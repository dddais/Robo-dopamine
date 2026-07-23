#!/usr/bin/env python3
"""Orchestrate the held-out target-bbox RoboRewardBench experiment.

Stages:

1. ``prepare``: freeze the 10-example discovery / 19-example evaluation split;
2. ``rank``: discover bbox-attending heads on discovery only;
3. ``experiment``: run baseline and four causal/control conditions, optionally
   sharded over GPUs;
4. ``metrics``: merge shards and compute paired bootstrap statistics plus
   optional post-hoc ordinal accuracy;
5. ``curve``: render the intervention dose-response from saved predictions;
6. ``video``: render baseline/candidate endpoint attention heatmap videos.

The expensive model stages are resumable.  Existing outputs are accepted only
when their provenance signatures still match the requested run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from roborewardbench.attention_mask.dataset import SELECTION_MODES, sha256_file
from roborewardbench.attention_mask.io import file_identity, model_identity


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GROUNDING_DIR = (
    REPO_ROOT
    / "roborewardbench"
    / "dopamine_eval"
    / "outputs"
    / "counterfactual_reward1"
)
DEFAULT_DATASET_ROOT = REPO_ROOT.parent / "data" / "RoboRewardBench_counterfactual_reward1"
DEFAULT_MODEL = REPO_ROOT / "pretrained_models" / "Robo-Dopamine-GRM-2.0-8B-Preview"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "attention" / "roborewardbench_counterfactual_reward1"
STAGES = ("prepare", "rank", "experiment", "metrics", "curve", "video", "all")


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve(strict=False)


def _python(args: argparse.Namespace) -> str:
    return str(_resolve(args.python_bin)) if args.python_bin else sys.executable


def _gpus(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--gpus must contain at least one device index")
    return values


def _run(
    command: Sequence[str],
    *,
    log_path: Path,
    gpu: str | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[runner] {' '.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; log={log_path}\n"
            + "\n".join(tail)
        )


def _rank_is_current(
    rank_path: Path,
    split_path: Path,
    model_path: Path,
    *,
    allow_smoke: bool = False,
    score_mode: str = "excess_mass",
) -> bool:
    if not rank_path.is_file():
        return False
    try:
        ranking = json.loads(rank_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    current_model = model_identity(model_path)
    return bool(
        (
            ranking.get("complete_discovery_partition")
            or (
                allow_smoke
                and int(ranking.get("n_discovery", 0)) == 1
                and int((ranking.get("args") or {}).get("max_examples", 0)) == 1
            )
        )
        and (ranking.get("split_manifest") or {}).get("sha256") == sha256_file(split_path)
        and ((ranking.get("model") or {}).get("config") or {}).get("sha256")
        == current_model["config"].get("sha256")
        and (ranking.get("model") or {}).get("inventory_fingerprint")
        == current_model.get("inventory_fingerprint")
        and str((ranking.get("args") or {}).get("score_mode")) == score_mode
        and (((ranking.get("code") or {}).get("rank_heads") or {}).get("sha256"))
        == file_identity(
            REPO_ROOT / "roborewardbench" / "attention_mask" / "rank_heads.py"
        ).get("sha256")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES, nargs="?", default="all")
    parser.add_argument("--grounding-dir", default=str(DEFAULT_GROUNDING_DIR))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--selection-mode",
        default="manual_correct_ready",
        choices=SELECTION_MODES,
        help=(
            "Evaluation bbox selection. Use auto_detected with --fixed-head-ranking "
            "for automatic-grounding transfer experiments."
        ),
    )
    parser.add_argument(
        "--fixed-head-ranking",
        default=None,
        help=(
            "Frozen ranking JSON produced outside this RoboRewardBench split. "
            "When set, the rank stage is skipped and discovery linkage is not required."
        ),
    )
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--top-ks", default="8,64")
    parser.add_argument("--biases", default="0,2,4,6")
    parser.add_argument(
        "--target-role",
        default="both",
        choices=["before", "after", "both", "after_high"],
        help=(
            "Image-span scope for the intervention. after_high reproduces the "
            "original success experiment by modifying only after_cam_high."
        ),
    )
    parser.add_argument(
        "--ranking-score",
        default="excess_mass",
        choices=["excess_mass", "raw_mass"],
    )
    parser.add_argument(
        "--intervention",
        default="boost_suppress",
        choices=["boost_suppress", "suppress_image"],
    )
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-pixels", type=int, default=76800)
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--video-top-k",
        type=int,
        default=64,
        help="Number of discovery-ranked heads used in candidate-target heatmaps.",
    )
    parser.add_argument(
        "--video-bias",
        type=float,
        default=4.0,
        help="Attention-logit bias used for the candidate-target heatmap.",
    )
    parser.add_argument(
        "--video-max-examples",
        type=int,
        default=None,
        help="Optional cap on frozen evaluation examples rendered in the heatmap video.",
    )
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--video-alpha", type=float, default=0.65)
    parser.add_argument("--video-save-png-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-rank", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Rank/evaluate one example with top-8 and biases 0,2; outputs are marked incomplete.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grounding_dir = _resolve(args.grounding_dir)
    dataset_root = _resolve(args.dataset_root)
    model_path = _resolve(args.model_path)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    split_path = output_root / "split_manifest.json"
    fixed_rank_path = (
        _resolve(args.fixed_head_ranking) if args.fixed_head_ranking is not None else None
    )
    rank_path = (
        fixed_rank_path
        if fixed_rank_path is not None
        else output_root / "head_discovery" / "head_ranking.json"
    )
    metrics_path = output_root / "metrics.json"
    python = _python(args)
    gpus = _gpus(args.gpus)
    stages = set(STAGES[:-1]) if args.stage == "all" else {args.stage}
    existing_shards = sorted((output_root / "shards").glob("shard_*"))
    if args.stage in {"metrics", "curve"} and args.num_shards is None and existing_shards:
        indices = [int(path.name.rsplit("_", 1)[1]) for path in existing_shards]
        if indices != list(range(len(indices))):
            raise ValueError(f"Existing shard directories are not contiguous: {indices}")
        num_shards = len(indices)
    else:
        num_shards = args.num_shards if args.num_shards is not None else len(gpus)
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.video_top_k <= 0:
        raise ValueError("--video-top-k must be positive")
    if args.video_max_examples is not None and args.video_max_examples <= 0:
        raise ValueError("--video-max-examples must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if not 0.0 <= args.video_alpha <= 1.0:
        raise ValueError("--video-alpha must be in [0, 1]")
    if args.video_save_png_frames < 0:
        raise ValueError("--video-save-png-frames cannot be negative")
    if fixed_rank_path is not None and not fixed_rank_path.is_file():
        raise FileNotFoundError(f"Fixed head ranking not found: {fixed_rank_path}")
    if fixed_rank_path is None and args.selection_mode != "manual_correct_ready":
        raise ValueError(
            "Non-manual selection modes require --fixed-head-ranking; in-split head "
            "discovery remains bound to the audited holdout protocol."
        )
    if "experiment" in stages and num_shards > len(gpus):
        raise ValueError(
            f"num_shards={num_shards} exceeds the {len(gpus)} GPU(s) listed in --gpus"
        )
    if "prepare" in stages:
        if split_path.is_file():
            print(f"[runner] reusing frozen split {split_path}", flush=True)
        else:
            command = [
                python,
                "-m",
                "roborewardbench.attention_mask.dataset",
                "--grounding-dir",
                str(grounding_dir),
                "--output",
                str(split_path),
            ]
            if fixed_rank_path is not None:
                command.extend(
                    [
                        "--strategy",
                        "evaluation_only",
                        "--selection-mode",
                        args.selection_mode,
                    ]
                )
            _run(
                command,
                log_path=logs / "prepare.log",
                gpu=None,
            )

    if any(stage in stages for stage in ("rank", "experiment", "metrics", "video")) and not split_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {split_path}. Run the prepare stage first."
        )

    if "rank" in stages:
        if fixed_rank_path is not None:
            print(f"[runner] using external fixed ranking {rank_path}", flush=True)
        elif not args.force_rank and _rank_is_current(
            rank_path,
            split_path,
            model_path,
            allow_smoke=args.smoke,
            score_mode=args.ranking_score,
        ):
            print(f"[runner] reusing current ranking {rank_path}", flush=True)
        else:
            command = [
                python,
                "-m",
                "roborewardbench.attention_mask.rank_heads",
                "--grounding-dir",
                str(grounding_dir),
                "--split-manifest",
                str(split_path),
                "--model-path",
                str(model_path),
                "--output",
                str(rank_path),
                "--target-role",
                args.target_role,
                "--score-mode",
                args.ranking_score,
                "--dtype",
                args.dtype,
                "--max-pixels",
                str(args.max_pixels),
                "--min-pixels",
                str(args.min_pixels),
            ]
            if args.smoke:
                command.extend(["--max-examples", "1"])
            _run(command, log_path=logs / "rank.log", gpu=gpus[0])

    if "experiment" in stages:
        if not rank_path.is_file():
            raise FileNotFoundError(
                f"Head ranking not found: {rank_path}. Run the rank stage first."
            )
        top_ks = "8" if args.smoke else args.top_ks
        biases = "0,2" if args.smoke else args.biases

        def run_shard(shard_index: int) -> None:
            shard_dir = output_root / "shards" / f"shard_{shard_index:03d}"
            command = [
                python,
                "-m",
                "roborewardbench.attention_mask.run_experiment",
                "--grounding-dir",
                str(grounding_dir),
                "--split-manifest",
                str(split_path),
                "--head-ranking",
                str(rank_path),
                "--model-path",
                str(model_path),
                "--output-dir",
                str(shard_dir),
                "--target-role",
                args.target_role,
                "--selection-mode",
                args.selection_mode,
                "--top-ks",
                top_ks,
                "--biases",
                biases,
                "--intervention",
                args.intervention,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--seed",
                str(args.seed),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(num_shards),
                "--dtype",
                args.dtype,
                "--max-pixels",
                str(args.max_pixels),
                "--min-pixels",
                str(args.min_pixels),
            ]
            if args.decode_only:
                command.append("--decode-only")
            if fixed_rank_path is not None:
                command.append("--external-fixed-ranking")
            if args.smoke:
                command.extend(["--max-examples", "1", "--allow-incomplete-ranking"])
            _run(
                command,
                log_path=logs / f"experiment_shard_{shard_index:03d}.log",
                gpu=gpus[shard_index],
            )

        with ThreadPoolExecutor(max_workers=num_shards) as pool:
            futures = {pool.submit(run_shard, index): index for index in range(num_shards)}
            for future in as_completed(futures):
                future.result()

    result_paths = [
        output_root / "shards" / f"shard_{index:03d}" / "results.jsonl"
        for index in range(num_shards)
    ]
    if "metrics" in stages:
        missing = [path for path in result_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing shard results: {missing}")
        metadata_path = dataset_root / "test" / "metadata.jsonl"
        command = [
            python,
            "-m",
            "roborewardbench.attention_mask.metrics",
            "--results",
            *[str(path) for path in result_paths],
            "--output",
            str(metrics_path),
            "--markdown",
            str(output_root / "metrics.md"),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--bootstrap-seed",
            str(args.seed),
        ]
        if metadata_path.is_file():
            command.extend(["--metadata", str(metadata_path)])
        _run(command, log_path=logs / "metrics.log", gpu=None)

    if "curve" in stages:
        missing = [path for path in result_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing shard results: {missing}")
        command = [
            python,
            "-m",
            "roborewardbench.attention_mask.curve",
            "--results",
            *[str(path) for path in result_paths],
            "--output-dir",
            str(output_root / "artifacts" / "curves"),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--bootstrap-seed",
            str(args.seed),
        ]
        _run(command, log_path=logs / "curve.log", gpu=None)

    if "video" in stages:
        if not rank_path.is_file():
            raise FileNotFoundError(
                f"Head ranking not found: {rank_path}. Run the rank stage first."
            )
        video_top_k = 8 if args.smoke else args.video_top_k
        video_bias = 2.0 if args.smoke else args.video_bias
        command = [
            python,
            "-m",
            "roborewardbench.attention_mask.visualize",
            "--grounding-dir",
            str(grounding_dir),
            "--split-manifest",
            str(split_path),
            "--head-ranking",
            str(rank_path),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(output_root / "artifacts" / "attention_video"),
            "--target-role",
            args.target_role,
            "--selection-mode",
            args.selection_mode,
            "--top-k",
            str(video_top_k),
            "--swap-bias",
            str(video_bias),
            "--intervention",
            args.intervention,
            "--fps",
            str(args.video_fps),
            "--alpha",
            str(args.video_alpha),
            "--save-png-frames",
            str(args.video_save_png_frames),
            "--seed",
            str(args.seed),
            "--dtype",
            args.dtype,
            "--max-pixels",
            str(args.max_pixels),
            "--min-pixels",
            str(args.min_pixels),
        ]
        if args.decode_only:
            command.append("--decode-only")
        if fixed_rank_path is not None:
            command.append("--external-fixed-ranking")
        if args.smoke:
            command.extend(["--max-examples", "1", "--allow-incomplete-ranking"])
        elif args.video_max_examples is not None:
            command.extend(["--max-examples", str(args.video_max_examples)])
        _run(command, log_path=logs / "video.log", gpu=gpus[0])

    print(f"[runner] requested stage(s) complete under {output_root}", flush=True)


if __name__ == "__main__":
    main()
