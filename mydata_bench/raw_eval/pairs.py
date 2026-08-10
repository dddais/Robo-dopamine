"""Instruction-conditioned, same-video raw GRM baseline."""

from __future__ import annotations

import hashlib
import math
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from ..config import section
from ..data import load_episodes
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    latest_by_id,
    provenance,
    read_jsonl,
    sha256_file,
    stable_shard,
    write_json,
    write_jsonl,
)
from ..protocol import (
    multiview_endpoint_payload,
    native_endpoint_payload,
    parse_score,
    progress,
    progress_to_reward,
    system_prompt,
)
from ..schemas import EpisodeRecord, SCHEMA_VERSION
from ..video import extract_endpoints
from .runner import VLLMGRM, sampling_kwargs


SIDES = ("counterfactual", "original")


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    value = section(config, "paired_raw_eval")
    required = ("dataset_root", "output_dir", "model_path")
    missing = [key for key in required if not value.get(key)]
    if missing:
        raise ValueError(f"paired_raw_eval is missing required keys: {', '.join(missing)}")
    return value


def _pair_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one pair for every fail row and its declared source_suc_id."""
    cfg = _cfg(config)
    rows = list(
        load_episodes(
            cfg["dataset_root"],
            str(cfg.get("split", "all")),
            compute_hash=True,
        )
    )
    successes = {
        row.example_id: row
        for row in rows
        if row.instruction_video_match is True
    }
    pairs: list[dict[str, Any]] = []
    for counter in sorted(
        (row for row in rows if row.instruction_video_match is False),
        key=lambda row: row.example_id,
    ):
        original = successes.get(str(counter.source_suc_id))
        if original is None:
            raise ValueError(
                f"Missing source success {counter.source_suc_id!r} for {counter.example_id}"
            )
        if counter.video_sha256 != original.video_sha256:
            raise ValueError(f"View-content mismatch for {counter.example_id}")
        if counter.task == original.task:
            raise ValueError(f"Instructions are identical for {counter.example_id}")
        pair_id = hashlib.sha256(
            f"{counter.video_sha256}\0{counter.example_id}".encode("utf-8")
        ).hexdigest()
        pairs.append(
            {
                "pair_id": pair_id,
                "video_sha256": counter.video_sha256,
                "video_path": counter.video_path,
                "view_paths": counter.views,
                "subset": counter.subset,
                "counterfactual_example_id": counter.example_id,
                "counterfactual_task": counter.task,
                "counterfactual_reward": 1,
                "original_example_id": original.example_id,
                "original_task": original.task,
                "original_reward": 5,
                "source_suc_id": original.example_id,
                "original_candidate_example_ids": [original.example_id],
                "original_candidate_tasks": [original.task],
            }
        )
    return pairs


def prepare(config: dict[str, Any]) -> Path:
    """Freeze the same-video instruction pairs without running any model."""
    cfg = _cfg(config)
    output_dir = Path(cfg["output_dir"]).resolve()
    pairs = _pair_rows(config)
    expected = cfg.get("expected_pairs")
    if expected is not None and len(pairs) != int(expected):
        raise ValueError(f"Expected {int(expected)} paired videos, found {len(pairs)}")
    write_jsonl(output_dir / "pairs.jsonl", pairs)
    summary = {
        "num_pairs": len(pairs),
        "num_model_inputs": len(pairs) * len(SIDES),
        "num_unique_video_sha256": len({row["video_sha256"] for row in pairs}),
        "subsets": dict(sorted(Counter(row["subset"] for row in pairs).items())),
        "counterfactual_reward": 1,
        "original_reward": 5,
        "duplicate_original_metadata_pairs": sum(
            len(row["original_candidate_example_ids"]) > 1 for row in pairs
        ),
    }
    summary["pair_fingerprint"] = jsonl_fingerprint(pairs)
    write_json(output_dir / "pair_manifest_summary.json", summary)
    write_json(
        output_dir / "pair_prepare_manifest.json",
        {
            **provenance(sys.argv, config, Path(__file__).resolve().parents[2]),
            "pair_manifest_summary": summary,
        },
    )
    return output_dir / "pairs.jsonl"


def jsonl_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Hash the canonical pair rows without a dependency on output file bytes."""
    from ..io import object_fingerprint

    return object_fingerprint(rows)


def _run_manifest(config: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = _cfg(config)
    prompt_mode = str(cfg.get("prompt_mode", "official"))
    prompt = system_prompt(prompt_mode)
    return {
        **provenance(sys.argv, config, Path(__file__).resolve().parents[2]),
        "model_fingerprint": artifact_fingerprint(cfg["model_path"]),
        "metadata_sha256": sha256_file(
            Path(cfg["dataset_root"]) / "metadata.jsonl"
        ),
        "raw_protocol": {
            "prompt_mode": prompt_mode,
            "prompt_template_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "sampling": sampling_kwargs(cfg),
        },
        "source_fingerprints": {
            "rewardbench/protocol.py": sha256_file(Path(__file__).resolve().parents[1] / "protocol.py"),
            "rewardbench/raw_eval/runner.py": sha256_file(Path(__file__).resolve().parent / "runner.py"),
            "rewardbench/raw_eval/pairs.py": sha256_file(Path(__file__).resolve()),
        },
        "pair_manifest_fingerprint": jsonl_fingerprint(pairs),
    }


def run(config: dict[str, Any], *, dry_run: bool = False, retry_failed: bool = False) -> Path:
    cfg = _cfg(config)
    output_dir = Path(cfg["output_dir"]).resolve()
    pairs_path = output_dir / "pairs.jsonl"
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Run prepare first; missing {pairs_path}")
    pairs = list(read_jsonl(pairs_path))
    expected = cfg.get("expected_pairs")
    if expected is not None and len(pairs) != int(expected):
        raise ValueError(f"Expected {int(expected)} frozen pairs, found {len(pairs)}")
    shard_id = int(cfg.get("shard_id", 0))
    num_shards = int(cfg.get("num_shards", 1))
    records_path = output_dir / f"paired_records.shard-{shard_id:02d}.jsonl"
    previous = latest_by_id(read_jsonl(records_path), key="record_id") if records_path.exists() else {}
    manifest_path = output_dir / ("pair_run_manifest.json" if num_shards == 1 else f"pair_run_manifest.shard-{shard_id:02d}.json")
    manifest = _run_manifest(config, pairs)
    manifest.update({"shard_id": shard_id, "num_shards": num_shards})
    write_json(manifest_path, manifest)
    local_pairs = [
        row for row in pairs if stable_shard(str(row["pair_id"]), num_shards) == shard_id
    ]
    engine = None if dry_run else VLLMGRM(cfg)
    blank_goal = cfg.get(
        "blank_goal", str(Path(__file__).resolve().parents[2] / "examples" / "blank_goal.png")
    )
    prompt_mode = str(cfg.get("prompt_mode", "official"))
    for pair in local_pairs:
        frame_dir = output_dir / "frames" / str(pair["pair_id"])
        try:
            frames = extract_endpoints(
                str(pair["pair_id"]),
                str(pair["video_sha256"]),
                str(pair["video_path"]),
                frame_dir,
            )
        except Exception as exc:
            for side in SIDES:
                record_id = f"{pair['pair_id']}:{side}"
                old = previous.get(record_id)
                if old and (old.get("status") == "ok" or not retry_failed):
                    continue
                append_jsonl(
                    records_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_id": record_id,
                        "pair_id": pair["pair_id"],
                        "video_sha256": pair["video_sha256"],
                        "subset": pair["subset"],
                        "side": side,
                        "attempt": int(old.get("attempt", 0)) + 1 if old else 1,
                        "status": "invalid",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            continue
        for side in SIDES:
            record_id = f"{pair['pair_id']}:{side}"
            old = previous.get(record_id)
            if old and (old.get("status") == "ok" or not retry_failed):
                continue
            task_key, id_key, reward_key = (f"{side}_task", f"{side}_example_id", f"{side}_reward")
            episode = EpisodeRecord(
                example_id=str(pair[id_key]),
                video_path=str(pair["video_path"]),
                task=str(pair[task_key]),
                reward=int(pair[reward_key]),
                subset=str(pair["subset"]),
                video_sha256=str(pair["video_sha256"]),
                view_paths=tuple(sorted(dict(pair.get("view_paths", {})).items())),
            )
            base = {
                "schema_version": SCHEMA_VERSION,
                "record_id": record_id,
                "pair_id": pair["pair_id"],
                "video_sha256": pair["video_sha256"],
                "subset": pair["subset"],
                "side": side,
                "example_id": episode.example_id,
                "reward": episode.reward,
                "task": episode.task,
                "attempt": int(old.get("attempt", 0)) + 1 if old else 1,
            }
            try:
                frames_by_view = {
                    view: extract_endpoints(
                        episode.example_id,
                        episode.video_sha256,
                        path,
                        frame_dir / view,
                    )
                    for view, path in episode.views.items()
                }
                payload = (
                    multiview_endpoint_payload(
                        episode,
                        frames_by_view,
                        blank_goal,
                        prompt_mode=prompt_mode,
                    )
                    if {"front", "left_wrist", "right_wrist"} <= set(frames_by_view)
                    else native_endpoint_payload(
                        episode, frames, blank_goal, prompt_mode=prompt_mode
                    )
                )
                if {"reward", "gpt5_mini_check"} & payload.keys():
                    raise AssertionError("Label leakage into pair model payload")
                if dry_run:
                    output, signed, status = "<score>0%</score>", 0.0, "dry_run"
                else:
                    assert engine is not None
                    output = engine.infer(payload)
                    signed, status = parse_score(output), "ok"
                append_jsonl(
                    records_path,
                    {
                        **base,
                        "frame_record": frames.to_dict(),
                        "frame_records": {
                            view: record.to_dict()
                            for view, record in frames_by_view.items()
                        }
                        or None,
                        "protocol": payload["protocol"],
                        "prompt_mode": payload["prompt_mode"],
                        "raw_output": output,
                        "signed_score": signed,
                        "progress": progress(signed),
                        "status": status,
                    },
                )
            except Exception as exc:
                append_jsonl(
                    records_path,
                    {
                        **base,
                        "status": "invalid",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
    return records_path


def _paired_bootstrap(rows: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, Any]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["subset"])].append(row)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = []
        for values in strata.values():
            draw.extend(rng.choice(values) for _ in values)
        estimates.append(mean(float(item["delta_signed_score"]) for item in draw))
    estimates.sort()
    return {
        "estimate": mean(float(item["delta_signed_score"]) for item in rows),
        "ci95": [
            estimates[max(0, math.floor(0.025 * (samples - 1)))],
            estimates[min(samples - 1, math.ceil(0.975 * (samples - 1)))],
        ],
        "samples": samples,
        "seed": seed,
    }


def _sign_flip_pvalue(deltas: list[float], *, samples: int, seed: int) -> float:
    observed = abs(mean(deltas))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        value = abs(mean(delta if rng.choice((False, True)) else -delta for delta in deltas))
        extreme += value >= observed
    return (extreme + 1) / (samples + 1)


def score(run_dir: str | Path, *, bootstrap_samples: int = 10_000, seed: int = 20260724) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    pairs = list(read_jsonl(run_dir / "pairs.jsonl"))
    expected_ids = {f"{row['pair_id']}:{side}" for row in pairs for side in SIDES}
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("paired_records.shard-*.jsonl")):
        latest.update(latest_by_id(read_jsonl(path), key="record_id"))
    invalid = [row for row in latest.values() if row.get("status") != "ok"]
    missing = sorted(expected_ids - set(latest))
    unexpected = sorted(set(latest) - expected_ids)
    pair_rows = []
    for pair in pairs:
        counter = latest.get(f"{pair['pair_id']}:counterfactual")
        original = latest.get(f"{pair['pair_id']}:original")
        if not counter or not original or counter.get("status") != "ok" or original.get("status") != "ok":
            continue
        delta = float(original["signed_score"]) - float(counter["signed_score"])
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "video_sha256": pair["video_sha256"],
                "subset": pair["subset"],
                "counterfactual_example_id": counter["example_id"],
                "original_example_id": original["example_id"],
                "counterfactual_signed_score": counter["signed_score"],
                "original_signed_score": original["signed_score"],
                "delta_signed_score": delta,
                "counterfactual_prediction": progress_to_reward(float(counter["progress"])),
                "original_prediction": progress_to_reward(float(original["progress"])),
            }
        )
    complete = not invalid and not missing and not unexpected and len(pair_rows) == len(pairs)
    result: dict[str, Any] = {
        "formal_scoring_ready": complete,
        "expected_pairs": len(pairs),
        "paired_valid": len(pair_rows),
        "invalid_count": len(invalid),
        "missing_record_ids": missing,
        "unexpected_record_ids": unexpected,
    }
    if pair_rows:
        deltas = [float(row["delta_signed_score"]) for row in pair_rows]
        result["instruction_conditioned_score_difference"] = {
            "definition": "original_reward5_signed_score - counterfactual_reward1_signed_score",
            "mean": mean(deltas),
            "median": median(deltas),
            "original_greater_fraction": mean(value > 0 for value in deltas),
            "equal_fraction": mean(value == 0 for value in deltas),
            "counterfactual_greater_fraction": mean(value < 0 for value in deltas),
            "paired_cluster_stratified_bootstrap": _paired_bootstrap(
                pair_rows, samples=bootstrap_samples, seed=seed
            ),
            "two_sided_sign_flip_pvalue": _sign_flip_pvalue(
                deltas, samples=bootstrap_samples, seed=seed
            ),
        }
        result["prediction_direction"] = {
            "original_prediction_gt_counterfactual_fraction": mean(
                row["original_prediction"] > row["counterfactual_prediction"] for row in pair_rows
            ),
            "equal_prediction_fraction": mean(
                row["original_prediction"] == row["counterfactual_prediction"] for row in pair_rows
            ),
            "counterfactual_prediction_gt_original_fraction": mean(
                row["original_prediction"] < row["counterfactual_prediction"] for row in pair_rows
            ),
        }
    write_jsonl(run_dir / "paired_scores.jsonl", pair_rows)
    write_json(run_dir / "paired_metrics.json", result)
    write_json(run_dir / "completion.json", result)
    write_json(run_dir / "invalid.json", invalid)
    summary = result.get("instruction_conditioned_score_difference", {})
    (run_dir / "paired_metrics.md").write_text(
        "# Paired Raw GRM Metrics\n\n"
        f"- Formal scoring ready: `{complete}`\n"
        f"- Valid pairs: {len(pair_rows)} / {len(pairs)}\n"
        f"- Mean original−counterfactual signed-score difference: {summary.get('mean')}\n"
        f"- 95% bootstrap CI: {summary.get('paired_cluster_stratified_bootstrap', {}).get('ci95')}\n"
        f"- Two-sided sign-flip p-value: {summary.get('two_sided_sign_flip_pvalue')}\n",
        encoding="utf-8",
    )
    return result
