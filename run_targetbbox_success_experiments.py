#!/usr/bin/env python3
"""Run target-only bbox head-ranking and steering experiments.

This is a small orchestration helper for the 2026-07-08 rerun after fixing bbox
grounding contamination.  It keeps every command, log, and expected output under
one output directory so the experiment can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent
TASKS = {
    "carrot": "pick the carrot and put it on yellow plate",
    "cube": "pick the white cube and put it on yellow plate",
    "bottle": "pick the bottle and put it on yellow plate",
}
DEFAULT_SOURCE_ROOTS = {
    "carrot": "results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_1_carrot/blank/inter20",
    "bottle": "results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_3_bottle/blank/inter20",
    "cube": "results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_4_cube/blank/inter20",
}
# MODES = ("forward", "incremental", "backward")
MODES = ("forward",)
# QUERIES = ("last_prompt", "decode")
QUERIES = ("last_prompt",)
TOP_KS = (8, 64)
VALID_QUERIES = {"last_prompt", "decode"}
CURVE_MODE = ["forward"]

@dataclass(frozen=True)
class Job:
    name: str
    cmd: list[str]
    output: Path
    log: Path
    gpu: int
    output_kind: str = "json"
    provenance: Optional[dict[str, str]] = None


def sample_path_for(root: Path, task: str, mode: str) -> Path:
    task_text = TASKS[task].replace(" ", "_")
    matches = []
    for search_root in [root / task, root]:
        if not search_root.exists():
            continue
        matches.extend(search_root.glob(f"**/*{mode}_mode_{task_text}*/sample.json"))
    deduped = {}
    for path in matches:
        deduped[str(path.resolve())] = path.resolve()
    matches = sorted(deduped.values())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one sample for {task}/{mode}, got {len(matches)}: {matches}")
    return matches[0]


def resolve_source_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def source_root_for_task(args: argparse.Namespace, task: str) -> Path:
    task_override = getattr(args, f"{task}_source_root", None)
    if task_override:
        return resolve_source_root(task_override)
    if args.source_root:
        return resolve_source_root(args.source_root)
    return resolve_source_root(DEFAULT_SOURCE_ROOTS[task])


def resolved_source_roots(args: argparse.Namespace) -> dict[str, str]:
    return {task: str(source_root_for_task(args, task)) for task in TASKS}


def conda_base() -> Optional[Path]:
    conda = shutil.which("conda")
    if conda is None:
        return None
    conda_path = Path(conda).resolve(strict=False)
    if conda_path.parent.name == "bin":
        return conda_path.parent.parent
    return None


def env_prefix_for_args(args: argparse.Namespace) -> Optional[Path]:
    if args.python_bin:
        py = resolve_source_root(args.python_bin)
        return py.parent.parent if py.parent.name == "bin" else None
    env_name = args.conda_env
    if not env_name:
        return None
    env_path = Path(env_name)
    if env_path.is_absolute() or "/" in env_name:
        return resolve_source_root(env_path)
    active = os.environ.get("CONDA_PREFIX")
    if active and Path(active).name == env_name:
        return Path(active).resolve(strict=False)
    base = conda_base()
    if base is None:
        return None
    return (base / "envs" / env_name).resolve(strict=False)


def python_cmd(args: argparse.Namespace) -> list[str]:
    if args.python_bin:
        return [str(resolve_source_root(args.python_bin))]
    prefix = env_prefix_for_args(args)
    if prefix is not None:
        py = prefix / "bin" / "python"
        if py.exists():
            return [str(py)]
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "python"]
    return [sys.executable]


def env_prefix_from_cmd(cmd: list[str]) -> Optional[Path]:
    if not cmd:
        return None
    first = Path(cmd[0])
    if first.name.startswith("python") and first.parent.name == "bin":
        return first.parent.parent.resolve(strict=False)
    return None


def subprocess_env(cmd: list[str], gpu: Optional[int] = None) -> dict[str, str]:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    prefix = env_prefix_from_cmd(cmd)
    if prefix is not None:
        env["CONDA_PREFIX"] = str(prefix)
        env["PATH"] = f"{prefix / 'bin'}:{env.get('PATH', '')}"
        cc = prefix / "bin" / "x86_64-conda-linux-gnu-cc"
        cxx = prefix / "bin" / "x86_64-conda-linux-gnu-c++"
        if cc.exists():
            env["CC"] = str(cc)
        if cxx.exists():
            env["CXX"] = str(cxx)
    return env


def is_complete(path: Path, kind: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if kind == "json":
        try:
            json.loads(path.read_text())
        except Exception:
            return False
    return True


def canonicalize_path(value: str | Path) -> str:
    return str(resolve_source_root(value))


def json_field(data: dict, dotted_key: str):
    cur = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def values_match(actual, expected: str) -> bool:
    if isinstance(actual, list):
        if len(actual) != 1:
            return False
        actual = actual[0]
    if actual is None:
        return False
    actual_s = str(actual)
    expected_s = str(expected)
    if (
        actual_s.endswith(".json")
        or expected_s.endswith(".json")
        or "/" in actual_s
        or "/" in expected_s
    ):
        return canonicalize_path(actual_s) == canonicalize_path(expected_s)
    return actual_s == expected_s


def matches_provenance(path: Path, kind: str, provenance: Optional[dict[str, str]]) -> bool:
    if not provenance:
        return True
    if kind != "json":
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    for key, expected in provenance.items():
        if not values_match(json_field(data, key), expected):
            return False
    return True


def output_is_current(job: Job) -> bool:
    if not is_complete(job.output, job.output_kind):
        return False
    return matches_provenance(job.output, job.output_kind, job.provenance)


def query_args(query: str) -> list[str]:
    if query == "decode":
        return ["--query-mode", "generate", "--generate-query-stage", "predict_token"]
    if query == "last_prompt":
        return ["--query-mode", "last_prompt"]
    raise ValueError(f"Unknown query preset {query!r}; expected one of {sorted(VALID_QUERIES)}")


def rank_jobs(args: argparse.Namespace, out_root: Path, logs: Path) -> list[Job]:
    jobs: list[Job] = []
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    n = 0
    py = python_cmd(args)
    for task in TASKS:
        src_root = source_root_for_task(args, task)
        for mode in MODES:
            sample_json = sample_path_for(src_root, task, mode)
            for query in QUERIES:
                if query not in VALID_QUERIES:
                    raise ValueError(
                        f"Invalid query preset {query!r}. QUERIES must be a tuple/list like "
                        "('last_prompt',) or ('last_prompt', 'decode')."
                    )
                out = out_root / f"rank_{task}_{mode}_{query}" / "head_ranking.json"
                log = logs / f"rank_{task}_{mode}_{query}.log"
                cmd = py + [
                    "rank_heads_by_bbox.py",
                    "--sample-json", str(sample_json),
                    "--target-label", args.target_label,
                    "--grounding-box-threshold", str(args.grounding_box_threshold),
                    "--num-samples", str(args.rank_num_samples),
                    "--top-k", "100",
                    "--output", str(out),
                ] + query_args(query)
                jobs.append(Job(
                    f"rank_{task}_{mode}_{query}",
                    cmd,
                    out,
                    log,
                    gpus[n % len(gpus)],
                    provenance={"args.sample_json": str(sample_json)},
                ))
                n += 1
    return jobs


def curve_jobs(args: argparse.Namespace, out_root: Path, logs: Path) -> list[Job]:
    jobs: list[Job] = []
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    n = 0
    py = python_cmd(args)
    for dataset_task in TASKS:
        src_root = source_root_for_task(args, dataset_task)
        for curve_mode in args.curve_modes:
            sample_json = sample_path_for(src_root, dataset_task, curve_mode)
            for inference_task, instruction in TASKS.items():
                for query in QUERIES:
                    rank = out_root / f"rank_{inference_task}_{curve_mode}_{query}" / "head_ranking.json"
                    for top_k in args.top_ks:
                        out_dir = out_root / (
                            f"curve_data-{dataset_task}_instr-{inference_task}_"
                            f"{curve_mode}_{query}_top{top_k}_full"
                        )
                        out = out_dir / "curve.json"
                        log = logs / (
                            f"curve_data-{dataset_task}_instr-{inference_task}_"
                            f"{curve_mode}_{query}_top{top_k}.log"
                        )
                        cmd = py + [
                            "steer_progress_curve.py",
                            "--sample-json", str(sample_json),
                            "--override-task", instruction,
                            "--head-ranking-json", str(rank),
                            "--target-label", args.target_label,
                            "--grounding-box-threshold", str(args.grounding_box_threshold),
                            "--top-k", str(top_k),
                            "--swap-bias", str(args.swap_bias),
                            "--random-control", "low_ranked",
                            "--wrong-region-samples", str(args.wrong_region_samples),
                            "--output-dir", str(out_dir),
                        ]
                        jobs.append(Job(
                            f"curve_data-{dataset_task}_instr-{inference_task}_{curve_mode}_{query}_top{top_k}",
                            cmd,
                            out,
                            log,
                            gpus[n % len(gpus)],
                            provenance={
                                "args.sample_json": str(sample_json),
                                "args.head_ranking_json": str(rank),
                                "args.override_task": instruction,
                            },
                        ))
                        n += 1
    return jobs


def video_jobs(args: argparse.Namespace, out_root: Path, logs: Path) -> list[Job]:
    jobs: list[Job] = []
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    n = 0
    py = python_cmd(args)
    specs = [
        ("carrot", "carrot"),
        ("cube", "cube"),
        ("bottle", "bottle"),
    ]
    for dataset_task, inference_task in specs:
        for query in QUERIES:
            if query not in VALID_QUERIES:
                raise ValueError(
                    f"Invalid query preset {query!r}. QUERIES must be a tuple/list like "
                    "('last_prompt',) or ('last_prompt', 'decode')."
                )
            src_root = source_root_for_task(args, dataset_task)
            sample_json = sample_path_for(src_root, dataset_task, "incremental")
            rank = out_root / f"rank_{inference_task}_incremental_{query}" / "head_ranking.json"
            out_dir = out_root / f"video_data-{dataset_task}_instr-{inference_task}_{query}_top64_8frames"
            out = out_dir / "attention_video_manifest.json"
            log = logs / f"video_data-{dataset_task}_instr-{inference_task}_{query}_top64.log"
            cmd = py + [
                "visualize_stage3_head_attention.py",
                "--sample-json", str(sample_json),
                "--override-task", TASKS[inference_task],
                "--head-ranking-json", str(rank),
                "--target-label", args.target_label,
                "--grounding-box-threshold", str(args.grounding_box_threshold),
                "--top-k", "64",
                "--num-samples", str(args.video_num_samples),
                "--sample-strategy", "even",
                "--swap-bias", str(args.swap_bias),
                "--output-dir", str(out_dir),
            ] + query_args(query)
            jobs.append(Job(
                f"video_data-{dataset_task}_instr-{inference_task}_{query}_top64",
                cmd,
                out,
                log,
                gpus[n % len(gpus)],
                provenance={
                    "args.sample_json": str(sample_json),
                    "args.head_ranking_json": str(rank),
                    "args.override_task": TASKS[inference_task],
                },
            ))
            n += 1
    return jobs


def run_jobs(jobs: Iterable[Job], max_parallel: int, skip_existing: bool) -> list[dict]:
    pending = list(jobs)
    results: list[dict] = []
    active: list[tuple[subprocess.Popen, Job, float]] = []
    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.pop(0)
            if skip_existing and output_is_current(job):
                print(f"[runner] skip complete {job.name}: {job.output}", flush=True)
                results.append({"name": job.name, "status": "skipped", "output": str(job.output), "log": str(job.log)})
                continue
            if skip_existing and is_complete(job.output, job.output_kind):
                print(f"[runner] rerun stale {job.name}: {job.output}", flush=True)
            job.log.parent.mkdir(parents=True, exist_ok=True)
            job.output.parent.mkdir(parents=True, exist_ok=True)
            env = subprocess_env(job.cmd, gpu=job.gpu)
            cmd_text = " ".join(shlex.quote(x) for x in job.cmd)
            print(f"[runner] start gpu={job.gpu} {job.name}", flush=True)
            with job.log.open("w") as f:
                f.write(f"# gpu={job.gpu}\n# cmd={cmd_text}\n\n")
            log_f = job.log.open("a")
            proc = subprocess.Popen(job.cmd, cwd=Path(__file__).resolve().parent, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            proc._codex_log_file = log_f  # type: ignore[attr-defined]
            active.append((proc, job, time.time()))
        time.sleep(5)
        still_active: list[tuple[subprocess.Popen, Job, float]] = []
        for proc, job, start in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, job, start))
                continue
            proc._codex_log_file.close()  # type: ignore[attr-defined]
            elapsed = time.time() - start
            status = "completed" if rc == 0 and output_is_current(job) else "failed"
            print(f"[runner] {status} rc={rc} elapsed={elapsed/60:.1f}m {job.name}", flush=True)
            results.append({
                "name": job.name,
                "status": status,
                "returncode": rc,
                "elapsed_sec": elapsed,
                "output": str(job.output),
                "log": str(job.log),
            })
            if status == "failed":
                tail = job.log.read_text(errors="replace").splitlines()[-40:]
                print("\n".join(tail), flush=True)
                raise SystemExit(f"job failed: {job.name}; see {job.log}")
        active = still_active
    return results


def run_analysis(args: argparse.Namespace, out_root: Path) -> None:
    cmd = python_cmd(args) + [
        "analyze_top100_head_overlap.py",
        "--root", str(out_root),
        "--top-k", "100",
        "--output-dir", str(out_root / "top100_head_analysis"),
    ]
    print(f"[runner] analysis: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=subprocess_env(cmd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run target-only bbox Stage-2/Stage-3 success experiments")
    ap.add_argument("--output-root", default="results/attention/3data_3instruction_prompt_20260720")
    ap.add_argument(
        "--source-root",
        default=None,
        help="Legacy common source root for all tasks. If unset, carrot/cube/bottle "
             "default to their own success episodes.",
    )
    ap.add_argument("--carrot-source-root", default=DEFAULT_SOURCE_ROOTS["carrot"])
    ap.add_argument("--cube-source-root", default=DEFAULT_SOURCE_ROOTS["cube"])
    ap.add_argument("--bottle-source-root", default=DEFAULT_SOURCE_ROOTS["bottle"])
    ap.add_argument("--conda-env", default="robo-dopamine")
    ap.add_argument(
        "--python-bin",
        default=None,
        help="Python executable to use for jobs. Defaults to <conda-env>/bin/python, "
             "avoiding conda-run compiler activation issues.",
    )
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--target-label", default="after_cam_high")
    ap.add_argument("--grounding-box-threshold", type=float, default=0.12)
    ap.add_argument("--rank-num-samples", type=int, default=12)
    ap.add_argument("--top-ks", type=int, nargs="+", default=list(TOP_KS))
    ap.add_argument(
        "--curve-modes",
        nargs="+",
        choices=MODES,
        default= CURVE_MODE,
        help="Which trajectory/eval modes to use for Stage-3 curve jobs. "
             "Default preserves the original incremental-only behavior.",
    )
    ap.add_argument("--swap-bias", type=float, default=6.0)
    ap.add_argument("--wrong-region-samples", type=int, default=1,
                    help="For candidate_wrong: number of independent wrong regions drawn "
                         "from the bbox-complement pool, averaged to reduce single-draw noise.")
    ap.add_argument("--video-num-samples", type=int, default=30)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--stage", choices=["rank", "analysis", "curve", "video", "all"], default="all")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    logs = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "runner_config.json").write_text(json.dumps({
        **vars(args),
        "resolved_source_roots": resolved_source_roots(args),
    }, indent=2))

    all_results: list[dict] = []
    if args.stage in {"rank", "all"}:
        all_results.extend(run_jobs(rank_jobs(args, out_root, logs), args.max_parallel, args.skip_existing))
    if args.stage in {"analysis", "all"}:
        run_analysis(args, out_root)
    if args.stage in {"curve", "all"}:
        all_results.extend(run_jobs(curve_jobs(args, out_root, logs), args.max_parallel, args.skip_existing))
    if args.stage in {"video", "all"}:
        all_results.extend(run_jobs(video_jobs(args, out_root, logs), args.max_parallel, args.skip_existing))

    manifest_path = out_root / "runner_manifest.json"
    existing = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            manifest_args = manifest.get("args", {})
            same_root = canonicalize_path(manifest_args.get("output_root", out_root)) == canonicalize_path(out_root)
            old_sources = manifest.get("resolved_source_roots") or manifest_args.get("resolved_source_roots") or {}
            new_sources = resolved_source_roots(args)
            same_sources = old_sources == new_sources
            if same_root and same_sources:
                existing = manifest.get("jobs", [])
        except Exception:
            existing = []
    manifest_path.write_text(json.dumps({
        "args": vars(args),
        "resolved_source_roots": resolved_source_roots(args),
        "jobs": existing + all_results,
    }, indent=2))
    print(f"[runner] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
