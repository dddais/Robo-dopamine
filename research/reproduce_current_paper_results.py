#!/usr/bin/env python3
"""Reproduce the current no-GPU Memory-GRM paper artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_MD = ROOT / "research_outputs/current_paper_reproduction.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/current_paper_reproduction.json"


PIPELINE = [
    {
        "name": "cached_grm_aggregation",
        "script": "research/aggregate_cached_grm_results.py",
        "outputs": [
            "research_outputs/cached_grm_aggregation.md",
            "research_outputs/cached_grm_metrics.csv",
            "research_outputs/cached_grm_rows.csv",
        ],
    },
    {
        "name": "trajectory_memory_features",
        "script": "research/analyze_trajectory_memory_features.py",
        "outputs": [
            "research_outputs/trajectory_memory_features.md",
            "research_outputs/trajectory_memory_metrics.csv",
            "research_outputs/trajectory_memory_cases.csv",
        ],
    },
    {
        "name": "benchmark_v0_validation",
        "script": "research/validate_benchmark_v0.py",
        "outputs": [
            "research_outputs/benchmark_v0_validation.md",
            "research_outputs/benchmark_v0_validation.json",
        ],
    },
    {
        "name": "radio_event_memory_mvp",
        "script": "research/evaluate_radio_event_memory_mvp.py",
        "outputs": [
            "research_outputs/radio_event_memory_mvp.md",
            "research_outputs/radio_event_memory_mvp.json",
        ],
    },
    {
        "name": "benchmark_v0_event_memory",
        "script": "research/evaluate_benchmark_v0_event_memory.py",
        "outputs": [
            "research_outputs/benchmark_v0_event_memory_eval.md",
            "research_outputs/benchmark_v0_event_memory_eval.json",
        ],
    },
    {
        "name": "radio_event_counterfactuals",
        "script": "research/evaluate_radio_event_counterfactuals.py",
        "outputs": [
            "research_outputs/radio_event_counterfactuals.md",
            "research_outputs/radio_event_counterfactuals.json",
        ],
    },
    {
        "name": "paper_ready_summary",
        "script": "research/make_paper_ready_summary.py",
        "outputs": [
            "research_outputs/paper_ready_results_summary.md",
            "research_outputs/paper_ready_results_summary.json",
        ],
    },
    {
        "name": "claim_evidence_ledger",
        "script": "research/make_claim_evidence_ledger.py",
        "outputs": [
            "research_outputs/claim_evidence_ledger.md",
            "research_outputs/claim_evidence_ledger.json",
        ],
    },
    {
        "name": "radio_intake_inventory",
        "script": "research/inventory_radio_intake_candidates.py",
        "outputs": [
            "research_outputs/radio_intake_candidates.md",
            "research_outputs/radio_intake_candidates.json",
        ],
    },
    {
        "name": "radio_intake_keyframes",
        "script": "research/extract_radio_intake_keyframes.py",
        "outputs": [
            "research_outputs/radio_intake_keyframes.md",
            "research_outputs/radio_intake_keyframes.json",
        ],
    },
    {
        "name": "pending_radio_event_window",
        "script": "research/extract_pending_radio_event_window.py",
        "outputs": [
            "research_outputs/radio_intake_event_windows/xzx_episode_1_sub2_event_window.json",
            "research_outputs/radio_intake_event_windows/xzx_episode_1_sub2_event_window.md",
            "research_outputs/radio_intake_event_windows/xzx_episode_1_sub2_window_540_649/contact_sheet.png",
        ],
    },
    {
        "name": "paper_latex_tables",
        "script": "research/export_paper_latex_tables.py",
        "outputs": [
            "research_outputs/paper_tables.tex",
        ],
    },
    {
        "name": "radio_case_figure",
        "script": "research/make_radio_case_figure.py",
        "outputs": [
            "research_outputs/figures/radio_memory_grm_case.png",
            "research_outputs/figures/radio_memory_grm_case.pdf",
            "research_outputs/figures/radio_memory_grm_case.tex",
            "research_outputs/figures/radio_memory_grm_case.md",
            "research_outputs/figures/radio_memory_grm_case.json",
        ],
    },
    {
        "name": "manuscript_package_check",
        "script": "research/check_manuscript_package.py",
        "outputs": [
            "research_outputs/manuscript_package_check.md",
            "research_outputs/manuscript_package_check.json",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed step and report all failures.",
    )
    return parser.parse_args()


def run_step(step: dict) -> dict:
    cmd = [sys.executable, step["script"]]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    outputs = []
    for rel_path in step["outputs"]:
        path = ROOT / rel_path
        outputs.append(
            {
                "path": rel_path,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return {
        "name": step["name"],
        "script": step["script"],
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "outputs": outputs,
    }


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_invariants() -> dict:
    validation = load_json_if_exists(ROOT / "research_outputs/benchmark_v0_validation.json")
    benchmark = load_json_if_exists(ROOT / "research_outputs/benchmark_v0_event_memory_eval.json")
    paper = load_json_if_exists(ROOT / "research_outputs/paper_ready_results_summary.json")
    ledger = load_json_if_exists(ROOT / "research_outputs/claim_evidence_ledger.json")
    intake = load_json_if_exists(ROOT / "research_outputs/radio_intake_candidates.json")
    intake_keyframes = load_json_if_exists(ROOT / "research_outputs/radio_intake_keyframes.json")
    event_window = load_json_if_exists(ROOT / "research_outputs/radio_intake_event_windows/xzx_episode_1_sub2_event_window.json")
    manuscript = load_json_if_exists(ROOT / "research_outputs/manuscript_package_check.json")
    episodes = load_json_if_exists(ROOT / "benchmark_v0/episodes.json")
    live_episode_ids = []
    if isinstance(episodes, list):
        live_episode_ids = [item.get("episode_id") for item in episodes]
    return {
        "benchmark_v0_valid": validation.get("valid") if validation else None,
        "benchmark_v0_errors": validation.get("num_errors") if validation else None,
        "benchmark_v0_warnings": validation.get("num_warnings") if validation else None,
        "num_evaluated": benchmark.get("aggregate", {}).get("num_evaluated") if benchmark else None,
        "live_benchmark_episode_ids": live_episode_ids,
        "verified_non_markovian_episodes": (
            paper.get("data_scope", {}).get("verified_non_markovian_episodes")
            if paper
            else None
        ),
        "manuscript_package_valid": manuscript.get("valid") if manuscript else None,
        "manuscript_package_errors": manuscript.get("num_errors") if manuscript else None,
        "manuscript_package_warnings": manuscript.get("num_warnings") if manuscript else None,
        "claim_ledger_valid": ledger.get("valid") if ledger else None,
        "claim_ledger_adds_labels": (
            ledger.get("data_scope", {}).get("adds_benchmark_labels") if ledger else None
        ),
        "radio_intake_candidates": intake.get("num_candidates") if intake else None,
        "radio_intake_pending": intake.get("num_pending") if intake else None,
        "radio_intake_adds_labels": intake.get("adds_benchmark_labels") if intake else None,
        "radio_intake_keyframes": intake_keyframes.get("num_extracted") if intake_keyframes else None,
        "radio_intake_keyframes_adds_labels": (
            intake_keyframes.get("adds_benchmark_labels") if intake_keyframes else None
        ),
        "pending_radio_event_window": event_window.get("candidate_episode_id") if event_window else None,
        "pending_radio_event_window_adds_labels": (
            event_window.get("adds_benchmark_labels") if event_window else None
        ),
        "pending_radio_event_window_frames": event_window.get("frames_extracted") if event_window else None,
    }


def write_report(payload: dict, out_md: Path) -> None:
    lines = [
        "# Current Paper Reproduction / 当前论文结果复现",
        "",
        "This pipeline reruns the current no-GPU Memory-GRM artifacts from cached results and human-verified event labels. It does not rerun Robo-Dopamine GRM inference and does not add benchmark labels.",
        "",
        "当前 pipeline 基于已有缓存结果和人工核验事件标签，重新生成当前 Memory-GRM 论文产物。它不会重新运行 Robo-Dopamine GRM 推理，也不会新增 benchmark 标签。",
        "",
        "## Data Scope / 数据边界",
        "",
        "- Verified non-Markovian benchmark episode: `xzx_radio_sub23` only.",
        "- Cached carrot/cube/bottle cases are visible-state baseline diagnostics and candidate material only.",
        "",
        "## Steps / 步骤",
        "",
        "| Step | Status | Seconds | Key outputs |",
        "|---|---:|---:|---|",
    ]
    for step in payload["steps"]:
        status = "PASS" if step["returncode"] == 0 else "FAIL"
        outputs = "<br>".join(
            f"`{item['path']}`" + ("" if item["exists"] else " (missing)")
            for item in step["outputs"]
        )
        lines.append(
            f"| `{step['name']}` | {status} | {step['elapsed_sec']:.2f} | {outputs} |"
        )

    inv = payload["invariants"]
    lines += [
        "",
        "## Invariants / 不变量",
        "",
        f"- Benchmark v0 valid: `{inv['benchmark_v0_valid']}`",
        f"- Benchmark v0 errors/warnings: `{inv['benchmark_v0_errors']}` / `{inv['benchmark_v0_warnings']}`",
        f"- Evaluated Benchmark v0 episodes: `{inv['num_evaluated']}`",
        f"- Live benchmark episode ids: `{', '.join(inv['live_benchmark_episode_ids'])}`",
        f"- Verified non-Markovian episodes in paper summary: `{', '.join(inv['verified_non_markovian_episodes'] or [])}`",
        f"- Manuscript package valid: `{inv['manuscript_package_valid']}`",
        f"- Manuscript package errors/warnings: `{inv['manuscript_package_errors']}` / `{inv['manuscript_package_warnings']}`",
        f"- Claim ledger valid / adds labels: `{inv['claim_ledger_valid']}` / `{inv['claim_ledger_adds_labels']}`",
        f"- Radio intake candidates / pending / adds labels: `{inv['radio_intake_candidates']}` / `{inv['radio_intake_pending']}` / `{inv['radio_intake_adds_labels']}`",
        f"- Radio intake keyframe sheets / adds labels: `{inv['radio_intake_keyframes']}` / `{inv['radio_intake_keyframes_adds_labels']}`",
        f"- Pending radio event window / frames / adds labels: `{inv['pending_radio_event_window']}` / `{inv['pending_radio_event_window_frames']}` / `{inv['pending_radio_event_window_adds_labels']}`",
        "",
        "## Reproducibility / 可复现性",
        "",
        "Command:",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/reproduce_current_paper_results.py",
        "```",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    steps = []
    failed = False
    for step in PIPELINE:
        result = run_step(step)
        steps.append(result)
        if result["returncode"] != 0:
            failed = True
            if not args.keep_going:
                break

    payload = {
        "pipeline_type": "no_gpu_cached_reproduction",
        "uses_gpu": False,
        "adds_benchmark_labels": False,
        "steps": steps,
        "invariants": summarize_invariants(),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, args.out_md)

    print(f"steps={len(steps)}")
    print(f"failed={str(failed).lower()}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
