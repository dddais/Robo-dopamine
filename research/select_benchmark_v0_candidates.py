#!/usr/bin/env python3
"""Select Benchmark v0 candidate cases from existing cached GRM analyses."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHED_ROWS = ROOT / "research_outputs/cached_grm_rows.csv"
TRAJECTORY_ROWS = ROOT / "research_outputs/trajectory_memory_cases.csv"
OUT_JSON = ROOT / "research_outputs/benchmark_v0_candidate_cases.json"
OUT_MD = ROOT / "research_outputs/benchmark_v0_candidate_cases.md"


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: str) -> bool:
    return str(value).lower() == "true"


def top_high_scoring_negatives(rows: list[dict], limit: int) -> list[dict]:
    candidates = [
        row
        for row in rows
        if row["model_tag"] == "GRM-2.0-8B"
        and row["source_summary"] == "summary_GRM8B_new"
        and as_bool(row["task_matches_scene"])
        and not as_bool(row["goal_success"])
    ]
    candidates.sort(key=lambda row: float(row["avg_progress"]), reverse=True)
    out = []
    for row in candidates[:limit]:
        out.append(
            {
                "priority": "high_scoring_negative",
                "data_tag": row["data_tag"],
                "scene_object": row["scene_object"],
                "task_tag": row["task_tag"],
                "interval": int(row["interval"]),
                "avg_progress": float(row["avg_progress"]),
                "forward_progress": float(row["forward_progress"]),
                "incremental_progress": float(row["incremental_progress"]),
                "backward_progress": float(row["backward_progress"]),
                "why": (
                    "Failed or non-goal-success episode with high final fused GRM score; "
                    "inspect for visually plausible failure, transient violation, or missing event labels."
                ),
            }
        )
    return out


def top_score_regressions(rows: list[dict], limit: int) -> list[dict]:
    candidates = [
        row
        for row in rows
        if as_bool(row["task_matches_scene"]) and not as_bool(row["goal_success"])
    ]
    candidates.sort(key=lambda row: float(row["max_drawdown_avg"]), reverse=True)
    out = []
    for row in candidates[:limit]:
        out.append(
            {
                "priority": "large_score_regression",
                "data_tag": row["data_tag"],
                "scene_object": row["scene_object"],
                "task_tag": row["task_tag"],
                "interval": int(row["interval"]),
                "baseline_final_avg": float(row["baseline_final_avg"]),
                "trajectory_mean_avg": float(row["trajectory_mean_avg"]),
                "max_drawdown_avg": float(row["max_drawdown_avg"]),
                "neg_hop_sum_avg": float(row["neg_hop_sum_avg"]),
                "why": (
                    "Trajectory has large progress drawdown; inspect for slip, drop, recovery, "
                    "or other transient events that scalar final progress may miss."
                ),
            }
        )
    return out


def dedupe(cases: list[dict], limit: int) -> list[dict]:
    seen = set()
    out = []
    for case in cases:
        key = (case["priority"], case["data_tag"], case["task_tag"], case["interval"])
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
        if len(out) >= limit:
            break
    return out


def write_report(cases: list[dict]) -> None:
    lines = [
        "# Benchmark v0 Candidate Cases / Benchmark v0 候选案例",
        "",
        "## English Summary",
        "",
        "This report selects cached cases for qualitative inspection and baseline diagnosis. It uses only existing CSV outputs and does not rerun GRM. These candidates are not current non-Markovian benchmark labels.",
        "",
        "Selection policy:",
        "",
        "- high-scoring negatives: failed/non-goal-success episodes that GRM scores highly;",
        "- large score regressions: episodes with strong trajectory drawdown that may contain transient events.",
        "",
        "## 中文总结",
        "",
        "本报告从已有缓存结果中筛选定性检查和 baseline 诊断候选案例；只读取现有 CSV，不重新运行 GRM。这些候选不是当前非马尔可夫 benchmark 标签。",
        "",
        "筛选策略：",
        "",
        "- high-scoring negatives：失败或非目标成功轨迹，但 GRM 给出高分；",
        "- large score regressions：分数轨迹有明显回落，可能包含 slip、drop、recovery 等短暂事件。",
        "",
        "## Candidates / 候选列表",
        "",
        "| Priority | Data | Task | Interval | Main score | Why inspect |",
        "|---|---|---|---:|---:|---|",
    ]
    for case in cases:
        if case["priority"] == "high_scoring_negative":
            score = case["avg_progress"]
        else:
            score = case["max_drawdown_avg"]
        lines.append(
            f"| {case['priority']} | `{case['data_tag']}` | `{case['task_tag']}` | {case['interval']} | {score:.2f} | {case['why']} |"
        )

    lines += [
        "",
        "## Recommended Next Manual Step / 建议下一步人工检查",
        "",
        "English: Start with the top 3 high-scoring negatives and the top 3 large-regression cases. For each case, extract/contact-sheet keyframes and mark visible events. Keep a case as non-Markovian only if manual inspection proves that history changes the success/failure label under similar final visual evidence; otherwise keep it as a visible-state baseline diagnostic.",
        "",
        "中文：优先检查前 3 个高分负例和前 3 个大回落案例。每个案例先抽关键帧/contact sheet，再标注可见事件。只有当人工检查证明在相似终态视觉证据下历史会改变成败标签时，才纳入非马尔可夫 benchmark；否则保留为可见状态 baseline 诊断。",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cached_rows = load_csv(CACHED_ROWS)
    trajectory_rows = load_csv(TRAJECTORY_ROWS)
    cases = dedupe(
        top_high_scoring_negatives(cached_rows, 6)
        + top_score_regressions(trajectory_rows, 6),
        limit=10,
    )
    OUT_JSON.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(cases)
    print(f"selected={len(cases)}")
    print(f"wrote={OUT_JSON}")
    print(f"wrote={OUT_MD}")
    for case in cases:
        print(case["priority"], case["data_tag"], case["task_tag"], case["interval"])


if __name__ == "__main__":
    main()
