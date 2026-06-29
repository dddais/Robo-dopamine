#!/usr/bin/env python3
"""Generate paper-ready tables from existing Memory-GRM experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "research_outputs/paper_ready_results_summary.md"
OUT_JSON = ROOT / "research_outputs/paper_ready_results_summary.json"

METRICS_CSV = ROOT / "research_outputs/cached_grm_metrics.csv"
TRAJ_CSV = ROOT / "research_outputs/trajectory_memory_metrics.csv"
BENCH_JSON = ROOT / "research_outputs/benchmark_v0_event_memory_eval.json"
COUNTERFACTUAL_JSON = ROOT / "research_outputs/radio_event_counterfactuals.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def pct_from_percent(value: float) -> str:
    return f"{value:.2f}%"


def find_metric_row(rows: list[dict], **criteria: object) -> dict:
    for row in rows:
        ok = True
        for key, expected in criteria.items():
            value = row.get(key)
            if isinstance(expected, int):
                ok = ok and int(value) == expected
            else:
                ok = ok and value == expected
        if ok:
            return row
    raise KeyError(f"No metric row for {criteria}")


def cached_grm_table() -> list[dict]:
    rows = load_csv(METRICS_CSV)
    specs = [
        ("fused final, interval 10", {"model": "GRM-2.0-8B", "interval": 10, "score": "avg_progress"}),
        ("fused final, interval 20", {"model": "GRM-2.0-8B", "interval": 20, "score": "avg_progress"}),
        ("forward only, interval 10", {"model": "GRM-2.0-8B", "interval": 10, "score": "forward_progress"}),
        ("forward only, interval 20", {"model": "GRM-2.0-8B", "interval": 20, "score": "forward_progress"}),
    ]
    table = []
    for label, criteria in specs:
        row = find_metric_row(rows, **criteria)
        table.append(
            {
                "setting": label,
                "auroc": as_float(row, "auc"),
                "best_f1": as_float(row, "f1"),
                "accuracy": as_float(row, "accuracy"),
            }
        )
    return table


def trajectory_table() -> list[dict]:
    rows = load_csv(TRAJ_CSV)
    feature_order = [
        "baseline_final_avg",
        "trajectory_mean_avg",
        "minus_drawdown_0.5",
        "minus_neg_hop_0.25",
        "grid_memory_score",
    ]
    labels = {
        "baseline_final_avg": "baseline final average",
        "trajectory_mean_avg": "trajectory mean average",
        "minus_drawdown_0.5": "final - 0.5 drawdown",
        "minus_neg_hop_0.25": "final - 0.25 negative-hop",
        "grid_memory_score": "best grid scalar-memory score",
    }
    table = []
    for feature in feature_order:
        row = find_metric_row(rows, feature=feature)
        table.append(
            {
                "feature": labels[feature],
                "auroc": as_float(row, "auc"),
                "best_f1": as_float(row, "f1"),
                "accuracy": as_float(row, "accuracy"),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
            }
        )
    return table


def benchmark_table(bench: dict) -> list[dict]:
    accuracy = bench["aggregate"]["accuracy"]
    episode = bench["episodes"][0]
    grm = episode["result"]["grm"]
    return [
        {
            "monitor": "Final-only GRM",
            "accuracy": accuracy["final_only_grm"],
            "decision": episode["result"]["decisions"]["final_only_grm"]["success"],
            "evidence": f"final fused progress {grm['fused_final_progress']:.2f}%",
        },
        {
            "monitor": "Score-memory GRM",
            "accuracy": accuracy["score_memory_grm"],
            "decision": episode["result"]["decisions"]["score_memory_grm"]["success"],
            "evidence": f"peak fused progress {grm['fused_peak_progress']:.2f}%",
        },
        {
            "monitor": "Event-Latched GRM",
            "accuracy": accuracy["event_latched_grm"],
            "decision": episode["result"]["decisions"]["event_latched_grm"]["success"],
            "evidence": episode["non_markovian_evidence"],
        },
    ]


def counterfactual_table(counterfactual: dict) -> list[dict]:
    rows = []
    for key, label in [
        ("final_only_grm", "Final-only GRM"),
        ("score_memory_grm", "Score-memory GRM"),
        ("event_latched_grm", "Event-Latched GRM"),
    ]:
        rows.append(
            {
                "monitor": label,
                "accuracy": counterfactual["accuracy"][key],
                "balanced_accuracy": counterfactual["balanced_accuracy"][key],
                "valid_history_recall": counterfactual["positive_recall"][key],
                "invalid_history_recall": counterfactual["negative_recall"][key],
            }
        )
    return rows


def build_payload() -> dict:
    bench = load_json(BENCH_JSON)
    counterfactual = load_json(COUNTERFACTUAL_JSON)
    return {
        "source_files": {
            "cached_grm_metrics": str(METRICS_CSV.relative_to(ROOT)),
            "trajectory_memory_metrics": str(TRAJ_CSV.relative_to(ROOT)),
            "benchmark_v0_eval": str(BENCH_JSON.relative_to(ROOT)),
            "radio_counterfactuals": str(COUNTERFACTUAL_JSON.relative_to(ROOT)),
        },
        "data_scope": {
            "verified_non_markovian_episodes": ["xzx_radio_sub23"],
            "note": (
                "Only the turn-on-radio episode is currently verified as "
                "non-Markovian. Cached carrot/cube/bottle cases are baseline "
                "diagnostics, not non-Markovian benchmark labels."
            ),
        },
        "cached_grm_visible_state": cached_grm_table(),
        "score_memory_visible_state": trajectory_table(),
        "benchmark_v0_radio": benchmark_table(bench),
        "radio_event_counterfactuals": counterfactual_table(counterfactual),
    }


def write_report(payload: dict) -> None:
    lines = [
        "# Paper-Ready Results Summary / 论文结果汇总",
        "",
        "## Data Scope / 数据边界",
        "",
        "English: The current verified non-Markovian benchmark contains one episode: `xzx_radio_sub23` / turn-on-radio. Cached carrot/cube/bottle cases are used only as visible-state baseline diagnostics and candidate material.",
        "",
        "中文：当前已核验非马尔可夫 benchmark 只有一个 episode：`xzx_radio_sub23` / turn-on-radio。已缓存 carrot/cube/bottle 案例只作为可见状态 baseline 诊断和候选材料。",
        "",
        "## Table 1: Visible-State GRM Baseline / 可见状态 GRM baseline",
        "",
        "| Setting | AUROC | Best F1 | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["cached_grm_visible_state"]:
        lines.append(
            f"| {row['setting']} | {pct(row['auroc'])} | {row['best_f1']:.3f} | {pct(row['accuracy'])} |"
        )

    lines += [
        "",
        "Interpretation: Robo-Dopamine GRM-2.0-8B is already strong for ordinary visible-state success/failure separation. The paper should not claim that GRM is broadly weak.",
        "",
        "## Table 2: Score-Only Temporal Memory / 仅分数轨迹记忆",
        "",
        "| Feature | AUROC | Best F1 | Accuracy | FP | FN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["score_memory_visible_state"]:
        lines.append(
            f"| {row['feature']} | {pct(row['auroc'])} | {row['best_f1']:.3f} | {pct(row['accuracy'])} | {row['fp']} | {row['fn']} |"
        )

    lines += [
        "",
        "Interpretation: Scalar trajectory evidence helps on visible-state cached data, but it cannot encode hidden event predicates when score evidence is unchanged.",
        "",
        "## Table 3: Current Benchmark v0 Radio Result / 当前 Benchmark v0 radio 结果",
        "",
        "| Monitor | Accuracy on labeled Benchmark v0 | Decision | Evidence |",
        "|---|---:|---|---|",
    ]
    for row in payload["benchmark_v0_radio"]:
        decision = "success" if row["decision"] else "not success"
        lines.append(
            f"| {row['monitor']} | {pct(row['accuracy'])} | {decision} | {row['evidence']} |"
        )

    lines += [
        "",
        "Interpretation: On the verified turn-on-radio episode, final-only GRM misses the success because the decisive evidence is an intermediate event. Event-Latched GRM recovers the success by remembering `button_press` and `indicator_green`.",
        "",
        "## Table 4: Radio Event Counterfactuals / Radio 事件反事实",
        "",
        "| Monitor | Accuracy | Balanced accuracy | Valid-history recall | Invalid-history recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["radio_event_counterfactuals"]:
        lines.append(
            f"| {row['monitor']} | {pct(row['accuracy'])} | {pct(row['balanced_accuracy'])} | {pct(row['valid_history_recall'])} | {pct(row['invalid_history_recall'])} |"
        )

    lines += [
        "",
        "Interpretation: All counterfactual variants share the same GRM score curve. Final-only and score-memory monitors collapse to constant decisions, while Event-Latched GRM separates valid and invalid event histories.",
        "",
        "## Claims Supported Now / 当前可支撑的论文主张",
        "",
        "- GRM is a strong visible-state progress baseline.",
        "- A single verified turn-on-radio case demonstrates hidden intermediate success evidence.",
        "- Event-latched memory can represent required events, order violations, and negative latches under identical scalar score evidence.",
        "- The current evidence is a feasibility study, not a statistically complete non-Markovian benchmark.",
        "",
        "## Claims Not Yet Supported / 当前不能声称",
        "",
        "- Do not claim a completed large-scale non-Markovian benchmark.",
        "- Do not claim automatic event detection; current key radio events are human-verified.",
        "- Do not count carrot/cube/bottle cached candidates as non-Markovian labels without manual proof of final-state-similar, history-different episodes.",
        "",
        "## Reproducibility / 可复现性",
        "",
        "Generated from:",
        "",
    ]
    for name, path in payload["source_files"].items():
        lines.append(f"- `{name}`: `{path}`")

    lines += [
        "",
        "Command:",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/make_paper_ready_summary.py",
        "```",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"wrote={OUT_MD}")
    print(f"wrote={OUT_JSON}")
    print(f"verified_non_markovian={','.join(payload['data_scope']['verified_non_markovian_episodes'])}")


if __name__ == "__main__":
    main()
