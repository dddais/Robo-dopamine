#!/usr/bin/env python3
"""Run event-label counterfactual stress tests for the radio case."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from statistics import mean

from evaluate_radio_event_memory_mvp import evaluate, fmt_pct, load_json


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/xzx_episode_1_sub23_memory_grm/run_summary.json"
EVENTS_PATH = ROOT / "benchmark_v0/event_annotations/xzx_radio_sub23_events.json"
OUT_MD = ROOT / "research_outputs/radio_event_counterfactuals.md"
OUT_JSON = ROOT / "research_outputs/radio_event_counterfactuals.json"


def remove_events(events_doc: dict, names: set[str]) -> dict:
    variant = copy.deepcopy(events_doc)
    variant["events"] = [
        event for event in variant["events"] if event["event"] not in names
    ]
    return variant


def set_negative_latch(events_doc: dict, latch_name: str) -> dict:
    variant = copy.deepcopy(events_doc)
    variant.setdefault("negative_event_latches", {})[latch_name] = True
    return variant


def swap_event_order(events_doc: dict, earlier_name: str, later_name: str) -> dict:
    variant = copy.deepcopy(events_doc)
    events = {event["event"]: event for event in variant["events"]}
    earlier = events[earlier_name]
    later = events[later_name]
    earlier["frame_id"], later["frame_id"] = later["frame_id"], earlier["frame_id"]
    earlier["time_index"], later["time_index"] = later["time_index"], earlier["time_index"]
    earlier["notes"] = (
        earlier.get("notes", "")
        + " Synthetic counterfactual: event order was swapped for stress testing."
    ).strip()
    later["notes"] = (
        later.get("notes", "")
        + " Synthetic counterfactual: event order was swapped for stress testing."
    ).strip()
    return variant


def variants(events_doc: dict) -> list[dict]:
    return [
        {
            "variant_id": "observed_success",
            "label": True,
            "events_doc": copy.deepcopy(events_doc),
            "description": "Human-verified observed radio success event chain.",
        },
        {
            "variant_id": "missing_indicator_green",
            "label": False,
            "events_doc": remove_events(events_doc, {"indicator_green"}),
            "description": "Synthetic invalid history: the green indicator event is absent.",
        },
        {
            "variant_id": "missing_button_press",
            "label": False,
            "events_doc": remove_events(events_doc, {"button_press"}),
            "description": "Synthetic invalid history: the required switch press event is absent.",
        },
        {
            "variant_id": "indicator_before_button_press",
            "label": False,
            "events_doc": swap_event_order(
                events_doc, "button_press", "indicator_green"
            ),
            "description": "Synthetic invalid history: required event order is violated.",
        },
        {
            "variant_id": "forbidden_contact_latched",
            "label": False,
            "events_doc": set_negative_latch(events_doc, "forbidden_contact"),
            "description": "Synthetic invalid history: a negative event latch is active.",
        },
    ]


def decision_correct(result: dict, monitor: str, label: bool) -> bool:
    return bool(result["decisions"][monitor]["success"]) == label


def summarize(summary: dict, events_doc: dict) -> dict:
    rows = []
    for variant in variants(events_doc):
        result = evaluate(
            summary,
            variant["events_doc"],
            final_threshold=70.0,
            peak_threshold=70.0,
        )
        rows.append(
            {
                "variant_id": variant["variant_id"],
                "label": variant["label"],
                "description": variant["description"],
                "result": result,
                "correct": {
                    monitor: decision_correct(result, monitor, variant["label"])
                    for monitor in (
                        "final_only_grm",
                        "score_memory_grm",
                        "event_latched_grm",
                    )
                },
            }
        )

    accuracy = {}
    balanced_accuracy = {}
    positive_recall = {}
    negative_recall = {}
    for monitor in ("final_only_grm", "score_memory_grm", "event_latched_grm"):
        accuracy[monitor] = mean(1.0 if row["correct"][monitor] else 0.0 for row in rows)
        positives = [row for row in rows if row["label"]]
        negatives = [row for row in rows if not row["label"]]
        positive_recall[monitor] = mean(
            1.0 if row["correct"][monitor] else 0.0 for row in positives
        )
        negative_recall[monitor] = mean(
            1.0 if row["correct"][monitor] else 0.0 for row in negatives
        )
        balanced_accuracy[monitor] = (
            positive_recall[monitor] + negative_recall[monitor]
        ) / 2.0

    return {
        "stress_test_type": "event_label_counterfactuals_on_single_radio_trajectory",
        "source_summary": str(SUMMARY_PATH.relative_to(ROOT)),
        "source_events": str(EVENTS_PATH.relative_to(ROOT)),
        "num_variants": len(rows),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "variants": rows,
        "limitation": (
            "These are event-label counterfactuals over one observed trajectory. "
            "They test monitor logic under identical GRM score evidence; they are "
            "not additional real robot episodes."
        ),
    }


def decision_text(result: dict, monitor: str) -> str:
    return "success" if result["decisions"][monitor]["success"] else "not success"


def correct_text(value: bool) -> str:
    return "correct" if value else "wrong"


def write_report(payload: dict, out_md: Path) -> None:
    first = payload["variants"][0]["result"]["grm"]
    lines = [
        "# Radio Event Counterfactual Stress Test / Radio 事件反事实压力测试",
        "",
        "## English Summary",
        "",
        "This stress test reuses the same radio trajectory and the same cached GRM score curve, then changes only the event-memory labels. It checks whether a monitor can distinguish missing required events, order violations, and negative latches when scalar visual progress evidence is identical.",
        "",
        f"- Source trajectory: `{payload['source_summary']}`",
        f"- Source event labels: `{payload['source_events']}`",
        f"- Counterfactual variants: {payload['num_variants']}",
        f"- Shared fused final progress: {fmt_pct(first['fused_final_progress'])}",
        f"- Shared fused peak progress: {fmt_pct(first['fused_peak_progress'])}",
        "",
        "## 中文总结",
        "",
        "该压力测试复用同一条 radio 轨迹和同一条 GRM 分数曲线，只改变事件记忆标签。它用于检查：当标量视觉进度证据完全相同时，监控器能否区分缺失必要事件、顺序违规和负事件锁存。",
        "",
        f"- 源轨迹：`{payload['source_summary']}`",
        f"- 源事件标签：`{payload['source_events']}`",
        f"- 反事实变体数量：{payload['num_variants']}",
        f"- 共享 fused final progress：{fmt_pct(first['fused_final_progress'])}",
        f"- 共享 fused peak progress：{fmt_pct(first['fused_peak_progress'])}",
        "",
        "## Aggregate Results / 汇总结果",
        "",
        "| Monitor | Accuracy | Balanced accuracy | Valid-history recall | Invalid-history recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for monitor, label in [
        ("final_only_grm", "Final-only GRM / 只看终态 GRM"),
        ("score_memory_grm", "Score-memory GRM / 分数轨迹记忆 GRM"),
        ("event_latched_grm", "Event-Latched GRM / 事件锁存 GRM"),
    ]:
        lines.append(
            f"| {label} | "
            f"{fmt_pct(100.0 * payload['accuracy'][monitor])} | "
            f"{fmt_pct(100.0 * payload['balanced_accuracy'][monitor])} | "
            f"{fmt_pct(100.0 * payload['positive_recall'][monitor])} | "
            f"{fmt_pct(100.0 * payload['negative_recall'][monitor])} |"
        )

    lines += [
        "",
        "## Variant Results / 变体结果",
        "",
        "| Variant | Label | Final-only | Score-memory | Event-latched | Counterfactual meaning |",
        "|---|---:|---|---|---|---|",
    ]
    for row in payload["variants"]:
        result = row["result"]
        lines.append(
            "| "
            f"`{row['variant_id']}` | "
            f"{str(row['label']).lower()} | "
            f"{decision_text(result, 'final_only_grm')} ({correct_text(row['correct']['final_only_grm'])}) | "
            f"{decision_text(result, 'score_memory_grm')} ({correct_text(row['correct']['score_memory_grm'])}) | "
            f"{decision_text(result, 'event_latched_grm')} ({correct_text(row['correct']['event_latched_grm'])}) | "
            f"{row['description']} |"
        )

    lines += [
        "",
        "## Interpretation / 解释",
        "",
        "English: Final-only GRM makes the same decision for all histories because the final score is unchanged. Its high raw accuracy here is a class-imbalance artifact because four of five variants are invalid. Score-memory GRM also makes the same decision for all histories because the peak score is unchanged. Event-Latched GRM changes its decision when required events are missing, out of order, or invalidated by a negative latch.",
        "",
        "中文：Final-only GRM 对所有历史给出同一判定，因为最终分数不变。这里 raw accuracy 较高只是类别不均衡造成的表象，因为 5 个变体中 4 个是 invalid。Score-memory GRM 也对所有历史给出同一判定，因为峰值分数不变。Event-Latched GRM 会在必要事件缺失、顺序错误或负事件锁存时改变判定。",
        "",
        "## Limitation / 限制",
        "",
        payload["limitation"],
        "",
        "Command:",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/evaluate_radio_event_counterfactuals.py",
        "```",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = load_json(SUMMARY_PATH)
    events_doc = load_json(EVENTS_PATH)
    payload = summarize(summary, events_doc)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, OUT_MD)

    print(f"variants={payload['num_variants']}")
    for monitor, value in payload["accuracy"].items():
        print(f"{monitor}_accuracy={100.0 * value:.2f}%")
    for monitor, value in payload["balanced_accuracy"].items():
        print(f"{monitor}_balanced_accuracy={100.0 * value:.2f}%")
    print(f"wrote={OUT_MD}")
    print(f"wrote={OUT_JSON}")


if __name__ == "__main__":
    main()
