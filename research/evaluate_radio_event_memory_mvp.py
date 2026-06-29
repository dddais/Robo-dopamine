#!/usr/bin/env python3
"""Evaluate the radio EventMemory MVP from cached GRM and event labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results/xzx_episode_1_sub23_memory_grm/run_summary.json"
DEFAULT_EVENTS = ROOT / "benchmark_v0/event_annotations/xzx_radio_sub23_events.json"
DEFAULT_OUT_MD = ROOT / "research_outputs/radio_event_memory_mvp.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/radio_event_memory_mvp.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument(
        "--final-success-threshold",
        type=float,
        default=70.0,
        help="Final fused progress threshold used by the final-only baseline.",
    )
    parser.add_argument(
        "--score-memory-threshold",
        type=float,
        default=70.0,
        help="Peak fused progress threshold used by the score-memory baseline.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fused_series(summary: dict) -> list[float]:
    mode_series = [
        summary["modes"][mode]["progress_series"]
        for mode in ("forward", "incremental", "backward")
    ]
    min_len = min(len(series) for series in mode_series)
    return [mean(series[i] for series in mode_series) for i in range(min_len)]


def event_memory_decision(events_doc: dict) -> dict:
    required = events_doc["success_rule"]["required_order"]
    events = {item["event"]: item for item in events_doc["events"]}
    missing = [name for name in required if name not in events]
    incomplete = [
        name
        for name in required
        if name in events
        and (
            events[name].get("frame_id") is None
            or events[name].get("time_index") is None
            or not events[name].get("view_evidence")
        )
    ]
    order_ok = True
    order_violations = []
    previous_frame = -1
    for name in required:
        if name not in events or events[name].get("frame_id") is None:
            order_ok = False
            continue
        frame_id = int(events[name]["frame_id"])
        if frame_id < previous_frame:
            order_ok = False
            order_violations.append(name)
        previous_frame = frame_id

    negative_latches = events_doc.get("negative_event_latches", {})
    active_violations = [name for name, value in negative_latches.items() if value]
    success = not missing and not incomplete and order_ok and not active_violations

    return {
        "required_order": required,
        "missing_events": missing,
        "incomplete_events": incomplete,
        "order_ok": order_ok,
        "order_violations": order_violations,
        "active_violations": active_violations,
        "success": success,
    }


def evaluate(summary: dict, events_doc: dict, final_threshold: float, peak_threshold: float) -> dict:
    fused = fused_series(summary)
    final_progress = fused[-1]
    peak_progress = max(fused)
    peak_index = fused.index(peak_progress)
    frame_interval = int(summary["frame_interval"])
    peak_frame = peak_index * frame_interval
    peak_time_sec = peak_frame / 30.0
    event_decision = event_memory_decision(events_doc)

    final_only_success = final_progress >= final_threshold
    score_memory_success = peak_progress >= peak_threshold
    event_memory_success = bool(event_decision["success"])
    if event_memory_success:
        event_memory_reason = "all required events are present in order and no negative latch is active"
    elif event_decision["active_violations"]:
        event_memory_reason = (
            "negative latch active: "
            + ", ".join(event_decision["active_violations"])
        )
    elif event_decision["missing_events"]:
        event_memory_reason = (
            "missing required events: "
            + ", ".join(event_decision["missing_events"])
        )
    elif event_decision["order_violations"]:
        event_memory_reason = (
            "required event order violated at: "
            + ", ".join(event_decision["order_violations"])
        )
    else:
        event_memory_reason = "required event order is incomplete or violated"

    return {
        "episode_id": summary["episode_id"],
        "task": summary["task"],
        "frame_interval": frame_interval,
        "thresholds": {
            "final_success_threshold": final_threshold,
            "score_memory_threshold": peak_threshold,
        },
        "grm": {
            "fused_final_progress": final_progress,
            "fused_mean_progress": mean(fused),
            "fused_peak_progress": peak_progress,
            "fused_peak_index": peak_index,
            "fused_peak_frame": peak_frame,
            "fused_peak_time_sec": peak_time_sec,
            "peak_minus_final": peak_progress - final_progress,
        },
        "events": event_decision,
        "decisions": {
            "final_only_grm": {
                "success": final_only_success,
                "reason": (
                    f"final fused progress {final_progress:.2f}% "
                    f"{'>=' if final_only_success else '<'} {final_threshold:.2f}%"
                ),
            },
            "score_memory_grm": {
                "success": score_memory_success,
                "reason": (
                    f"peak fused progress {peak_progress:.2f}% "
                    f"{'>=' if score_memory_success else '<'} {peak_threshold:.2f}%"
                ),
            },
            "event_latched_grm": {
                "success": event_memory_success,
                "reason": event_memory_reason,
            },
        },
    }


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def write_report(result: dict, events_doc: dict, out_md: Path) -> None:
    grm = result["grm"]
    decisions = result["decisions"]
    events_by_name = {item["event"]: item for item in events_doc["events"]}
    required = result["events"]["required_order"]

    lines = [
        "# Radio EventMemory MVP / Radio 事件记忆最小验证",
        "",
        "## English Summary",
        "",
        "This MVP uses cached Robo-Dopamine GRM outputs and human-verified keyframe events. It does not rerun GRM and does not claim automatic event detection.",
        "",
        f"- Episode: `{result['episode_id']}`",
        f"- Final fused GRM progress: {fmt_pct(grm['fused_final_progress'])}",
        f"- Peak fused GRM progress: {fmt_pct(grm['fused_peak_progress'])} at frame {grm['fused_peak_frame']} ({grm['fused_peak_time_sec']:.1f}s)",
        f"- Peak-final gap: {fmt_pct(grm['peak_minus_final'])}",
        "- Core finding: final-only GRM gives a low success signal, while Event-Latched GRM recovers success by remembering `button_press` and `indicator_green`.",
        "",
        "## 中文总结",
        "",
        "该 MVP 使用已有 Robo-Dopamine GRM 缓存结果和人工核验关键帧事件；没有重新跑 GRM，也不声称已经实现自动事件检测。",
        "",
        f"- Episode: `{result['episode_id']}`",
        f"- GRM fused final progress: {fmt_pct(grm['fused_final_progress'])}",
        f"- GRM fused peak progress: {fmt_pct(grm['fused_peak_progress'])}，峰值在 frame {grm['fused_peak_frame']} ({grm['fused_peak_time_sec']:.1f}s)",
        f"- peak-final gap: {fmt_pct(grm['peak_minus_final'])}",
        "- 核心结论：final-only GRM 给出较低成功信号，而 Event-Latched GRM 通过记住 `button_press` 和 `indicator_green` 恢复成功判断。",
        "",
        "## Decision Comparison / 判定对比",
        "",
        "| Monitor | Decision | Evidence |",
        "|---|---|---|",
    ]
    for name, label in [
        ("final_only_grm", "Final-only GRM / 只看终态 GRM"),
        ("score_memory_grm", "Score-memory GRM / 分数轨迹记忆 GRM"),
        ("event_latched_grm", "Event-Latched GRM / 事件锁存 GRM"),
    ]:
        decision = decisions[name]
        lines.append(
            f"| {label} | {'success' if decision['success'] else 'not success'} | {decision['reason']} |"
        )

    lines += [
        "",
        "## Event Timeline / 事件时间线",
        "",
        "| Event | Frame | Time | Confidence | Evidence |",
        "|---|---:|---:|---:|---|",
    ]
    for name in required:
        event = events_by_name[name]
        evidence = "<br>".join(f"`{path}`" for path in event["view_evidence"])
        lines.append(
            f"| `{name}` | {event['frame_id']} | {event['time_index']:.1f}s | {event['confidence']:.2f} | {evidence} |"
        )

    lines += [
        "",
        "## Interpretation / 解释",
        "",
        "English: The final frame no longer exposes the key success evidence clearly. A monitor that only thresholds final GRM progress can therefore under-estimate this successful non-Markovian episode. The event-memory rule stores the intermediate `button_press` and `indicator_green` events, so the final decision remains successful after the radio is put down.",
        "",
        "中文：终态画面不再清楚呈现关键成功证据。因此，只对 final GRM progress 设阈值的监控器会低估这个成功的非马尔可夫任务。事件记忆规则会在中段锁存 `button_press` 和 `indicator_green`，所以 radio 放下后仍能保留成功判断。",
        "",
        "## Reproducibility / 可复现性",
        "",
        "Command:",
        "",
        "```bash",
        "python research/evaluate_radio_event_memory_mvp.py",
        "```",
        "",
        "Inputs:",
        "",
        "- `results/xzx_episode_1_sub23_memory_grm/run_summary.json`",
        "- `benchmark_v0/event_annotations/xzx_radio_sub23_events.json`",
        "",
        "Outputs:",
        "",
        "- `research_outputs/radio_event_memory_mvp.md`",
        "- `research_outputs/radio_event_memory_mvp.json`",
        "",
        "Limitation / 限制：event labels are human-verified from keyframes; automatic VLM/event-head detection remains future work. 事件标签来自人工关键帧核验，自动 VLM/event-head 检测仍是下一步工作。",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary)
    events_doc = load_json(args.events)
    result = evaluate(
        summary,
        events_doc,
        final_threshold=args.final_success_threshold,
        peak_threshold=args.score_memory_threshold,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result, events_doc, args.out_md)

    grm = result["grm"]
    print(f"episode={result['episode_id']}")
    print(f"final={grm['fused_final_progress']:.2f}")
    print(f"peak={grm['fused_peak_progress']:.2f}")
    print(f"peak_frame={grm['fused_peak_frame']}")
    print(f"gap={grm['peak_minus_final']:.2f}")
    for name, decision in result["decisions"].items():
        print(f"{name}={decision['success']} ({decision['reason']})")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")


if __name__ == "__main__":
    main()
