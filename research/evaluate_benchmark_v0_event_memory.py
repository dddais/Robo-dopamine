#!/usr/bin/env python3
"""Evaluate EventMemory monitors over Benchmark v0 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from evaluate_radio_event_memory_mvp import evaluate, fmt_pct, load_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = ROOT / "benchmark_v0/episodes.json"
DEFAULT_BENCHMARK_ROOT = ROOT / "benchmark_v0"
DEFAULT_OUT_MD = ROOT / "research_outputs/benchmark_v0_event_memory_eval.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/benchmark_v0_event_memory_eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
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


def resolve_root_path(path_value: str | Path, base: Path = ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base / path


def resolve_event_path(path_value: str | Path, benchmark_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    benchmark_relative = benchmark_root / path
    if benchmark_relative.exists():
        return benchmark_relative
    return ROOT / path


def summarize_non_markovian_evidence(events_doc: dict) -> str:
    non_markovian = events_doc.get("success_rule", {}).get("non_markovian_events", [])
    events = {item["event"]: item for item in events_doc.get("events", [])}
    evidence = []
    for name in non_markovian:
        event = events.get(name)
        if not event:
            evidence.append(f"{name}: missing")
            continue
        evidence.append(f"{name}: frame {event.get('frame_id')}")
    return "; ".join(evidence) if evidence else "event rule only"


def bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def decision_correct(decision: dict, label: bool | None) -> bool | None:
    if label is None:
        return None
    return bool(decision["success"]) == label


def evaluate_episode(
    episode: dict,
    benchmark_root: Path,
    final_threshold: float,
    peak_threshold: float,
) -> dict:
    cached_paths = episode.get("cached_pred_path", {})
    if "summary" not in cached_paths:
        return {
            "episode_id": episode.get("episode_id", "unknown"),
            "status": "skipped",
            "reason": "missing cached_pred_path.summary",
        }
    if "event_labels" not in episode:
        return {
            "episode_id": episode.get("episode_id", "unknown"),
            "status": "skipped",
            "reason": "missing event_labels",
        }

    summary_path = resolve_root_path(cached_paths["summary"])
    event_path = resolve_event_path(episode["event_labels"], benchmark_root)
    if not summary_path.exists():
        return {
            "episode_id": episode.get("episode_id", "unknown"),
            "status": "skipped",
            "reason": f"missing summary file: {summary_path}",
        }
    if not event_path.exists():
        return {
            "episode_id": episode.get("episode_id", "unknown"),
            "status": "skipped",
            "reason": f"missing event file: {event_path}",
        }

    summary = load_json(summary_path)
    events_doc = load_json(event_path)
    result = evaluate(summary, events_doc, final_threshold, peak_threshold)
    label = bool_or_none(episode.get("success_label"))
    decisions = result["decisions"]

    return {
        "episode_id": episode["episode_id"],
        "status": "evaluated",
        "task": episode.get("task", result.get("task")),
        "success_label": label,
        "label_status": episode.get("label_status"),
        "annotation_status": events_doc.get("annotation_status"),
        "non_markovian_rule": episode.get("non_markovian_rule", {}),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "event_path": str(event_path.relative_to(ROOT)),
        "non_markovian_evidence": summarize_non_markovian_evidence(events_doc),
        "result": result,
        "correct": {
            "final_only_grm": decision_correct(decisions["final_only_grm"], label),
            "score_memory_grm": decision_correct(decisions["score_memory_grm"], label),
            "event_latched_grm": decision_correct(decisions["event_latched_grm"], label),
        },
    }


def aggregate(evaluations: list[dict]) -> dict:
    evaluated = [item for item in evaluations if item["status"] == "evaluated"]
    skipped = [item for item in evaluations if item["status"] != "evaluated"]
    labeled = [item for item in evaluated if item.get("success_label") is not None]

    monitor_names = ["final_only_grm", "score_memory_grm", "event_latched_grm"]
    accuracy = {}
    for name in monitor_names:
        correctness = [item["correct"][name] for item in labeled]
        accuracy[name] = (
            mean(1.0 if value else 0.0 for value in correctness)
            if correctness
            else None
        )

    final_scores = [
        item["result"]["grm"]["fused_final_progress"] for item in evaluated
    ]
    peak_scores = [item["result"]["grm"]["fused_peak_progress"] for item in evaluated]

    return {
        "num_episodes": len(evaluations),
        "num_evaluated": len(evaluated),
        "num_skipped": len(skipped),
        "num_labeled": len(labeled),
        "accuracy": accuracy,
        "mean_fused_final_progress": mean(final_scores) if final_scores else None,
        "mean_fused_peak_progress": mean(peak_scores) if peak_scores else None,
        "skipped": skipped,
        "data_scope_note": (
            "Benchmark v0 currently contains one human-verified non-Markovian "
            "episode, xzx_radio_sub23. Cached carrot/cube candidates are not "
            "counted as non-Markovian benchmark cases under the current data."
        ),
    }


def decision_text(result: dict, name: str) -> str:
    decision = result["decisions"][name]
    return "success" if decision["success"] else "not success"


def correct_text(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "correct" if value else "wrong"


def write_report(payload: dict, out_md: Path) -> None:
    aggregate_result = payload["aggregate"]
    evaluations = payload["episodes"]
    evaluated = [item for item in evaluations if item["status"] == "evaluated"]
    skipped = [item for item in evaluations if item["status"] != "evaluated"]

    lines = [
        "# Benchmark v0 EventMemory Evaluation / Benchmark v0 事件记忆评估",
        "",
        "## English Summary",
        "",
        "This report evaluates the current Benchmark v0 through a generic episode loader. It reuses cached GRM summaries and human-verified event labels; it does not rerun GRM and does not claim automatic event detection.",
        "",
        f"- Episodes in `benchmark_v0/episodes.json`: {aggregate_result['num_episodes']}",
        f"- Evaluated episodes: {aggregate_result['num_evaluated']}",
        f"- Labeled episodes: {aggregate_result['num_labeled']}",
        "- Current non-Markovian data scope: only `xzx_radio_sub23` is counted as a verified non-Markovian benchmark episode.",
        "- Cached carrot/cube candidates remain qualitative inspection material, not non-Markovian benchmark labels.",
        "",
        "## 中文总结",
        "",
        "本报告通过通用 episode loader 评估当前 Benchmark v0。它复用已有 GRM summary 和人工核验事件标签；不重新运行 GRM，也不声称自动事件检测已经完成。",
        "",
        f"- `benchmark_v0/episodes.json` 中 episode 数量：{aggregate_result['num_episodes']}",
        f"- 已评估 episode 数量：{aggregate_result['num_evaluated']}",
        f"- 有成功标签的 episode 数量：{aggregate_result['num_labeled']}",
        "- 当前非马尔可夫数据范围：只有 `xzx_radio_sub23` 被计入已核验非马尔可夫 benchmark episode。",
        "- 已缓存 carrot/cube 候选只作为定性检查材料，不作为当前非马尔可夫 benchmark 标签。",
        "",
        "## Aggregate Results / 汇总结果",
        "",
        "| Monitor | Accuracy on labeled Benchmark v0 |",
        "|---|---:|",
    ]
    for name, label in [
        ("final_only_grm", "Final-only GRM / 只看终态 GRM"),
        ("score_memory_grm", "Score-memory GRM / 分数轨迹记忆 GRM"),
        ("event_latched_grm", "Event-Latched GRM / 事件锁存 GRM"),
    ]:
        value = aggregate_result["accuracy"][name]
        text = "n/a" if value is None else fmt_pct(100.0 * value)
        lines.append(f"| {label} | {text} |")

    if aggregate_result["mean_fused_final_progress"] is not None:
        lines += [
            "",
            f"- Mean fused final progress: {fmt_pct(aggregate_result['mean_fused_final_progress'])}",
            f"- Mean fused peak progress: {fmt_pct(aggregate_result['mean_fused_peak_progress'])}",
        ]

    lines += [
        "",
        "## Episode Results / 单 episode 结果",
        "",
        "| Episode | Rule | Label | Final | Peak | Final-only | Score-memory | Event-latched | Non-Markovian evidence |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for item in evaluated:
        result = item["result"]
        grm = result["grm"]
        rule_type = item["non_markovian_rule"].get("type", "unknown")
        label = item["success_label"]
        label_text = "n/a" if label is None else str(label).lower()
        lines.append(
            "| "
            f"`{item['episode_id']}` | "
            f"`{rule_type}` | "
            f"{label_text} | "
            f"{fmt_pct(grm['fused_final_progress'])} | "
            f"{fmt_pct(grm['fused_peak_progress'])} | "
            f"{decision_text(result, 'final_only_grm')} ({correct_text(item['correct']['final_only_grm'])}) | "
            f"{decision_text(result, 'score_memory_grm')} ({correct_text(item['correct']['score_memory_grm'])}) | "
            f"{decision_text(result, 'event_latched_grm')} ({correct_text(item['correct']['event_latched_grm'])}) | "
            f"{item['non_markovian_evidence']} |"
        )

    if skipped:
        lines += [
            "",
            "## Skipped Episodes / 跳过的 episode",
            "",
            "| Episode | Reason |",
            "|---|---|",
        ]
        for item in skipped:
            lines.append(f"| `{item['episode_id']}` | {item['reason']} |")

    lines += [
        "",
        "## Reproducibility / 可复现性",
        "",
        "Command:",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/evaluate_benchmark_v0_event_memory.py",
        "```",
        "",
        "Inputs:",
        "",
        "- `benchmark_v0/episodes.json`",
        "- `results/xzx_episode_1_sub23_memory_grm/run_summary.json`",
        "- `benchmark_v0/event_annotations/xzx_radio_sub23_events.json`",
        "",
        "Outputs:",
        "",
        "- `research_outputs/benchmark_v0_event_memory_eval.md`",
        "- `research_outputs/benchmark_v0_event_memory_eval.json`",
        "",
        "Limitation / 限制：the current evaluated benchmark has one verified non-Markovian episode. The next publishable step is collecting or constructing additional final-state-similar histories, not re-labeling ordinary visible-state failures as non-Markovian cases.",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    episodes = load_json(args.episodes)
    evaluations = [
        evaluate_episode(
            episode,
            args.benchmark_root,
            args.final_success_threshold,
            args.score_memory_threshold,
        )
        for episode in episodes
    ]
    payload = {
        "episodes_path": str(args.episodes.relative_to(ROOT)),
        "thresholds": {
            "final_success_threshold": args.final_success_threshold,
            "score_memory_threshold": args.score_memory_threshold,
        },
        "aggregate": aggregate(evaluations),
        "episodes": evaluations,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(payload, args.out_md)

    print(f"episodes={payload['aggregate']['num_episodes']}")
    print(f"evaluated={payload['aggregate']['num_evaluated']}")
    print(f"labeled={payload['aggregate']['num_labeled']}")
    for name, value in payload["aggregate"]["accuracy"].items():
        text = "n/a" if value is None else f"{100.0 * value:.2f}%"
        print(f"{name}_accuracy={text}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")


if __name__ == "__main__":
    main()
