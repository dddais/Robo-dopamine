#!/usr/bin/env python3
"""Validate Benchmark v0 episode and event annotation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "benchmark_v0"
DEFAULT_EPISODES = BENCHMARK_ROOT / "episodes.json"
DEFAULT_OUT_MD = ROOT / "research_outputs/benchmark_v0_validation.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/benchmark_v0_validation.json"
EXPECTED_NON_MARKOVIAN_EPISODES = {"xzx_radio_sub23"}
REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "task",
    "video_path",
    "cached_pred_path",
    "success_label",
    "label_status",
    "event_labels",
    "non_markovian_rule",
}
REQUIRED_EVENT_FIELDS = {
    "event",
    "time_index",
    "frame_id",
    "view_evidence",
    "confidence",
    "source",
}
REQUIRED_LATCHES = {
    "drop",
    "slip",
    "wrong_object",
    "wrong_target",
    "collision",
    "forbidden_contact",
    "order_violation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def root_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def event_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    benchmark_relative = BENCHMARK_ROOT / path
    if benchmark_relative.exists():
        return benchmark_relative
    return ROOT / path


def add_issue(issues: list[dict], severity: str, episode_id: str, message: str) -> None:
    issues.append({"severity": severity, "episode_id": episode_id, "message": message})


def validate_path_map(
    issues: list[dict],
    episode_id: str,
    field_name: str,
    path_map: object,
    required_keys: set[str] | None = None,
) -> None:
    if not isinstance(path_map, dict):
        add_issue(issues, "error", episode_id, f"`{field_name}` must be an object")
        return
    if required_keys:
        missing = sorted(required_keys - set(path_map))
        if missing:
            add_issue(issues, "error", episode_id, f"`{field_name}` missing keys: {missing}")
    for key, value in path_map.items():
        path = root_path(value)
        if not path.exists():
            add_issue(issues, "error", episode_id, f"`{field_name}.{key}` does not exist: {value}")


def validate_events_doc(episode: dict, events_doc: dict, issues: list[dict]) -> dict:
    episode_id = episode["episode_id"]
    annotation_status = events_doc.get("annotation_status")
    if episode.get("label_status") != annotation_status:
        add_issue(
            issues,
            "warning",
            episode_id,
            f"episode label_status `{episode.get('label_status')}` differs from annotation_status `{annotation_status}`",
        )

    if events_doc.get("episode_id") != episode_id:
        add_issue(
            issues,
            "error",
            episode_id,
            f"event annotation episode_id `{events_doc.get('episode_id')}` does not match",
        )

    success_rule = events_doc.get("success_rule", {})
    required_order = success_rule.get("required_order", [])
    non_markovian_events = success_rule.get("non_markovian_events", [])
    if not isinstance(required_order, list) or not required_order:
        add_issue(issues, "error", episode_id, "`success_rule.required_order` must be a non-empty list")
        required_order = []
    if not isinstance(non_markovian_events, list):
        add_issue(issues, "error", episode_id, "`success_rule.non_markovian_events` must be a list")
        non_markovian_events = []

    events = events_doc.get("events", [])
    if not isinstance(events, list):
        add_issue(issues, "error", episode_id, "`events` must be a list")
        events = []

    by_name = {}
    confidences = []
    for event in events:
        if not isinstance(event, dict):
            add_issue(issues, "error", episode_id, "event entry must be an object")
            continue
        missing = sorted(REQUIRED_EVENT_FIELDS - set(event))
        name = str(event.get("event", "unknown"))
        if missing:
            add_issue(issues, "error", episode_id, f"event `{name}` missing fields: {missing}")
        if name in by_name:
            add_issue(issues, "error", episode_id, f"duplicate event `{name}`")
        by_name[name] = event

        frame_id = event.get("frame_id")
        time_index = event.get("time_index")
        confidence = event.get("confidence")
        if not isinstance(frame_id, int):
            add_issue(issues, "error", episode_id, f"event `{name}` frame_id must be int")
        if not isinstance(time_index, (int, float)):
            add_issue(issues, "error", episode_id, f"event `{name}` time_index must be numeric")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            add_issue(issues, "error", episode_id, f"event `{name}` confidence must be in [0, 1]")
        else:
            confidences.append(float(confidence))

        evidence = event.get("view_evidence", [])
        if not isinstance(evidence, list) or not evidence:
            add_issue(issues, "error", episode_id, f"event `{name}` needs non-empty view_evidence")
        else:
            for evidence_path in evidence:
                if not root_path(evidence_path).exists():
                    add_issue(
                        issues,
                        "error",
                        episode_id,
                        f"event `{name}` evidence does not exist: {evidence_path}",
                    )

    for name in required_order:
        if name not in by_name:
            add_issue(issues, "error", episode_id, f"required event `{name}` is missing")
    for name in non_markovian_events:
        if name not in by_name:
            add_issue(issues, "error", episode_id, f"non-Markovian event `{name}` is missing")

    previous = -1
    order_ok = True
    for name in required_order:
        event = by_name.get(name)
        if not event or not isinstance(event.get("frame_id"), int):
            order_ok = False
            continue
        if event["frame_id"] < previous:
            add_issue(issues, "error", episode_id, f"required event order violated at `{name}`")
            order_ok = False
        previous = event["frame_id"]

    latches = events_doc.get("negative_event_latches", {})
    if not isinstance(latches, dict):
        add_issue(issues, "error", episode_id, "`negative_event_latches` must be an object")
        latches = {}
    missing_latches = sorted(REQUIRED_LATCHES - set(latches))
    if missing_latches:
        add_issue(issues, "error", episode_id, f"missing negative latches: {missing_latches}")
    for name, value in latches.items():
        if not isinstance(value, bool):
            add_issue(issues, "error", episode_id, f"negative latch `{name}` must be bool")

    return {
        "annotation_status": annotation_status,
        "required_events": required_order,
        "non_markovian_events": non_markovian_events,
        "num_events": len(events),
        "order_ok": order_ok,
        "mean_confidence": mean(confidences) if confidences else None,
        "active_negative_latches": [name for name, value in latches.items() if value is True],
    }


def validate_episode(episode: dict, seen_ids: set[str]) -> dict:
    episode_id = str(episode.get("episode_id", "unknown"))
    issues: list[dict] = []
    missing = sorted(REQUIRED_EPISODE_FIELDS - set(episode))
    if missing:
        add_issue(issues, "error", episode_id, f"episode missing fields: {missing}")

    if episode_id in seen_ids:
        add_issue(issues, "error", episode_id, "duplicate episode_id")
    seen_ids.add(episode_id)

    if episode_id not in EXPECTED_NON_MARKOVIAN_EPISODES:
        add_issue(
            issues,
            "error",
            episode_id,
            "unexpected non-Markovian episode in current data scope",
        )

    validate_path_map(
        issues,
        episode_id,
        "video_path",
        episode.get("video_path"),
        required_keys={"front", "left_wrist", "right_wrist"},
    )
    validate_path_map(
        issues,
        episode_id,
        "cached_pred_path",
        episode.get("cached_pred_path"),
        required_keys={"forward", "incremental", "backward", "summary"},
    )

    if not isinstance(episode.get("success_label"), bool):
        add_issue(issues, "error", episode_id, "`success_label` must be bool")

    rule = episode.get("non_markovian_rule", {})
    if not isinstance(rule, dict):
        add_issue(issues, "error", episode_id, "`non_markovian_rule` must be an object")
        rule = {}
    required_events = rule.get("required_events", [])
    if not isinstance(required_events, list) or not required_events:
        add_issue(issues, "error", episode_id, "`non_markovian_rule.required_events` must be a non-empty list")

    events_file = event_path(episode.get("event_labels", ""))
    event_summary = None
    if not events_file.exists():
        add_issue(issues, "error", episode_id, f"event_labels file does not exist: {episode.get('event_labels')}")
    else:
        events_doc = load_json(events_file)
        if not isinstance(events_doc, dict):
            add_issue(issues, "error", episode_id, "event_labels file must contain an object")
        else:
            event_summary = validate_events_doc(episode, events_doc, issues)
            if required_events and event_summary:
                if required_events != event_summary["required_events"]:
                    add_issue(
                        issues,
                        "error",
                        episode_id,
                        "episode required_events differ from event success_rule.required_order",
                    )

    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    return {
        "episode_id": episode_id,
        "valid": not errors,
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "event_summary": event_summary,
        "issues": issues,
    }


def validate(episodes_path: Path) -> dict:
    episodes = load_json(episodes_path)
    if not isinstance(episodes, list):
        return {
            "valid": False,
            "num_episodes": 0,
            "num_errors": 1,
            "num_warnings": 0,
            "episodes": [],
            "issues": [
                {
                    "severity": "error",
                    "episode_id": "global",
                    "message": "`episodes.json` must contain a list",
                }
            ],
        }

    seen_ids: set[str] = set()
    episode_results = [validate_episode(episode, seen_ids) for episode in episodes]
    issues = [issue for result in episode_results for issue in result["issues"]]
    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    return {
        "valid": not errors,
        "num_episodes": len(episodes),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "expected_non_markovian_episodes": sorted(EXPECTED_NON_MARKOVIAN_EPISODES),
        "episodes": episode_results,
        "issues": issues,
    }


def write_report(result: dict, out_md: Path) -> None:
    status = "PASS" if result["valid"] else "FAIL"
    lines = [
        "# Benchmark v0 Validation / Benchmark v0 校验",
        "",
        "## Summary / 总结",
        "",
        f"- Status: **{status}**",
        f"- Episodes: {result['num_episodes']}",
        f"- Errors: {result['num_errors']}",
        f"- Warnings: {result['num_warnings']}",
        f"- Expected current non-Markovian episodes: `{', '.join(result.get('expected_non_markovian_episodes', []))}`",
        "",
        "This validator checks file existence, required fields, event order, non-Markovian event presence, negative latch types, and the current data-scope invariant that only `xzx_radio_sub23` is a verified non-Markovian episode.",
        "",
        "该校验器检查文件存在性、必需字段、事件顺序、非马尔可夫事件是否存在、负事件 latch 类型，以及当前数据边界：只有 `xzx_radio_sub23` 是已核验非马尔可夫 episode。",
        "",
        "## Episodes / Episodes",
        "",
        "| Episode | Valid | Events | Order OK | Non-Markovian events | Mean confidence | Issues |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for episode in result["episodes"]:
        summary = episode.get("event_summary") or {}
        mean_conf = summary.get("mean_confidence")
        mean_conf_text = "n/a" if mean_conf is None else f"{mean_conf:.2f}"
        lines.append(
            "| "
            f"`{episode['episode_id']}` | "
            f"{str(episode['valid']).lower()} | "
            f"{summary.get('num_events', 'n/a')} | "
            f"{str(summary.get('order_ok', 'n/a')).lower()} | "
            f"`{', '.join(summary.get('non_markovian_events', []))}` | "
            f"{mean_conf_text} | "
            f"{episode['num_errors']} errors / {episode['num_warnings']} warnings |"
        )

    if result["issues"]:
        lines += [
            "",
            "## Issues / 问题",
            "",
            "| Severity | Episode | Message |",
            "|---|---|---|",
        ]
        for issue in result["issues"]:
            lines.append(
                f"| {issue['severity']} | `{issue['episode_id']}` | {issue['message']} |"
            )

    lines += [
        "",
        "## Reproducibility / 可复现性",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/validate_benchmark_v0.py",
        "```",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = validate(args.episodes)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result, args.out_md)
    print(f"status={'PASS' if result['valid'] else 'FAIL'}")
    print(f"episodes={result['num_episodes']}")
    print(f"errors={result['num_errors']}")
    print(f"warnings={result['num_warnings']}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
