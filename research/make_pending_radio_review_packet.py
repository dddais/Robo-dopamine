#!/usr/bin/env python3
"""Build a human-review packet for a pending turn-on-radio candidate.

The packet is deliberately non-mutating: it reads keyframe/window/scaffold
artifacts and writes review guidance under research_outputs only. It does not
create benchmark labels and does not modify benchmark_v0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = "xzx_episode_1_sub2"
DEFAULT_SCaffold_ID = "xzx_radio_sub2_pending_review"
DEFAULT_OUT_MD = ROOT / "research_outputs/pending_radio_review_packet.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/pending_radio_review_packet.json"
REQUIRED_EVENTS = ["grasp", "lift", "button_press", "indicator_green", "place", "release"]
EVENT_PRIORS = {
    "grasp": "Use the full-trajectory contact sheet first; refine with dense frames if the object is already lifted before the GRM event window.",
    "lift": "Use the full-trajectory contact sheet first; record the earliest frame where the radio is clearly off the table.",
    "button_press": "Inspect the left-wrist view around the GRM jump, especially frames 600, 610, 620, and 630.",
    "indicator_green": "Inspect the left-wrist view around and after the GRM jump, especially frames 620, 630, and 640.",
    "place": "Use the later full-trajectory frames and final frame; add a denser late window if the release is not visible.",
    "release": "Use the later full-trajectory frames and final frame; add a denser late window if hand separation is ambiguous.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--scaffold-id", default=DEFAULT_SCaffold_ID)
    parser.add_argument(
        "--keyframe-manifest",
        type=Path,
        default=None,
        help="Full-trajectory keyframe manifest. Defaults to the candidate's radio intake keyframe manifest.",
    )
    parser.add_argument(
        "--event-window-summary",
        type=Path,
        default=None,
        help="Dense event-window summary JSON. Defaults to the candidate's pending event-window JSON.",
    )
    parser.add_argument(
        "--event-window-manifest",
        type=Path,
        default=None,
        help="Dense event-window manifest. If omitted, the path is read from the summary JSON.",
    )
    parser.add_argument(
        "--scaffold-episode",
        type=Path,
        default=None,
        help="Scaffolded episode entry JSON. Defaults to research_outputs/scaffolded_radio_episodes/<scaffold-id>/episode_entry.json.",
    )
    parser.add_argument(
        "--scaffold-events",
        type=Path,
        default=None,
        help="Scaffolded event template JSON. Defaults to research_outputs/scaffolded_radio_episodes/<scaffold-id>/<scaffold-id>_events.json.",
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    keyframe_manifest = args.keyframe_manifest or (
        ROOT / "research_outputs/radio_intake_keyframes" / args.candidate_id / "manifest.json"
    )
    event_window_summary = args.event_window_summary or (
        ROOT / "research_outputs/radio_intake_event_windows" / f"{args.candidate_id}_event_window.json"
    )
    scaffold_dir = ROOT / "research_outputs/scaffolded_radio_episodes" / args.scaffold_id
    scaffold_episode = args.scaffold_episode or scaffold_dir / "episode_entry.json"
    scaffold_events = args.scaffold_events or scaffold_dir / f"{args.scaffold_id}_events.json"
    return {
        "keyframe_manifest": keyframe_manifest,
        "event_window_summary": event_window_summary,
        "scaffold_episode": scaffold_episode,
        "scaffold_events": scaffold_events,
    }


def manifest_frames(manifest: dict) -> list[dict]:
    rows = []
    for item in manifest.get("frames", []):
        rows.append(
            {
                "frame_id": item.get("frame_id"),
                "time_sec": item.get("time_sec"),
                "views": item.get("views", {}),
            }
        )
    return rows


def collect_grm_jumps(event_manifest: dict) -> list[dict]:
    jumps = []
    for mode_summary in event_manifest.get("window", {}).get("mode_summaries", []):
        pred_file = mode_summary.get("pred_file")
        for rank, item in enumerate(mode_summary.get("top_positive_jumps", []), start=1):
            jumps.append(
                {
                    "rank_within_mode": rank,
                    "pred_file": pred_file,
                    "delta": item.get("delta"),
                    "before": item.get("before"),
                    "after": item.get("after"),
                    "frames": item.get("frames", []),
                    "step_id": item.get("step_id"),
                }
            )
    return sorted(jumps, key=lambda item: item.get("delta") or 0.0, reverse=True)


def unresolved_scaffold_events(scaffold_events: dict) -> list[str]:
    unresolved = []
    for item in scaffold_events.get("events", []):
        if item.get("frame_id") is None or item.get("time_index") is None or not item.get("view_evidence"):
            unresolved.append(item.get("event"))
    return unresolved


def make_event_review_rows(keyframes: list[dict], window_frames: list[dict]) -> list[dict]:
    keyframe_ids = [item["frame_id"] for item in keyframes]
    window_frame_ids = [item["frame_id"] for item in window_frames]
    dense_views = {}
    for item in window_frames:
        dense_views[item["frame_id"]] = item.get("views", {})

    rows = []
    for event in REQUIRED_EVENTS:
        if event in {"button_press", "indicator_green"}:
            candidate_frames = [fid for fid in window_frame_ids if fid is not None]
            primary_source = "dense_event_window"
        elif event in {"grasp", "lift"}:
            candidate_frames = [fid for fid in keyframe_ids if fid is not None and fid <= 500]
            primary_source = "full_trajectory_keyframes"
        else:
            candidate_frames = [fid for fid in keyframe_ids if fid is not None and fid >= 500]
            primary_source = "full_trajectory_keyframes"

        rows.append(
            {
                "event": event,
                "status": "needs_human_review",
                "primary_source": primary_source,
                "candidate_frames": candidate_frames,
                "preferred_views": ["cam_left_wrist"] if event in {"button_press", "indicator_green"} else ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                "guidance": EVENT_PRIORS[event],
                "evidence_template": [
                    dense_views.get(frame, {}) for frame in candidate_frames[:3]
                ]
                if primary_source == "dense_event_window"
                else [],
            }
        )
    return rows


def build_payload(args: argparse.Namespace) -> dict:
    paths = default_paths(args)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing required review inputs: {missing_text}")

    event_window_summary = load_json(paths["event_window_summary"])
    if args.event_window_manifest:
        event_window_manifest_path = args.event_window_manifest
    else:
        manifest_rel = event_window_summary.get("manifest")
        if not manifest_rel:
            raise ValueError("event-window summary does not contain a manifest path")
        event_window_manifest_path = ROOT / manifest_rel
    if not event_window_manifest_path.exists():
        raise FileNotFoundError(f"Missing event-window manifest: {event_window_manifest_path}")

    keyframe_manifest = load_json(paths["keyframe_manifest"])
    event_window_manifest = load_json(event_window_manifest_path)
    scaffold_episode = load_json(paths["scaffold_episode"])
    scaffold_events = load_json(paths["scaffold_events"])
    keyframes = manifest_frames(keyframe_manifest)
    window_frames = manifest_frames(event_window_manifest)
    review_rows = make_event_review_rows(keyframes, window_frames)

    blocking_reasons = [
        "Local image viewer is unavailable in the current sandbox, so no new visual event labels are asserted.",
        "The scaffold still has null frame/time/evidence fields for required events.",
        "The scaffold episode still has a null success_label and TODO cached_pred_path entries.",
    ]
    if event_window_summary.get("adds_benchmark_labels") is not False:
        blocking_reasons.append("Event-window summary does not explicitly state adds_benchmark_labels=false.")
    if keyframe_manifest.get("benchmark_label_status") != "not_a_new_benchmark_label":
        blocking_reasons.append("Keyframe manifest does not explicitly mark benchmark_label_status=not_a_new_benchmark_label.")

    payload = {
        "candidate_episode_id": args.candidate_id,
        "scaffold_episode_id": args.scaffold_id,
        "review_status": "pending_human_verification",
        "adds_benchmark_labels": False,
        "modifies_live_benchmark": False,
        "live_benchmark_inclusion_allowed": False,
        "inputs": {
            "keyframe_manifest": rel(paths["keyframe_manifest"]),
            "keyframe_contact_sheet": rel(paths["keyframe_manifest"].parent / "contact_sheet.png"),
            "event_window_summary": rel(paths["event_window_summary"]),
            "event_window_manifest": rel(event_window_manifest_path),
            "event_window_contact_sheet": event_window_summary.get("contact_sheet"),
            "scaffold_episode": rel(paths["scaffold_episode"]),
            "scaffold_events": rel(paths["scaffold_events"]),
        },
        "video_metadata": keyframe_manifest.get("video_metadata", {}),
        "dense_window": {
            "window_start": event_window_summary.get("window_start"),
            "window_end": event_window_summary.get("window_end"),
            "suggested_frame_min": event_window_summary.get("suggested_frame_min"),
            "suggested_frame_max": event_window_summary.get("suggested_frame_max"),
            "frames_extracted": event_window_summary.get("frames_extracted"),
        },
        "top_grm_positive_jumps": collect_grm_jumps(event_window_manifest)[:6],
        "event_review_rows": review_rows,
        "unresolved_required_events": unresolved_scaffold_events(scaffold_events),
        "success_label": scaffold_episode.get("success_label"),
        "label_status": scaffold_episode.get("label_status"),
        "cached_pred_paths": scaffold_episode.get("cached_pred_path", {}),
        "blocking_reasons": blocking_reasons,
        "required_human_decisions": [
            "Confirm or reject button_press with frame_id, time_index, view_evidence, confidence, and notes.",
            "Confirm or reject indicator_green with frame_id, time_index, view_evidence, confidence, and notes.",
            "Confirm grasp/lift/place/release event order or request additional dense windows.",
            "Set success_label only after all required events and negative latches are reviewed.",
            "Replace cached_pred_path TODOs before any live Benchmark v0 inclusion.",
        ],
    }
    return payload


def fmt_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "NA"


def write_report(payload: dict, out_md: Path) -> None:
    lines = [
        "# Pending Radio Human-Review Packet / Radio 待审核复核包",
        "",
        "This packet organizes evidence for a pending turn-on-radio candidate. It is not a label file and it does not modify Benchmark v0.",
        "",
        "该复核包只整理待审核证据，不是标签文件，也不会修改 Benchmark v0。",
        "",
        "## Scope / 范围",
        "",
        f"- Candidate episode: `{payload['candidate_episode_id']}`",
        f"- Scaffold episode: `{payload['scaffold_episode_id']}`",
        f"- Review status: `{payload['review_status']}`",
        f"- Adds benchmark labels: `{payload['adds_benchmark_labels']}`",
        f"- Modifies live benchmark: `{payload['modifies_live_benchmark']}`",
        f"- Live benchmark inclusion allowed now: `{payload['live_benchmark_inclusion_allowed']}`",
        "",
        "## Evidence Files / 证据文件",
        "",
    ]
    for label, path in payload["inputs"].items():
        lines.append(f"- {label}: `{path}`")

    dense = payload["dense_window"]
    lines += [
        "",
        "## Dense Window / 密集窗口",
        "",
        f"- Frame window: `{dense['window_start']}` to `{dense['window_end']}`",
        f"- GRM-suggested frames: `{dense['suggested_frame_min']}` to `{dense['suggested_frame_max']}`",
        f"- Frames extracted: `{dense['frames_extracted']}`",
        "",
        "## Top GRM Positive Jumps / GRM 主要正跳变",
        "",
        "| Delta | Before | After | Frames | Prediction file |",
        "|---:|---:|---:|---|---|",
    ]
    for item in payload["top_grm_positive_jumps"]:
        frames = ", ".join(str(frame) for frame in item.get("frames", []))
        lines.append(
            f"| {fmt_float(item.get('delta'))} | {fmt_float(item.get('before'))} | {fmt_float(item.get('after'))} | `{frames}` | `{item.get('pred_file')}` |"
        )

    lines += [
        "",
        "## Event Review Worksheet / 事件复核表",
        "",
        "| Event | Status | Primary source | Candidate frames | Preferred views | Guidance |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["event_review_rows"]:
        frames = ", ".join(str(frame) for frame in row["candidate_frames"])
        views = ", ".join(row["preferred_views"])
        lines.append(
            f"| `{row['event']}` | `{row['status']}` | `{row['primary_source']}` | `{frames}` | `{views}` | {row['guidance']} |"
        )

    lines += [
        "",
        "## Current Blockers / 当前阻塞",
        "",
    ]
    for reason in payload["blocking_reasons"]:
        lines.append(f"- {reason}")

    lines += [
        "",
        "## Required Human Decisions / 需要人工决定",
        "",
    ]
    for decision in payload["required_human_decisions"]:
        lines.append(f"- {decision}")

    lines += [
        "",
        "## Benchmark Gate / Benchmark 接入门槛",
        "",
        "Keep this candidate outside `benchmark_v0/episodes.json` until all required events, negative latches, success label, and cached GRM paths are complete and reviewed.",
        "",
        "在必需事件、负事件 latch、成功标签和 GRM 缓存路径全部完成并复核前，不要把该候选加入 `benchmark_v0/episodes.json`。",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, args.out_md)
    print(f"candidate={payload['candidate_episode_id']}")
    print(f"review_status={payload['review_status']}")
    print(f"adds_benchmark_labels={str(payload['adds_benchmark_labels']).lower()}")
    print(f"live_benchmark_inclusion_allowed={str(payload['live_benchmark_inclusion_allowed']).lower()}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")


if __name__ == "__main__":
    main()
