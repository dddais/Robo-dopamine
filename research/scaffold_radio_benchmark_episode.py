#!/usr/bin/env python3
"""Scaffold Benchmark v0 files for a new turn-on-radio episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "benchmark_v0"
DEFAULT_TASK = (
    "pick up the red radio, press the switch with the left hand until the "
    "indicator light turns green, then put the radio down"
)
REQUIRED_EVENTS = ["grasp", "lift", "button_press", "indicator_green", "place", "release"]
NEGATIVE_LATCHES = [
    "drop",
    "slip",
    "wrong_object",
    "wrong_target",
    "collision",
    "forbidden_contact",
    "order_violation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to aligned episode directory containing cam_high/cam_left_wrist/cam_right_wrist mp4 files.",
    )
    parser.add_argument(
        "--label",
        choices=["success", "failure", "unknown"],
        default="unknown",
        help="Initial success label for the scaffold.",
    )
    parser.add_argument(
        "--case-type",
        choices=[
            "radio_success_hidden_green",
            "radio_no_green",
            "radio_no_press",
            "radio_wrong_order",
            "radio_negative_latch",
            "radio_visual_decoy",
        ],
        default="radio_success_hidden_green",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "research_outputs/scaffolded_radio_episodes",
        help="Directory where scaffold files are written by default. This avoids modifying the live benchmark until labels are verified.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def validate_videos(data_dir: Path) -> dict:
    videos = {
        "front": data_dir / "cam_high.mp4",
        "left_wrist": data_dir / "cam_left_wrist.mp4",
        "right_wrist": data_dir / "cam_right_wrist.mp4",
    }
    missing = [str(path) for path in videos.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required videos: {missing}")

    infos = {name: video_info(path) for name, path in videos.items()}
    return {
        "videos": videos,
        "infos": infos,
    }


def label_value(label: str) -> bool | None:
    if label == "success":
        return True
    if label == "failure":
        return False
    return None


def make_episode(args: argparse.Namespace, video_data: dict) -> dict:
    data_dir = Path(args.data_dir)
    infos = video_data["infos"]
    front = infos["front"]
    left = infos["left_wrist"]
    return {
        "episode_id": args.episode_id,
        "task": args.task,
        "video_path": {
            "front": rel(data_dir / "cam_high.mp4"),
            "left_wrist": rel(data_dir / "cam_left_wrist.mp4"),
            "right_wrist": rel(data_dir / "cam_right_wrist.mp4"),
        },
        "video_metadata": {
            "frames": front["frames"],
            "fps": front["fps"],
            "front_resolution": [front["width"], front["height"]],
            "wrist_resolution": [left["width"], left["height"]],
        },
        "cached_pred_path": {
            "forward": f"TODO/results/{args.episode_id}/forward/pred_vllm.json",
            "incremental": f"TODO/results/{args.episode_id}/incremental/pred_vllm.json",
            "backward": f"TODO/results/{args.episode_id}/backward/pred_vllm.json",
            "summary": f"TODO/results/{args.episode_id}/run_summary.json",
            "progress_curve": f"TODO/results/{args.episode_id}/progress_curve.png",
        },
        "object": "red_radio",
        "success_label": label_value(args.label),
        "label_status": "pending_human_verification",
        "event_labels": f"event_annotations/{args.episode_id}_events.json",
        "non_markovian_rule": {
            "type": args.case_type,
            "required_events": REQUIRED_EVENTS,
            "description": (
                "The episode succeeds only if the radio is picked up, the switch is pressed "
                "with the left hand, and the indicator light turns green before the radio is put down. "
                "The final frame may not show the green light."
            ),
        },
        "notes": (
            "Scaffold only. Fill event labels from keyframe inspection, run GRM, "
            "replace cached_pred_path TODOs, then run research/validate_benchmark_v0.py."
        ),
    }


def make_events(args: argparse.Namespace) -> dict:
    return {
        "episode_id": args.episode_id,
        "annotation_status": "pending_human_verification",
        "task": args.task,
        "success_rule": {
            "type": "ordered_required_events",
            "required_order": REQUIRED_EVENTS,
            "non_markovian_events": ["button_press", "indicator_green"],
            "description": (
                "The episode is successful only if the switch is pressed, the indicator turns green, "
                "and the radio is released after those events."
            ),
        },
        "events": [
            {
                "event": name,
                "time_index": None,
                "frame_id": None,
                "view_evidence": [],
                "confidence": None,
                "source": "pending_human_verification",
                "notes": "TODO: fill from keyframe inspection.",
            }
            for name in REQUIRED_EVENTS
        ],
        "negative_event_latches": {name: False for name in NEGATIVE_LATCHES},
        "human_correction_notes": (
            "TODO: document keyframe inspection, ambiguous events, and whether this is a real "
            "episode or constructed/counterfactual history."
        ),
    }


def write_json(path: Path, data: object, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(path: Path, args: argparse.Namespace, episode_path: Path, events_path: Path, video_data: dict) -> None:
    infos = video_data["infos"]
    lines = [
        "# Scaffolded Radio Benchmark Episode / Radio Benchmark 样本模板",
        "",
        "This scaffold is intentionally written outside the live benchmark by default. Review and complete it before copying into `benchmark_v0/episodes.json` and `benchmark_v0/event_annotations/`.",
        "",
        "该模板默认写入 research_outputs，不直接修改 live benchmark。请先完成人工核验和 GRM 路径替换，再合并到 `benchmark_v0/episodes.json` 和 `benchmark_v0/event_annotations/`。",
        "",
        "## Episode / 样本",
        "",
        f"- Episode id: `{args.episode_id}`",
        f"- Case type: `{args.case_type}`",
        f"- Initial label: `{args.label}`",
        f"- Data dir: `{args.data_dir}`",
        f"- Episode template: `{rel(episode_path)}`",
        f"- Event template: `{rel(events_path)}`",
        "",
        "## Video Metadata / 视频元数据",
        "",
        "| View | Frames | FPS | Resolution |",
        "|---|---:|---:|---|",
    ]
    for view, info in infos.items():
        lines.append(
            f"| `{view}` | {info['frames']} | {info['fps']:.2f} | {info['width']}x{info['height']} |"
        )
    lines += [
        "",
        "## Required Next Steps / 后续步骤",
        "",
        "1. Extract event-window keyframes for `grasp`, `lift`, `button_press`, `indicator_green`, `place`, and `release`.",
        "2. Fill frame ids, timestamps, evidence paths, confidence, and notes in the event template.",
        "3. Set `success_label` and `label_status` only after human verification.",
        "4. Run GRM for forward/incremental/backward modes and replace `cached_pred_path` TODOs.",
        "5. Copy finalized entries into the live benchmark.",
        "6. Run validation and evaluation:",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/validate_benchmark_v0.py",
        "conda run -n robo-dopamine python research/evaluate_benchmark_v0_event_memory.py",
        "conda run -n robo-dopamine python research/make_paper_ready_summary.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    video_data = validate_videos(data_dir)
    out_dir = args.out_dir / args.episode_id
    episode_path = out_dir / "episode_entry.json"
    events_path = out_dir / f"{args.episode_id}_events.json"
    report_path = out_dir / "README.md"

    episode = make_episode(args, video_data)
    events = make_events(args)
    write_json(episode_path, episode, args.force)
    write_json(events_path, events, args.force)
    write_report(report_path, args, episode_path, events_path, video_data)

    print(f"episode_template={episode_path}")
    print(f"event_template={events_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
