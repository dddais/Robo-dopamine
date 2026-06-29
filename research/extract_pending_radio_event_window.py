#!/usr/bin/env python3
"""Extract a dense review window for a pending radio intake candidate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = "xzx_episode_1_sub2"
DEFAULT_DATA_DIR = ROOT / "aligned_data/xzx_episode_1_sub2"
DEFAULT_PRED_ROOT = ROOT / "results/xzx_episode_1_sub2_new_prompt_ref_inter30"
DEFAULT_OUT_ROOT = ROOT / "research_outputs/radio_intake_event_windows"
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--margin", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def step_frames(step_id: str) -> list[int]:
    return [int(item) for item in re.findall(r"(?:bf|af)_(\d{6})", step_id)]


def infer_window(pred_root: Path, margin: int) -> dict:
    pred_files = sorted(pred_root.rglob("pred_vllm.json"))
    if not pred_files:
        raise FileNotFoundError(f"No pred_vllm.json under {pred_root}")

    summaries = []
    suggested_frames: list[int] = []
    for pred_file in pred_files:
        data = json.loads(pred_file.read_text(encoding="utf-8"))
        progress = [float(row.get("progress", 0.0)) * 100.0 for row in data]
        jumps = []
        for idx in range(1, len(progress)):
            step_id = data[idx].get("id", "")
            jumps.append(
                {
                    "index": idx,
                    "delta": progress[idx] - progress[idx - 1],
                    "before": progress[idx - 1],
                    "after": progress[idx],
                    "step_id": step_id,
                    "frames": step_frames(step_id),
                }
            )
        top = sorted(jumps, key=lambda item: item["delta"], reverse=True)[:3]
        focus = top[:1]
        for item in focus:
            suggested_frames.extend(item["frames"])
        summaries.append(
            {
                "pred_file": rel(pred_file),
                "num_steps": len(progress),
                "final_progress": progress[-1] if progress else None,
                "max_progress": max(progress) if progress else None,
                "top_positive_jumps": top,
            }
        )

    if not suggested_frames:
        raise RuntimeError("Could not infer event window from prediction ids")
    start = max(0, min(suggested_frames) - margin)
    end = max(suggested_frames) + margin
    return {
        "pred_root": rel(pred_root),
        "suggested_frame_min": min(suggested_frames),
        "suggested_frame_max": max(suggested_frames),
        "window_start": start,
        "window_end": end,
        "mode_summaries": summaries,
    }


def video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def extract_frame(video_path: Path, frame_id: int, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return True


def make_contact_sheet(manifest: dict, out_path: Path) -> None:
    thumb_size = (185, 185)
    label_h = 24
    margin = 8
    rows = len(CAMS)
    cols = len(manifest["frames"])
    sheet = Image.new(
        "RGB",
        (cols * (thumb_size[0] + margin) + margin, rows * (thumb_size[1] + label_h + margin) + margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for col, record in enumerate(manifest["frames"]):
        for row, cam in enumerate(CAMS):
            view_path = record["views"].get(cam)
            if view_path is None:
                continue
            img = Image.open(ROOT / view_path).convert("RGB")
            img.thumbnail(thumb_size)
            x = margin + col * (thumb_size[0] + margin)
            y = margin + row * (thumb_size[1] + label_h + margin)
            label = f"{cam} f{record['frame_id']} t={record['time_sec']:.1f}s"
            draw.text((x, y), label, fill=(0, 0, 0))
            sheet.paste(img, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def extract_window(args: argparse.Namespace, window: dict) -> dict:
    high_info = video_info(args.data_dir / "cam_high.mp4")
    start = max(0, int(window["window_start"]))
    end = min(high_info["frames"] - 1, int(window["window_end"]))
    frame_ids = list(range(start, end + 1, args.stride))
    for anchor in (window["suggested_frame_min"], window["suggested_frame_max"]):
        if 0 <= anchor < high_info["frames"]:
            frame_ids.append(anchor)
    frame_ids = sorted(set(frame_ids))

    window_dir_name = f"{args.candidate_id}_window_{start}_{end}"
    out_dir = args.out_root / window_dir_name
    manifest = {
        "candidate_episode_id": args.candidate_id,
        "source": rel(args.data_dir),
        "annotation_status": "pending_human_verification",
        "benchmark_label_status": "not_a_new_benchmark_label",
        "adds_benchmark_label": False,
        "purpose": "dense review window inferred from cached GRM progress jumps",
        "video_metadata": high_info,
        "window": window,
        "frame_stride": args.stride,
        "frames": [],
    }
    for frame_id in frame_ids:
        record = {"frame_id": frame_id, "time_sec": frame_id / high_info["fps"], "views": {}}
        for cam in CAMS:
            out_path = out_dir / f"frame_{frame_id:06d}_{cam}.png"
            ok = extract_frame(args.data_dir / f"{cam}.mp4", frame_id, out_path)
            record["views"][cam] = rel(out_path) if ok else None
        manifest["frames"].append(record)

    manifest_path = out_dir / "manifest.json"
    contact_sheet_path = out_dir / "contact_sheet.png"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(manifest, contact_sheet_path)
    return {
        "candidate_episode_id": args.candidate_id,
        "adds_benchmark_labels": False,
        "source": rel(args.data_dir),
        "manifest": rel(manifest_path),
        "contact_sheet": rel(contact_sheet_path),
        "frames_extracted": len(frame_ids),
        "window_start": start,
        "window_end": end,
        "suggested_frame_min": window["suggested_frame_min"],
        "suggested_frame_max": window["suggested_frame_max"],
    }


def write_report(summary: dict, out_root: Path) -> None:
    report_path = out_root / f"{summary['candidate_episode_id']}_event_window.md"
    lines = [
        f"# Pending Radio Event Window: {summary['candidate_episode_id']}",
        "",
        "This is a review package only. It does not add a Benchmark v0 label and does not modify `benchmark_v0/episodes.json`.",
        "",
        "## Summary",
        "",
        f"- Candidate: `{summary['candidate_episode_id']}`",
        f"- Adds benchmark labels: `{summary['adds_benchmark_labels']}`",
        f"- Dense window: frames `{summary['window_start']}` to `{summary['window_end']}`",
        f"- GRM-suggested frames: `{summary['suggested_frame_min']}` to `{summary['suggested_frame_max']}`",
        f"- Frames extracted: `{summary['frames_extracted']}`",
        f"- Contact sheet: `{summary['contact_sheet']}`",
        f"- Manifest: `{summary['manifest']}`",
        "",
        "## Manual Review Checklist",
        "",
        "- Inspect left-wrist frames around the GRM jump for switch contact and green indicator.",
        "- Assign `button_press` and `indicator_green` only if visible evidence is clear.",
        "- Keep this candidate pending until event labels, success label, and cached prediction paths are reviewed.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    window = infer_window(args.pred_root, args.margin)
    summary = extract_window(args, window)
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / f"{args.candidate_id}_event_window.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, args.out_root)
    print(f"candidate={args.candidate_id}")
    print(f"adds_benchmark_labels={str(summary['adds_benchmark_labels']).lower()}")
    print(f"window={summary['window_start']}..{summary['window_end']}")
    print(f"frames={summary['frames_extracted']}")
    print(f"wrote={summary_path}")
    print(f"contact_sheet={summary['contact_sheet']}")


if __name__ == "__main__":
    main()
