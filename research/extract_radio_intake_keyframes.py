#!/usr/bin/env python3
"""Extract review keyframes for pending radio intake candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "research_outputs/radio_intake_candidates.json"
DEFAULT_OUT_DIR = ROOT / "research_outputs/radio_intake_keyframes"
DEFAULT_REPORT = ROOT / "research_outputs/radio_intake_keyframes.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/radio_intake_keyframes.json"
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=14)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_pending_candidates(path: Path, max_candidates: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        item
        for item in payload["candidates"]
        if item.get("annotation_status") == "pending_human_verification"
    ]
    return candidates[:max_candidates]


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


def sample_frame_ids(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0]
    ids = [round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]
    return sorted(set(int(item) for item in ids))


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
    thumb_size = (170, 170)
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
            rel_path = record["views"].get(cam)
            if rel_path is None:
                continue
            img = Image.open(ROOT / rel_path).convert("RGB")
            img.thumbnail(thumb_size)
            x = margin + col * (thumb_size[0] + margin)
            y = margin + row * (thumb_size[1] + label_h + margin)
            label = f"{cam} f{record['frame_id']} t={record['time_sec']:.1f}s"
            draw.text((x, y), label, fill=(0, 0, 0))
            sheet.paste(img, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def extract_candidate(candidate: dict, out_dir: Path, num_frames: int) -> dict:
    episode_id = candidate["candidate_episode_id"]
    data_dir = ROOT / candidate["data_dir"]
    front_info = video_info(data_dir / "cam_high.mp4")
    frame_ids = sample_frame_ids(front_info["frames"], num_frames)
    candidate_dir = out_dir / episode_id
    manifest = {
        "candidate_episode_id": episode_id,
        "source": candidate["data_dir"],
        "annotation_status": candidate["annotation_status"],
        "benchmark_label_status": "not_a_new_benchmark_label",
        "purpose": "manual review before any possible Benchmark v0 inclusion",
        "video_metadata": front_info,
        "frames": [],
    }
    for frame_id in frame_ids:
        record = {"frame_id": frame_id, "time_sec": frame_id / front_info["fps"], "views": {}}
        for cam in CAMS:
            video_path = data_dir / f"{cam}.mp4"
            out_path = candidate_dir / f"frame_{frame_id:06d}_{cam}.png"
            ok = extract_frame(video_path, frame_id, out_path)
            record["views"][cam] = rel(out_path) if ok else None
        manifest["frames"].append(record)
    manifest_path = candidate_dir / "manifest.json"
    contact_sheet_path = candidate_dir / "contact_sheet.png"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(manifest, contact_sheet_path)
    return {
        "candidate_episode_id": episode_id,
        "source": candidate["data_dir"],
        "frames_sampled": len(frame_ids),
        "video_frames": front_info["frames"],
        "fps": front_info["fps"],
        "manifest": rel(manifest_path),
        "contact_sheet": rel(contact_sheet_path),
        "adds_benchmark_label": False,
    }


def write_report(payload: dict, report: Path) -> None:
    lines = [
        "# Radio Intake Keyframes / Radio Pending-Candidate Keyframes",
        "",
        "This report indexes contact sheets for pending radio-like candidates. These files are review material only and do not add benchmark labels.",
        "",
        "Current live Benchmark v0 still contains only `xzx_radio_sub23`. The candidates below must be human-verified for `button_press` and `indicator_green` before any benchmark inclusion.",
        "",
        "## Summary",
        "",
        f"- Extracted candidates: `{payload['num_extracted']}`",
        f"- Adds benchmark labels: `{payload['adds_benchmark_labels']}`",
        "",
        "## Extracted Candidates",
        "",
        "| Candidate | Video frames | Sampled frames | Contact sheet | Manifest |",
        "|---|---:|---:|---|---|",
    ]
    for item in payload["extracted"]:
        lines.append(
            f"| `{item['candidate_episode_id']}` | {item['video_frames']} | {item['frames_sampled']} | `{item['contact_sheet']}` | `{item['manifest']}` |"
        )
    lines += [
        "",
        "## Manual Review Checklist",
        "",
        "- Locate candidate `grasp`, `lift`, `button_press`, `indicator_green`, `place`, and `release` events.",
        "- Record exact frame ids and evidence views before setting a success label.",
        "- Mark the candidate as non-Markovian only if the task label depends on hidden intermediate events or event order.",
        "- Keep the candidate outside `benchmark_v0/episodes.json` until event labels and cached GRM paths are complete.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/extract_radio_intake_keyframes.py",
        "```",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidates = load_pending_candidates(args.inventory, args.max_candidates)
    extracted = [extract_candidate(item, args.out_dir, args.num_frames) for item in candidates]
    payload = {
        "adds_benchmark_labels": False,
        "source_inventory": rel(args.inventory),
        "num_extracted": len(extracted),
        "extracted": extracted,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, args.report)
    print(f"extracted={len(extracted)}")
    print("adds_benchmark_labels=false")
    print(f"wrote={args.report}")
    print(f"wrote={args.out_json}")


if __name__ == "__main__":
    main()
