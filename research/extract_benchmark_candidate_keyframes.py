#!/usr/bin/env python3
"""Extract keyframes/contact sheets for selected Benchmark v0 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "research_outputs/benchmark_v0_candidate_cases.json"
DEFAULT_OUT_DIR = ROOT / "benchmark_v0/keyframes/candidate_cases"
DEFAULT_REPORT = ROOT / "research_outputs/benchmark_v0_candidate_keyframes.md"
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-episodes", type=int, default=6)
    parser.add_argument("--num-frames", type=int, default=12)
    return parser.parse_args()


def load_candidates(path: Path, max_episodes: int) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    out = []
    for record in records:
        data_tag = record["data_tag"]
        if data_tag in seen:
            continue
        seen.add(data_tag)
        out.append(record)
        if len(out) >= max_episodes:
            break
    return out


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
    return sorted(set(int(i) for i in ids))


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
    thumb_size = (180, 180)
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
            if not rel_path:
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


def extract_episode(case: dict, out_root: Path, num_frames: int) -> dict:
    data_tag = case["data_tag"]
    data_dir = ROOT / "aligned_data" / data_tag
    high_info = video_info(data_dir / "cam_high.mp4")
    frame_ids = sample_frame_ids(high_info["frames"], num_frames)
    episode_dir = out_root / data_tag
    manifest = {
        "episode_id": data_tag,
        "source": str(data_dir.relative_to(ROOT)),
        "candidate_priority": case["priority"],
        "task_tag": case["task_tag"],
        "scene_object": case["scene_object"],
        "selection_reason": case["why"],
        "video_metadata": {
            "frames": high_info["frames"],
            "fps": high_info["fps"],
            "front_resolution": [high_info["width"], high_info["height"]],
        },
        "frames": [],
    }
    for frame_id in frame_ids:
        record = {"frame_id": frame_id, "time_sec": frame_id / high_info["fps"], "views": {}}
        for cam in CAMS:
            video_path = data_dir / f"{cam}.mp4"
            out_path = episode_dir / f"frame_{frame_id:06d}_{cam}.png"
            ok = extract_frame(video_path, frame_id, out_path)
            record["views"][cam] = str(out_path.relative_to(ROOT)) if ok else None
        manifest["frames"].append(record)
    manifest_path = episode_dir / "manifest.json"
    contact_sheet_path = episode_dir / "contact_sheet.png"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(manifest, contact_sheet_path)
    return {
        "data_tag": data_tag,
        "priority": case["priority"],
        "task_tag": case["task_tag"],
        "scene_object": case["scene_object"],
        "manifest": str(manifest_path.relative_to(ROOT)),
        "contact_sheet": str(contact_sheet_path.relative_to(ROOT)),
        "frames": len(frame_ids),
        "video_frames": high_info["frames"],
        "fps": high_info["fps"],
    }


def write_report(extracted: list[dict], report_path: Path) -> None:
    lines = [
        "# Candidate Keyframes / 候选案例关键帧",
        "",
        "## English Summary",
        "",
        "This report indexes contact sheets for manual inspection and baseline diagnosis. It uses existing local videos only and does not rerun GRM. These candidates are not current non-Markovian benchmark labels.",
        "",
        "## 中文总结",
        "",
        "本报告索引人工检查和 baseline 诊断所需的 contact sheet。它只使用已有本地视频，不重新运行 GRM。这些候选不是当前非马尔可夫 benchmark 标签。",
        "",
        "## Extracted Episodes / 已抽帧案例",
        "",
        "| Data | Priority | Task | Frames | Contact sheet | Manifest |",
        "|---|---|---|---:|---|---|",
    ]
    for item in extracted:
        lines.append(
            f"| `{item['data_tag']}` | {item['priority']} | `{item['task_tag']}` | {item['frames']} | `{item['contact_sheet']}` | `{item['manifest']}` |"
        )
    lines += [
        "",
        "## Manual Inspection Checklist / 人工检查清单",
        "",
        "English:",
        "",
        "- Decide whether the failure is visible from the final state or depends on history.",
        "- Mark candidate events such as `grasp`, `lift`, `drop`, `slip`, `place`, `release`, `wrong_object`, and `wrong_target`.",
        "- Keep the case as non-Markovian only if event history changes the success/failure label under similar final visual evidence.",
        "- Otherwise, keep the case as a visible-state baseline diagnostic.",
        "",
        "中文：",
        "",
        "- 判断失败是否仅靠终态可见，还是依赖历史事件。",
        "- 标注候选事件，例如 `grasp`, `lift`, `drop`, `slip`, `place`, `release`, `wrong_object`, `wrong_target`。",
        "- 只有当历史事件会在相似终态视觉证据下改变成败标签时，才把该案例作为非马尔可夫样本。",
        "- 否则保留为可见状态 baseline 诊断样本。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases = load_candidates(args.candidates, args.max_episodes)
    extracted = [extract_episode(case, args.out_dir, args.num_frames) for case in cases]
    write_report(extracted, args.report)
    print(f"extracted={len(extracted)}")
    print(f"wrote={args.report}")
    for item in extracted:
        print(item["data_tag"], item["contact_sheet"])


if __name__ == "__main__":
    main()
