#!/usr/bin/env python3
"""Extract keyframes for radio sub23 event annotation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "aligned_data/xzx_episode_1_sub23"
OUT_DIR = ROOT / "benchmark_v0/keyframes/xzx_radio_sub23"

FRAME_IDS = [
    0,
    30,
    120,
    210,
    240,
    330,
    420,
    450,
    480,
    570,
    630,
    660,
    720,
    840,
    849,
]


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "episode_id": "xzx_radio_sub23",
        "source": str(DATA_DIR.relative_to(ROOT)),
        "purpose": "manual correction of button_press and indicator_green events",
        "frames": [],
    }
    for frame_id in FRAME_IDS:
        record = {"frame_id": frame_id, "time_sec": frame_id / 30.0, "views": {}}
        for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            video_path = DATA_DIR / f"{cam}.mp4"
            out_path = OUT_DIR / f"frame_{frame_id:06d}_{cam}.png"
            ok = extract_frame(video_path, frame_id, out_path)
            record["views"][cam] = str(out_path.relative_to(ROOT)) if ok else None
        manifest["frames"].append(record)
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Extracted {len(FRAME_IDS)} frame sets to {OUT_DIR}")


if __name__ == "__main__":
    main()

