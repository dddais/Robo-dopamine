#!/usr/bin/env python3
"""Extract dense radio task frames around the switch/indicator event."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "aligned_data/xzx_episode_1_sub23"
OUT_DIR = ROOT / "benchmark_v0/keyframes/xzx_radio_sub23_event_window"
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# Dense window around the observed red-to-green indicator transition.
FRAME_IDS = list(range(510, 691, 15))


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
    thumb_size = (220, 220)
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "episode_id": "xzx_radio_sub23",
        "source": str(DATA_DIR.relative_to(ROOT)),
        "purpose": "dense manual inspection of button_press and indicator_green events",
        "frame_interval": 15,
        "frames": [],
    }
    for frame_id in FRAME_IDS:
        record = {"frame_id": frame_id, "time_sec": frame_id / 30.0, "views": {}}
        for cam in CAMS:
            video_path = DATA_DIR / f"{cam}.mp4"
            out_path = OUT_DIR / f"frame_{frame_id:06d}_{cam}.png"
            ok = extract_frame(video_path, frame_id, out_path)
            record["views"][cam] = str(out_path.relative_to(ROOT)) if ok else None
        manifest["frames"].append(record)

    manifest_path = OUT_DIR / "manifest.json"
    contact_sheet_path = OUT_DIR / "contact_sheet.png"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(manifest, contact_sheet_path)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {contact_sheet_path}")
    print(f"Extracted {len(FRAME_IDS)} frame sets to {OUT_DIR}")


if __name__ == "__main__":
    main()
