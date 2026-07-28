from __future__ import annotations

from pathlib import Path

import cv2

from .io import sha256_file
from .schemas import FrameRecord


def uniform_indices(frame_count: int, count: int) -> list[int]:
    if frame_count < 1 or count < 1:
        return []
    if frame_count <= count:
        return list(range(frame_count))
    return [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]


def _read_at(cap: cv2.VideoCapture, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def extract_endpoints(
    example_id: str,
    video_sha256: str,
    video_path: str | Path,
    output_dir: str | Path,
    *,
    max_fallback: int = 300,
) -> FrameRecord:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = _read_at(cap, 0)
    if first is None:
        cap.release()
        raise RuntimeError(f"Cannot decode first frame: {video_path}")
    start = max(0, reported - 1)
    last = None
    last_index = -1
    fallback = 0
    for candidate in range(start, max(-1, start - max_fallback - 1), -1):
        last = _read_at(cap, candidate)
        if last is not None:
            last_index = candidate
            fallback = start - candidate
            break
    if last is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        index = -1
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            index += 1
            last = frame
        last_index = index
        fallback = max(0, start - index)
    cap.release()
    if last is None or last_index < 0:
        raise RuntimeError(f"Cannot decode terminal frame: {video_path}")
    first_path = output_dir / "first.png"
    last_path = output_dir / "last.png"
    if not cv2.imwrite(str(first_path), first) or not cv2.imwrite(str(last_path), last):
        raise RuntimeError(f"Cannot write endpoint frames under {output_dir}")
    height, width = first.shape[:2]
    if last.shape[:2] != first.shape[:2]:
        # Keep actual terminal frame; width/height describe the target endpoint.
        height, width = last.shape[:2]
    return FrameRecord(
        example_id=example_id,
        video_sha256=video_sha256,
        first_index=0,
        last_index=last_index,
        first_path=str(first_path.resolve()),
        last_path=str(last_path.resolve()),
        width=width,
        height=height,
        first_sha256=sha256_file(first_path),
        last_sha256=sha256_file(last_path),
        reported_frame_count=reported,
        last_decode_fallback=fallback,
    )


def extract_uniform(
    video_path: str | Path, output_dir: str | Path, count: int = 8
) -> list[tuple[int, str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results: list[tuple[int, str]] = []
    for index in uniform_indices(frame_count, count):
        frame = _read_at(cap, index)
        if frame is None:
            continue
        path = output_dir / f"frame_{index:06d}.png"
        cv2.imwrite(str(path), frame)
        results.append((index, str(path.resolve())))
    cap.release()
    return results

