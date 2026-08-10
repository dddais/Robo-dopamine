from __future__ import annotations

import re
from pathlib import Path
from typing import Any

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


def extract_frame_at(
    video_path: str | Path,
    output_path: str | Path,
    frame_index: int,
) -> tuple[int, str]:
    """Decode one exact source frame and cache it at a deterministic path."""
    path = Path(output_path)
    if path.is_file():
        return int(frame_index), str(path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame = _read_at(cap, int(frame_index))
    cap.release()
    if frame is None:
        raise RuntimeError(
            f"Cannot decode frame {int(frame_index)} from {video_path}"
        )
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Cannot write frame to {path}")
    return int(frame_index), str(path.resolve())


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


def extract_uniform_image_sequence(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    count: int = 8,
) -> tuple[list[str], dict[str, Any]]:
    """Decode an auditable fixed-length sequence of independent images.

    Unlike a Qwen ``video`` item, every returned path is intended to become a
    separate ``image`` item and therefore a separate image-token span.  The
    indices are uniformly spaced over the decoded rollout and always include
    the source terminal frame.  Videos shorter than ``count`` are rejected so
    this ablation cannot silently change its configured image count.
    """
    if count < 2:
        raise ValueError("image sequence count must be at least two")
    source = Path(video_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cached = sorted(destination.glob("image_*_source_*.png"))
    cached_indices: list[int] = []
    if len(cached) == count:
        for ordinal, path in enumerate(cached):
            match = re.fullmatch(
                r"image_(\d+)_source_(\d+)\.png", path.name
            )
            if match is None or int(match.group(1)) != ordinal:
                cached_indices = []
                break
            cached_indices.append(int(match.group(2)))
    if len(cached_indices) == count:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return [str(path.resolve()) for path in cached], {
            "input_representation": "uniform_independent_images_v1",
            "requested_image_count": int(count),
            "image_count": len(cached),
            "decoded_frame_count": cached_indices[-1] + 1,
            "selected_source_indices": cached_indices,
            "terminal_source_index": cached_indices[-1],
            "terminal_frame_in_last_image": True,
            "source_fps": source_fps,
            "width": width,
            "height": height,
            "source_video_path": str(source),
        }
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded: list[Any] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded.append(frame)
    cap.release()
    if len(decoded) < count:
        raise ValueError(
            f"Image-sequence protocol requires {count} decoded frames, "
            f"but {source} has {len(decoded)}"
        )
    indices = uniform_indices(len(decoded), count)
    if len(indices) != count or indices[-1] != len(decoded) - 1:
        raise RuntimeError("Uniform image sampling did not retain the terminal frame")
    paths: list[str] = []
    for ordinal, source_index in enumerate(indices):
        path = destination / f"image_{ordinal:02d}_source_{source_index:06d}.png"
        if not path.is_file() and not cv2.imwrite(str(path), decoded[source_index]):
            raise RuntimeError(f"Cannot write sampled image: {path}")
        paths.append(str(path))
    return paths, {
        "input_representation": "uniform_independent_images_v1",
        "requested_image_count": int(count),
        "image_count": len(paths),
        "decoded_frame_count": len(decoded),
        "selected_source_indices": [int(value) for value in indices],
        "terminal_source_index": len(decoded) - 1,
        "terminal_frame_in_last_image": True,
        "source_fps": source_fps,
        "width": width,
        "height": height,
        "source_video_path": str(source),
    }
