"""Canonical model-native media layouts shared by baseline and attention code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..protocol import IMAGE_LABELS


def grm_multiview_image_paths(
    frames: dict[str, dict[str, Any]], blank_goal: str | Path
) -> list[str]:
    blank = Path(blank_goal).resolve()
    if not blank.is_file():
        raise FileNotFoundError(blank)
    missing_views = {"front", "left_wrist", "right_wrist"} - frames.keys()
    if missing_views:
        raise ValueError(f"Missing GRM frame views: {sorted(missing_views)}")
    front = frames["front"]
    left = frames["left_wrist"]
    right = frames["right_wrist"]
    paths = [
        front["first_path"],
        str(blank),
        front["first_path"],
        left["first_path"],
        right["first_path"],
        front["last_path"],
        left["last_path"],
        right["last_path"],
    ]
    if len(paths) != len(IMAGE_LABELS):
        raise AssertionError("GRM layout no longer matches IMAGE_LABELS")
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing GRM input images: {missing}")
    return [str(Path(path).resolve()) for path in paths]

