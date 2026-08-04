from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from ..io import artifact_fingerprint, object_fingerprint
from ..schemas import validate_bbox


class Grounder(ABC):
    backend: str

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @property
    def fingerprint(self) -> str:
        model_path = self.config.get("model_path")
        return object_fingerprint(
            {
                "backend": self.backend,
                "config": self.config,
                "model_artifact": artifact_fingerprint(model_path)
                if model_path
                else "unspecified",
            }
        )

    @abstractmethod
    def candidates(self, image_path: str, queries: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def ground(self, image_path: str, queries: list[str]) -> dict[str, Any] | None:
        candidates = self.candidates(image_path, queries)
        return self.select(image_path, candidates, len(queries))

    def select(
        self, image_path: str, candidates: list[dict[str, Any]], query_count: int
    ) -> dict[str, Any] | None:
        if not candidates:
            return None
        with Image.open(image_path) as image:
            width, height = image.size
        legal = []
        for candidate in candidates:
            try:
                candidate["bbox"] = list(validate_bbox(candidate["bbox"], width, height))
                legal.append(candidate)
            except ValueError:
                continue
        if not legal:
            return None
        return max(
            legal,
            key=lambda row: (
                float(row["score"]),
                -int(row.get("query_priority", query_count)),
            ),
        )

    def track(
        self, video_path: str, query: str, anchor_indices: list[int]
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{self.backend} video tracking is not implemented")


def mask_to_bbox(mask) -> tuple[float, float, float, float] | None:
    import numpy as np

    array = np.asarray(mask).astype(bool)
    ys, xs = np.where(array)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def select_temporal_pair(
    first_image_path: str,
    last_image_path: str,
    first_candidates: list[dict[str, Any]],
    last_candidates: list[dict[str, Any]],
    *,
    query_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Select endpoint boxes jointly using confidence, query identity and appearance."""
    import cv2
    import numpy as np

    images = [cv2.imread(first_image_path), cv2.imread(last_image_path)]
    if any(image is None for image in images):
        raise RuntimeError("Cannot decode endpoint image for pair selection")

    def legal(rows, image):
        height, width = image.shape[:2]
        output = []
        for row in rows:
            try:
                row["bbox"] = list(validate_bbox(row["bbox"], width, height))
                output.append(row)
            except ValueError:
                continue
        return output

    first = legal(first_candidates, images[0])
    last = legal(last_candidates, images[1])
    if not first or not last:
        return (
            max(first, key=lambda row: row["score"]) if first else None,
            max(last, key=lambda row: row["score"]) if last else None,
            "one_or_both_endpoints_have_no_legal_candidate",
        )

    def histogram(image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(48, dtype=np.float32)
        hist = cv2.calcHist([crop], [0, 1, 2], None, [16, 16, 16], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).reshape(-1)
        return hist

    first_hist = [histogram(images[0], row["bbox"]) for row in first]
    last_hist = [histogram(images[1], row["bbox"]) for row in last]
    best = None
    for first_index, first_row in enumerate(first):
        for last_index, last_row in enumerate(last):
            appearance = float(
                cv2.compareHist(
                    first_hist[first_index], last_hist[last_index], cv2.HISTCMP_CORREL
                )
            )
            appearance = max(0.0, min(1.0, (appearance + 1) / 2))
            same_query = float(first_row.get("query") == last_row.get("query"))
            priority = 1 - (
                int(first_row.get("query_priority", query_count))
                + int(last_row.get("query_priority", query_count))
            ) / max(1, 2 * query_count)
            confidence = (float(first_row["score"]) + float(last_row["score"])) / 2
            score = 0.50 * confidence + 0.30 * appearance + 0.15 * same_query + 0.05 * priority
            candidate = (score, -first_index, -last_index, first_row, last_row)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    assert best is not None
    best[3]["pair_selection_score"] = best[0]
    best[4]["pair_selection_score"] = best[0]
    return best[3], best[4], "joint_confidence_query_and_crop_appearance"
