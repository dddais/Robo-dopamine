"""GroundingDINO inference and endpoint consistency utilities."""

from __future__ import annotations

import gc
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DetectionCandidate:
    label: str
    score: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox"] = list(self.bbox)
        return payload


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in first]
    bx1, by1, bx2, by2 = [float(x) for x in second]
    left = max(ax1, bx1)
    top = max(ay1, by1)
    right = min(ax2, bx2)
    bottom = min(ay2, by2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(
    first: Sequence[float],
    second: Sequence[float],
    image_size: Sequence[int],
) -> float:
    width, height = [float(x) for x in image_size]
    ax1, ay1, ax2, ay2 = [float(x) for x in first]
    bx1, by1, bx2, by2 = [float(x) for x in second]
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    diagonal = max(math.hypot(width, height), 1.0)
    return math.hypot(acx - bcx, acy - bcy) / diagonal


def area_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in first]
    bx1, by1, bx2, by2 = [float(x) for x in second]
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return area_b / area_a if area_a > 0 else math.inf


def pair_consistency(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    *,
    image_size: Sequence[int],
    minimum_iou: float = 0.30,
    maximum_center_distance: float = 0.12,
) -> dict[str, Any]:
    """Measure whether endpoint detections plausibly represent one instance.

    This is a quality proxy rather than ground-truth accuracy.  It is useful for
    this reward=1 slice because the instruction target is generally stationary,
    but camera motion, occlusion, or a detector identity switch can reduce it.
    """

    if not before or not after:
        return {
            "available": False,
            "consistent": False,
            "iou": None,
            "center_distance": None,
            "area_ratio": None,
        }
    first = before["bbox"]
    second = after["bbox"]
    iou = box_iou(first, second)
    center_distance = normalized_center_distance(first, second, image_size)
    ratio = area_ratio(first, second)
    scale_consistent = 0.5 <= ratio <= 2.0
    consistent = iou >= minimum_iou or (
        center_distance <= maximum_center_distance and scale_consistent
    )
    return {
        "available": True,
        "consistent": bool(consistent),
        "iou": iou,
        "center_distance": center_distance,
        "area_ratio": ratio,
    }


_COLOR_ALIASES = {
    "grey": "gray",
    "silver": "metallic",
    "tan": "beige",
}
_COLOR_WORDS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "beige", "brown", "white", "black", "dark", "gray", "metallic",
}


def explicit_colors(attributes: Sequence[str]) -> list[str]:
    colors: list[str] = []
    for attribute in attributes:
        for token in str(attribute).lower().replace("/", " ").split():
            token = _COLOR_ALIASES.get(token, token)
            if token in _COLOR_WORDS and token not in colors:
                colors.append(token)
    return colors


def color_match_fraction(
    image_path: str | Path,
    bbox: Sequence[float],
    colors: Sequence[str],
) -> float | None:
    """Return the strongest simple HSV color fraction for explicit attributes.

    This is deliberately a reranking cue rather than a detector replacement.
    When no explicit color word exists it returns ``None`` and has no effect.
    """

    normalized = [_COLOR_ALIASES.get(str(color).lower(), str(color).lower()) for color in colors]
    normalized = [color for color in normalized if color in _COLOR_WORDS]
    if not normalized:
        return None
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    clamped = _clamp_box(bbox, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    crop = image[int(y1) : max(int(y1) + 1, int(math.ceil(y2))), int(x1) : max(int(x1) + 1, int(math.ceil(x2)))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    masks = {
        "red": (((h <= 8) | (h >= 172)) & (s >= 70) & (v >= 45)),
        "orange": ((h >= 5) & (h <= 25) & (s >= 65) & (v >= 70)),
        "yellow": ((h >= 18) & (h <= 40) & (s >= 55) & (v >= 80)),
        "green": ((h >= 35) & (h <= 90) & (s >= 45) & (v >= 45)),
        "blue": ((h >= 88) & (h <= 135) & (s >= 55) & (v >= 45)),
        "purple": ((h >= 125) & (h <= 165) & (s >= 45) & (v >= 45)),
        "pink": (((h <= 8) | (h >= 165)) & (s >= 25) & (s <= 180) & (v >= 120)),
        "beige": ((h >= 7) & (h <= 35) & (s >= 18) & (s <= 170) & (v >= 70)),
        "brown": ((h >= 5) & (h <= 30) & (s >= 55) & (v >= 25) & (v <= 180)),
        "white": ((s <= 45) & (v >= 170)),
        "black": (v <= 65),
        "dark": (v <= 90),
        "gray": ((s <= 45) & (v >= 60) & (v <= 210)),
        "metallic": ((s <= 55) & (v >= 65) & (v <= 230)),
    }
    return max(float(np.mean(masks[color])) for color in normalized)


def _area_fraction(box: Sequence[float], image_size: Sequence[int]) -> float:
    width, height = [float(value) for value in image_size]
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(width * height, 1.0)


def select_temporal_candidate_pair(
    before_candidates: Sequence[Mapping[str, Any]],
    after_candidates: Sequence[Mapping[str, Any]],
    *,
    before_image_path: str | Path,
    after_image_path: str | Path,
    image_size: Sequence[int],
    attributes: Sequence[str] = (),
    target_type: str = "object",
) -> dict[str, Any]:
    """Jointly select the most plausible stationary target instance.

    RoboRewardBench reward=1 counterfactual examples usually leave the
    instruction target unchanged.  Joint selection therefore uses detector
    confidence, endpoint overlap/center stability, explicit color attributes,
    and a penalty for near-full-image hallucination boxes.  It is disabled as a
    hard prior for region targets, where a very large box can be legitimate.
    """

    if not before_candidates or not after_candidates:
        return {
            "available": False,
            "before": before_candidates[0] if before_candidates else None,
            "after": after_candidates[0] if after_candidates else None,
            "pair_score": None,
            "pair_margin": None,
            "pairs_considered": 0,
            "color_attributes": explicit_colors(attributes),
        }
    colors = explicit_colors(attributes)
    scored: list[dict[str, Any]] = []
    for before in before_candidates:
        for after in after_candidates:
            iou = box_iou(before["bbox"], after["bbox"])
            center = normalized_center_distance(before["bbox"], after["bbox"], image_size)
            ratio = area_ratio(before["bbox"], after["bbox"])
            score_mean = (float(before["score"]) + float(after["score"])) / 2.0
            area_mean = (
                _area_fraction(before["bbox"], image_size)
                + _area_fraction(after["bbox"], image_size)
            ) / 2.0
            color_before = color_match_fraction(before_image_path, before["bbox"], colors)
            color_after = color_match_fraction(after_image_path, after["bbox"], colors)
            color_values = [value for value in (color_before, color_after) if value is not None]
            color_mean = sum(color_values) / len(color_values) if color_values else None
            stability = 0.25 * iou + 0.10 * max(0.0, 1.0 - center / 0.15)
            scale_penalty = 0.08 * min(abs(math.log(max(ratio, 1e-6))), 2.5)
            large_box_penalty = 0.0
            if target_type != "region" and area_mean > 0.65:
                large_box_penalty = 0.30 * min(1.0, (area_mean - 0.65) / 0.35)
            # An explicit instruction color is a strong identity cue in these
            # multi-object scenes.  It receives more weight than a small raw
            # detector-score difference, while temporal stability remains an
            # independent requirement for steering readiness.
            color_bonus = 0.80 * color_mean if color_mean is not None else 0.0
            pair_score = score_mean + stability + color_bonus - scale_penalty - large_box_penalty
            scored.append(
                {
                    "before": dict(before),
                    "after": dict(after),
                    "pair_score": pair_score,
                    "detector_score_mean": score_mean,
                    "iou": iou,
                    "center_distance": center,
                    "area_ratio": ratio,
                    "area_fraction_mean": area_mean,
                    "color_match_mean": color_mean,
                    "large_box_penalty": large_box_penalty,
                    "scale_penalty": scale_penalty,
                }
            )
    scored.sort(key=lambda row: float(row["pair_score"]), reverse=True)
    best = scored[0]
    # ``None`` means there is no runner-up.  Avoid ``math.inf`` because JSON's
    # standard number grammar does not permit Infinity.
    margin = (
        float(best["pair_score"]) - float(scored[1]["pair_score"])
        if len(scored) > 1
        else None
    )
    return {
        "available": True,
        **best,
        "pair_margin": margin,
        "pairs_considered": len(scored),
        "color_attributes": colors,
    }


def _clamp_box(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _deduplicate_candidates(
    candidates: Sequence[DetectionCandidate],
    *,
    maximum_iou: float = 0.90,
    top_k: int = 10,
) -> list[DetectionCandidate]:
    output: list[DetectionCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(box_iou(candidate.bbox, kept.bbox) >= maximum_iou for kept in output):
            continue
        output.append(candidate)
        if len(output) >= top_k:
            break
    return output


class GroundingDinoGrounder:
    """Batched wrapper around the local Hugging Face GroundingDINO checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        detection_threshold: float = 0.15,
        text_threshold: float = 0.15,
        accept_threshold: float = 0.25,
        top_k: int = 10,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = device
        self.detection_threshold = float(detection_threshold)
        self.text_threshold = float(text_threshold)
        self.accept_threshold = float(accept_threshold)
        self.top_k = int(top_k)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        # Keep float32.  GroundingDINO decoder paths in some Transformers
        # versions produce dtype mismatches when the model is forced to bf16.
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self._torch = torch

    @staticmethod
    def make_query_text(queries: Sequence[str]) -> str:
        cleaned: list[str] = []
        seen: set[str] = set()
        for query in queries:
            value = str(query).strip().lower().strip(" .")
            if value and value not in seen:
                cleaned.append(value)
                seen.add(value)
        return " . ".join(cleaned) + " ." if cleaned else ""

    def detect_batch(self, items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Detect one target-query set in each image item."""

        if not items:
            return []
        images: list[Image.Image] = []
        sizes: list[tuple[int, int]] = []
        texts: list[str] = []
        try:
            for item in items:
                image = Image.open(item["image_path"]).convert("RGB")
                images.append(image)
                sizes.append((image.height, image.width))
                query_text = self.make_query_text(item["queries"])
                if not query_text:
                    raise ValueError(f"empty GroundingDINO query for {item.get('example_id')}")
                texts.append(query_text)

            inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with self._torch.inference_mode():
                outputs = self.model(**inputs)
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.detection_threshold,
                text_threshold=self.text_threshold,
                target_sizes=sizes,
            )

            records: list[dict[str, Any]] = []
            for item, image, query_text, result in zip(items, images, texts, processed):
                labels = result.get("text_labels")
                if labels is None:
                    labels = result.get("labels", [])
                candidates: list[DetectionCandidate] = []
                for raw_box, raw_label, raw_score in zip(
                    result.get("boxes", []), labels, result.get("scores", [])
                ):
                    if isinstance(raw_label, (list, tuple)):
                        raw_label = " ".join(str(value) for value in raw_label)
                    box = _clamp_box(raw_box.tolist(), image.width, image.height)
                    if box is None:
                        continue
                    candidates.append(
                        DetectionCandidate(
                            label=str(raw_label).strip().lower(),
                            score=float(raw_score),
                            bbox=box,
                        )
                    )
                selected_candidates = _deduplicate_candidates(
                    candidates,
                    top_k=self.top_k,
                )
                selected = selected_candidates[0] if selected_candidates else None
                records.append(
                    {
                        "example_id": item["example_id"],
                        "frame_role": item["frame_role"],
                        "image_path": str(item["image_path"]),
                        "image_size": [image.width, image.height],
                        "queries": list(item["queries"]),
                        "query_text": query_text,
                        "candidates": [candidate.to_dict() for candidate in selected_candidates],
                        "selected": selected.to_dict() if selected else None,
                        "detected": selected is not None,
                        "accepted": bool(selected and selected.score >= self.accept_threshold),
                        "accept_threshold": self.accept_threshold,
                        "detection_threshold": self.detection_threshold,
                        "text_threshold": self.text_threshold,
                        "error": None,
                    }
                )
            return records
        finally:
            for image in images:
                image.close()

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        if getattr(self, "processor", None) is not None:
            del self.processor
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def __enter__(self) -> "GroundingDinoGrounder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
