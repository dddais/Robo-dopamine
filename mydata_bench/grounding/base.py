from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from ..io import artifact_fingerprint, object_fingerprint, sha256_file
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
                "implementation": "semantic_instance_grounding_v2",
                "source_sha256": {
                    name: sha256_file(Path(__file__).parent / name)
                    for name in ("base.py", "parser.py", "pipeline.py", "sam3.py", "dino.py", "visual.py", "recovery.py")
                },
                "config": self.config,
                "model_artifact": artifact_fingerprint(model_path)
                if model_path
                else "unspecified",
                "visual_model_artifact": artifact_fingerprint(self.config["visual_resolver"]["model_path"])
                if self.config.get("visual_resolver", {}).get("enabled") else None,
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
                -int(row.get("query_priority", query_count)),
                float(row["score"]),
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


def _box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the same instance returned by multiple open-vocabulary queries."""
    def duplicate(row, other):
        group = row.get("visual_instance_group")
        if group and group == other.get("visual_instance_group") and (
            row.get("visual_instance_index") != other.get("visual_instance_index")
        ):
            # Overlapping boxes can still represent explicitly distinct instances.
            return False
        return _box_iou(row["bbox"], other["bbox"]) >= 0.8

    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: (
        int(value.get("query_priority", 0)), -float(value.get("score", 0))
    )):
        if not any(duplicate(row, other) for other in kept):
            kept.append(row)
    return kept


def select_unambiguous_candidate(
    candidates: list[dict[str, Any]], *, score_margin: float = 0.05
) -> tuple[dict[str, Any] | None, str]:
    rows = _deduplicate_candidates(candidates)
    if not rows:
        return None, "target_not_detected"
    # Semantic specificity takes precedence over scores from different prompts.
    priority = min(int(row.get("query_priority", 0)) for row in rows)
    rows = sorted((r for r in rows if int(r.get("query_priority", 0)) == priority),
                  key=lambda r: float(r["score"]), reverse=True)
    if len(rows) > 1 and float(rows[0]["score"]) - float(rows[1]["score"]) <= score_margin:
        return None, "ambiguous_target_candidates"
    return rows[0], "first_frame_semantic_detection"


def select_relational_candidate(
    image_path: str,
    target_candidates: list[dict[str, Any]],
    reference_candidates: list[dict[str, Any]],
    relation: str,
    *,
    ordinal_index: int | None = None,
    ambiguity_margin: float = 0.05,
    geometry_margin: float = 0.01,
    reference_consensus: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Resolve a relational noun phrase using first-frame box geometry.

    SAM3 supplies entity proposals; this function, rather than the tokenizer or
    language model, implements the exact left/right/nearest/farthest operator.
    """
    import math

    with Image.open(image_path) as image:
        width, height = image.size

    def legal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            value = dict(row)
            try:
                value["bbox"] = list(validate_bbox(value["bbox"], width, height))
            except ValueError:
                continue
            output.append(value)
        return _deduplicate_candidates(output)

    targets = legal(target_candidates)
    references = legal(reference_candidates)

    def center(row: dict[str, Any]) -> tuple[float, float]:
        box = row["bbox"]
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    if relation.startswith("nth_from_"):
        direction = relation.removeprefix("nth_from_")
        if direction not in {"left", "right", "top", "bottom"}:
            raise ValueError(f"Unsupported spatial relation: {relation}")
        if type(ordinal_index) is not int or ordinal_index < 1:
            raise ValueError("ordinal_index must be a positive one-based integer")
        if len(targets) < ordinal_index:
            return None, None, "insufficient_instances_for_ordinal"
        axis = 0 if direction in {"left", "right"} else 1
        ordered = sorted(targets, key=lambda row: center(row)[axis],
                         reverse=direction in {"right", "bottom"})
        selected = ordered[ordinal_index - 1]
        tolerance = geometry_margin * (width if axis == 0 else height)
        if any(abs(center(selected)[axis] - center(other)[axis]) <= tolerance
               for other in ordered if other is not selected):
            return None, None, "ambiguous_ordinal_geometry"
        selected = {**selected, "relation": relation, "ordinal_index": ordinal_index,
                    "instance_count": len(ordered)}
        return selected, None, f"first_frame_geometry_{relation}"
    if not targets or not references:
        return None, None, "relational_target_or_reference_not_detected"
    reference, _ = select_unambiguous_candidate(references, score_margin=ambiguity_margin)
    if reference is None:
        if reference_consensus:
            # An ambiguous reference need not make the target ambiguous. Require
            # every plausible reference to independently select the same target.
            priority = min(int(row.get("query_priority", 0)) for row in references)
            plausible = [row for row in references if int(row.get("query_priority", 0)) == priority]
            best_score = max(float(row["score"]) for row in plausible)
            plausible = [row for row in plausible if best_score - float(row["score"]) <= ambiguity_margin]
            choices = [select_relational_candidate(
                image_path, targets, [row], relation,
                ambiguity_margin=ambiguity_margin, geometry_margin=geometry_margin,
            )[0] for row in plausible]
            if choices and all(row is not None for row in choices) and all(
                _box_iou(choices[0]["bbox"], row["bbox"]) >= 0.8 for row in choices[1:]
            ):
                selected = dict(choices[0])
                for key in ("reference_bbox", "reference_query", "reference_center_distance"):
                    selected.pop(key, None)
                selected["reference_consensus_boxes"] = [list(row["bbox"]) for row in plausible]
                return selected, None, f"first_frame_geometry_{relation}_reference_consensus"
        return None, None, "ambiguous_reference_candidates"
    rx = (reference["bbox"][0] + reference["bbox"][2]) / 2
    ry = (reference["bbox"][1] + reference["bbox"][3]) / 2
    # A reference such as "purple cup" can also appear in the generic "cup"
    # result. Exclude that same physical instance before applying the relation.
    targets = [row for row in targets if _box_iou(row["bbox"], reference["bbox"]) < 0.6]
    if not targets:
        return None, reference, "only_reference_instance_detected"

    if relation in {"left_of", "right_of"}:
        sign = -1 if relation == "left_of" else 1
        valid = [row for row in targets if sign * (center(row)[0] - rx) > geometry_margin * width]
        if not valid:
            return None, reference, f"no_candidate_satisfies_{relation}"
        selected, reason = select_unambiguous_candidate(valid, score_margin=ambiguity_margin)
        if selected is None:
            return None, reference, "ambiguous_relational_candidates"
    elif relation in {"closest_to", "farthest_from"}:
        def distance(row: dict[str, Any]) -> float:
            x, y = center(row)
            return math.hypot(x - rx, y - ry)

        selected = (
            min(targets, key=lambda row: (distance(row), -float(row.get("score", 0))))
            if relation == "closest_to"
            else max(targets, key=lambda row: (distance(row), float(row.get("score", 0))))
        )
        selected["reference_center_distance"] = distance(selected)
        if any(abs(distance(selected) - distance(other)) <= geometry_margin * max(width, height)
               for other in targets if other is not selected):
            return None, reference, "ambiguous_relational_geometry"
    else:
        raise ValueError(f"Unsupported spatial relation: {relation}")
    selected["relation"] = relation
    selected["reference_bbox"] = list(reference["bbox"])
    selected["reference_query"] = reference.get("query")
    return selected, reference, f"first_frame_geometry_{relation}"


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
