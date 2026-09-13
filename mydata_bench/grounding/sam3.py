from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import cv2
from PIL import Image

from ..io import sha256_file
from .base import Grounder, _box_iou, _deduplicate_candidates, mask_to_bbox


def _clip_bbox_to_image(
    bbox: list[float] | tuple[float, ...], width: int, height: int
) -> list[float] | None:
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        return None
    clipped = [
        max(0.0, min(float(width), float(bbox[0]))),
        max(0.0, min(float(height), float(bbox[1]))),
        max(0.0, min(float(width), float(bbox[2]))),
        max(0.0, min(float(height), float(bbox[3]))),
    ]
    return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None


def _interior_point(mask, bbox, width: int, height: int) -> list[float]:
    """A mask centroid can fall on background for a ring or curved thin object."""
    if mask is not None:
        array = np.asarray(mask).squeeze().astype(np.uint8)
        if array.ndim == 2 and array.any():
            distance = cv2.distanceTransform(np.pad(array, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            y, x = np.unravel_index(np.argmax(distance), distance.shape)
            return [(float(x) + 0.5) / array.shape[1], (float(y) + 0.5) / array.shape[0]]
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height)]


def _interior_points(mask, bbox, width: int, height: int, count: int = 1) -> list[list[float]]:
    """Spread positive prompts inside the detected mask, keeping a safe first point."""
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("tracking_positive_points must be an integer in 1..8")
    first = _interior_point(mask, bbox, width, height)
    if count == 1 or mask is None:
        return [first]
    array = np.asarray(mask).squeeze().astype(np.uint8)
    if array.ndim != 2 or not array.any():
        return [first]
    distance = cv2.distanceTransform(np.pad(array, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    ys, xs = np.where(distance >= max(1.0, float(distance.max()) * 0.25))
    if not len(xs):
        return [first]
    choices = np.stack((xs, ys), axis=1)
    selected = [choices[np.argmax(distance[ys, xs])]]
    for _ in range(count - 1):
        separation = np.min(np.sum(
            (choices[:, None, :] - np.array(selected)[None, :, :]) ** 2, axis=2,
        ), axis=1)
        index = np.argmax(separation)
        if separation[index] < 16:
            break
        selected.append(choices[index])
    return [[(float(x) + 0.5) / array.shape[1], (float(y) + 0.5) / array.shape[0]]
            for x, y in selected]


class SAM3Grounder(Grounder):
    backend = "sam3"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._video_predictor = None
        self._candidate_cache = OrderedDict()
        self.last_tracking_diagnostics: dict[str, Any] = {}
        self._visual_resolver = None

    def visual_candidates(self, image_path, queries, candidates):
        if self._visual_resolver is None:
            from .visual import VisualCategoryResolver
            self._visual_resolver = VisualCategoryResolver(self.config['visual_resolver'])
        return self._visual_resolver.resolve(image_path, queries, candidates)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "SAM3 is unavailable in this environment. Create the isolated "
                "'rewardbench-sam3' environment described in "
                "rewardbench/MIGRATION_DEPLOYMENT_EVALUATION.md."
            ) from exc
        self._torch = torch
        model_path = self.config["model_path"]
        self._processor = Sam3Processor.from_pretrained(model_path)
        self._model = Sam3Model.from_pretrained(model_path)
        self._device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()

    def _query_variants(self, query: str) -> list[str]:
        aliases = self.config.get("query_aliases", {})
        variants = [query]
        # Longest suffix first preserves modifiers ("red cup" -> "red small bowl").
        for name in sorted(aliases, key=len, reverse=True):
            if query == name or query.endswith(" " + name):
                values = aliases[name]
                if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                    raise ValueError("query_aliases values must be lists of strings")
                prefix = query[:-len(name)]
                variants.extend(prefix + value for value in values)
                break
        return list(dict.fromkeys(variants))

    def _image_candidates(self, image: Image.Image, query: str) -> list[dict[str, Any]]:
        width, height = image.size
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        result = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=float(self.config.get("threshold", 0.3)),
            mask_threshold=float(self.config.get("mask_threshold", 0.5)),
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        rows = []
        boxes = result.get("boxes")
        for index, (mask, score) in enumerate(zip(result["masks"], result["scores"])):
            mask_array = mask.detach().cpu().numpy().astype(np.uint8)
            bbox = boxes[index].detach().cpu().tolist() if boxes is not None else mask_to_bbox(mask_array)
            if bbox is None or not np.isfinite(bbox).all():
                continue
            bbox = _clip_bbox_to_image(bbox, width, height)
            if bbox is not None and mask_array.any():
                rows.append({"bbox": bbox, "score": float(score.detach().cpu()), "_mask": mask_array})
        return rows

    def candidates(self, image_path: str, queries: list[str]) -> list[dict[str, Any]]:
        if not queries:
            return []
        key = (sha256_file(image_path), tuple(queries))
        if key in self._candidate_cache:
            self._candidate_cache.move_to_end(key)
            return [dict(row) for row in self._candidate_cache[key]]
        self._load()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        crops = [(0, 0, width, height)]
        grid = int(self.config.get("detection_tile_grid", 1))
        if grid > 1:
            overlap = float(self.config.get("detection_tile_overlap", 0.25))
            if not 0 <= overlap < 1 or grid > 4:
                raise ValueError("Require tile grid <= 4 and overlap in [0, 1)")
            cw = min(width, round(width / (grid - (grid - 1) * overlap)))
            ch = min(height, round(height / (grid - (grid - 1) * overlap)))
            crops += [(int(x), int(y), int(x) + cw, int(y) + ch)
                      for y in np.linspace(0, height - ch, grid)
                      for x in np.linspace(0, width - cw, grid)]
        rows = []
        for priority, semantic_query in enumerate(queries):
            for query in self._query_variants(semantic_query):
                for left, top, right, bottom in crops:
                    crop = image.crop((left, top, right, bottom))
                    for row in self._image_candidates(crop, query):
                        x1, y1, x2, y2 = row["bbox"]
                        # Interior crop boundaries can split an instance in two;
                        # the overlapping view or full image must supply it whole.
                        if ((left > 0 and x1 <= 2) or (top > 0 and y1 <= 2)
                            or (right < width and x2 >= crop.width - 2)
                            or (bottom < height and y2 >= crop.height - 2)):
                            continue
                        mask = np.zeros((height, width), dtype=np.uint8)
                        mask[top:bottom, left:right] = row["_mask"]
                        rows.append({**row, "bbox": [x1 + left, y1 + top, x2 + left, y2 + top],
                                     "_mask": mask, "query": query, "label": semantic_query,
                                     "semantic_query": semantic_query, "query_priority": priority,
                                     "detection_crop": [left, top, right, bottom]})
        rows = _deduplicate_candidates(rows)
        rows.sort(key=lambda row: (row["query_priority"], -row["score"]))
        rows = rows[:int(self.config.get("top_n", 20))]
        self._candidate_cache[key] = rows
        while len(self._candidate_cache) > int(self.config.get("candidate_cache_size", 16)):
            self._candidate_cache.popitem(last=False)
        return [dict(row) for row in rows]

    def _load_video_predictor(self):
        if self._video_predictor is not None:
            return self._video_predictor
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "Official sam3 package is required for video tracking in rewardbench-sam3"
            ) from exc
        self._video_predictor = build_sam3_video_predictor(
            checkpoint_path=self.config.get(
                "checkpoint_path", str(Path(self.config["model_path"]) / "sam3.pt")
            ),
        )
        return self._video_predictor

    @staticmethod
    def _one_track_output(
        outputs: dict[str, Any], frame_index: int, width: int, height: int, obj_id: int | None
    ) -> dict[str, Any] | None:
        ids = np.asarray(outputs.get("out_obj_ids", [])).reshape(-1)
        scores = np.asarray(outputs.get("out_probs", [])).reshape(-1)
        boxes = np.asarray(outputs.get("out_boxes_xywh", [])).reshape(-1, 4)
        if not len(boxes):
            return None
        if len(ids) != len(boxes) or len(scores) != len(boxes):
            return None
        matches = (
            np.flatnonzero(ids == obj_id)
            if obj_id is not None and len(ids)
            else np.array([], dtype=int)
        )
        if obj_id is not None and not len(matches):
            return None
        index = int(matches[0]) if len(matches) else int(np.argmax(scores))
        if not np.isfinite(boxes[index]).all() or not np.isfinite(scores[index]):
            return None
        x, y, w, h = (float(value) for value in boxes[index])
        bbox = [
            max(0.0, min(float(width), x * width)),
            max(0.0, min(float(height), y * height)),
            max(0.0, min(float(width), (x + w) * width)),
            max(0.0, min(float(height), (y + h) * height)),
        ]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            return None
        masks = outputs.get("out_binary_masks")
        mask = None
        if masks is not None and len(masks) > index:
            mask = np.asarray(masks[index]).squeeze().astype(np.uint8)
            if mask.ndim != 2 or not mask.any():
                return None
        return {
            "frame_index": int(frame_index),
            "bbox": bbox,
            "score": float(scores[index]) if len(scores) > index else None,
            "obj_id": int(ids[index]) if len(ids) > index else obj_id,
            "_mask": mask,
        }

    def track(
        self, video_path: str, bbox: list[float] | tuple[float, ...], anchor_index: int = 0,
        *, anchor_mask=None, terminal_index: int | None = None,
        propagation_direction: str = "forward",
    ) -> list[dict[str, Any]]:
        """Bind the detected instance, then propagate only its ID, with offline refinement."""
        self.last_tracking_diagnostics = {"identity_policy": "strict_obj_id", "refined": False}
        predictor = self._load_video_predictor()
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video for tracking: {video_path}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            capture.release()
        if width < 1 or height < 1:
            raise RuntimeError(f"Invalid video dimensions: {video_path}")
        if propagation_direction not in {"forward", "backward"}:
            raise ValueError("propagation_direction must be forward or backward")
        terminal_index = (frame_count - 1 if propagation_direction == "forward" else 0) if terminal_index is None else int(terminal_index)
        lower, upper = sorted((anchor_index, terminal_index))
        if not 0 <= lower <= upper < frame_count or (
            (propagation_direction == "forward" and anchor_index > terminal_index)
            or (propagation_direction == "backward" and anchor_index < terminal_index)
        ):
            raise ValueError("Tracking endpoints must lie inside the video in propagation order")
        expected_indices = range(lower, upper + 1)
        prompt_mode = self.config.get("tracking_prompt", "visual_box")
        if prompt_mode not in {"visual_box", "mask_refine"}:
            raise ValueError("tracking_prompt must be visual_box or mask_refine")
        self.last_tracking_diagnostics = {
            "identity_policy": "strict_obj_id", "refined": False, "prompt_mode": prompt_mode,
            "propagation_direction": propagation_direction,
        }
        x1, y1, x2, y2 = bbox
        normalized_xywh = [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]
        response = predictor.handle_request(
            request={"type": "start_session", "resource_path": video_path}
        )
        session_id = response["session_id"]
        min_iou = float(self.config.get("tracking_anchor_min_iou", 0.3))
        try:
            prompt = predictor.handle_request(request={
                "type": "add_prompt", "session_id": session_id, "frame_index": int(anchor_index),
                "text": "visual", "bounding_boxes": [normalized_xywh], "bounding_box_labels": [1],
            })
            outputs = prompt.get("outputs", {})
            observed_ids = set(int(i) for i in outputs.get("out_obj_ids", []))
            anchors = [self._one_track_output(outputs, anchor_index, width, height, i)
                       for i in sorted(observed_ids)]
            anchors = [r for r in anchors if r is not None and _box_iou(r["bbox"], bbox) >= min_iou]
            initial = max(anchors, key=lambda r: (_box_iou(r["bbox"], bbox), r["score"])) if anchors else None
            obj_id = initial["obj_id"] if initial else None

            def consume_propagation() -> dict[int, dict[str, Any]]:
                propagated = {int(anchor_index): initial} if initial is not None else {}
                for item in predictor.handle_stream_request({
                    "type": "propagate_in_video", "session_id": session_id,
                    "propagation_direction": propagation_direction, "start_frame_index": int(anchor_index),
                }):
                    outputs = item.get("outputs", {})
                    observed_ids.update(int(i) for i in outputs.get("out_obj_ids", []))
                    # With no matched anchor this pass only builds the official
                    # predictor's cache. It may not bind an arbitrary later ID.
                    if obj_id is None:
                        continue
                    row = self._one_track_output(outputs, int(item["frame_index"]), width, height, obj_id)
                    if row is not None:
                        if row["frame_index"] not in {anchor_index, terminal_index}:
                            row.pop("_mask", None)
                        propagated[row["frame_index"]] = row
                    else:
                        propagated.pop(int(item["frame_index"]), None)
                return propagated

            # The installed official predictor requires a complete normal pass
            # before point prompts. Adding points to a fresh session asserts.
            by_frame = consume_propagation()
            has_gaps = any(i not in by_frame for i in expected_indices)
            if (prompt_mode == "mask_refine" or has_gaps) and (
                initial is not None or anchor_mask is not None or self.config.get("allow_box_center_seed", False)
            ):
                if obj_id is None:
                    obj_id = max(observed_ids, default=0) + 1
                seed_mask = anchor_mask if anchor_mask is not None else (initial or {}).get("_mask")
                positive_points = _interior_points(
                    seed_mask, bbox, width, height, self.config.get("tracking_positive_points", 1),
                )
                refined_prompt = predictor.handle_request(request={
                    "type": "add_prompt", "session_id": session_id, "frame_index": int(anchor_index),
                    "points": positive_points,
                    "point_labels": [1] * len(positive_points), "obj_id": int(obj_id),
                })
                self.last_tracking_diagnostics["prompt_point_count"] = len(positive_points)
                refined = self._one_track_output(
                    refined_prompt.get("outputs", {}), anchor_index, width, height, obj_id
                )
                if refined is not None and _box_iou(refined["bbox"], bbox) >= min_iou:
                    initial = refined
                    # Do not retain stale boxes at gaps in the corrected pass.
                    by_frame = consume_propagation()
                    self.last_tracking_diagnostics["refined"] = True
                else:
                    self.last_tracking_diagnostics["refinement_failure"] = "tracking_anchor_not_matched"
            missing = [i for i in expected_indices if i not in by_frame]
            longest_gap = current_gap = 0
            for i in expected_indices:
                current_gap = current_gap + 1 if i not in by_frame else 0
                longest_gap = max(longest_gap, current_gap)
            self.last_tracking_diagnostics.update({
                "obj_id": initial["obj_id"] if initial else None,
                "tracked_frame_count": len(by_frame), "missing_frame_count": len(missing),
                "frame_coverage": 1 - len(missing) / len(expected_indices),
                "longest_missing_run": longest_gap, "terminal_present": terminal_index in by_frame,
            })
            if initial is None:
                self.last_tracking_diagnostics["failure"] = "tracking_anchor_not_matched"
            return [by_frame[index] for index in sorted(by_frame)]
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
