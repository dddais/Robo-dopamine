from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import cv2
from PIL import Image

from .base import Grounder, mask_to_bbox


def _clip_bbox_to_image(
    bbox: list[float] | tuple[float, ...], width: int, height: int
) -> list[float] | None:
    clipped = [
        max(0.0, min(float(width), float(bbox[0]))),
        max(0.0, min(float(height), float(bbox[1]))),
        max(0.0, min(float(width), float(bbox[2]))),
        max(0.0, min(float(height), float(bbox[3]))),
    ]
    return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None


class SAM3Grounder(Grounder):
    backend = "sam3"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._video_predictor = None

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

    def candidates(self, image_path: str, queries: list[str]) -> list[dict[str, Any]]:
        self._load()
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        rows = []
        for priority, query in enumerate(queries):
            inputs = self._processor(images=image, text=query, return_tensors="pt").to(
                self._device
            )
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            result = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=float(self.config.get("threshold", 0.3)),
                mask_threshold=float(self.config.get("mask_threshold", 0.5)),
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            boxes = result.get("boxes")
            masks = result.get("masks")
            for index, (mask, score) in enumerate(zip(masks, result["scores"])):
                mask_array = mask.detach().cpu().numpy().astype(np.uint8)
                bbox = (
                    [float(x) for x in boxes[index].detach().cpu().tolist()]
                    if boxes is not None
                    else mask_to_bbox(mask_array)
                )
                if bbox is None:
                    continue
                bbox = _clip_bbox_to_image(bbox, width, height)
                if bbox is None:
                    continue
                rows.append(
                    {
                        "bbox": bbox,
                        "score": float(score.detach().cpu()),
                        "label": query,
                        "query": query,
                        "query_priority": priority,
                        "_mask": mask_array,
                    }
                )
        rows.sort(key=lambda row: (-row["score"], row["query_priority"]))
        return rows[: int(self.config.get("top_n", 10))]

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
        matches = (
            np.flatnonzero(ids == obj_id)
            if obj_id is not None and len(ids)
            else np.array([], dtype=int)
        )
        index = int(matches[0]) if len(matches) else int(np.argmax(scores)) if len(scores) else 0
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
        return {
            "frame_index": int(frame_index),
            "bbox": bbox,
            "score": float(scores[index]) if len(scores) > index else None,
            "obj_id": int(ids[index]) if len(ids) > index else obj_id,
            "_mask": mask,
        }

    def track(
        self, video_path: str, bbox: list[float] | tuple[float, ...], anchor_index: int = 0
    ) -> list[dict[str, Any]]:
        """Track one first-frame instance from a box prompt through the video."""
        predictor = self._load_video_predictor()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video for tracking: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        if width < 1 or height < 1:
            raise RuntimeError(f"Invalid video dimensions: {video_path}")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        normalized_xywh = [
            x1 / width,
            y1 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]
        response = predictor.handle_request(
            request={"type": "start_session", "resource_path": video_path}
        )
        session_id = response["session_id"]
        try:
            prompt = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(anchor_index),
                    "text": "visual",
                    "bounding_boxes": [normalized_xywh],
                    "bounding_box_labels": [1],
                }
            )
            initial = self._one_track_output(
                prompt.get("outputs", {}), int(anchor_index), width, height, None
            )
            obj_id = initial.get("obj_id") if initial else None
            by_frame: dict[int, dict[str, Any]] = {}
            if initial is not None:
                by_frame[int(anchor_index)] = initial

            def consume_propagation() -> None:
                for item in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": int(anchor_index),
                    }
                ):
                    row = self._one_track_output(
                        item.get("outputs", {}),
                        int(item["frame_index"]),
                        width,
                        height,
                        obj_id,
                    )
                    if row is not None:
                        by_frame[row["frame_index"]] = row

            consume_propagation()
            terminal_index = frame_count - 1
            # A visual box prompt is detector-driven. Some SAM3 releases do
            # not retain it to the terminal frame. After the required normal
            # propagation has populated the cache, promote the same object id
            # into the instance tracker with one positive mask-centroid point.
            if (
                initial is not None
                and obj_id is not None
                and terminal_index not in by_frame
            ):
                mask = initial.get("_mask")
                point: list[float]
                if mask is not None:
                    mask_array = np.asarray(mask).squeeze()
                    ys, xs = np.nonzero(mask_array)
                    if len(xs):
                        point = [
                            (float(xs.mean()) + 0.5) / float(mask_array.shape[1]),
                            (float(ys.mean()) + 0.5) / float(mask_array.shape[0]),
                        ]
                    else:
                        point = [
                            (x1 + x2) / (2.0 * width),
                            (y1 + y2) / (2.0 * height),
                        ]
                else:
                    point = [
                        (x1 + x2) / (2.0 * width),
                        (y1 + y2) / (2.0 * height),
                    ]
                refined_prompt = predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": int(anchor_index),
                        "points": [point],
                        "point_labels": [1],
                        "obj_id": int(obj_id),
                    }
                )
                refined = self._one_track_output(
                    refined_prompt.get("outputs", {}),
                    int(anchor_index),
                    width,
                    height,
                    int(obj_id),
                )
                if refined is not None:
                    initial = refined
                    by_frame[int(anchor_index)] = refined
                consume_propagation()
            ordered = [by_frame[index] for index in sorted(by_frame)]
            # Full-resolution masks dominate memory on long videos. Only the
            # two endpoint masks are consumed downstream.
            for row in ordered[1:-1]:
                row.pop("_mask", None)
            return ordered
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
