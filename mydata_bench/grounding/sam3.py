from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import Grounder, mask_to_bbox


class SAM3Grounder(Grounder):
    backend = "sam3"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._model = None
        self._processor = None

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

    def track(
        self, video_path: str, query: str, anchor_indices: list[int]
    ) -> list[dict[str, Any]]:
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as exc:
            raise RuntimeError(
                "Official sam3 package is required for video tracking in rewardbench-sam3"
            ) from exc
        predictor = build_sam3_video_predictor(
            checkpoint_path=self.config.get("checkpoint_path", str(Path(self.config["model_path"]) / "sam3.pt"))
        )
        response = predictor.handle_request(
            request={"type": "start_session", "resource_path": video_path}
        )
        session_id = response["session_id"]
        anchor = anchor_indices[0] if anchor_indices else 0
        response = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": anchor,
                "text": query,
            }
        )
        return response.get("outputs", [])
