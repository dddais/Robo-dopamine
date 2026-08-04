from __future__ import annotations

from typing import Any

from PIL import Image

from .base import Grounder


class GroundingDINOGrounder(Grounder):
    backend = "grounding_dino"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("GroundingDINO requires torch and transformers") from exc
        self._torch = torch
        model_path = self.config["model_path"]
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
        device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device).eval()
        self._device = device

    def candidates(self, image_path: str, queries: list[str]) -> list[dict[str, Any]]:
        self._load()
        image = Image.open(image_path).convert("RGB")
        rows = []
        top_n = int(self.config.get("top_n", 10))
        for priority, query in enumerate(queries):
            inputs = self._processor(images=image, text=query, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            result = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=float(self.config.get("box_threshold", 0.12)),
                text_threshold=float(self.config.get("text_threshold", 0.10)),
                target_sizes=[image.size[::-1]],
            )[0]
            labels = result.get("text_labels", result.get("labels", []))
            for box, score, label in zip(result["boxes"], result["scores"], labels):
                rows.append(
                    {
                        "bbox": [float(value) for value in box.detach().cpu().tolist()],
                        "score": float(score.detach().cpu()),
                        "label": str(label),
                        "query": query,
                        "query_priority": priority,
                    }
                )
        rows.sort(key=lambda row: (-row["score"], row["query_priority"]))
        return rows[:top_n]

