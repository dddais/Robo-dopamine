"""GroundingDINO-based target-object bbox extraction from a free-form task string.

Usage:
    g = TaskGrounding()
    box = g.ground_best(image_path, task="pick the white cube and put it on the plate")
    # box.bbox == [x1, y1, x2, y2] in pixel coords

The target object is the noun phrase following the action verb ("pick"/"grasp"/...),
not destination/context objects such as "plate". The phrase is sent to GroundingDINO
as a single-phrase query ending with a dot, which the model requires for correct
phrase-boundary tokenization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image


# Action verbs in robot manipulation tasks. The first one in the task string
# marks where the target noun phrase begins.
_ACTION_WORDS = {
    "pick", "grasp", "grab", "lift", "take", "move", "place", "put",
    "push", "pull", "drop", "insert", "pour", "fold", "stack",
    "open", "close", "release",
}
# Words that mark the end of the target noun phrase (connectors, prepositions).
_BOUNDARY_WORDS = {
    "and", "then", "to", "into", "in", "on", "onto", "from", "inside",
    "with", "near", "at", "before", "after",
}
# Words stripped from inside the target phrase but do not end it.
_FILLER_WORDS = {
    "the", "a", "an", "it", "this", "that", "is", "are",
}
_NOISE_WORDS = _ACTION_WORDS | _BOUNDARY_WORDS | _FILLER_WORDS


@dataclass
class GroundingBox:
    label: str
    score: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coords
    query: str = ""


def _normalize_task_tokens(task: str) -> List[str]:
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", task.lower())
    return [t for t in cleaned.split() if t]


def _ensure_trailing_dot(text: str) -> str:
    """GroundingDINO requires text queries to end with a dot for correct matching."""
    text = text.strip()
    if not text.endswith("."):
        text = text + " ."
    return text


def task_to_phrases(task: str) -> str:
    """Convert a free-form task string into a multi-object GroundingDINO query.

    GroundingDINO matches each '.'-separated phrase independently. We strip
    action verbs and fillers, then rejoin the surviving tokens with '.' so
    multi-object tasks ("pick the white cube and put it on the plate") become
    "white cube . plate .", giving the detector both targets in one pass.
    """
    tokens = _normalize_task_tokens(task)
    phrases: List[str] = []
    current: List[str] = []
    for t in tokens:
        if t in _NOISE_WORDS:
            if current:
                phrases.append(" ".join(current))
                current = []
        else:
            current.append(t)
    if current:
        phrases.append(" ".join(current))
    if not phrases:
        return _ensure_trailing_dot(task.lower())
    seen = set()
    unique = [p for p in phrases if not (p in seen or seen.add(p))]
    return _ensure_trailing_dot(" . ".join(unique))


def task_to_target_phrase(task: str) -> str:
    """Extract the primary manipulated object phrase from a robot task.

    For "pick the white cube and put it on yellow plate" the target is
    "white cube", not "yellow plate". Falls back to the first noun phrase
    derived by task_to_phrases() if no action verb is found.
    """
    tokens = _normalize_task_tokens(task)
    if not tokens:
        return _ensure_trailing_dot(task.lower().strip())

    start = None
    for i, tok in enumerate(tokens):
        if tok in _ACTION_WORDS:
            start = i + 1
            if start < len(tokens) and tokens[start] == "up":
                start += 1
            break

    if start is None:
        phrases = task_to_phrases(task).split(" . ")
        first = phrases[0].strip().rstrip(".").strip() if phrases else ""
        return _ensure_trailing_dot(first) if first else _ensure_trailing_dot(task.lower().strip())

    current: List[str] = []
    for tok in tokens[start:]:
        if tok in _BOUNDARY_WORDS or tok in _ACTION_WORDS:
            break
        current.append(tok)
    phrase = " ".join(t for t in current if t not in _FILLER_WORDS).strip()
    if phrase:
        return _ensure_trailing_dot(phrase)

    phrases = task_to_phrases(task).split(" . ")
    first = phrases[0].strip().rstrip(".").strip() if phrases else ""
    return _ensure_trailing_dot(first) if first else _ensure_trailing_dot(task.lower().strip())


class TaskGrounding:
    """Wrap HuggingFace GroundingDINO for single-image target-object detection.

    Loads in float32: the bfloat16 path has a known linear/mat1 dtype mismatch
    in transformers' GroundingDINO decoder.
    """

    def __init__(
        self,
        model_path: str = "../model/grounding-dino-base",
        device: str = "cuda:0",
        box_threshold: float = 0.30,
        text_threshold: float = 0.20,
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device).eval()
        self._torch = torch

    def ground(
        self,
        image_path: str,
        task: str,
        phrase_query: Optional[str] = None,
    ) -> List[GroundingBox]:
        """Detect objects in one image matching the task-derived phrase.

        Args:
            image_path: path to the image.
            task: free-form task description; passed through task_to_target_phrase
                unless phrase_query is given.
            phrase_query: override the auto-derived phrase (e.g. "white cube .").

        Returns:
            List of GroundingBox sorted by score descending. Empty list if
            GroundingDINO finds nothing above threshold.
        """
        torch = self._torch
        image = Image.open(image_path).convert("RGB")
        image_size = image.size[::-1]  # (H, W) for target_sizes
        text = phrase_query if phrase_query is not None else task_to_target_phrase(task)
        if not text.strip():
            return []

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image_size],
        )[0]

        boxes: List[GroundingBox] = []
        labels = results.get("text_labels")
        if labels is None:
            labels = results.get("labels", [])
        for bbox, label, score in zip(results["boxes"], labels, results["scores"]):
            if isinstance(label, (list, tuple)):
                label = " ".join(str(x) for x in label)
            boxes.append(
                GroundingBox(
                    label=str(label).strip().lower(),
                    score=float(score),
                    bbox=[float(x) for x in bbox.tolist()],
                    query=text,
                )
            )
        boxes.sort(key=lambda b: b.score, reverse=True)
        return boxes

    def ground_best(
        self,
        image_path: str,
        task: str,
        phrase_query: Optional[str] = None,
    ) -> Optional[GroundingBox]:
        """Return the highest-confidence target-object box, or None."""
        boxes = self.ground(image_path, task, phrase_query=phrase_query)
        return boxes[0] if boxes else None
