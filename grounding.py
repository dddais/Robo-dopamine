"""GroundingDINO-based task-relevant bbox labeling for GRM attention probes.

Two papers that study text->image attention heads (Localization Heads, CVPR 2025;
Gaze Heads, 2026) both rely on an external spatial ground truth: RefCOCO bboxes
or comic-panel boundaries. GRM has neither. To rank heads by whether they
actually attend to the task-relevant object (rather than to "something sharp
anywhere in the frame"), we need per-frame bboxes for the nouns in the task
description. This module wraps HuggingFace GroundingDINO to produce them from
the free-form task string.

Usage:
    g = TaskGrounding()
    boxes = g.ground(image_path, task="pick the white cube and put it on the plate")
    # boxes = [{"label": "white cube", "score": 0.63, "bbox": [x1,y1,x2,y2]}, ...]

The text query passed to GroundingDINO is derived from the task by splitting on
action verbs ("pick", "put", "place", ...) and connectors, and joining the
remaining noun phrases with '.' — GroundingDINO uses '.' as the phrase
separator and matches each phrase independently, so this lets the model locate
multiple task objects in one forward pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image


# Common action verbs / fillers in robot manipulation tasks. Stripping them
# leaves the noun phrases GroundingDINO should ground. Tuned for the GRM task
# style ("pick X and put it in Y", "place X on Y", "open the drawer").
_TASK_NOISE_WORDS = {
    "pick", "put", "place", "grasp", "grab", "release", "open", "close",
    "push", "pull", "lift", "drop", "insert", "pour", "fold", "stack",
    "the", "a", "an", "it", "into", "in", "on", "onto", "to", "from",
    "and", "then", "is", "are", "this", "that",
}


@dataclass
class GroundingBox:
    label: str
    score: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coords


def task_to_phrases(task: str) -> str:
    """Convert a free-form task string into a GroundingDINO phrase query.

    GroundingDINO matches each '.'-separated phrase independently. We strip
    action verbs and fillers, then rejoin the surviving tokens with '.' so
    multi-object tasks ("pick the white cube and put it on the plate") become
    "white cube . plate", giving the detector both targets in one pass.
    """
    # Normalize whitespace and punctuation, lowercase.
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", task.lower())
    tokens = [t for t in cleaned.split() if t]
    kept = [t for t in tokens if t not in _TASK_NOISE_WORDS]
    # Collapse consecutive kept tokens into phrases. A single-word phrase is
    # fine for GroundingDINO; multi-word phrases (e.g. "white cube") survive
    # because we keep them adjacent. We split phrases wherever a noise word was
    # removed so "white cube ... plate" becomes ["white cube", "plate"].
    phrases: List[str] = []
    current: List[str] = []
    prev_was_noise = False
    for t in tokens:
        if t in _TASK_NOISE_WORDS:
            if current:
                phrases.append(" ".join(current))
                current = []
            prev_was_noise = True
        else:
            # If a noise word sat between two noun tokens of the same object
            # (rare, e.g. "cube of sugar"), we still break — acceptable loss.
            current.append(t)
            prev_was_noise = False
    if current:
        phrases.append(" ".join(current))
    if not phrases:
        # Fallback: hand the whole task string to the detector.
        return task.lower()
    # Deduplicate while preserving order.
    seen = set()
    unique = [p for p in phrases if not (p in seen or seen.add(p))]
    return " . ".join(unique)


class TaskGrounding:
    """Wrap HuggingFace GroundingDINO for single-image task-relevant detection.

    Loads in float32: the bfloat16 path has a known linear/mat1 dtype mismatch
    in transformers' GroundingDINO decoder (verified on transformers 4.57.0).
    The base model is ~900MB so float32 VRAM cost is acceptable alongside GRM.
    """

    def __init__(
        self,
        model_path: str = "./model/grounding-dino-base",
        device: str = "cuda:0",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
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
        """Detect task-relevant objects in one image.

        Args:
            image_path: path to the image.
            task: free-form task description; passed through task_to_phrases
                unless phrase_query is given.
            phrase_query: override the auto-derived phrase query (e.g. "white cube . plate").

        Returns:
            List of GroundingBox sorted by score descending. Empty list if
            GroundingDINO finds nothing above threshold.
        """
        torch = self._torch
        image = Image.open(image_path).convert("RGB")
        image_size = image.size[::-1]  # (H, W) for target_sizes
        text = phrase_query if phrase_query is not None else task_to_phrases(task)
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
            # transformers < 4.51 returns string labels under "labels".
            labels = results.get("labels", [])
        for bbox, label, score in zip(results["boxes"], labels, results["scores"]):
            if isinstance(label, (list, tuple)):
                label = " ".join(str(x) for x in label)
            boxes.append(
                GroundingBox(
                    label=str(label).strip().lower(),
                    score=float(score),
                    bbox=[float(x) for x in bbox.tolist()],
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
        """Return the single highest-confidence box, or None."""
        boxes = self.ground(image_path, task, phrase_query=phrase_query)
        return boxes[0] if boxes else None


def ground_samples(
    samples: Sequence[dict],
    grounding: TaskGrounding,
    focus_labels: Sequence[str],
    sample_to_image_paths: dict,
    box_threshold: Optional[float] = None,
) -> dict:
    """Run GroundingDINO on every focus image of every sample.

    Returns a box_map keyed by sample id, then by focus label:
        {sample_id: {label: [x1,y1,x2,y2]}}

    The output is in the exact shape consumed by
    scan_localization_heads_best.load_box_map when written to JSON as
    {sample_id/label/image_basename: [x1,y1,x2,y2]}, so it can be saved and
    reloaded without re-running GroundingDINO.
    """
    if box_threshold is not None:
        grounding.box_threshold = float(box_threshold)

    box_map: dict = {}
    for sample in samples:
        sample_id = sample.get("id", "")
        image_paths = sample_to_image_paths.get(sample_id, {})
        per_label: dict = {}
        for label in focus_labels:
            path = image_paths.get(label)
            if path is None or not Path(path).exists():
                continue
            box = grounding.ground_best(path, sample["task"])
            if box is not None:
                per_label[label] = box.bbox
        if per_label:
            box_map[sample_id] = per_label
    return box_map
