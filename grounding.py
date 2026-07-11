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
multiple task objects in one forward pass.  For steering/ranking, however, we
use only the primary manipulated object (the noun phrase after "pick"/"grasp")
so destination objects such as plates do not compete with the target object.
"""

from __future__ import annotations

import json
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
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
    query: str = ""
    reject_reason: str = ""
    quality: float = 0.0
    source: str = "detector"


def _normalize_task_tokens(task: str) -> List[str]:
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", task.lower())
    return [t for t in cleaned.split() if t]


def _clean_phrase_tokens(tokens: Sequence[str]) -> str:
    kept = [t for t in tokens if t not in _TASK_NOISE_WORDS]
    return " ".join(kept).strip()


def task_to_phrases(task: str) -> str:
    """Convert a free-form task string into a multi-object GroundingDINO query.

    GroundingDINO matches each '.'-separated phrase independently. We strip
    action verbs and fillers, then rejoin the surviving tokens with '.' so
    multi-object tasks ("pick the white cube and put it on the plate") become
    "white cube . plate", giving the detector both targets in one pass.
    """
    # Normalize whitespace and punctuation, lowercase.
    tokens = _normalize_task_tokens(task)
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


def task_to_target_phrase(task: str) -> str:
    """Extract the primary manipulated object phrase from a robot task.

    For tasks like "pick the white cube and put it on yellow plate", steering
    should use "white cube" as the target bbox, not "yellow plate".  The latter
    is a destination/context object and letting it compete in the same
    GroundingDINO query can make `ground_best()` select the wrong box.
    """
    tokens = _normalize_task_tokens(task)
    if not tokens:
        return task.lower().strip()

    action_words = {"pick", "grasp", "grab", "lift", "take", "move", "place", "put"}
    stop_words = {
        "and", "then", "to", "into", "in", "on", "onto", "from", "inside",
        "with", "near", "at", "before", "after",
    }

    start = None
    for i, tok in enumerate(tokens):
        if tok in action_words:
            start = i + 1
            # "pick up the cube" / "take up the cube"
            if start < len(tokens) and tokens[start] == "up":
                start += 1
            break

    if start is None:
        # Fallback to the first derived noun phrase.
        phrases = task_to_phrases(task).split(" . ")
        return phrases[0].strip() if phrases and phrases[0].strip() else task.lower().strip()

    current: List[str] = []
    for tok in tokens[start:]:
        if tok in stop_words or tok in action_words:
            break
        current.append(tok)

    phrase = _clean_phrase_tokens(current)
    if phrase:
        return phrase

    phrases = task_to_phrases(task).split(" . ")
    return phrases[0].strip() if phrases and phrases[0].strip() else task.lower().strip()


def target_phrase_queries(task: str) -> List[str]:
    """Return target-only phrase queries to try, ordered by preference.

    GroundingDINO sometimes fails on the literal GRM object name.  We still keep
    the search target-only: aliases may describe the same manipulated object,
    but destination/context objects such as the yellow plate are never included.
    """
    phrase = task_to_target_phrase(task)
    queries = [phrase]
    if "carrot" in phrase:
        queries.extend(["orange carrot", "orange object"])
    elif "cube" in phrase:
        queries.extend(["cube"])
    elif "bottle" in phrase:
        queries.extend(["plastic bottle", "blue bottle"])
    out: List[str] = []
    seen = set()
    for q in queries:
        q = q.strip().lower()
        if q and q not in seen:
            out.append(q)
            seen.add(q)
    return out


def _box_stats(box: GroundingBox, image_path: str) -> dict:
    with Image.open(image_path) as im:
        width, height = im.size
    x1, y1, x2, y2 = box.bbox
    w = max(0.0, min(float(width), x2) - max(0.0, x1))
    h = max(0.0, min(float(height), y2) - max(0.0, y1))
    area_frac = (w * h) / max(1.0, float(width * height))
    aspect = w / max(h, 1e-6)
    cx = (max(0.0, x1) + min(float(width), x2)) / 2.0
    cy = (max(0.0, y1) + min(float(height), y2)) / 2.0
    return {
        "width": w,
        "height": h,
        "area_frac": area_frac,
        "aspect": aspect,
        "cx_frac": cx / max(1.0, float(width)),
        "cy_frac": cy / max(1.0, float(height)),
    }


def grounding_box_to_record(box: Optional[GroundingBox], image_path: str = "") -> Optional[dict]:
    """Serialize a GroundingBox plus lightweight QA stats for experiment logs."""
    if box is None:
        return None
    rec = {
        "label": box.label,
        "score": float(box.score),
        "bbox": [float(x) for x in box.bbox],
        "query": box.query,
        "reject_reason": box.reject_reason,
        "quality": float(box.quality),
        "source": box.source,
    }
    if image_path:
        try:
            rec.update({f"box_{k}": float(v) for k, v in _box_stats(box, image_path).items()})
            rec.update({f"color_{k}": float(v) for k, v in _crop_color_stats(image_path, box).items()})
        except Exception as exc:
            rec["qa_error"] = str(exc)
    return rec


def _clone_box(box: GroundingBox) -> GroundingBox:
    return GroundingBox(
        label=box.label,
        score=float(box.score),
        bbox=[float(x) for x in box.bbox],
        query=box.query,
        reject_reason=box.reject_reason,
        quality=float(box.quality),
        source=box.source,
    )


def _crop_color_stats(image_path: str, box: GroundingBox) -> dict:
    with Image.open(image_path).convert("RGB") as im:
        arr = np.asarray(im)
    x1, y1, x2, y2 = [int(round(v)) for v in box.bbox]
    x1 = max(0, min(arr.shape[1], x1))
    x2 = max(0, min(arr.shape[1], x2))
    y1 = max(0, min(arr.shape[0], y1))
    y2 = max(0, min(arr.shape[0], y2))
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return {"orange_frac": 0.0, "white_frac": 0.0, "blue_frac": 0.0, "dark_frac": 0.0}
    r = crop[..., 0].astype(np.float32)
    g = crop[..., 1].astype(np.float32)
    b = crop[..., 2].astype(np.float32)
    orange = (r > 120) & (g > 45) & (g < 190) & (b < 135) & (r > g * 1.05) & (g > b * 1.02)
    white = (r > 145) & (g > 145) & (b > 145) & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) < 75)
    blue = (b > 80) & (b > r * 1.1) & (b > g * 0.8)
    dark = (r < 65) & (g < 65) & (b < 65)
    return {
        "orange_frac": float(orange.mean()),
        "white_frac": float(white.mean()),
        "blue_frac": float(blue.mean()),
        "dark_frac": float(dark.mean()),
    }


def _mask_components(mask: np.ndarray) -> List[dict]:
    try:
        import cv2
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        comps = []
        for idx in range(1, num):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area <= 0:
                continue
            comps.append({
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "width": w,
                "height": h,
                "area": area,
                "fill": area / max(1, w * h),
                "cx": float(centroids[idx][0]),
                "cy": float(centroids[idx][1]),
            })
        return comps
    except Exception:
        # Tiny fallback to avoid a hard dependency in non-experiment contexts.
        from scipy import ndimage
        labels, num = ndimage.label(mask)
        slices = ndimage.find_objects(labels)
        comps = []
        for idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            ys, xs = slc
            area = int((labels[slc] == idx).sum())
            w = int(xs.stop - xs.start)
            h = int(ys.stop - ys.start)
            comps.append({
                "x1": int(xs.start), "y1": int(ys.start),
                "x2": int(xs.stop), "y2": int(ys.stop),
                "width": w, "height": h, "area": area,
                "fill": area / max(1, w * h),
                "cx": float((xs.start + xs.stop) / 2),
                "cy": float((ys.start + ys.stop) / 2),
            })
        return comps


def _expand_box(comp: dict, width: int, height: int, pad: int = 6) -> List[float]:
    return [
        float(max(0, comp["x1"] - pad)),
        float(max(0, comp["y1"] - pad)),
        float(min(width, comp["x2"] + pad)),
        float(min(height, comp["y2"] + pad)),
    ]


def _box_motion_penalty(prev: GroundingBox, curr: GroundingBox, prev_path: str, curr_path: str, task: str) -> float:
    """Penalty for implausible bbox jumps between adjacent trajectory frames."""
    ps = _box_stats(prev, prev_path)
    cs = _box_stats(curr, curr_path)
    dx = ps["cx_frac"] - cs["cx_frac"]
    dy = ps["cy_frac"] - cs["cy_frac"]
    dist = math.sqrt(dx * dx + dy * dy)
    area_ratio = max(ps["area_frac"], 1e-6) / max(cs["area_frac"], 1e-6)
    area_penalty = min(abs(math.log(area_ratio)), 2.0)
    target = task_to_target_phrase(task)
    max_step = 0.30 if ("carrot" in target or "cube" in target) else 0.26
    penalty = 0.45 * dist + 0.12 * area_penalty
    if dist > max_step:
        penalty += 1.5 + 5.0 * (dist - max_step)
    return float(penalty)


def _color_fallback_box(image_path: str, task: str) -> Optional[GroundingBox]:
    target = task_to_target_phrase(task)
    with Image.open(image_path).convert("RGB") as im:
        arr = np.asarray(im)
    height, width = arr.shape[:2]
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)

    candidates: List[Tuple[float, dict]] = []
    if "carrot" in target:
        mask = (r > 120) & (g > 50) & (g < 175) & (b < 130) & ((r - g) > 22) & ((g - b) > 8)
        for comp in _mask_components(mask):
            area_frac = comp["area"] / max(1, width * height)
            box_area_frac = (comp["width"] * comp["height"]) / max(1, width * height)
            aspect = comp["width"] / max(1, comp["height"])
            if comp["area"] < 120 or box_area_frac > 0.035:
                continue
            if not (0.25 <= aspect <= 2.8):
                continue
            # Avoid the large orange plate: it is broad, flat, and low fill after
            # thresholding; carrots are smaller and usually denser/narrower.
            if comp["width"] > 115 and comp["height"] > 80:
                continue
            quality = comp["area"] * comp["fill"] / max(1.0, comp["width"] + comp["height"])
            candidates.append((quality, comp))
        query = "color_fallback:carrot_orange"
    elif "cube" in target:
        mx = np.maximum.reduce([r, g, b])
        mn = np.minimum.reduce([r, g, b])
        mask = (r > 145) & (g > 145) & (b > 145) & ((mx - mn) < 75)
        for comp in _mask_components(mask):
            area_frac = comp["area"] / max(1, width * height)
            box_area_frac = (comp["width"] * comp["height"]) / max(1, width * height)
            aspect = comp["width"] / max(1, comp["height"])
            if comp["area"] < 250 or box_area_frac > 0.025:
                continue
            if not (0.45 <= aspect <= 1.8):
                continue
            if comp["cy"] < 120:
                continue
            quality = comp["area"] * comp["fill"]
            candidates.append((quality, comp))
        query = "color_fallback:white_cube"
    elif "bottle" in target:
        mask = (b > 95) & (b > r + 25) & (b > g * 0.75)
        for comp in _mask_components(mask):
            box_area_frac = (comp["width"] * comp["height"]) / max(1, width * height)
            aspect = comp["width"] / max(1, comp["height"])
            if comp["area"] < 250 or box_area_frac > 0.08:
                continue
            if aspect > 1.4:
                continue
            # Discard blue table/background blobs; target bottle components are
            # compact and not full-frame.
            if comp["width"] > 180 or comp["height"] > 240:
                continue
            quality = comp["area"] * comp["fill"] / max(1.0, aspect)
            candidates.append((quality, comp))
        query = "color_fallback:blue_bottle"
    else:
        return None

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _quality, comp = candidates[0]
    return GroundingBox(
        label=query.split(":", 1)[1],
        score=0.0,
        bbox=_expand_box(comp, width, height),
        query=query,
        source="color_fallback",
    )


def _target_box_quality(box: GroundingBox, image_path: str, task: str) -> Tuple[bool, str, float]:
    """Conservative target-box QA.

    The detector can return a high-scoring microwave/table/plate box even for a
    target-only phrase.  For causal experiments, a false target box is worse
    than skipping the frame, so these filters reject broad context boxes and
    target-specific obvious mismatches.
    """
    target = task_to_target_phrase(task)
    stats = _box_stats(box, image_path)
    colors = _crop_color_stats(image_path, box)
    area = stats["area_frac"]
    aspect = stats["aspect"]
    score = float(box.score)
    quality = score

    if area <= 0:
        return False, "empty_box", quality
    if area > 0.075:
        return False, f"too_large_area={area:.3f}", quality
    if stats["width"] < 8 or stats["height"] < 8:
        return False, "too_small", quality

    if "carrot" in target:
        if area > 0.035:
            return False, f"carrot_area_too_large={area:.3f}", quality
        if colors["orange_frac"] < 0.35:
            return False, f"carrot_not_orange={colors['orange_frac']:.3f}", quality
        if colors["white_frac"] > 0.25 and colors["white_frac"] > colors["orange_frac"] * 0.8:
            return False, f"carrot_likely_white_object={colors['white_frac']:.3f}", quality
        if stats["width"] > 115 and stats["height"] > 80:
            return False, "carrot_likely_plate", quality
        # Prefer compact orange detections over large orange plate detections.
        quality = score + 0.25 * colors["orange_frac"] - 0.35 * colors["white_frac"] - 0.8 * area
    elif "cube" in target:
        if area > 0.03:
            return False, f"cube_area_too_large={area:.3f}", quality
        if area < 0.003:
            return False, f"cube_area_too_small={area:.3f}", quality
        if colors["white_frac"] < 0.15:
            return False, f"cube_not_white={colors['white_frac']:.3f}", quality
        if stats["cx_frac"] < 0.38 and stats["cy_frac"] < 0.35:
            return False, "cube_likely_background", quality
        if stats["cy_frac"] < 0.24:
            return False, "cube_likely_top_background", quality
        top_penalty = 0.04 if stats["cy_frac"] < 0.34 else 0.0
        quality = score + 0.20 * colors["white_frac"] - 0.4 * area - top_penalty
    elif "bottle" in target:
        if area > 0.075:
            return False, f"bottle_area_too_large={area:.3f}", quality
        if stats["height"] < stats["width"] * 0.75:
            return False, f"bottle_not_tall_aspect={aspect:.2f}", quality
        if colors["orange_frac"] > 0.30:
            return False, f"bottle_likely_plate_or_carrot={colors['orange_frac']:.3f}", quality
        if colors["blue_frac"] < 0.10 and area < 0.012:
            return False, f"bottle_weak_color={colors['blue_frac']:.3f}", quality
        quality = score + 0.15 * colors["blue_frac"] - 0.25 * area
    return True, "ok", quality


class TaskGrounding:
    """Wrap HuggingFace GroundingDINO for single-image task-relevant detection.

    Loads in float32: the bfloat16 path has a known linear/mat1 dtype mismatch
    in transformers' GroundingDINO decoder (verified on transformers 4.57.0).
    The base model is ~900MB so float32 VRAM cost is acceptable alongside GRM.
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
        target_only: bool = True,
    ) -> Optional[GroundingBox]:
        """Return the highest-confidence target-object box, or None.

        By default this uses `task_to_target_phrase(task)`, not all noun
        phrases from the task.  This avoids selecting destination/context boxes
        such as "yellow plate" when the manipulated object is "white cube".
        Pass `target_only=False` or an explicit `phrase_query` for diagnostics
        that intentionally need all task objects.
        """
        if phrase_query is not None:
            queries = [phrase_query]
        elif target_only:
            queries = target_phrase_queries(task)
        else:
            queries = [task_to_phrases(task)]

        accepted: List[Tuple[float, GroundingBox]] = []
        rejected: List[GroundingBox] = []
        old_threshold = self.box_threshold
        try:
            for query in queries:
                # Alias fallbacks need access to lower-scoring candidates so QA
                # can choose compact target-like boxes over high-scoring context.
                if target_only and phrase_query is None:
                    self.box_threshold = min(old_threshold, 0.03)
                boxes = self.ground(image_path, task, phrase_query=query)
                for box in boxes:
                    box.query = query
                    if target_only:
                        ok, reason, quality = _target_box_quality(box, image_path, task)
                    else:
                        ok, reason, quality = True, "ok", float(box.score)
                    if ok:
                        box.quality = float(quality)
                        box.source = "detector"
                        accepted.append((quality, box))
                    else:
                        box.reject_reason = reason
                        rejected.append(box)
        finally:
            self.box_threshold = old_threshold

        if accepted:
            accepted.sort(key=lambda item: item[0], reverse=True)
            accepted[0][1].reject_reason = ""
            return accepted[0][1]
        if target_only:
            fallback = _color_fallback_box(image_path, task)
            if fallback is not None:
                ok, reason, quality = _target_box_quality(fallback, image_path, task)
                if ok:
                    fallback.reject_reason = ""
                    fallback.quality = float(quality) if quality else 0.01
                    return fallback
                fallback.reject_reason = reason
            return None
        return None

    def ground_candidates(
        self,
        image_path: str,
        task: str,
        phrase_query: Optional[str] = None,
        target_only: bool = True,
        include_color_fallback: bool = True,
        max_candidates: int = 8,
    ) -> List[GroundingBox]:
        """Return QA-accepted candidate boxes sorted by single-frame quality.

        `ground_best()` is intentionally stateless. For trajectory experiments
        we need more than one candidate per frame so a temporal smoother can
        avoid sudden jumps to context objects such as a microwave or plate.
        """
        if phrase_query is not None:
            queries = [phrase_query]
        elif target_only:
            queries = target_phrase_queries(task)
        else:
            queries = [task_to_phrases(task)]

        accepted: List[GroundingBox] = []
        seen = set()
        old_threshold = self.box_threshold
        try:
            for query in queries:
                if target_only and phrase_query is None:
                    self.box_threshold = min(old_threshold, 0.03)
                boxes = self.ground(image_path, task, phrase_query=query)
                for box in boxes:
                    box.query = query
                    box.source = "detector"
                    if target_only:
                        ok, reason, quality = _target_box_quality(box, image_path, task)
                    else:
                        ok, reason, quality = True, "ok", float(box.score)
                    if not ok:
                        box.reject_reason = reason
                        continue
                    key = tuple(round(float(x), 1) for x in box.bbox)
                    if key in seen:
                        continue
                    seen.add(key)
                    box.reject_reason = ""
                    box.quality = float(quality)
                    accepted.append(box)
        finally:
            self.box_threshold = old_threshold

        if target_only and include_color_fallback:
            fallback = _color_fallback_box(image_path, task)
            if fallback is not None:
                ok, reason, quality = _target_box_quality(fallback, image_path, task)
                if ok:
                    fallback.reject_reason = ""
                    fallback.quality = float(quality) if quality else 0.01
                    key = tuple(round(float(x), 1) for x in fallback.bbox)
                    if key not in seen:
                        accepted.append(fallback)
                else:
                    fallback.reject_reason = reason

        accepted.sort(key=lambda b: float(b.quality), reverse=True)
        return [_clone_box(b) for b in accepted[:max(1, int(max_candidates))]]

    def ground_best_sequence(
        self,
        image_paths: Sequence[str],
        tasks: Sequence[str] | str,
        target_only: bool = True,
        max_candidates: int = 8,
        allow_missing: bool = True,
        write_json: Optional[str | Path] = None,
    ) -> List[Optional[GroundingBox]]:
        """Ground a trajectory with temporal consistency.

        The per-frame detector is allowed to be noisy. This routine first keeps
        all QA-accepted target candidates, then chooses a path that trades off
        candidate quality against adjacent-frame bbox jumps. A None state is
        available when `allow_missing` is true; skipping a suspicious frame is
        safer than steering/ranking against a wrong object.
        """
        if isinstance(tasks, str):
            task_list = [tasks for _ in image_paths]
        else:
            task_list = list(tasks)
        if len(task_list) != len(image_paths):
            raise ValueError(f"tasks length {len(task_list)} != image_paths length {len(image_paths)}")

        cand_seq: List[List[GroundingBox]] = []
        for path, task in zip(image_paths, task_list):
            cand_seq.append(self.ground_candidates(
                path,
                task,
                target_only=target_only,
                include_color_fallback=True,
                max_candidates=max_candidates,
            ))

        states: List[List[Optional[GroundingBox]]] = []
        for cands in cand_seq:
            state = list(cands)
            if allow_missing:
                state.append(None)
            states.append(state)

        if not states:
            return []

        dp: List[List[float]] = []
        back: List[List[Optional[int]]] = []
        for i, state in enumerate(states):
            costs: List[float] = []
            prev_idx: List[Optional[int]] = []
            for j, box in enumerate(state):
                if box is None:
                    unary = 1.15
                else:
                    unary = max(0.0, 1.0 - float(box.quality))
                    if box.source == "color_fallback":
                        unary += 0.25
                if i == 0:
                    costs.append(unary)
                    prev_idx.append(None)
                    continue

                best_cost = float("inf")
                best_prev: Optional[int] = None
                for k, prev_box in enumerate(states[i - 1]):
                    prev_cost = dp[i - 1][k]
                    if box is None and prev_box is None:
                        trans = 0.05
                    elif box is None or prev_box is None:
                        trans = 0.45
                    else:
                        trans = _box_motion_penalty(prev_box, box, image_paths[i - 1], image_paths[i], task_list[i])
                    total = prev_cost + trans + unary
                    if total < best_cost:
                        best_cost = total
                        best_prev = k
                costs.append(best_cost)
                prev_idx.append(best_prev)
            dp.append(costs)
            back.append(prev_idx)

        end_idx = min(range(len(dp[-1])), key=lambda idx: dp[-1][idx])
        selected: List[Optional[GroundingBox]] = [None for _ in states]
        idx: Optional[int] = end_idx
        for i in range(len(states) - 1, -1, -1):
            if idx is None:
                break
            selected[i] = _clone_box(states[i][idx]) if states[i][idx] is not None else None
            idx = back[i][idx]

        if write_json is not None:
            records = []
            for i, (path, task, cands, chosen) in enumerate(zip(image_paths, task_list, cand_seq, selected)):
                records.append({
                    "index": i,
                    "image": str(path),
                    "task": task,
                    "chosen": grounding_box_to_record(chosen, path),
                    "candidates": [grounding_box_to_record(c, path) for c in cands],
                })
            out = Path(write_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(records, indent=2))

        return selected


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
