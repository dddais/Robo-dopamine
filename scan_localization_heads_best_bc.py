#!/usr/bin/env python3
"""
Offline attention probe for Robo-Dopamine GRM.

This script follows the spirit of localization-head analyses:

1. Run GRM with Transformers/eager attention rather than vLLM.
2. Measure how much each text-layer/head attends to each image span.
3. Prefer heads with high image-attention mass and low spatial entropy.
4. Save heatmap overlays for the top heads.
5. If a target bbox is supplied, quantify how much heatmap mass falls on it.

It is intentionally separate from examples/inference.py because the normal vLLM
path is optimized for reward inference and does not expose attention maps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
MODEL_PATH ="./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview"
# MODEL_PATH ="/home/dais/workspace/Robo-Dopamine/train/checkpoints/obj_instruction_mismatch_finetune"
CAM_HIGH="/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot/cam_high.mp4"
CAM_LEFT="/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot/cam_left_wrist.mp4"
CAM_RIGHT="/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot/cam_right_wrist.mp4"
GOAL_IMAGE="/home/dais/workspace/Robo-Dopamine/examples/blank_goal.png"
TASK="pick the white cube and put it in the plate"
EVAL_MODE="incremental"
FRAME_INTERVAL=20
MAX_SAMPLES=1
SAMPLE_INDEX=0
LAYERS="all"
FOCUS_IMAGES="after_cam_high,after_cam_left_wrist,after_cam_right_wrist"
OUT_DIR="./results/grm_attention_probe_new_cube_incremental_4"

SAVE_ATTENTION_VIDEOS=True
ATTENTION_VIDEO_TOP_K=2
ATTENTION_VIDEO_FPS=0
ATTENTION_VIDEO_ALPHA=0.45
MAX_SAMPLES=0
SAMPLE_INDEX=-1

SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START - Robot Front Image (task just starting): <image>
- REFERENCE END - Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = "just beginning"; REFERENCE END = "fully completed."
   - Visually estimate how far BEFORE and AFTER are along this START->END continuum.
2) Direction:
   - AFTER better than BEFORE -> positive score.
   - AFTER worse than BEFORE -> negative score.
   - Essentially the same -> 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged -> 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""

IMAGE_LABELS = [
    "reference_start",
    "reference_end",
    "before_cam_high",
    "before_cam_left_wrist",
    "before_cam_right_wrist",
    "after_cam_high",
    "after_cam_left_wrist",
    "after_cam_right_wrist",
]


@dataclass
class ImageSpan:
    label: str
    path: str
    start: int
    end: int
    grid_thw: Tuple[int, int, int]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_pngs_sorted(dir_path: Path) -> List[Path]:
    return sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".png")


def get_frame_count(path: Path) -> Tuple[str, int]:
    if path.is_dir():
        files = list_pngs_sorted(path)
        if not files:
            raise RuntimeError(f"No PNG frames found in directory: {path}")
        return "dir", len(files)

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Invalid frame count from video: {path}")
    return "video", n


def make_sample_indices_by_interval(num_frames: int, interval: int) -> List[int]:
    if num_frames < 1:
        return []
    indices = list(range(0, num_frames, interval))
    last_idx = num_frames - 1
    if not indices or indices[-1] != last_idx:
        indices.append(last_idx)
    return indices


def save_frames(src_path: Path, out_dir: Path, indices: Sequence[int], src_type: str) -> None:
    ensure_dir(out_dir)
    if src_type == "dir":
        files = list_pngs_sorted(src_path)
        for idx in indices:
            if 0 <= idx < len(files):
                shutil.copyfile(files[idx], out_dir / f"frame_{idx:06d}.png")
        return

    import cv2

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARN] Failed to read frame {idx} from {src_path}")
                continue
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    finally:
        cap.release()


def build_samples_json(run_root: Path, task: str, indices: Sequence[int], ref_end_path: str, mode: str) -> List[Dict]:
    cache_root = run_root / ".cache"
    items: List[Dict] = []
    if len(indices) < 2:
        return items

    for k in range(len(indices) - 1):
        af = indices[k + 1]
        if mode == "incremental":
            bf = indices[k]
            bf_id = f"bf_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "forward":
            bf = indices[0]
            bf_id = f"start_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist" / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "backward":
            bf_id = "goal"
            bf_images = [ref_end_path, ref_end_path, ref_end_path]
        else:
            raise ValueError(f"Unknown eval mode: {mode}")

        af_images = [
            str(cache_root / "cam_high" / f"frame_{af:06d}.png"),
            str(cache_root / "cam_left_wrist" / f"frame_{af:06d}.png"),
            str(cache_root / "cam_right_wrist" / f"frame_{af:06d}.png"),
        ]
        items.append(
            {
                "id": f"step-{run_root.name}-{k:04d}-{bf_id}-af_{af:06d}",
                "task": task,
                "image": [
                    str(cache_root / "cam_high" / f"frame_{0:06d}.png"),
                    ref_end_path,
                    bf_images[0],
                    bf_images[1],
                    bf_images[2],
                    af_images[0],
                    af_images[1],
                    af_images[2],
                ],
            }
        )
    return items


def samples_from_videos(args: argparse.Namespace, out_dir: Path) -> List[Dict]:
    missing = [name for name in ["cam_high", "cam_left", "cam_right", "task", "goal_image"] if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "Either pass --sample-json, or provide all raw inputs: "
            "--cam-high --cam-left --cam-right --task --goal-image. "
            f"Missing: {', '.join(missing)}"
        )

    run_root = out_dir
    cache_root = run_root / ".cache"
    cam_dirs = {
        "cam_high": cache_root / "cam_high",
        "cam_left_wrist": cache_root / "cam_left_wrist",
        "cam_right_wrist": cache_root / "cam_right_wrist",
    }
    for path in cam_dirs.values():
        ensure_dir(path)

    paths = [Path(args.cam_high), Path(args.cam_left), Path(args.cam_right)]
    types_counts = [get_frame_count(path) for path in paths]
    counts = [count for _, count in types_counts]
    if len(set(counts)) != 1:
        raise ValueError(f"Frame count mismatch among cameras: {counts}")

    indices = make_sample_indices_by_interval(counts[0], args.frame_interval)
    for src, key, (src_type, _) in zip(paths, cam_dirs.keys(), types_counts):
        save_frames(src, cam_dirs[key], indices, src_type)

    ref_end_path = cache_root / "ref_end.png"
    shutil.copyfile(args.goal_image, ref_end_path)
    samples = build_samples_json(run_root, args.task, indices, str(ref_end_path), mode=args.eval_mode)
    with open(run_root / "sample.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    return samples


def parse_layer_spec(spec: str, num_layers: int, skip_early: int = 0) -> List[int]:
    """Parse a layer specification into an explicit list of layer indices.

    Early layers (0 .. skip_early-1) are excluded because, as Kang et al. CVPR 2025
    §4 note, "the early layers are known to operate differently from the other
    layers" — their attention is dominated by raw embedding similarity and
    positional bias rather than learned semantic grounding. Letting them compete
    in the localization ranking surfaces layer-0 artifacts (e.g. a head whose
    high image-attention mass comes from token-embedding cosine similarity, not
    from genuine target focusing).
    """
    if spec == "all":
        base = list(range(num_layers))
    else:
        layers = set()
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if part == "last":
                layers.add(num_layers - 1)
            elif "-" in part:
                a, b = part.split("-", 1)
                layers.update(range(int(a), int(b) + 1))
            else:
                layers.add(int(part))
        base = sorted(x for x in layers if 0 <= x < num_layers)

    if skip_early > 0:
        kept = [x for x in base if x >= skip_early]
        if not kept:
            print(f"[WARN] --skip-early-layers={skip_early} removed every layer; ignoring.")
            return base
        return kept
    return base


def parse_focus_labels(raw: str) -> List[str]:
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in labels if x not in IMAGE_LABELS]
    if unknown:
        raise ValueError(f"Unknown focus labels {unknown}. Valid labels: {IMAGE_LABELS}")
    return labels


def parse_box(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    vals = [float(x) for x in re.split(r"[, ]+", raw.strip()) if x]
    if len(vals) != 4:
        raise ValueError("--target-box must contain four numbers: x1,y1,x2,y2")
    return vals


def load_box_map(args: argparse.Namespace) -> Dict[str, List[float]]:
    box_map: Dict[str, List[float]] = {}
    shared = parse_box(args.target_box)
    if shared is not None:
        for label in parse_focus_labels(args.focus_images):
            box_map[label] = shared

    if args.target_box_json:
        with open(args.target_box_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, val in raw.items():
            if not isinstance(val, list) or len(val) != 4:
                raise ValueError(f"Box for {key!r} must be [x1, y1, x2, y2]")
            box_map[key] = [float(x) for x in val]
    return box_map


def dtype_from_arg(torch, raw: str):
    if raw == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[raw]


def build_prompt(processor, task: str, analysis_suffix: Optional[str], score_suffix: bool = False) -> str:
    prompt = SYSTEM_PROMPT.format(task=task)
    if analysis_suffix:
        prompt = prompt.rstrip() + "\n\nAttention probe question: " + analysis_suffix.strip() + "\n"
    parts = prompt.split("<image>")
    if len(parts) != 9:
        raise RuntimeError(f"Expected 8 <image> placeholders, got {len(parts) - 1}")

    content: List[Dict] = [{"type": "text", "text": parts[0]}]
    for idx in range(8):
        content.append({"type": "image"})
        content.append({"type": "text", "text": parts[idx + 1]})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Teacher-force a canonical score so the numeric token exists in input_ids
    # and can serve as the localization query (Kang's "last text token" analogue).
    if score_suffix:
        text = text + "<score>0%</score>"
    return text


def find_contiguous_spans(ids: Sequence[int], token_id: int) -> List[Tuple[int, int]]:
    spans = []
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] != token_id:
            i += 1
            continue
        start = i
        while i < n and ids[i] == token_id:
            i += 1
        spans.append((start, i))
    return spans


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    starts = []
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            starts.append(i)
    return starts


def select_query_positions(
    tokenizer,
    input_ids: Sequence[int],
    image_spans: Sequence[ImageSpan],
    mode: str,
    tail_tokens: int,
    query_text: Optional[str],
    special_ids: Iterable[int],
) -> Tuple[List[int], str]:
    last_image_end = max(span.end for span in image_spans)
    special = set(special_ids)

    if query_text:
        needle = tokenizer(query_text, add_special_tokens=False)["input_ids"]
        starts = find_subsequence(input_ids, needle)
        positions: List[int] = []
        for start in starts:
            positions.extend(range(start, start + len(needle)))
        after = [pos for pos in positions if pos >= last_image_end and input_ids[pos] not in special]
        if after:
            return after, f"query_text:{query_text}"
        print(
            f"[WARN] Query text {query_text!r} was not found after the image tokens. "
            "In a causal decoder, text before images cannot attend to later image tokens. "
            "Falling back to tail query tokens."
        )

    if mode == "score":
        # Teacher-forced canonical score: query on the numeric token of "<score>0%</score>".
        # This mirrors Kang et al. using the last input text token as the representative
        # query, applied to a GRM whose output is a score string rather than a free-form answer.
        needle = tokenizer("0", add_special_tokens=False)["input_ids"]
        starts = find_subsequence(input_ids, needle)
        after = [pos for pos in starts if pos >= last_image_end and input_ids[pos] not in special]
        if after:
            return after, "score:0%"
        print("[WARN] score-mode needle '0' not found after images; falling back to tail.")

    valid_after = [idx for idx in range(last_image_end, len(input_ids)) if input_ids[idx] not in special]
    if not valid_after:
        raise RuntimeError("Could not find non-special query tokens after the image spans.")

    if mode == "all_after_images":
        return valid_after, mode
    if mode == "tail":
        return valid_after[-tail_tokens:], f"tail:{tail_tokens}"
    raise ValueError(f"Unknown query mode: {mode}")


# Sentinel for pathological attention maps (empty / all-constant / no peak).
# Must be a LARGE value so that bad heads rank LAST under "lower entropy = better".
# We use log(num_cells) + 1, i.e. the entropy of a uniform distribution + 1,
# so it sits just above any legitimate entropy value for the same grid size.
ENTROPY_SENTINEL_HIGH = float("inf")


def spatial_entropy(grid: np.ndarray, threshold_rel: float) -> float:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return ENTROPY_SENTINEL_HIGH
    arr = arr - np.nanmin(arr)
    max_val = float(np.nanmax(arr))
    if max_val <= 0:
        # All cells identical: no localization signal. Penalize, do not reward.
        return ENTROPY_SENTINEL_HIGH
    mask = arr >= (threshold_rel * max_val)
    if not mask.any():
        return ENTROPY_SENTINEL_HIGH

    # 8-connectivity, matching Kang et al. CVPR 2025 §4.1.
    visited = np.zeros(mask.shape, dtype=bool)
    sizes: List[int] = []
    h, w = mask.shape
    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    for r in range(h):
        for c in range(w):
            if visited[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)

    total = float(sum(sizes))
    probs = [size / total for size in sizes if size > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def vector_to_grid(vec: np.ndarray, grid_thw: Tuple[int, int, int], spatial_merge_size: int) -> np.ndarray:
    """Reshape a flat image-token attention vector into a 2D spatial grid.

    For video inputs (t > 1) the time dimension is averaged out, returning a
    single (gh, gw) map. The GRM probe use case is single-frame (t == 1), so
    this averaging is a no-op there; the behavior is documented to avoid
    surprises when feeding multi-frame inputs.
    """
    t, h, w = grid_thw
    gh = max(1, h // spatial_merge_size)
    gw = max(1, w // spatial_merge_size)
    expected = int(t * gh * gw)
    if vec.size != expected:
        # Some processors already report the merged grid. Fall back to the raw h*w layout.
        raw_expected = int(t * h * w)
        if vec.size == raw_expected:
            gh, gw = h, w
            expected = raw_expected
        else:
            raise ValueError(f"Cannot reshape {vec.size} image tokens into grid {grid_thw} with merge={spatial_merge_size}")
    return vec.reshape(t, gh, gw).mean(axis=0)


def normalize_heatmap(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float64)
    arr = arr - np.nanmin(arr)
    denom = np.nanmax(arr)
    if denom <= 0:
        return np.zeros_like(arr)
    return arr / denom


def resize_heatmap_to_image(grid: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    heat = normalize_heatmap(grid)
    img = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255))
    img = img.resize(image_size, resample=Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def mask_grid_from_box(image_path: str, grid_shape: Tuple[int, int], box: Sequence[float]) -> np.ndarray:
    with Image.open(image_path) as im:
        width, height = im.size
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))

    gh, gw = grid_shape
    yy = (np.arange(gh) + 0.5) * height / gh
    xx = (np.arange(gw) + 0.5) * width / gw
    xmask = (xx >= x1) & (xx <= x2)
    ymask = (yy >= y1) & (yy <= y2)
    return np.outer(ymask, xmask)


def target_metrics(grid: np.ndarray, image_path: str, box: Sequence[float], threshold_rel: float) -> Dict[str, float]:
    heat = np.asarray(grid, dtype=np.float64)
    mask = mask_grid_from_box(image_path, heat.shape, box)
    total_mass = float(heat.sum())
    target_mass = float(heat[mask].sum()) if mask.any() else 0.0
    outside = ~mask
    target_area = float(mask.sum())
    outside_area = float(outside.sum())
    target_density = target_mass / max(target_area, 1.0)
    outside_density = float(heat[outside].sum()) / max(outside_area, 1.0)
    pred = normalize_heatmap(heat) >= threshold_rel
    union = np.logical_or(pred, mask)
    inter = np.logical_and(pred, mask)
    return {
        "target_mass": target_mass,
        "target_fraction": target_mass / max(total_mass, 1e-12),
        "target_density": target_density,
        "outside_density": outside_density,
        "density_ratio": target_density / max(outside_density, 1e-12),
        "iou": float(inter.sum() / max(union.sum(), 1)),
        "target_grid_area": target_area,
    }


def save_overlay(image_path: str, grid: np.ndarray, out_path: Path, title: str, box: Optional[Sequence[float]]) -> None:
    with Image.open(image_path).convert("RGB") as im:
        image = np.asarray(im)
        heat = resize_heatmap_to_image(grid, im.size)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.imshow(image)
    ax.imshow(heat, cmap="jet", alpha=0.45)
    if box is not None:
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=2))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_overlay_frame(
    image_path: str,
    grid: np.ndarray,
    box: Optional[Sequence[float]],
    alpha: float,
    title: Optional[str] = None,
) -> np.ndarray:
    import cv2

    with Image.open(image_path).convert("RGB") as im:
        image = np.asarray(im).copy()
        heat = resize_heatmap_to_image(grid, im.size)

    heat_u8 = np.uint8(np.clip(heat, 0, 1) * 255)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = np.clip(
        (1.0 - alpha) * image.astype(np.float32) + alpha * heat_color.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    if box is not None:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        max_chars = max(24, overlay.shape[1] // 8)
        text = title if len(title) <= max_chars else title[: max_chars - 3] + "..."
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(overlay, (0, 0), (min(overlay.shape[1], tw + 8), th + baseline + 8), (0, 0, 0), -1)
        cv2.putText(overlay, text, (4, th + 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def frame_id_from_path(path: str, fallback: int) -> str:
    match = re.search(r"frame_(\d+)", Path(path).stem)
    if match:
        return match.group(1)
    return f"{fallback:04d}"


def infer_attention_video_fps(args: argparse.Namespace) -> float:
    if args.attention_video_fps > 0:
        return float(args.attention_video_fps)
    if args.sample_json:
        return 5.0
    try:
        import cv2

        cap = cv2.VideoCapture(str(args.cam_high))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        if fps > 0:
            return max(0.1, fps / max(int(args.frame_interval), 1))
    except Exception:
        pass
    return 5.0


def save_top_head_attention_videos(
    sample_results: Sequence[Dict[str, Dict[str, np.ndarray]]],
    sample_spans_by_label: Sequence[Dict[str, ImageSpan]],
    metas: Sequence[Dict],
    top_heads: Sequence[Dict],
    focus_labels: Sequence[str],
    box_map: Dict[str, List[float]],
    out_dir: Path,
    args: argparse.Namespace,
) -> List[Dict]:
    import cv2

    video_dir = out_dir / "attention_videos"
    ensure_dir(video_dir)
    fps = infer_attention_video_fps(args)
    selected_heads = list(top_heads)
    if args.attention_video_top_k > 0:
        selected_heads = selected_heads[: args.attention_video_top_k]

    manifest: List[Dict] = []
    for head in selected_heads:
        li = int(head["layer_table_index"])
        hi = int(head["head"])
        for label in focus_labels:
            out_path = video_dir / f"{safe_filename(head['label'])}_{safe_filename(label)}.mp4"
            writer = None
            frame_size: Optional[Tuple[int, int]] = None
            num_frames = 0
            try:
                for frame_idx, (result, span_by_label) in enumerate(zip(sample_results, sample_spans_by_label)):
                    span = span_by_label[label]
                    box = lookup_box(box_map, label, span)
                    s_img = float(result[label]["s_img"][li, hi])
                    title = f"{head['label']} {label} frame={frame_id_from_path(span.path, frame_idx)} Simg={s_img:.4f}"
                    frame = render_overlay_frame(
                        span.path,
                        result[label]["heatmaps"][li, hi],
                        box,
                        alpha=float(args.attention_video_alpha),
                        title=title,
                    )
                    height, width = frame.shape[:2]
                    if writer is None:
                        frame_size = (width, height)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open video writer for {out_path}")
                    elif frame_size is not None and (width, height) != frame_size:
                        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    num_frames += 1
            finally:
                if writer is not None:
                    writer.release()

            manifest.append(
                {
                    "head": head["label"],
                    "label": label,
                    "path": str(out_path),
                    "fps": fps,
                    "num_frames": num_frames,
                }
            )
            print(f"[video] wrote {out_path} ({num_frames} frames @ {fps:.2f} fps)")

    return manifest


def save_matrix_plot(table: np.ndarray, out_path: Path, title: str, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    im = ax.imshow(table, aspect="auto", cmap=cmap)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def infer_image_spans(inputs, config, image_paths: Sequence[str]) -> List[ImageSpan]:
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    image_token_id = int(getattr(config, "image_token_id", 151655))
    token_spans = find_contiguous_spans(ids, image_token_id)
    grids = inputs.get("image_grid_thw")
    if grids is None:
        raise RuntimeError("Processor did not return image_grid_thw; cannot map image tokens back to grids.")
    grid_list = [tuple(int(x) for x in row) for row in grids.detach().cpu().tolist()]
    if len(token_spans) != len(image_paths):
        raise RuntimeError(f"Found {len(token_spans)} image token spans for {len(image_paths)} images.")
    if len(grid_list) != len(image_paths):
        raise RuntimeError(f"Found {len(grid_list)} image grids for {len(image_paths)} images.")
    return [
        ImageSpan(label=IMAGE_LABELS[i], path=image_paths[i], start=start, end=end, grid_thw=grid_list[i])
        for i, (start, end) in enumerate(token_spans)
    ]


def move_inputs_to_device(torch, inputs, device, dtype):
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if key in {"pixel_values", "pixel_values_videos"}:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def load_model_and_processor(args: argparse.Namespace):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    dtype = dtype_from_arg(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        if args.max_pixels is not None:
            processor.image_processor.max_pixels = args.max_pixels
        if args.min_pixels is not None:
            processor.image_processor.min_pixels = args.min_pixels

    kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "device_map": None if args.device_map == "none" else args.device_map,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=dtype, **kwargs)
    except Exception as first_error:
        cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        if getattr(cfg, "model_type", "") != "qwen3_vl":
            raise
        from transformers import Qwen3VLForConditionalGeneration

        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, dtype=dtype, **kwargs)
        except TypeError:
            model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=dtype, **kwargs)
        except Exception:
            raise first_error

    model.eval()
    model.config.output_attentions = True
    model.config.use_cache = False
    return torch, model, processor, dtype


def scan_sample(
    torch,
    model,
    processor,
    sample: Dict,
    args: argparse.Namespace,
    dtype,
    focus_labels: Sequence[str],
    layers: Optional[List[int]],
) -> Tuple[Dict, Dict[str, Dict[str, np.ndarray]], Dict[str, ImageSpan], List[int], str]:
    image_paths = sample["image"]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    score_suffix = args.query_mode == "score"
    prompt = build_prompt(processor, sample["task"], args.analysis_suffix, score_suffix=score_suffix)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    spans = infer_image_spans(inputs, model.config, image_paths)
    span_by_label = {span.label: span for span in spans}

    ids = inputs["input_ids"][0].detach().cpu().tolist()
    special_ids = [
        int(getattr(model.config, "image_token_id", 151655)),
        int(getattr(model.config, "video_token_id", 151656)),
        int(getattr(model.config, "vision_start_token_id", 151652)),
        int(getattr(model.config, "vision_end_token_id", 151653)),
    ]
    query_positions, query_desc = select_query_positions(
        processor.tokenizer,
        ids,
        spans,
        mode=args.query_mode,
        tail_tokens=args.tail_query_tokens,
        query_text=args.query_text,
        special_ids=special_ids,
    )

    device = next(model.parameters()).device
    inputs = move_inputs_to_device(torch, inputs, device, dtype)
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model returned no attentions. Ensure attn_implementation='eager'.")
    if layers is None:
        selected_layers = list(range(len(attentions)))
    else:
        selected_layers = layers

    spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    per_label: Dict[str, Dict[str, List[np.ndarray]]] = {
        label: {"s_img": [], "entropy": [], "heatmaps": []} for label in focus_labels
    }

    for layer_idx in selected_layers:
        attn = attentions[layer_idx]
        # Shape: [batch, heads, query_len, key_len].
        attn_cpu = attn[0, :, query_positions, :].detach().float().cpu()
        for label in focus_labels:
            span = span_by_label[label]
            block = attn_cpu[:, :, span.start : span.end].mean(dim=1).numpy()
            s_img = block.sum(axis=1)
            heatmaps = []
            entropies = []
            for head_idx in range(block.shape[0]):
                grid = vector_to_grid(block[head_idx], span.grid_thw, spatial_merge_size)
                heatmaps.append(grid)
                entropies.append(spatial_entropy(grid, args.entropy_threshold_rel))
            per_label[label]["s_img"].append(s_img)
            per_label[label]["entropy"].append(np.asarray(entropies, dtype=np.float64))
            per_label[label]["heatmaps"].append(np.stack(heatmaps, axis=0))

    packed: Dict[str, Dict[str, np.ndarray]] = {}
    for label, vals in per_label.items():
        packed[label] = {
            "s_img": np.stack(vals["s_img"], axis=0),
            "entropy": np.stack(vals["entropy"], axis=0),
            "heatmaps": np.stack(vals["heatmaps"], axis=0),
        }

    meta = {
        "sample_id": sample.get("id"),
        "task": sample["task"],
        "image_grid_thw": [span.grid_thw for span in spans],
        "query_positions": query_positions,
        "query_description": query_desc,
        "layers": selected_layers,
    }
    return meta, packed, span_by_label, query_positions, query_desc


def lookup_box(box_map: Dict[str, List[float]], label: str, span: ImageSpan) -> Optional[List[float]]:
    # Explicit membership checks; avoids the falsy trap of `dict.get(x) or dict.get(y)`
    # when a box legitimately equals a falsy-like value.
    if label in box_map:
        return box_map[label]
    base = Path(span.path).name
    if base in box_map:
        return box_map[base]
    if span.path in box_map:
        return box_map[span.path]
    return None


def rank_heads(
    aggregate: Dict[str, Dict[str, np.ndarray]],
    target_tables: Dict[str, Dict[str, np.ndarray]],
    focus_labels: Sequence[str],
    rank_by: str,
    top_k: int,
    layer_ids: Sequence[int],
    s_img_floor: float = 0.0,
) -> List[Dict]:
    s = np.mean([aggregate[label]["s_img"] for label in focus_labels], axis=0)
    e = np.mean([aggregate[label]["entropy"] for label in focus_labels], axis=0)

    if rank_by == "localization":
        # Stage 1 (Kang Criterion 1): drop heads whose image-attention mass is
        # below the floor. These heads simply do not look at the image enough
        # to be localization candidates; letting them compete on entropy alone
        # is what previously pushed near-empty maps (entropy=0 via the bug)
        # to the top.
        mask_pass = s >= s_img_floor
        if not mask_pass.any():
            print(f"[WARN] No head passes s_img_floor={s_img_floor}; disabling floor.")
            mask_pass = np.ones_like(s, dtype=bool)

        s_eff = s.copy()
        e_eff = e.copy()
        # Mask out failed heads so they cannot win after normalization.
        s_eff[~mask_pass] = s.min()
        e_eff[~mask_pass] = float(np.nanmax(e[np.isfinite(e)])) if np.isfinite(e).any() else 0.0

        s_norm = (s_eff - s_eff.min()) / max(float(s_eff.max() - s_eff.min()), 1e-12)
        # Replace inf entropy with the max finite entropy before normalization
        # so pathological maps do not dominate the (1 - e_norm) term.
        finite_e = np.where(np.isfinite(e_eff), e_eff, -1.0)
        e_max = float(finite_e.max()) if finite_e.size else 0.0
        e_for_norm = np.where(np.isfinite(e_eff), e_eff, e_max)
        e_norm = (e_for_norm - e_for_norm.min()) / max(float(e_for_norm.max() - e_for_norm.min()), 1e-12)
        score = s_norm * (1.0 - e_norm)
        score[~mask_pass] = -np.inf
    else:
        metric_stack = []
        for label in focus_labels:
            if label in target_tables and rank_by in target_tables[label]:
                metric_stack.append(target_tables[label][rank_by])
        if not metric_stack:
            print(f"[WARN] No target metric {rank_by!r}; falling back to localization score.")
            return rank_heads(aggregate, target_tables, focus_labels, "localization", top_k, layer_ids, s_img_floor)
        score = np.nanmean(metric_stack, axis=0)

    flat = np.argsort(score.reshape(-1))[::-1]
    num_heads = score.shape[1]
    result = []
    for flat_idx in flat[:top_k]:
        li = int(flat_idx // num_heads)
        hi = int(flat_idx % num_heads)
        if not np.isfinite(score[li, hi]):
            continue
        result.append(
            {
                "layer": int(layer_ids[li]),
                "layer_table_index": li,
                "head": hi,
                "label": f"L{layer_ids[li]}_H{hi}",
                "s_img": float(s[li, hi]),
                "entropy": float(e[li, hi]),
                "score": float(score[li, hi]),
            }
        )
    return result


def aggregate_samples(sample_results: List[Dict[str, Dict[str, np.ndarray]]], focus_labels: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    aggregate: Dict[str, Dict[str, np.ndarray]] = {}
    for label in focus_labels:
        aggregate[label] = {
            "s_img": np.mean([res[label]["s_img"] for res in sample_results], axis=0),
            "entropy": np.mean([res[label]["entropy"] for res in sample_results], axis=0),
            "heatmaps": np.mean([res[label]["heatmaps"] for res in sample_results], axis=0),
        }
    return aggregate


def compute_target_tables(
    aggregate: Dict[str, Dict[str, np.ndarray]],
    focus_labels: Sequence[str],
    span_by_label: Dict[str, ImageSpan],
    box_map: Dict[str, List[float]],
    threshold_rel: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    tables: Dict[str, Dict[str, np.ndarray]] = {}
    for label in focus_labels:
        box = lookup_box(box_map, label, span_by_label[label])
        if box is None:
            continue
        heatmaps = aggregate[label]["heatmaps"]
        shape = heatmaps.shape[:2]
        metric_names = ["target_mass", "target_fraction", "target_density", "outside_density", "density_ratio", "iou"]
        label_tables = {name: np.full(shape, np.nan, dtype=np.float64) for name in metric_names}
        for li in range(shape[0]):
            for hi in range(shape[1]):
                metrics = target_metrics(heatmaps[li, hi], span_by_label[label].path, box, threshold_rel)
                for name in metric_names:
                    label_tables[name][li, hi] = metrics[name]
        tables[label] = label_tables
    return tables


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "top_heads")

    if args.sample_json:
        with open(args.sample_json, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        samples = samples_from_videos(args, out_dir)

    if args.sample_index is not None and args.sample_index >= 0:
        samples = [samples[args.sample_index]]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No samples to scan.")

    focus_labels = parse_focus_labels(args.focus_images)
    box_map = load_box_map(args)

    torch, model, processor, dtype = load_model_and_processor(args)
    num_layers = int(getattr(model.config.text_config, "num_hidden_layers", getattr(model.config, "num_hidden_layers", 0)))
    selected_layers = parse_layer_spec(args.layers, num_layers, skip_early=int(args.skip_early_layers))

    sample_results: List[Dict[str, Dict[str, np.ndarray]]] = []
    sample_spans_by_label: List[Dict[str, ImageSpan]] = []
    metas = []
    last_span_by_label: Optional[Dict[str, ImageSpan]] = None
    for sample in samples:
        print(f"[scan] {sample.get('id', '<sample>')}")
        meta, packed, span_by_label, _, _ = scan_sample(
            torch, model, processor, sample, args, dtype, focus_labels, selected_layers
        )
        sample_results.append(packed)
        sample_spans_by_label.append(span_by_label)
        metas.append(meta)
        last_span_by_label = span_by_label

    assert last_span_by_label is not None
    aggregate = aggregate_samples(sample_results, focus_labels)
    target_tables = compute_target_tables(
        aggregate, focus_labels, last_span_by_label, box_map, args.entropy_threshold_rel
    )
    top_heads = rank_heads(
        aggregate, target_tables, focus_labels, args.rank_by, args.top_k, selected_layers,
        s_img_floor=float(args.s_img_floor),
    )

    per_image_json = []
    for label in focus_labels:
        entry = {
            "label": label,
            "path": last_span_by_label[label].path,
            "grid_thw": last_span_by_label[label].grid_thw,
            "s_img_mean_table": aggregate[label]["s_img"].tolist(),
            "entropy_mean_table": aggregate[label]["entropy"].tolist(),
        }
        if label in target_tables:
            entry["target_metrics"] = {name: value.tolist() for name, value in target_tables[label].items()}
        per_image_json.append(entry)

    attention_video_manifest: List[Dict] = []
    out_json = {
        "model_path": args.model_path,
        "sample_json": args.sample_json,
        "num_samples": len(samples),
        "num_images": 8,
        "focus_image_labels": list(focus_labels),
        "query_mode": args.query_mode,
        "query_text": args.query_text,
        "analysis_suffix": args.analysis_suffix,
        "rank_by": args.rank_by,
        "s_img_floor": float(args.s_img_floor),
        "skip_early_layers": int(args.skip_early_layers),
        "layers": selected_layers,
        "max_pixels": args.max_pixels,
        "samples": metas,
        "per_image": per_image_json,
        "top_heads": top_heads,
        "attention_videos": attention_video_manifest,
    }
    mean_s = np.mean([aggregate[label]["s_img"] for label in focus_labels], axis=0)
    mean_e = np.mean([aggregate[label]["entropy"] for label in focus_labels], axis=0)
    save_matrix_plot(mean_s, out_dir / "head_scan_Simg.png", "Mean image-attention mass")
    save_matrix_plot(mean_e, out_dir / "head_scan_entropy.png", "Mean spatial entropy", cmap="magma")

    if len(top_heads) >= 2:
        xs = [head["entropy"] for head in top_heads]
        ys = [head["s_img"] for head in top_heads]
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        ax.scatter(xs, ys)
        for head, x, y in zip(top_heads, xs, ys):
            ax.annotate(head["label"], (x, y), fontsize=7)
        ax.set_xlabel("spatial entropy")
        ax.set_ylabel("image-attention mass")
        ax.set_title("Top heads")
        fig.tight_layout()
        fig.savefig(out_dir / "head_scan_scatter.png")
        plt.close(fig)

    for head in top_heads:
        li = head["layer_table_index"]
        hi = head["head"]
        for label in focus_labels:
            span = last_span_by_label[label]
            box = lookup_box(box_map, label, span)
            title = f"{head['label']} {label} Simg={aggregate[label]['s_img'][li, hi]:.4f}"
            if label in target_tables:
                frac = target_tables[label]["target_fraction"][li, hi]
                iou = target_tables[label]["iou"][li, hi]
                title += f" target_frac={frac:.3f} IoU={iou:.3f}"
            out_path = out_dir / "top_heads" / f"{head['label']}_{label}.png"
            save_overlay(span.path, aggregate[label]["heatmaps"][li, hi], out_path, title, box)

    if args.save_attention_videos:
        attention_video_manifest.extend(
            save_top_head_attention_videos(
                sample_results,
                sample_spans_by_label,
                metas,
                top_heads,
                focus_labels,
                box_map,
                out_dir,
                args,
            )
        )

    with open(out_dir / "head_scan.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote {out_dir / 'head_scan.json'}")
    print("[top heads]")
    for head in top_heads:
        print(
            f"  {head['label']}: score={head['score']:.4f} "
            f"Simg={head['s_img']:.4f} entropy={head['entropy']:.4f}"
        )


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default= MODEL_PATH)
    parser.add_argument("--sample-json", help="Existing GRM sample.json produced by examples/inference.py")
    parser.add_argument("--out-dir", default= OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="Maximum samples to scan. Use 0 to scan all samples.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=SAMPLE_INDEX,
        help="Scan one sample by index. Use -1 to keep all samples before --max-samples is applied.",
    )

    parser.add_argument("--cam-high",default = CAM_HIGH)
    parser.add_argument("--cam-left",default = CAM_LEFT)
    parser.add_argument("--cam-right",default = CAM_RIGHT)
    parser.add_argument("--goal-image",default = GOAL_IMAGE)
    parser.add_argument("--task",default = TASK)
    parser.add_argument("--frame-interval", type=int, default=FRAME_INTERVAL)
    parser.add_argument("--eval-mode", default=EVAL_MODE, choices=["incremental", "forward", "backward"])

    parser.add_argument("--focus-images", default=FOCUS_IMAGES)
    parser.add_argument("--layers", default=LAYERS, help="Layer spec: all, last, 0,4-10, or comma-separated ranges")
    parser.add_argument(
        "--skip-early-layers",
        type=int,
        default=2,
        help=(
            "Exclude this many early layers from the scan. Kang et al. CVPR 2025 §4 "
            "exclude the first 2 layers because their attention reflects raw embedding "
            "similarity / positional bias rather than learned grounding. Set to 0 to "
            "include layer 0 (will likely surface a L0 head as a known artifact)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rank-by", default="localization", choices=["localization", "target_fraction", "target_density", "density_ratio", "iou"])
    parser.add_argument(
        "--save-attention-videos",
        default = SAVE_ATTENTION_VIDEOS,
        help="Write MP4 overlays over all scanned samples for each selected top head and focus image.",
    )
    parser.add_argument("--attention-video-top-k", type=int, default=ATTENTION_VIDEO_TOP_K, help="Limit how many ranked heads are rendered to video. 0 means all --top-k heads.")
    parser.add_argument(
        "--attention-video-fps", type=float, default=ATTENTION_VIDEO_FPS, help="FPS for attention videos. 0 infers source FPS / frame interval for raw videos, or 5 FPS for sample JSON.")
    parser.add_argument("--attention-video-alpha", type=float, default=ATTENTION_VIDEO_ALPHA, help="Heatmap overlay opacity for attention videos.")

    parser.add_argument(
        "--query-mode",
        default="score",
        choices=["score", "tail", "all_after_images"],
        help=(
            "How to pick the localization query token. 'score' (default) teacher-forces "
            "'<score>0%</score>' and uses the numeric token, mirroring Kang et al.'s use of "
            "the last input text token for an LVLM whose output is text. 'tail' uses the last "
            "N non-special prompt tokens (the instruction text, NOT the score token) and tends "
            "to surface early-layer heads that read the prompt rather than genuine grounding heads."
        ),
    )
    parser.add_argument("--tail-query-tokens", type=int, default=32)
    parser.add_argument("--query-text", help="Optional text to use as query positions; must appear after images to see image tokens")
    parser.add_argument(
        "--analysis-suffix",
        help="Optional extra text appended after the images before the assistant prompt, e.g. 'Focus on the target object.'",
    )
    parser.add_argument(
        "--s-img-floor",
        type=float,
        default=0.01,
        help=(
            "Kang Criterion-1 threshold tau: heads whose mean image-attention mass is below "
            "this value are excluded from localization ranking. Set to 0 to disable."
        ),
    )

    parser.add_argument("--target-box", help="Shared target bbox for focus images: x1,y1,x2,y2")
    parser.add_argument(
        "--target-box-json",
        help=(
            "JSON mapping image labels, basenames, or full image paths to [x1,y1,x2,y2]. "
            "Labels include after_cam_high, after_cam_left_wrist, after_cam_right_wrist."
        ),
    )

    parser.add_argument("--entropy-threshold-rel", type=float, default=0.6)
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=22500,
        help="Lower than inference.py by default to keep output_attentions memory manageable. Use 76800 to match inference.py.",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto", help="'auto' or 'none'")
    return parser


if __name__ == "__main__":
    run(make_arg_parser().parse_args())
