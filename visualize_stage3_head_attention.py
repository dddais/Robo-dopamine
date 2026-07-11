#!/usr/bin/env python3
"""Visualize one Stage-3 candidate head under baseline vs bbox steering.

This is a companion artifact for steer_grm_heads.py.  It selects one candidate
head from the stage-2 bbox-mass ranking, runs the same samples twice:

  1. baseline: no attention hook
  2. candidate_target: stage-3 candidate heads steered to the GroundingDINO bbox

For both conditions it extracts the selected head's attention from the chosen
query row(s) to the target image span, maps the vector back to the image-token
grid, overlays it on the target frame with the bbox, and writes MP4 videos.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_localization_heads_best import (  # noqa: E402
    IMAGE_LABELS,
    ImageSpan,
    _run_forward_for_query,
    build_prompt,
    frame_id_from_path,
    infer_image_spans,
    load_model_and_processor,
    move_inputs_to_device,
    render_overlay_frame,
    safe_filename,
    select_query_positions,
    vector_to_grid,
)
from grounding import GroundingBox, TaskGrounding  # noqa: E402
from steer_grm_heads import (  # noqa: E402
    HeadSpec,
    build_smoothed_bbox_sequence,
    bbox_to_token_positions,
    group_heads_by_layer,
    make_steering_hook,
    num_query_heads,
    parse_heads,
    register_layer_hooks,
    remove_handles,
    target_image_paths_for_samples,
)


@dataclass
class VisualContext:
    sample: dict
    inputs: dict
    span_by_label: Dict[str, ImageSpan]
    target_span: ImageSpan
    target_box: GroundingBox
    target_positions: List[int]
    other_positions: List[int]
    query_positions: List[int]
    query_desc: str


def select_samples(samples: List[dict], num_samples: Optional[int], strategy: str) -> List[dict]:
    if num_samples is None or num_samples <= 0 or num_samples >= len(samples):
        return list(samples)
    if strategy == "first":
        return list(samples[:num_samples])
    step = max(1, len(samples) // max(1, num_samples))
    return list(samples[::step][:num_samples])


def apply_task_override(samples: List[dict], override_task: Optional[str]) -> List[dict]:
    if override_task is None:
        return samples
    out: List[dict] = []
    for sample in samples:
        copied = dict(sample)
        copied["original_task"] = sample.get("task")
        copied["task"] = override_task
        out.append(copied)
    return out


def choose_visual_head(candidate_heads: Sequence[HeadSpec], args: argparse.Namespace) -> HeadSpec:
    if args.layer is not None or args.head is not None:
        if args.layer is None or args.head is None:
            raise ValueError("--layer and --head must be provided together")
        return HeadSpec(layer=int(args.layer), head=int(args.head), label=f"L{int(args.layer)}H{int(args.head)}")
    if not candidate_heads:
        raise ValueError("No candidate heads available")
    idx = max(0, min(int(args.head_index), len(candidate_heads) - 1))
    return candidate_heads[idx]


def build_visual_context(
    torch,
    model,
    processor,
    sample: dict,
    grounding: TaskGrounding,
    target_label: str,
    dtype,
    spatial_merge_size: int,
    args: argparse.Namespace,
    target_box: Optional[GroundingBox] = None,
    allow_single_frame_grounding: bool = True,
) -> VisualContext:
    image_paths = sample["image"]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    prompt = build_prompt(processor, sample["task"], analysis_suffix=None, score_suffix=False)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    spans = infer_image_spans(inputs, model.config, image_paths)
    span_by_label = {s.label: s for s in spans}
    target_span = span_by_label.get(target_label)
    if target_span is None:
        raise RuntimeError(f"Target label {target_label!r} not found in sample image spans")

    if target_box is None and allow_single_frame_grounding:
        target_box = grounding.ground_best(target_span.path, sample["task"])
    if target_box is None:
        raise RuntimeError(f"GroundingDINO found no box for {target_label} in {target_span.path}")
    target_positions = bbox_to_token_positions(
        target_span,
        target_box.bbox,
        spatial_merge_size,
        target_span.path,
    )
    if not target_positions:
        raise RuntimeError(f"Grounding bbox mapped to zero image tokens for {target_span.path}")
    target_set = set(target_positions)
    other_positions = [p for p in range(target_span.start, target_span.end) if p not in target_set]

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
        score_query_tokens="digits",
    )

    device = next(model.parameters()).device
    inputs = move_inputs_to_device(torch, inputs, device, dtype)
    return VisualContext(
        sample=sample,
        inputs=inputs,
        span_by_label=span_by_label,
        target_span=target_span,
        target_box=target_box,
        target_positions=target_positions,
        other_positions=other_positions,
        query_positions=query_positions,
        query_desc=query_desc,
    )


def run_condition_attentions(
    torch,
    model,
    processor,
    ctx: VisualContext,
    condition: str,
    candidate_heads: Sequence[HeadSpec],
    n_qheads: int,
    swap_bias: float,
    device: str,
    args: argparse.Namespace,
) -> Tuple:
    handles = []
    if condition == "candidate_target":
        hook_by_layer: Dict[int, object] = {}
        for layer_idx, hlist in group_heads_by_layer(candidate_heads).items():
            hook_by_layer[layer_idx] = make_steering_hook(
                head_indices=hlist,
                target_positions=ctx.target_positions,
                other_positions=ctx.other_positions,
                n_query_heads=n_qheads,
                device=device,
                swap_bias=swap_bias,
            )
        handles = register_layer_hooks(model, hook_by_layer)

    try:
        attentions, query_positions, query_desc = _run_forward_for_query(
            torch,
            model,
            ctx.inputs,
            args.query_mode,
            args.generate_max_new_tokens,
            ctx.query_positions,
            processor.tokenizer,
            args.score_query_tokens,
            args.generate_query_stage,
        )
        if attentions is None:
            raise RuntimeError("Model returned no attentions; use eager attention implementation")
        ctx.query_positions = list(query_positions)
        ctx.query_desc = query_desc
        return attentions
    finally:
        if handles:
            remove_handles(handles)


def extract_head_grid(
    attentions,
    ctx: VisualContext,
    visual_head: HeadSpec,
    spatial_merge_size: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    layer = int(visual_head.layer)
    head = int(visual_head.head)
    if layer < 0 or layer >= len(attentions):
        raise RuntimeError(f"Layer {layer} is outside returned attentions ({len(attentions)} layers)")
    attn = attentions[layer][0, head].detach().float().cpu().numpy()  # [Q, K]
    q_idx = [int(q) for q in ctx.query_positions]
    span = ctx.target_span
    vec = attn[q_idx, span.start:span.end].mean(axis=0)
    grid = vector_to_grid(vec, span.grid_thw, spatial_merge_size)
    local_target = [p - span.start for p in ctx.target_positions if span.start <= p < span.end]
    bbox_mass = float(vec[local_target].sum()) if local_target else 0.0
    span_mass = float(vec.sum())
    return grid, {
        "span_mass": span_mass,
        "bbox_mass": bbox_mass,
        "bbox_fraction_of_span": bbox_mass / max(span_mass, 1e-12),
        "grid_min": float(np.nanmin(grid)),
        "grid_max": float(np.nanmax(grid)),
        "grid_sum": float(np.nansum(grid)),
    }


def write_video(
    cv2,
    out_path: Path,
    frames: Sequence[np.ndarray],
    fps: float,
) -> None:
    if not frames:
        raise RuntimeError(f"No frames to write for {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def make_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage-3 baseline vs candidate-target attention video for one head")
    ap.add_argument("--model-path", default="./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview")
    ap.add_argument("--head-ranking-json", required=True)
    ap.add_argument("--ranking", default=None, choices=["mean", "max", "median", "selection_frequency"])
    ap.add_argument("--sample-json", required=True)
    ap.add_argument("--override-task", default=None,
                    help="Replace sample['task'] at inference time. The visualized prompt "
                         "and GroundingDINO target bbox both use this task.")
    ap.add_argument("--target-label", default="after_cam_high", choices=IMAGE_LABELS)
    ap.add_argument("--top-k", type=int, default=8, help="Candidate heads used for candidate_target steering")
    ap.add_argument("--head-index", type=int, default=0, help="Which candidate head to visualize")
    ap.add_argument("--layer", type=int, default=None, help="Override visualized head layer")
    ap.add_argument("--head", type=int, default=None, help="Override visualized head index")
    ap.add_argument("--grounding-model", default="../model/grounding-dino-base")
    ap.add_argument("--grounding-box-threshold", type=float, default=0.12)
    ap.add_argument("--swap-bias", type=float, default=6.0)
    ap.add_argument("--query-mode", default="last_prompt", choices=["last_prompt", "tail", "all_after_images", "generate"])
    ap.add_argument("--tail-query-tokens", type=int, default=5)
    ap.add_argument("--query-text", default=None)
    ap.add_argument("--score-query-tokens", default="digits")
    ap.add_argument("--generate-query-stage", default="predict_token", choices=["predict_token", "score_token"])
    ap.add_argument("--generate-max-new-tokens", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--sample-strategy", default="even", choices=["even", "first"])
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--save-png-frames", type=int, default=2, help="Save first N rendered frames per condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="none", help="'none' or a transformers device_map value such as 'auto'")
    ap.add_argument("--max-pixels", type=int, default=76800)
    ap.add_argument("--min-pixels", type=int, default=12544)
    ap.add_argument("--output-dir", required=True)
    return ap


def main() -> None:
    args = make_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    samples_all = json.loads(Path(args.sample_json).read_text())
    original_task = samples_all[0].get("task") if samples_all else None
    samples_all = apply_task_override(samples_all, args.override_task)
    samples = select_samples(samples_all, args.num_samples, args.sample_strategy)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage3-video] loading GRM from {args.model_path}")
    torch, model, processor, dtype = load_model_and_processor(args)
    model.config.output_attentions = True
    model.config.use_cache = False
    n_qheads = num_query_heads(model)
    spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[stage3-video] n_samples={len(samples)} n_qheads={n_qheads} spatial_merge_size={spatial_merge_size}")

    candidate_heads = parse_heads(Path(args.head_ranking_json), args.top_k, ranking=args.ranking)
    visual_head = choose_visual_head(candidate_heads, args)
    head_label = f"L{visual_head.layer}H{visual_head.head}"
    print(f"[stage3-video] steering heads: {[(h.layer, h.head) for h in candidate_heads]}")
    print(f"[stage3-video] visualized head: {head_label}")

    print(f"[stage3-video] loading GroundingDINO from {args.grounding_model}")
    grounding = TaskGrounding(
        model_path=args.grounding_model,
        device=device,
        box_threshold=args.grounding_box_threshold,
    )
    bbox_sequence = build_smoothed_bbox_sequence(
        grounding,
        samples,
        args.target_label,
        write_json=out_dir / "bbox_sequence.json",
    )

    import cv2

    frames_by_condition: Dict[str, List[np.ndarray]] = {"baseline": [], "candidate_target": []}
    per_sample: List[dict] = []
    for si, sample in enumerate(samples):
        print(f"[stage3-video] sample {si + 1}/{len(samples)}: {sample.get('id', '')}")
        try:
            ctx = build_visual_context(
                torch,
                model,
                processor,
                sample,
                grounding,
                args.target_label,
                dtype,
                spatial_merge_size,
                args,
                target_box=bbox_sequence[si] if si < len(bbox_sequence) else None,
                allow_single_frame_grounding=False,
            )
        except Exception as exc:
            print(f"    [skip] {exc}")
            per_sample.append({"sample_id": sample.get("id"), "skipped": True, "reason": str(exc)})
            continue

        sample_rec = {
            "sample_id": sample.get("id"),
            "task": sample.get("task"),
            "target_label": args.target_label,
            "target_image": ctx.target_span.path,
            "frame_id": frame_id_from_path(ctx.target_span.path, si),
            "query_desc": ctx.query_desc,
            "query_positions": ctx.query_positions,
            "bbox": ctx.target_box.bbox,
            "bbox_label": ctx.target_box.label,
            "bbox_score": ctx.target_box.score,
            "bbox_query": ctx.target_box.query,
            "bbox_quality": ctx.target_box.quality,
            "bbox_source": ctx.target_box.source,
            "n_target_tokens": len(ctx.target_positions),
            "conditions": {},
        }

        for condition in ("baseline", "candidate_target"):
            attentions = run_condition_attentions(
                torch,
                model,
                processor,
                ctx,
                condition,
                candidate_heads,
                n_qheads,
                args.swap_bias,
                device,
                args,
            )
            grid, metrics = extract_head_grid(attentions, ctx, visual_head, spatial_merge_size)
            metrics["query_desc"] = ctx.query_desc
            metrics["query_positions"] = [int(q) for q in ctx.query_positions]
            del attentions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            title = (
                f"{condition} {head_label} {args.target_label} "
                f"frame={sample_rec['frame_id']} bbox={metrics['bbox_mass']:.4f}"
            )
            frame = render_overlay_frame(
                ctx.target_span.path,
                grid,
                ctx.target_box.bbox,
                alpha=float(args.alpha),
                title=title,
            )
            frames_by_condition[condition].append(frame)
            sample_rec["conditions"][condition] = metrics
            print(
                f"    {condition}: bbox_mass={metrics['bbox_mass']:.6f} "
                f"span_mass={metrics['span_mass']:.6f}"
            )

        per_sample.append(sample_rec)

    videos = {}
    for condition, frames in frames_by_condition.items():
        out_path = out_dir / f"{safe_filename(condition)}_{head_label}_{safe_filename(args.target_label)}.mp4"
        write_video(cv2, out_path, frames, args.fps)
        videos[condition] = {
            "path": str(out_path),
            "num_frames": len(frames),
            "fps": float(args.fps),
        }
        for idx, frame in enumerate(frames[: max(0, int(args.save_png_frames))]):
            png_path = out_dir / "frames" / f"{safe_filename(condition)}_{head_label}_{idx:03d}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(png_path)

    manifest = {
        "args": vars(args),
        "original_task": original_task,
        "inference_task": samples[0].get("task") if samples else None,
        "selected_head": {"layer": visual_head.layer, "head": visual_head.head, "label": head_label},
        "steering_heads": [{"layer": h.layer, "head": h.head, "label": h.label} for h in candidate_heads],
        "conditions": {
            "baseline": "no hook",
            "candidate_target": "top-k candidate heads receive +swap_bias on bbox tokens and -swap_bias on other target-view tokens",
        },
        "videos": videos,
        "n_input_samples": len(samples),
        "n_rendered_frames": {k: len(v) for k, v in frames_by_condition.items()},
        "per_sample": per_sample,
    }
    manifest_path = out_dir / "attention_video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[stage3-video] wrote {manifest_path}")
    for condition, info in videos.items():
        print(f"[stage3-video] wrote {condition}: {info['path']} ({info['num_frames']} frames)")


if __name__ == "__main__":
    main()
