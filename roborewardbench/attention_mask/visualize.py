#!/usr/bin/env python3
"""Render endpoint attention heatmaps for baseline and target-bbox intervention.

The intervention and head selection are exactly the same as ``run_experiment``.
For each frozen evaluation example this module extracts the last-prompt query
attention of all selected heads, averages their high-camera spatial maps, and
writes baseline/candidate MP4s plus a side-by-side comparison.

RoboRewardBench supplies a video but the GRM benchmark protocol scores only its
decoded before/after endpoints.  Consequently these MP4s sequence endpoint
examples; they are not temporal attention tracking through the source videos.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .dataset import (
    examples_fingerprint,
    load_attention_examples,
    load_split_partition,
    sha256_file,
)
from .io import file_identity, model_identity, object_fingerprint, strict_dump
from .masking import (
    Head,
    ImageSpan,
    ROLE_LABELS,
    intervention_positions,
    merged_grid_shape,
    registered_mask_hooks,
    target_position_set,
)
from .modeling import (
    ensure_blank_goal,
    last_prompt_query_position,
    load_grm,
    model_dimensions,
    prepare_inputs,
)
from .run_experiment import (
    choose_head_groups,
    validate_ranking_linkage,
    validate_ranking_model,
)


CONDITIONS = ("baseline", "candidate_target")


def endpoint_labels(target_role: str) -> tuple[str, ...]:
    """Return the high-camera spans actually included in the intervention role."""

    labels = ROLE_LABELS.get(target_role)
    if labels is None:
        raise ValueError(f"Unknown target role: {target_role}")
    return tuple(label for label in labels if label.endswith("cam_high"))


def aggregate_head_grid(
    attentions: Sequence[Any],
    *,
    heads: Sequence[Head],
    query_position: int,
    span: ImageSpan,
    spatial_merge_size: int,
    target_positions: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Average selected-head attention over a single image-token span."""

    if not heads:
        raise ValueError("At least one head is required for visualization")
    vectors: list[np.ndarray] = []
    per_head_span_mass: list[float] = []
    per_head_bbox_mass: list[float] = []
    local_target = sorted(
        {
            int(position) - span.start
            for position in target_positions
            if span.start <= int(position) < span.end
        }
    )
    if not local_target:
        raise ValueError(f"{span.label}: no target tokens fall inside the visualized span")

    for head in heads:
        if not 0 <= head.layer < len(attentions):
            raise ValueError(f"Layer {head.layer} is absent from returned attentions")
        attention = attentions[head.layer]
        if attention.ndim != 4:
            raise RuntimeError(
                f"Expected attention [batch,heads,query,key], got {tuple(attention.shape)}"
            )
        if not 0 <= head.head < int(attention.shape[1]):
            raise ValueError(f"Head {head.head} is absent from layer {head.layer}")
        if not 0 <= query_position < int(attention.shape[2]):
            raise ValueError(
                f"Query position {query_position} exceeds layer {head.layer} "
                f"query length {int(attention.shape[2])}"
            )
        if span.end > int(attention.shape[3]):
            raise ValueError(
                f"{span.label} token end {span.end} exceeds attention key length "
                f"{int(attention.shape[3])}"
            )
        vector = (
            attention[0, head.head, query_position, span.start : span.end]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        if vector.shape != (span.token_count,):
            raise RuntimeError(f"{span.label}: unexpected attention vector shape {vector.shape}")
        vectors.append(vector)
        per_head_span_mass.append(float(vector.sum()))
        per_head_bbox_mass.append(float(vector[local_target].sum()))

    mean_vector = np.stack(vectors, axis=0).mean(axis=0)
    temporal, grid_h, grid_w = merged_grid_shape(span, spatial_merge_size)
    grid = mean_vector.reshape(temporal, grid_h, grid_w).mean(axis=0)
    span_mass = float(mean_vector.sum())
    bbox_mass = float(mean_vector[local_target].sum())
    return grid, {
        "num_aggregated_heads": len(heads),
        "query_position": int(query_position),
        "span_mass": span_mass,
        "bbox_mass": bbox_mass,
        "bbox_fraction_of_span": bbox_mass / max(span_mass, 1e-12),
        "mean_per_head_span_mass": float(np.mean(per_head_span_mass)),
        "mean_per_head_bbox_mass": float(np.mean(per_head_bbox_mass)),
        "grid_min": float(grid.min()),
        "grid_max": float(grid.max()),
        "grid_sum": float(grid.sum()),
        "target_token_count_in_span": len(local_target),
    }


def _forward_attentions(torch, model, inputs: Mapping[str, Any]):
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model returned no attentions; eager attention is required")
    return attentions


def _fit_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def render_heatmap_frame(
    image_path: str | Path,
    grid: np.ndarray,
    bbox: Sequence[float],
    *,
    vmax: float,
    alpha: float,
    title: str,
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    """Overlay a raw attention grid with shared baseline/candidate color scale."""

    import cv2

    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB"))
    height, width = image.shape[:2]
    safe_vmax = max(float(vmax), np.finfo(np.float32).eps)
    normalized = np.clip(np.asarray(grid, dtype=np.float32) / safe_vmax, 0.0, 1.0)
    # Nearest-neighbor rendering keeps the actual merged visual-token cells
    # visible and avoids cubic overshoot that can make a coarse cell appear
    # shifted outside the pixel-space bbox.
    dense = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_NEAREST)
    colored_bgr = cv2.applyColorMap(
        np.asarray(np.clip(dense * 255.0, 0, 255), dtype=np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    blend_weight = (float(alpha) * dense)[..., None]
    overlay = np.clip(
        image.astype(np.float32) * (1.0 - blend_weight)
        + colored.astype(np.float32) * blend_weight,
        0,
        255,
    ).astype(np.uint8)
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    cv2.rectangle(
        overlay,
        (max(0, x1), max(0, y1)),
        (min(width - 1, x2), min(height - 1, y2)),
        (0, 255, 0),
        thickness=max(1, round(min(width, height) / 160)),
    )

    header_height = 54
    fitted = _fit_rgb(
        overlay,
        int(canvas_width),
        int(canvas_height) - header_height,
    )
    canvas = np.zeros((int(canvas_height), int(canvas_width), 3), dtype=np.uint8)
    canvas[header_height:] = fitted
    cv2.putText(
        canvas,
        title,
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_video(path: str | Path, frames: Sequence[np.ndarray], *, fps: float) -> None:
    import cv2

    if not frames:
        raise ValueError(f"No frames were supplied for {path}")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {destination}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("Every video frame must have the same dimensions")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Video writer produced no data: {destination}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--head-ranking", required=True)
    parser.add_argument(
        "--external-fixed-ranking",
        action="store_true",
        help="Treat --head-ranking as a frozen head set produced outside this split.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-mode", default="manual_correct_ready")
    parser.add_argument("--partition", default="evaluation", choices=["evaluation"])
    parser.add_argument(
        "--target-role",
        default="both",
        choices=["before", "after", "both", "after_high"],
    )
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--swap-bias", type=float, default=4.0)
    parser.add_argument(
        "--intervention",
        default="boost_suppress",
        choices=["boost_suppress", "suppress_image"],
    )
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--save-png-frames", type=int, default=4)
    parser.add_argument("--canvas-width", type=int, default=960)
    parser.add_argument("--canvas-height", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--max-pixels", type=int, default=76800)
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument(
        "--allow-incomplete-ranking",
        action="store_true",
        help="Allow a one-example smoke ranking; never use its video for final claims.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not math.isfinite(args.swap_bias) or args.swap_bias < 0:
        raise ValueError("--swap-bias must be finite and non-negative")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.save_png_frames < 0:
        raise ValueError("--save-png-frames cannot be negative")
    if args.canvas_width <= 0 or args.canvas_height <= 54:
        raise ValueError("Canvas dimensions are too small")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    grounding_dir = Path(args.grounding_dir).expanduser().resolve()
    split_path = Path(args.split_manifest).expanduser().resolve()
    ranking_path = Path(args.head_ranking).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))

    evaluation_ids, split_data = load_split_partition(split_path, args.partition)
    expected_mode = str(split_data[args.partition].get("selection_mode"))
    if args.selection_mode != expected_mode:
        raise ValueError(
            f"selection_mode={args.selection_mode} disagrees with frozen split "
            f"selection_mode={expected_mode}"
        )
    validate_ranking_linkage(
        ranking,
        evaluation_ids=evaluation_ids,
        split_sha256=sha256_file(split_path),
        target_role=args.target_role,
        external_fixed_ranking=args.external_fixed_ranking,
        allow_incomplete_ranking=args.allow_incomplete_ranking,
    )
    examples_all = load_attention_examples(
        grounding_dir,
        selection_mode=args.selection_mode,
        example_ids=evaluation_ids,
    )
    if examples_fingerprint(examples_all) != split_data["evaluation"].get(
        "dataset_fingerprint"
    ):
        raise ValueError("Evaluation grounding changed after the split was frozen")
    examples = list(examples_all)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("No frozen evaluation examples were selected")

    signature_payload = {
        "schema_version": 1,
        "grounding_dir": str(grounding_dir),
        "split_manifest": file_identity(split_path),
        "head_ranking": file_identity(ranking_path),
        "head_set_mode": (
            "external_fixed" if args.external_fixed_ranking else "in_split_discovery"
        ),
        "model": model_identity(model_path),
        "code": {
            "visualize": file_identity(Path(__file__)),
            "masking": file_identity(Path(__file__).with_name("masking.py")),
            "modeling": file_identity(Path(__file__).with_name("modeling.py")),
            "run_experiment": file_identity(Path(__file__).with_name("run_experiment.py")),
        },
        "evaluation_dataset_fingerprint": examples_fingerprint(examples_all),
        "selected_example_ids": [example.example_id for example in examples],
        "target_role": args.target_role,
        "top_k": args.top_k,
        "swap_bias": args.swap_bias,
        "intervention": args.intervention,
        "decode_only": args.decode_only,
        "fps": args.fps,
        "alpha": args.alpha,
        "save_png_frames": args.save_png_frames,
        "canvas_width": args.canvas_width,
        "canvas_height": args.canvas_height,
        "dtype": args.dtype,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
    }
    run_signature = object_fingerprint(signature_payload)
    manifest_path = output_dir / "attention_video_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_paths = [
            Path(str(block["path"]))
            for block in (existing.get("videos") or {}).values()
            if isinstance(block, Mapping) and block.get("path")
        ]
        if (
            existing.get("run_signature") == run_signature
            and len(video_paths) == 3
            and all(path.is_file() and path.stat().st_size > 0 for path in video_paths)
        ):
            print(f"[attention-video] reusing current artifacts in {output_dir}")
            return

    blank_goal = ensure_blank_goal(output_dir / "blank_goal.png")
    print(f"[attention-video] loading {model_path}", flush=True)
    torch, model, processor, dtype = load_grm(
        model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        output_attentions=True,
    )
    model.config.use_cache = False
    num_layers, num_heads, spatial_merge_size = model_dimensions(model)
    current_model = model_identity(model_path)
    validate_ranking_model(
        ranking,
        current_model_identity=current_model,
        num_layers=num_layers,
        num_heads=num_heads,
        external_fixed_ranking=args.external_fixed_ranking,
    )
    candidate_heads, _ = choose_head_groups(
        ranking,
        args.top_k,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    labels = endpoint_labels(args.target_role)
    frames: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    comparison_frames: list[np.ndarray] = []
    saved_pngs: dict[str, list[str]] = {condition: [] for condition in CONDITIONS}
    per_example: list[dict[str, Any]] = []

    for example_index, example in enumerate(examples, 1):
        print(
            f"[attention-video] {example_index}/{len(examples)} {example.example_id}",
            flush=True,
        )
        item = example.model_item(blank_goal)
        if "reward" in item:
            raise AssertionError("Reward leaked into the visualization model item")
        inputs, spans = prepare_inputs(torch, model, processor, item, dtype)
        span_by_label = {span.label: span for span in spans}
        positions = target_position_set(
            spans,
            before_bbox=example.before_bbox,
            after_bbox=example.after_bbox,
            before_image_size=example.before_image_size,
            after_image_size=example.after_image_size,
            spatial_merge_size=spatial_merge_size,
            target_role=args.target_role,
        )
        query_position = last_prompt_query_position(inputs, spans, model.config)
        condition_grids: dict[str, dict[str, np.ndarray]] = {
            condition: {} for condition in CONDITIONS
        }
        condition_metrics: dict[str, dict[str, Any]] = {
            condition: {} for condition in CONDITIONS
        }

        for condition in CONDITIONS:
            if condition == "baseline":
                attentions = _forward_attentions(torch, model, inputs)
            else:
                suppress, boost = intervention_positions(args.intervention, positions)
                with registered_mask_hooks(
                    model,
                    heads=candidate_heads,
                    suppress_positions=suppress,
                    boost_positions=boost,
                    num_query_heads=num_heads,
                    swap_bias=args.swap_bias,
                    decode_only=args.decode_only,
                ):
                    attentions = _forward_attentions(torch, model, inputs)
            for label in labels:
                grid, metrics = aggregate_head_grid(
                    attentions,
                    heads=candidate_heads,
                    query_position=query_position,
                    span=span_by_label[label],
                    spatial_merge_size=spatial_merge_size,
                    target_positions=positions.per_span_target[label],
                )
                condition_grids[condition][label] = grid
                condition_metrics[condition][label] = metrics
            del attentions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        example_record: dict[str, Any] = {
            "example_id": example.example_id,
            "subset": example.subset,
            "task": example.task,
            "target_phrase": example.target_phrase,
            "grounding_fingerprint": example.grounding_fingerprint,
            "query_position": int(query_position),
            "prompt_length": int(inputs["input_ids"].shape[1]),
            "conditions": condition_metrics,
            "endpoints": {},
        }
        for label in labels:
            is_before = label.startswith("before_")
            image_path = example.before_path if is_before else example.after_path
            bbox = example.before_bbox if is_before else example.after_bbox
            joint_max = max(
                float(condition_grids[condition][label].max())
                for condition in CONDITIONS
            )
            endpoint_record = {
                "image_path": str(image_path),
                "bbox": list(bbox),
                "merged_token_grid_shape": list(
                    condition_grids["baseline"][label].shape
                ),
                "target_token_count": len(positions.per_span_target[label]),
                "shared_raw_attention_vmax": joint_max,
            }
            rendered: dict[str, np.ndarray] = {}
            endpoint_name = "before" if is_before else "after"
            for condition in CONDITIONS:
                metrics = condition_metrics[condition][label]
                title = (
                    f"{example_index:02d}/{len(examples):02d} | {endpoint_name} | "
                    f"{condition} | bbox_mass={metrics['bbox_mass']:.5f}"
                )
                frame = render_heatmap_frame(
                    image_path,
                    condition_grids[condition][label],
                    bbox,
                    vmax=joint_max,
                    alpha=args.alpha,
                    title=title,
                    canvas_width=args.canvas_width,
                    canvas_height=args.canvas_height,
                )
                frames[condition].append(frame)
                rendered[condition] = frame
                if len(saved_pngs[condition]) < args.save_png_frames:
                    png_path = (
                        output_dir
                        / "frames"
                        / f"{condition}_{len(saved_pngs[condition]):03d}.png"
                    )
                    png_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame).save(png_path)
                    saved_pngs[condition].append(str(png_path))
            comparison_frames.append(
                np.concatenate(
                    [rendered["baseline"], rendered["candidate_target"]],
                    axis=1,
                )
            )
            example_record["endpoints"][endpoint_name] = endpoint_record
        per_example.append(example_record)

    video_paths = {
        "baseline": output_dir / "baseline_attention.mp4",
        "candidate_target": output_dir / "candidate_target_attention.mp4",
        "comparison": output_dir / "baseline_vs_candidate_attention.mp4",
    }
    write_video(video_paths["baseline"], frames["baseline"], fps=args.fps)
    write_video(
        video_paths["candidate_target"],
        frames["candidate_target"],
        fps=args.fps,
    )
    write_video(video_paths["comparison"], comparison_frames, fps=args.fps)

    manifest = {
        **signature_payload,
        "run_signature": run_signature,
        "visualization_semantics": (
            "Each MP4 sequences frozen before/after high-camera endpoint frames across "
            "evaluation examples. It is not frame-by-frame tracking through source videos."
        ),
        "attention_semantics": (
            "Raw post-softmax attention from the last non-special prompt query is averaged "
            "over the same top-k heads used for intervention. Baseline and candidate use "
            "one shared color maximum per example/endpoint."
        ),
        "mask_semantics": (
            "candidate_target applies +bias to target bbox keys and -bias to other selected "
            "endpoint-image keys in selected heads; text and reference images remain zero. "
            "The intervention covers all role camera copies, while visualization renders "
            "the high-camera copy only."
        ),
        "decode_only_note": (
            "decode_only=True intentionally leaves this prefill-query heatmap unchanged; "
            "the experiment intervention then begins only during autoregressive decode."
            if args.decode_only
            else None
        ),
        "selected_heads": [
            {"layer": head.layer, "head": head.head} for head in candidate_heads
        ],
        "num_examples": len(examples),
        "full_evaluation_num_examples": len(examples_all),
        "complete_evaluation_partition": len(examples) == len(examples_all),
        "num_endpoint_frames_per_condition": len(frames["baseline"]),
        "videos": {
            name: {
                "path": str(path),
                "num_frames": (
                    len(comparison_frames)
                    if name == "comparison"
                    else len(frames[name])
                ),
                "fps": args.fps,
            }
            for name, path in video_paths.items()
        },
        "saved_png_frames": saved_pngs,
        "per_example": per_example,
    }
    strict_dump(manifest, manifest_path)
    print(f"[attention-video] wrote {manifest_path}", flush=True)
    for name, path in video_paths.items():
        print(f"[attention-video] wrote {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
