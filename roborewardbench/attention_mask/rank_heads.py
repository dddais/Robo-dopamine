#!/usr/bin/env python3
"""Discover GRM heads that attend to frozen target bboxes.

This stage must use only the ``discovery`` partition.  The output records all
discovery IDs and their frozen dataset fingerprint; ``run_experiment.py`` then
refuses to evaluate any overlapping example.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import (
    examples_fingerprint,
    load_attention_examples,
    load_split_partition,
    sha256_file,
)
from .io import file_identity, model_identity, strict_dump
from .masking import target_position_set
from .modeling import (
    ensure_blank_goal,
    last_prompt_query_position,
    load_grm,
    model_dimensions,
    prepare_inputs,
)


RANKING_NAMES = ("mean", "median", "max", "selection_frequency")
SCORE_MODES = ("excess_mass", "raw_mass")
OUTPUT_SCHEMA_VERSION = 1


def aggregate_rankings(
    mass_arrays: Sequence[np.ndarray],
    *,
    top_fraction_per_sample: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    if not mass_arrays:
        return {name: [] for name in RANKING_NAMES}
    shape = mass_arrays[0].shape
    if len(shape) != 2 or any(array.shape != shape for array in mass_arrays):
        raise ValueError("Every per-sample mass array must have shape [layers, heads]")
    stack = np.stack(mass_arrays, axis=0).astype(np.float64)
    if not np.isfinite(stack).all():
        raise ValueError("Attention mass contains NaN or Infinity")
    num_layers, num_heads = shape
    n_total = num_layers * num_heads
    n_selected = max(1, int(round(n_total * float(top_fraction_per_sample))))
    hits = np.zeros(shape, dtype=np.float64)
    for array in stack:
        order = np.argsort(array.reshape(-1), kind="stable")
        selected = order[-n_selected:]
        hits.reshape(-1)[selected] += 1.0

    values = {
        "mean": stack.mean(axis=0),
        "median": np.median(stack, axis=0),
        "max": stack.max(axis=0),
        "selection_frequency": hits / len(stack),
    }
    rankings: dict[str, list[dict[str, Any]]] = {}
    for name, array in values.items():
        rows = [
            {
                "layer": layer,
                "head": head,
                "score": float(array[layer, head]),
            }
            for layer in range(num_layers)
            for head in range(num_heads)
        ]
        rows.sort(key=lambda row: (-row["score"], row["layer"], row["head"]))
        rankings[name] = rows
    return rankings


def bbox_ranking_score(
    target_mass: np.ndarray,
    role_image_mass: np.ndarray,
    *,
    target_token_count: int,
    role_image_token_count: int,
    score_mode: str,
) -> np.ndarray:
    """Return a per-head bbox score with an explicit area-confound correction.

    Raw bbox mass is proportional to bbox area even for spatially uniform
    attention.  ``excess_mass`` subtracts that uniform-within-selected-images
    expectation.  A full-image bbox therefore contributes exactly zero rather
    than dominating discovery as a generic image-attention sample.
    """

    if score_mode == "raw_mass":
        return np.asarray(target_mass, dtype=np.float64)
    if score_mode != "excess_mass":
        raise ValueError(f"Unknown score_mode={score_mode!r}; expected one of {SCORE_MODES}")
    if role_image_token_count <= 0:
        raise ValueError("role_image_token_count must be positive")
    if not 0 < target_token_count <= role_image_token_count:
        raise ValueError("target_token_count must be in 1..role_image_token_count")
    area_fraction = target_token_count / role_image_token_count
    return (
        np.asarray(target_mass, dtype=np.float64)
        - area_fraction * np.asarray(role_image_mass, dtype=np.float64)
    )


def collect_head_mass(
    torch,
    model,
    processor,
    dtype,
    example,
    blank_goal: Path,
    *,
    target_role: str,
    spatial_merge_size: int,
    score_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    item = example.model_item(blank_goal)
    inputs, spans = prepare_inputs(torch, model, processor, item, dtype)
    positions = target_position_set(
        spans,
        before_bbox=example.before_bbox,
        after_bbox=example.after_bbox,
        before_image_size=example.before_image_size,
        after_image_size=example.after_image_size,
        spatial_merge_size=spatial_merge_size,
        target_role=target_role,
    )
    query_position = last_prompt_query_position(inputs, spans, model.config)
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model returned no attentions; eager attention is required")
    target_indices = list(positions.target)
    role_image_indices = list(positions.target) + list(positions.other_image)
    raw_target_mass = np.zeros(
        (len(attentions), int(attentions[0].shape[1])), dtype=np.float64
    )
    image_mass = np.zeros_like(raw_target_mass)
    for layer, attention in enumerate(attentions):
        # One post-image prompt query row, matching the existing
        # run_targetbbox_success_experiments.py default.
        row = attention[0, :, query_position, :].detach().float().cpu().numpy()
        raw_target_mass[layer] = row[:, target_indices].sum(axis=-1)
        image_mass[layer] = row[:, role_image_indices].sum(axis=-1)
    mass = bbox_ranking_score(
        raw_target_mass,
        image_mass,
        target_token_count=len(target_indices),
        role_image_token_count=len(role_image_indices),
        score_mode=score_mode,
    )
    del attentions, outputs
    return mass, {
        "example_id": example.example_id,
        "subset": example.subset,
        "grounding_fingerprint": example.grounding_fingerprint,
        "query_position": int(query_position),
        "prompt_length": int(inputs["input_ids"].shape[1]),
        "target_token_count": len(target_indices),
        "role_image_token_count": len(role_image_indices),
        "bbox_token_fraction": len(target_indices) / len(role_image_indices),
        "score_mode": score_mode,
        "max_target_mass": float(mass.max()),
        "mean_target_mass": float(mass.mean()),
        "max_raw_target_mass": float(raw_target_mass.max()),
        "mean_raw_target_mass": float(raw_target_mass.mean()),
        "max_role_image_mass": float(image_mass.max()),
        "mass": mass.tolist(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-mode", default="manual_correct")
    parser.add_argument("--partition", default="discovery", choices=["discovery"])
    parser.add_argument(
        "--target-role",
        default="both",
        choices=["before", "after", "both", "after_high"],
    )
    parser.add_argument("--default-ranking", default="mean", choices=RANKING_NAMES)
    parser.add_argument("--score-mode", default="excess_mass", choices=SCORE_MODES)
    parser.add_argument("--skip-early-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--max-pixels", type=int, default=76800)
    parser.add_argument("--min-pixels", type=int, default=12544)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    split_ids, split_data = load_split_partition(args.split_manifest, args.partition)
    expected_mode = str(split_data[args.partition].get("selection_mode"))
    if args.selection_mode != expected_mode:
        raise ValueError(
            f"selection_mode={args.selection_mode} disagrees with split manifest "
            f"{args.partition}.selection_mode={expected_mode}"
        )
    examples_all = load_attention_examples(
        args.grounding_dir,
        selection_mode=args.selection_mode,
        example_ids=split_ids,
    )
    current_fingerprint = examples_fingerprint(examples_all)
    expected_fingerprint = split_data["discovery"].get("dataset_fingerprint")
    if current_fingerprint != expected_fingerprint:
        raise ValueError("Discovery grounding records changed after the split was frozen")
    examples = list(examples_all)
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max-examples must be positive")
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("The discovery partition is empty")

    output_path = Path(args.output).expanduser().resolve()
    blank_goal = ensure_blank_goal(output_path.parent / "blank_goal.png")
    print(f"[rank] loading {args.model_path}", flush=True)
    torch, model, processor, dtype = load_grm(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        output_attentions=True,
    )
    num_layers, num_heads, spatial_merge = model_dimensions(model)
    if not 0 <= args.skip_early_layers < num_layers:
        raise ValueError("--skip-early-layers is outside the model layer range")

    per_sample: list[dict[str, Any]] = []
    mass_arrays: list[np.ndarray] = []
    for index, example in enumerate(examples, 1):
        print(f"[rank] {index}/{len(examples)} {example.example_id}", flush=True)
        mass, record = collect_head_mass(
            torch,
            model,
            processor,
            dtype,
            example,
            blank_goal,
            target_role=args.target_role,
            spatial_merge_size=spatial_merge,
            score_mode=args.score_mode,
        )
        mass_arrays.append(mass)
        per_sample.append(record)

    rankings = aggregate_rankings(mass_arrays)
    filtered = [
        row
        for row in rankings[args.default_ranking]
        if int(row["layer"]) >= args.skip_early_layers
    ]
    selected_ids = [example.example_id for example in examples]
    result: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": f"last_prompt_bbox_{args.score_mode}",
        "interpretation": (
            "Heads are ranked on the discovery partition only. excess_mass is "
            "target bbox attention minus the uniform-within-selected-images "
            "expectation, removing bbox-area and full-frame confounding."
        ),
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output"}
        },
        "model": model_identity(args.model_path),
        "code": {
            "rank_heads": file_identity(Path(__file__)),
            "dataset": file_identity(Path(__file__).with_name("dataset.py")),
            "masking": file_identity(Path(__file__).with_name("masking.py")),
            "modeling": file_identity(Path(__file__).with_name("modeling.py")),
        },
        "num_layers": num_layers,
        "num_heads": num_heads,
        "spatial_merge_size": spatial_merge,
        "target_role": args.target_role,
        "default_ranking": args.default_ranking,
        "skip_early_layers": args.skip_early_layers,
        "top_heads": filtered[: max(1, int(args.top_k))],
        "rankings": rankings,
        "n_discovery": len(examples),
        "discovery_ids": selected_ids,
        "discovery_dataset_fingerprint": examples_fingerprint(examples),
        "full_discovery_dataset_fingerprint": current_fingerprint,
        "split_manifest": {
            "path": str(Path(args.split_manifest).expanduser().resolve()),
            "sha256": sha256_file(args.split_manifest),
            "strategy": split_data.get("strategy"),
            "partition_fingerprint": split_data["discovery"].get("dataset_fingerprint"),
        },
        "per_sample": per_sample,
        "elapsed_seconds": time.monotonic() - started,
    }
    # A capped smoke run must never masquerade as the complete discovery split.
    result["complete_discovery_partition"] = set(selected_ids) == set(split_ids)
    strict_dump(result, output_path)
    print(
        f"[rank] wrote {output_path}; n={len(examples)}, "
        f"top-{min(args.top_k, len(filtered))} from {args.default_ranking}",
        flush=True,
    )
    print(
        json.dumps(result["top_heads"][: min(10, len(result["top_heads"]))], indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
