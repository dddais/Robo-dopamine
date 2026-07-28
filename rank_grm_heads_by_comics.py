#!/usr/bin/env python3
"""Discover GRM gaze heads with the gaze-heads six-panel comic protocol.

For every comic strip, ask one question per panel and collect the raw
post-softmax attention mass from the final prompt token to each panel's image
tokens.  The gaze score of a language-model head is the diagonal mean of the
queried-panel x attended-panel matrix, averaged over complete strips.

The output intentionally follows ``rank_heads_by_bbox.py``'s
``top_heads``/``rankings`` schema so the existing GRM curve and visualization
scripts can consume a comic ranking without any steering-specific changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from tqdm.auto import tqdm

from scan_localization_heads_best import (
    ImageSpan,
    infer_image_spans,
    load_model_and_processor,
    move_inputs_to_device,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GAZE_HEADS_ROOT = REPO_ROOT.parent / "gaze-heads"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Discover GRM gaze heads from six-panel comics")
    ap.add_argument(
        "--model-path",
        default="./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview",
    )
    ap.add_argument(
        "--gaze-heads-root",
        default=str(DEFAULT_GAZE_HEADS_ROOT),
        help="Checkout containing gaze_heads/data.py, gaze.py, and regions.py.",
    )
    ap.add_argument(
        "--comics-root",
        default=str(DEFAULT_GAZE_HEADS_ROOT / "data" / "comics"),
        help="comicN/p1.png..pN.png folders, or the raw COMICS root with --use-raw.",
    )
    ap.add_argument(
        "--use-raw",
        action="store_true",
        help="Sample consecutive panel windows from the raw COMICS corpus.",
    )
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--n-panels", type=int, default=6)
    ap.add_argument("--target-height", type=int, default=256)
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-early-layers", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    ap.add_argument(
        "--device-map",
        default="none",
        help="'none' loads the model on the current CUDA device; otherwise pass a transformers device_map.",
    )
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional processor cap. Unset preserves the model/gaze-heads image resolution.",
    )
    ap.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional processor floor. Unset preserves the model/gaze-heads image resolution.",
    )
    ap.add_argument(
        "--output",
        default="results/attention/grm_comic_gaze/rank_comics_last_prompt_comic_gaze/head_ranking.json",
    )
    return ap


def _load_gaze_heads_helpers(root: Path):
    root = root.expanduser().resolve(strict=False)
    required = [
        root / "gaze_heads" / "data.py",
        root / "gaze_heads" / "gaze.py",
        root / "gaze_heads" / "regions.py",
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"not a complete gaze-heads checkout: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from gaze_heads.common import seed_everything
    from gaze_heads.data import build_random_raw_strip, build_strip, list_comic_dirs
    from gaze_heads.gaze import panel_query_prompt
    from gaze_heads.regions import assign_panels_to_tokens, region_positions_from_ids

    return (
        seed_everything,
        build_random_raw_strip,
        build_strip,
        list_comic_dirs,
        panel_query_prompt,
        assign_panels_to_tokens,
        region_positions_from_ids,
    )


def _ranking_rows(diagonal: np.ndarray, off_diagonal: np.ndarray) -> list[dict]:
    if diagonal.shape != off_diagonal.shape:
        raise ValueError(
            f"diagonal/off-diagonal shape mismatch: {diagonal.shape} vs {off_diagonal.shape}"
        )
    rows = []
    n_layers, n_heads = diagonal.shape
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            diag = float(diagonal[layer_idx, head_idx])
            off = float(off_diagonal[layer_idx, head_idx])
            rows.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "score": diag,
                    "off_diagonal_mean": off,
                    "diagonal_minus_off_diagonal": diag - off,
                }
            )
    # Protocol-faithful primary ranking: raw diagonal mass, not selectivity.
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def _span_record(
    span: ImageSpan,
    *,
    merged_grid_shape: Sequence[int],
    panel_column_ranges: Sequence[Sequence[int]],
) -> dict:
    return {
        "label": span.label,
        "path": span.path,
        "start": int(span.start),
        "end": int(span.end),
        "n_image_tokens": int(span.end - span.start),
        "grid_thw": [int(v) for v in span.grid_thw],
        "merged_grid_shape": [int(v) for v in merged_grid_shape],
        "panel_column_ranges": [
            [int(col_start), int(col_end)]
            for col_start, col_end in panel_column_ranges
        ],
    }


def _prepare_comic_inputs(processor: Any, strip_image: Any, query: str):
    """Use the same single-image user message as gaze-heads discovery."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": strip_image},
                {"type": "text", "text": query},
            ],
        }
    ]
    return processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def _collect_panel_masses(
    torch,
    model,
    model_inputs: dict,
    query_position: int,
    region_positions: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return raw region attention with shape [layers, heads, panels]."""
    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("model returned no attentions; eager attention is required")

    n_layers = len(attentions)
    n_heads = int(attentions[0].shape[1])
    masses = np.zeros((n_layers, n_heads, len(region_positions)), dtype=np.float64)
    for layer_idx, attention in enumerate(attentions):
        # [1, H, query_len, key_len] -> [H, key_len]
        row = attention[0, :, query_position, :].detach().float().cpu().numpy()
        for region_idx, positions in enumerate(region_positions):
            masses[layer_idx, :, region_idx] = row[:, list(positions)].sum(axis=-1)
    return masses


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.n_panels < 2:
        raise ValueError("--n-panels must be at least 2")
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")

    gaze_root = Path(args.gaze_heads_root)
    comics_root = Path(args.comics_root).expanduser().resolve(strict=False)
    (
        seed_everything,
        build_random_raw_strip,
        build_strip,
        list_comic_dirs,
        panel_query_prompt,
        assign_panels_to_tokens,
        region_positions_from_ids,
    ) = _load_gaze_heads_helpers(gaze_root)
    seed_everything(args.seed)

    if args.use_raw:
        rng = np.random.RandomState(args.seed)

        def strip_source() -> Iterator:
            for _ in range(args.n_samples):
                try:
                    yield build_random_raw_strip(
                        raw_panel_root=comics_root,
                        rng=rng,
                        n_panels=args.n_panels,
                        target_height=args.target_height,
                        gap=args.gap,
                    )
                except (RuntimeError, FileNotFoundError) as exc:
                    print(f"[comic-gaze] raw strip sampling failed: {exc}", flush=True)
                    return

        n_planned = args.n_samples
    else:
        comic_dirs = list_comic_dirs(comics_root, n_panels=args.n_panels)
        if not comic_dirs:
            raise FileNotFoundError(
                f"No valid {args.n_panels}-panel comics under {comics_root}. "
                f"Run {gaze_root / 'download_data.py'} or pass --comics-root."
            )
        comic_dirs = comic_dirs[: args.n_samples]

        def strip_source() -> Iterator:
            for comic_dir in comic_dirs:
                yield build_strip(
                    panel_dir=comic_dir,
                    n_panels=args.n_panels,
                    target_height=args.target_height,
                    gap=args.gap,
                )

        n_planned = len(comic_dirs)

    print(f"[comic-gaze] loading GRM from {args.model_path}", flush=True)
    torch, model, processor, dtype = load_model_and_processor(args)
    text_config = getattr(model.config, "text_config", model.config)
    n_layers = int(getattr(text_config, "num_hidden_layers"))
    n_heads = int(getattr(text_config, "num_attention_heads"))
    spatial_merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    device = next(model.parameters()).device
    print(
        f"[comic-gaze] layers={n_layers} heads={n_heads} spatial_merge={spatial_merge}",
        flush=True,
    )

    gaze_sum = np.zeros(
        (args.n_panels, n_layers, n_heads, args.n_panels),
        dtype=np.float64,
    )
    valid_names: list[str] = []
    skipped: list[dict] = []
    sample_audit: list[dict] = []

    for strip in tqdm(strip_source(), total=n_planned, desc="GRM comic-gaze discovery"):
        per_query: list[np.ndarray] = []
        per_query_audit: list[dict] = []
        sample_error = None

        for panel_index in range(1, args.n_panels + 1):
            query = panel_query_prompt(panel_index, n_panels=args.n_panels)
            try:
                encoded = _prepare_comic_inputs(processor, strip.strip, query)
                spans = infer_image_spans(
                    encoded,
                    model.config,
                    [f"comic:{strip.name}"],
                )
                if len(spans) != 1:
                    raise RuntimeError(f"expected one comic image span, got {len(spans)}")
                span = spans[0]

                region_ids, merged_shape, panel_ranges = assign_panels_to_tokens(
                    image_grid_thw=encoded["image_grid_thw"],
                    panel_widths=strip.panel_widths,
                    spatial_merge=spatial_merge,
                )
                expected_tokens = int(span.end - span.start)
                if int(region_ids.shape[0]) != expected_tokens:
                    raise RuntimeError(
                        f"panel mapping has {region_ids.shape[0]} tokens for "
                        f"an image span of {expected_tokens}"
                    )
                region_map = region_positions_from_ids(
                    span.start,
                    region_ids,
                    args.n_panels,
                )
                region_positions = [region_map[idx] for idx in range(args.n_panels)]
                if any(not positions for positions in region_positions):
                    raise RuntimeError(
                        "one or more panels received no image tokens: "
                        f"{[len(positions) for positions in region_positions]}"
                    )
                mapped = set().union(*(set(positions) for positions in region_positions))
                expected = set(range(span.start, span.end))
                if mapped != expected:
                    raise RuntimeError(
                        f"panel mapping covered {len(mapped)}/{len(expected)} image tokens"
                    )

                query_position = int(encoded["input_ids"].shape[-1] - 1)
                model_inputs = move_inputs_to_device(
                    torch,
                    encoded,
                    device,
                    dtype,
                )
                mass = _collect_panel_masses(
                    torch,
                    model,
                    model_inputs,
                    query_position,
                    region_positions,
                )
                if mass.shape != (n_layers, n_heads, args.n_panels):
                    raise RuntimeError(
                        f"unexpected attention mass shape {mass.shape}; "
                        f"expected {(n_layers, n_heads, args.n_panels)}"
                    )
                per_query.append(mass)
                per_query_audit.append(
                    {
                        "queried_panel": panel_index,
                        "query": query,
                        "query_positions": [query_position],
                        "query_detail": "last_prompt",
                        "n_panel_tokens": [
                            len(positions) for positions in region_positions
                        ],
                        "image_span": _span_record(
                            span,
                            merged_grid_shape=merged_shape,
                            panel_column_ranges=panel_ranges,
                        ),
                    }
                )
            except Exception as exc:
                sample_error = f"panel {panel_index}: {exc}"
                break

        # Keep only complete queried-panel x attended-panel matrices.
        if sample_error is not None or len(per_query) != args.n_panels:
            print(f"[comic-gaze] skip {strip.name}: {sample_error}", flush=True)
            skipped.append({"name": strip.name, "error": sample_error})
            continue

        gaze_sum += np.stack(per_query, axis=0)
        valid_names.append(strip.name)
        sample_audit.append(
            {
                "name": strip.name,
                "panel_paths": [str(path) for path in strip.panel_paths],
                "panel_widths": [int(width) for width in strip.panel_widths],
                "strip_size": [int(value) for value in strip.strip.size],
                "queries": per_query_audit,
            }
        )

    if not valid_names:
        raise RuntimeError("No complete comic strips were processed successfully")

    mean_matrix = gaze_sum / float(len(valid_names))
    diagonal = np.zeros((n_layers, n_heads), dtype=np.float64)
    off_diagonal = np.zeros((n_layers, n_heads), dtype=np.float64)
    for query_idx in range(args.n_panels):
        diagonal += mean_matrix[query_idx, :, :, query_idx]
        off_cols = [idx for idx in range(args.n_panels) if idx != query_idx]
        off_diagonal += np.take(
            mean_matrix[query_idx],
            off_cols,
            axis=-1,
        ).mean(axis=-1)
    diagonal /= float(args.n_panels)
    off_diagonal /= float(args.n_panels)

    ranked = _ranking_rows(diagonal, off_diagonal)
    filtered = [
        row for row in ranked
        if int(row["layer"]) >= args.skip_early_layers
    ]
    out_path = Path(args.output).expanduser().resolve(strict=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gaze_path = out_path.parent / "comic_gaze_scores.npy"
    matrix_path = out_path.parent / "mean_panel_attention.npy"
    offdiag_path = out_path.parent / "comic_off_diagonal_scores.npy"
    np.save(gaze_path, diagonal.astype(np.float32))
    np.save(matrix_path, mean_matrix.astype(np.float32))
    np.save(offdiag_path, off_diagonal.astype(np.float32))

    result = {
        "args": {
            **vars(args),
            "ranking_method": "comic-gaze",
            "query_mode": "last_prompt",
        },
        "model": "Robo-Dopamine-GRM",
        "num_layers": n_layers,
        "num_heads": n_heads,
        "default_ranking": "gaze_mean",
        "skip_early_layers": args.skip_early_layers,
        "n_samples": n_planned,
        "n_valid_samples": len(valid_names),
        "top_heads": filtered,
        "rankings": {"gaze_mean": ranked},
        "comic_gaze": {
            "protocol_source": str(gaze_root.resolve(strict=False)),
            "score_kind": "raw_image_token_attention_diagonal_mean",
            "query_mode": "last_prompt",
            "complete_strip_only": True,
            "n_panels": args.n_panels,
            "n_valid_samples": len(valid_names),
            "sample_names": valid_names,
            "skipped": skipped,
            "gaze_scores_npy": str(gaze_path),
            "mean_panel_attention_npy": str(matrix_path),
            "off_diagonal_scores_npy": str(offdiag_path),
            "sample_audit": sample_audit,
        },
    }
    out_path.write_text(json.dumps(result, indent=2))

    print(f"[comic-gaze] valid strips: {len(valid_names)}/{n_planned}", flush=True)
    print(f"[comic-gaze] top-{min(args.top_k, len(filtered))} heads:", flush=True)
    for row in filtered[: args.top_k]:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} "
            f"diag={row['score']:.6f} "
            f"offdiag={row['off_diagonal_mean']:.6f}",
            flush=True,
        )
    print(f"[comic-gaze] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
