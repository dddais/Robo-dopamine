"""Rank attention heads by raw attention mass on GroundingDINO bbox tokens.

This is the gaze-heads-style head selection adapted for GRM. The Localization
Heads criterion used by scan_localization_heads_best.py (high s_img + low spatial
entropy) finds heads whose attention is spatially concentrated, but says nothing
about WHAT the head is looking at — a head locked onto a table corner will rank
high while ignoring the task object entirely. For causal steering we need heads
that actually attend to the task-relevant object, so the ranking must be
conditioned on the object's location.

The gaze score analogue for GRM:
    score(l, h) = sum of post-softmax attention that head (l, h) places on the
                  image tokens inside the GroundingDINO bbox for the task object

This mirrors gaze_heads.aggregate_region_attention: raw (un-normalized) attention
summed over a region. A head that ignores images scores ~0; a head that spreads
attention everywhere also scores low; only a head that (a) looks at images and
(b) concentrates on the object scores high.

Aggregation across samples: bbox location is NOT stable across frames (an object
gets picked up, moved, occluded), so we compute the score per-sample and report
four aggregations — mean, max, median, selection_frequency — so the ranking can
be cross-checked. Default ranking is by mean (matches gaze_heads' strip averaging).

Output: head_ranking_by_bbox.json, consumed by steer_grm_heads.py via
--head-ranking-json (instead of the entropy-based --head-scan-json).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_localization_heads_best import (  # noqa: E402
    IMAGE_LABELS,
    ImageSpan,
    _run_forward_for_query,
    build_prompt,
    infer_image_spans,
    load_model_and_processor,
    move_inputs_to_device,
    select_query_positions,
)
from grounding import GroundingBox, TaskGrounding, grounding_box_to_record  # noqa: E402
from steer_grm_heads import bbox_to_token_positions  # noqa: E402


def _target_image_path_for_sample(sample: dict, target_label: str) -> Optional[str]:
    idx = IMAGE_LABELS.index(target_label)
    if idx >= len(sample.get("image", [])):
        return None
    path = sample["image"][idx]
    return str(path) if Path(path).exists() else None


def build_bbox_sequence(
    grounding: TaskGrounding,
    samples: Sequence[dict],
    target_label: str,
    out_json: Optional[Path] = None,
) -> List[Optional[GroundingBox]]:
    image_paths: List[str] = []
    tasks: List[str] = []
    index_map: List[int] = []
    for idx, sample in enumerate(samples):
        path = _target_image_path_for_sample(sample, target_label)
        if path is None:
            continue
        image_paths.append(path)
        tasks.append(sample["task"])
        index_map.append(idx)

    selected_full: List[Optional[GroundingBox]] = [None for _ in samples]
    if image_paths:
        seq_json = None
        if out_json is not None:
            seq_json = out_json
        selected = grounding.ground_best_sequence(image_paths, tasks, write_json=seq_json)
        for idx, box in zip(index_map, selected):
            selected_full[idx] = box
    return selected_full


def collect_per_head_bbox_mass(
    torch,
    model,
    processor,
    sample: dict,
    args: argparse.Namespace,
    dtype,
    target_label: str,
    grounding: TaskGrounding,
    spatial_merge_size: int,
    target_box: Optional[GroundingBox] = None,
    allow_single_frame_grounding: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[GroundingBox]]:
    """Run one forward and return per-head raw attention mass on the bbox tokens.

    Returns (mass[L, H], query_positions, box_label) or (None, None, None) if
    GroundingDINO produced no box for this sample's target view.

    mass[l, h] = sum over query_positions of sum over bbox_tokens of
                 post-softmax attention. Query positions follow args.query_mode
                 so the ranking is consistent with whichever query semantics
                 the downstream steering / scan uses.
    """
    image_paths = sample["image"]
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in image_paths]
    score_suffix = args.query_mode == "score"
    prompt = build_prompt(processor, sample["task"], args.analysis_suffix, score_suffix=score_suffix)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    spans = infer_image_spans(inputs, model.config, image_paths)
    span_by_label = {s.label: s for s in spans}

    target_span = span_by_label.get(target_label)
    if target_span is None:
        return None, None, None

    # GroundingDINO box for the target view. Prefer the trajectory-smoothed box
    # supplied by main(); fall back to stateless single-frame grounding for
    # backwards-compatible ad hoc calls.
    gbox = target_box
    if gbox is None and allow_single_frame_grounding:
        gbox = grounding.ground_best(target_span.path, sample["task"])
    if gbox is None:
        return None, None, None
    target_positions = bbox_to_token_positions(target_span, gbox.bbox, spatial_merge_size, target_span.path)
    if not target_positions:
        return None, None, None

    ids = inputs["input_ids"][0].detach().cpu().tolist()
    special_ids = [
        int(getattr(model.config, "image_token_id", 151655)),
        int(getattr(model.config, "video_token_id", 151656)),
        int(getattr(model.config, "vision_start_token_id", 151652)),
        int(getattr(model.config, "vision_end_token_id", 151653)),
    ]
    query_positions, _ = select_query_positions(
        processor.tokenizer, ids, spans, mode=args.query_mode,
        tail_tokens=args.tail_query_tokens, query_text=args.query_text,
        special_ids=special_ids, score_query_tokens=args.score_query_tokens,
    )

    device = next(model.parameters()).device
    inputs = move_inputs_to_device(torch, inputs, device, dtype)
    attentions, query_positions, _ = _run_forward_for_query(
        torch, model, inputs, args.query_mode,
        getattr(args, "generate_max_new_tokens", 64), query_positions, processor.tokenizer,
        args.score_query_tokens, args.generate_query_stage,
    )

    target_set = set(target_positions)
    n_layers = len(attentions)
    n_heads = int(attentions[0].shape[1])
    mass = np.zeros((n_layers, n_heads), dtype=np.float64)

    for li in range(n_layers):
        # [1, H, query_len, key_len] -> [H, query_len, key_len]
        attn = attentions[li][0].detach().float().cpu().numpy()
        # Sum attention over query positions, then over target key positions.
        # query_positions selects the score-deciding query rows.
        # Using mean over query positions keeps the magnitude comparable to a
        # single-row attention distribution (rows sum to 1, so sum of |Q| rows
        # would inflate mass by |Q|).
        q_idx = list(query_positions)
        sub = attn[:, q_idx, :]            # [H, |Q|, key_len]
        per_query_mass = sub[:, :, list(target_positions)].sum(axis=-1)  # [H, |Q|]
        mass[li] = per_query_mass.mean(axis=-1)  # [H]

    return mass, query_positions, gbox


def aggregate_and_rank(
    per_sample: List[dict],
    num_layers: int,
    num_heads: int,
) -> Dict[str, List[dict]]:
    """Build four rankings from per-sample [L,H] mass arrays.

    mean / max / median: aggregate the mass array across samples, then rank.
    selection_frequency: per sample, mark heads in the top-k by mass as "hit";
        rank by hit count. This rewards heads that localize the object
        consistently across frames even if their absolute mass varies.
    """
    valid = [ps for ps in per_sample if ps["mass"] is not None]
    if not valid:
        return {k: [] for k in ("mean", "max", "median", "selection_frequency")}

    stack = np.stack([ps["mass"] for ps in valid], axis=0)  # [N, L, H]
    top_k_per_sample = max(1, int(0.05 * num_layers * num_heads))  # top 5% per frame

    hits = np.zeros((num_layers, num_heads), dtype=np.float64)
    for arr in stack:
        flat = arr.reshape(-1)
        if flat.size <= top_k_per_sample:
            threshold = float(np.min(flat))
        else:
            threshold = float(np.sort(flat)[-top_k_per_sample])
        hits += (arr >= threshold).astype(np.float64)

    agg = {
        "mean": stack.mean(axis=0),
        "max": stack.max(axis=0),
        "median": np.median(stack, axis=0),
        "selection_frequency": hits,
    }

    rankings: Dict[str, List[dict]] = {}
    for name, arr in agg.items():
        ranked = []
        for li in range(num_layers):
            for hi in range(num_heads):
                ranked.append({"layer": li, "head": hi, "score": float(arr[li, hi])})
        ranked.sort(key=lambda r: r["score"], reverse=True)
        rankings[name] = ranked
    return rankings


def main():
    ap = argparse.ArgumentParser(description="Rank GRM heads by raw attention mass on GroundingDINO bbox tokens")
    ap.add_argument("--model-path", default="./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview")
    ap.add_argument("--sample-json", action="append", required=True, help="Can be repeated to mix episodes/tasks")
    ap.add_argument("--target-label", default="after_cam_high", choices=IMAGE_LABELS,
                    help="Which GRM view the bbox is grounded on")
    ap.add_argument("--grounding-model", default="../model/grounding-dino-base")
    ap.add_argument("--grounding-box-threshold", type=float, default=0.12)
    # Query semantics — kept identical to scan_localization_heads_best so the
    # ranking is consistent with whatever attention the downstream analysis uses.
    ap.add_argument("--query-mode", default="last_prompt",
                    choices=["last_prompt", "score", "tail", "all_after_images", "generate"])
    ap.add_argument("--tail-query-tokens", type=int, default=5)
    ap.add_argument("--query-text", default=None)
    ap.add_argument("--score-query-tokens", default="digits")
    ap.add_argument("--generate-query-stage", default="predict_token", choices=["predict_token", "score_token"])
    ap.add_argument("--generate-max-new-tokens", type=int, default=64)
    ap.add_argument("--analysis-suffix", default=None)
    ap.add_argument("--skip-early-layers", type=int, default=2,
                    help="Drop the top heads from these earliest layers when printing top-k, "
                         "since early-layer heads rarely drive the score decision. "
                         "The full per-layer table is still written to the JSON.")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device-map", default="none")
    ap.add_argument("--max-pixels", type=int, default=76800)
    ap.add_argument("--min-pixels", type=int, default=12544)
    ap.add_argument("--num-samples", type=int, default=None, help="Cap samples per sample-json")
    ap.add_argument("--top-k", type=int, default=8, help="Top-k to report in the printed summary")
    ap.add_argument("--default-ranking", default="mean", choices=["mean", "max", "median", "selection_frequency"],
                    help="Which aggregation the output JSON's top-level top_heads uses")
    ap.add_argument("--output", default="./results/rank_heads_by_bbox/head_ranking.json")
    args = ap.parse_args()

    # Load GRM
    print(f"[rank-by-bbox] loading GRM from {args.model_path}")
    torch, model, processor, dtype = load_model_and_processor(args)
    num_layers = int(getattr(model.config.text_config, "num_hidden_layers",
                             getattr(model.config, "num_hidden_layers", 0)))
    num_heads = int(getattr(model.config.text_config, "num_attention_heads",
                            getattr(model.config, "num_attention_heads", 0)))
    spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    print(f"[rank-by-bbox] num_layers={num_layers} num_heads={num_heads} "
          f"target_label={args.target_label}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[rank-by-bbox] loading GroundingDINO from {args.grounding_model}")
    grounding = TaskGrounding(
        model_path=args.grounding_model, device=device,
        box_threshold=args.grounding_box_threshold,
    )

    per_sample: List[dict] = []
    sample_idx_global = 0
    for sj_path in args.sample_json:
        samples_all = json.loads(Path(sj_path).read_text())
        bbox_seq_json = Path(args.output).parent / f"bbox_sequence_{Path(sj_path).parent.name}_{args.target_label}.json"
        bbox_sequence_all = build_bbox_sequence(grounding, samples_all, args.target_label, bbox_seq_json)
        indexed = list(enumerate(samples_all))
        if args.num_samples is not None:
            step = max(1, len(indexed) // max(1, args.num_samples))
            indexed = indexed[::step][: args.num_samples]
        for original_idx, sample in indexed:
            print(f"[rank-by-bbox] sample {sample_idx_global}: {sample.get('id','')[:60]}")
            target_box = bbox_sequence_all[original_idx] if bbox_sequence_all else None
            mass, qpos, box_label = collect_per_head_bbox_mass(
                torch, model, processor, sample, args, dtype,
                args.target_label, grounding, spatial_merge_size,
                target_box=target_box,
                allow_single_frame_grounding=False,
            )
            if mass is None:
                print(f"    [skip] no grounding box on {args.target_label}")
            else:
                print(f"    box_label={box_label.label!r} query_positions={len(qpos) if qpos else 0} "
                      f"max mass={float(mass.max()):.4f}")
            per_sample.append({
                "sample_id": sample.get("id"),
                "task": sample.get("task"),
                "mass": (mass.tolist() if mass is not None else None),
                "box_label": (box_label.label if box_label is not None else None),
                "bbox": grounding_box_to_record(box_label, _target_image_path_for_sample(sample, args.target_label) or ""),
            })
            sample_idx_global += 1

    rankings = aggregate_and_rank(per_sample, num_layers, num_heads)
    n_valid = sum(1 for ps in per_sample if ps["mass"] is not None)
    print(f"\n[rank-by-bbox] valid samples (with bbox): {n_valid}/{len(per_sample)}")

    print(f"\n[rank-by-bbox] top-{args.top_k} by {args.default_ranking} (layers >= {args.skip_early_layers}):")
    filtered = [r for r in rankings[args.default_ranking] if r["layer"] >= args.skip_early_layers]
    for r in filtered[: args.top_k]:
        print(f"  L{r['layer']:2d} H{r['head']:2d}  score={r['score']:.4f}")

    # Cross-check: print top-k for each aggregation so the user can see whether
    # different aggregations agree (robust) or diverge (fragile).
    print(f"\n[rank-by-bbox] top-{args.top_k} cross-aggregation (layers >= {args.skip_early_layers}):")
    for name in ("mean", "max", "median", "selection_frequency"):
        f = [r for r in rankings[name] if r["layer"] >= args.skip_early_layers]
        top = [(r["layer"], r["head"]) for r in f[: args.top_k]]
        print(f"  {name:20s}: {top}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "args": vars(args),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "target_label": args.target_label,
        "default_ranking": args.default_ranking,
        "skip_early_layers": args.skip_early_layers,
        "n_samples": len(per_sample),
        "n_valid_samples": n_valid,
        # top_heads uses the default aggregation, filtered to layers >=
        # skip_early_layers so consumers (steer_grm_heads) get score-relevant
        # heads directly. The unfiltered rankings[name] tables are kept for
        # inspection.
        "top_heads": [r for r in rankings[args.default_ranking] if r["layer"] >= args.skip_early_layers],
        "rankings": rankings,
        "per_sample": per_sample,
    }, indent=2))
    print(f"\n[rank-by-bbox] wrote {out_path}")


if __name__ == "__main__":
    main()
