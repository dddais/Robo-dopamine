"""Causal steering experiment for GRM localization heads.

Hypothesis: a set of attention heads, identified by scan_localization_heads_best.py
as "localization heads" (high s_img, low spatial entropy, IoU with GroundingDINO
bbox), causally drive the model's reward score. If so, steering their attention
onto the task-relevant object should systematically shift the score in the
direction predicted by the steering; steering onto a wrong region, or steering
random heads, should have a much weaker effect.

Design (four-group within-sample contrast, Gaze Heads style):

  Group                  | heads            | bias region
  -----------------------|------------------|---------------------------
  baseline (no steer)    | —                | —
  candidate + target     | top heads        | GroundingDINO bbox tokens
  candidate + wrong      | top heads        | random off-target tokens
  random   + target      | low-ranked heads | GroundingDINO bbox tokens
  all      + target      | every head       | GroundingDINO bbox tokens

For each group we run a forward pass that teacher-forces the canonical
`<score>0%</score>` suffix (so the score token is always present) and read the
model's predicted score. Steering is applied by adding a per-head additive bias
to the pre-softmax attention mask of the selected heads: +delta on the target
region's image tokens and -delta on the other image tokens of the same image
span (the LocalizationHeads/GazeHeads "boost_suppress" recipe). The bias is
applied during prefill so the steering shapes the score-computing forward pass.

Metrics per sample, aggregated across samples:
  - score(group): mean predicted score in the group
  - score_shift(group) = score(group) - score(baseline)
  - direction_flip_rate: fraction of samples where candidate+target and
    candidate+wrong move the score in opposite directions
  - control_gap = |candidate+target shift| - |random+target shift|: how much
    extra leverage the candidate heads have over random heads for the same
    target region. A positive, large control_gap is the causal signature.

The intervention mechanism mirrors gaze_heads/steering.py:make_static_attention_mask_hook,
adapted for Qwen3-VL (verified: attention_mask enters Qwen3VLTextAttention as
[1,1,q,k] and is broadcast against [1, num_q_heads, q, k] inside eager attention,
so a [1, num_q_heads, 1, k] additive bias broadcasts correctly).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Reuse scan script's prompt builder / model loader / span inference. This is
# not a soft dependency: this script is part of the same probe toolchain.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_localization_heads_best import (  # noqa: E402
    IMAGE_LABELS,
    ImageSpan,
    build_prompt,
    infer_image_spans,
    load_model_and_processor,
    move_inputs_to_device,
)
from grounding import GroundingBox, TaskGrounding, grounding_box_to_record  # noqa: E402


SCORE_RE = re.compile(r"<score>\s*([+-]?\d+(?:\.\d+)?)\s*%\s*</score>", re.IGNORECASE)


def target_image_paths_for_samples(samples: Sequence[dict], target_label: str) -> List[Optional[str]]:
    idx = IMAGE_LABELS.index(target_label)
    out: List[Optional[str]] = []
    for sample in samples:
        images = sample.get("image", [])
        if idx >= len(images) or not Path(images[idx]).exists():
            out.append(None)
        else:
            out.append(str(images[idx]))
    return out


def build_smoothed_bbox_sequence(
    grounding: Optional[TaskGrounding],
    samples: Sequence[dict],
    target_label: str,
    write_json: Optional[Path] = None,
) -> List[Optional[GroundingBox]]:
    if grounding is None:
        return [None for _ in samples]
    paths_all = target_image_paths_for_samples(samples, target_label)
    image_paths: List[str] = []
    tasks: List[str] = []
    index_map: List[int] = []
    for idx, (sample, path) in enumerate(zip(samples, paths_all)):
        if path is None:
            continue
        image_paths.append(path)
        tasks.append(sample["task"])
        index_map.append(idx)
    selected_full: List[Optional[GroundingBox]] = [None for _ in samples]
    if image_paths:
        selected = grounding.ground_best_sequence(image_paths, tasks, write_json=write_json)
        for idx, box in zip(index_map, selected):
            selected_full[idx] = box
    return selected_full


@dataclass
class HeadSpec:
    layer: int
    head: int
    label: str = ""


@dataclass
class SteeringConfig:
    swap_bias: float = 4.0
    """Additive bias magnitude added to the pre-softmax attention mask.

    4.0 is a moderate value: it is large enough to dominate the unscaled
    log-attention range in a single forward pass (softmax(4) ~ 0.98 vs the
    rest), but small enough that suppress boost interactions stay numerically
    stable in bfloat16. Sweep this if effects saturate or vanish."""


def parse_heads(head_json: Path, top_k: int, focus_label: Optional[str] = None, ranking: Optional[str] = None) -> List[HeadSpec]:
    """Read candidate heads from a ranking JSON.

    Works with both stage-1 output (scan_localization_heads_best's head_scan.json,
    ranked by entropy/s_img) and stage-2 output (rank_heads_by_bbox's
    head_ranking.json, ranked by raw bbox attention mass). Both write a
    top-level `top_heads` list of {layer, head, ...}.

    For stage-2 output, `ranking` selects which aggregation's table to use
    ("mean"/"max"/"median"/"selection_frequency"); default None reads the
    pre-filtered `top_heads` (the default_ranking, already skip_early_layers-
    filtered at rank time).
    """
    data = json.loads(head_json.read_text())

    if ranking is not None:
        rankings = data.get("rankings") or {}
        if ranking not in rankings:
            raise ValueError(f"ranking={ranking!r} not found; available: {list(rankings.keys())}")
        top_heads = rankings[ranking]
        skip = int(data.get("skip_early_layers", 0))
        top_heads = [r for r in top_heads if int(r["layer"]) >= skip]
    else:
        top_heads = data.get("top_heads") or []

    if not top_heads:
        raise ValueError(f"{head_json} has no top_heads; rerun the ranking script")
    selected: List[HeadSpec] = []
    for entry in top_heads:
        # focus_label only meaningful for stage-1 output (entry["label"] is an
        # image label like "after_cam_high"). Stage-2 entries use "L{l}_H{h}"
        # head ids, so focus_label is ignored there.
        if focus_label is not None and entry.get("label", "").startswith("L"):
            pass
        elif focus_label is not None and entry.get("label") != focus_label:
            continue
        layer = int(entry["layer"])
        head = int(entry["head"])
        selected.append(HeadSpec(layer=layer, head=head, label=entry.get("label", "")))
        if len(selected) >= top_k:
            break
    if not selected:
        raise ValueError(f"no heads in {head_json} matched focus_label={focus_label!r}")
    return selected


def group_heads_by_layer(heads: Sequence[HeadSpec]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = {}
    for h in heads:
        grouped.setdefault(int(h.layer), []).append(int(h.head))
    return grouped


def lm_layers(model):
    """Return the ModuleList of language-model decoder layers for Qwen3-VL GRM."""
    return model.model.language_model.layers


def num_query_heads(model) -> int:
    cfg = getattr(model.config, "text_config", model.config)
    n = getattr(cfg, "num_attention_heads", None)
    if n is None:
        # Fall back to q_proj out_features // head_dim on layer 0.
        sa = lm_layers(model)[0].self_attn
        n = int(sa.q_proj.out_features // sa.head_dim)
    return int(n)


def select_random_heads(num_layers: int, num_heads: int, k: int, rng: random.Random, exclude: Optional[set] = None) -> List[HeadSpec]:
    exclude = exclude or set()
    pool = [(l, h) for l in range(num_layers) for h in range(num_heads) if (l, h) not in exclude]
    rng.shuffle(pool)
    return [HeadSpec(layer=l, head=h, label="random") for l, h in pool[:k]]


def select_low_ranked_heads(
    head_json: Path,
    k: int,
    rng: random.Random,
    exclude: Optional[set] = None,
    ranking: Optional[str] = None,
    low_rank_fraction: float = 0.25,
) -> Tuple[List[HeadSpec], dict]:
    """Sample control heads from the low bbox-mass segment of a stage-2 ranking.

    This mirrors gaze-heads' non-gaze control: the random-target group should be
    heads that are unlikely to carry target-object attention, not arbitrary
    heads that may include moderately localizing heads.
    """
    data = json.loads(head_json.read_text())
    rankings = data.get("rankings") or {}
    ranking_name = ranking or data.get("default_ranking") or "mean"
    if ranking_name in rankings:
        rows = rankings[ranking_name]
        source = f"rankings.{ranking_name}"
    else:
        rows = data.get("top_heads") or []
        source = "top_heads"

    skip = int(data.get("skip_early_layers", 0))
    exclude = exclude or set()
    clean_rows = []
    seen = set()
    for row in rows:
        pair = (int(row["layer"]), int(row["head"]))
        if pair in seen or pair[0] < skip:
            continue
        seen.add(pair)
        clean_rows.append(row)
    if not clean_rows:
        raise ValueError(f"{head_json} has no ranking rows available for low-ranked controls")

    frac = max(0.0, min(float(low_rank_fraction), 1.0))
    low_pool_size = max(int(k), int(math.ceil(len(clean_rows) * frac)))
    low_rows = list(clean_rows[-low_pool_size:])
    pool = [r for r in low_rows if (int(r["layer"]), int(r["head"])) not in exclude]

    if len(pool) < k:
        # If the chosen low segment is too small after exclusions, expand from
        # the bottom upward before giving up. This preserves the "as low-ranked
        # as possible" control while keeping head count matched to candidates.
        pool = [
            r for r in reversed(clean_rows)
            if (int(r["layer"]), int(r["head"])) not in exclude
        ]
    if len(pool) < k:
        raise ValueError(
            f"Only {len(pool)} low-ranked control heads available in {head_json}, "
            f"but k={k} was requested"
        )

    rng.shuffle(pool)
    selected = [
        HeadSpec(layer=int(r["layer"]), head=int(r["head"]), label="low_rank_control")
        for r in pool[:k]
    ]
    meta = {
        "type": "low_ranked",
        "ranking_json": str(head_json),
        "ranking": ranking_name,
        "source": source,
        "skip_early_layers": skip,
        "low_rank_fraction": frac,
        "ranked_heads_available": len(clean_rows),
        "low_pool_size": low_pool_size,
        "eligible_low_pool_size": len(pool),
        "selected_heads": [(h.layer, h.head) for h in selected],
    }
    return selected, meta


def all_heads(num_layers: int, num_heads: int) -> List[HeadSpec]:
    return [HeadSpec(layer=l, head=h, label="all") for l in range(num_layers) for h in range(num_heads)]


def make_steering_hook(
    head_indices: Sequence[int],
    target_positions: Sequence[int],
    other_positions: Sequence[int],
    n_query_heads: int,
    device: str,
    swap_bias: float,
):
    """Build a forward pre-hook (with_kwargs) that biases selected heads' attention.

    target_positions receive +swap_bias; other_positions receive -swap_bias.
    Positions outside both lists are left at 0 (unchanged). The bias tensor is
    [1, n_query_heads, 1, kv_len]; when added to the incoming [1,1,q,kv] mask it
    broadcasts to [1, n_query_heads, q, kv] exactly as HF eager attention
    expects (modeling_qwen3_vl.py:158 `attn_weights + causal_mask`).

    Applied at prefill (we do NOT gate on mask.shape[-2]==1), because the score
    is computed during the prefill forward of the teacher-forced prompt.
    """
    h_idx = [int(h) for h in head_indices]
    tgt = [int(p) for p in target_positions]
    oth = [int(p) for p in other_positions]
    if not h_idx or (not tgt and not oth):
        def no_op(_module, args, kwargs):
            return None
        return no_op

    all_pos = tgt + oth
    base_len = max(all_pos) + 1
    base_bias = torch.zeros((1, n_query_heads, 1, base_len), dtype=torch.float32, device=device)
    h_tensor = torch.tensor(h_idx, dtype=torch.long, device=device)
    if tgt:
        t_tensor = torch.tensor(tgt, dtype=torch.long, device=device)
        base_bias[0, h_tensor[:, None], 0, t_tensor[None, :]] = float(swap_bias)
    if oth:
        o_tensor = torch.tensor(oth, dtype=torch.long, device=device)
        base_bias[0, h_tensor[:, None], 0, o_tensor[None, :]] = -float(swap_bias)

    dtype_cache: Dict = {}

    def hook(_module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            return None
        key = (mask.dtype, mask.device)
        if key not in dtype_cache:
            dtype_cache[key] = base_bias.to(mask.device, mask.dtype)
        bias = dtype_cache[key]
        kv_len = int(mask.shape[-1])
        if kv_len <= base_len:
            padded = bias[:, :, :, :kv_len]
        else:
            pad_len = kv_len - base_len
            padded = torch.nn.functional.pad(bias, (0, pad_len))
        new_kwargs = dict(kwargs)
        new_kwargs["attention_mask"] = mask + padded
        return args, new_kwargs

    return hook


def register_layer_hooks(model, hook_by_layer: Dict[int, object]) -> List:
    handles = []
    layers = lm_layers(model)
    for layer_idx, hook_fn in hook_by_layer.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        handles.append(layers[layer_idx].self_attn.register_forward_pre_hook(hook_fn, with_kwargs=True))
    return handles


def remove_handles(handles: Sequence) -> None:
    for h in handles:
        h.remove()


def bbox_to_token_positions(span: ImageSpan, box: Sequence[float], spatial_merge_size: int, image_path: str) -> List[int]:
    """Map a pixel-space bbox to the image-token positions it covers.

    Uses grid-cell / bbox rectangle INTERSECTION (not center-in-box). The scan
    script's mask_grid_from_box uses center-point sampling, which is fine for
    IoU evaluation but drops small bboxes that fall between cell centers — for
    steering we want every cell overlapping the bbox to receive the bias, so we
    cannot reuse it. Token layout is row-major over (gh, gw) after
    spatial_merge_size folding, matching vector_to_grid in the scan script.
    Multi-frame spans (t>1) get the mask replicated across time; GRM probes are
    single-frame.
    """
    from PIL import Image
    if image_path and Path(image_path).exists():
        with Image.open(image_path) as im:
            width, height = im.size
    else:
        # Fall back to square assumption; caller should pass image_path.
        width = height = 256
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return []

    gh = max(1, span.grid_thw[1] // spatial_merge_size)
    gw = max(1, span.grid_thw[2] // spatial_merge_size)
    # Cell boundaries along each axis.
    xs = np.linspace(0, width, gw + 1)
    ys = np.linspace(0, height, gh + 1)
    # A cell [j] is selected if its extent [xs[j], xs[j+1]] intersects [x1, x2].
    x_keep = (xs[:-1] < x2) & (xs[1:] > x1)
    y_keep = (ys[:-1] < y2) & (ys[1:] > y1)
    grid = np.outer(y_keep, x_keep)
    flat = grid.reshape(-1)
    t = span.grid_thw[0]
    if t != 1:
        flat = np.broadcast_to(flat[None], (t, flat.shape[0])).reshape(-1)
    return [span.start + i for i, keep in enumerate(flat) if keep]


def random_token_positions(span: ImageSpan, n_target: int, rng: random.Random, spatial_merge_size: int, image_path: Optional[str]) -> List[int]:
    """Pick a random set of image-token positions inside the span, matching the
    count of the target box (so candidate+target and candidate+wrong apply bias
    to the same number of tokens)."""
    total = (span.end - span.start)
    k = max(1, min(n_target, total))
    return [span.start + i for i in rng.sample(range(total), k)]


def random_positions_from_pool(positions: Sequence[int], n_target: int, rng: random.Random) -> List[int]:
    """Pick random token positions from a pre-filtered pool.

    Used for the wrong-region control: the pool is bbox-complement image tokens,
    so the boosted wrong region cannot overlap the true target bbox.
    """
    pool = [int(p) for p in positions]
    if not pool:
        return []
    k = max(1, min(int(n_target), len(pool)))
    return rng.sample(pool, k)


@dataclass
class SampleContext:
    sample: dict
    inputs: dict
    span_by_label: Dict[str, ImageSpan]
    target_span: Optional[ImageSpan]
    target_box: Optional[GroundingBox]
    target_positions: List[int]
    other_positions: List[int]


def build_sample_context(
    torch,
    model,
    processor,
    sample: dict,
    grounding: Optional[TaskGrounding],
    target_label: str,
    dtype,
    rng: random.Random,
    spatial_merge_size: int,
    target_box: Optional[GroundingBox] = None,
    allow_single_frame_grounding: bool = True,
) -> SampleContext:
    image_paths = sample["image"]
    images = [__import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB") for p in image_paths]
    # score_suffix=False: do NOT teacher-force a score. The model generates its
    # own score so steering acts on the actual generation (causal).
    prompt = build_prompt(processor, sample["task"], analysis_suffix=None, score_suffix=False)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    spans = infer_image_spans(inputs, model.config, image_paths)
    span_by_label = {s.label: s for s in spans}
    device = next(model.parameters()).device
    inputs = move_inputs_to_device(torch, inputs, device, dtype)

    target_span = span_by_label.get(target_label)
    gbox = target_box
    target_positions: List[int] = []
    other_positions: List[int] = []

    if target_span is not None and grounding is not None:
        path = target_span.path
        if gbox is None and allow_single_frame_grounding:
            gbox = grounding.ground_best(path, sample["task"])
        if gbox is not None:
            target_positions = bbox_to_token_positions(target_span, gbox.bbox, spatial_merge_size, path)
            other_positions = [
                p for p in range(target_span.start, target_span.end) if p not in set(target_positions)
            ]
    return SampleContext(
        sample=sample,
        inputs=inputs,
        span_by_label=span_by_label,
        target_span=target_span,
        target_box=gbox,
        target_positions=target_positions,
        other_positions=other_positions,
    )


def parse_score(text: str) -> Optional[float]:
    m = SCORE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1)) / 100.0
    except ValueError:
        return None


def generate_score(model, processor, inputs, torch, max_new_tokens: int = 12) -> Optional[float]:
    """Generate the model's score via a short greedy decode and parse it.

    The prompt ends right after the assistant role header + `<score>` opener is
    NOT included — we let the model emit `<score>...%</score>` itself so the
    steering bias acts on the actual generation, not on a teacher-forced value.
    This is the causal question: does steering change what the model freely
    outputs?

    Uses use_cache=True for speed (one prefill + a few decode steps). The
    steering mask hooks (registered outside) fire on both prefill and decode.
    """
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
            output_attentions=False,
            return_dict_in_generate=True,
        )
    gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
    return parse_score(text)


def _run_with_bias(
    torch,
    model,
    processor,
    ctx: SampleContext,
    heads: Sequence[HeadSpec],
    target_positions: Sequence[int],
    other_positions: Sequence[int],
    swap_bias: float,
    n_qheads: int,
    device: str,
) -> Optional[float]:
    """Register steering hooks for the given positions and run one generation."""
    hook_by_layer: Dict[int, object] = {}
    heads_by_layer = group_heads_by_layer(heads)
    for layer_idx, hlist in heads_by_layer.items():
        hook_by_layer[layer_idx] = make_steering_hook(
            head_indices=hlist,
            target_positions=target_positions,
            other_positions=other_positions,
            n_query_heads=n_qheads,
            device=device,
            swap_bias=swap_bias,
        )

    handles = register_layer_hooks(model, hook_by_layer)
    try:
        return generate_score(model, processor, ctx.inputs, torch)
    finally:
        remove_handles(handles)


def run_group(
    torch,
    model,
    processor,
    ctx: SampleContext,
    heads: Sequence[HeadSpec],
    bias_target: bool,
    swap_bias: float,
    n_qheads: int,
    device: str,
    rng: random.Random,
    spatial_merge_size: int,
    wrong_region_samples: int = 1,
) -> Optional[float]:
    """Run one steering condition and return the predicted score.

    For bias_target=False (candidate_wrong), wrong_region_samples > 1 draws
    that many independent wrong regions from the bbox-complement pool and
    returns the mean score, reducing the noise of a single random draw. Each
    draw reseeds from the same rng state so the sample set depends only on
    (seed, step, draw_index), not on unrelated conditions run beforehand.
    """
    if bias_target and ctx.target_span is not None and ctx.target_positions:
        target_positions = ctx.target_positions
        other_positions = ctx.other_positions
        return _run_with_bias(
            torch, model, processor, ctx, heads,
            target_positions, other_positions,
            swap_bias, n_qheads, device,
        )

    # wrong-region control: random bbox-complement positions, same count as target
    n_target = len(ctx.target_positions) if ctx.target_positions else 4
    if ctx.target_span is None or not ctx.target_positions or not ctx.other_positions:
        # No real bbox available (or bbox covers the full span); a strict
        # off-target control cannot be constructed for this sample.
        return None

    n_samples = max(1, int(wrong_region_samples))
    rng_state = rng.getstate()
    scores: List[float] = []
    for _ in range(n_samples):
        wrong_positions = random_positions_from_pool(ctx.other_positions, n_target, rng)
        other_positions = [p for p in range(ctx.target_span.start, ctx.target_span.end) if p not in set(wrong_positions)]
        s = _run_with_bias(
            torch, model, processor, ctx, heads,
            wrong_positions, other_positions,
            swap_bias, n_qheads, device,
        )
        if s is not None:
            scores.append(s)
    rng.setstate(rng_state)
    if not scores:
        return None
    return float(np.mean(scores)) if len(scores) > 1 else scores[0]


def aggregate_results(per_sample: List[dict]) -> dict:
    groups = ["baseline", "candidate_target", "candidate_wrong", "random_target", "all_target"]
    means: Dict[str, float] = {}
    for g in groups:
        vals = [ps["scores"][g] for ps in per_sample if ps["scores"].get(g) is not None]
        means[g] = float(np.mean(vals)) if vals else float("nan")

    shifts: Dict[str, float] = {}
    base = means["baseline"]
    for g in groups[1:]:
        shifts[g] = (means[g] - base) if (not math.isnan(means[g]) and not math.isnan(base)) else float("nan")

    # direction flip rate: candidate_target and candidate_wrong move opposite
    flip_count = 0
    flip_total = 0
    for ps in per_sample:
        ct = ps["scores"].get("candidate_target")
        cw = ps["scores"].get("candidate_wrong")
        bs = ps["scores"].get("baseline")
        if ct is None or cw is None or bs is None:
            continue
        flip_total += 1
        if (ct - bs) * (cw - bs) < 0:
            flip_count += 1
    flip_rate = (flip_count / flip_total) if flip_total else float("nan")

    # control gap: |candidate_target shift| - |random_target shift|
    ct_shifts = []
    rt_shifts = []
    for ps in per_sample:
        ct = ps["scores"].get("candidate_target")
        rt = ps["scores"].get("random_target")
        bs = ps["scores"].get("baseline")
        if ct is not None and bs is not None:
            ct_shifts.append(abs(ct - bs))
        if rt is not None and bs is not None:
            rt_shifts.append(abs(rt - bs))
    control_gap = (float(np.mean(ct_shifts)) - float(np.mean(rt_shifts))) if (ct_shifts and rt_shifts) else float("nan")

    return {
        "n_samples": len(per_sample),
        "mean_score": means,
        "mean_score_shift_from_baseline": shifts,
        "direction_flip_rate": flip_rate,
        "control_gap_abs_shift": control_gap,
    }


def main():
    ap = argparse.ArgumentParser(description="Causal steering of GRM localization heads")
    ap.add_argument("--model-path", default="./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview")
    ap.add_argument("--head-scan-json", default=None,
                    help="Stage-1 candidate heads (scan_localization_heads_best output). "
                         "Mutually exclusive with --head-ranking-json.")
    ap.add_argument("--head-ranking-json", default=None,
                    help="Stage-2 candidate heads (rank_heads_by_bbox output). Preferred for "
                         "causal steering because heads are ranked by bbox attention mass, not "
                         "just spatial concentration.")
    ap.add_argument("--ranking", default=None,
                    choices=["mean", "max", "median", "selection_frequency"],
                    help="Aggregation to use from stage-2 JSON. Default: the pre-filtered "
                         "top_heads (rank script's default_ranking). Ignored for stage-1 input.")
    ap.add_argument("--sample-json", required=True)
    ap.add_argument("--target-label", default="after_cam_high", choices=IMAGE_LABELS)
    ap.add_argument("--focus-label", default=None, help="Filter candidate heads by this scan label")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--random-control", default="low_ranked", choices=["low_ranked", "uniform"],
                    help="How to choose random_target control heads. low_ranked samples from "
                         "the bottom segment of the stage-2 bbox-mass ranking, matching "
                         "gaze-heads' non-gaze control. uniform preserves the old behavior.")
    ap.add_argument("--low-rank-fraction", type=float, default=0.25,
                    help="Bottom fraction of the ranking used as the low-ranked control pool.")
    ap.add_argument("--grounding-model", default="../model/grounding-dino-base")
    ap.add_argument("--grounding-box-threshold", type=float, default=0.12)
    ap.add_argument("--no-grounding", action="store_true", help="Skip GroundingDINO; steering will use random target regions only")
    ap.add_argument("--swap-bias", type=float, default=4.0)
    ap.add_argument("--wrong-region-samples", type=int, default=1,
                    help="For candidate_wrong: number of independent wrong regions drawn "
                         "from the bbox-complement pool, averaged to reduce single-draw noise.")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device-map", default="none")
    ap.add_argument("--max-pixels", type=int, default=76800, help="Match scan_localization_heads_best default so grid_thw is comparable")
    ap.add_argument("--min-pixels", type=int, default=12544)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=None, help="Limit number of samples processed")
    ap.add_argument("--sample-strategy", default="even", choices=["even", "first"],
                    help="How to subsample when --num-samples is set. 'even' matches "
                         "rank_heads_by_bbox.py and covers the whole trajectory; 'first' "
                         "keeps the old prefix-only behavior.")
    ap.add_argument("--output", default="./results/steer_grm_heads/results.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    samples = json.loads(Path(args.sample_json).read_text())
    if args.num_samples:
        if args.sample_strategy == "even":
            step = max(1, len(samples) // max(1, args.num_samples))
            samples = samples[::step][: args.num_samples]
        else:
            samples = samples[: args.num_samples]

    print(f"[steer] loading GRM model ...")
    torch, model, processor, dtype = load_model_and_processor(args)
    # scan's loader forces use_cache=False (it does single forwards with
    # output_attentions). Causal steering needs autoregressive score generation,
    # so re-enable KV cache for speed. attn_implementation stays eager so the
    # mask hooks fire.
    model.config.use_cache = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_qheads = num_query_heads(model)
    num_layers = len(lm_layers(model))
    spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    print(f"[steer] num_layers={num_layers} num_query_heads={n_qheads} spatial_merge_size={spatial_merge_size}")

    head_json_path = args.head_ranking_json or args.head_scan_json
    if head_json_path is None:
        ap.error("one of --head-scan-json or --head-ranking-json is required")
    candidate_heads = parse_heads(Path(head_json_path), args.top_k, focus_label=args.focus_label, ranking=args.ranking)
    head_source = "stage-2 (bbox mass)" if args.head_ranking_json else "stage-1 (entropy/mass)"
    print(f"[steer] candidate heads ({len(candidate_heads)}, {head_source}): {[(h.layer, h.head) for h in candidate_heads]}")
    candidate_set = {(h.layer, h.head) for h in candidate_heads}
    random_control_meta: dict
    if args.random_control == "low_ranked" and args.head_ranking_json:
        random_heads, random_control_meta = select_low_ranked_heads(
            Path(args.head_ranking_json),
            len(candidate_heads),
            rng,
            exclude=candidate_set,
            ranking=args.ranking,
            low_rank_fraction=args.low_rank_fraction,
        )
    else:
        random_heads = select_random_heads(num_layers, n_qheads, len(candidate_heads), rng, exclude=candidate_set)
        random_control_meta = {
            "type": "uniform",
            "selected_heads": [(h.layer, h.head) for h in random_heads],
        }
    print(f"[steer] random/control heads ({random_control_meta['type']}): {[(h.layer, h.head) for h in random_heads]}")
    all_h = all_heads(num_layers, n_qheads)

    grounding = None
    if not args.no_grounding:
        print(f"[steer] loading GroundingDINO ...")
        grounding = TaskGrounding(model_path=args.grounding_model, device=device, box_threshold=args.grounding_box_threshold)
    bbox_sequence = build_smoothed_bbox_sequence(
        grounding,
        samples,
        args.target_label,
        write_json=Path(args.output).parent / "bbox_sequence.json" if grounding is not None else None,
    )

    per_sample: List[dict] = []
    for si, sample in enumerate(samples):
        print(f"[steer] sample {si+1}/{len(samples)}: {sample.get('id', '')} task={sample.get('task','')!r}")
        try:
            ctx = build_sample_context(
                torch, model, processor, sample, grounding,
                target_label=args.target_label, dtype=dtype, rng=rng,
                spatial_merge_size=spatial_merge_size,
                target_box=bbox_sequence[si] if si < len(bbox_sequence) else None,
                allow_single_frame_grounding=False,
            )
        except Exception as e:
            print(f"    [warn] context build failed: {e}")
            continue
        if not ctx.target_positions and not args.no_grounding:
            print(f"    [warn] no grounding box for target; using random target positions")
        if not ctx.target_positions:
            # Even with --no-grounding we need some target positions; sample randomly.
            if ctx.target_span is not None:
                ctx.target_positions = random_token_positions(ctx.target_span, 8, rng, spatial_merge_size, ctx.target_span.path)
                ctx.other_positions = [p for p in range(ctx.target_span.start, ctx.target_span.end) if p not in set(ctx.target_positions)]

        scores: Dict[str, Optional[float]] = {}
        # Baseline: no hooks.
        scores["baseline"] = generate_score(model, processor, ctx.inputs, torch)
        scores["candidate_target"] = run_group(torch, model, processor, ctx, candidate_heads, True, args.swap_bias, n_qheads, device, rng, spatial_merge_size)
        scores["candidate_wrong"] = run_group(torch, model, processor, ctx, candidate_heads, False, args.swap_bias, n_qheads, device, rng, spatial_merge_size, wrong_region_samples=args.wrong_region_samples)
        scores["random_target"] = run_group(torch, model, processor, ctx, random_heads, True, args.swap_bias, n_qheads, device, rng, spatial_merge_size)
        scores["all_target"] = run_group(torch, model, processor, ctx, all_h, True, args.swap_bias, n_qheads, device, rng, spatial_merge_size)

        print(f"    scores: {scores}")
        per_sample.append({
            "sample_id": sample.get("id"),
            "task": sample.get("task"),
            "target_label": args.target_label,
            "bbox": grounding_box_to_record(ctx.target_box, ctx.target_span.path if ctx.target_span is not None else ""),
            "n_target_tokens": len(ctx.target_positions),
            "candidate_heads": [(h.layer, h.head) for h in candidate_heads],
            "random_control_heads": [(h.layer, h.head) for h in random_heads],
            "scores": scores,
        })

    summary = aggregate_results(per_sample)
    print("\n[steer] SUMMARY")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "args": vars(args),
        "random_control": random_control_meta,
        "summary": summary,
        "per_sample": per_sample,
    }, indent=2))
    print(f"[steer] wrote {out_path}")


if __name__ == "__main__":
    main()
