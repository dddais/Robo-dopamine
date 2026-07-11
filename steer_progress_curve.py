"""Trace GRM task progress over a full trajectory under several steering conditions.

For each step of a single trajectory's sample.json (one task, one instruction),
run the model N times with different attention-mask biases and record the
emitted score. Then convert raw scores to progress using the same mode-specific
logic as examples/inference.py and plot the resulting curves so a human can
visually compare conditions.

Conditions (mirrors steer_grm_heads.py):
    baseline          no intervention
    candidate_target  candidate heads steered onto GroundingDINO bbox of after_<cam>
    candidate_wrong   candidate heads steered onto a random off-target region (control)
    random_target     low-ranked/control heads steered onto target region
    all_target        all heads steered onto target region (upper bound)

Output:
    <output_dir>/curve.csv           per-step scores and accumulated progress
    <output_dir>/curve.json          same data + metadata
    <output_dir>/progress_curve.png  matplotlib comparison plot

Reuses steer_grm_heads.py for everything except the per-step loop and plotting.
"""
import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import steer_grm_heads as S
from grounding import grounding_box_to_record

HEAD_RANKING_JSON = "/home/dais/workspace/Robo-Dopamine/results/attention/corresponding_success_20260708/rank_carrot_incremental_last_prompt/head_ranking.json"
SAMPLE_JSON = "/home/dais/workspace/Robo-Dopamine/results/attention/success_pick3suc_1_carrot_samples/cube/forward_mode_pick_the_white_cube_and_put_it_on_yellow_plate/sample.json"

SAMPLE_ID_RE = re.compile(r"af_(\d+)")


def parse_after_frame(sample_id: str) -> Optional[int]:
    """Extract the after-frame number from the step id, e.g. '...af_000560' -> 560."""
    m = SAMPLE_ID_RE.search(sample_id)
    return int(m.group(1)) if m else None


def accumulate_incremental(raw_scores: List[Optional[float]]) -> List[Optional[float]]:
    """GRM incremental-mode progress formula (see examples/inference.py ~L594).

    prog_0 = raw_0
    prog_t = prog_{t-1} + (1 - prog_{t-1}) * raw_t   if raw_t >= 0
             prog_{t-1} + prog_{t-1} * raw_t         if raw_t < 0

    None scores propagate as None (curve breaks at that step).
    """
    out: List[Optional[float]] = []
    prev = 0.0
    for i, r in enumerate(raw_scores):
        if r is None:
            out.append(None)
            continue
        if i == 0:
            curr = r
        elif r >= 0:
            curr = prev + (1 - prev) * r
        else:
            curr = prev + prev * r
        out.append(curr)
        prev = curr
    return out




def infer_eval_mode(samples: List[dict], sample_json: str = "") -> str:
    """Infer forward/incremental/backward from sample ids or the sample path."""
    haystack = " ".join([sample_json] + [str(s.get("id", "")) for s in samples[:3]])
    for mode in ("incremental", "forward", "backward"):
        if f"{mode}_mode" in haystack or f"{mode}-" in haystack:
            return mode
    return "incremental"


def scores_to_progress(raw_scores: List[Optional[float]], eval_mode: str) -> List[Optional[float]]:
    """Convert model raw score fractions to progress for the selected eval mode."""
    if eval_mode == "incremental":
        return accumulate_incremental(raw_scores)
    if eval_mode == "forward":
        return [None if r is None else r for r in raw_scores]
    if eval_mode == "backward":
        return [None if r is None else max(0.0, min(1.0, 1.0 + r)) for r in raw_scores]
    raise ValueError(f"Unknown eval_mode: {eval_mode}")


def trace_curve(
    torch,
    model,
    processor,
    grounding,
    samples: List[dict],
    candidate_heads: List[S.HeadSpec],
    random_heads: List[S.HeadSpec],
    all_heads: List[S.HeadSpec],
    target_label: str,
    swap_bias: float,
    spatial_merge_size: int,
    rng: random.Random,
    dtype,
    bbox_sequence: Optional[List] = None,
    wrong_region_samples: int = 1,
) -> Dict[str, List[Optional[float]]]:
    """Run all conditions across all steps. Returns {condition: [score_per_step]}."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_qheads = S.num_query_heads(model)
    conditions = ["baseline", "candidate_target", "candidate_wrong", "random_target", "all_target"]
    scores: Dict[str, List[Optional[float]]] = {c: [] for c in conditions}

    for i, sample in enumerate(samples):
        target_box = bbox_sequence[i] if bbox_sequence is not None and i < len(bbox_sequence) else None
        ctx = S.build_sample_context(
            torch, model, processor, sample, grounding, target_label, dtype, rng, spatial_merge_size,
            target_box=target_box,
            allow_single_frame_grounding=False,
        )
        if ctx.target_positions:
            tgt_desc = f"bbox@{len(ctx.target_positions)}tok"
        else:
            tgt_desc = "no-bbox"
        print(f"  step {i:2d}/{len(samples)} af={parse_after_frame(sample['id'])} {tgt_desc}", flush=True)

        # baseline: no hooks
        scores["baseline"].append(S.generate_score(model, processor, ctx.inputs, torch))

        # candidate × target
        scores["candidate_target"].append(
            S.run_group(torch, model, processor, ctx, candidate_heads, bias_target=True,
                        swap_bias=swap_bias, n_qheads=n_qheads, device=device,
                        rng=rng, spatial_merge_size=spatial_merge_size)
        )
        # candidate × wrong region (control: same heads, off-target region)
        scores["candidate_wrong"].append(
            S.run_group(torch, model, processor, ctx, candidate_heads, bias_target=False,
                        swap_bias=swap_bias, n_qheads=n_qheads, device=device,
                        rng=rng, spatial_merge_size=spatial_merge_size,
                        wrong_region_samples=wrong_region_samples)
        )
        # random non-candidate heads × target region (control: head specificity)
        scores["random_target"].append(
            S.run_group(torch, model, processor, ctx, random_heads, bias_target=True,
                        swap_bias=swap_bias, n_qheads=n_qheads, device=device,
                        rng=rng, spatial_merge_size=spatial_merge_size)
        )
        # all heads × target region (upper bound on intervention strength)
        scores["all_target"].append(
            S.run_group(torch, model, processor, ctx, all_heads, bias_target=True,
                        swap_bias=swap_bias, n_qheads=n_qheads, device=device,
                        rng=rng, spatial_merge_size=spatial_merge_size)
        )

    return scores


def apply_task_override(samples: List[dict], override_task: Optional[str]) -> List[dict]:
    """Return samples with task text replaced for cross-instruction inference."""
    if override_task is None:
        return samples
    out: List[dict] = []
    for sample in samples:
        copied = dict(sample)
        copied["original_task"] = sample.get("task")
        copied["task"] = override_task
        out.append(copied)
    return out


def plot_curves(
    frames: List[int],
    raw_scores: Dict[str, List[Optional[float]]],
    progress: Dict[str, List[Optional[float]]],
    task: str,
    out_png: Path,
    fps: int = 30,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "baseline":         ("#2196F3", "-",  2.0, "o"),
        "candidate_target": ("#E91E63", "-",  2.6, "s"),
        "candidate_wrong":  ("#9E9E9E", "--", 1.6, None),
        "random_target":    ("#FF9800", "--", 1.6, None),
        "all_target":       ("#4CAF50", ":",  1.8, None),
    }
    labels = {
        "baseline":         "Baseline (no steering)",
        "candidate_target": "Candidate heads × target bbox",
        "candidate_wrong":  "Candidate heads × wrong region",
        "random_target":    "Control heads × target bbox",
        "all_target":       "All heads × target bbox",
    }

    time_sec = [f / fps for f in frames]
    fig, (ax_raw, ax_prog) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for cond in ["baseline", "candidate_target", "candidate_wrong", "random_target", "all_target"]:
        color, ls, lw, mk = style[cond]
        xs = []
        rs = []
        for x, v in zip(time_sec, raw_scores[cond]):
            if v is not None:
                xs.append(x)
                rs.append(v * 100)
        if not xs:
            continue
        ax_raw.plot(xs, rs, ls, color=color, linewidth=lw, marker=mk, markersize=4,
                    label=f"{labels[cond]}")

        xs2, ps = [], []
        for x, v in zip(time_sec, progress[cond]):
            if v is not None:
                xs2.append(x)
                ps.append(v * 100)
        if xs2:
            ax_prog.plot(xs2, ps, ls, color=color, linewidth=lw, marker=mk, markersize=4,
                         label=f"{labels[cond]}  (final={ps[-1]:.1f}%)")

    ax_raw.set_ylabel("Raw step score (%)", fontsize=12)
    ax_raw.set_title(f"Per-step score under steering — task: \"{task.strip()}\"", fontsize=13)
    ax_raw.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(fontsize=10, loc="best")

    ax_prog.set_xlabel("Time (s)", fontsize=12)
    ax_prog.set_ylabel("Accumulated progress (%)", fontsize=12)
    ax_prog.set_title("Accumulated progress (GRM incremental formula)", fontsize=13)
    ax_prog.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_prog.axhline(100, color="gray", linewidth=0.5, linestyle=":")
    ax_prog.set_xlim(0, time_sec[-1] if time_sec else 1)
    ax_prog.set_ylim(min(-5, ax_prog.get_ylim()[0]), max(105, ax_prog.get_ylim()[1]))
    ax_prog.grid(True, alpha=0.3)
    ax_prog.legend(fontsize=10, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Trace GRM progress curve under attention steering")
    ap.add_argument("--model-path", default="./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview")
    ap.add_argument("--head-ranking-json", default=HEAD_RANKING_JSON,
                    help="Stage-2 candidate head JSON (rank_heads_by_bbox output)")
    ap.add_argument("--ranking", default=None,
                    choices=["mean", "max", "median", "selection_frequency"])
    ap.add_argument("--sample-json", default=SAMPLE_JSON,
                    help="A trajectory's sample.json (multiple steps for one task)")
    ap.add_argument("--override-task", default=None,
                    help="Replace sample['task'] at inference time. Useful for running "
                         "one success trajectory under another instruction; bbox grounding "
                         "also follows the overridden target object.")
    ap.add_argument("--target-label", default="after_cam_high", choices=S.IMAGE_LABELS)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--random-control", default="low_ranked", choices=["low_ranked", "uniform"],
                    help="How to choose random_target control heads. low_ranked samples from "
                         "the bottom segment of the stage-2 bbox-mass ranking; uniform "
                         "preserves the old behavior.")
    ap.add_argument("--low-rank-fraction", type=float, default=0.25,
                    help="Bottom fraction of the ranking used as the low-ranked control pool.")
    ap.add_argument("--grounding-model", default="../model/grounding-dino-base")
    ap.add_argument("--grounding-box-threshold", type=float, default=0.12)
    ap.add_argument("--no-grounding", action="store_true")
    ap.add_argument("--swap-bias", type=float, default=6.0)
    ap.add_argument("--wrong-region-samples", type=int, default=1,
                    help="For candidate_wrong: number of independent wrong regions drawn "
                         "from the bbox-complement pool, averaged to reduce single-draw noise.")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device-map", default="none")
    ap.add_argument("--max-pixels", type=int, default=76800)
    ap.add_argument("--min-pixels", type=int, default=12544)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Limit number of steps (useful for quick smoke tests)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--output-dir", default="./results/steer_progress_curve")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    samples = json.loads(Path(args.sample_json).read_text())
    original_task = samples[0].get("task") if samples else None
    samples = apply_task_override(samples, args.override_task)
    if args.num_samples:
        samples = samples[: args.num_samples]
    print(f"[curve] trajectory: {len(samples)} steps, task={samples[0]['task']!r}")
    if args.override_task is not None:
        print(f"[curve] original task: {original_task!r}")

    print(f"[curve] loading GRM model ...")
    torch, model, processor, dtype = S.load_model_and_processor(args)
    model.config.use_cache = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_qheads = S.num_query_heads(model)
    num_layers = len(S.lm_layers(model))
    spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    print(f"[curve] num_layers={num_layers} num_query_heads={n_qheads} spatial_merge_size={spatial_merge_size}")

    candidate_heads = S.parse_heads(Path(args.head_ranking_json), args.top_k, ranking=args.ranking)
    print(f"[curve] candidate heads ({len(candidate_heads)}): {[(h.layer, h.head) for h in candidate_heads]}")
    candidate_set = {(h.layer, h.head) for h in candidate_heads}
    if args.random_control == "low_ranked":
        random_heads, random_control_meta = S.select_low_ranked_heads(
            Path(args.head_ranking_json),
            len(candidate_heads),
            rng,
            exclude=candidate_set,
            ranking=args.ranking,
            low_rank_fraction=args.low_rank_fraction,
        )
    else:
        random_heads = S.select_random_heads(num_layers, n_qheads, len(candidate_heads), rng, exclude=candidate_set)
        random_control_meta = {
            "type": "uniform",
            "selected_heads": [(h.layer, h.head) for h in random_heads],
        }
    print(f"[curve] random/control heads ({random_control_meta['type']}): {[(h.layer, h.head) for h in random_heads]}")
    all_h = S.all_heads(num_layers, n_qheads)

    grounding = None
    if not args.no_grounding:
        print(f"[curve] loading GroundingDINO ...")
        grounding = S.TaskGrounding(model_path=args.grounding_model, device=device, box_threshold=args.grounding_box_threshold)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox_sequence = S.build_smoothed_bbox_sequence(
        grounding,
        samples,
        args.target_label,
        write_json=out_dir / "bbox_sequence.json" if grounding is not None else None,
    )

    print(f"[curve] tracing {len(samples)} steps × 5 conditions ...")
    raw_scores = trace_curve(
        torch, model, processor, grounding, samples,
        candidate_heads, random_heads, all_h,
        target_label=args.target_label, swap_bias=args.swap_bias,
        spatial_merge_size=spatial_merge_size, rng=rng, dtype=dtype,
        bbox_sequence=bbox_sequence,
        wrong_region_samples=args.wrong_region_samples,
    )

    eval_mode = infer_eval_mode(samples, args.sample_json)
    progress: Dict[str, List[Optional[float]]] = {c: scores_to_progress(raw_scores[c], eval_mode) for c in raw_scores}
    frames = [parse_after_frame(s["id"]) or (i * 20) for i, s in enumerate(samples)]

    # CSV
    csv_path = out_dir / "curve.csv"
    with csv_path.open("w") as f:
        f.write("step,frame," + ",".join(f"{c}_raw,{c}_progress" for c in raw_scores) + "\n")
        for i, fr in enumerate(frames):
            row = [str(i), str(fr)]
            for c in raw_scores:
                r = raw_scores[c][i]
                p = progress[c][i]
                row.append("" if r is None else f"{r:.4f}")
                row.append("" if p is None else f"{p:.4f}")
            f.write(",".join(row) + "\n")

    # JSON
    json_path = out_dir / "curve.json"
    json_path.write_text(json.dumps({
        "args": vars(args),
        "original_task": original_task,
        "inference_task": samples[0].get("task") if samples else None,
        "eval_mode": eval_mode,
        "candidate_heads": [[c.layer, c.head] for c in candidate_heads],
        "random_control": random_control_meta,
        "random_control_heads": [[h.layer, h.head] for h in random_heads],
        "frames": frames,
        "bbox_sequence": [
            grounding_box_to_record(
                box,
                S.target_image_paths_for_samples(samples, args.target_label)[i] or "",
            )
            for i, box in enumerate(bbox_sequence)
        ],
        "raw_scores": raw_scores,
        "progress": progress,
        "final_progress_pct": {c: (progress[c][-1] * 100 if progress[c] and progress[c][-1] is not None else None) for c in progress},
    }, indent=2))

    # PNG
    png_path = out_dir / "progress_curve.png"
    plot_curves(frames, raw_scores, progress, samples[0]["task"], png_path, fps=args.fps)

    print(f"\n[curve] wrote:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {png_path}")
    print(f"\n[curve] final accumulated progress:")
    for c in ["baseline", "candidate_target", "candidate_wrong", "random_target", "all_target"]:
        p = progress[c][-1] if progress[c] else None
        print(f"  {c:18s} = {(p*100 if p is not None else float('nan')):.1f}%")


if __name__ == "__main__":
    main()
