#!/usr/bin/env python3
"""Analyze top-100 bbox-ranked GRM heads across tasks, modes, and samples.

This is an offline analysis helper: it consumes rank_heads_by_bbox.py outputs
and does not load GRM.  It reports:

  - top-100 layer distributions for every task/mode/query ranking
  - pairwise top-100 head overlap across rankings
  - within-ranking per-sample top-100 stability
  - whether tail top-100 heads still carry target-bbox attention mass compared
    with low-ranked controls
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np


RANK_DIR_RE = re.compile(r"^rank_(?P<task>.+?)_(?P<mode>forward|incremental|backward)_(?P<query>last_prompt|decode)$")


def head_key(row: dict) -> tuple[int, int]:
    return int(row["layer"]), int(row["head"])


def fmt_head(pair: tuple[int, int]) -> str:
    return f"L{pair[0]}H{pair[1]}"


def parse_rank_name(path: Path) -> dict:
    m = RANK_DIR_RE.match(path.parent.name)
    if not m:
        return {"task": "unknown", "mode": "unknown", "query": "unknown", "name": path.parent.name}
    d = m.groupdict()
    d["name"] = path.parent.name
    return d


def get_ranking_rows(data: dict, ranking: str | None) -> list[dict]:
    ranking_name = ranking or data.get("default_ranking") or "mean"
    rows = (data.get("rankings") or {}).get(ranking_name)
    if rows is None:
        rows = data.get("top_heads") or []
    skip = int(data.get("skip_early_layers", 0))
    return [r for r in rows if int(r["layer"]) >= skip]


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(mean(vals)) if vals else float("nan")


def safe_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(median(vals)) if vals else float("nan")


def describe(values: list[float]) -> dict:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def sample_mass_stack(data: dict) -> np.ndarray | None:
    valid = [ps for ps in data.get("per_sample", []) if ps.get("mass") is not None]
    if not valid:
        return None
    return np.asarray([ps["mass"] for ps in valid], dtype=np.float64)


def per_sample_top_set(arr: np.ndarray, k: int, skip_layers: int = 0) -> set[tuple[int, int]]:
    masked = np.array(arr, copy=True)
    if skip_layers > 0:
        masked[:skip_layers, :] = -np.inf
    flat = masked.reshape(-1)
    k = max(1, min(int(k), flat.size))
    # argpartition is enough; exact order is irrelevant for set overlap.
    idxs = np.argpartition(flat, -k)[-k:]
    n_heads = arr.shape[1]
    return {(int(i // n_heads), int(i % n_heads)) for i in idxs if np.isfinite(flat[i])}


def top5_threshold(arr: np.ndarray, skip_layers: int = 0) -> float:
    masked = np.array(arr, copy=True)
    if skip_layers > 0:
        masked[:skip_layers, :] = -np.inf
    flat = masked.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan")
    k = max(1, int(0.05 * flat.size))
    return float(np.partition(flat, -k)[-k])


def bin_rows(rows: list[dict], n_total: int, low_rank_fraction: float) -> dict[str, list[dict]]:
    bins = {
        "rank_001_008": rows[:8],
        "rank_009_032": rows[8:32],
        "rank_033_064": rows[32:64],
        "rank_065_100": rows[64:100],
        "rank_101_128": rows[100:128],
    }
    low_n = max(1, int(math.ceil(n_total * max(0.0, min(float(low_rank_fraction), 1.0)))))
    bins[f"bottom_{int(low_rank_fraction * 100):02d}pct"] = rows[-low_n:]
    return bins


def analyze_one(path: Path, top_k: int, ranking: str | None, low_rank_fraction: float) -> dict:
    data = json.loads(path.read_text())
    meta = parse_rank_name(path)
    rows = get_ranking_rows(data, ranking)
    top_rows = rows[:top_k]
    top_set = {head_key(r) for r in top_rows}
    skip = int(data.get("skip_early_layers", 0))
    n_heads = int(data.get("num_heads", 32))
    n_ranked = len(rows)

    layer_counts: dict[int, int] = {}
    for l, _h in top_set:
        layer_counts[l] = layer_counts.get(l, 0) + 1

    stack = sample_mass_stack(data)
    sample_overlap_rows = []
    mass_bin_rows = []
    if stack is not None:
        for si, arr in enumerate(stack):
            sample_top = per_sample_top_set(arr, top_k, skip_layers=skip)
            inter = len(top_set & sample_top)
            sample_overlap_rows.append({
                **meta,
                "sample_index": si,
                "sample_top100_overlap_count": inter,
                "sample_top100_overlap_frac": inter / max(1, top_k),
                "sample_top100_jaccard": inter / max(1, len(top_set | sample_top)),
            })

        thresholds = [top5_threshold(arr, skip_layers=skip) for arr in stack]
        for bin_name, bin_members in bin_rows(rows, n_ranked, low_rank_fraction).items():
            pairs = [head_key(r) for r in bin_members]
            masses = []
            hit_count = 0
            total_count = 0
            positive_1e4 = 0
            positive_1e3 = 0
            for arr, thr in zip(stack, thresholds):
                for l, h in pairs:
                    if l < arr.shape[0] and h < arr.shape[1]:
                        v = float(arr[l, h])
                        masses.append(v)
                        total_count += 1
                        if v >= thr:
                            hit_count += 1
                        if v > 1e-4:
                            positive_1e4 += 1
                        if v > 1e-3:
                            positive_1e3 += 1
            d = describe(masses)
            mass_bin_rows.append({
                **meta,
                "bin": bin_name,
                "n_heads": len(pairs),
                "mass_mean": d["mean"],
                "mass_median": d["median"],
                "mass_min": d["min"],
                "mass_max": d["max"],
                "sample_top5_hit_rate": hit_count / max(1, total_count),
                "frac_mass_gt_1e-4": positive_1e4 / max(1, total_count),
                "frac_mass_gt_1e-3": positive_1e3 / max(1, total_count),
            })

    return {
        "meta": meta,
        "path": str(path),
        "top_heads": top_rows,
        "top_set": top_set,
        "layer_counts": layer_counts,
        "n_ranked": n_ranked,
        "n_valid_samples": int(data.get("n_valid_samples", 0)),
        "sample_overlap_rows": sample_overlap_rows,
        "mass_bin_rows": mass_bin_rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze top-100 GRM bbox heads from existing ranking JSONs")
    ap.add_argument("--root", default="./results/attention/corresponding_success_20260708")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--ranking", default=None, choices=["mean", "max", "median", "selection_frequency"])
    ap.add_argument("--low-rank-fraction", type=float, default=0.25)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir) if args.output_dir else root / f"top{args.top_k}_head_analysis"
    paths = sorted(root.glob("rank_*/*head_ranking.json"))
    paths = [p for p in paths if RANK_DIR_RE.match(p.parent.name)]
    if not paths:
        raise FileNotFoundError(f"No rank_*/*head_ranking.json files found under {root}")

    analyses = [analyze_one(p, args.top_k, args.ranking, args.low_rank_fraction) for p in paths]

    layer_rows = []
    top_head_rows = []
    sample_overlap_rows = []
    mass_bin_rows = []
    for a in analyses:
        meta = a["meta"]
        for layer, count in sorted(a["layer_counts"].items()):
            layer_rows.append({**meta, "layer": layer, "count": count})
        for rank, row in enumerate(a["top_heads"], start=1):
            top_head_rows.append({
                **meta,
                "rank": rank,
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "score": float(row["score"]),
            })
        sample_overlap_rows.extend(a["sample_overlap_rows"])
        mass_bin_rows.extend(a["mass_bin_rows"])

    pairwise_rows = []
    for i, a in enumerate(analyses):
        for b in analyses[i + 1:]:
            aset = a["top_set"]
            bset = b["top_set"]
            inter = len(aset & bset)
            pairwise_rows.append({
                "name_a": a["meta"]["name"],
                "task_a": a["meta"]["task"],
                "mode_a": a["meta"]["mode"],
                "query_a": a["meta"]["query"],
                "name_b": b["meta"]["name"],
                "task_b": b["meta"]["task"],
                "mode_b": b["meta"]["mode"],
                "query_b": b["meta"]["query"],
                "overlap_count": inter,
                "overlap_frac_of_top_k": inter / max(1, args.top_k),
                "jaccard": inter / max(1, len(aset | bset)),
            })

    # Aggregate summaries by useful groupings.
    group_rows = []
    groupings = {
        "all_pairs": lambda r: True,
        "same_task": lambda r: r["task_a"] == r["task_b"],
        "same_mode": lambda r: r["mode_a"] == r["mode_b"],
        "same_query": lambda r: r["query_a"] == r["query_b"],
        "same_task_mode": lambda r: r["task_a"] == r["task_b"] and r["mode_a"] == r["mode_b"],
        "same_task_query": lambda r: r["task_a"] == r["task_b"] and r["query_a"] == r["query_b"],
        "same_mode_query": lambda r: r["mode_a"] == r["mode_b"] and r["query_a"] == r["query_b"],
    }
    for name, pred in groupings.items():
        rows = [r for r in pairwise_rows if pred(r)]
        group_rows.append({
            "group": name,
            "n_pairs": len(rows),
            "overlap_count_mean": safe_mean(r["overlap_count"] for r in rows),
            "overlap_count_median": safe_median(r["overlap_count"] for r in rows),
            "overlap_frac_mean": safe_mean(r["overlap_frac_of_top_k"] for r in rows),
            "jaccard_mean": safe_mean(r["jaccard"] for r in rows),
        })

    query_mode_rows = []
    for query in ("last_prompt", "decode"):
        rows = [r for r in pairwise_rows if r["query_a"] == query and r["query_b"] == query]
        query_mode_rows.append({
            "query": query,
            "n_pairs": len(rows),
            "overlap_count_mean": safe_mean(r["overlap_count"] for r in rows),
            "overlap_count_median": safe_median(r["overlap_count"] for r in rows),
            "jaccard_mean": safe_mean(r["jaccard"] for r in rows),
        })
    for mode in ("forward", "incremental", "backward"):
        rows = [r for r in pairwise_rows if r["mode_a"] == mode and r["mode_b"] == mode]
        query_mode_rows.append({
            "query": f"mode:{mode}",
            "n_pairs": len(rows),
            "overlap_count_mean": safe_mean(r["overlap_count"] for r in rows),
            "overlap_count_median": safe_median(r["overlap_count"] for r in rows),
            "jaccard_mean": safe_mean(r["jaccard"] for r in rows),
        })

    # Union/core statistics.
    all_top_sets = [a["top_set"] for a in analyses]
    union = set().union(*all_top_sets)
    counts: dict[tuple[int, int], int] = {}
    for s in all_top_sets:
        for pair in s:
            counts[pair] = counts.get(pair, 0) + 1
    recurring = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    core_rows = [
        {"head": fmt_head(pair), "layer": pair[0], "head_index": pair[1], "n_rankings_top100": count}
        for pair, count in recurring
    ]

    write_csv(out_dir / "top100_heads.csv", top_head_rows)
    write_csv(out_dir / "layer_distribution.csv", layer_rows)
    write_csv(out_dir / "pairwise_overlap.csv", pairwise_rows)
    write_csv(out_dir / "pairwise_overlap_summary.csv", group_rows + query_mode_rows)
    write_csv(out_dir / "per_sample_overlap.csv", sample_overlap_rows)
    write_csv(out_dir / "mass_bins.csv", mass_bin_rows)
    write_csv(out_dir / "recurring_heads.csv", core_rows)

    report = {
        "args": vars(args),
        "n_rankings": len(analyses),
        "rankings": [{**a["meta"], "path": a["path"], "n_valid_samples": a["n_valid_samples"]} for a in analyses],
        "top100_union_size": len(union),
        "heads_in_all_rankings": [fmt_head(pair) for pair, count in recurring if count == len(analyses)],
        "heads_in_at_least_half_rankings": [fmt_head(pair) for pair, count in recurring if count >= math.ceil(len(analyses) / 2)],
        "pairwise_overlap_summary": group_rows,
        "query_mode_overlap_summary": query_mode_rows,
        "sample_overlap_summary": {
            "mean_overlap_frac": safe_mean(r["sample_top100_overlap_frac"] for r in sample_overlap_rows),
            "median_overlap_frac": safe_median(r["sample_top100_overlap_frac"] for r in sample_overlap_rows),
            "mean_jaccard": safe_mean(r["sample_top100_jaccard"] for r in sample_overlap_rows),
        },
        "mass_bin_summary": mass_bin_rows,
        "outputs": {
            "top100_heads_csv": str(out_dir / "top100_heads.csv"),
            "layer_distribution_csv": str(out_dir / "layer_distribution.csv"),
            "pairwise_overlap_csv": str(out_dir / "pairwise_overlap.csv"),
            "pairwise_overlap_summary_csv": str(out_dir / "pairwise_overlap_summary.csv"),
            "per_sample_overlap_csv": str(out_dir / "per_sample_overlap.csv"),
            "mass_bins_csv": str(out_dir / "mass_bins.csv"),
            "recurring_heads_csv": str(out_dir / "recurring_heads.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))

    print(f"[top100-analysis] rankings={len(analyses)} top{args.top_k}_union={len(union)}")
    print(f"[top100-analysis] sample overlap mean={report['sample_overlap_summary']['mean_overlap_frac']:.3f}")
    print(f"[top100-analysis] wrote {out_dir}")


if __name__ == "__main__":
    main()
