#!/usr/bin/env python3
"""Summarize target-only bbox success rerun outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


RANK_RE = re.compile(r"rank_(?P<task>.+?)_(?P<mode>forward|incremental|backward)_(?P<query>last_prompt|decode)$")
CURVE_RE = re.compile(
    r"curve_data-(?P<data>.+?)_instr-(?P<instr>.+?)_"
    r"(?P<mode>forward|incremental|backward)_"
    r"(?P<query>last_prompt|decode)_top(?P<topk>\d+)_full$"
)


def safe_mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    kept = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(mean(kept)) if kept else None


def last_non_none(vals: list[Optional[float]]) -> Optional[float]:
    for v in reversed(vals):
        if v is not None:
            return float(v)
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rankings(root: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    top_rows = []
    for path in sorted(root.glob("rank_*/*head_ranking.json")):
        match = RANK_RE.match(path.parent.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        meta = match.groupdict()
        top = data.get("top_heads", [])[:100]
        per_sample = data.get("per_sample", [])
        valid = [ps for ps in per_sample if ps.get("mass") is not None]
        labels = {}
        for ps in valid:
            label = ps.get("box_label") or ""
            labels[label] = labels.get(label, 0) + 1
        scores = [float(r["score"]) for r in top]
        rows.append({
            **meta,
            "path": str(path),
            "n_samples": data.get("n_samples"),
            "n_valid_samples": data.get("n_valid_samples"),
            "box_labels": json.dumps(labels, sort_keys=True),
            "top100_score_mean": safe_mean(scores),
            "top100_score_min": min(scores) if scores else None,
            "top100_score_max": max(scores) if scores else None,
            "tail65_100_score_mean": safe_mean(float(r["score"]) for r in top[64:100]),
            "tail65_100_positive_frac": safe_mean(1.0 if float(r["score"]) > 1e-4 else 0.0 for r in top[64:100]),
        })
        for i, r in enumerate(top, start=1):
            top_rows.append({
                **meta,
                "rank": i,
                "layer": int(r["layer"]),
                "head": int(r["head"]),
                "score": float(r["score"]),
            })
    return rows, top_rows


def summarize_curves(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("curve_*_full/curve.json")):
        match = CURVE_RE.match(path.parent.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        meta = match.groupdict()
        raw = data.get("raw_scores", {})
        progress = data.get("progress", {})
        row = {
            **meta,
            "topk": int(meta["topk"]),
            "path": str(path),
            "original_task": data.get("original_task"),
            "inference_task": data.get("inference_task"),
            "eval_mode": data.get("eval_mode"),
            "n_steps": len(data.get("frames", [])),
        }
        bbox_sequence = data.get("bbox_sequence", [])
        if bbox_sequence:
            bbox_valid = sum(1 for b in bbox_sequence if b is not None)
            row["bbox_valid_steps"] = bbox_valid
            row["bbox_none_count"] = len(bbox_sequence) - bbox_valid
            row["bbox_coverage_frac"] = bbox_valid / max(1, len(bbox_sequence))
        else:
            row["bbox_valid_steps"] = None
            row["bbox_none_count"] = None
            row["bbox_coverage_frac"] = None
        for cond in ["baseline", "candidate_target", "candidate_wrong", "random_target", "all_target"]:
            rvals = raw.get(cond, [])
            pvals = progress.get(cond, [])
            row[f"{cond}_raw_mean_pct"] = None if safe_mean(rvals) is None else safe_mean(rvals) * 100
            final = last_non_none(pvals)
            row[f"{cond}_final_progress_pct"] = None if final is None else final * 100
            row[f"{cond}_none_count"] = sum(1 for v in rvals if v is None)
        base_mean = row.get("baseline_raw_mean_pct")
        ct_mean = row.get("candidate_target_raw_mean_pct")
        rt_mean = row.get("random_target_raw_mean_pct")
        cw_mean = row.get("candidate_wrong_raw_mean_pct")
        row["candidate_target_shift_raw_pct"] = None if base_mean is None or ct_mean is None else ct_mean - base_mean
        row["random_target_shift_raw_pct"] = None if base_mean is None or rt_mean is None else rt_mean - base_mean
        row["candidate_wrong_shift_raw_pct"] = None if base_mean is None or cw_mean is None else cw_mean - base_mean
        if row["candidate_target_shift_raw_pct"] is not None and row["random_target_shift_raw_pct"] is not None:
            row["control_gap_abs_raw_pct"] = abs(row["candidate_target_shift_raw_pct"]) - abs(row["random_target_shift_raw_pct"])
        else:
            row["control_gap_abs_raw_pct"] = None
        rows.append(row)
    return rows


def summarize_videos(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("video_*/attention_video_manifest.json")):
        data = json.loads(path.read_text())
        args = data.get("args", {})
        selected = data.get("selected_head", {})
        per_sample = [ps for ps in data.get("per_sample", []) if not ps.get("skipped")]
        b = []
        c = []
        for ps in per_sample:
            conds = ps.get("conditions", {})
            if "baseline" in conds:
                b.append(float(conds["baseline"].get("bbox_mass", float("nan"))))
            if "candidate_target" in conds:
                c.append(float(conds["candidate_target"].get("bbox_mass", float("nan"))))
        bm = safe_mean(b)
        cm = safe_mean(c)
        rows.append({
            "name": path.parent.name,
            "path": str(path),
            "sample_json": args.get("sample_json"),
            "override_task": args.get("override_task"),
            "top_k": args.get("top_k"),
            "query_mode": args.get("query_mode"),
            "selected_head": selected.get("label"),
            "n_input_samples": data.get("n_input_samples"),
            "n_rendered_baseline": data.get("n_rendered_frames", {}).get("baseline"),
            "n_rendered_candidate": data.get("n_rendered_frames", {}).get("candidate_target"),
            "baseline_bbox_mass_mean": bm,
            "candidate_bbox_mass_mean": cm,
            "candidate_over_baseline_ratio": None if bm in (None, 0.0) or cm is None else cm / bm,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize target-only bbox rerun outputs")
    ap.add_argument("--root", default="results/attention/targetbbox_success_sweep_20260708")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_rows, top_rows = summarize_rankings(root)
    curve_rows = summarize_curves(root)
    video_rows = summarize_videos(root)

    write_csv(out_dir / "ranking_summary.csv", rank_rows)
    write_csv(out_dir / "ranking_top100.csv", top_rows)
    write_csv(out_dir / "curve_summary.csv", curve_rows)
    write_csv(out_dir / "video_summary.csv", video_rows)

    analysis_path = root / "top100_head_analysis" / "summary.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    summary = {
        "root": str(root),
        "n_rankings": len(rank_rows),
        "n_curves": len(curve_rows),
        "n_videos": len(video_rows),
        "ranking_valid_samples_min": min((int(r["n_valid_samples"]) for r in rank_rows), default=None),
        "ranking_valid_samples_max": max((int(r["n_valid_samples"]) for r in rank_rows), default=None),
        "top100_overlap": {
            "union_size": analysis.get("top100_union_size"),
            "heads_in_all_rankings": analysis.get("heads_in_all_rankings", []),
            "heads_in_at_least_half_rankings_count": len(analysis.get("heads_in_at_least_half_rankings", [])),
            "pairwise_overlap_summary": analysis.get("pairwise_overlap_summary", []),
            "sample_overlap_summary": analysis.get("sample_overlap_summary", {}),
        },
        "curve_bbox_coverage_min": min(
            (float(r["bbox_coverage_frac"]) for r in curve_rows if r.get("bbox_coverage_frac") is not None),
            default=None,
        ),
        "curve_bbox_coverage_below_full": [
            {
                "data": r["data"],
                "instr": r["instr"],
                "query": r["query"],
                "topk": r["topk"],
                "bbox_valid_steps": r.get("bbox_valid_steps"),
                "bbox_none_count": r.get("bbox_none_count"),
                "bbox_coverage_frac": r.get("bbox_coverage_frac"),
            }
            for r in curve_rows
            if r.get("bbox_coverage_frac") is not None and float(r["bbox_coverage_frac"]) < 1.0
        ],
        "curve_outputs_csv": str(out_dir / "curve_summary.csv"),
        "ranking_outputs_csv": str(out_dir / "ranking_summary.csv"),
        "video_outputs_csv": str(out_dir / "video_summary.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
