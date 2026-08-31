#!/usr/bin/env python3
"""Calibrate a global adaptive-router threshold on frozen development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import object_fingerprint, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ranking", required=True)
    parser.add_argument("--development-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--success-branch-scale", type=float, default=1.0 / 3.0)
    parser.add_argument("--fail-branch-scale", type=float, default=1.0)
    parser.add_argument("--success-margin", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.success_branch_scale <= 1:
        raise ValueError("success branch scale must be in (0, 1]")
    if not 0 < args.fail_branch_scale <= 2:
        raise ValueError("fail branch scale must be in (0, 2]")
    if args.success_margin <= 0:
        raise ValueError("success margin must be positive")

    source_path = Path(args.source_ranking).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("inference_uses_labels") is not False:
        raise ValueError("Source adaptive artifact must exclude inference labels")
    latest = {}
    for row in read_jsonl(Path(args.development_records).resolve()):
        if row.get("condition") == "baseline" and row.get("status") == "ok":
            latest[str(row["example_id"])] = row
    success_scores = []
    fail_scores = []
    for example_id, row in latest.items():
        gate = row.get("adaptive_gate")
        if not isinstance(gate, dict) or gate.get("inference_uses_labels") is not False:
            raise ValueError("Development baseline is missing a label-free gate")
        score = float(gate["router_score"])
        if example_id.startswith("suc/"):
            success_scores.append(score)
        elif example_id.startswith("fail/"):
            fail_scores.append(score)
        else:
            raise ValueError(f"Unknown development class in {example_id}")
    if not success_scores or not fail_scores:
        raise RuntimeError("Calibration requires both development classes")
    threshold = min(success_scores) - args.success_margin

    success_ranking = []
    for row in source["success_ranking"]:
        item = dict(row)
        item["pre_calibration_steering_multiplier"] = float(
            item.get("steering_multiplier", 1.0)
        )
        item["steering_multiplier"] = (
            item["pre_calibration_steering_multiplier"]
            * args.success_branch_scale
        )
        success_ranking.append(item)
    fail_ranking = []
    for row in source["fail_ranking"]:
        item = dict(row)
        item["pre_calibration_steering_multiplier"] = float(
            item.get("steering_multiplier", 1.0)
        )
        item["steering_multiplier"] = (
            item["pre_calibration_steering_multiplier"] * args.fail_branch_scale
        )
        fail_ranking.append(item)
    router = dict(source["router"])
    router.update(
        {
            "fail_branch_if_score_below": threshold,
            "threshold_rule": "min frozen screening-development success score minus fixed 0.05 margin",
            "screening_development_success_count": len(success_scores),
            "screening_development_fail_count": len(fail_scores),
            "screening_development_success_below_threshold": sum(
                score < threshold for score in success_scores
            ),
            "screening_development_fail_below_threshold": sum(
                score < threshold for score in fail_scores
            ),
            "screening_development_labels_used_for_global_calibration": True,
            "development_labels_model_facing": False,
        }
    )
    artifact = {
        **{
            key: value
            for key, value in source.items()
            if key not in {"fingerprint", "ranking", "router", "success_ranking"}
        },
        "schema_version": (
            "conflict-gated-branch-ranking-v3-development-calibrated-asymmetric-dose"
            if args.fail_branch_scale != 1.0
            else "conflict-gated-branch-ranking-v2-development-calibrated"
        ),
        "ranking_source": "development_calibrated_spatial_conflict_router_with_asymmetric_safe_dose",
        "source_adaptive_fingerprint": source["fingerprint"],
        "router": router,
        "success_branch_scale": args.success_branch_scale,
        "fail_branch_scale": args.fail_branch_scale,
        "success_effective_base_bias": 6.0 * args.success_branch_scale,
        "fail_effective_base_bias": 6.0 * args.fail_branch_scale,
        "success_ranking": success_ranking,
        "fail_ranking": fail_ranking,
        "ranking": success_ranking,
        "inference_uses_labels": False,
        "labels_model_facing": False,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = "v3" if args.fail_branch_scale != 1.0 else "v2"
    path = output / f"adaptive_conflict_ranking_{suffix}.json"
    write_json(path, artifact)
    print(path)


if __name__ == "__main__":
    main()
