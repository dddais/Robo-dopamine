#!/usr/bin/env python3
"""Derive conservative conflict-gated success/fail head branches."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .auto_explore_causal_rank import _load_candidate_pool, _load_fallback_rows, _section
from .config import load_config
from .io import object_fingerprint, read_jsonl, write_json


def latest(paths: list[Path], prefix: str) -> dict[tuple[int, int, str], dict]:
    result = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("status") == "ok" and str(row.get("example_id", "")).startswith(prefix):
                result[(int(row["layer"]), int(row["head"]), str(row["video_sha256"]))] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    _attention, causal = _section(load_config(args.config))

    positive_path = Path(causal["positive_contrastive_ranking_path"]).resolve()
    positive_data = json.loads(positive_path.read_text(encoding="utf-8"))
    positive_rows = [
        dict(row) for row in positive_data["ranking"]
        if row.get("success_target_correct_margin_delta") is not None
    ]
    positive_by_head = {(int(row["layer"]), int(row["head"])): row for row in positive_rows}
    negative_candidates = _load_candidate_pool(causal)
    success_dir = Path(causal["negative_success_profile_dir"]).resolve()
    fail_dir = Path(causal["negative_fail_profile_dir"]).resolve()
    shards = int(causal.get("negative_profile_num_head_shards", 1))
    negative_success = latest(
        [success_dir / f"causal_profile.head-shard-{i:02d}-of-{shards:02d}.jsonl" for i in range(shards)],
        "suc/",
    )
    negative_fail = latest(
        [fail_dir / f"causal_profile.head-shard-{i:02d}-of-{shards:02d}.jsonl" for i in range(shards)],
        "fail/",
    )

    directional = []
    for row in positive_rows:
        directional.append({**row, "steering_multiplier": 1.0, "validated_profile_bias": 6.0})
    for evidence in negative_candidates:
        key = (int(evidence["layer"]), int(evidence["head"]))
        suc = {k[2]: row for k, row in negative_success.items() if k[:2] == key}
        fail = {k[2]: row for k, row in negative_fail.items() if k[:2] == key}
        videos = sorted(set(suc) & set(fail))
        if len(videos) != 32:
            raise RuntimeError(f"Negative direction {key} has {len(videos)}/32 pairs")
        directional.append(
            {
                "layer": key[0],
                "head": key[1],
                "source_ranks": evidence["source_ranks"],
                "best_source_rank": evidence["best_source_rank"],
                "success_target_correct_margin_delta": mean(float(suc[v]["target_margin_delta"]) for v in videos),
                "fail_target_correct_margin_delta": mean(-float(fail[v]["target_margin_delta"]) for v in videos),
                "success_spatial_correct_margin_delta": mean(float(suc[v]["spatial_margin_delta"]) for v in videos),
                "fail_spatial_correct_margin_delta": mean(-float(fail[v]["spatial_margin_delta"]) for v in videos),
                "steering_multiplier": -1.0,
                "validated_profile_bias": -6.0,
            }
        )

    profiled_keys = set(positive_by_head)
    fallback_rows = []
    for row in _load_fallback_rows(causal):
        key = (int(row["layer"]), int(row["head"]))
        if key in profiled_keys:
            continue
        fallback_rows.append(
            {
                "layer": key[0],
                "head": key[1],
                "source_ranks": {"raw_mass": int(row.get("rank", len(fallback_rows) + 1))},
                "best_source_rank": int(row.get("rank", len(fallback_rows) + 1)),
                "steering_multiplier": 1.0,
                "fallback_only": True,
            }
        )

    def branch(side: str) -> tuple[list[dict], int]:
        target_field = f"{side}_target_correct_margin_delta"
        spatial_field = f"{side}_spatial_correct_margin_delta"
        best = {}
        for row in directional:
            score = float(row[target_field]) + float(row[spatial_field])
            key = (int(row["layer"]), int(row["head"]))
            previous = best.get(key)
            if previous is None or score > previous[0]:
                best[key] = (score, row)
        safe = []
        harmful = []
        for score, row in best.values():
            item = dict(row)
            target = float(item[target_field])
            spatial = float(item[spatial_field])
            item["branch_side"] = side
            item["branch_score"] = score
            item["branch_safe"] = target > 0 and spatial > 0
            (safe if item["branch_safe"] else harmful).append(item)
        safe.sort(key=lambda row: (-row["branch_score"], row["best_source_rank"], row["layer"], row["head"]))
        harmful.sort(key=lambda row: (-row["branch_score"], row["best_source_rank"], row["layer"], row["head"]))
        if len(safe) < 8:
            raise RuntimeError(f"{side} branch has only {len(safe)} safe heads")
        maximum = max(float(row[target_field]) + float(row[spatial_field]) for row in safe)
        floor = 0.1
        for row in safe:
            magnitude = max(
                floor,
                math.sqrt((float(row[target_field]) + float(row[spatial_field])) / maximum),
            )
            row["steering_multiplier"] = (
                -magnitude if float(row["steering_multiplier"]) < 0 else magnitude
            )
        need = max(0, 64 - len(safe))
        start = len(fallback_rows) - 64 - need
        padding_source = fallback_rows[start : start + need]
        padding_keys = {(row["layer"], row["head"]) for row in padding_source}
        padding = [
            {
                **row,
                "branch_side": side,
                "branch_safe": False,
                "safe_padding": True,
                "steering_multiplier": floor,
            }
            for row in padding_source
        ]
        remaining = [
            row for row in fallback_rows if (row["layer"], row["head"]) not in padding_keys
        ]
        ordered = [*safe, *padding, *harmful, *remaining]
        if len(ordered) != 896 or len({(r["layer"], r["head"]) for r in ordered}) != 896:
            raise RuntimeError(f"{side} branch is not a unique 896-head ranking")
        return [{**row, "rank": index} for index, row in enumerate(ordered, start=1)], len(safe)

    success_ranking, success_safe = branch("success")
    fail_ranking, fail_safe = branch("fail")

    probe_data = json.loads(Path(causal["probe_ranking_path"]).resolve().read_text(encoding="utf-8"))
    probe = [dict(row) for row in probe_data["ranking"][:8]]
    positive_success_paths = [
        Path(causal["positive_success_profile_dir"]).resolve()
        / f"causal_profile.head-shard-{i:02d}-of-{int(causal.get('positive_profile_num_head_shards', 2)):02d}.jsonl"
        for i in range(int(causal.get("positive_profile_num_head_shards", 2)))
    ]
    positive_fail_paths = [
        Path(causal["positive_fail_profile_dir"]).resolve()
        / f"causal_profile.head-shard-{i:02d}-of-{int(causal.get('positive_profile_num_head_shards', 2)):02d}.jsonl"
        for i in range(int(causal.get("positive_profile_num_head_shards", 2)))
    ]
    positive_success = latest(positive_success_paths, "suc/")
    positive_fail = latest(positive_fail_paths, "fail/")
    videos = sorted({key[2] for key in positive_success} & {key[2] for key in positive_fail})
    if len(videos) != 32:
        raise RuntimeError(f"Probe router has {len(videos)}/32 paired videos")
    features, labels, groups = [], [], []
    for group, video in enumerate(videos):
        for side, pos, neg, outcome in (
            ("success", positive_success, negative_success, 1.0),
            ("fail", positive_fail, negative_fail, -1.0),
        ):
            values = []
            for head in probe:
                source = pos if float(head["steering_multiplier"]) > 0 else neg
                row = source[(int(head["layer"]), int(head["head"]), video)]
                values.append(abs(float(head["steering_multiplier"])) * float(row["spatial_margin_delta"]))
            features.append(values)
            labels.append(outcome)
            groups.append(group)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    group_ids = np.asarray(groups, dtype=np.int64)

    def fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = x[train].mean(axis=0)
        scale = np.maximum(x[train].std(axis=0), 1e-6)
        design = np.column_stack([np.ones(train.sum()), (x[train] - center) / scale])
        penalty = np.diag([0.0] + [1.0] * x.shape[1])
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y[train])
        return center, scale, coefficients

    out_of_fold = np.zeros(len(y), dtype=np.float64)
    for group in range(32):
        train = group_ids != group
        test = ~train
        center, scale, coefficients = fit(train)
        design = np.column_stack([np.ones(test.sum()), (x[test] - center) / scale])
        out_of_fold[test] = design @ coefficients
    full = np.ones(len(y), dtype=bool)
    center, scale, coefficients = fit(full)
    full_scores = np.column_stack([np.ones(len(y)), (x - center) / scale]) @ coefficients
    threshold = min(
        float(out_of_fold[y > 0].min()),
        float(full_scores[y > 0].min()),
    ) - 0.05
    router = {
        "schema_version": "ridge-spatial-conflict-router-v1",
        "feature_definition": "per-probe-head target_minus_wrong reward5 margin at explicit signed bias6, multiplied by abs development weight",
        "probe_bias": 6.0,
        "probe_heads": probe,
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients_with_intercept": coefficients.tolist(),
        "ridge_penalty": 1.0,
        "fail_branch_if_score_below": threshold,
        "threshold_rule": "min(full-fit success score, leave-video-out success score)-0.05",
        "leave_video_out_accuracy": float(np.mean(np.sign(out_of_fold) == y)),
        "leave_video_out_success_accuracy": float(np.mean(out_of_fold[y > 0] > 0)),
        "leave_video_out_fail_accuracy": float(np.mean(out_of_fold[y < 0] < 0)),
        "leave_video_out_success_below_threshold": int(np.sum(out_of_fold[y > 0] < threshold)),
        "leave_video_out_fail_below_threshold": int(np.sum(out_of_fold[y < 0] < threshold)),
        "development_labels_model_facing": False,
    }
    artifact = {
        "schema_version": "conflict-gated-branch-ranking-v1",
        "ranking_source": "paired_development_class_specific_signed_causal_heads_with_internal_spatial_router",
        "labels_model_facing": False,
        "inference_uses_labels": False,
        "success_safe_head_count": success_safe,
        "fail_safe_head_count": fail_safe,
        "router": router,
        "success_ranking": success_ranking,
        "fail_ranking": fail_ranking,
        "ranking": success_ranking,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    output = Path(causal["adaptive_output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "adaptive_conflict_ranking.json"
    write_json(path, artifact)
    print(path)


if __name__ == "__main__":
    main()
