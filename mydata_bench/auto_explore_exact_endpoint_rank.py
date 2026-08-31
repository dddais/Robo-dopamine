#!/usr/bin/env python3
"""Derive development-only head rankings for exact reward endpoints.

The earlier causal ranker uses the reward-5-vs-rest margin and reverses its
sign for failed instructions.  That is a useful ordinal objective, but moving
probability away from reward 5 does not specifically move it toward reward 1.
This incremental ranker reuses the already-frozen single-head profiles and
requires a signed head to improve the exact correct-choice margin on both
development sides: 5-vs-{1..4} for success and 1-vs-{2..5} for failure.  It
also requires the target-region effect to exceed the equal-size wrong-region
effect on both sides.  Evaluation examples and labels are never read.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import load_config
from .io import object_fingerprint, read_jsonl, write_json


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _choice_margin(stats: dict[str, Any], label: int) -> float:
    logits = [float(value) for value in stats["choice_logits"]]
    if len(logits) != 5 or label not in {1, 5}:
        raise ValueError("Exact endpoint margin requires five logits and label 1 or 5")
    index = label - 1
    alternatives = logits[:index] + logits[index + 1 :]
    return logits[index] - _logsumexp(alternatives)


def _task_balanced(rows: list[dict[str, Any]], field: str) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[str(row["subset"])].append(float(row[field]))
    if not by_task:
        raise ValueError("Cannot aggregate an empty profile")
    return mean(mean(values) for values in by_task.values())


def _read_profile_dir(
    directory: Path,
    *,
    shard_count: int,
    expected_side: str,
    label: int,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, int, str]] = set()
    for shard_id in range(shard_count):
        path = directory / f"causal_profile.head-shard-{shard_id:02d}-of-{shard_count:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            if row.get("status") != "ok":
                raise ValueError(f"Non-ok profile row in {path}: {row.get('status')}")
            if str(row.get("development_side")) != expected_side:
                raise ValueError(f"Unexpected development side in {path}")
            example_id = str(row["example_id"])
            required_prefix = "suc/" if expected_side == "success" else "fail/"
            if not example_id.startswith(required_prefix):
                raise ValueError(f"Profile row has wrong class prefix: {example_id}")
            key = (int(row["layer"]), int(row["head"]))
            unique_key = (*key, example_id)
            if unique_key in seen:
                raise ValueError(f"Duplicate profile key: {unique_key}")
            seen.add(unique_key)
            item = dict(row)
            baseline_margin = _choice_margin(item["baseline"], label)
            target_margin = _choice_margin(item["target"], label)
            wrong_margin = _choice_margin(item["wrong"], label)
            item["exact_target_correct_margin_delta"] = target_margin - baseline_margin
            item["exact_spatial_correct_margin_delta"] = target_margin - wrong_margin
            grouped[key].append(item)
    counts = {len(rows) for rows in grouped.values()}
    if len(counts) != 1:
        raise ValueError(f"Incomplete per-head profiles in {directory}: counts={sorted(counts)}")
    return grouped


def _direction_rows(
    success: dict[tuple[int, int], list[dict[str, Any]]],
    failure: dict[tuple[int, int], list[dict[str, Any]]],
    *,
    direction: int,
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(success) & set(failure)):
        success_target = _task_balanced(success[key], "exact_target_correct_margin_delta")
        success_spatial = _task_balanced(success[key], "exact_spatial_correct_margin_delta")
        fail_target = _task_balanced(failure[key], "exact_target_correct_margin_delta")
        fail_spatial = _task_balanced(failure[key], "exact_spatial_correct_margin_delta")
        weakest = min(success_target, fail_target, success_spatial, fail_spatial)
        target_floor = min(success_target, fail_target)
        spatial_floor = min(success_spatial, fail_spatial)
        source = success[key][0]
        rows.append(
            {
                "layer": key[0],
                "head": key[1],
                "steering_multiplier": float(direction),
                "validated_profile_bias": abs(float(source["bias"])),
                "development_success_count": len(success[key]),
                "development_fail_count": len(failure[key]),
                "success_exact_target_correct_margin_delta": success_target,
                "fail_exact_target_correct_margin_delta": fail_target,
                "success_exact_spatial_correct_margin_delta": success_spatial,
                "fail_exact_spatial_correct_margin_delta": fail_spatial,
                "weakest_exact_endpoint_effect": weakest,
                "exact_endpoint_causal_score": target_floor + 0.5 * spatial_floor,
                "source_ranks": dict(source.get("source_ranks", {})),
            }
        )
    return rows


def derive(config: dict[str, Any]) -> Path:
    section = config.get("exact_endpoint_rank")
    if not isinstance(section, dict):
        raise ValueError("Expected exact_endpoint_rank configuration section")
    positive_shards = int(section.get("positive_profile_num_head_shards", 2))
    negative_shards = int(section.get("negative_profile_num_head_shards", 1))
    positive_success = _read_profile_dir(
        Path(section["positive_success_profile_dir"]).resolve(),
        shard_count=positive_shards,
        expected_side="success",
        label=5,
    )
    positive_fail = _read_profile_dir(
        Path(section["positive_fail_profile_dir"]).resolve(),
        shard_count=positive_shards,
        expected_side="fail",
        label=1,
    )
    negative_success = _read_profile_dir(
        Path(section["negative_success_profile_dir"]).resolve(),
        shard_count=negative_shards,
        expected_side="success",
        label=5,
    )
    negative_fail = _read_profile_dir(
        Path(section["negative_fail_profile_dir"]).resolve(),
        shard_count=negative_shards,
        expected_side="fail",
        label=1,
    )
    candidates = [
        *_direction_rows(positive_success, positive_fail, direction=1),
        *_direction_rows(negative_success, negative_fail, direction=-1),
    ]
    # A head can appear in both signed pools.  Keep only the stronger measured
    # direction so the final attention hook never receives duplicate heads.
    best_by_head: dict[tuple[int, int], dict[str, Any]] = {}
    for row in candidates:
        key = (int(row["layer"]), int(row["head"]))
        prior = best_by_head.get(key)
        if prior is None or (
            float(row["weakest_exact_endpoint_effect"]),
            float(row["exact_endpoint_causal_score"]),
        ) > (
            float(prior["weakest_exact_endpoint_effect"]),
            float(prior["exact_endpoint_causal_score"]),
        ):
            best_by_head[key] = row
    safe = [
        row for row in best_by_head.values()
        if float(row["weakest_exact_endpoint_effect"]) > 0
    ]
    safe.sort(
        key=lambda row: (
            -float(row["weakest_exact_endpoint_effect"]),
            -float(row["exact_endpoint_causal_score"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    minimum = int(section.get("minimum_safe_heads", 8))
    top_k = int(section.get("safe_padding_top_k", 64))
    if not 8 <= minimum <= top_k:
        raise ValueError("minimum_safe_heads must be between 8 and safe_padding_top_k")
    if len(safe) < minimum:
        raise RuntimeError(f"Only {len(safe)} exact-endpoint-safe heads; need {minimum}")
    safe = safe[:top_k]

    fallback_path = Path(section["fallback_ranking_path"]).resolve()
    fallback_data = json.loads(fallback_path.read_text(encoding="utf-8"))
    fallback_rows = fallback_data.get("ranking")
    if not isinstance(fallback_rows, list) or len(fallback_rows) < top_k * 2:
        raise ValueError(f"Incomplete fallback ranking at {fallback_path}")
    profiled = set(best_by_head)
    available = [
        dict(row) for row in fallback_rows
        if (int(row["layer"]), int(row["head"])) not in profiled
    ]
    need = top_k - len(safe)
    if len(available) < need + top_k:
        raise RuntimeError("Insufficient fallback heads while reserving low-rank control")
    start = len(available) - top_k - need
    padding_multiplier = float(section.get("padding_multiplier", 0.1))
    if not 0 < padding_multiplier <= 1:
        raise ValueError("padding_multiplier must be in (0, 1]")
    padding = []
    for row in available[start : start + need]:
        padding.append(
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "steering_multiplier": padding_multiplier,
                "safe_padding": True,
                "padding_basis": "lowest_fallback_rank_reserving_disjoint_tail_control",
                "source_ranks": dict(row.get("source_ranks", {})),
            }
        )
    selected = {(int(row["layer"]), int(row["head"])) for row in [*safe, *padding]}
    remaining = [
        dict(row) for row in fallback_rows
        if (int(row["layer"]), int(row["head"])) not in selected
    ]
    ranking = [
        {**row, "rank": index}
        for index, row in enumerate([*safe, *padding, *remaining], start=1)
    ]
    keys = [(int(row["layer"]), int(row["head"])) for row in ranking]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Exact endpoint ranking contains duplicate heads")
    artifact = {
        "schema_version": "paired-exact-endpoint-causal-safe-padding-ranking-v1",
        "ranking_source": "existing_development_single_head_profiles_only",
        "ranking_method_detail": (
            "signed heads must improve task-balanced 5-vs-rest success margin, "
            "1-vs-rest fail margin, and target-minus-wrong spatial effect on both sides; "
            "no screening or held-out rows read; full-strength safe heads plus frozen low-mass padding"
        ),
        "labels_model_facing": False,
        "development_labels_used_for_head_selection": True,
        "inference_uses_labels": False,
        "success_endpoint": 5,
        "fail_endpoint": 1,
        "positive_profiled_head_count": len(positive_success),
        "negative_profiled_head_count": len(negative_success),
        "exact_endpoint_safe_head_count": len(safe),
        "safe_positive_head_count": sum(float(row["steering_multiplier"]) > 0 for row in safe),
        "safe_negative_head_count": sum(float(row["steering_multiplier"]) < 0 for row in safe),
        "safe_padding_head_count": len(padding),
        "padding_multiplier": padding_multiplier,
        "fallback_ranking_fingerprint": str(fallback_data.get("fingerprint")),
        "ranking": ranking,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    output = Path(section["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "exact_endpoint_causal_ranking.json"
    write_json(path, artifact)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(derive(load_config(args.config)))


if __name__ == "__main__":
    main()
