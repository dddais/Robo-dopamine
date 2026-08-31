#!/usr/bin/env python3
"""Label-isolated causal head profiling for cross-model auto exploration.

This exploratory tool deliberately operates only on the independent successful
ranking trajectories.  It measures whether steering one head toward the
audited target changes the teacher-forced reward-5 margin, and whether that
change is spatially specific relative to an equal-size wrong-region control.
Evaluation cohort labels and records are never read here.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .attention_eval.masking import Head
from .config import load_config
from .io import append_jsonl, object_fingerprint, read_jsonl, write_json
from .qwen_eval.attention import QwenAttentionRuntime


SCORE_KINDS = ("raw_mass", "excess_mass", "visual_enrichment")
ANSWER_PREFIX = "ANSWER: "


def _section(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    attention = config.get("attention_steer")
    causal = config.get("causal_rank")
    if not isinstance(attention, dict) or not isinstance(causal, dict):
        raise ValueError("Expected attention_steer and causal_rank mappings")
    return attention, causal


def _load_candidate_pool(causal: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_path = causal.get("explicit_candidate_ranking_path")
    if explicit_path is not None:
        data = json.loads(Path(explicit_path).resolve().read_text(encoding="utf-8"))
        field = str(causal.get("explicit_candidate_ranking_field", "ranking"))
        rows = data.get(field)
        if not isinstance(rows, list):
            raise ValueError(f"Explicit candidate artifact has no {field!r} list")
        tail_count = int(causal.get("explicit_candidate_tail_count", 64))
        if tail_count < 8 or len(rows) < tail_count:
            raise ValueError("Explicit candidate tail must contain at least 8 heads")
        selected = rows[-tail_count:]
        if len({(int(row["layer"]), int(row["head"])) for row in selected}) != tail_count:
            raise ValueError("Explicit candidate tail contains duplicate heads")
        fingerprint = str(data.get("fingerprint", object_fingerprint(data)))
        return [
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "source_ranks": {"explicit_tail": len(rows) - tail_count + index},
                "best_source_rank": len(rows) - tail_count + index,
                "source_fingerprints": {"explicit_tail": fingerprint},
            }
            for index, row in enumerate(selected, start=1)
        ]
    directory = Path(causal["source_ranking_dir"]).resolve()
    depth = int(causal.get("candidate_depth_per_metric", 64))
    if depth < 64:
        raise ValueError("candidate_depth_per_metric must be at least 64")
    evidence: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    source_fingerprints = {}
    for kind in SCORE_KINDS:
        path = directory / f"consensus_ranking_{kind}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("ranking")
        if not isinstance(rows, list) or len(rows) < depth:
            raise ValueError(f"Incomplete {kind} ranking at {path}")
        source_fingerprints[kind] = str(data.get("fingerprint", object_fingerprint(data)))
        for index, row in enumerate(rows[:depth], start=1):
            key = (int(row["layer"]), int(row["head"]))
            evidence[key][kind] = index
    candidates = []
    for (layer, head), ranks in evidence.items():
        candidates.append(
            {
                "layer": layer,
                "head": head,
                "source_ranks": dict(sorted(ranks.items())),
                "best_source_rank": min(ranks.values()),
                "mean_missing_worst_rank": mean(
                    ranks.get(kind, depth + 1) for kind in SCORE_KINDS
                ),
                "source_fingerprints": source_fingerprints,
            }
        )
    candidates.sort(
        key=lambda row: (
            row["best_source_rank"],
            row["mean_missing_worst_rank"],
            row["layer"],
            row["head"],
        )
    )
    filter_path = causal.get("candidate_filter_ranking_path")
    if filter_path is not None:
        direction = int(causal.get("candidate_filter_direction", -1))
        if direction not in {-1, 1}:
            raise ValueError("candidate_filter_direction must be -1 or 1")
        filter_data = json.loads(Path(filter_path).resolve().read_text(encoding="utf-8"))
        filter_rows = filter_data.get("ranking")
        if not isinstance(filter_rows, list):
            raise ValueError(f"No ranking list in candidate filter {filter_path}")
        aligned = {
            (int(row["layer"]), int(row["head"]))
            for row in filter_rows
            if row.get("success_target_correct_margin_delta") is not None
            and row.get("fail_target_correct_margin_delta") is not None
            and direction * float(row["success_target_correct_margin_delta"]) > 0
            and direction * float(row["fail_target_correct_margin_delta"]) > 0
        }
        candidates = [
            row for row in candidates if (int(row["layer"]), int(row["head"])) in aligned
        ]
        if not candidates:
            raise ValueError("Candidate effect-sign filter selected no heads")
        return candidates
    if len(candidates) < 64:
        raise ValueError("Candidate union contains fewer than 64 heads")
    return candidates


def _load_fallback_rows(causal: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the complete source ranking for disjoint low-rank controls.

    These rows can only be appended after every causally profiled candidate and
    therefore never enter candidate top-64 selection.
    """
    path = Path(causal["source_ranking_dir"]).resolve() / "consensus_ranking_raw_mass.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("ranking")
    if not isinstance(rows, list) or len(rows) < 128:
        raise ValueError(f"Incomplete fallback ranking at {path}")
    return rows


def _ranking_samples(causal: dict[str, Any]) -> list[dict[str, Any]]:
    ranking_rows = list(read_jsonl(Path(causal["ranking_inputs_file"]).resolve()))
    if bool(causal.get("contrastive_fail_development", False)):
        ranking_videos = {str(row["video_sha256"]) for row in ranking_rows}
        cohort = list(read_jsonl(Path(causal["development_cohort_inputs_file"]).resolve()))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cohort:
            if (
                str(row.get("video_sha256")) in ranking_videos
                and str(row.get("example_id", "")).startswith("fail/")
            ):
                grouped[str(row["video_sha256"])].append(row)
        rows = [
            sorted(grouped[video], key=lambda row: str(row["example_id"]))[0]
            for video in sorted(ranking_videos)
            if grouped.get(video)
        ]
    else:
        rows = ranking_rows
    expected = int(causal.get("expected_success_samples", 34))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} ranking samples, found {len(rows)}")
    forbidden = {"reward", "label", "expected_reward", "native_prediction", "progress"}
    for row in rows:
        required_prefix = (
            "fail/" if bool(causal.get("contrastive_fail_development", False)) else "suc/"
        )
        if not str(row.get("example_id", "")).startswith(required_prefix):
            raise ValueError(f"Causal profiling expected only {required_prefix} development rows")
        leaked = forbidden & set(row)
        if leaked:
            raise ValueError(f"Ranking row contains forbidden outcome fields: {sorted(leaked)}")
    return rows


def _extended_inputs(runtime: QwenAttentionRuntime, prepared) -> dict[str, Any]:
    torch = runtime.torch
    tokenizer = runtime.processor.tokenizer
    prefix_ids = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    if tokenizer.decode(prefix_ids) != ANSWER_PREFIX:
        raise RuntimeError("ANSWER prefix does not round-trip through tokenizer")
    result = dict(prepared.inputs)
    input_ids = result["input_ids"]
    prefix = torch.tensor(prefix_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
    result["input_ids"] = torch.cat([input_ids, prefix], dim=1)
    original_length = int(input_ids.shape[1])
    for key, value in list(result.items()):
        if key == "input_ids" or not torch.is_tensor(value):
            continue
        if value.ndim != 2 or int(value.shape[1]) != original_length:
            continue
        if key == "attention_mask":
            extension = torch.ones(
                (value.shape[0], len(prefix_ids)), dtype=value.dtype, device=value.device
            )
        else:
            extension = value[:, -1:].expand(value.shape[0], len(prefix_ids))
        result[key] = torch.cat([value, extension], dim=1)
    return result


def _stalled_prefix_sample(sample: dict[str, Any], cutoff_index: int) -> dict[str, Any]:
    """Freeze an 8-image success trajectory at an early observed state."""
    paths = list(sample.get("image_paths", []))
    indices = [int(value) for value in sample.get("image_source_indices", [])]
    if len(paths) != 8 or len(indices) != 8 or not 0 <= cutoff_index < 7:
        raise ValueError("Stalled temporal control requires 8 images and cutoff in [0, 6]")
    result = dict(sample)
    result["example_id"] = f"{sample['example_id']}#stalled_t{cutoff_index}"
    result["image_paths"] = [
        path if index <= cutoff_index else paths[cutoff_index]
        for index, path in enumerate(paths)
    ]
    result["image_source_indices"] = [
        value if index <= cutoff_index else indices[cutoff_index]
        for index, value in enumerate(indices)
    ]
    record = dict(sample.get("image_sampling_record", {}))
    record.update(
        {
            "selected_source_indices": result["image_source_indices"],
            "terminal_source_index": indices[cutoff_index],
            "terminal_frame_in_last_image": False,
            "stalled_prefix_control": True,
            "stalled_prefix_cutoff_index": cutoff_index,
            "source_terminal_frame_omitted": True,
        }
    )
    result["image_sampling_record"] = record
    return result


def _choice_stats(runtime: QwenAttentionRuntime, inputs: dict[str, Any]) -> dict[str, Any]:
    torch = runtime.torch
    tokenizer = runtime.processor.tokenizer
    choice_ids = []
    for value in range(1, 6):
        ids = tokenizer.encode(str(value), add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Reward choice {value} is not a single token")
        choice_ids.append(ids[0])
    with torch.inference_mode():
        logits = runtime.model(**inputs, use_cache=False).logits[0, -1, choice_ids].float()
    log_probs = torch.log_softmax(logits, dim=0)
    probabilities = torch.softmax(logits, dim=0)
    margin5 = logits[4] - torch.logsumexp(logits[:4], dim=0)
    expected = sum((index + 1) * probabilities[index] for index in range(5))
    return {
        "choice_logits": [float(value) for value in logits.cpu().tolist()],
        "choice_log_probs": [float(value) for value in log_probs.cpu().tolist()],
        "choice_probabilities": [float(value) for value in probabilities.cpu().tolist()],
        "reward5_margin": float(margin5.cpu()),
        "expected_reward": float(expected.cpu()),
    }


def profile(config: dict[str, Any], *, head_shard_id: int, num_head_shards: int) -> Path:
    attention, causal = _section(config)
    if num_head_shards < 1 or not 0 <= head_shard_id < num_head_shards:
        raise ValueError("Require 0 <= head_shard_id < num_head_shards")
    candidates = _load_candidate_pool(causal)
    selected = [
        row for index, row in enumerate(candidates) if index % num_head_shards == head_shard_id
    ]
    samples = _ranking_samples(causal)
    output = Path(causal["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = output / f"causal_profile.head-shard-{head_shard_id:02d}-of-{num_head_shards:02d}.jsonl"
    fingerprint = object_fingerprint(
        {
            "candidate_pool": candidates,
            "sample_fingerprints": [row.get("sample_fingerprint") for row in samples],
            "bias": float(causal.get("profiling_bias", 6)),
            "query_scope": str(causal.get("query_scope", "all")),
            "negative_scope": str(causal.get("negative_scope", "none")),
            "answer_prefix": ANSWER_PREFIX,
            "temporal_stalled_prefix_index": causal.get("temporal_stalled_prefix_index"),
            "contrastive_fail_development": bool(causal.get("contrastive_fail_development", False)),
        }
    )
    done = {
        (int(row["layer"]), int(row["head"]), str(row["example_id"]))
        for row in read_jsonl(records)
        if row.get("status") == "ok" and row.get("profiling_fingerprint") == fingerprint
    } if records.exists() else set()
    runtime = QwenAttentionRuntime(attention)
    bias = float(causal.get("profiling_bias", 6))
    query_scope = str(causal.get("query_scope", "all"))
    negative_scope = str(causal.get("negative_scope", "none"))
    for sample in samples:
        prepared = runtime.prepare(sample)
        target = runtime.target_positions(sample, prepared)
        wrong, wrong_mode = runtime.wrong_control_positions(prepared, target)
        base_inputs = _extended_inputs(runtime, prepared)
        baseline = _choice_stats(runtime, base_inputs)
        stalled_index = causal.get("temporal_stalled_prefix_index")
        if stalled_index is not None:
            stalled_sample = _stalled_prefix_sample(sample, int(stalled_index))
            stalled_prepared = runtime.prepare(stalled_sample)
            stalled_target = runtime.target_positions(stalled_sample, stalled_prepared)
            stalled_inputs = _extended_inputs(runtime, stalled_prepared)
            stalled_baseline = _choice_stats(runtime, stalled_inputs)
        else:
            stalled_sample = None
            stalled_prepared = None
            stalled_target = None
            stalled_inputs = None
            stalled_baseline = None
        for candidate in selected:
            key = (candidate["layer"], candidate["head"], str(sample["example_id"]))
            if key in done:
                continue
            head = Head(candidate["layer"], candidate["head"])
            try:
                target_diagnostics: dict[str, Any] = {}
                with runtime.steering_hooks(
                    [head], target, prepared.visual_positions, bias, query_scope,
                    negative_scope, prepared.spans, target_diagnostics,
                ):
                    target_stats = _choice_stats(runtime, base_inputs)
                wrong_diagnostics: dict[str, Any] = {}
                with runtime.steering_hooks(
                    [head], wrong, prepared.visual_positions, bias, query_scope,
                    negative_scope, prepared.spans, wrong_diagnostics,
                ):
                    wrong_stats = _choice_stats(runtime, base_inputs)
                if stalled_prepared is not None:
                    stalled_diagnostics: dict[str, Any] = {}
                    assert stalled_target is not None and stalled_inputs is not None
                    assert stalled_baseline is not None
                    with runtime.steering_hooks(
                        [head], stalled_target, stalled_prepared.visual_positions, bias,
                        query_scope, negative_scope, stalled_prepared.spans,
                        stalled_diagnostics,
                    ):
                        stalled_stats = _choice_stats(runtime, stalled_inputs)
                    stalled_margin_delta = (
                        stalled_stats["reward5_margin"]
                        - stalled_baseline["reward5_margin"]
                    )
                    temporal_margin_delta = (
                        target_stats["reward5_margin"]
                        - baseline["reward5_margin"]
                        - stalled_margin_delta
                    )
                    stalled_expected_delta = (
                        stalled_stats["expected_reward"]
                        - stalled_baseline["expected_reward"]
                    )
                    temporal_expected_delta = (
                        target_stats["expected_reward"]
                        - baseline["expected_reward"]
                        - stalled_expected_delta
                    )
                else:
                    stalled_diagnostics = None
                    stalled_stats = None
                    stalled_margin_delta = None
                    temporal_margin_delta = None
                    stalled_expected_delta = None
                    temporal_expected_delta = None
                append_jsonl(
                    records,
                    {
                        "schema_version": "causal-head-profile-v1",
                        "profiling_fingerprint": fingerprint,
                        "example_id": sample["example_id"],
                        "video_sha256": sample["video_sha256"],
                        "subset": sample.get("subset"),
                        "layer": head.layer,
                        "head": head.head,
                        "source_ranks": candidate["source_ranks"],
                        "bias": bias,
                        "query_scope": query_scope,
                        "negative_scope": negative_scope,
                        "control_region": wrong_mode,
                        "baseline": baseline,
                        "target": target_stats,
                        "wrong": wrong_stats,
                        "target_margin_delta": target_stats["reward5_margin"] - baseline["reward5_margin"],
                        "wrong_margin_delta": wrong_stats["reward5_margin"] - baseline["reward5_margin"],
                        "spatial_margin_delta": target_stats["reward5_margin"] - wrong_stats["reward5_margin"],
                        "target_expected_reward_delta": target_stats["expected_reward"] - baseline["expected_reward"],
                        "spatial_expected_reward_delta": target_stats["expected_reward"] - wrong_stats["expected_reward"],
                        "stalled_prefix_index": stalled_index,
                        "stalled_baseline": stalled_baseline,
                        "stalled_target": stalled_stats,
                        "stalled_target_margin_delta": stalled_margin_delta,
                        "temporal_margin_delta": temporal_margin_delta,
                        "stalled_target_expected_reward_delta": stalled_expected_delta,
                        "temporal_expected_reward_delta": temporal_expected_delta,
                        "target_hook_diagnostics": target_diagnostics,
                        "wrong_hook_diagnostics": wrong_diagnostics,
                        "stalled_hook_diagnostics": stalled_diagnostics,
                        "labels_model_facing": False,
                        "development_side": (
                            "fail" if bool(causal.get("contrastive_fail_development", False))
                            else "success"
                        ),
                        "status": "ok",
                    },
                )
            except Exception as exc:
                append_jsonl(
                    records,
                    {
                        "schema_version": "causal-head-profile-v1",
                        "profiling_fingerprint": fingerprint,
                        "example_id": sample["example_id"],
                        "layer": head.layer,
                        "head": head.head,
                        "status": "invalid",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
    return records


def _task_balanced(rows: list[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subset"))].append(float(row[field]))
    return mean(mean(values) for values in grouped.values())


def aggregate(config: dict[str, Any], *, num_head_shards: int) -> Path:
    _attention, causal = _section(config)
    candidates = _load_candidate_pool(causal)
    output = Path(causal["output_dir"]).resolve()
    paths = [
        output / f"causal_profile.head-shard-{index:02d}-of-{num_head_shards:02d}.jsonl"
        for index in range(num_head_shards)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing causal profile shards: {missing}")
    latest: dict[tuple[int, int, str], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            key = (int(row["layer"]), int(row["head"]), str(row["example_id"]))
            latest[key] = row
    expected_samples = int(causal.get("expected_success_samples", 34))
    by_head: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    invalid = []
    for key, row in latest.items():
        if row.get("status") == "ok":
            by_head[key[:2]].append(row)
        else:
            invalid.append(row)
    summaries = []
    evidence_by_head = {(row["layer"], row["head"]): row for row in candidates}
    for key, evidence in evidence_by_head.items():
        rows = by_head.get(key, [])
        if len(rows) != expected_samples:
            raise RuntimeError(f"Head {key} has {len(rows)}/{expected_samples} valid samples")
        target = _task_balanced(rows, "target_margin_delta")
        spatial = _task_balanced(rows, "spatial_margin_delta")
        target_expected = _task_balanced(rows, "target_expected_reward_delta")
        spatial_expected = _task_balanced(rows, "spatial_expected_reward_delta")
        temporal_profile = all(row.get("temporal_margin_delta") is not None for row in rows)
        if temporal_profile:
            stalled = _task_balanced(rows, "stalled_target_margin_delta")
            temporal = _task_balanced(rows, "temporal_margin_delta")
            stalled_expected = _task_balanced(rows, "stalled_target_expected_reward_delta")
            temporal_expected = _task_balanced(rows, "temporal_expected_reward_delta")
            safety_tier = (
                0 if target > 0 and spatial > 0 and temporal > 0 and stalled <= 0
                else 1 if target >= 0 and spatial > 0 and temporal > 0
                else 2
            )
            causal_score = target + spatial + 2 * temporal
        else:
            stalled = temporal = stalled_expected = temporal_expected = None
            safety_tier = 0 if target > 0 and spatial > 0 else 1 if target >= 0 else 2
            causal_score = target + spatial
        summaries.append(
            {
                "layer": key[0],
                "head": key[1],
                "source_ranks": evidence["source_ranks"],
                "best_source_rank": evidence["best_source_rank"],
                "n_success_samples": len(rows),
                "n_subsets": len({str(row.get("subset")) for row in rows}),
                "task_balanced_target_margin_delta": target,
                "task_balanced_spatial_margin_delta": spatial,
                "task_balanced_target_expected_reward_delta": target_expected,
                "task_balanced_spatial_expected_reward_delta": spatial_expected,
                "task_balanced_stalled_target_margin_delta": stalled,
                "task_balanced_temporal_margin_delta": temporal,
                "task_balanced_stalled_target_expected_reward_delta": stalled_expected,
                "task_balanced_temporal_expected_reward_delta": temporal_expected,
                "median_target_margin_delta": median(float(row["target_margin_delta"]) for row in rows),
                "median_spatial_margin_delta": median(float(row["spatial_margin_delta"]) for row in rows),
                "target_positive_fraction": mean(float(row["target_margin_delta"]) > 0 for row in rows),
                "spatial_positive_fraction": mean(float(row["spatial_margin_delta"]) > 0 for row in rows),
                "safety_tier": safety_tier,
                # Frozen before inference: within each tier, reward-5 target
                # preservation and target-vs-wrong specificity have equal weight.
                "causal_score": causal_score,
            }
        )
    summaries.sort(
        key=lambda row: (
            row["safety_tier"],
            -row["causal_score"],
            -row["task_balanced_target_margin_delta"],
            row["best_source_rank"],
            row["layer"],
            row["head"],
        )
    )
    profiled = {(row["layer"], row["head"]) for row in summaries}
    fallback = []
    for row in _load_fallback_rows(causal):
        key = (int(row["layer"]), int(row["head"]))
        if key in profiled:
            continue
        fallback.append(
            {
                "layer": key[0],
                "head": key[1],
                "source_ranks": {"raw_mass": int(row.get("rank", len(fallback) + 1))},
                "best_source_rank": int(row.get("rank", len(fallback) + 1)),
                "n_success_samples": 0,
                "n_subsets": 0,
                "task_balanced_target_margin_delta": None,
                "task_balanced_spatial_margin_delta": None,
                "task_balanced_target_expected_reward_delta": None,
                "task_balanced_spatial_expected_reward_delta": None,
                "median_target_margin_delta": None,
                "median_spatial_margin_delta": None,
                "target_positive_fraction": None,
                "spatial_positive_fraction": None,
                "safety_tier": 3,
                "causal_score": None,
                "fallback_only": True,
            }
        )
    ordered = [*summaries, *fallback]
    ranking = [{**row, "rank": index} for index, row in enumerate(ordered, start=1)]
    artifact = {
        "schema_version": "causal-head-ranking-v1",
        "ranking_source": "independent_success_causal_target_vs_wrong_profile",
        "ranking_method_detail": (
            "union_top64_raw_excess_enrichment; all_34_independent_success_trajectories; "
            "teacher_forced_reward5_margin; target_boost_vs_equal_size_wrong_control; "
            + (
                "full_success_vs_early_stalled_temporal_control; temporal margin double weight; "
                "positive_full_target_spatial_temporal_tiers"
                if summaries and summaries[0].get("task_balanced_temporal_margin_delta") is not None
                else "positive_target_and_spatial_tier_then_equal_weight_causal_score"
            )
        ),
        "labels_model_facing": False,
        "expected_success_samples": expected_samples,
        "candidate_count": len(candidates),
        "total_ranking_head_count": len(ranking),
        "fallback_head_count": len(fallback),
        "invalid_record_count": len(invalid),
        "ranking": ranking,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    path = output / "causal_ranking.json"
    write_json(path, artifact)
    safe_rows = [row for row in summaries if row["safety_tier"] <= 1]
    target_top_k = int(causal.get("safe_padding_top_k", 64))
    if not 8 <= len(safe_rows) <= target_top_k:
        raise RuntimeError(
            f"Safe causal head count {len(safe_rows)} cannot support top8 or padding"
        )
    padding_needed = target_top_k - len(safe_rows)
    # Draw padding from the lowest raw-mass unprofiled heads, while leaving at
    # least target_top_k still-lower heads for a disjoint low-rank control.
    if len(fallback) < padding_needed + target_top_k:
        raise RuntimeError("Insufficient unprofiled heads for safe padding and control")
    padding_start = len(fallback) - target_top_k - padding_needed
    padding_source = fallback[padding_start : padding_start + padding_needed]
    padding_keys = {(row["layer"], row["head"]) for row in padding_source}
    padding = [
        {
            **row,
            "safety_tier": 2,
            "fallback_only": True,
            "safe_padding": True,
            "padding_basis": "lowest_raw_mass_unprofiled_reserving_disjoint_tail_control",
        }
        for row in padding_source
    ]
    harmful_profiled = [row for row in summaries if row["safety_tier"] > 1]
    remaining_fallback = [
        row for row in fallback if (row["layer"], row["head"]) not in padding_keys
    ]
    safe_ordered = [*safe_rows, *padding, *harmful_profiled, *remaining_fallback]
    safe_ranking = [
        {**row, "rank": index} for index, row in enumerate(safe_ordered, start=1)
    ]
    safe_artifact = {
        **{key: value for key, value in artifact.items() if key not in {"ranking", "fingerprint"}},
        "schema_version": "causal-safe-padding-ranking-v1",
        "ranking_source": "independent_success_causal_profile_with_inert_mass_padding",
        "ranking_method_detail": (
            artifact["ranking_method_detail"]
            + "; safe target-nonnegative heads first; pad to top64 with lowest-raw-mass "
            "unprofiled heads while reserving a disjoint lower tail for controls"
        ),
        "safe_causal_head_count": len(safe_rows),
        "safe_padding_head_count": len(padding),
        "safe_padding_target_top_k": target_top_k,
        "ranking": safe_ranking,
    }
    safe_artifact["fingerprint"] = object_fingerprint(safe_artifact)
    write_json(output / "causal_safe_padding_ranking.json", safe_artifact)
    return path


def aggregate_contrastive(config: dict[str, Any], *, num_head_shards: int) -> Path:
    """Combine paired success/fail development effects without held-out labels."""
    _attention, causal = _section(config)
    if not bool(causal.get("contrastive_fail_development", False)):
        raise ValueError("aggregate-contrastive requires contrastive_fail_development")
    candidates = _load_candidate_pool(causal)
    output = Path(causal["output_dir"]).resolve()
    fail_profile_dir = Path(causal.get("fail_profile_dir", output)).resolve()
    fail_paths = [
        fail_profile_dir / f"causal_profile.head-shard-{index:02d}-of-{num_head_shards:02d}.jsonl"
        for index in range(num_head_shards)
    ]
    success_dir = Path(causal["success_profile_dir"]).resolve()
    success_shards = int(causal.get("success_profile_num_head_shards", 2))
    success_paths = [
        success_dir / f"causal_profile.head-shard-{index:02d}-of-{success_shards:02d}.jsonl"
        for index in range(success_shards)
    ]
    missing = [str(path) for path in [*fail_paths, *success_paths] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing contrastive profile shards: {missing}")

    def latest_rows(paths: list[Path], prefix: str) -> dict[tuple[int, int, str], dict[str, Any]]:
        latest = {}
        for path in paths:
            for row in read_jsonl(path):
                if str(row.get("example_id", "")).startswith(prefix):
                    latest[(int(row["layer"]), int(row["head"]), str(row["video_sha256"]))] = row
        return latest

    success = latest_rows(success_paths, "suc/")
    fail = latest_rows(fail_paths, "fail/")
    expected = int(causal.get("expected_success_samples", 34))
    summaries = []
    for evidence in candidates:
        head_key = (int(evidence["layer"]), int(evidence["head"]))
        success_rows = {
            key[2]: row for key, row in success.items()
            if key[:2] == head_key and row.get("status") == "ok"
        }
        fail_rows = {
            key[2]: row for key, row in fail.items()
            if key[:2] == head_key and row.get("status") == "ok"
        }
        videos = sorted(set(success_rows) & set(fail_rows))
        if len(videos) != expected:
            raise RuntimeError(
                f"Head {head_key} has {len(videos)}/{expected} paired development videos"
            )
        suc_target = mean(float(success_rows[v]["target_margin_delta"]) for v in videos)
        fail_target = mean(-float(fail_rows[v]["target_margin_delta"]) for v in videos)
        suc_spatial = mean(float(success_rows[v]["spatial_margin_delta"]) for v in videos)
        fail_spatial = mean(-float(fail_rows[v]["spatial_margin_delta"]) for v in videos)
        paired_target = mean(
            (
                float(success_rows[v]["target_margin_delta"])
                - float(fail_rows[v]["target_margin_delta"])
            ) / 2
            for v in videos
        )
        paired_spatial = mean(
            (
                float(success_rows[v]["spatial_margin_delta"])
                - float(fail_rows[v]["spatial_margin_delta"])
            ) / 2
            for v in videos
        )
        safety_tier = (
            0 if suc_target > 0 and fail_target > 0 and paired_spatial > 0
            else 1 if paired_target > 0 and min(suc_target, fail_target) >= 0
            else 2
        )
        summaries.append(
            {
                "layer": head_key[0],
                "head": head_key[1],
                "source_ranks": evidence["source_ranks"],
                "best_source_rank": evidence["best_source_rank"],
                "n_paired_development_videos": len(videos),
                "success_target_correct_margin_delta": suc_target,
                "fail_target_correct_margin_delta": fail_target,
                "success_spatial_correct_margin_delta": suc_spatial,
                "fail_spatial_correct_margin_delta": fail_spatial,
                "paired_target_correct_margin_delta": paired_target,
                "paired_spatial_correct_margin_delta": paired_spatial,
                "safety_tier": safety_tier,
                "causal_score": paired_target + paired_spatial,
            }
        )
    summaries.sort(
        key=lambda row: (
            row["safety_tier"],
            -row["causal_score"],
            -min(
                row["success_target_correct_margin_delta"],
                row["fail_target_correct_margin_delta"],
            ),
            row["best_source_rank"],
            row["layer"],
            row["head"],
        )
    )
    profiled = {(row["layer"], row["head"]) for row in summaries}
    fallback = []
    for row in _load_fallback_rows(causal):
        key = (int(row["layer"]), int(row["head"]))
        if key in profiled:
            continue
        fallback.append(
            {
                "layer": key[0], "head": key[1],
                "source_ranks": {"raw_mass": int(row.get("rank", len(fallback) + 1))},
                "best_source_rank": int(row.get("rank", len(fallback) + 1)),
                "safety_tier": 3, "causal_score": None, "fallback_only": True,
            }
        )
    safe_rows = [row for row in summaries if row["safety_tier"] <= 1]
    top_k = int(causal.get("safe_padding_top_k", 64))
    minimum_safe_heads = int(causal.get("minimum_positive_safe_heads", 8))
    if not 1 <= minimum_safe_heads <= 8:
        raise ValueError("minimum_positive_safe_heads must be between 1 and 8")
    if not minimum_safe_heads <= len(safe_rows) <= top_k:
        raise RuntimeError(f"Contrastive safe head count {len(safe_rows)} cannot support top8/padding")
    need = top_k - len(safe_rows)
    if len(fallback) < need + top_k:
        raise RuntimeError("Insufficient fallback heads for contrastive padding/control")
    start = len(fallback) - top_k - need
    padding_source = fallback[start : start + need]
    padding_keys = {(row["layer"], row["head"]) for row in padding_source}
    padding = [
        {
            **row, "safety_tier": 2, "safe_padding": True,
            "padding_basis": "lowest_raw_mass_unprofiled_reserving_disjoint_tail_control",
        }
        for row in padding_source
    ]
    harmful = [row for row in summaries if row["safety_tier"] > 1]
    remaining = [
        row for row in fallback if (row["layer"], row["head"]) not in padding_keys
    ]
    ranking = [
        {**row, "rank": index}
        for index, row in enumerate([*safe_rows, *padding, *harmful, *remaining], start=1)
    ]
    artifact = {
        "schema_version": "paired-contrastive-causal-safe-padding-ranking-v1",
        "ranking_source": "paired_same_video_success_fail_development_causal_profile",
        "ranking_method_detail": (
            "one deterministic fail instruction per each of 34 ranking videos; "
            "teacher-forced correct-endpoint margin effects; paired target and spatial specificity; "
            "held-out 697-record labels excluded; lowest-raw-mass padding to top64"
        ),
        "labels_model_facing": False,
        "development_labels_used_for_head_selection": True,
        "development_video_count": expected,
        "candidate_count": len(candidates),
        "minimum_positive_safe_heads": minimum_safe_heads,
        "safe_causal_head_count": len(safe_rows),
        "safe_padding_head_count": len(padding),
        "total_ranking_head_count": len(ranking),
        "ranking": ranking,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    path = output / "contrastive_causal_safe_padding_ranking.json"
    write_json(path, artifact)
    return path


def aggregate_bidirectional(config: dict[str, Any]) -> Path:
    """Merge explicitly validated +bias and -bias paired causal heads."""
    _attention, causal = _section(config)
    positive_path = Path(causal["positive_contrastive_ranking_path"]).resolve()
    positive_data = json.loads(positive_path.read_text(encoding="utf-8"))
    positive_rows = [
        row for row in positive_data.get("ranking", [])
        if row.get("success_target_correct_margin_delta") is not None
    ]
    if not positive_rows:
        raise ValueError(f"No profiled positive rows in {positive_path}")

    negative_candidates = _load_candidate_pool(causal)
    shards = int(causal.get("negative_profile_num_head_shards", 1))
    success_dir = Path(causal["negative_success_profile_dir"]).resolve()
    fail_dir = Path(causal["negative_fail_profile_dir"]).resolve()
    success_paths = [
        success_dir / f"causal_profile.head-shard-{index:02d}-of-{shards:02d}.jsonl"
        for index in range(shards)
    ]
    fail_paths = [
        fail_dir / f"causal_profile.head-shard-{index:02d}-of-{shards:02d}.jsonl"
        for index in range(shards)
    ]
    missing = [str(path) for path in [*success_paths, *fail_paths] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing signed profile shards: {missing}")

    def latest(paths: list[Path], prefix: str) -> dict[tuple[int, int, str], dict[str, Any]]:
        rows = {}
        for path in paths:
            for row in read_jsonl(path):
                if row.get("status") == "ok" and str(row.get("example_id", "")).startswith(prefix):
                    rows[(int(row["layer"]), int(row["head"]), str(row["video_sha256"]))] = row
        return rows

    success = latest(success_paths, "suc/")
    fail = latest(fail_paths, "fail/")
    expected = int(causal.get("expected_success_samples", 32))

    def safety_tier(suc_target: float, fail_target: float, paired_spatial: float) -> int:
        if suc_target > 0 and fail_target > 0 and paired_spatial > 0:
            return 0
        if min(suc_target, fail_target) >= 0:
            return 1
        return 2

    signed_rows = []
    for row in positive_rows:
        item = dict(row)
        item["steering_multiplier"] = 1.0
        item["validated_profile_bias"] = 6.0
        item["safety_tier"] = safety_tier(
            float(item["success_target_correct_margin_delta"]),
            float(item["fail_target_correct_margin_delta"]),
            float(item["paired_spatial_correct_margin_delta"]),
        )
        signed_rows.append(item)

    for evidence in negative_candidates:
        key = (int(evidence["layer"]), int(evidence["head"]))
        success_rows = {
            row_key[2]: row for row_key, row in success.items() if row_key[:2] == key
        }
        fail_rows = {row_key[2]: row for row_key, row in fail.items() if row_key[:2] == key}
        videos = sorted(set(success_rows) & set(fail_rows))
        if len(videos) != expected:
            raise RuntimeError(f"Signed head {key} has {len(videos)}/{expected} pairs")
        suc_target = mean(float(success_rows[v]["target_margin_delta"]) for v in videos)
        fail_target = mean(-float(fail_rows[v]["target_margin_delta"]) for v in videos)
        suc_spatial = mean(float(success_rows[v]["spatial_margin_delta"]) for v in videos)
        fail_spatial = mean(-float(fail_rows[v]["spatial_margin_delta"]) for v in videos)
        paired_target = mean(
            (float(success_rows[v]["target_margin_delta"]) - float(fail_rows[v]["target_margin_delta"])) / 2
            for v in videos
        )
        paired_spatial = mean(
            (float(success_rows[v]["spatial_margin_delta"]) - float(fail_rows[v]["spatial_margin_delta"])) / 2
            for v in videos
        )
        signed_rows.append(
            {
                "layer": key[0],
                "head": key[1],
                "source_ranks": evidence["source_ranks"],
                "best_source_rank": evidence["best_source_rank"],
                "n_paired_development_videos": len(videos),
                "success_target_correct_margin_delta": suc_target,
                "fail_target_correct_margin_delta": fail_target,
                "success_spatial_correct_margin_delta": suc_spatial,
                "fail_spatial_correct_margin_delta": fail_spatial,
                "paired_target_correct_margin_delta": paired_target,
                "paired_spatial_correct_margin_delta": paired_spatial,
                "safety_tier": safety_tier(suc_target, fail_target, paired_spatial),
                "causal_score": paired_target + paired_spatial,
                "steering_multiplier": -1.0,
                "validated_profile_bias": -6.0,
            }
        )

    best_by_head: dict[tuple[int, int], dict[str, Any]] = {}
    for row in signed_rows:
        key = (int(row["layer"]), int(row["head"]))
        previous = best_by_head.get(key)
        if previous is None or (
            int(row["safety_tier"]), -float(row["causal_score"])
        ) < (
            int(previous["safety_tier"]), -float(previous["causal_score"])
        ):
            best_by_head[key] = row
    signed_rows = list(best_by_head.values())
    signed_rows.sort(
        key=lambda row: (
            row["safety_tier"],
            -float(row["causal_score"]),
            -min(
                float(row["success_target_correct_margin_delta"]),
                float(row["fail_target_correct_margin_delta"]),
            ),
            int(row["best_source_rank"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    safe = [row for row in signed_rows if int(row["safety_tier"]) <= 1]
    top_k = int(causal.get("safe_padding_top_k", 64))
    if not 8 <= len(safe) <= top_k:
        raise RuntimeError(f"Bidirectional safe head count {len(safe)} cannot support top8")

    profiled = {(int(row["layer"]), int(row["head"])) for row in signed_rows}
    fallback = []
    for row in _load_fallback_rows(causal):
        key = (int(row["layer"]), int(row["head"]))
        if key in profiled:
            continue
        fallback.append(
            {
                "layer": key[0],
                "head": key[1],
                "source_ranks": {"raw_mass": int(row.get("rank", len(fallback) + 1))},
                "best_source_rank": int(row.get("rank", len(fallback) + 1)),
                "safety_tier": 3,
                "causal_score": None,
                "steering_multiplier": 1.0,
                "fallback_only": True,
            }
        )
    need = top_k - len(safe)
    if len(fallback) < need + top_k:
        raise RuntimeError("Insufficient fallback heads for signed padding/control")
    start = len(fallback) - top_k - need
    padding_source = fallback[start : start + need]
    padding_keys = {(row["layer"], row["head"]) for row in padding_source}
    padding = [
        {
            **row,
            "safety_tier": 2,
            "safe_padding": True,
            "padding_basis": "lowest_raw_mass_unprofiled_reserving_disjoint_tail_control",
        }
        for row in padding_source
    ]
    harmful = [row for row in signed_rows if int(row["safety_tier"]) > 1]
    remaining = [
        row for row in fallback if (row["layer"], row["head"]) not in padding_keys
    ]
    ranking = [
        {**row, "rank": index}
        for index, row in enumerate([*safe, *padding, *harmful, *remaining], start=1)
    ]
    artifact = {
        "schema_version": "paired-bidirectional-causal-safe-padding-ranking-v1",
        "ranking_source": "paired_same_video_explicit_positive_and_negative_bias_causal_profile",
        "ranking_method_detail": (
            "positive direction from +6 paired profile; candidate negative direction explicitly rerun at -6; "
            "success/fail correct endpoint margins and paired spatial specificity; held-out labels excluded; "
            "per-head signed steering multiplier; lowest-raw-mass padding to top64"
        ),
        "labels_model_facing": False,
        "development_labels_used_for_head_selection": True,
        "development_video_count": expected,
        "positive_profiled_head_count": len(positive_rows),
        "negative_profiled_head_count": len(negative_candidates),
        "safe_causal_head_count": len(safe),
        "safe_positive_head_count": sum(row["steering_multiplier"] > 0 for row in safe),
        "safe_negative_head_count": sum(row["steering_multiplier"] < 0 for row in safe),
        "safe_padding_head_count": len(padding),
        "total_ranking_head_count": len(ranking),
        "ranking": ranking,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    output = Path(causal["bidirectional_output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "bidirectional_causal_safe_padding_ranking.json"
    write_json(path, artifact)
    return path


def derive_weighted(config: dict[str, Any]) -> Path:
    """Scale signed heads by their weakest paired causal evidence."""
    _attention, causal = _section(config)
    source = Path(causal["bidirectional_ranking_path"]).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data.get("ranking")
    if not isinstance(rows, list) or len(rows) < 64:
        raise ValueError(f"Incomplete bidirectional ranking at {source}")
    causal_strengths = []
    for row in rows[:64]:
        if row.get("success_target_correct_margin_delta") is None:
            continue
        strength = max(
            0.0,
            min(
                float(row["success_target_correct_margin_delta"]),
                float(row["fail_target_correct_margin_delta"]),
            ),
        ) + max(0.0, float(row["paired_spatial_correct_margin_delta"]))
        causal_strengths.append(strength)
    if not causal_strengths or max(causal_strengths) <= 0:
        raise ValueError("Weighted ranking has no positive causal strength")
    scale = max(causal_strengths)
    floor = float(causal.get("weighted_multiplier_floor", 0.1))
    if not 0 < floor <= 1:
        raise ValueError("weighted_multiplier_floor must be in (0, 1]")
    weighted = []
    for row in rows:
        item = dict(row)
        sign = -1.0 if float(row.get("steering_multiplier", 1.0)) < 0 else 1.0
        if row.get("success_target_correct_margin_delta") is not None and int(row.get("safety_tier", 9)) <= 1:
            strength = max(
                0.0,
                min(
                    float(row["success_target_correct_margin_delta"]),
                    float(row["fail_target_correct_margin_delta"]),
                ),
            ) + max(0.0, float(row["paired_spatial_correct_margin_delta"]))
            magnitude = max(floor, math.sqrt(strength / scale))
            item["causal_weight_strength"] = strength
            item["causal_weight_normalizer"] = scale
        elif bool(row.get("safe_padding", False)):
            magnitude = floor
            item["causal_weight_strength"] = 0.0
            item["causal_weight_normalizer"] = scale
        else:
            magnitude = 1.0
        item["steering_multiplier"] = sign * magnitude
        weighted.append(item)
    artifact = {
        **{key: value for key, value in data.items() if key not in {"ranking", "fingerprint"}},
        "schema_version": "paired-bidirectional-causal-weighted-safe-padding-ranking-v1",
        "ranking_source": "paired_bidirectional_causal_profile_strength_weighted",
        "ranking_method_detail": (
            str(data.get("ranking_method_detail", ""))
            + "; abs multiplier=sqrt((min(success_correct,fail_correct)+positive_paired_spatial)/max), "
            + f"floor={floor}; safe padding uses the same nonzero floor"
        ),
        "source_bidirectional_fingerprint": str(data.get("fingerprint")),
        "weighted_multiplier_floor": floor,
        "ranking": weighted,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    output = Path(causal["weighted_output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "bidirectional_causal_weighted_ranking.json"
    write_json(path, artifact)
    return path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    profile_parser = sub.add_parser("profile")
    profile_parser.add_argument("--config", required=True)
    profile_parser.add_argument("--head-shard-id", type=int, required=True)
    profile_parser.add_argument("--num-head-shards", type=int, required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--config", required=True)
    aggregate_parser.add_argument("--num-head-shards", type=int, required=True)
    contrastive_parser = sub.add_parser("aggregate-contrastive")
    contrastive_parser.add_argument("--config", required=True)
    contrastive_parser.add_argument("--num-head-shards", type=int, required=True)
    bidirectional_parser = sub.add_parser("aggregate-bidirectional")
    bidirectional_parser.add_argument("--config", required=True)
    weighted_parser = sub.add_parser("derive-weighted")
    weighted_parser.add_argument("--config", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "profile":
        print(profile(config, head_shard_id=args.head_shard_id, num_head_shards=args.num_head_shards))
    elif args.command == "aggregate":
        print(aggregate(config, num_head_shards=args.num_head_shards))
    elif args.command == "aggregate-contrastive":
        print(aggregate_contrastive(config, num_head_shards=args.num_head_shards))
    elif args.command == "aggregate-bidirectional":
        print(aggregate_bidirectional(config))
    else:
        print(derive_weighted(config))


if __name__ == "__main__":
    main()
