from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from ..data import load_episodes
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    deterministic_merge,
    object_fingerprint,
    provenance,
    read_jsonl,
    stable_shard,
    write_json,
)
from ..protocol import progress, progress_to_reward
from ..schemas import SCHEMA_VERSION
from .dataset import load_partition
from .masking import (
    QUERY_SCOPES,
    Head,
    matched_wrong_position_set,
    select_low_ranked_heads,
)
from .ranking import aggregate_in_domain, consensus_ranking
from .runtime import AttentionRuntime
from .stats import (
    exact_mcnemar_pvalue,
    holm,
    paired_cluster_bootstrap,
    paired_sign_flip_pvalue,
)


def _attention_prompt_protocol(attention: dict[str, Any]) -> dict[str, Any]:
    import hashlib

    from ..protocol import system_prompt

    prompt_mode = str(attention.get("prompt_mode", "official"))
    return {
        "prompt_mode": prompt_mode,
        "prompt_sha256": hashlib.sha256(
            system_prompt(prompt_mode).encode("utf-8")
        ).hexdigest(),
        "decoding": "greedy_attention_runtime",
        "steering_query_scope": str(attention.get("steering_query_scope", "all")),
        "query_scope_sensitivity": list(
            attention.get("query_scope_sensitivity", [])
        ),
    }


def _query_scopes(attention: dict[str, Any]) -> tuple[str, list[str]]:
    primary = str(attention.get("steering_query_scope", "all"))
    sensitivity = list(
        dict.fromkeys(str(value) for value in attention.get("query_scope_sensitivity", []))
    )
    invalid = [value for value in [primary, *sensitivity] if value not in QUERY_SCOPES]
    if invalid:
        choices = ", ".join(sorted(QUERY_SCOPES))
        raise ValueError(
            f"Unknown attention query scope(s) {invalid}; choose from {choices}"
        )
    return primary, sensitivity


def rank(
    config: dict[str, Any],
    source: str,
    *,
    dry_run: bool = False,
    retry_failed: bool = False,
) -> Path:
    attention = config["attention_eval"]
    output_dir = Path(attention["output_dir"]).resolve()
    if source == "consensus":
        result = consensus_ranking(
            attention["consensus_ranking_paths"],
            expected_layers=int(attention.get("num_layers", 36)),
            expected_heads=int(attention.get("num_heads", 32)),
            skip_early_layers=int(attention.get("skip_early_layers", 0)),
        )
        path = output_dir / "consensus_ranking.json"
        write_json(path, result)
        return path
    samples, split = load_partition(output_dir, "discovery")
    shard_id = int(attention.get("shard_id", 0))
    num_shards = int(attention.get("num_shards", 1))
    samples = [
        row for row in samples if stable_shard(row["video_sha256"], num_shards) == shard_id
    ]
    records_path = (
        output_dir / "in_domain_mass.jsonl"
        if num_shards == 1
        else output_dir / f"in_domain_mass.shard-{shard_id:02d}.jsonl"
    )
    runtime = None if dry_run else AttentionRuntime(attention)
    previous = {}
    if records_path.exists():
        for row in read_jsonl(records_path):
            previous[row["example_id"]] = row
    for sample in samples:
        old = previous.get(sample["example_id"])
        if old and (old.get("status") == "ok" or not retry_failed):
            continue
        try:
            if dry_run:
                layers = int(attention.get("num_layers", 36))
                heads = int(attention.get("num_heads", 32))
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": sample["example_id"],
                    "video_sha256": sample["video_sha256"],
                    "partition": "discovery",
                    "raw_mass": [[0.0] * heads for _ in range(layers)],
                    "excess_mass": [[0.0] * heads for _ in range(layers)],
                    "status": "ok",
                }
            else:
                assert runtime is not None
                row = runtime.collect_mass(sample)
            append_jsonl(records_path, row)
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": sample["example_id"],
                    "video_sha256": sample["video_sha256"],
                    "partition": "discovery",
                    "status": "invalid",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    aggregate_path = records_path
    if num_shards > 1:
        shard_paths = [
            output_dir / f"in_domain_mass.shard-{index:02d}.jsonl"
            for index in range(num_shards)
        ]
        if not all(path.exists() for path in shard_paths):
            return records_path
        aggregate_path = output_dir / "in_domain_mass.jsonl"
        deterministic_merge(shard_paths, aggregate_path)
    result = aggregate_in_domain(
        list(read_jsonl(aggregate_path)),
        num_layers=int(attention.get("num_layers", 36)),
        num_heads=int(attention.get("num_heads", 32)),
        skip_layers=int(attention.get("skip_early_layers", 2)),
    )
    if set(result["per_sample_example_ids"]) - set(split["discovery"]):
        raise AssertionError("Evaluation leakage into in-domain ranking")
    path = output_dir / "in_domain_ranking.json"
    write_json(path, result)
    return path


def _load_ranking(path: Path) -> tuple[list[dict], str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("ranking")
    if not isinstance(rows, list):
        raise ValueError(f"No ranking list in {path}")
    return rows, str(data.get("ranking_source", "unknown")), str(data.get("fingerprint", object_fingerprint(data)))


def steer(config: dict[str, Any], *, dry_run: bool = False, retry_failed: bool = False) -> Path:
    attention = config["attention_eval"]
    output_dir = Path(attention["output_dir"]).resolve()
    steering_partition = str(attention.get("steering_partition", "evaluation"))
    samples, split = load_partition(output_dir, steering_partition)
    shard_id = int(attention.get("shard_id", 0))
    num_shards = int(attention.get("num_shards", 1))
    samples = [
        row for row in samples if stable_shard(row["video_sha256"], num_shards) == shard_id
    ]
    ranking_path = Path(attention.get("ranking_path", output_dir / "consensus_ranking.json")).resolve()
    ranking, ranking_source, ranking_fingerprint = _load_ranking(ranking_path)
    if ranking_source == "in_domain_discovery_only":
        discovery = set(json.loads((output_dir / "split.json").read_text())["discovery"])
        if set(row.get("example_id") for row in ranking if "example_id" in row) - discovery:
            raise AssertionError("In-domain ranking contains non-discovery records")
    records_path = (
        output_dir / "steering.jsonl"
        if num_shards == 1
        else output_dir / f"steering.shard-{shard_id:02d}.jsonl"
    )
    runtime = None if dry_run else AttentionRuntime(attention)
    top_values = [int(x) for x in attention.get("top_k_sensitivity", [8, 64])]
    biases = [float(x) for x in attention.get("bias_sensitivity", [0, 2, 4, 6])]
    primary_k = int(attention.get("top_k", 8))
    primary_bias = float(attention.get("swap_bias", 6))
    primary_scope, scope_values = _query_scopes(attention)
    scope_conditions = list(
        dict.fromkeys(
            str(value)
            for value in attention.get(
                "query_scope_sensitivity_conditions", ["candidate_target"]
            )
        )
    )
    allowed_scope_conditions = {
        "candidate_target",
        "candidate_wrong",
        "low_rank_target",
    }
    invalid_scope_conditions = [
        value for value in scope_conditions if value not in allowed_scope_conditions
    ]
    if invalid_scope_conditions:
        raise ValueError(
            "query_scope_sensitivity_conditions contains unsupported values: "
            f"{invalid_scope_conditions}"
        )
    previous_ids = (
        {row["example_id"] for row in read_jsonl(records_path)}
        if records_path.exists()
        else set()
    )
    for sample in samples:
        if sample["example_id"] in previous_ids and not retry_failed:
            continue
        try:
            candidate = [Head(int(row["layer"]), int(row["head"])) for row in ranking[:primary_k]]
            low = select_low_ranked_heads(ranking, primary_k, candidate)
            if dry_run:
                baseline = {
                    "raw_output": "<score>0%</score>",
                    "signed_score": 0.0,
                    "hook_diagnostics": {
                        "dry_run": True,
                        "query_scope": primary_scope,
                    },
                }
                target_positions = [10, 11, 12, 13]
                image_positions = list(range(10, 30))
                target_spans = []
            else:
                assert runtime is not None
                inputs, spans = runtime.prepare(sample)
                del inputs
                target_positions, image_positions, target_spans = runtime.target_positions(
                    sample, spans, attention.get("intervention_location", "after_cam_high")
                )
                baseline = runtime.generate(
                    sample,
                    heads=candidate,
                    selected_positions=target_positions,
                    image_positions=image_positions,
                    bias=0,
                    query_scope=primary_scope,
                )
            baseline_record = _record(
                sample, ranking_source, ranking_fingerprint, (), 0, "baseline",
                baseline, target_positions, image_positions,
            )
            append_jsonl(records_path, baseline_record)
            if dry_run:
                wrong = [20, 21, 22, 23]
            else:
                if len(target_spans) != 1:
                    wrong = None
                else:
                    wrong = matched_wrong_position_set(
                        target_spans[0],
                        target_positions,
                        spatial_merge_size=runtime.spatial_merge_size,
                    )
            conditions = [
                ("candidate_target", candidate, target_positions),
                ("candidate_wrong", candidate, wrong),
                ("low_rank_target", low, target_positions),
            ]
            if bool(attention.get("include_all_heads_control", True)):
                all_heads = [
                    Head(layer, head)
                    for layer in range(int(attention.get("num_layers", 36)))
                    for head in range(int(attention.get("num_heads", 32)))
                ]
                conditions.append(("all_target", all_heads, target_positions))
            primary_condition_results = {}
            for name, heads, positions in conditions:
                if positions is None:
                    append_jsonl(
                        records_path,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "example_id": sample["example_id"],
                            "video_sha256": sample["video_sha256"],
                            "ranking_source": ranking_source,
                            "condition": name,
                            "query_scope": primary_scope,
                            "status": "missing_control",
                            "reason": "equal-size non-overlapping wrong region unavailable",
                        },
                    )
                    continue
                result = (
                    {
                        "raw_output": "<score>0%</score>",
                        "signed_score": 0.0,
                        "hook_diagnostics": {
                            "dry_run": True,
                            "query_scope": primary_scope,
                        },
                    }
                    if dry_run
                    else runtime.generate(
                        sample,
                        heads=heads,
                        selected_positions=positions,
                        image_positions=image_positions,
                        bias=primary_bias,
                        query_scope=primary_scope,
                    )
                )
                primary_condition_results[name] = result
                append_jsonl(
                    records_path,
                    _record(
                        sample, ranking_source, ranking_fingerprint, heads,
                        primary_bias, name, result, positions, image_positions,
                    ),
                )
            scope_specs = {
                "candidate_target": (candidate, target_positions),
                "candidate_wrong": (candidate, wrong),
                "low_rank_target": (low, target_positions),
            }
            for scope in scope_values:
                for base_name in scope_conditions:
                    heads, positions = scope_specs[base_name]
                    name = f"query_scope_{scope}_{base_name}"
                    if positions is None:
                        append_jsonl(
                            records_path,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "example_id": sample["example_id"],
                                "video_sha256": sample["video_sha256"],
                                "ranking_source": ranking_source,
                                "condition": name,
                                "query_scope": scope,
                                "status": "missing_control",
                                "reason": (
                                    "equal-size non-overlapping wrong region unavailable"
                                ),
                            },
                        )
                        continue
                    if scope == primary_scope and base_name in primary_condition_results:
                        result = dict(primary_condition_results[base_name])
                        result["hook_diagnostics"] = {
                            **result.get("hook_diagnostics", {}),
                            "exact_primary_condition_reuse": True,
                            "query_scope": scope,
                        }
                    elif dry_run:
                        result = {
                            "raw_output": "<score>0%</score>",
                            "signed_score": 0.0,
                            "hook_diagnostics": {
                                "dry_run": True,
                                "query_scope": scope,
                            },
                        }
                    else:
                        result = runtime.generate(
                            sample,
                            heads=heads,
                            selected_positions=positions,
                            image_positions=image_positions,
                            bias=primary_bias,
                            query_scope=scope,
                        )
                    append_jsonl(
                        records_path,
                        _record(
                            sample,
                            ranking_source,
                            ranking_fingerprint,
                            heads,
                            primary_bias,
                            name,
                            result,
                            positions,
                            image_positions,
                        ),
                    )
            # Predeclared dose/top-k sensitivity. bias=0 reuses baseline exactly.
            if bool(attention.get("run_sensitivity", True)):
                for top_k in top_values:
                    heads = [Head(int(row["layer"]), int(row["head"])) for row in ranking[:top_k]]
                    for bias in biases:
                        name = f"sensitivity_candidate_target_k{top_k}_bias{bias:g}"
                        if bias == 0:
                            result = dict(baseline)
                            result["hook_diagnostics"] = {
                                **baseline.get("hook_diagnostics", {}),
                                "exact_baseline_reuse": True,
                            }
                        elif dry_run:
                            result = {
                                "raw_output": "<score>0%</score>",
                                "signed_score": 0.0,
                                "hook_diagnostics": {
                                    "dry_run": True,
                                    "query_scope": primary_scope,
                                },
                            }
                        else:
                            result = runtime.generate(
                                sample,
                                heads=heads,
                                selected_positions=target_positions,
                                image_positions=image_positions,
                                bias=bias,
                                query_scope=primary_scope,
                            )
                        append_jsonl(
                            records_path,
                            _record(
                                sample, ranking_source, ranking_fingerprint, heads,
                                bias, name, result, target_positions, image_positions,
                            ),
                        )
            if attention.get("run_duplicate_location_sensitivity", True):
                if dry_run:
                    duplicate_positions = target_positions
                    duplicate_image_positions = image_positions
                    duplicate_result = {
                        "raw_output": "<score>0%</score>",
                        "signed_score": 0.0,
                        "hook_diagnostics": {
                            "dry_run": True,
                            "query_scope": primary_scope,
                        },
                    }
                else:
                    duplicate_positions, duplicate_image_positions, _ = (
                        runtime.target_positions(sample, spans, "after_all_duplicates")
                    )
                    duplicate_result = runtime.generate(
                        sample,
                        heads=candidate,
                        selected_positions=duplicate_positions,
                        image_positions=duplicate_image_positions,
                        bias=primary_bias,
                        query_scope=primary_scope,
                    )
                append_jsonl(
                    records_path,
                    _record(
                        sample,
                        ranking_source,
                        ranking_fingerprint,
                        candidate,
                        primary_bias,
                        "sensitivity_after_all_duplicates_k8_bias6",
                        duplicate_result,
                        duplicate_positions,
                        duplicate_image_positions,
                    ),
                )
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": sample["example_id"],
                    "video_sha256": sample["video_sha256"],
                    "status": "invalid",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    if attention.get("run_paired", True):
        _run_paired(
            attention,
            output_dir,
            samples,
            ranking_source,
            ranking_fingerprint,
            ranking,
            runtime,
            dry_run=dry_run,
            retry_failed=retry_failed,
        )
    write_json(
        output_dir / f"steering_manifest.shard-{shard_id:02d}.json",
        {
            **provenance(sys.argv, config, Path(__file__).resolve().parents[2]),
            "attention_prompt_protocol": _attention_prompt_protocol(attention),
            "split_fingerprint": split["fingerprint"],
            "steering_partition": steering_partition,
            "steering_population_count": len(samples),
            "ranking_fingerprint": ranking_fingerprint,
            "model_fingerprint": artifact_fingerprint(attention["model_path"]),
            "confirmatory": ranking_source == "frozen_cross_domain_consensus"
            and Path(attention["grounding_run"]).name == "grounding_dino"
            and str(attention.get("eligibility_mode", "audited")) == "audited",
            "eligibility_mode": str(attention.get("eligibility_mode", "audited")),
            "shard_id": shard_id,
            "num_shards": num_shards,
        },
    )
    if num_shards > 1:
        shard_paths = [
            output_dir / f"steering.shard-{index:02d}.jsonl"
            for index in range(num_shards)
        ]
        if all(path.exists() for path in shard_paths):
            deterministic_merge(shard_paths, output_dir / "steering.jsonl")
            manifests = [
                json.loads(
                    (output_dir / f"steering_manifest.shard-{index:02d}.json").read_text()
                )
                for index in range(num_shards)
            ]
            write_json(
                output_dir / "steering_manifest.json",
                {
                    **manifests[0],
                    "shards": [
                        {
                            "shard_id": item["shard_id"],
                            "config_fingerprint": item["config_fingerprint"],
                        }
                        for item in manifests
                    ],
                },
            )
    else:
        write_json(
            output_dir / "steering_manifest.json",
            json.loads(
                (output_dir / f"steering_manifest.shard-{shard_id:02d}.json").read_text()
            ),
        )
    return records_path


def _record(sample, ranking_source, ranking_fingerprint, heads, bias, condition, result, bbox, image):
    hook_diagnostics = result.get("hook_diagnostics", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": sample["example_id"],
        "video_sha256": sample["video_sha256"],
        "ranking_source": ranking_source,
        "ranking_fingerprint": ranking_fingerprint,
        "heads": [[head.layer, head.head] for head in heads],
        "bias": float(bias),
        "condition": condition,
        "query_scope": hook_diagnostics.get("query_scope"),
        "raw_output": result["raw_output"],
        "signed_score": result["signed_score"],
        "progress": progress(result["signed_score"]),
        "bbox_positions": list(bbox),
        "image_positions": list(image),
        "hook_diagnostics": hook_diagnostics,
        "status": (
            "dry_run"
            if hook_diagnostics.get("dry_run")
            else "ok"
        ),
    }


def _run_paired(
    attention: dict[str, Any],
    output_dir: Path,
    evaluation_samples: list[dict],
    ranking_source: str,
    ranking_fingerprint: str,
    ranking: list[dict],
    runtime: AttentionRuntime | None,
    *,
    dry_run: bool,
    retry_failed: bool,
) -> None:
    primary_scope, _ = _query_scopes(attention)
    pair_path = output_dir / "paired_reward1_reward5.jsonl"
    if not pair_path.exists():
        return
    sample_by_id = {row["example_id"]: row for row in evaluation_samples}
    candidate = [
        Head(int(row["layer"]), int(row["head"]))
        for row in ranking[: int(attention.get("top_k", 8))]
    ]
    shard_id = int(attention.get("shard_id", 0))
    num_shards = int(attention.get("num_shards", 1))
    output = (
        output_dir / "paired_steering.jsonl"
        if num_shards == 1
        else output_dir / f"paired_steering.shard-{shard_id:02d}.jsonl"
    )
    ranking_output = (
        output_dir / "paired_attention_ranking.jsonl"
        if num_shards == 1
        else output_dir / f"paired_attention_ranking.shard-{shard_id:02d}.jsonl"
    )
    previous_pairs = (
        {row["pair_id"] for row in read_jsonl(output)}
        if output.exists()
        else set()
    )
    for pair in read_jsonl(pair_path):
        if pair["pair_id"] in previous_pairs and not retry_failed:
            continue
        counter = sample_by_id.get(pair["counterfactual_example_id"])
        original = sample_by_id.get(pair["original_example_id"])
        if counter is None or original is None:
            continue
        samples = {"counterfactual": counter, "original": original}
        if not dry_run:
            assert runtime is not None
            counter_mass = runtime.collect_mass(counter)
            original_mass = runtime.collect_mass(original)
            counter_rank = _mass_ranking(counter_mass["excess_mass"])
            original_rank = _mass_ranking(original_mass["excess_mass"])
            top_count = min(64, len(counter_rank))
            append_jsonl(
                ranking_output,
                {
                    "pair_id": pair["pair_id"],
                    "video_sha256": pair["video_sha256"],
                    "counterfactual_example_id": counter["example_id"],
                    "original_example_id": original["example_id"],
                    "counterfactual_top_heads": counter_rank[:top_count],
                    "original_top_heads": original_rank[:top_count],
                    "top64_overlap": len(
                        {
                            (row["layer"], row["head"])
                            for row in counter_rank[:top_count]
                        }
                        & {
                            (row["layer"], row["head"])
                            for row in original_rank[:top_count]
                        }
                    )
                    / top_count,
                    "all_head_rank_correlation": _rank_correlation(
                        counter_rank, original_rank
                    ),
                    "status": "ok",
                },
            )
        for task_source, task_sample in samples.items():
            for bbox_source, bbox_sample in samples.items():
                dynamic = dict(task_sample)
                dynamic["last"] = dict(task_sample["last"])
                dynamic["last"]["bbox"] = bbox_sample["last"]["bbox"]
                if dry_run:
                    result = {
                        "raw_output": "<score>0%</score>",
                        "signed_score": 0.0,
                        "hook_diagnostics": {
                            "dry_run": True,
                            "query_scope": primary_scope,
                        },
                    }
                    bbox_positions, image_positions = [], []
                else:
                    assert runtime is not None
                    inputs, spans = runtime.prepare(dynamic)
                    del inputs
                    bbox_positions, image_positions, _ = runtime.target_positions(
                        dynamic, spans, attention.get("intervention_location", "after_cam_high")
                    )
                    result = runtime.generate(
                        dynamic,
                        heads=candidate,
                        selected_positions=bbox_positions,
                        image_positions=image_positions,
                        bias=float(attention.get("swap_bias", 6)),
                        query_scope=primary_scope,
                    )
                append_jsonl(
                    output,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "pair_id": pair["pair_id"],
                        "video_sha256": pair["video_sha256"],
                        "task_source": task_source,
                        "bbox_source": bbox_source,
                        "task_example_id": task_sample["example_id"],
                        "bbox_example_id": bbox_sample["example_id"],
                        "ranking_source": ranking_source,
                        "ranking_fingerprint": ranking_fingerprint,
                        "heads": [[head.layer, head.head] for head in candidate],
                        "bias": float(attention.get("swap_bias", 6)),
                        "query_scope": primary_scope,
                        "bbox_positions": bbox_positions,
                        "image_positions": image_positions,
                        "raw_output": result["raw_output"],
                        "signed_score": result["signed_score"],
                        "hook_diagnostics": result.get("hook_diagnostics", {}),
                        "status": "dry_run" if dry_run else "ok",
                    },
                )
    if num_shards > 1:
        paths = [
            output_dir / f"paired_steering.shard-{index:02d}.jsonl"
            for index in range(num_shards)
        ]
        if all(path.exists() for path in paths):
            deterministic_merge(paths, output_dir / "paired_steering.jsonl")
        rank_paths = [
            output_dir / f"paired_attention_ranking.shard-{index:02d}.jsonl"
            for index in range(num_shards)
        ]
        if all(path.exists() for path in rank_paths):
            deterministic_merge(
                rank_paths, output_dir / "paired_attention_ranking.jsonl"
            )


def _mass_ranking(excess_mass) -> list[dict]:
    rows = [
        {"layer": layer, "head": head, "score": float(score)}
        for layer, layer_values in enumerate(excess_mass)
        for head, score in enumerate(layer_values)
        if layer >= 2
    ]
    rows.sort(key=lambda row: (-row["score"], row["layer"], row["head"]))
    return rows


def _rank_correlation(first: list[dict], second: list[dict]) -> float:
    first_rank = {
        (row["layer"], row["head"]): index for index, row in enumerate(first)
    }
    second_rank = {
        (row["layer"], row["head"]): index for index, row in enumerate(second)
    }
    pairs = sorted(set(first_rank) & set(second_rank))
    count = len(pairs)
    if count < 2:
        return 0.0
    squared = sum(
        (first_rank[pair] - second_rank[pair]) ** 2 for pair in pairs
    )
    return 1 - 6 * squared / (count * (count * count - 1))


def metrics(run_dir: str | Path, config: dict[str, Any] | None = None) -> dict:
    run_dir = Path(run_dir).resolve()
    rows = list(read_jsonl(run_dir / "steering.jsonl"))
    grouped: dict[str, dict[str, dict]] = {}
    metadata = {
        row["example_id"]: row for row in read_jsonl(run_dir / "eligible.jsonl")
    } if (run_dir / "eligible.jsonl").exists() else {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault(row["example_id"], {})[row["condition"]] = row
    contrasts = []
    for example_id, conditions in grouped.items():
        required = {"baseline", "candidate_target", "low_rank_target"}
        if not required <= conditions.keys():
            continue
        baseline = conditions["baseline"]["signed_score"]
        target = conditions["candidate_target"]["signed_score"]
        row = {
            "example_id": example_id,
            "video_sha256": conditions["baseline"]["video_sha256"],
            "target_shift": target - baseline,
            "head_specificity": target - conditions["low_rank_target"]["signed_score"],
            "spatial_specificity": (
                target - conditions["candidate_wrong"]["signed_score"]
                if "candidate_wrong" in conditions
                else None
            ),
            "bbox_mass_increased": _bbox_mass_increased(conditions),
            "subset": metadata.get(example_id, {}).get("subset"),
        }
        contrasts.append(row)
    samples = int((config or {}).get("attention_eval", {}).get("bootstrap_samples", 10_000))
    summary = {
        "n_formal_contrasts": len(contrasts),
        "estimands": {
            field: paired_cluster_bootstrap(contrasts, field, samples=samples)
            for field in ("target_shift", "spatial_specificity", "head_specificity")
        },
        "hook_diagnostics": {
            "bbox_mass_increase_rate": (
                sum(row["bbox_mass_increased"] is True for row in contrasts)
                / sum(row["bbox_mass_increased"] is not None for row in contrasts)
                if any(row["bbox_mass_increased"] is not None for row in contrasts)
                else None
            )
        },
    }
    configured_scope_values = list(
        dict.fromkeys(
            str(value)
            for value in (config or {})
            .get("attention_eval", {})
            .get("query_scope_sensitivity", [])
        )
    )
    discovered_scope_values = [
        scope
        for scope in sorted(QUERY_SCOPES)
        if any(
            f"query_scope_{scope}_candidate_target" in conditions
            for conditions in grouped.values()
        )
    ]
    scope_values = configured_scope_values or discovered_scope_values
    if scope_values:
        all_target_by_id = {
            example_id: conditions["query_scope_all_candidate_target"]
            for example_id, conditions in grouped.items()
            if "query_scope_all_candidate_target" in conditions
        }
        scope_summary = {}
        for scope in scope_values:
            scope_rows = []
            for example_id, conditions in grouped.items():
                target_key = f"query_scope_{scope}_candidate_target"
                if "baseline" not in conditions or target_key not in conditions:
                    continue
                baseline = conditions["baseline"]
                target = conditions[target_key]
                value = {
                    "example_id": example_id,
                    "video_sha256": baseline["video_sha256"],
                    "subset": metadata.get(example_id, {}).get("subset"),
                    "target_shift": (
                        float(target["signed_score"])
                        - float(baseline["signed_score"])
                    ),
                    "bbox_mass_increased": _bbox_mass_increased(
                        {"baseline": baseline, "candidate_target": target}
                    ),
                }
                wrong_key = f"query_scope_{scope}_candidate_wrong"
                if wrong_key in conditions:
                    value["spatial_specificity"] = (
                        float(target["signed_score"])
                        - float(conditions[wrong_key]["signed_score"])
                    )
                low_key = f"query_scope_{scope}_low_rank_target"
                if low_key in conditions:
                    value["head_specificity"] = (
                        float(target["signed_score"])
                        - float(conditions[low_key]["signed_score"])
                    )
                if example_id in all_target_by_id:
                    value["candidate_score_minus_all_scope"] = (
                        float(target["signed_score"])
                        - float(all_target_by_id[example_id]["signed_score"])
                    )
                scope_rows.append(value)
            scope_summary[scope] = {
                "n": len(scope_rows),
                "estimands": {
                    field: paired_cluster_bootstrap(
                        scope_rows, field, samples=samples
                    )
                    for field in (
                        "target_shift",
                        "spatial_specificity",
                        "head_specificity",
                        "candidate_score_minus_all_scope",
                    )
                },
                "bbox_mass_increase_rate": (
                    sum(row["bbox_mass_increased"] is True for row in scope_rows)
                    / sum(
                        row["bbox_mass_increased"] is not None for row in scope_rows
                    )
                    if any(
                        row["bbox_mass_increased"] is not None for row in scope_rows
                    )
                    else None
                ),
            }
        summary["query_scope_ablation"] = {
            "reference_scope": "all",
            "scopes": scope_summary,
        }
    pvalues = {
        field: paired_sign_flip_pvalue(contrasts, field, samples=samples)
        for field in ("target_shift", "spatial_specificity", "head_specificity")
    }
    summary["two_sided_pvalues"] = pvalues
    specificity_adjusted = holm(
        {key: pvalues[key] for key in ("spatial_specificity", "head_specificity")}
    )
    summary["specificity_holm_adjusted"] = specificity_adjusted
    hook_rate = summary["hook_diagnostics"]["bbox_mass_increase_rate"]
    expected_direction = all(
        summary["estimands"][field]["mean"] is not None
        and summary["estimands"][field]["mean"] > 0
        for field in ("target_shift", "spatial_specificity", "head_specificity")
    )
    target_head_specific_pattern = bool(
        pvalues["target_shift"] is not None
        and pvalues["target_shift"] < 0.05
        and all(
            value is not None and value < 0.05
            for value in specificity_adjusted.values()
        )
        and hook_rate is not None
        and hook_rate > 0.5
        and expected_direction
    )
    eligibility_mode = str(
        (config or {}).get("attention_eval", {}).get(
            "eligibility_mode", "audited"
        )
    )
    # An unaudited box can be useful for screening but cannot establish the
    # target/head-specific *causal* conclusion, regardless of its statistics.
    summary["target_head_specific_causal_effect_supported"] = (
        target_head_specific_pattern
        if eligibility_mode == "audited"
        else False
    )
    if eligibility_mode == "auto_valid_grounding":
        summary["exploratory_target_head_specific_pattern"] = (
            target_head_specific_pattern
        )
    if (run_dir / "split.json").exists():
        split = json.loads((run_dir / "split.json").read_text())
        steering_partition = str(
            (config or {}).get("attention_eval", {}).get("steering_partition", "")
        )
        if not steering_partition and (run_dir / "steering_manifest.json").exists():
            steering_manifest = json.loads((run_dir / "steering_manifest.json").read_text())
            steering_partition = str(steering_manifest.get("steering_partition", "evaluation"))
        summary["analysis_partition"] = steering_partition or "evaluation"
        if eligibility_mode == "auto_valid_grounding":
            eligible = list(read_jsonl(run_dir / "eligible.jsonl"))
            summary["formal_gate"] = {
                "status": "exploratory_unaudited_auto_grounding",
                "eligibility_mode": eligibility_mode,
                "population_count": len(eligible),
                "population_subset_count": len({row.get("subset") for row in eligible}),
                "reason": "Uses automatic grounding boxes without human endpoint audit.",
            }
        elif steering_partition == "all_eligible":
            eligible = list(read_jsonl(run_dir / "eligible.jsonl"))
            summary["formal_gate"] = {
                "status": "exploratory_all_eligible_followup",
                "population_count": len(eligible),
                "population_subset_count": len({row.get("subset") for row in eligible}),
                "reason": "Includes discovery samples; not a held-out evaluation result.",
            }
        else:
            summary["formal_gate"] = split.get("formal_gate")
    paired_path = run_dir / "paired_steering.jsonl"
    if paired_path.exists():
        paired_groups: dict[tuple[str, str], dict[str, dict]] = {}
        for row in read_jsonl(paired_path):
            if row.get("status") == "ok":
                paired_groups.setdefault(
                    (row["pair_id"], row["task_source"]), {}
                )[row["bbox_source"]] = row
        paired_contrasts = []
        for (pair_id, task_source), values in paired_groups.items():
            if {"counterfactual", "original"} <= values.keys():
                paired_contrasts.append(
                    {
                        "pair_id": pair_id,
                        "video_sha256": values["original"]["video_sha256"],
                        "task_source": task_source,
                        "counterfactual_minus_original_bbox": (
                            values["counterfactual"]["signed_score"]
                            - values["original"]["signed_score"]
                        ),
                    }
                )
        summary["paired_instruction_conditioned"] = {
            "n_task_pair_contrasts": len(paired_contrasts),
            "counterfactual_minus_original_bbox": paired_cluster_bootstrap(
                paired_contrasts,
                "counterfactual_minus_original_bbox",
                samples=samples,
            ),
        }
    paired_manifest = run_dir / "paired_reward1_reward5.jsonl"
    if paired_manifest.exists():
        summary["paired_non_destructive_reward5"] = _paired_non_destructive_metrics(
            list(read_jsonl(paired_manifest)),
            grouped,
            samples=samples,
            non_inferiority_margin=float(
                (config or {}).get("attention_eval", {}).get(
                    "non_inferiority_margin", -0.05
                )
            ),
        )
    expected_reward = (config or {}).get("attention_eval", {}).get(
        "expected_reward_for_metrics_only"
    )
    if expected_reward is not None:
        summary["single_label_cohort"] = _single_label_cohort_metrics(
            grouped,
            metadata=metadata,
            expected_reward=int(expected_reward),
            samples=samples,
            non_inferiority_margin=float(
                (config or {}).get("attention_eval", {}).get(
                    "non_inferiority_margin", -0.05
                )
            ),
        )
    write_json(run_dir / "attention_metrics.json", summary)
    lines = [
        "# Causal Attention Metrics",
        "",
        f"- Formal contrasts: {summary['n_formal_contrasts']}",
        f"- Target/head-specific causal effect supported: "
        f"`{summary['target_head_specific_causal_effect_supported']}`",
        "",
        "| Estimand | Mean | 95% cluster bootstrap CI |",
        "|---|---:|---:|",
    ]
    for name, value in summary["estimands"].items():
        lines.append(f"| {name} | {value['mean']} | {value['ci95']} |")
    (run_dir / "attention_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def _reward_distribution(rows: list[dict], field: str) -> dict[str, int]:
    return {
        str(label): sum(int(row[field]) == label for row in rows)
        for label in range(1, 6)
    }


def _single_label_cohort_metrics(
    grouped: dict[str, dict[str, dict]],
    *,
    metadata: dict[str, dict] | None = None,
    expected_reward: int,
    samples: int,
    non_inferiority_margin: float,
) -> dict:
    """Post-inference metrics for a frozen, one-label independent cohort.

    ``expected_reward`` comes from an analysis-only config key.  It is never
    added to ``EpisodeRecord.model_payload``, parser input, grounder input, or
    ranking input.
    """
    if not 1 <= expected_reward <= 5:
        raise ValueError("expected_reward_for_metrics_only must be in 1..5")
    rows: list[dict] = []
    for example_id, conditions in grouped.items():
        if not {"baseline", "candidate_target"} <= conditions.keys():
            continue
        baseline = conditions["baseline"]
        candidate = conditions["candidate_target"]
        row = {
            "example_id": example_id,
            "video_sha256": baseline["video_sha256"],
            "subset": (metadata or {}).get(example_id, {}).get("subset"),
            "baseline_signed_score": float(baseline["signed_score"]),
            "candidate_target_signed_score": float(candidate["signed_score"]),
            "candidate_minus_baseline": float(candidate["signed_score"])
            - float(baseline["signed_score"]),
            "baseline_label": progress_to_reward(progress(float(baseline["signed_score"]))),
            "candidate_target_label": progress_to_reward(
                progress(float(candidate["signed_score"]))
            ),
        }
        row["baseline_correct"] = row["baseline_label"] == expected_reward
        row["candidate_correct"] = (
            row["candidate_target_label"] == expected_reward
        )
        for condition in ("candidate_wrong", "low_rank_target", "all_target"):
            if condition in conditions:
                row[f"{condition}_label"] = progress_to_reward(
                    progress(float(conditions[condition]["signed_score"]))
                )
        if "candidate_wrong" in conditions:
            row["spatial_specificity"] = float(candidate["signed_score"]) - float(
                conditions["candidate_wrong"]["signed_score"]
            )
        if "low_rank_target" in conditions:
            row["head_specificity"] = float(candidate["signed_score"]) - float(
                conditions["low_rank_target"]["signed_score"]
            )
        rows.append(row)
    delta = paired_cluster_bootstrap(rows, "candidate_minus_baseline", samples=samples)
    lower = delta["ci95"][0]
    if lower is None:
        decision = "not_estimable"
    elif lower >= non_inferiority_margin:
        decision = "non_inferiority_supported"
    elif delta["ci95"][1] < non_inferiority_margin:
        decision = "evidence_of_harm_beyond_margin"
    else:
        decision = "non_inferiority_not_established"
    baseline_at_target = [row for row in rows if row["baseline_label"] == expected_reward]
    target_flips = [
        row
        for row in rows
        if row["baseline_label"] == expected_reward
        and row["candidate_target_label"] != expected_reward
    ]
    condition_fields = {
        "baseline": "baseline_label",
        "candidate_target": "candidate_target_label",
        "candidate_wrong": "candidate_wrong_label",
        "low_rank_target": "low_rank_target_label",
        "all_target": "all_target_label",
    }
    return {
        "analysis_scope": "frozen_independent_single_label_cohort",
        "expected_reward_metrics_only": expected_reward,
        "n_baseline_candidate": len(rows),
        "non_inferiority_margin_signed_score": non_inferiority_margin,
        "candidate_minus_baseline": delta,
        "candidate_minus_baseline_two_sided_sign_flip_pvalue": paired_sign_flip_pvalue(
            rows, "candidate_minus_baseline", samples=samples
        ),
        "expected_reward_exact_mcnemar_pvalue_record_level": exact_mcnemar_pvalue(
            rows, "baseline_correct", "candidate_correct"
        ),
        "non_inferiority_decision": decision,
        "label_distribution": {
            name: _reward_distribution(rows, field)
            if any(field in row for row in rows)
            else None
            for name, field in condition_fields.items()
        },
        "expected_reward_prediction_rate": {
            "baseline": sum(row["baseline_label"] == expected_reward for row in rows) / len(rows)
            if rows
            else None,
            "candidate_target": sum(
                row["candidate_target_label"] == expected_reward for row in rows
            )
            / len(rows)
            if rows
            else None,
        },
        "retention_given_baseline_at_expected_reward": sum(
            row["candidate_target_label"] == expected_reward for row in baseline_at_target
        )
        / len(baseline_at_target)
        if baseline_at_target
        else None,
        "baseline_expected_reward_to_other_flip_count": len(target_flips),
        "baseline_expected_reward_to_other_flip_example_ids": [
            row["example_id"] for row in target_flips
        ],
        "controls": {
            "spatial_specificity": paired_cluster_bootstrap(
                rows, "spatial_specificity", samples=samples
            ),
            "head_specificity": paired_cluster_bootstrap(
                rows, "head_specificity", samples=samples
            ),
        },
    }


def _paired_non_destructive_metrics(
    pairs: list[dict],
    grouped: dict[str, dict[str, dict]],
    *,
    samples: int,
    non_inferiority_margin: float,
) -> dict:
    """Summarize original reward=5 preservation without touching model inputs.

    The pair manifest is a label-bearing analysis artifact.  It is consulted
    only here, after model output has been written, to identify the original
    instruction side and report its reward=5 adaptation metrics.
    """
    original_rows: list[dict] = []
    complete_pair_ids: list[str] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        counter = grouped.get(str(pair["counterfactual_example_id"]), {})
        original = grouped.get(str(pair["original_example_id"]), {})
        # Complete-pair accounting remains explicit even though Δ5 itself is
        # computed from the original-instruction own-target comparison.
        if {"baseline", "candidate_target"} <= counter.keys() and {
            "baseline", "candidate_target"
        } <= original.keys():
            complete_pair_ids.append(pair_id)
        required = {"baseline", "candidate_target"}
        if not required <= original.keys():
            continue
        baseline = original["baseline"]
        candidate = original["candidate_target"]
        baseline_label = progress_to_reward(progress(float(baseline["signed_score"])))
        candidate_label = progress_to_reward(progress(float(candidate["signed_score"])))
        row = {
            "pair_id": pair_id,
            "video_sha256": str(pair["video_sha256"]),
            "subset": pair.get("subset"),
            "baseline_signed_score": float(baseline["signed_score"]),
            "candidate_target_signed_score": float(candidate["signed_score"]),
            "delta5": float(candidate["signed_score"]) - float(baseline["signed_score"]),
            "baseline_label": baseline_label,
            "candidate_target_label": candidate_label,
            "baseline_reward5": baseline_label == 5,
            "candidate_target_reward5": candidate_label == 5,
            "reward5_to_less_than5": baseline_label == 5 and candidate_label < 5,
        }
        if "candidate_wrong" in original:
            row["spatial_specificity"] = float(candidate["signed_score"]) - float(
                original["candidate_wrong"]["signed_score"]
            )
            row["candidate_wrong_label"] = progress_to_reward(
                progress(float(original["candidate_wrong"]["signed_score"]))
            )
        if "low_rank_target" in original:
            row["head_specificity"] = float(candidate["signed_score"]) - float(
                original["low_rank_target"]["signed_score"]
            )
            row["low_rank_target_label"] = progress_to_reward(
                progress(float(original["low_rank_target"]["signed_score"]))
            )
        original_rows.append(row)

    delta = paired_cluster_bootstrap(original_rows, "delta5", samples=samples)
    ci_lower = delta["ci95"][0]
    if ci_lower is None:
        decision = "not_estimable"
    elif ci_lower >= non_inferiority_margin:
        decision = "non_inferiority_supported"
    elif delta["ci95"][1] < non_inferiority_margin:
        decision = "evidence_of_harm_beyond_margin"
    else:
        decision = "non_inferiority_not_established"
    baseline_reward5 = [row for row in original_rows if row["baseline_reward5"]]
    flips = [row for row in original_rows if row["reward5_to_less_than5"]]
    result = {
        "analysis_scope": "same_video_exact_pair_original_reward5_instruction",
        "candidate_pair_count": len(pairs),
        "complete_pair_count": len(complete_pair_ids),
        "original_instruction_with_baseline_and_own_target_count": len(original_rows),
        "non_inferiority_margin_signed_score": non_inferiority_margin,
        "delta5_own_target_minus_baseline": delta,
        "delta5_two_sided_sign_flip_pvalue": paired_sign_flip_pvalue(
            original_rows, "delta5", samples=samples
        ),
        "reward5_exact_mcnemar_pvalue_record_level": exact_mcnemar_pvalue(
            original_rows, "baseline_reward5", "candidate_target_reward5"
        ),
        "non_inferiority_decision": decision,
        "label_distribution": {
            "baseline": _reward_distribution(original_rows, "baseline_label"),
            "candidate_target": _reward_distribution(original_rows, "candidate_target_label"),
            "candidate_wrong": _reward_distribution(original_rows, "candidate_wrong_label")
            if any("candidate_wrong_label" in row for row in original_rows)
            else None,
            "low_rank_target": _reward_distribution(original_rows, "low_rank_target_label")
            if any("low_rank_target_label" in row for row in original_rows)
            else None,
        },
        "reward5_prediction_rate": {
            "baseline": sum(row["baseline_reward5"] for row in original_rows) / len(original_rows)
            if original_rows
            else None,
            "candidate_target": sum(row["candidate_target_reward5"] for row in original_rows)
            / len(original_rows)
            if original_rows
            else None,
        },
        "reward5_retention_given_baseline_reward5": sum(
            row["candidate_target_reward5"] for row in baseline_reward5
        )
        / len(baseline_reward5)
        if baseline_reward5
        else None,
        "reward5_to_less_than5_flip_count": len(flips),
        "reward5_to_less_than5_flip_pair_ids": [row["pair_id"] for row in flips],
        "controls": {
            "spatial_specificity": paired_cluster_bootstrap(
                original_rows, "spatial_specificity", samples=samples
            ),
            "head_specificity": paired_cluster_bootstrap(
                original_rows, "head_specificity", samples=samples
            ),
        },
    }
    return result


def _bbox_mass_increased(conditions: dict[str, dict]) -> bool | None:
    target = conditions["candidate_target"].get("hook_diagnostics", {}).get("bbox_attention_mass")
    baseline = conditions["baseline"].get("hook_diagnostics", {}).get("bbox_attention_mass")
    if target is None or baseline is None:
        return None
    return float(target) > float(baseline)
