"""Model-specific head ranking and paired causal steering for custom manifests."""

from __future__ import annotations

import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from ..attention_eval.masking import (
    Head,
    bbox_to_token_positions,
    matched_wrong_position_set,
    select_low_ranked_heads,
)
from ..config import section
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    object_fingerprint,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


CAUSAL_SCHEMA_VERSION = "my_dataset.causal_attention.v1"


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = section(config, "my_dataset_causal")
    required = {"model_family", "input_manifest", "output_dir", "model_path"}
    missing = sorted(key for key in required if not cfg.get(key))
    if missing:
        raise ValueError(f"my_dataset_causal is missing: {', '.join(missing)}")
    if cfg["model_family"] not in {"roboreward", "qwen", "grm"}:
        raise ValueError("model_family must be roboreward, qwen, or grm")
    return cfg


def _samples(cfg: dict[str, Any], expected_partition: str) -> list[dict[str, Any]]:
    path = Path(cfg["input_manifest"]).resolve()
    rows = list(read_jsonl(path))
    wrong = sorted({str(row.get("partition")) for row in rows} - {expected_partition})
    if wrong:
        raise ValueError(
            f"{expected_partition} operation received other partitions: {wrong}"
        )
    if not rows:
        raise ValueError(f"Empty {expected_partition} input manifest: {path}")
    return rows


def _runtime(cfg: dict[str, Any]):
    if cfg["model_family"] in {"roboreward", "qwen"}:
        from ..qwen_eval.attention import QwenAttentionRuntime

        runtime_cfg = dict(cfg)
        runtime_cfg.setdefault("protocol", "roborewardbench_native")
        return QwenAttentionRuntime(runtime_cfg)
    from ..attention_eval.runtime import AttentionRuntime

    return AttentionRuntime(cfg)


def _ranking_rows(
    arrays: np.ndarray,
    *,
    skip_early_layers: int,
) -> list[dict[str, Any]]:
    if arrays.ndim != 3:
        raise ValueError(f"Expected sample×layer×head attention array, got {arrays.shape}")
    aggregate = arrays.mean(axis=0)
    rows = [
        {"layer": layer, "head": head, "score": float(aggregate[layer, head])}
        for layer in range(skip_early_layers, aggregate.shape[0])
        for head in range(aggregate.shape[1])
    ]
    rows.sort(key=lambda row: (-row["score"], row["layer"], row["head"]))
    return rows


def rank_heads(config: dict[str, Any], *, retry_failed: bool = False) -> Path:
    cfg = _cfg(config)
    samples = _samples(cfg, "discovery")
    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "ranking_mass.jsonl"
    previous = (
        {str(row["example_id"]): row for row in read_jsonl(records_path)}
        if records_path.is_file()
        else {}
    )
    runtime = _runtime(cfg)
    successful = []
    for sample in samples:
        old = previous.get(str(sample["example_id"]))
        if old and old.get("status") == "ok":
            successful.append(old)
            continue
        if old and not retry_failed:
            raise RuntimeError(
                f"Previous ranking failure for {sample['example_id']}; inspect {records_path} "
                "before using --retry-failed"
            )
        try:
            row = {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "partition": "discovery",
                "group_id": sample["group_id"],
                "task_id": sample["task_id"],
                **runtime.collect_mass(sample),
            }
        except Exception as exc:
            row = {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "example_id": sample["example_id"],
                "group_id": sample["group_id"],
                "partition": "discovery",
                "status": "invalid",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        append_jsonl(records_path, row)
        if row.get("status") != "ok":
            raise RuntimeError(f"Ranking failed for {sample['example_id']}: {row.get('error')}")
        successful.append(row)
    score_kind = str(cfg.get("ranking_score_kind", "excess_mass"))
    if score_kind not in {"raw_mass", "excess_mass", "visual_enrichment"}:
        raise ValueError("ranking_score_kind must be raw_mass, excess_mass, or visual_enrichment")
    if score_kind == "visual_enrichment" and any(score_kind not in row for row in successful):
        raise ValueError("Selected runtime does not provide visual_enrichment")
    arrays = np.asarray([row[score_kind] for row in successful], dtype=np.float64)
    ranking = _ranking_rows(
        arrays,
        skip_early_layers=int(cfg.get("skip_early_layers", 8)),
    )
    artifact = {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "model_family": cfg["model_family"],
        "partition": "discovery",
        "method": f"terminal_last_prompt_{score_kind}_mean_skip_early_layers",
        "ranking_score_kind": score_kind,
        "skip_early_layers": int(cfg.get("skip_early_layers", 8)),
        "sample_count": len(successful),
        "group_count": len({row["group_id"] for row in successful}),
        "input_manifest": str(Path(cfg["input_manifest"]).resolve()),
        "input_manifest_sha256": sha256_file(cfg["input_manifest"]),
        "model_fingerprint": artifact_fingerprint(cfg["model_path"]),
        "ranking": ranking,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    path = output_dir / "ranking.json"
    write_json(path, artifact)
    return path


def _heads(rows: list[dict[str, Any]], count: int) -> list[Head]:
    values = [Head(int(row["layer"]), int(row["head"])) for row in rows[:count]]
    if len(values) != count:
        raise ValueError(f"Ranking contains fewer than {count} heads")
    return values


def _layer_matched_random(
    candidate: Iterable[Head],
    excluded: Iterable[Head],
    num_heads: int,
) -> list[Head]:
    blocked = {(value.layer, value.head) for value in excluded}
    result = []
    for value in candidate:
        choices = [
            Head(value.layer, head)
            for head in range(num_heads)
            if (value.layer, head) not in blocked
        ]
        if not choices:
            raise ValueError(f"No layer-matched random head available in layer {value.layer}")
        selected = choices[0]
        result.append(selected)
        blocked.add((selected.layer, selected.head))
    return result


def _result_record(
    sample: dict[str, Any],
    condition: str,
    heads: Iterable[Head],
    bias: float,
    result: dict[str, Any],
    ranking_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "example_id": sample["example_id"],
        "group_id": sample["group_id"],
        "task_id": sample["task_id"],
        "task_family": sample["task_family"],
        "partition": sample["partition"],
        "condition": condition,
        "heads": [[head.layer, head.head] for head in heads],
        "bias": float(bias),
        "ranking_fingerprint": ranking_fingerprint,
        "status": "ok",
        **result,
    }


def check_zero_bias_equivalence(config: dict[str, Any]) -> Path:
    """Verify that the attention runtime is invariant to an inactive bias hook."""
    cfg = _cfg(config)
    partition = str(cfg.get("partition", "validation"))
    samples = _samples(cfg, partition)
    limit = int(cfg.get("equivalence_limit", 20))
    if limit:
        samples = samples[:limit]
    runtime = _runtime(cfg)
    canonical = {}
    baseline_run = cfg.get("baseline_run")
    if baseline_run:
        root = Path(baseline_run).resolve()
        records = []
        for path in sorted(root.glob("records.shard-*.jsonl")):
            records.extend(read_jsonl(path))
        canonical = {
            str(row["example_id"]): row
            for row in records
            if row.get("status") == "ok"
        }
    output_dir = Path(cfg["output_dir"]).resolve()
    records_path = output_dir / f"equivalence_{partition}.jsonl"
    rows = []
    for sample in samples:
        if cfg["model_family"] in {"roboreward", "qwen"}:
            prepared = runtime.prepare(sample)
            target = runtime.target_positions(sample, prepared)
            visual = prepared.visual_positions
            generate = lambda: runtime.generate(
                sample,
                prepared=prepared,
                heads=(),
                selected_positions=target,
                visual_positions=visual,
                bias=0,
                query_scope=str(cfg.get("steering_query_scope", "all")),
            )
        else:
            _inputs, spans = runtime.prepare(sample)
            target, visual, _target_spans = runtime.target_positions(
                sample, spans, str(cfg.get("intervention_location", "after_cam_high"))
            )
            generate = lambda: runtime.generate(
                sample,
                prepared=(_inputs, spans),
                heads=(),
                selected_positions=target,
                image_positions=visual,
                bias=0,
                query_scope=str(cfg.get("steering_query_scope", "all")),
            )
        first = generate()
        second = generate()
        runtime_equal = (
            first.get("raw_output") == second.get("raw_output")
            and first.get("native_prediction") == second.get("native_prediction")
            and first.get("signed_score") == second.get("signed_score")
        )
        canonical_row = canonical.get(str(sample["example_id"]))
        canonical_equal = None
        if canonical_row is not None:
            canonical_equal = (
                canonical_row.get("raw_output") == first.get("raw_output")
                and canonical_row.get("native_prediction") == first.get("native_prediction")
                and canonical_row.get("signed_score") == first.get("signed_score")
            )
        rows.append(
            {
                "schema_version": CAUSAL_SCHEMA_VERSION,
                "example_id": sample["example_id"],
                "group_id": sample["group_id"],
                "partition": partition,
                "runtime_repeat_exact": runtime_equal,
                "canonical_baseline_exact": canonical_equal,
                "first": {
                    key: first.get(key)
                    for key in ("raw_output", "native_prediction", "signed_score")
                },
                "second": {
                    key: second.get(key)
                    for key in ("raw_output", "native_prediction", "signed_score")
                },
                "status": "ok" if runtime_equal else "mismatch",
            }
        )
    write_jsonl(records_path, rows)
    require_canonical = bool(cfg.get("require_canonical_equivalence", False))
    passed = all(row["runtime_repeat_exact"] for row in rows) and (
        not require_canonical
        or all(row["canonical_baseline_exact"] is True for row in rows)
    )
    write_json(
        output_dir / f"equivalence_{partition}_manifest.json",
        {
            "schema_version": CAUSAL_SCHEMA_VERSION,
            "partition": partition,
            "sample_count": len(rows),
            "runtime_repeat_exact": all(row["runtime_repeat_exact"] for row in rows),
            "canonical_comparison_available": bool(canonical),
            "canonical_exact_count": sum(row["canonical_baseline_exact"] is True for row in rows),
            "require_canonical_equivalence": require_canonical,
            "passed": passed,
            "labels_opened": False,
        },
    )
    if not passed:
        raise RuntimeError(f"Zero-bias equivalence gate failed; inspect {records_path}")
    return records_path


def steer(config: dict[str, Any], *, retry_failed: bool = False) -> Path:
    cfg = _cfg(config)
    partition = str(cfg.get("partition", "validation"))
    if partition not in {"validation", "test"}:
        raise ValueError("steering partition must be validation or test")
    samples = _samples(cfg, partition)
    ranking_path = Path(cfg["ranking_path"]).resolve()
    ranking_artifact = json.loads(ranking_path.read_text(encoding="utf-8"))
    if ranking_artifact.get("partition") != "discovery":
        raise ValueError("Steering ranking must come from discovery")
    ranking = ranking_artifact["ranking"]
    top_k = int(cfg.get("top_k", 8))
    candidate = _heads(ranking, top_k)
    low = select_low_ranked_heads(ranking, top_k, candidate)
    num_heads = int(cfg.get("num_heads", 32))
    random_heads = _layer_matched_random(candidate, [*candidate, *low], num_heads)
    bias = float(cfg.get("swap_bias", 6))
    scope = str(cfg.get("steering_query_scope", "all"))
    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"steering_{partition}.jsonl"
    done: dict[str, set[str]] = defaultdict(set)
    if records_path.is_file():
        for row in read_jsonl(records_path):
            if row.get("status") == "ok":
                done[str(row["example_id"])].add(str(row["condition"]))
    runtime = _runtime(cfg)
    expected = {
        "baseline",
        "candidate_target",
        "candidate_wrong",
        "low_rank_target",
        "layer_matched_random_target",
    }
    validation_top_k = [int(value) for value in cfg.get("validation_top_k", [top_k])]
    validation_biases = [float(value) for value in cfg.get("validation_biases", [bias])]
    if partition == "validation":
        expected.update(
            f"validation_candidate_target_k{k}_bias{value:g}"
            for k in validation_top_k
            for value in validation_biases
        )
    for sample in samples:
        if expected <= done[str(sample["example_id"])]:
            continue
        if done[str(sample["example_id"])] and not retry_failed:
            raise RuntimeError(
                f"Incomplete paired conditions for {sample['example_id']}; inspect "
                f"{records_path} before using --retry-failed"
            )
        try:
            if cfg["model_family"] in {"roboreward", "qwen"}:
                prepared = runtime.prepare(sample)
                target = runtime.target_positions(sample, prepared)
                visual = prepared.visual_positions
                if sample.get("wrong_region_bbox") is not None:
                    with Image.open(sample["last_image_path"]) as image:
                        wrong = bbox_to_token_positions(
                            prepared.target_span,
                            sample["wrong_region_bbox"],
                            image.size,
                            runtime.merge_size,
                        )
                    wrong_source = "audited_same_target_image"
                elif bool(cfg.get("require_audited_wrong_region", True)):
                    raise ValueError("Audited wrong_region_bbox is required")
                else:
                    wrong, wrong_source = runtime.wrong_control_positions(prepared, target)
                generate = lambda heads, positions, value: runtime.generate(
                    sample,
                    prepared=prepared,
                    heads=heads,
                    selected_positions=positions,
                    visual_positions=visual,
                    bias=value,
                    query_scope=scope,
                )
            else:
                _inputs, spans = runtime.prepare(sample)
                target, visual, target_spans = runtime.target_positions(
                    sample, spans, str(cfg.get("intervention_location", "after_cam_high"))
                )
                if len(target_spans) != 1:
                    raise ValueError("Primary GRM control requires one target image span")
                if sample.get("wrong_region_bbox") is not None:
                    with Image.open(sample["last_image_path"]) as image:
                        wrong = bbox_to_token_positions(
                            target_spans[0],
                            sample["wrong_region_bbox"],
                            image.size,
                            runtime.spatial_merge_size,
                        )
                    wrong_source = "audited_same_target_image"
                elif bool(cfg.get("require_audited_wrong_region", True)):
                    raise ValueError("Audited wrong_region_bbox is required")
                else:
                    wrong = matched_wrong_position_set(
                        target_spans[0],
                        target,
                        spatial_merge_size=runtime.spatial_merge_size,
                    )
                    wrong_source = "same_target_span_farthest_region"
                if wrong is None:
                    raise ValueError("No equal-size disjoint wrong region for GRM sample")
                generate = lambda heads, positions, value: runtime.generate(
                    sample,
                    prepared=(_inputs, spans),
                    heads=heads,
                    selected_positions=positions,
                    image_positions=visual,
                    bias=value,
                    query_scope=scope,
                )
            if not wrong or set(wrong) & set(target):
                raise ValueError("Wrong-region tokens must be non-empty and disjoint from target")
            if len(wrong) != len(target):
                raise ValueError(
                    "Audited wrong region must map to the same number of visual tokens as target"
                )
            conditions = [
                ("baseline", [], target, 0.0),
                ("candidate_target", candidate, target, bias),
                ("candidate_wrong", candidate, wrong, bias),
                ("low_rank_target", low, target, bias),
                ("layer_matched_random_target", random_heads, target, bias),
            ]
            for name, heads, positions, value in conditions:
                result = generate(heads, positions, value)
                result.setdefault("hook_diagnostics", {})["control_region"] = (
                    wrong_source if name == "candidate_wrong" else "audited_target"
                )
                append_jsonl(
                    records_path,
                    _result_record(
                        sample,
                        name,
                        heads,
                        value,
                        result,
                        str(ranking_artifact["fingerprint"]),
                    ),
                )
            if partition == "validation":
                for count in validation_top_k:
                    grid_heads = _heads(ranking, count)
                    for grid_bias in validation_biases:
                        name = (
                            f"validation_candidate_target_k{count}_bias{grid_bias:g}"
                        )
                        result = generate(grid_heads, target, grid_bias)
                        result.setdefault("hook_diagnostics", {})[
                            "validation_grid"
                        ] = True
                        append_jsonl(
                            records_path,
                            _result_record(
                                sample,
                                name,
                                grid_heads,
                                grid_bias,
                                result,
                                str(ranking_artifact["fingerprint"]),
                            ),
                        )
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    "schema_version": CAUSAL_SCHEMA_VERSION,
                    "example_id": sample["example_id"],
                    "group_id": sample["group_id"],
                    "partition": partition,
                    "status": "invalid",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
    write_json(
        output_dir / f"steering_{partition}_manifest.json",
        {
            "schema_version": CAUSAL_SCHEMA_VERSION,
            "partition": partition,
            "input_manifest": str(Path(cfg["input_manifest"]).resolve()),
            "input_manifest_sha256": sha256_file(cfg["input_manifest"]),
            "ranking_path": str(ranking_path),
            "ranking_fingerprint": ranking_artifact["fingerprint"],
            "top_k": top_k,
            "bias": bias,
            "query_scope": scope,
            "conditions": sorted(expected),
            "validation_top_k": validation_top_k if partition == "validation" else None,
            "validation_biases": validation_biases if partition == "validation" else None,
            "model_fingerprint": artifact_fingerprint(cfg["model_path"]),
            "labels_opened": False,
        },
    )
    return records_path
