#!/usr/bin/env python3
"""Analyze paired attention-mask effects and optional ordinal accuracy.

Continuous paired score shifts are primary.  If benchmark metadata is supplied,
fixed-bin accuracy is reported only as a secondary, post-hoc metric alongside
continuous and interval ordinal errors; reward labels never enter generation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from roborewardbench.data import load_metadata_reference
from roborewardbench.metrics import compute_metrics

from .io import assert_unique, read_jsonl, strict_dump
from .run_experiment import result_key


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def stratified_bootstrap_ci(
    values_by_subset: Mapping[str, Sequence[float]],
    *,
    statistic: Callable[[Sequence[float]], float] = lambda values: sum(values) / len(values),
    samples: int = 10000,
    seed: int = 0,
) -> list[float] | None:
    """Bootstrap examples within subset, then macro-average subset statistics."""

    clean = {
        subset: [float(value) for value in values]
        for subset, values in values_by_subset.items()
        if values
    }
    if samples <= 0 or not clean:
        return None
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        subset_statistics = []
        for subset in sorted(clean):
            values = clean[subset]
            resampled = [rng.choice(values) for _ in values]
            subset_statistics.append(float(statistic(resampled)))
        draws.append(sum(subset_statistics) / len(subset_statistics))
    draws.sort()
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def example_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 10000,
    seed: int = 0,
) -> list[float] | None:
    """Bootstrap the observed examples directly for a paired micro-mean CI.

    This interval remains informative when most subsets contain only one
    example.  It is descriptive for the audited sample and must not be read as
    population uncertainty under purposefully selected sampling.
    """

    clean = [float(value) for value in values]
    if samples <= 0 or not clean:
        return None
    rng = random.Random(seed)
    draws = [
        sum(rng.choice(clean) for _ in clean) / len(clean)
        for _ in range(samples)
    ]
    draws.sort()
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def summarize_values(
    values_by_subset: Mapping[str, Sequence[float]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    per_subset = {
        subset: float(sum(values) / len(values))
        for subset, values in sorted(values_by_subset.items())
        if values
    }
    all_values = [
        float(value)
        for values in values_by_subset.values()
        for value in values
    ]
    result: dict[str, Any] = {
        "n": len(all_values),
        "num_subsets": len(per_subset),
        "micro_mean": _mean(all_values),
        "macro_subset_mean": _mean(list(per_subset.values())),
        "median": (float(statistics.median(all_values)) if all_values else None),
        "sample_std": (
            float(statistics.stdev(all_values)) if len(all_values) >= 2 else None
        ),
        "negative_rate": (
            sum(value < 0.0 for value in all_values) / len(all_values)
            if all_values
            else None
        ),
        "zero_rate": (
            sum(value == 0.0 for value in all_values) / len(all_values)
            if all_values
            else None
        ),
        "positive_rate": (
            sum(value > 0.0 for value in all_values) / len(all_values)
            if all_values
            else None
        ),
        "per_subset_mean": per_subset,
    }
    example_ci = example_bootstrap_mean_ci(
        all_values,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    if example_ci is not None:
        result["micro_mean_example_bootstrap_95ci"] = example_ci
    ci = stratified_bootstrap_ci(
        values_by_subset,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    if ci is not None:
        result["macro_subset_mean_95ci"] = ci
    return result


def config_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("condition")),
        row.get("top_k"),
        float(row.get("swap_bias", 0.0)),
        str(row.get("intervention")),
        str(row.get("target_role")),
    )


def config_name(key: tuple[Any, ...]) -> str:
    condition, top_k, bias, intervention, target_role = key
    top = "all" if top_k is None else str(top_k)
    return (
        f"{condition}|top_k={top}|bias={float(bias):g}|"
        f"{intervention}|role={target_role}"
    )


def _valid_score(row: Mapping[str, Any] | None) -> float | None:
    if row is None or row.get("status") != "ok" or row.get("score") is None:
        return None
    value = float(row["score"])
    return value if math.isfinite(value) else None


def _rows_by_config(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = config_key(row)
        example_id = str(row["example_id"])
        if example_id in grouped[key]:
            raise ValueError(f"Duplicate config/example record: {config_name(key)} {example_id}")
        grouped[key][example_id] = row
    return dict(grouped)


def analyze_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    metadata_records: Mapping[str, Mapping[str, Any]] | None = None,
    shard_completions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No experiment records were provided")
    assert_unique((result_key(row) for row in rows), what="experiment result keys")
    family_signatures = {str(row.get("run_family_signature")) for row in rows}
    if len(family_signatures) != 1:
        raise ValueError("Input results do not belong to one experiment family")
    grouped = _rows_by_config(rows)
    baseline_keys = [key for key in grouped if key[0] == "baseline"]
    if len(baseline_keys) != 1:
        raise ValueError(f"Expected one baseline configuration, found {len(baseline_keys)}")
    baseline_rows = grouped[baseline_keys[0]]

    configurations: dict[str, Any] = {}
    paired_shift_cache: dict[tuple[Any, ...], dict[str, float]] = {}
    expected_example_ids = set(baseline_rows)
    for index, key in enumerate(sorted(grouped, key=config_name)):
        records = grouped[key]
        statuses = Counter(str(row.get("status")) for row in records.values())
        missing_examples = sorted(expected_example_ids - set(records))
        valid_scores_by_subset: dict[str, list[float]] = defaultdict(list)
        shifts_by_subset: dict[str, list[float]] = defaultdict(list)
        absolute_shifts_by_subset: dict[str, list[float]] = defaultdict(list)
        paired: dict[str, float] = {}
        for example_id, row in records.items():
            score = _valid_score(row)
            if score is None:
                continue
            subset = str(row.get("subset", "unknown"))
            valid_scores_by_subset[subset].append(score)
            baseline = _valid_score(baseline_rows.get(example_id))
            if baseline is not None:
                shift = score - baseline
                paired[example_id] = shift
                shifts_by_subset[subset].append(shift)
                absolute_shifts_by_subset[subset].append(abs(shift))
        paired_shift_cache[key] = paired
        configurations[config_name(key)] = {
            "condition": key[0],
            "top_k": key[1],
            "swap_bias": key[2],
            "intervention": key[3],
            "target_role": key[4],
            "num_records": len(records),
            "expected_num_records": len(expected_example_ids),
            "missing_example_count": len(missing_examples),
            "missing_example_ids": missing_examples,
            "coverage_rate": (
                len(records) / len(expected_example_ids)
                if expected_example_ids
                else None
            ),
            "status_counts": dict(sorted(statuses.items())),
            "invalid_rate": (
                (len(records) - statuses.get("ok", 0)) / len(records)
                if records
                else None
            ),
            "invalid_or_missing_rate": (
                1.0 - statuses.get("ok", 0) / len(expected_example_ids)
                if expected_example_ids
                else None
            ),
            "score": summarize_values(
                valid_scores_by_subset,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + index,
            ),
            "paired_score_shift_vs_baseline": summarize_values(
                shifts_by_subset,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 1000 + index,
            ),
            "paired_absolute_shift_vs_baseline": summarize_values(
                absolute_shifts_by_subset,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 2000 + index,
            ),
        }

    contrasts: dict[str, Any] = {}
    candidate_keys = [key for key in grouped if key[0] == "candidate_target"]
    for index, candidate_key in enumerate(sorted(candidate_keys, key=config_name)):
        _condition, top_k, bias, intervention, target_role = candidate_key
        controls = {
            "wrong_region": (
                "candidate_wrong",
                top_k,
                bias,
                intervention,
                target_role,
            ),
            "low_rank_heads": (
                "low_rank_target",
                top_k,
                bias,
                intervention,
                target_role,
            ),
            "all_heads": (
                "all_target",
                None,
                bias,
                intervention,
                target_role,
            ),
        }
        for control_name, control_key in controls.items():
            if control_key not in paired_shift_cache:
                continue
            candidate_shifts = paired_shift_cache[candidate_key]
            control_shifts = paired_shift_cache[control_key]
            common_ids = sorted(set(candidate_shifts) & set(control_shifts))
            signed_by_subset: dict[str, list[float]] = defaultdict(list)
            leverage_by_subset: dict[str, list[float]] = defaultdict(list)
            opposite_directions = 0
            candidate_larger = 0
            for example_id in common_ids:
                subset = str(grouped[candidate_key][example_id].get("subset", "unknown"))
                candidate_shift = candidate_shifts[example_id]
                control_shift = control_shifts[example_id]
                signed_by_subset[subset].append(candidate_shift - control_shift)
                leverage_by_subset[subset].append(
                    abs(candidate_shift) - abs(control_shift)
                )
                opposite_directions += int(candidate_shift * control_shift < 0.0)
                candidate_larger += int(abs(candidate_shift) > abs(control_shift))
            name = f"{config_name(candidate_key)} vs {control_name}"
            contrasts[name] = {
                "candidate_config": config_name(candidate_key),
                "control_config": config_name(control_key),
                "paired_ids": len(common_ids),
                "strict_opposite_direction_rate": (
                    opposite_directions / len(common_ids) if common_ids else None
                ),
                "candidate_larger_absolute_shift_rate": (
                    candidate_larger / len(common_ids) if common_ids else None
                ),
                "signed_difference_of_shifts": summarize_values(
                    signed_by_subset,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + 3000 + index,
                ),
                "absolute_leverage_difference": summarize_values(
                    leverage_by_subset,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed + 4000 + index,
                ),
            }

    zero_bias_differences: list[float] = []
    for key, shifts in paired_shift_cache.items():
        if key[0] != "baseline" and float(key[2]) == 0.0:
            zero_bias_differences.extend(abs(value) for value in shifts.values())

    ordinal: dict[str, Any] | None = None
    if metadata_records is not None:
        ordinal = {}
        for key, records in grouped.items():
            joined: list[dict[str, Any]] = []
            missing_metadata: list[str] = []
            for example_id, row in records.items():
                metadata = metadata_records.get(example_id)
                if metadata is None:
                    missing_metadata.append(example_id)
                    continue
                status = str(row.get("status"))
                joined_row: dict[str, Any] = {
                    "id": example_id,
                    "subset": str(metadata["subset"]),
                    "reward": int(metadata["reward"]),
                    "task": str(metadata["task"]),
                    "split": "test",
                    "status": status,
                }
                score = _valid_score(row)
                if score is not None:
                    joined_row["progress"] = min(max(score, 0.0), 1.0)
                joined.append(joined_row)
            valid_joined = [row for row in joined if row["status"] == "ok"]
            if not valid_joined:
                ordinal[config_name(key)] = {
                    "error": "no valid metadata-joined predictions",
                    "missing_metadata_ids": missing_metadata,
                }
                continue
            computed = compute_metrics(
                joined,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            ordinal[config_name(key)] = {
                "num_records": computed["num_records"],
                "num_valid": computed["num_valid"],
                "invalid_rate": computed["invalid_rate"],
                "fixed_bin_exact_accuracy": computed["discrete_classification"][
                    "exact_accuracy"
                ],
                "fixed_bin_within_one_accuracy": computed["discrete_classification"][
                    "within_one_accuracy"
                ],
                "fixed_bin_mae": computed["benchmark_compatible_fixed_bin"],
                "continuous_ordinal_mae": computed["continuous_ordinal"],
                "interval_ordinal_mae": computed["interval_ordinal"],
                "missing_metadata_ids": missing_metadata,
            }

    completeness: dict[str, Any]
    if shard_completions is None:
        completeness = {
            "verified": False,
            "complete": None,
            "reason": "shard completion records were not supplied",
        }
    else:
        completion_families = {
            str(value.get("run_family_signature")) for value in shard_completions
        }
        shard_counts = {int(value.get("num_shards", -1)) for value in shard_completions}
        shard_indices = {int(value.get("shard_index", -1)) for value in shard_completions}
        expected_shards = next(iter(shard_counts)) if len(shard_counts) == 1 else -1
        expected_indices = set(range(expected_shards)) if expected_shards > 0 else set()
        all_success = all(bool(value.get("complete_shard")) for value in shard_completions)
        family_match = completion_families == family_signatures
        record_counts_match = all(
            int(value.get("result_record_count", -1))
            == int(value.get("_observed_result_record_count", -2))
            for value in shard_completions
        )
        signatures_match = all(
            {str(item) for item in value.get("_observed_run_signatures", [])}
            == {str(value.get("run_signature"))}
            for value in shard_completions
        )
        selected_ids_match = all(
            {str(item) for item in value.get("_observed_example_ids", [])}
            == {str(item) for item in value.get("selected_ids", [])}
            for value in shard_completions
        )
        complete = bool(
            all_success
            and family_match
            and record_counts_match
            and signatures_match
            and selected_ids_match
            and len(shard_counts) == 1
            and shard_indices == expected_indices
        )
        completeness = {
            "verified": True,
            "complete": complete,
            "all_shards_marked_complete": all_success,
            "family_match": family_match,
            "record_counts_match": record_counts_match,
            "run_signatures_match": signatures_match,
            "selected_ids_match": selected_ids_match,
            "expected_num_shards": expected_shards,
            "observed_shard_indices": sorted(shard_indices),
            "expected_shard_indices": sorted(expected_indices),
        }

    return {
        "schema_version": 1,
        "run_family_signature": next(iter(family_signatures)),
        "num_input_records": len(rows),
        "num_examples": len({str(row["example_id"]) for row in rows}),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "resampling": "examples within subset; equal-weight macro average over subsets",
        },
        "input_completeness": completeness,
        "primary_estimand": (
            "Within-example continuous GRM score shift relative to baseline; "
            "candidate-target effects are contrasted with wrong-region and "
            "low-ranked-head controls."
        ),
        "configurations": configurations,
        "paired_control_contrasts": contrasts,
        "zero_bias_invariance_check": {
            "n": len(zero_bias_differences),
            "max_absolute_difference": (
                max(zero_bias_differences) if zero_bias_differences else None
            ),
            "passed_exactly": (
                all(value == 0.0 for value in zero_bias_differences)
                if zero_bias_differences
                else None
            ),
        },
        "posthoc_ordinal_metrics": ordinal,
        "interpretation_boundaries": [
            "Ground-truth reward is joined only after model generation.",
            "Fixed-bin exact accuracy depends on arbitrary discretization thresholds; "
            "continuous and interval ordinal errors are reported alongside it.",
            "This purposefully audited held-out subset is not an official full-test score.",
            "Bootstrap intervals describe stability within the 19 audited examples; "
            "purposeful sampling does not support population-level confidence claims.",
            "The subset-stratified macro interval is conditional on the observed subsets "
            "and may be narrow because most subsets contain one example; use the direct "
            "example-bootstrap micro interval as the primary uncertainty summary.",
            "A causal claim requires candidate-target effects to exceed both wrong-region "
            "and low-ranked-head controls with adequate paired coverage.",
        ],
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# RoboRewardBench attention-mask experiment",
        "",
        f"- Examples: {result['num_examples']}",
        f"- Result records: {result['num_input_records']}",
        f"- Run family: `{result['run_family_signature']}`",
        "",
        "## Paired continuous effects",
        "",
        "| Configuration | valid / expected | mean shift | macro shift | example-bootstrap 95% CI | mean abs(shift) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, block in result["configurations"].items():
        shift = block["paired_score_shift_vs_baseline"]
        absolute = block["paired_absolute_shift_vs_baseline"]
        valid = block["status_counts"].get("ok", 0)
        ci = shift.get("micro_mean_example_bootstrap_95ci")
        ci_text = (
            f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "N/A"
        )
        lines.append(
            f"| `{name}` | {valid}/{block['expected_num_records']} | "
            f"{_format(shift['micro_mean'])} | {_format(shift['macro_subset_mean'])} | "
            f"{ci_text} | {_format(absolute['micro_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Control contrasts",
            "",
            "| Candidate vs control | paired n | signed Δshift | absolute leverage Δ |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, block in result["paired_control_contrasts"].items():
        signed = block["signed_difference_of_shifts"]["macro_subset_mean"]
        leverage = block["absolute_leverage_difference"]["macro_subset_mean"]
        lines.append(
            f"| `{name}` | {block['paired_ids']} | {_format(signed)} | {_format(leverage)} |"
        )

    invariant = result["zero_bias_invariance_check"]
    completeness = result["input_completeness"]
    lines.extend(
        [
            "",
            "## Sanity check",
            "",
            f"- Complete formal input: `{completeness.get('complete')}` "
            f"(verified={completeness.get('verified')}).",
            f"- Zero-bias exact invariance: `{invariant['passed_exactly']}`; "
            f"max absolute difference = `{_format(invariant['max_absolute_difference'])}`.",
        ]
    )
    if result.get("posthoc_ordinal_metrics") is not None:
        lines.extend(
            [
                "",
                "## Post-hoc ordinal metrics",
                "",
                "Accuracy is threshold-dependent. Consult the JSON for fixed-bin exact "
                "accuracy together with continuous and interval ordinal MAE.",
            ]
        )
    lines.extend(["", "## Interpretation boundaries", ""])
    lines.extend(f"- {value}" for value in result["interpretation_boundaries"])
    return "\n".join(lines) + "\n"


def _format(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = []
    completions = []
    all_completions_present = True
    for result_path in args.results:
        shard_rows = read_jsonl(result_path)
        rows.extend(shard_rows)
        completion_path = Path(result_path).expanduser().resolve().parent / "completion.json"
        if not completion_path.is_file():
            all_completions_present = False
            continue
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["_observed_result_record_count"] = len(shard_rows)
        completion["_observed_run_signatures"] = sorted(
            {str(row.get("run_signature")) for row in shard_rows}
        )
        completion["_observed_example_ids"] = sorted(
            {str(row.get("example_id")) for row in shard_rows}
        )
        completions.append(completion)
    metadata = (
        load_metadata_reference(args.metadata)["records"]
        if args.metadata is not None
        else None
    )
    result = analyze_records(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        metadata_records=metadata,
        shard_completions=(
            completions if all_completions_present and completions else None
        ),
    )
    strict_dump(result, args.output)
    markdown_path = (
        Path(args.markdown).expanduser().resolve()
        if args.markdown
        else Path(args.output).expanduser().resolve().with_suffix(".md")
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
