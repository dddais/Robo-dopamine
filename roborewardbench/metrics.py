"""Metrics and validation-only calibration for RoboRewardBench.

The official benchmark requires an integer prediction in ``{1, ..., 5}``.
Continuous Robo-Dopamine progress is therefore reported in three ways:

1. benchmark-compatible fixed-bin MAE for leaderboard comparability;
2. continuous ordinal MAE, which does not round predictions;
3. interval ordinal MAE, which does not penalize a continuous prediction
   anywhere inside the ground-truth label's fixed quantization cell.

An optional monotonic calibrator may be fitted on validation predictions and
then frozen before evaluating the test set. It must never be fitted on test.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from roborewardbench.data import (
    CATEGORY_NAMES_ZH,
    CATEGORY_ORDER,
    classify_source,
    metadata_matches_records,
)

DEFAULT_THRESHOLDS = (0.125, 0.375, 0.625, 0.875)
VALID_LABELS = (1, 2, 3, 4, 5)
EXPECTED_TEST_SUBSETS = frozenset({
    "austin_sirius_dataset_converted_externally_to_rlds",
    "berkeley_autolab_ur5",
    "berkeley_fanuc_manipulation",
    "berkeley_mvp_converted_externally_to_rlds",
    "berkeley_rpt_converted_externally_to_rlds",
    "bridge",
    "cmu_play_fusion",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "droid",
    "fractal20220817_data",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "jaco_play",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "robo_arena",
    "roboturk",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "taco_play",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "viola",
})


def _clip_progress(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Progress must be finite, got {value!r}")
    return min(max(value, 0.0), 1.0)


def _validate_label(value: Any) -> int:
    label = int(value)
    if label not in VALID_LABELS or float(value) != label:
        raise ValueError(f"Ground-truth reward must be an integer in 1..5, got {value!r}")
    return label


def _validate_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in thresholds)
    if len(values) != 4:
        raise ValueError(f"Expected four thresholds, got {len(values)}")
    if not all(0.0 <= x <= 1.0 for x in values):
        raise ValueError("Thresholds must lie in [0, 1]")
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError("Thresholds must be non-decreasing")
    return values


def apply_thresholds(progress: float, thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> int:
    """Map progress to 1..5 using explicit half-open threshold intervals."""

    value = _clip_progress(progress)
    cuts = _validate_thresholds(thresholds)
    return 1 + sum(value >= threshold for threshold in cuts)


def continuous_ordinal_error(progress: float, label: int) -> float:
    """Absolute error on the unrounded ordinal coordinate ``1 + 4p``."""

    return abs((1.0 + 4.0 * _clip_progress(progress)) - _validate_label(label))


def interval_ordinal_error(progress: float, label: int) -> float:
    """Distance to the label's fixed quantization cell, measured in label units.

    This is zero throughout the region that fixed rounding maps to the correct
    class. Outside that region it grows continuously rather than jumping by one.
    """

    ordinal = 1.0 + 4.0 * _clip_progress(progress)
    target = _validate_label(label)
    lower = 1.0 if target == 1 else target - 0.5
    upper = 5.0 if target == 5 else target + 0.5
    if ordinal < lower:
        return lower - ordinal
    if ordinal > upper:
        return ordinal - upper
    return 0.0


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a mean over an empty sequence")
    return float(sum(values) / len(values))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile over an empty sequence")
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _macro_bootstrap_ci(
    per_subset_errors: Mapping[str, Sequence[float]],
    samples: int,
    seed: int,
) -> list[float] | None:
    if samples <= 0:
        return None
    rng = random.Random(seed)
    subset_names = sorted(per_subset_errors)
    bootstrapped = []
    for _ in range(samples):
        subset_means = []
        for subset in subset_names:
            errors = per_subset_errors[subset]
            subset_means.append(_mean([rng.choice(errors) for _ in range(len(errors))]))
        bootstrapped.append(_mean(subset_means))
    bootstrapped.sort()
    return [_percentile(bootstrapped, 0.025), _percentile(bootstrapped, 0.975)]


def compute_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    progress_field: str = "progress",
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
    expected_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute official and continuous metrics from valid prediction records.

    Every record must contain ``progress``, ``reward``, and ``subset``. Records
    with a non-``ok`` status are counted but excluded from metric denominators.
    """

    cuts = _validate_thresholds(thresholds)
    rows = list(records)
    valid_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if not valid_rows:
        raise ValueError("No valid prediction records were provided")

    official_by_subset: dict[str, list[float]] = defaultdict(list)
    continuous_by_subset: dict[str, list[float]] = defaultdict(list)
    interval_by_subset: dict[str, list[float]] = defaultdict(list)
    official_by_category: dict[str, list[float]] = defaultdict(list)
    continuous_by_category: dict[str, list[float]] = defaultdict(list)
    interval_by_category: dict[str, list[float]] = defaultdict(list)
    exact_by_subset: dict[str, list[float]] = defaultdict(list)
    within_one_by_subset: dict[str, list[float]] = defaultdict(list)
    label_stats: dict[int, list[float]] = defaultdict(list)
    prediction_counts: Counter[int] = Counter()
    true_label_counts: Counter[int] = Counter()
    confusion_counts: dict[int, Counter[int]] = defaultdict(Counter)
    signed_errors: list[float] = []

    for row in valid_rows:
        progress = _clip_progress(row[progress_field])
        label = _validate_label(row["reward"])
        subset = str(row["subset"])
        predicted_label = apply_thresholds(progress, cuts)
        official_error = float(abs(predicted_label - label))
        continuous_error = continuous_ordinal_error(progress, label)
        interval_error = interval_ordinal_error(progress, label)
        official_by_subset[subset].append(official_error)
        continuous_by_subset[subset].append(continuous_error)
        interval_by_subset[subset].append(interval_error)
        exact_by_subset[subset].append(float(predicted_label == label))
        within_one_by_subset[subset].append(float(official_error <= 1.0))
        prediction_counts[predicted_label] += 1
        true_label_counts[label] += 1
        confusion_counts[label][predicted_label] += 1
        signed_errors.append(float(predicted_label - label))
        if row.get("id") is not None:
            category = classify_source(str(row["id"]), label)
            official_by_category[category].append(official_error)
            continuous_by_category[category].append(continuous_error)
            interval_by_category[category].append(interval_error)
        label_stats[label].append(float(predicted_label))

    def summarize(by_subset: Mapping[str, Sequence[float]]) -> dict[str, Any]:
        subset_mae = {name: _mean(errors) for name, errors in sorted(by_subset.items())}
        all_errors = [error for errors in by_subset.values() for error in errors]
        result: dict[str, Any] = {
            "macro_mae": _mean(list(subset_mae.values())),
            "micro_mae": _mean(all_errors),
            "per_subset_mae": subset_mae,
        }
        ci = _macro_bootstrap_ci(by_subset, bootstrap_samples, bootstrap_seed)
        if ci is not None:
            result["macro_mae_95ci"] = ci
        return result

    official = summarize(official_by_subset)
    continuous = summarize(continuous_by_subset)
    interval = summarize(interval_by_subset)

    def summarize_accuracy(by_subset: Mapping[str, Sequence[float]]) -> dict[str, Any]:
        per_subset = {name: _mean(values) for name, values in sorted(by_subset.items())}
        all_values = [value for values in by_subset.values() for value in values]
        result: dict[str, Any] = {
            "macro_accuracy": _mean(list(per_subset.values())),
            "micro_accuracy": _mean(all_values),
            "per_subset_accuracy": per_subset,
        }
        ci = _macro_bootstrap_ci(by_subset, bootstrap_samples, bootstrap_seed)
        if ci is not None:
            result["macro_accuracy_95ci"] = ci
        return result

    discrete_classification = {
        "definition": "Fixed-bin ordinal labels using the thresholds reported above.",
        "exact_accuracy": summarize_accuracy(exact_by_subset),
        "within_one_accuracy": summarize_accuracy(within_one_by_subset),
        "mean_predicted_label": sum(
            label * count for label, count in prediction_counts.items()
        ) / len(valid_rows),
        "mean_true_label": sum(
            label * count for label, count in true_label_counts.items()
        ) / len(valid_rows),
        "mean_signed_error": _mean(signed_errors),
        "overprediction_rate": _mean([float(error > 0.0) for error in signed_errors]),
        "underprediction_rate": _mean([float(error < 0.0) for error in signed_errors]),
        "prediction_counts": {
            str(label): prediction_counts[label] for label in VALID_LABELS
        },
        "true_label_counts": {
            str(label): true_label_counts[label] for label in VALID_LABELS
        },
        "confusion_matrix": {
            str(true_label): {
                str(predicted_label): confusion_counts[true_label][predicted_label]
                for predicted_label in VALID_LABELS
            }
            for true_label in VALID_LABELS
        },
    }
    source_category_metrics = {
        category: {
            "name_zh": CATEGORY_NAMES_ZH[category],
            "count": len(official_by_category[category]),
            "benchmark_compatible_fixed_bin_mae": _mean(official_by_category[category]),
            "continuous_ordinal_mae": _mean(continuous_by_category[category]),
            "interval_ordinal_mae": _mean(interval_by_category[category]),
        }
        for category in CATEGORY_ORDER
        if official_by_category[category]
    }
    oxe_subsets = [name for name in official_by_subset if name != "robo_arena"]
    observed_subsets = set(official_by_subset)
    splits = {str(row.get("split", "")).lower() for row in valid_rows}
    rows_by_id = {
        str(row["id"]): row for row in rows if row.get("id") is not None
    }
    metadata_exact_match = False
    metadata_match_problems = ["metadata_not_provided"]
    if expected_records is not None:
        metadata_exact_match, metadata_match_problems = metadata_matches_records(
            rows_by_id,
            expected_records,
        )
    official_comparable = (
        len(rows) == 2831
        and len(valid_rows) == 2831
        and len(rows_by_id) == 2831
        and observed_subsets == EXPECTED_TEST_SUBSETS
        and splits == {"test"}
        and metadata_exact_match
    )

    return {
        "num_records": len(rows),
        "num_valid": len(valid_rows),
        "num_invalid": len(rows) - len(valid_rows),
        "invalid_rate": (len(rows) - len(valid_rows)) / len(rows) if rows else 0.0,
        "num_subsets": len(official_by_subset),
        "missing_expected_subsets": sorted(EXPECTED_TEST_SUBSETS - observed_subsets),
        "unexpected_subsets": sorted(observed_subsets - EXPECTED_TEST_SUBSETS),
        "metric_scope": "valid_predictions_only",
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "resampling_unit": "examples_within_each_subset",
        },
        "official_comparable": official_comparable,
        "metadata_exact_match": metadata_exact_match,
        "metadata_match_problems": metadata_match_problems[:20],
        "official_comparability_requirements": {
            "split": "test",
            "num_records": 2831,
            "num_valid": 2831,
            "num_unique_ids": 2831,
            "subsets": sorted(EXPECTED_TEST_SUBSETS),
            "exact_metadata_identity_and_labels": True,
        },
        "thresholds": list(cuts),
        "progress_field": progress_field,
        "benchmark_compatible_fixed_bin": official,
        "discrete_classification": discrete_classification,
        "continuous_ordinal": continuous,
        "interval_ordinal": interval,
        "source_category_metrics": source_category_metrics,
        "roboarena_mae": (
            _mean(official_by_subset["robo_arena"]) if "robo_arena" in official_by_subset else None
        ),
        "oxe_macro_mae": (
            _mean([_mean(official_by_subset[name]) for name in oxe_subsets]) if oxe_subsets else None
        ),
        "prediction_mean_by_label": {
            str(label): _mean(predictions) for label, predictions in sorted(label_stats.items())
        },
    }


def _pav_isotonic(values: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[float]]:
    """Fit an increasing least-squares isotonic mapping with PAV."""

    ordered = sorted(zip(values, labels), key=lambda item: item[0])
    unique_x: list[float] = []
    sums: list[float] = []
    weights: list[int] = []
    for x, label in ordered:
        x = _clip_progress(x)
        target = (_validate_label(label) - 1) / 4.0
        if unique_x and x == unique_x[-1]:
            sums[-1] += target
            weights[-1] += 1
        else:
            unique_x.append(x)
            sums.append(target)
            weights.append(1)

    blocks: list[dict[str, float | int]] = []
    for idx, (total, weight) in enumerate(zip(sums, weights)):
        blocks.append({"start": idx, "end": idx, "sum": total, "weight": weight})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left["sum"]) / int(left["weight"])
            right_mean = float(right["sum"]) / int(right["weight"])
            if left_mean <= right_mean:
                break
            blocks[-2:] = [{
                "start": int(left["start"]),
                "end": int(right["end"]),
                "sum": float(left["sum"]) + float(right["sum"]),
                "weight": int(left["weight"]) + int(right["weight"]),
            }]

    fitted = [0.0] * len(unique_x)
    for block in blocks:
        mean = float(block["sum"]) / int(block["weight"])
        for idx in range(int(block["start"]), int(block["end"]) + 1):
            fitted[idx] = mean
    return unique_x, fitted


def _interpolate(x: float, knots_x: Sequence[float], knots_y: Sequence[float]) -> float:
    if x <= knots_x[0]:
        return float(knots_y[0])
    if x >= knots_x[-1]:
        return float(knots_y[-1])
    for idx in range(1, len(knots_x)):
        if x <= knots_x[idx]:
            x0, x1 = knots_x[idx - 1], knots_x[idx]
            y0, y1 = knots_y[idx - 1], knots_y[idx]
            if x1 == x0:
                return float(y1)
            alpha = (x - x0) / (x1 - x0)
            return float(y0 + alpha * (y1 - y0))
    raise AssertionError("Interpolation fell through")


def fit_monotonic_calibration(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit a monotonic calibration map from validation predictions only."""

    rows = [row for row in records if row.get("status", "ok") == "ok"]
    if not rows:
        raise ValueError("No valid validation predictions were provided")
    splits = {str(row.get("split", "")).lower() for row in rows}
    if splits != {"validation"} and splits != {"val"}:
        raise ValueError(
            "Calibration input must contain only validation records with split='validation' or split='val'"
        )

    knots_x, knots_y = _pav_isotonic(
        [float(row["progress"]) for row in rows],
        [int(row["reward"]) for row in rows],
    )
    if len(knots_x) == 1:
        thresholds = list(DEFAULT_THRESHOLDS)
    else:
        thresholds = []
        for target in DEFAULT_THRESHOLDS:
            candidates = []
            for idx in range(1, len(knots_x)):
                y0, y1 = knots_y[idx - 1], knots_y[idx]
                if y0 <= target <= y1:
                    if y1 == y0:
                        candidates.append((knots_x[idx - 1] + knots_x[idx]) / 2.0)
                    else:
                        alpha = (target - y0) / (y1 - y0)
                        candidates.append(knots_x[idx - 1] + alpha * (knots_x[idx] - knots_x[idx - 1]))
            if candidates:
                thresholds.append(float(candidates[0]))
            elif target < knots_y[0]:
                thresholds.append(0.0)
            else:
                thresholds.append(1.0)

    return {
        "schema_version": 1,
        "fit_split": "validation",
        "method": "isotonic_pav",
        "num_examples": len(rows),
        "knots_x": [0.0, *knots_x, 1.0] if knots_x[0] > 0.0 and knots_x[-1] < 1.0 else (
            [0.0, *knots_x] if knots_x[0] > 0.0 else (
                [*knots_x, 1.0] if knots_x[-1] < 1.0 else knots_x
            )
        ),
        "knots_y": [0.0, *knots_y, 1.0] if knots_x[0] > 0.0 and knots_x[-1] < 1.0 else (
            [0.0, *knots_y] if knots_x[0] > 0.0 else (
                [*knots_y, 1.0] if knots_x[-1] < 1.0 else knots_y
            )
        ),
        "thresholds": list(_validate_thresholds(thresholds)),
    }


def calibrate_progress(progress: float, calibration: Mapping[str, Any]) -> float:
    """Apply a saved monotonic validation calibrator to a progress value."""

    knots_x = [float(x) for x in calibration["knots_x"]]
    knots_y = [float(y) for y in calibration["knots_y"]]
    if not knots_x or len(knots_x) != len(knots_y):
        raise ValueError("Invalid calibration knots")
    return _clip_progress(_interpolate(_clip_progress(progress), knots_x, knots_y))


def save_calibration(calibration: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(dict(calibration), handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def load_calibration(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("fit_split") != "validation":
        raise ValueError("Calibration artifact was not marked as validation-fitted")
    _validate_thresholds(calibration["thresholds"])
    return calibration
