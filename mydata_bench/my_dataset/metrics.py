"""Group-aware scoring for same-video instruction counterfactuals.

Inference records are deliberately label-free.  This module is the only part
of the custom-dataset baseline pipeline that opens the scoring labels and joins
them to completed model outputs.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

from ..io import latest_by_id, read_jsonl, write_json, write_jsonl
from ..protocol import progress_to_reward
from .data import load_labels, load_model_inputs


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _load_latest_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(run_dir.glob("records.shard-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No shard records found under {run_dir}")
    latest: dict[str, dict[str, Any]] = {}
    for path in paths:
        latest.update(latest_by_id(read_jsonl(path)))
    return latest


def _join_records(
    records: dict[str, dict[str, Any]],
    inputs: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_by_id = {str(row["example_id"]): row for row in inputs}
    label_by_id = {str(row["example_id"]): row for row in labels}
    expected_ids = set(input_by_id)
    if expected_ids != set(label_by_id):
        raise ValueError("Model-input and scoring-label IDs differ")

    invalid = [row for row in records.values() if row.get("status") != "ok"]
    missing = sorted(expected_ids - set(records))
    unexpected = sorted(set(records) - expected_ids)
    joined: list[dict[str, Any]] = []
    for example_id in sorted(expected_ids & set(records)):
        record = records[example_id]
        if record.get("status") != "ok" or record.get("progress") is None:
            continue
        input_row = input_by_id[example_id]
        label = label_by_id[example_id]
        if str(record.get("group_id")) != str(input_row["group_id"]):
            raise ValueError(f"Output group mismatch for {example_id}")
        progress = min(1.0, max(0.0, float(record["progress"])))
        prediction = int(record.get("native_prediction", progress_to_reward(progress)))
        reward = int(label["protocol_reward"])
        joined.append(
            {
                "example_id": example_id,
                "group_id": str(input_row["group_id"]),
                "task_id": str(input_row["task_id"]),
                "task_family": str(input_row["task_family"]),
                "instruction": str(input_row["instruction"]),
                "instruction_video_match": bool(label["instruction_video_match"]),
                "protocol_reward": reward,
                "prediction": prediction,
                "progress": progress,
                "continuous_ordinal_prediction": 1 + 4 * progress,
                "absolute_error": abs(prediction - reward),
                "continuous_absolute_error": abs(1 + 4 * progress - reward),
                "correct": prediction == reward,
                "raw_output": record.get("raw_output"),
                "signed_score": record.get("signed_score"),
                "model_family": record.get("model_family"),
                "protocol": record.get("protocol"),
            }
        )
    completion = {
        "expected_count": len(expected_ids),
        "record_count": len(records),
        "valid_count": len(joined),
        "invalid_count": len(invalid),
        "missing_example_ids": missing,
        "unexpected_example_ids": unexpected,
        "formal_scoring_ready": not invalid and not missing and not unexpected,
    }
    return joined, completion


def _example_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    absolute = [float(row["absolute_error"]) for row in rows]
    continuous = [float(row["continuous_absolute_error"]) for row in rows]
    return {
        "n": len(rows),
        "mae": mean(absolute),
        "continuous_ordinal_mae": mean(continuous),
        "exact_accuracy": mean(bool(row["correct"]) for row in rows),
        "within_one_accuracy": mean(value <= 1 for value in absolute),
        "mean_prediction": mean(float(row["prediction"]) for row in rows),
        "mean_progress": mean(float(row["progress"]) for row in rows),
    }


def _build_group_rows(
    joined: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[str(row["group_id"])].append(row)
    group_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for group_id, rows in sorted(grouped.items()):
        matched = [row for row in rows if row["instruction_video_match"]]
        counterfactuals = [row for row in rows if not row["instruction_video_match"]]
        if len(matched) != 1 or not counterfactuals:
            raise ValueError(
                f"Group {group_id} must contain one matched and at least one counterfactual row"
            )
        original = matched[0]
        margins = [
            float(original["progress"]) - float(counter["progress"])
            for counter in counterfactuals
        ]
        signed_margins = [
            float(original["signed_score"]) - float(counter["signed_score"])
            for counter in counterfactuals
            if original.get("signed_score") is not None
            and counter.get("signed_score") is not None
        ]
        max_counter = max(float(row["progress"]) for row in counterfactuals)
        pairwise_wins = [value > 0 for value in margins]
        pairwise_ties = [value == 0 for value in margins]
        for counter, margin in zip(counterfactuals, margins):
            pair_rows.append(
                {
                    "group_id": group_id,
                    "task_id": original["task_id"],
                    "task_family": original["task_family"],
                    "matched_example_id": original["example_id"],
                    "counterfactual_example_id": counter["example_id"],
                    "matched_progress": original["progress"],
                    "counterfactual_progress": counter["progress"],
                    "progress_margin": margin,
                    "matched_greater": margin > 0,
                    "tie": margin == 0,
                }
            )
        group_rows.append(
            {
                "group_id": group_id,
                "task_id": original["task_id"],
                "task_family": original["task_family"],
                "num_counterfactuals": len(counterfactuals),
                "matched_example_id": original["example_id"],
                "matched_prediction": original["prediction"],
                "matched_progress": original["progress"],
                "matched_reward5_correct": original["prediction"] == 5,
                "counterfactual_predicted_one_rate": mean(
                    row["prediction"] == 1 for row in counterfactuals
                ),
                "counterfactual_overprediction_rate": mean(
                    row["prediction"] > 1 for row in counterfactuals
                ),
                "counterfactual_high_false_positive_rate": mean(
                    row["prediction"] >= 4 for row in counterfactuals
                ),
                "group_exact_accuracy": mean(bool(row["correct"]) for row in rows),
                "group_mae": mean(float(row["absolute_error"]) for row in rows),
                "group_continuous_ordinal_mae": mean(
                    float(row["continuous_absolute_error"]) for row in rows
                ),
                "pairwise_accuracy": mean(pairwise_wins),
                "pairwise_tie_rate": mean(pairwise_ties),
                "mean_progress_margin": mean(margins),
                "minimum_progress_margin": min(margins),
                "mean_signed_score_margin": mean(signed_margins)
                if signed_margins
                else None,
                "strict_top1_correct": float(original["progress"]) > max_counter,
                "top1_including_ties": float(original["progress"]) >= max_counter,
            }
        )
    return group_rows, pair_rows


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_groups": 0}
    numeric_fields = (
        "pairwise_accuracy",
        "pairwise_tie_rate",
        "mean_progress_margin",
        "minimum_progress_margin",
        "strict_top1_correct",
        "top1_including_ties",
        "matched_reward5_correct",
        "counterfactual_predicted_one_rate",
        "counterfactual_overprediction_rate",
        "counterfactual_high_false_positive_rate",
        "group_exact_accuracy",
        "group_mae",
        "group_continuous_ordinal_mae",
    )
    result: dict[str, Any] = {"n_groups": len(rows)}
    for field in numeric_fields:
        result[field] = mean(float(row[field]) for row in rows)
    signed = [
        float(row["mean_signed_score_margin"])
        for row in rows
        if row.get("mean_signed_score_margin") is not None
    ]
    result["mean_signed_score_margin"] = _mean_or_none(signed)
    return result


def _stratified_group_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["task_id"])].append(row)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw: list[dict[str, Any]] = []
        for task_id in sorted(strata):
            values = strata[task_id]
            draw.extend(rng.choice(values) for _ in values)
        estimates.append(float(statistic(draw)))
    return {
        "estimate": float(statistic(rows)),
        "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "samples": samples,
        "seed": seed,
        "cluster": "source_group",
        "stratum": "task_id",
    }


def _by_group_field(
    group_rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        grouped[str(row[field])].append(row)
    return {key: _group_summary(values) for key, values in sorted(grouped.items())}


def _by_example_field(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: _example_metrics(values) for key, values in sorted(grouped.items())}


def score_run(
    run_dir: str | Path,
    *,
    inputs_path: str | Path,
    labels_path: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 20260803,
) -> dict[str, Any]:
    """Join labels after inference and write group-aware baseline metrics."""
    run_dir = Path(run_dir).resolve()
    inputs = load_model_inputs(inputs_path)
    labels = load_labels(labels_path)
    records = _load_latest_records(run_dir)
    joined, completion = _join_records(records, inputs, labels)
    group_rows, pair_rows = _build_group_rows(joined) if joined else ([], [])

    matched = [row for row in joined if row["instruction_video_match"]]
    counterfactuals = [row for row in joined if not row["instruction_video_match"]]
    confusion = {str(y): {str(p): 0 for p in range(1, 6)} for y in (1, 5)}
    for row in joined:
        confusion[str(row["protocol_reward"])][str(row["prediction"])] += 1
    result: dict[str, Any] = {
        "metric_contract": "same_video_counterfactual_group_v1",
        "completion": completion,
        "num_examples": len(joined),
        "num_groups": len(group_rows),
        "num_pairs": len(pair_rows),
        "example_micro": _example_metrics(joined),
        "matched_reward5": _example_metrics(matched),
        "counterfactual_reward1": _example_metrics(counterfactuals),
        "group_macro": _group_summary(group_rows),
        "example_by_task_id": _by_example_field(joined, "task_id"),
        "example_by_task_family": _by_example_field(joined, "task_family"),
        "by_task_id": _by_group_field(group_rows, "task_id"),
        "by_task_family": _by_group_field(group_rows, "task_family"),
        "prediction_counts": dict(
            sorted(Counter(str(row["prediction"]) for row in joined).items())
        ),
        "confusion_matrix": confusion,
        "inference_unit": "source_group",
        "labels_joined_only_during_scoring": True,
    }
    if group_rows and bootstrap_samples:
        statistics: dict[str, Callable[[list[dict[str, Any]]], float]] = {
            "pairwise_accuracy": lambda values: mean(
                float(row["pairwise_accuracy"]) for row in values
            ),
            "mean_progress_margin": lambda values: mean(
                float(row["mean_progress_margin"]) for row in values
            ),
            "strict_top1_accuracy": lambda values: mean(
                float(row["strict_top1_correct"]) for row in values
            ),
            "group_macro_exact_accuracy": lambda values: mean(
                float(row["group_exact_accuracy"]) for row in values
            ),
            "counterfactual_overprediction_rate": lambda values: mean(
                float(row["counterfactual_overprediction_rate"]) for row in values
            ),
        }
        result["cluster_stratified_bootstrap"] = {
            name: _stratified_group_bootstrap(
                group_rows, statistic, samples=bootstrap_samples, seed=seed + index
            )
            for index, (name, statistic) in enumerate(statistics.items())
        }

    scoring_dir = run_dir / "scoring"
    write_jsonl(scoring_dir / "joined_scores.jsonl", joined)
    write_jsonl(scoring_dir / "pair_scores.jsonl", pair_rows)
    write_jsonl(scoring_dir / "group_scores.jsonl", group_rows)
    write_json(scoring_dir / "metrics.json", result)
    write_json(scoring_dir / "completion.json", completion)
    invalid = [row for row in records.values() if row.get("status") != "ok"]
    write_json(scoring_dir / "invalid.json", invalid)
    group = result["group_macro"]
    bootstrap = result.get("cluster_stratified_bootstrap", {})
    (scoring_dir / "metrics.md").write_text(
        "# 同视频 counterfactual baseline 指标\n\n"
        f"- 正式评分是否完整：`{completion['formal_scoring_ready']}`\n"
        f"- 有效样本 / source groups / 配对数：{len(joined)} / {len(group_rows)} / {len(pair_rows)}\n"
        f"- Group-macro 配对准确率：{group.get('pairwise_accuracy')}\n"
        f"- 严格 group top-1 准确率：{group.get('strict_top1_correct')}\n"
        f"- matched−counterfactual 平均 progress margin：{group.get('mean_progress_margin')}\n"
        f"- 配对准确率 95% CI：{bootstrap.get('pairwise_accuracy', {}).get('ci95')}\n"
        f"- Counterfactual 高估率：{group.get('counterfactual_overprediction_rate')}\n"
        f"- Matched reward=5 Exact accuracy：{result['matched_reward5'].get('exact_accuracy')}\n"
        f"- Counterfactual reward=1 Exact accuracy：{result['counterfactual_reward1'].get('exact_accuracy')}\n",
        encoding="utf-8",
    )
    return result
