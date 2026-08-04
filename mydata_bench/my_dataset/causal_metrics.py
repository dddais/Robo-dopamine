"""Label-joined paired metrics for custom-dataset causal steering."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from ..io import read_jsonl, write_json, write_jsonl
from ..protocol import progress
from .data import load_labels


REQUIRED_CONDITIONS = {
    "baseline",
    "candidate_target",
    "candidate_wrong",
    "low_rank_target",
    "layer_matched_random_target",
}


def _group_ranking(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    pairwise = []
    strict = []
    pair_count = 0
    for values in groups.values():
        matched = [row for row in values if int(row["protocol_reward"]) == 5]
        failures = [row for row in values if int(row["protocol_reward"]) == 1]
        if len(matched) != 1 or not failures:
            continue
        matched_value = float(matched[0][field])
        comparisons = [matched_value > float(row[field]) for row in failures]
        pair_count += len(comparisons)
        pairwise.append(mean(comparisons))
        strict.append(all(comparisons))
    return {
        "group_count": len(pairwise),
        "pair_count": pair_count,
        "group_macro_pairwise_accuracy": mean(pairwise) if pairwise else None,
        "strict_top1_accuracy": mean(strict) if strict else None,
    }


def _prediction(row: dict[str, Any]) -> int:
    if "native_prediction" in row:
        return int(row["native_prediction"])
    value = float(row["signed_score"])
    return max(1, min(5, round(progress(value) * 4) + 1))


def _effect(row: dict[str, Any]) -> float:
    if "progress" in row:
        return float(row["progress"])
    return progress(float(row["signed_score"]))


def _bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = []
        for task_id in sorted(by_task):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in by_task[task_id]:
                groups[str(row["group_id"])].append(row)
            keys = sorted(groups)
            for _index in keys:
                draw.extend(groups[rng.choice(keys)])
        values.append(statistic(draw))
    values.sort()
    return {
        "estimate": statistic(rows),
        "ci95": [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]],
        "samples": samples,
        "cluster": "group_id",
        "stratum": "task_id",
    }


def score_steering(
    records_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260803,
) -> dict[str, Any]:
    records = [row for row in read_jsonl(records_path) if row.get("status") == "ok"]
    labels = {str(row["example_id"]): row for row in load_labels(labels_path)}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        grouped[str(row["example_id"])][str(row["condition"])] = row
    contrasts = []
    incomplete = {}
    for example_id, conditions in sorted(grouped.items()):
        missing = REQUIRED_CONDITIONS - conditions.keys()
        if missing:
            incomplete[example_id] = sorted(missing)
            continue
        label = labels.get(example_id)
        if label is None:
            raise ValueError(f"No label for steering record {example_id}")
        baseline = conditions["baseline"]
        target = conditions["candidate_target"]
        baseline_prediction = _prediction(baseline)
        target_prediction = _prediction(target)
        reward = int(label["protocol_reward"])
        fail = reward == 1
        suc = reward == 5
        row = {
            "example_id": example_id,
            "group_id": baseline["group_id"],
            "task_id": baseline["task_id"],
            "task_family": baseline["task_family"],
            "protocol_reward": reward,
            "baseline_prediction": baseline_prediction,
            "candidate_prediction": target_prediction,
            "baseline_progress": _effect(baseline),
            "candidate_progress": _effect(target),
            "target_shift": _effect(target) - _effect(baseline),
            "spatial_specificity": _effect(target) - _effect(conditions["candidate_wrong"]),
            "low_rank_specificity": _effect(target) - _effect(conditions["low_rank_target"]),
            "random_head_specificity": _effect(target) - _effect(conditions["layer_matched_random_target"]),
            "fail_correction": bool(fail and baseline_prediction != 1 and target_prediction == 1),
            "suc_harm": bool(suc and baseline_prediction == 5 and target_prediction != 5),
            "baseline_exact": baseline_prediction == reward,
            "candidate_exact": target_prediction == reward,
        }
        contrasts.append(row)
    fail_rows = [row for row in contrasts if row["protocol_reward"] == 1]
    suc_rows = [row for row in contrasts if row["protocol_reward"] == 5]
    correction = mean(float(row["fail_correction"]) for row in fail_rows) if fail_rows else 0.0
    harm = mean(float(row["suc_harm"]) for row in suc_rows) if suc_rows else 0.0
    validation_grid = {}
    grid_conditions = sorted(
        {
            str(row["condition"])
            for row in records
            if str(row["condition"]).startswith("validation_candidate_target_")
        }
    )
    label_by_id = labels
    baseline_by_id = {
        example_id: values["baseline"]
        for example_id, values in grouped.items()
        if "baseline" in values
    }
    for condition in grid_conditions:
        grid_rows = [row for row in records if row["condition"] == condition]
        joined = []
        for row in grid_rows:
            example_id = str(row["example_id"])
            if example_id not in baseline_by_id or example_id not in label_by_id:
                continue
            reward = int(label_by_id[example_id]["protocol_reward"])
            before = _prediction(baseline_by_id[example_id])
            after = _prediction(row)
            joined.append(
                {
                    "reward": reward,
                    "group_id": row["group_id"],
                    "before": before,
                    "after": after,
                    "baseline_progress": _effect(baseline_by_id[example_id]),
                    "candidate_progress": _effect(row),
                    "baseline_exact": before == reward,
                    "candidate_exact": after == reward,
                    "fail_correction": reward == 1 and before != 1 and after == 1,
                    "suc_harm": reward == 5 and before == 5 and after != 5,
                }
            )
        fail_grid = [row for row in joined if row["reward"] == 1]
        suc_grid = [row for row in joined if row["reward"] == 5]
        correction_grid = (
            mean(float(row["fail_correction"]) for row in fail_grid) if fail_grid else 0.0
        )
        harm_grid = mean(float(row["suc_harm"]) for row in suc_grid) if suc_grid else 0.0
        validation_grid[condition] = {
            "n": len(joined),
            "fail_correction_rate": correction_grid,
            "suc_harm_rate": harm_grid,
            "balanced_net_correction": correction_grid - harm_grid,
            "exact_delta": (
                mean(
                    float(row["candidate_exact"]) - float(row["baseline_exact"])
                    for row in joined
                )
                if joined
                else None
            ),
            "baseline_ranking": _group_ranking(
                [
                    {
                        "group_id": row["group_id"],
                        "protocol_reward": row["reward"],
                        "progress": row["baseline_progress"],
                    }
                    for row in joined
                ],
                "progress",
            ),
            "candidate_ranking": _group_ranking(
                [
                    {
                        "group_id": row["group_id"],
                        "protocol_reward": row["reward"],
                        "progress": row["candidate_progress"],
                    }
                    for row in joined
                ],
                "progress",
            ),
        }
    statistics = {
        "target_shift": lambda values: mean(float(row["target_shift"]) for row in values),
        "spatial_specificity": lambda values: mean(float(row["spatial_specificity"]) for row in values),
        "low_rank_specificity": lambda values: mean(float(row["low_rank_specificity"]) for row in values),
        "random_head_specificity": lambda values: mean(float(row["random_head_specificity"]) for row in values),
        "exact_delta": lambda values: mean(float(row["candidate_exact"]) - float(row["baseline_exact"]) for row in values),
    }
    result = {
        "metric_contract": "my_dataset.causal_paired.v1",
        "complete_examples": len(contrasts),
        "incomplete_examples": incomplete,
        "fail_count": len(fail_rows),
        "suc_count": len(suc_rows),
        "fail_correction_rate": correction,
        "suc_harm_rate": harm,
        "balanced_net_correction": correction - harm,
        "baseline_ranking": _group_ranking(contrasts, "baseline_progress"),
        "candidate_ranking": _group_ranking(contrasts, "candidate_progress"),
        "condition_counts": dict(sorted(Counter(row["condition"] for row in records).items())),
        "validation_grid": validation_grid,
        "validation_selection_automatic": False,
        "labels_joined_only_during_scoring": True,
        "cluster_stratified_bootstrap": {
            name: _bootstrap(
                contrasts,
                statistic,
                samples=bootstrap_samples,
                seed=seed + index,
            )
            for index, (name, statistic) in enumerate(statistics.items())
        } if contrasts and bootstrap_samples else {},
    }
    destination = Path(output_dir)
    write_jsonl(destination / "contrasts.jsonl", contrasts)
    write_json(destination / "metrics.json", result)
    return result
