from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from statistics import mean
from typing import Callable

from .data import classify_source
from .protocol import progress_to_reward


def _valid(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("progress") is not None
        and row.get("reward") is not None
    ]


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    absolute = [abs(int(row["prediction"]) - int(row["reward"])) for row in rows]
    continuous = [
        abs(1 + 4 * float(row["progress"]) - int(row["reward"])) for row in rows
    ]
    signed = [int(row["prediction"]) - int(row["reward"]) for row in rows]
    return {
        "n": len(rows),
        "mae": mean(absolute),
        "continuous_ordinal_mae": mean(continuous),
        "exact_accuracy": mean(value == 0 for value in absolute),
        "within_one_accuracy": mean(value <= 1 for value in absolute),
        "mean_signed_error": mean(signed),
        "overprediction_rate": mean(value > 0 for value in signed),
        "underprediction_rate": mean(value < 0 for value in signed),
    }


def compute_metrics(rows: list[dict]) -> dict:
    valid = _valid(rows)
    for row in valid:
        row["prediction"] = progress_to_reward(float(row["progress"]))
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for row in valid:
        by_subset[str(row["subset"])].append(row)
    subset_metrics = {key: _summary(value) for key, value in sorted(by_subset.items())}
    macro_mae = (
        mean(value["mae"] for value in subset_metrics.values()) if subset_metrics else None
    )
    confusion = {str(y): {str(p): 0 for p in range(1, 6)} for y in range(1, 6)}
    for row in valid:
        confusion[str(row["reward"])][str(row["prediction"])] += 1
    prediction_counts = Counter(int(row["prediction"]) for row in valid)
    grouped = {}
    for field, key_fn in (
        ("reward", lambda row: str(row["reward"])),
        ("subset", lambda row: str(row["subset"])),
        ("source", lambda row: classify_source(str(row["subset"]))),
    ):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in valid:
            groups[key_fn(row)].append(row)
        grouped[field] = {key: _summary(value) for key, value in sorted(groups.items())}
    reward1 = [row for row in valid if int(row["reward"]) == 1]
    return {
        "adapter_metric": True,
        "official_native_discrete_output": False,
        "num_records": len(rows),
        "num_valid": len(valid),
        "num_invalid": len(rows) - len(valid),
        "micro": _summary(valid),
        "macro_subset_mae": macro_mae,
        "by": grouped,
        "confusion_matrix": confusion,
        "prediction_counts": {
            str(index): prediction_counts.get(index, 0) for index in range(1, 6)
        },
        "reward1": {
            "n": len(reward1),
            "predicted_one_rate": mean(row["prediction"] == 1 for row in reward1)
            if reward1
            else None,
            "overestimated_rate": mean(row["prediction"] > 1 for row in reward1)
            if reward1
            else None,
            "label_migration": dict(
                sorted(Counter(str(row["prediction"]) for row in reward1).items())
            ),
        },
    }


def clustered_stratified_bootstrap(
    rows: list[dict],
    statistic: Callable[[list[dict]], float],
    *,
    samples: int = 10_000,
    seed: int = 20260724,
) -> dict:
    valid = _valid(rows)
    strata: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in valid:
        strata[str(row["subset"])][str(row["video_sha256"])].append(row)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw: list[dict] = []
        for clusters in strata.values():
            keys = sorted(clusters)
            for _ in keys:
                draw.extend(clusters[rng.choice(keys)])
        estimates.append(float(statistic(draw)))
    estimates.sort()
    if not estimates:
        return {"samples": 0, "estimate": None, "ci95": [None, None]}
    lo = estimates[max(0, math.floor(0.025 * (len(estimates) - 1)))]
    hi = estimates[min(len(estimates) - 1, math.ceil(0.975 * (len(estimates) - 1)))]
    return {
        "samples": samples,
        "seed": seed,
        "estimate": float(statistic(valid)),
        "ci95": [lo, hi],
    }

