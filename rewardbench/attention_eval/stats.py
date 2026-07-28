from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, median


def paired_cluster_bootstrap(
    rows: list[dict],
    field: str,
    *,
    samples: int = 10_000,
    seed: int = 20260724,
) -> dict:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            clusters[str(row["video_sha256"])].append(float(value))
    keys = sorted(clusters)
    if not keys:
        return {"n": 0, "mean": None, "median": None, "ci95": [None, None]}
    cluster_values = {key: mean(values) for key, values in clusters.items()}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(cluster_values[rng.choice(keys)] for _ in keys))
    estimates.sort()
    return {
        "n_records": sum(len(value) for value in clusters.values()),
        "n_clusters": len(keys),
        "mean": mean(cluster_values.values()),
        "median": median(cluster_values.values()),
        "ci95": [
            estimates[int(0.025 * (samples - 1))],
            estimates[int(0.975 * (samples - 1))],
        ],
        "samples": samples,
        "seed": seed,
    }


def paired_sign_flip_pvalue(
    rows: list[dict], field: str, *, samples: int = 10_000, seed: int = 20260724
) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    observed = abs(mean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        estimate = abs(mean(value * rng.choice((-1, 1)) for value in values))
        extreme += estimate >= observed
    return (extreme + 1) / (samples + 1)


def holm(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted((value, key) for key, value in pvalues.items() if value is not None)
    adjusted: dict[str, float | None] = {key: None for key in pvalues}
    running = 0.0
    total = len(valid)
    for index, (value, key) in enumerate(valid):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[key] = running
    return adjusted

