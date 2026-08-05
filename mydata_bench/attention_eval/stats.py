from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, median


def _cluster_values(
    rows: list[dict], field: str
) -> tuple[dict[str, float], dict[str, str], int]:
    values_by_cluster: dict[str, list[float]] = defaultdict(list)
    subset_by_cluster: dict[str, str] = {}
    n_records = 0
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        cluster = str(row["video_sha256"])
        subset_value = row.get("subset")
        subset = str(subset_value) if subset_value is not None else "__all__"
        previous = subset_by_cluster.setdefault(cluster, subset)
        if previous != subset:
            raise ValueError(
                f"video cluster {cluster!r} appears in multiple subsets: "
                f"{previous!r} and {subset!r}"
            )
        values_by_cluster[cluster].append(float(value))
        n_records += 1
    return (
        {cluster: mean(values) for cluster, values in values_by_cluster.items()},
        subset_by_cluster,
        n_records,
    )


def paired_cluster_bootstrap(
    rows: list[dict],
    field: str,
    *,
    samples: int = 10_000,
    seed: int = 20260724,
) -> dict:
    cluster_values, subset_by_cluster, n_records = _cluster_values(rows, field)
    keys = sorted(cluster_values)
    if not keys:
        return {
            "n_records": 0,
            "n_clusters": 0,
            "n_strata": 0,
            "strata_cluster_counts": {},
            "mean": None,
            "median": None,
            "ci95": [None, None],
        }
    strata: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        strata[subset_by_cluster[key]].append(key)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resampled = [
            cluster_values[rng.choice(stratum_keys)]
            for stratum_keys in strata.values()
            for _ in stratum_keys
        ]
        estimates.append(mean(resampled))
    estimates.sort()
    return {
        "n_records": n_records,
        "n_clusters": len(keys),
        "n_strata": len(strata),
        "strata_cluster_counts": {
            subset: len(strata[subset]) for subset in sorted(strata)
        },
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
    cluster_values, _subset_by_cluster, _n_records = _cluster_values(rows, field)
    values = [cluster_values[key] for key in sorted(cluster_values)]
    if not values:
        return None
    observed = abs(mean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        estimate = abs(mean(value * rng.choice((-1, 1)) for value in values))
        extreme += estimate >= observed
    return (extreme + 1) / (samples + 1)


def exact_mcnemar_pvalue(
    rows: list[dict],
    baseline_field: str,
    candidate_field: str,
) -> float | None:
    """Two-sided exact McNemar test over paired Boolean correctness values."""
    discordant_baseline_only = 0
    discordant_candidate_only = 0
    for row in rows:
        baseline = row.get(baseline_field)
        candidate = row.get(candidate_field)
        if baseline is None or candidate is None:
            continue
        baseline = bool(baseline)
        candidate = bool(candidate)
        if baseline and not candidate:
            discordant_baseline_only += 1
        elif candidate and not baseline:
            discordant_candidate_only += 1
    total = discordant_baseline_only + discordant_candidate_only
    if total == 0:
        return 1.0
    lower = min(discordant_baseline_only, discordant_candidate_only)
    tail = sum(math.comb(total, value) for value in range(lower + 1)) / (2**total)
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted((value, key) for key, value in pvalues.items() if value is not None)
    adjusted: dict[str, float | None] = {key: None for key in pvalues}
    running = 0.0
    total = len(valid)
    for index, (value, key) in enumerate(valid):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[key] = running
    return adjusted
