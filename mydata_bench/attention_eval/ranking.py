from __future__ import annotations

from pathlib import Path
from statistics import mean, median

from ..io import object_fingerprint, read_jsonl, sha256_file, write_json


def consensus_ranking(
    paths: list[str | Path],
    *,
    expected_layers: int = 36,
    expected_heads: int = 32,
    skip_early_layers: int = 0,
) -> dict:
    if not paths:
        raise ValueError("Ranking aggregation requires at least one complete ranking")
    if not 0 <= skip_early_layers < expected_layers:
        raise ValueError("skip_early_layers must be in [0, expected_layers)")
    rank_maps = []
    fingerprints = {}
    total = expected_layers * expected_heads
    eligible_total = (expected_layers - skip_early_layers) * expected_heads
    for raw_path in paths:
        path = Path(raw_path).resolve()
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("num_layers", -1)) != expected_layers:
            raise ValueError(f"Layer count mismatch in {path}")
        if int(data.get("num_heads", -1)) != expected_heads:
            raise ValueError(f"Query-head count mismatch in {path}")
        rows = data.get("rankings", {}).get("mean")
        if not isinstance(rows, list) or len(rows) != total:
            raise ValueError(f"{path} is not a complete mean ranking")
        pairs = [(int(row["layer"]), int(row["head"])) for row in rows]
        if len(set(pairs)) != total:
            raise ValueError(f"{path} contains duplicate/missing heads")
        # ``rankings.mean`` is intentionally complete for diagnostics, while
        # each source's published ``top_heads`` has already applied its
        # skip-early-layer policy.  Filter before Borda normalization so the
        # excluded layers neither enter consensus nor distort remaining ranks.
        eligible_pairs = [pair for pair in pairs if pair[0] >= skip_early_layers]
        if len(eligible_pairs) != eligible_total:
            raise ValueError(f"{path} has an unexpected eligible-head count")
        denominator = max(1, eligible_total - 1)
        rank_maps.append(
            {pair: index / denominator for index, pair in enumerate(eligible_pairs)}
        )
        fingerprints[str(path)] = sha256_file(path)
    rows = []
    for layer in range(skip_early_layers, expected_layers):
        for head in range(expected_heads):
            normalized = [mapping[(layer, head)] for mapping in rank_maps]
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "normalized_borda_rank": mean(normalized),
                    "source_normalized_ranks": normalized,
                }
            )
    rows.sort(key=lambda row: (row["normalized_borda_rank"], row["layer"], row["head"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["score"] = 1 - row["normalized_borda_rank"]
    single_source = len(paths) == 1
    ranking_source = "independent_single_source_ranking" if single_source else "frozen_cross_domain_consensus"
    method = "single_source_normalized_rank" if single_source else "mean_normalized_borda_rank"
    result = {
        "ranking_source": ranking_source,
        "num_layers": expected_layers,
        "num_heads": expected_heads,
        "skip_early_layers": skip_early_layers,
        "eligible_head_count": eligible_total,
        "method": method,
        "ranking_fingerprints": fingerprints,
        "ranking": rows,
    }
    result["fingerprint"] = object_fingerprint(result)
    return result


def aggregate_in_domain(
    per_sample: list[dict], *, num_layers: int = 36, num_heads: int = 32, skip_layers: int = 2
) -> dict:
    import numpy as np

    # Mass files are append-only so retries can leave multiple records for one
    # example. Match the resume logic and treat the latest record as the
    # authoritative state before deciding which examples are valid.
    latest_by_example: dict[str, dict] = {}
    for row in per_sample:
        latest_by_example[str(row["example_id"])] = row
    valid = [row for row in latest_by_example.values() if row.get("status") == "ok"]
    if not valid:
        raise ValueError("No valid discovery attention records")
    raw = np.asarray([row["raw_mass"] for row in valid], dtype=np.float64)
    excess = np.asarray([row["excess_mass"] for row in valid], dtype=np.float64)
    if raw.shape[1:] != (num_layers, num_heads):
        raise ValueError(f"Unexpected attention shape {raw.shape}")
    top_count = max(1, int(round(0.05 * num_layers * num_heads)))
    hits = np.zeros((num_layers, num_heads), dtype=np.float64)
    for sample in excess:
        eligible = sample[skip_layers:].reshape(-1)
        chosen = np.argpartition(eligible, -top_count)[-top_count:]
        flat = hits[skip_layers:].reshape(-1)
        flat[chosen] += 1
    rows = []
    for layer in range(skip_layers, num_layers):
        for head in range(num_heads):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "mean_excess_mass": float(excess[:, layer, head].mean()),
                    "mean_raw_mass": float(raw[:, layer, head].mean()),
                    "median_raw_mass": float(np.median(raw[:, layer, head])),
                    "top5_selection_frequency": float(hits[layer, head] / len(valid)),
                }
            )
    rows.sort(
        key=lambda row: (
            -row["mean_excess_mass"],
            -row["mean_raw_mass"],
            row["layer"],
            row["head"],
        )
    )
    result = {
        "ranking_source": "in_domain_discovery_only",
        "num_layers": num_layers,
        "num_heads": num_heads,
        "skip_early_layers": skip_layers,
        "n_discovery_samples": len(valid),
        "ranking": rows,
        "per_sample_example_ids": [row["example_id"] for row in valid],
    }
    result["fingerprint"] = object_fingerprint(result)
    return result
