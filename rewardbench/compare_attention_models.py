#!/usr/bin/env python3
"""Reproducible cross-model ranking and steering comparison."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rewardbench.attention_eval.stats import (
    exact_mcnemar_pvalue,
    paired_cluster_bootstrap,
    paired_sign_flip_pvalue,
)
from rewardbench.io import object_fingerprint, read_jsonl, sha256_file, write_json
from rewardbench.protocol import progress, progress_to_reward


CONDITIONS = (
    "baseline",
    "candidate_target",
    "candidate_wrong",
    "low_rank_target",
)
EFFECT_FIELDS = (
    "target_shift",
    "spatial_specificity",
    "head_specificity",
    "prediction_delta",
    "absolute_error_change",
)


def _ranking(path: Path) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("ranking")
    if not isinstance(rows, list):
        raise ValueError(f"No complete consensus ranking in {path}")
    pairs = [(int(row["layer"]), int(row["head"])) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"Duplicate heads in {path}")
    return data, pairs


def _spearman(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> float:
    first_rank = {value: index for index, value in enumerate(first)}
    second_rank = {value: index for index, value in enumerate(second)}
    common = sorted(set(first_rank) & set(second_rank))
    count = len(common)
    if count < 2:
        raise ValueError("Spearman comparison requires at least two common heads")
    squared = sum(
        (first_rank[value] - second_rank[value]) ** 2 for value in common
    )
    return 1 - 6 * squared / (count * (count * count - 1))


def compare_rankings(
    models: dict[str, dict[str, Any]], *, top_k: int
) -> dict[str, Any]:
    loaded = {}
    for name, value in models.items():
        path = Path(value["ranking_path"]).resolve()
        data, ranking = _ranking(path)
        loaded[name] = {"path": path, "data": data, "ranking": ranking}
    dimensions = {
        (
            int(value["data"]["num_layers"]),
            int(value["data"]["num_heads"]),
            int(value["data"].get("skip_early_layers", 0)),
            len(value["ranking"]),
        )
        for value in loaded.values()
    }
    if len(dimensions) != 1:
        raise ValueError(f"Rankings do not share an eligible head universe: {dimensions}")
    head_universes = {
        frozenset(value["ranking"]) for value in loaded.values()
    }
    if len(head_universes) != 1:
        raise ValueError("Rankings have different eligible head identities")
    pairwise = {}
    for first, second in itertools.combinations(loaded, 2):
        first_top = set(loaded[first]["ranking"][:top_k])
        second_top = set(loaded[second]["ranking"][:top_k])
        intersection = sorted(first_top & second_top)
        union = first_top | second_top
        pairwise[f"{first}__vs__{second}"] = {
            "top_k": top_k,
            "intersection_count": len(intersection),
            "jaccard": len(intersection) / len(union),
            "shared_heads": [list(value) for value in intersection],
            "full_ranking_spearman": _spearman(
                loaded[first]["ranking"], loaded[second]["ranking"]
            ),
        }
    per_model = {}
    for name, value in loaded.items():
        per_model[name] = {
            "ranking_path": str(value["path"]),
            "ranking_sha256": sha256_file(value["path"]),
            "top_heads": [
                {"rank": index + 1, "layer": layer, "head": head}
                for index, (layer, head) in enumerate(value["ranking"][:top_k])
            ],
            "top_layer_counts": {
                str(layer): sum(
                    selected_layer == layer
                    for selected_layer, _head in value["ranking"][:top_k]
                )
                for layer in sorted(
                    {layer for layer, _head in value["ranking"][:top_k]}
                )
            },
        }
        root = value["path"].parent
        raw_path = root / "consensus_ranking_raw_mass.json"
        for alternative in ("excess_mass", "visual_enrichment"):
            alternative_path = root / f"consensus_ranking_{alternative}.json"
            if raw_path.is_file() and alternative_path.is_file():
                _raw_data, raw_ranking = _ranking(raw_path)
                _alt_data, alt_ranking = _ranking(alternative_path)
                raw_top, alt_top = set(raw_ranking[:top_k]), set(
                    alt_ranking[:top_k]
                )
                per_model[name].setdefault("metric_robustness", {})[
                    f"raw_mass__vs__{alternative}"
                ] = {
                    "top_k_jaccard": len(raw_top & alt_top)
                    / len(raw_top | alt_top),
                    "full_ranking_spearman": _spearman(
                        raw_ranking, alt_ranking
                    ),
                }
    return {
        "common_dimensions": list(next(iter(dimensions))),
        "per_model": per_model,
        "pairwise": pairwise,
    }


def _latest_conditions(run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    metadata = {}
    for name in ("eligible.jsonl", "cohort_inputs.jsonl"):
        path = run_dir / name
        if path.is_file():
            metadata.update(
                {
                    str(row["example_id"]): row
                    for row in read_jsonl(path)
                    if isinstance(row.get("example_id"), str)
                }
            )
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(run_dir / "steering.jsonl"):
        example_id = str(row.get("example_id"))
        condition = str(row.get("condition"))
        latest[(example_id, condition)] = row
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for (example_id, condition), row in latest.items():
        if row.get("status") != "ok" or condition not in CONDITIONS:
            continue
        value = dict(row)
        source = metadata.get(example_id, {})
        value["video_sha256"] = str(
            value.get("video_sha256") or source.get("video_sha256") or example_id
        )
        value["subset"] = value.get("subset") or source.get("subset")
        grouped.setdefault(example_id, {})[condition] = value
    return grouped


def _values(row: dict[str, Any]) -> tuple[float, int]:
    if row.get("native_prediction") is not None:
        prediction = int(row["native_prediction"])
        return (prediction - 1) / 4, prediction
    if row.get("progress") is not None:
        value = float(row["progress"])
        return value, progress_to_reward(value)
    if row.get("signed_score") is not None:
        value = progress(float(row["signed_score"]))
        return value, progress_to_reward(value)
    raise ValueError("Steering row has no recognized model output")


def steering_effect_rows(run_dir: Path, expected_reward: int) -> list[dict[str, Any]]:
    grouped = _latest_conditions(run_dir)
    rows = []
    for example_id, conditions in grouped.items():
        if not set(CONDITIONS) <= conditions.keys():
            continue
        baseline_value, baseline_prediction = _values(conditions["baseline"])
        target_value, target_prediction = _values(conditions["candidate_target"])
        wrong_value, _wrong_prediction = _values(conditions["candidate_wrong"])
        low_value, _low_prediction = _values(conditions["low_rank_target"])
        rows.append(
            {
                "example_id": example_id,
                "video_sha256": conditions["baseline"]["video_sha256"],
                "subset": conditions["baseline"].get("subset"),
                "target_shift": target_value - baseline_value,
                "spatial_specificity": target_value - wrong_value,
                "head_specificity": target_value - low_value,
                "prediction_delta": target_prediction - baseline_prediction,
                "absolute_error_change": abs(target_prediction - expected_reward)
                - abs(baseline_prediction - expected_reward),
                "baseline_correct": baseline_prediction == expected_reward,
                "candidate_correct": target_prediction == expected_reward,
                "corrected": baseline_prediction != expected_reward
                and target_prediction == expected_reward,
                "harmed": baseline_prediction == expected_reward
                and target_prediction != expected_reward,
            }
        )
    return rows


def _effect_summary(rows: list[dict[str, Any]], samples: int) -> dict[str, Any]:
    return {
        "n_records": len(rows),
        "n_video_clusters": len({row["video_sha256"] for row in rows}),
        "corrected_count": sum(row["corrected"] for row in rows),
        "harmed_count": sum(row["harmed"] for row in rows),
        "estimands": {
            field: paired_cluster_bootstrap(rows, field, samples=samples)
            for field in EFFECT_FIELDS
        },
        "continuous_two_sided_cluster_sign_flip_pvalues": {
            field: paired_sign_flip_pvalue(rows, field, samples=samples)
            for field in (
                "target_shift",
                "spatial_specificity",
                "head_specificity",
            )
        },
        "exact_mcnemar_pvalue_record_level": exact_mcnemar_pvalue(
            rows, "baseline_correct", "candidate_correct"
        ),
    }


def compare_steering(
    models: dict[str, dict[str, Any]], *, samples: int
) -> dict[str, Any]:
    by_reward = {}
    for reward in (1, 5):
        rows_by_model = {}
        per_model = {}
        for name, value in models.items():
            run_dir = Path(value[f"reward{reward}_run"]).resolve()
            rows = steering_effect_rows(run_dir, reward)
            rows_by_model[name] = {row["example_id"]: row for row in rows}
            per_model[name] = {
                "run_dir": str(run_dir),
                "steering_sha256": sha256_file(run_dir / "steering.jsonl"),
                **_effect_summary(rows, samples),
            }
        pairwise = {}
        for first, second in itertools.combinations(rows_by_model, 2):
            shared = sorted(
                set(rows_by_model[first]) & set(rows_by_model[second])
            )
            differences = []
            for example_id in shared:
                first_row = rows_by_model[first][example_id]
                second_row = rows_by_model[second][example_id]
                if first_row["video_sha256"] != second_row["video_sha256"]:
                    raise ValueError(
                        f"Cross-model video mismatch for {example_id}: "
                        f"{first} vs {second}"
                    )
                differences.append(
                    {
                        "example_id": example_id,
                        "video_sha256": first_row["video_sha256"],
                        "subset": first_row.get("subset")
                        or second_row.get("subset"),
                        **{
                            f"{field}_difference_first_minus_second": (
                                float(first_row[field])
                                - float(second_row[field])
                            )
                            for field in EFFECT_FIELDS
                        },
                    }
                )
            pairwise[f"{first}__vs__{second}"] = {
                "n_shared_records": len(shared),
                "first_minus_second_cluster_estimands": {
                    field: paired_cluster_bootstrap(
                        differences,
                        f"{field}_difference_first_minus_second",
                        samples=samples,
                    )
                    for field in EFFECT_FIELDS
                },
            }
        by_reward[str(reward)] = {
            "per_model": per_model,
            "pairwise": pairwise,
        }
    return by_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    models = spec.get("models")
    if not isinstance(models, dict) or len(models) < 2:
        raise ValueError("spec.models must define at least two models")
    result = {
        "spec_path": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "top_k": args.top_k,
        "bootstrap_samples": args.bootstrap_samples,
        "ranking": compare_rankings(models, top_k=args.top_k),
        "steering": compare_steering(
            models, samples=args.bootstrap_samples
        ),
        "interpretation_boundary": (
            "Cross-model effect differences are paired descriptive contrasts. "
            "Adapter and native protocols have different readouts; native Qwen "
            "and native RoboReward provide the strict checkpoint comparison."
        ),
    }
    result["fingerprint"] = object_fingerprint(result)
    write_json(args.output, result)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
