#!/usr/bin/env python3
"""Score label-free target-vs-control contrastive decoding branches."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from mydata_bench.data import load_episodes
from mydata_bench.attention_eval.stats import (
    exact_mcnemar_pvalue,
    paired_cluster_bootstrap,
)
from mydata_bench.io import read_jsonl, sha256_file, write_json


LABELS = tuple(range(1, 6))


def _logprobs(row: dict) -> dict[int, float]:
    values = row.get("discrete_label_logprobs", {}).get("values")
    if not isinstance(values, dict) or set(values) != {str(value) for value in LABELS}:
        raise ValueError(f"Missing discrete label log-probs for {row.get('example_id')}")
    result = {value: float(values[str(value)]) for value in LABELS}
    if max(result, key=result.get) != int(row["native_prediction"]):
        raise ValueError(
            "Recorded score-token argmax does not match generated prediction for "
            f"{row.get('example_id')} / {row.get('condition')}"
        )
    return result


def _summary(predictions: dict[str, int], labels: dict[str, int]) -> dict:
    errors = [abs(predictions[key] - labels[key]) for key in sorted(predictions)]
    success = [key for key in predictions if labels[key] == 5]
    failure = [key for key in predictions if labels[key] == 1]
    return {
        "n": len(predictions),
        "mae": mean(errors),
        "exact_accuracy": mean(error == 0 for error in errors),
        "suc_accuracy": mean(predictions[key] == 5 for key in success),
        "fail_accuracy": mean(predictions[key] == 1 for key in failure),
        "prediction_counts": dict(sorted(Counter(predictions.values()).items())),
    }


def _paired_summary(
    predictions: dict[str, int],
    baseline: dict[str, int],
    labels: dict[str, int],
    metadata: dict[str, dict],
    bootstrap_samples: int,
) -> dict:
    rows = []
    for example_id in sorted(predictions):
        base = baseline[example_id]
        candidate = predictions[example_id]
        label = labels[example_id]
        rows.append(
            {
                "example_id": example_id,
                "video_sha256": str(metadata[example_id]["video_sha256"]),
                "subset": metadata[example_id].get("subset"),
                "prediction_delta": candidate - base,
                "absolute_error_change": abs(candidate - label) - abs(base - label),
                "baseline_correct": base == label,
                "candidate_correct": candidate == label,
                "corrected": base != label and candidate == label,
                "harmed": base == label and candidate != label,
            }
        )
    return {
        "n": len(rows),
        "corrected_count": sum(row["corrected"] for row in rows),
        "harmed_count": sum(row["harmed"] for row in rows),
        "absolute_error_change_cluster_bootstrap": paired_cluster_bootstrap(
            rows, "absolute_error_change", samples=bootstrap_samples
        ),
        "prediction_delta_cluster_bootstrap": paired_cluster_bootstrap(
            rows, "prediction_delta", samples=bootstrap_samples
        ),
        "exact_mcnemar_pvalue_record_level": exact_mcnemar_pvalue(
            rows, "baseline_correct", "candidate_correct"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--control-kind",
        choices=("candidate_wrong", "candidate_target_only"),
        default="candidate_wrong",
    )
    parser.add_argument(
        "--alpha-values", default="0.25,0.5,1,2,4,8",
        help="Comma-separated non-negative contrastive strengths",
    )
    parser.add_argument("--exclude-ids", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--formulas",
        default="positive_plus_delta,baseline_plus_delta,control_plus_delta",
        help="Comma-separated subset of registered contrastive formulas",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")
    alphas = [float(value) for value in args.alpha_values.split(",")]
    if not alphas or any(value < 0 for value in alphas):
        raise ValueError("alpha-values must be non-empty and non-negative")
    allowed_formulas = {
        "positive_plus_delta",
        "baseline_plus_delta",
        "control_plus_delta",
    }
    formulas = [value for value in args.formulas.split(",") if value]
    if not formulas or any(value not in allowed_formulas for value in formulas):
        raise ValueError(f"formulas must be a subset of {sorted(allowed_formulas)}")

    latest = {}
    for row in read_jsonl(args.steering):
        if row.get("status") == "ok":
            latest[(str(row["example_id"]), str(row["condition"]))] = row
    ids = sorted({example_id for example_id, condition in latest if condition == "baseline"})
    excluded = (
        set(json.loads(args.exclude_ids.read_text(encoding="utf-8")))
        if args.exclude_ids
        else set()
    )
    ids = [example_id for example_id in ids if example_id not in excluded]
    if not ids:
        raise ValueError("No analysis IDs remain after exclusions")
    labels = {
        row.example_id: int(row.reward)
        for row in load_episodes(args.dataset_root, "all")
        if row.example_id in set(ids)
    }
    if set(labels) != set(ids):
        raise ValueError("Baseline IDs do not exactly resolve to dataset labels")
    baseline = {example_id: _logprobs(latest[(example_id, "baseline")]) for example_id in ids}
    metadata = {example_id: latest[(example_id, "baseline")] for example_id in ids}
    baseline_predictions = {
        example_id: max(values, key=values.get) for example_id, values in baseline.items()
    }
    top_values = sorted(
        int(condition.rsplit("k", 1)[1])
        for example_id, condition in latest
        if example_id == ids[0] and condition.startswith("candidate_target_k")
    )
    results = {}
    for top_k in top_values:
        target_name = f"candidate_target_k{top_k}"
        wrong_name = f"{args.control_kind}_k{top_k}"
        missing = [
            example_id for example_id in ids
            if (example_id, target_name) not in latest or (example_id, wrong_name) not in latest
        ]
        if missing:
            raise ValueError(f"Incomplete target/control branches for k={top_k}: {len(missing)}")
        target = {example_id: _logprobs(latest[(example_id, target_name)]) for example_id in ids}
        wrong = {example_id: _logprobs(latest[(example_id, wrong_name)]) for example_id in ids}
        by_formula = {}
        for formula in formulas:
            by_alpha = {}
            for alpha in alphas:
                predictions = {}
                for example_id in ids:
                    anchor = (
                        target[example_id]
                        if formula == "positive_plus_delta"
                        else wrong[example_id]
                        if formula == "control_plus_delta"
                        else baseline[example_id]
                    )
                    combined = {
                        label: anchor[label]
                        + alpha * (target[example_id][label] - wrong[example_id][label])
                        for label in LABELS
                    }
                    predictions[example_id] = max(combined, key=combined.get)
                summary = _summary(predictions, labels)
                summary["paired_vs_baseline"] = _paired_summary(
                    predictions,
                    baseline_predictions,
                    labels,
                    metadata,
                    args.bootstrap_samples,
                )
                by_alpha[str(alpha)] = summary
            by_formula[formula] = by_alpha
        results[str(top_k)] = by_formula
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "method": "target_vs_matched_control_discrete_contrastive_decoding",
            "labels_model_facing": False,
            "steering_path": str(args.steering.resolve()),
            "steering_sha256": sha256_file(args.steering),
            "alpha_values": alphas,
            "control_kind": args.control_kind,
            "formulas": formulas,
            "excluded_ids_file": str(args.exclude_ids.resolve()) if args.exclude_ids else None,
            "excluded_configured_count": len(excluded),
            "analysis_id_count": len(ids),
            "bootstrap_samples": args.bootstrap_samples,
            "baseline": _summary(baseline_predictions, labels),
            "by_top_k": results,
        },
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
