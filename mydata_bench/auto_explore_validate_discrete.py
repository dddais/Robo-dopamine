#!/usr/bin/env python3
"""Completeness and paired statistical validation for discrete steering runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .attention_eval.stats import (
    exact_mcnemar_pvalue,
    holm,
    paired_cluster_bootstrap,
    paired_sign_flip_pvalue,
)
from .io import object_fingerprint, read_jsonl, write_json


CONDITIONS = [
    "baseline",
    *[
        f"{kind}_k{k}"
        for k in (8, 32, 64)
        for kind in ("candidate_target", "candidate_wrong", "low_rank_target")
    ],
]


def label(example_id: str) -> int:
    if example_id.startswith("suc/"):
        return 5
    if example_id.startswith("fail/"):
        return 1
    raise ValueError(f"Unknown example class: {example_id}")


def metrics(rows: list[dict], condition: str) -> dict:
    values = [(int(row[condition]["native_prediction"]), label(row["example_id"])) for row in rows]
    return {
        "n": len(values),
        "mae": mean(abs(prediction - target) for prediction, target in values),
        "accuracy": mean(prediction == target for prediction, target in values),
        "prediction_distribution": dict(sorted(Counter(prediction for prediction, _ in values).items())),
    }


def paired_rows(rows: list[dict], condition: str) -> list[dict]:
    result = []
    for row in rows:
        target = label(row["example_id"])
        baseline = int(row["baseline"]["native_prediction"])
        candidate = int(row[condition]["native_prediction"])
        result.append(
            {
                "example_id": row["example_id"],
                "video_sha256": row["video_sha256"],
                "subset": row["subset"],
                "mae_delta": abs(candidate - target) - abs(baseline - target),
                "accuracy_delta": int(candidate == target) - int(baseline == target),
                "baseline_correct": baseline == target,
                "candidate_correct": candidate == target,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--heldout-ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected_ids = json.loads(Path(args.heldout_ids).resolve().read_text(encoding="utf-8"))
    expected = set(expected_ids)
    latest = {}
    paths = sorted(Path(args.results_dir).resolve().glob("steering.shard-*.jsonl"))
    if not paths:
        raise FileNotFoundError("No steering shard files found")
    for path in paths:
        for row in read_jsonl(path):
            example_id = row.get("example_id")
            condition = row.get("condition")
            if isinstance(example_id, str) and isinstance(condition, str):
                latest[(example_id, condition)] = row
    unexpected = sorted({key[0] for key in latest} - expected)
    if unexpected:
        raise RuntimeError(f"Unexpected held-out IDs: {unexpected[:5]}")
    missing = [
        (example_id, condition)
        for example_id in expected_ids
        for condition in CONDITIONS
        if (example_id, condition) not in latest
        or latest[(example_id, condition)].get("status") != "ok"
    ]
    if missing:
        raise RuntimeError(f"Incomplete held-out matrix: {len(missing)} missing/invalid, first={missing[:5]}")

    rows = []
    for example_id in expected_ids:
        conditions = {condition: latest[(example_id, condition)] for condition in CONDITIONS}
        baseline = conditions["baseline"]
        rows.append(
            {
                "example_id": example_id,
                "video_sha256": baseline["video_sha256"],
                "subset": baseline["subset"],
                **conditions,
            }
        )
    suc_rows = [row for row in rows if row["example_id"].startswith("suc/")]
    fail_rows = [row for row in rows if row["example_id"].startswith("fail/")]

    adaptive_enabled = any(
        isinstance(row["baseline"].get("adaptive_gate"), dict) for row in rows
    )
    adaptive_gate_audit = {"enabled": adaptive_enabled}
    if adaptive_enabled:
        gate_rows = []
        for row in rows:
            gates = [row[condition].get("adaptive_gate") for condition in CONDITIONS]
            if not all(isinstance(gate, dict) for gate in gates):
                raise RuntimeError("Adaptive gate is missing from one or more conditions")
            signatures = {
                (
                    gate.get("selected_branch"),
                    float(gate.get("router_score")),
                    tuple(float(value) for value in gate.get("features", [])),
                )
                for gate in gates
            }
            if len(signatures) != 1:
                raise RuntimeError("Conditions do not share one frozen sample-level gate")
            gate = gates[0]
            if gate.get("inference_uses_labels") is not False:
                raise RuntimeError("Adaptive gate does not explicitly exclude labels")
            gate_rows.append(
                {
                    "example_id": row["example_id"],
                    "selected_branch": str(gate["selected_branch"]),
                    "router_score": float(gate["router_score"]),
                }
            )

        def gate_summary(prefix: str | None) -> dict:
            selected = [
                row
                for row in gate_rows
                if prefix is None or row["example_id"].startswith(prefix)
            ]
            scores = [row["router_score"] for row in selected]
            return {
                "n": len(selected),
                "branch_counts": dict(
                    sorted(Counter(row["selected_branch"] for row in selected).items())
                ),
                "router_score_mean": mean(scores),
                "router_score_min": min(scores),
                "router_score_max": max(scores),
            }

        adaptive_gate_audit.update(
            {
                "all_conditions_share_sample_gate": True,
                "inference_uses_labels": False,
                "overall": gate_summary(None),
                "suc": gate_summary("suc/"),
                "fail": gate_summary("fail/"),
            }
        )

    summary = {}
    raw_pvalues = {}
    for k in (8, 32, 64):
        condition = f"candidate_target_k{k}"
        paired = paired_rows(rows, condition)
        paired_suc = paired_rows(suc_rows, condition)
        paired_fail = paired_rows(fail_rows, condition)
        summary[str(k)] = {
            "baseline": {
                "overall": metrics(rows, "baseline"),
                "suc": metrics(suc_rows, "baseline"),
                "fail": metrics(fail_rows, "baseline"),
            },
            "candidate": {
                "overall": metrics(rows, condition),
                "suc": metrics(suc_rows, condition),
                "fail": metrics(fail_rows, condition),
            },
            "cluster_bootstrap": {
                "mae_delta": paired_cluster_bootstrap(paired, "mae_delta"),
                "suc_accuracy_delta": paired_cluster_bootstrap(paired_suc, "accuracy_delta"),
                "fail_accuracy_delta": paired_cluster_bootstrap(paired_fail, "accuracy_delta"),
            },
            "sign_flip_pvalue": {
                "mae_delta": paired_sign_flip_pvalue(paired, "mae_delta"),
                "suc_accuracy_delta": paired_sign_flip_pvalue(paired_suc, "accuracy_delta"),
                "fail_accuracy_delta": paired_sign_flip_pvalue(paired_fail, "accuracy_delta"),
            },
            "mcnemar_pvalue": {
                "overall": exact_mcnemar_pvalue(paired, "baseline_correct", "candidate_correct"),
                "suc": exact_mcnemar_pvalue(paired_suc, "baseline_correct", "candidate_correct"),
                "fail": exact_mcnemar_pvalue(paired_fail, "baseline_correct", "candidate_correct"),
            },
        }
        for metric_name, value in summary[str(k)]["sign_flip_pvalue"].items():
            raw_pvalues[f"k{k}_{metric_name}"] = value

    per_task = {}
    for subset in sorted({row["subset"] for row in rows}):
        task_rows = [row for row in rows if row["subset"] == subset]
        per_task[subset] = {
            condition: metrics(task_rows, condition)
            for condition in ["baseline", "candidate_target_k8", "candidate_target_k32", "candidate_target_k64"]
        }

    distributions = {}
    for condition in CONDITIONS:
        distributions[condition] = {
            "suc": metrics(suc_rows, condition)["prediction_distribution"],
            "fail": metrics(fail_rows, condition)["prediction_distribution"],
        }

    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_sha256"]].append(row)
    pairwise = {}
    for condition in ["baseline", "candidate_target_k8", "candidate_target_k32", "candidate_target_k64"]:
        differences = []
        for video_rows in by_video.values():
            successes = [row for row in video_rows if row["example_id"].startswith("suc/")]
            failures = [row for row in video_rows if row["example_id"].startswith("fail/")]
            for success in successes:
                for failure in failures:
                    differences.append(
                        int(success[condition]["native_prediction"])
                        - int(failure[condition]["native_prediction"])
                    )
        pairwise[condition] = {
            "n_pairs": len(differences),
            "difference_distribution": dict(sorted(Counter(differences).items())),
            "positive_rate": mean(value > 0 for value in differences) if differences else None,
        }

    controls = {
        condition: {
            "overall": metrics(rows, condition),
            "suc": metrics(suc_rows, condition),
            "fail": metrics(fail_rows, condition),
        }
        for condition in CONDITIONS
        if condition.startswith("candidate_wrong") or condition.startswith("low_rank")
    }
    artifact = {
        "schema_version": "auto-explore-discrete-validation-v1",
        "verification_status": "ANALYZED",
        "results_dir": str(Path(args.results_dir).resolve()),
        "heldout_ids_file": str(Path(args.heldout_ids).resolve()),
        "input_shards": [str(path) for path in paths],
        "completeness": {
            "expected_example_count": len(expected_ids),
            "condition_count": len(CONDITIONS),
            "expected_record_count": len(expected_ids) * len(CONDITIONS),
            "valid_record_count": len(expected_ids) * len(CONDITIONS),
            "video_cluster_count": len(by_video),
        },
        "summary": summary,
        "holm_adjusted_sign_flip_pvalues": holm(raw_pvalues),
        "per_task": per_task,
        "prediction_distributions": distributions,
        "pairwise": pairwise,
        "controls": controls,
        "adaptive_gate_audit": adaptive_gate_audit,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    write_json(Path(args.output).resolve(), artifact)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
