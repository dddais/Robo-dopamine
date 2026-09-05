#!/usr/bin/env python3
"""Write a complete report for a fixed discrete contrastive decoder.

The inference JSONL remains immutable.  This tool reconstructs the label-free
target/control decoder for one pre-specified ``(top_k, formula, alpha)`` and
writes new Markdown and JSON audit artifacts.  Existing outputs are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from mydata_bench.io import read_jsonl, sha256_file, write_json
from mydata_bench.score_contrastive_discrete import _logprobs, _paired_summary
from mydata_bench.write_exp_records import (
    condition_groups,
    markdown_table,
    native_distribution,
    native_summary,
    pair_stats,
    pair_table,
    pct,
    task_accuracy_native,
)


LABELS = tuple(range(1, 6))
FORMULAS = ("positive_plus_delta", "baseline_plus_delta", "control_plus_delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New Markdown output")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--formula", choices=FORMULAS)
    parser.add_argument(
        "--mixture-weights",
        help="Convex baseline,target,control weights, for example 0.7,0.05,0.25",
    )
    parser.add_argument(
        "--control-kind",
        choices=("candidate_wrong", "candidate_target_only"),
        default="candidate_wrong",
    )
    parser.add_argument("--exclude-ids", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mixture_weights = None
    if args.mixture_weights:
        values = [float(value) for value in args.mixture_weights.split(",")]
        if len(values) != 3 or any(value < 0 for value in values):
            raise ValueError("mixture-weights must contain three non-negative values")
        if abs(sum(values) - 1.0) > 1e-12:
            raise ValueError("mixture-weights must sum to one")
        mixture_weights = tuple(values)
    if mixture_weights is None:
        if args.formula is None or args.alpha is None:
            raise ValueError("formula and alpha are required without mixture-weights")
        if args.alpha < 0:
            raise ValueError("alpha must be non-negative")
    elif args.formula is not None or args.alpha is not None:
        raise ValueError("mixture-weights cannot be combined with formula or alpha")
    json_output = args.output.with_suffix(".json")
    for path in (args.output, json_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")

    latest: dict[tuple[str, str], dict] = {}
    all_rows = list(read_jsonl(args.steering))
    for row in all_rows:
        if row.get("status") == "ok":
            latest[(str(row["example_id"]), str(row["condition"]))] = row

    baseline_name = "baseline"
    target_name = f"candidate_target_k{args.top_k}"
    control_name = f"{args.control_kind}_k{args.top_k}"
    ids = sorted(example_id for example_id, condition in latest if condition == baseline_name)
    excluded = (
        set(json.loads(args.exclude_ids.read_text(encoding="utf-8")))
        if args.exclude_ids
        else set()
    )
    ids = [example_id for example_id in ids if example_id not in excluded]
    if not ids:
        raise ValueError("No analysis IDs remain after exclusions")
    missing = [
        example_id
        for example_id in ids
        if (example_id, target_name) not in latest or (example_id, control_name) not in latest
    ]
    if missing:
        raise ValueError(f"Incomplete target/control branches: {len(missing)}")

    metadata_rows = list(read_jsonl(args.metadata))
    metadata = {str(row["id"]): row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError(f"Duplicate metadata IDs in {args.metadata}")
    absent = sorted(set(ids) - set(metadata))
    if absent:
        raise ValueError(f"Analysis IDs absent from metadata: {len(absent)}")

    baseline_probs = {example_id: _logprobs(latest[(example_id, baseline_name)]) for example_id in ids}
    target_probs = {example_id: _logprobs(latest[(example_id, target_name)]) for example_id in ids}
    control_probs = {example_id: _logprobs(latest[(example_id, control_name)]) for example_id in ids}

    def argmax(values: dict[int, float]) -> int:
        return max(values, key=values.get)

    baseline_predictions = {example_id: argmax(baseline_probs[example_id]) for example_id in ids}
    predictions: dict[str, int] = {}
    for example_id in ids:
        if mixture_weights is not None:
            combined = {
                label: mixture_weights[0] * baseline_probs[example_id][label]
                + mixture_weights[1] * target_probs[example_id][label]
                + mixture_weights[2] * control_probs[example_id][label]
                for label in LABELS
            }
        else:
            anchor = (
                target_probs[example_id]
                if args.formula == "positive_plus_delta"
                else control_probs[example_id]
                if args.formula == "control_plus_delta"
                else baseline_probs[example_id]
            )
            combined = {
                label: anchor[label]
                + args.alpha * (target_probs[example_id][label] - control_probs[example_id][label])
                for label in LABELS
            }
        predictions[example_id] = argmax(combined)

    def record(example_id: str, prediction: int, condition: str) -> dict:
        meta = metadata[example_id]
        split = str(meta.get("split") or example_id.split("/", 1)[0])
        return {
            "condition": condition,
            "example_id": example_id,
            "prediction": prediction,
            "label": 5 if split == "suc" else 1,
            "split": split,
            "task_id": str(meta["task_id"]),
            "source_suc_id": str(meta.get("source_suc_id") or example_id),
        }

    candidate_label = (
        "convex_mixture_k"
        f"{args.top_k}_b{mixture_weights[0]:g}_t{mixture_weights[1]:g}_c{mixture_weights[2]:g}"
        if mixture_weights is not None
        else f"{args.formula}_k{args.top_k}_alpha{args.alpha:g}"
    )
    baseline_records = [record(key, baseline_predictions[key], baseline_name) for key in ids]
    candidate_records = [record(key, predictions[key], candidate_label) for key in ids]
    groups = condition_groups(baseline_records + candidate_records)
    labels = {row["example_id"]: int(row["label"]) for row in candidate_records}
    video_metadata = {
        example_id: {
            "video_sha256": str(latest[(example_id, baseline_name)].get("video_sha256") or example_id),
            "subset": metadata[example_id].get("task_id"),
        }
        for example_id in ids
    }
    paired = _paired_summary(
        predictions,
        baseline_predictions,
        labels,
        video_metadata,
        args.bootstrap_samples,
    )

    summaries = {condition: native_summary(rows) for condition, rows in groups.items()}
    pairwise = {condition: pair_stats(rows, native=True) for condition, rows in groups.items()}
    distributions = {}
    task_accuracy = {}
    for condition, rows in groups.items():
        distributions[condition] = {}
        for scope, scoped in (
            ("suc", [row for row in rows if row["split"] == "suc"]),
            ("fail", [row for row in rows if row["split"] == "fail"]),
        ):
            counts = Counter(int(row["prediction"]) for row in scoped)
            distributions[condition][scope] = {
                "n": len(scoped),
                "counts": {str(label): counts[label] for label in LABELS},
            }
        task_accuracy[condition] = {}
        for task_id in sorted({str(row["task_id"]) for row in rows}):
            task_rows = [row for row in rows if str(row["task_id"]) == task_id]
            by_split = {}
            for scope, scoped in (
                ("all", task_rows),
                ("suc", [row for row in task_rows if row["split"] == "suc"]),
                ("fail", [row for row in task_rows if row["split"] == "fail"]),
            ):
                by_split[scope] = {
                    "n": len(scoped),
                    "correct": sum(row["prediction"] == row["label"] for row in scoped),
                }
            task_accuracy[condition][task_id] = by_split

    prediction_digest = hashlib.sha256(
        "\n".join(f"{key}\t{predictions[key]}" for key in sorted(predictions)).encode("utf-8")
    ).hexdigest()
    structured = {
        "method": (
            "fixed_baseline_target_control_convex_logprob_mixture"
            if mixture_weights is not None
            else "fixed_target_vs_control_discrete_contrastive_decoding"
        ),
        "labels_model_facing": False,
        "steering_path": str(args.steering.resolve()),
        "steering_sha256": sha256_file(args.steering),
        "source_row_count": len(all_rows),
        "status_counts": dict(sorted(Counter(str(row.get("status")) for row in all_rows).items())),
        "top_k": args.top_k,
        "alpha": args.alpha,
        "formula": args.formula,
        "mixture_weights": (
            {
                "baseline": mixture_weights[0],
                "target": mixture_weights[1],
                "control": mixture_weights[2],
            }
            if mixture_weights is not None
            else None
        ),
        "control_kind": args.control_kind,
        "excluded_ids_file": str(args.exclude_ids.resolve()) if args.exclude_ids else None,
        "excluded_configured_count": len(excluded),
        "analysis_id_count": len(ids),
        "bootstrap_samples": args.bootstrap_samples,
        "candidate_prediction_sha256": prediction_digest,
        "summaries": summaries,
        "paired_vs_baseline": paired,
        "task_accuracy": task_accuracy,
        "prediction_distributions": distributions,
        "pairwise": pairwise,
    }

    lines = [
        "# 固定对比解码完整实验记录",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-09-04",
        "- Verification Status: ANALYZED",
        "- Version Label: contrastive_exp_record_v1",
        "",
        "## 冻结方法与审计口径",
        "",
        (
            "- 方法：三分支 log-prob 凸组合，"
            f"weights=(baseline={mixture_weights[0]:g}, target={mixture_weights[1]:g}, control={mixture_weights[2]:g})，"
            f"top-k=`{args.top_k}`，control=`{args.control_kind}`。"
            if mixture_weights is not None
            else f"- 方法：`{args.formula}`，top-k=`{args.top_k}`，alpha=`{args.alpha:g}`，control=`{args.control_kind}`。"
        ),
        "- target/control 两个模型分支均只接收同一条无标签输入；真实标签仅在全部推理完成后由本报告器读取。",
        f"- 分析样本：{len(ids)}；配置排除 ID：{len(excluded)}；原始 steering SHA-256：`{structured['steering_sha256']}`。",
        "- MAE 为 `mean(abs(prediction-label))`；exact accuracy 要求预测严格等于 suc=5 / fail=1。",
        "- pairwise 定义为同视频 `suc prediction - fail prediction`，每条 fail 各形成一对。",
        "",
        "## 总览",
        "",
        markdown_table(
            ("condition", "n", "suc", "fail", "MAE", "总准确率", "suc 准确率", "fail 准确率"),
            [
                (
                    condition,
                    item["n"],
                    item["suc"],
                    item["fail"],
                    f"{item['mae']:.4f}",
                    pct(item["accuracy"]),
                    pct(item["suc_accuracy"]),
                    pct(item["fail_accuracy"]),
                )
                for condition, item in summaries.items()
            ],
        ),
        "",
        "## 配对推断",
        "",
        f"- corrected/harmed：{paired['corrected_count']}/{paired['harmed_count']}。",
        f"- record-level exact McNemar p：`{paired['exact_mcnemar_pvalue_record_level']:.6g}`。",
        f"- absolute-error change cluster bootstrap：`{json.dumps(paired['absolute_error_change_cluster_bootstrap'], ensure_ascii=False)}`。",
        f"- prediction delta cluster bootstrap：`{json.dumps(paired['prediction_delta_cluster_bootstrap'], ensure_ascii=False)}`。",
        "",
        "## 各 task accuracy",
    ]
    for condition, rows in groups.items():
        lines.extend(["", f"<details><summary>{condition}</summary>", "", task_accuracy_native(rows), "", "</details>"])
    lines.extend(["", "## suc/fail 与各 task 预测分布"])
    for condition, rows in groups.items():
        lines.extend(["", f"<details><summary>{condition}</summary>", "", native_distribution(rows), "", "</details>"])
    lines.extend(
        [
            "",
            "## Pairwise 区分度",
            "",
            pair_table(groups, list(groups), native=True),
            "",
            f"结构化审计结果：`{json_output.name}`。",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(json_output, structured)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(json_output.resolve())


if __name__ == "__main__":
    main()
