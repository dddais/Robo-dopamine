#!/usr/bin/env python3
"""Evaluate a label-blind spatial-counterfactual steering gate.

The gate consumes three predictions that already exist in an append-only
attention-steering run: baseline, target-region steering, and the matched
wrong-region control.  It accepts a target-steered prediction only when the
intervention reaches an ordinal endpoint (1 or 5) and the wrong-region
intervention does not reach the same endpoint.  No reward label, split name,
task ID, or pairing metadata is used by the gate.

Labels and metadata are joined only after all gated predictions have been
materialized.  This keeps inference-time arbitration auditable and makes the
script suitable for re-scoring immutable experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Sequence

from mydata_bench.attention_eval.stats import (
    exact_mcnemar_pvalue,
    paired_cluster_bootstrap,
    paired_sign_flip_pvalue,
)


CONDITIONS = ("baseline", "target", "wrong", "low_rank", "sc_gate")


def spatial_counterfactual_gate(
    baseline: int, target: int, wrong: int
) -> tuple[int, str]:
    """Return a prediction and a label-blind audit reason."""
    for name, value in (("baseline", baseline), ("target", target), ("wrong", wrong)):
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{name} prediction must be an integer in [1, 5]")
    if target in {1, 5} and target != wrong:
        return target, "accept_spatially_specific_endpoint"
    if target not in {1, 5}:
        return baseline, "reject_non_endpoint"
    return baseline, "reject_wrong_region_agreement"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def parse_spec(value: str) -> tuple[str, str, int]:
    try:
        alias, remainder = value.split("=", 1)
        experiment, top_k = remainder.rsplit(":", 1)
        top_k_int = int(top_k)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "spec must have form ALIAS=EXPERIMENT:TOP_K"
        ) from exc
    if not alias or not experiment or top_k_int < 1:
        raise argparse.ArgumentTypeError(
            "spec must have non-empty alias/experiment and positive TOP_K"
        )
    return alias, experiment, top_k_int


def load_metadata(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in read_jsonl(path):
        example_id = str(row["id"])
        if example_id in rows:
            raise ValueError(f"duplicate metadata id: {example_id}")
        rows[example_id] = row
    return rows


def reward_from_id(example_id: str) -> int:
    split = example_id.split("/", 1)[0]
    if split == "suc":
        return 5
    if split == "fail":
        return 1
    raise ValueError(f"cannot infer label from {example_id!r}")


def load_predictions(
    experiment_dir: Path, metadata: dict[str, dict], top_k: int
) -> list[dict]:
    names = {
        "baseline": "baseline",
        "target": f"candidate_target_k{top_k}",
        "wrong": f"candidate_wrong_k{top_k}",
        "low_rank": f"low_rank_target_k{top_k}",
    }
    wanted = set(names.values())
    latest: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(experiment_dir / "steering.jsonl"):
        condition = str(row.get("condition"))
        if condition in wanted:
            latest[(str(row.get("example_id")), condition)] = row

    baseline_ids = {
        example_id
        for (example_id, condition), row in latest.items()
        if condition == names["baseline"] and row.get("status") == "ok"
    }
    if not baseline_ids:
        raise ValueError(f"{experiment_dir.name}: no successful baseline records")
    unknown = baseline_ids - set(metadata)
    if unknown:
        raise ValueError(
            f"{experiment_dir.name}: baseline IDs absent from metadata: {sorted(unknown)[:5]}"
        )
    result = []
    incomplete: list[str] = []
    # The immutable baseline ID set defines the frozen experiment cohort.  The
    # dataset metadata may legitimately contain additional out-of-cohort rows.
    for example_id in sorted(baseline_ids):
        selected = {key: latest.get((example_id, condition)) for key, condition in names.items()}
        if not all(row and row.get("status") == "ok" for row in selected.values()):
            incomplete.append(example_id)
            continue
        predictions = {}
        for key, row in selected.items():
            value = row.get("native_prediction")
            if not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(
                    f"{experiment_dir.name}/{example_id}/{key}: invalid prediction {value!r}"
                )
            predictions[key] = value
        gated, reason = spatial_counterfactual_gate(
            predictions["baseline"], predictions["target"], predictions["wrong"]
        )
        meta = metadata[example_id]
        baseline_row = selected["baseline"]
        result.append(
            {
                "example_id": example_id,
                "video_sha256": str(baseline_row["video_sha256"]),
                "split": example_id.split("/", 1)[0],
                "task_id": str(meta["task_id"]),
                "source_suc_id": str(meta["source_suc_id"]),
                "label": reward_from_id(example_id),
                "predictions": {**predictions, "sc_gate": gated},
                "gate_reason": reason,
            }
        )
    if incomplete:
        raise ValueError(
            f"{experiment_dir.name}: {len(incomplete)} examples lack a complete strict "
            f"baseline/target/wrong/low-rank quartet; first={incomplete[:5]}"
        )
    return result


def summarize(rows: Sequence[dict], condition: str) -> dict:
    if not rows:
        return {"n": 0}
    predictions = [int(row["predictions"][condition]) for row in rows]
    errors = [abs(prediction - int(row["label"])) for prediction, row in zip(predictions, rows)]
    return {
        "n": len(rows),
        "mae": mean(errors),
        "exact_accuracy": mean(error == 0 for error in errors),
        "within_one_accuracy": mean(error <= 1 for error in errors),
        "mean_signed_error": mean(
            prediction - int(row["label"])
            for prediction, row in zip(predictions, rows)
        ),
        "prediction_counts": {
            str(value): Counter(predictions).get(value, 0) for value in range(1, 6)
        },
    }


def paired(rows: Sequence[dict], condition: str, *, samples: int) -> dict:
    paired_rows = []
    for row in rows:
        baseline = int(row["predictions"]["baseline"])
        candidate = int(row["predictions"][condition])
        label = int(row["label"])
        paired_rows.append(
            {
                "example_id": row["example_id"],
                "video_sha256": row["video_sha256"],
                # Task-stratified cluster resampling, matching existing runs.
                "subset": row["task_id"],
                "prediction_delta": candidate - baseline,
                "absolute_error_change": abs(candidate - label) - abs(baseline - label),
                "baseline_correct": baseline == label,
                "candidate_correct": candidate == label,
            }
        )
    corrected = sum(
        not row["baseline_correct"] and row["candidate_correct"] for row in paired_rows
    )
    harmed = sum(
        row["baseline_correct"] and not row["candidate_correct"] for row in paired_rows
    )
    return {
        "n": len(paired_rows),
        "corrected_count": corrected,
        "harmed_count": harmed,
        "absolute_error_change": paired_cluster_bootstrap(
            paired_rows, "absolute_error_change", samples=samples
        ),
        "prediction_delta": paired_cluster_bootstrap(
            paired_rows, "prediction_delta", samples=samples
        ),
        "absolute_error_change_cluster_sign_flip_pvalue": paired_sign_flip_pvalue(
            paired_rows, "absolute_error_change", samples=samples
        ),
        "exact_mcnemar_pvalue_record_level": exact_mcnemar_pvalue(
            paired_rows, "baseline_correct", "candidate_correct"
        ),
    }


def pairwise(rows: Sequence[dict], condition: str) -> dict:
    lookup = {str(row["example_id"]): row for row in rows}
    counts: Counter[str] = Counter()
    missing = 0
    total = 0
    for fail in (row for row in rows if row["split"] == "fail"):
        suc = lookup.get(str(fail["source_suc_id"]))
        if suc is None:
            missing += 1
            continue
        total += 1
        delta = int(suc["predictions"][condition]) - int(fail["predictions"][condition])
        counts["<0" if delta < 0 else str(delta)] += 1
    return {
        "candidate_fail_pairs": sum(row["split"] == "fail" for row in rows),
        "valid_pairs": total,
        "missing_source_suc": missing,
        "bins": {key: counts[key] for key in ("<0", "0", "1", "2", "3", "4")},
    }


def group_summaries(
    rows: Sequence[dict], key: Callable[[dict], str]
) -> dict[str, dict[str, dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {
        group: {condition: summarize(values, condition) for condition in CONDITIONS}
        for group, values in sorted(groups.items())
    }


def evaluate(rows: Sequence[dict], *, samples: int) -> dict:
    overall = {condition: summarize(rows, condition) for condition in CONDITIONS}
    baseline = overall["baseline"]
    gate = overall["sc_gate"]
    by_split = group_summaries(rows, lambda row: str(row["split"]))
    by_task = group_summaries(rows, lambda row: str(row["task_id"]))
    criteria = {
        "mae_decreased": gate["mae"] < baseline["mae"],
        "overall_exact_increased": gate["exact_accuracy"] > baseline["exact_accuracy"],
        "suc_exact_non_decreased": (
            by_split["suc"]["sc_gate"]["exact_accuracy"]
            >= by_split["suc"]["baseline"]["exact_accuracy"]
        ),
        "fail_exact_increased": (
            by_split["fail"]["sc_gate"]["exact_accuracy"]
            > by_split["fail"]["baseline"]["exact_accuracy"]
        ),
        "beats_wrong_exact": gate["exact_accuracy"] > overall["wrong"]["exact_accuracy"],
        "beats_low_rank_exact": gate["exact_accuracy"] > overall["low_rank"]["exact_accuracy"],
    }
    return {
        "n": len(rows),
        "gate_is_label_blind": True,
        "gate_rule": (
            "accept target iff target in {1,5} and target != wrong; otherwise baseline"
        ),
        "gate_reasons": dict(sorted(Counter(row["gate_reason"] for row in rows).items())),
        "overall": overall,
        "by_split": by_split,
        "by_task": by_task,
        "pairwise": {condition: pairwise(rows, condition) for condition in CONDITIONS},
        "paired_vs_baseline": {
            condition: paired(rows, condition, samples=samples)
            for condition in ("target", "wrong", "low_rank", "sc_gate")
        },
        "acceptance_criteria": criteria,
        "passes_all_acceptance_criteria": all(criteria.values()),
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(results: dict[str, dict]) -> str:
    lines = [
        "# Spatial-Counterfactual Endpoint Gate 验证",
        "",
        "## Material Passport",
        "",
        "- Artifact type: post-hoc mechanism validation over immutable real forward passes",
        "- Verification status: ANALYZED（未重新执行模型；全部输入来自完整 steering artifacts）",
        "- Label boundary: gate 仅见 baseline/target/wrong prediction；label、split、task 与 pair metadata 只在 gate 输出冻结后回连评分",
        "- Rule: target 为端点 1/5 且 wrong-region 不同意该端点时接受 target，否则回退 baseline",
        "",
        "## 总体结果",
        "",
        "| 模型/配置 | 条件 | n | Exact | MAE | suc Exact | fail Exact |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for alias, result in results.items():
        for condition in CONDITIONS:
            value = result["overall"][condition]
            lines.append(
                "| {} | {} | {} | {} | {:.4f} | {} | {} |".format(
                    alias,
                    condition,
                    value["n"],
                    pct(value["exact_accuracy"]),
                    value["mae"],
                    pct(result["by_split"]["suc"][condition]["exact_accuracy"]),
                    pct(result["by_split"]["fail"][condition]["exact_accuracy"]),
                )
            )
    lines.extend(["", "## 聚类配对效应", ""])
    for alias, result in results.items():
        gate = result["paired_vs_baseline"]["sc_gate"]
        effect = gate["absolute_error_change"]
        lines.extend(
            [
                f"### {alias}",
                "",
                f"- 绝对误差变化（cluster mean）：{effect['mean']:.4f}，95% CI "
                f"[{effect['ci95'][0]:.4f}, {effect['ci95'][1]:.4f}]。",
                f"- 修正/损害：{gate['corrected_count']}/{gate['harmed_count']}；"
                f"McNemar p={gate['exact_mcnemar_pvalue_record_level']:.4g}。",
                f"- 全部预设验收条件：{result['passes_all_acceptance_criteria']}。",
                "",
            ]
        )
    lines.extend(["## Pairwise suc−fail 区分度", ""])
    for alias, result in results.items():
        base = result["pairwise"]["baseline"]
        gate = result["pairwise"]["sc_gate"]
        lines.append(
            f"- {alias}: baseline {base['bins']} → sc_gate {gate['bins']} "
            f"（有效 pair {gate['valid_pairs']}）。"
        )
    lines.extend(["", "## 逐 task 稳定性", ""])
    for alias, result in results.items():
        improved_mae = 0
        nondecreased_exact = 0
        for values in result["by_task"].values():
            improved_mae += values["sc_gate"]["mae"] < values["baseline"]["mae"]
            nondecreased_exact += (
                values["sc_gate"]["exact_accuracy"]
                >= values["baseline"]["exact_accuracy"]
            )
        total = len(result["by_task"])
        lines.append(
            f"- {alias}: MAE 下降 {improved_mae}/{total} tasks；Exact 不下降 "
            f"{nondecreased_exact}/{total} tasks。"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--spec", action="append", type=parse_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    metadata = load_metadata(args.metadata.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    all_records: list[dict] = []
    seen_aliases: set[str] = set()
    for alias, experiment, top_k in args.spec:
        if alias in seen_aliases:
            raise ValueError(f"duplicate alias: {alias}")
        seen_aliases.add(alias)
        rows = load_predictions(args.experiments_root / experiment, metadata, top_k)
        all_results[alias] = {
            "source_experiment": experiment,
            "top_k": top_k,
            **evaluate(rows, samples=args.bootstrap_samples),
        }
        all_records.extend({"model_alias": alias, **row} for row in rows)

    with (output / "gated_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    payload = {
        "schema_version": "spatial-counterfactual-endpoint-gate-v1",
        "metadata_path": str(args.metadata.resolve()),
        "bootstrap_samples": args.bootstrap_samples,
        "results": all_results,
        "cross_model_pass": all(
            result["passes_all_acceptance_criteria"] for result in all_results.values()
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        render_markdown(all_results), encoding="utf-8"
    )
    print(output / "metrics.json")


if __name__ == "__main__":
    main()
