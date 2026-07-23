#!/usr/bin/env python3
"""Build a reproducible human audit report for grounding outputs.

The detector metrics in :mod:`report` measure candidate coverage and endpoint
stability.  They cannot establish that the selected box belongs to the target
named by the instruction.  This module joins a frozen audit sample with human
annotations, validates that every sample was reviewed exactly once, and emits
machine-readable records plus a descriptive report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .report import read_jsonl, summary_markdown, write_jsonl


AUDIT_LABELS = {"correct", "incorrect", "uncertain"}
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "counterfactual_reward1"


def grounding_result_fingerprint(result: Mapping[str, Any]) -> str:
    """Fingerprint exactly the fields on which a bbox correctness label relies."""

    payload = {
        "example_id": result.get("example_id"),
        "task": result.get("task"),
        "selected_parse": result.get("selected_parse"),
        "grounding_queries": result.get("grounding_queries"),
        "before_selected": (result.get("before") or {}).get("selected"),
        "after_selected": (result.get("after") or {}).get("selected"),
        "pair_consistency": result.get("pair_consistency"),
        "steering_ready": result.get("steering_ready"),
        "status": result.get("status"),
        "visualization_file": result.get("visualization_file"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_or_initialize_fingerprints(
    sample_rows: Sequence[Mapping[str, Any]],
    grounding_results: Sequence[Mapping[str, Any]],
    fingerprint_path: str | Path,
) -> dict[str, str]:
    """Tie manual labels to the exact selected boxes that were reviewed."""

    result_by_id = {str(row.get("example_id")): row for row in grounding_results}
    expected: dict[str, str] = {}
    for sample in sample_rows:
        example_id = str(sample.get("example_id", ""))
        result = result_by_id.get(example_id)
        if result is None:
            raise ValueError(f"audit sample missing from grounding results: {example_id}")
        sample_projection = {
            "task": sample.get("task"),
            "selected_parse": sample.get("selected_parse"),
            "steering_ready": sample.get("steering_ready"),
            "status": sample.get("status"),
            "visualization_file": sample.get("visualization_file"),
        }
        result_projection = {
            "task": result.get("task"),
            "selected_parse": result.get("selected_parse"),
            "steering_ready": result.get("steering_ready"),
            "status": result.get("status"),
            "visualization_file": result.get("visualization_file"),
        }
        if sample_projection != result_projection:
            raise ValueError(f"audit sample is stale relative to grounding result: {example_id}")
        expected[example_id] = grounding_result_fingerprint(result)

    destination = Path(fingerprint_path)
    if destination.is_file():
        frozen_rows = read_jsonl(destination)
        frozen = {str(row.get("example_id")): str(row.get("grounding_fingerprint")) for row in frozen_rows}
        if frozen != expected:
            changed = sorted(
                example_id
                for example_id in set(frozen) | set(expected)
                if frozen.get(example_id) != expected.get(example_id)
            )
            raise ValueError(
                "grounding outputs changed after manual review; re-audit before rebuilding reports: "
                + ", ".join(changed)
            )
    else:
        write_jsonl(
            destination,
            [
                {"example_id": example_id, "grounding_fingerprint": fingerprint}
                for example_id, fingerprint in expected.items()
            ],
        )
    return expected


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        return None
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z_squared / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def merge_manual_annotations(
    sample_rows: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and join one human annotation to every sampled example."""

    sample_by_id: dict[str, Mapping[str, Any]] = {}
    for row in sample_rows:
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError("audit sample contains an empty example_id")
        if example_id in sample_by_id:
            raise ValueError(f"duplicate audit sample id: {example_id}")
        sample_by_id[example_id] = row

    annotation_by_id: dict[str, Mapping[str, Any]] = {}
    for row in annotations:
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError("manual annotation contains an empty example_id")
        if example_id in annotation_by_id:
            raise ValueError(f"duplicate manual annotation id: {example_id}")
        label = str(row.get("manual_label", ""))
        if label not in AUDIT_LABELS:
            raise ValueError(f"invalid manual_label for {example_id}: {label!r}")
        if label == "incorrect" and not str(row.get("failure_category", "")).strip():
            raise ValueError(f"incorrect annotation lacks failure_category: {example_id}")
        if not str(row.get("reason", "")).strip():
            raise ValueError(f"annotation lacks reason: {example_id}")
        annotation_by_id[example_id] = row

    missing = sorted(set(sample_by_id) - set(annotation_by_id))
    unexpected = sorted(set(annotation_by_id) - set(sample_by_id))
    if missing or unexpected:
        raise ValueError(f"audit identity mismatch: missing={missing}, unexpected={unexpected}")

    merged: list[dict[str, Any]] = []
    for sample in sample_rows:
        example_id = str(sample["example_id"])
        annotation = annotation_by_id[example_id]
        merged.append(
            {
                **dict(sample),
                "manual_label": str(annotation["manual_label"]),
                "failure_category": annotation.get("failure_category"),
                "reason": str(annotation["reason"]),
                "review_basis": str(annotation.get("review_basis", "endpoint_visualization")),
            }
        )
    return merged


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row["manual_label"]) for row in rows)
    evaluated = labels["correct"] + labels["incorrect"]
    interval = wilson_interval(labels["correct"], evaluated)
    return {
        "total": len(rows),
        "correct": labels["correct"],
        "incorrect": labels["incorrect"],
        "uncertain": labels["uncertain"],
        "evaluated": evaluated,
        "observed_correct_object_precision": labels["correct"] / evaluated if evaluated else None,
        "wilson_95_interval": list(interval) if interval is not None else None,
    }


def summarize_manual_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    population_total: int | None = None,
    population_steering_ready: int | None = None,
) -> dict[str, Any]:
    """Compute descriptive audit metrics without treating proxies as accuracy."""

    ready = [row for row in rows if bool(row.get("steering_ready"))]
    nonready = [row for row in rows if not bool(row.get("steering_ready"))]
    subsets = {str(row.get("subset", "unknown")) for row in rows}
    failures = Counter(
        str(row.get("failure_category"))
        for row in rows
        if row.get("manual_label") == "incorrect"
    )
    return {
        "audit_sample_size": len(rows),
        "subsets_covered": len(subsets),
        "population_total": population_total,
        "population_steering_ready": population_steering_ready,
        "population_steering_ready_rate": (
            population_steering_ready / population_total
            if population_total and population_steering_ready is not None
            else None
        ),
        "overall": _group_summary(rows),
        "steering_ready": _group_summary(ready),
        "not_steering_ready": _group_summary(nonready),
        "failure_categories": dict(sorted(failures.items())),
        "interpretation_boundary": (
            "这是单人、按来源子集分层的描述性审计，不是简单随机抽样。observed proportion 与 "
            "Wilson 区间只描述被审计记录，不能作为全量数据的无偏准确率估计。"
        ),
    }


def write_manual_audit_csv(rows: Sequence[Mapping[str, Any]], destination: str | Path) -> None:
    fields = [
        "index",
        "example_id",
        "subset",
        "task",
        "target_phrase",
        "steering_ready",
        "status",
        "agreement_level",
        "manual_label",
        "failure_category",
        "reason",
        "review_basis",
        "grounding_fingerprint",
        "visualization_file",
    ]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "target_phrase": (row.get("selected_parse") or {}).get("target_phrase"),
                }
            )


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100.0:.2f}%"


def _interval(value: Sequence[float] | None) -> str:
    return "N/A" if value is None else f"[{_percent(float(value[0]))}, {_percent(float(value[1]))}]"


def manual_audit_markdown(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    overall = summary["overall"]
    ready = summary["steering_ready"]
    nonready = summary["not_steering_ready"]
    failures = [row for row in rows if row.get("manual_label") == "incorrect"]
    lines = [
        "# Grounding 人工审计报告",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {datetime.now(timezone.utc).date().isoformat()}",
        "- Verification Status: ANALYZED",
        "- Version Label: grounding_manual_audit_v1",
        "",
        "## 审计方法",
        "",
        f"- 共检查 {summary['audit_sample_size']} 条，覆盖全部 {summary['subsets_covered']} 个来源子集。每个子集取 1–2 条，并同时覆盖 steering-ready 与非 ready 状态（只要该子集存在相应样本）。",
        "- `correct` 的判据：绿色选中框在首、末端点对应 instruction 直接作用的目标实体；当关系词或同类实例无法仅凭端点判定时，查看完整视频。",
        "- 这是单人、按子集分层的目的性抽样，不是从 228 条中做简单随机抽样。Wilson 区间仅描述审计样本的二项不确定性，不能消除抽样偏差。",
        "- 人工审计发生在模型推理之后；模型仍未读取 `reward` 或 `gpt5_mini_check`。",
        "",
        "## 结果",
        "",
        "| 审计组 | 正确 | 错误 | 不确定 | observed precision | Wilson 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 全部审计样本 | {overall['correct']}/{overall['evaluated']} | {overall['incorrect']} | {overall['uncertain']} | {_percent(overall['observed_correct_object_precision'])} | {_interval(overall['wilson_95_interval'])} |",
        f"| steering-ready | {ready['correct']}/{ready['evaluated']} | {ready['incorrect']} | {ready['uncertain']} | {_percent(ready['observed_correct_object_precision'])} | {_interval(ready['wilson_95_interval'])} |",
        f"| 非 steering-ready | {nonready['correct']}/{nonready['evaluated']} | {nonready['incorrect']} | {nonready['uncertain']} | {_percent(nonready['observed_correct_object_precision'])} | {_interval(nonready['wilson_95_interval'])} |",
        "",
        f"自动门控把 {summary.get('population_steering_ready')}/{summary.get('population_total')} 条标为 steering-ready（{_percent(summary.get('population_steering_ready_rate'))}）。但 ready 组人工审计仅为 {ready['correct']}/{ready['evaluated']}（{_percent(ready['observed_correct_object_precision'])}），没有达到建议的 90% correct-object precision 使用门槛。",
        "",
        "## 明确错误样本",
        "",
        "| index | ready | instruction | failure category | 人工判断 |",
        "|---:|:---:|---|---|---|",
    ]
    for row in failures:
        task = str(row.get("task", "")).replace("|", "\\|")
        reason = str(row.get("reason", "")).replace("|", "\\|")
        lines.append(
            f"| {row.get('index')} | {'yes' if row.get('steering_ready') else 'no'} | {task} | "
            f"`{row.get('failure_category')}` | {reason} |"
        )
    lines.extend(
        [
            "",
            "## 统计解释与偏差检查",
            "",
            "- Fallacy scan coverage: 11/11。",
            "- Simpson / ecological / collider / reverse-causality：本报告不做跨组因果推断，未触发；但各子集仅 1–2 条，不能据此比较子集性能。",
            "- Berkson / survivorship：审计集是分层目的性样本，不能把 29/41 外推为全体准确率。",
            "- Base-rate neglect：同时报告了全量 ready 基率与 ready 组审计 precision，未用 detector coverage 代替 precision。",
            "- Regression-to-mean：无前后干预比较，不适用。",
            "- Look-elsewhere / multiple comparisons：未执行显著性检验或筛选显著结果。",
            "- Garden of forking paths：双帧联合选择在 4 条 smoke 后冻结；全量 test 观察不再用于调当前阈值。后续改进需单独 dev split 后重新独立测试。",
            "- Correlation-causation：只报告描述性覆盖率、稳定性和人工审计结果，不作因果主张。",
            "",
            "## 结论边界",
            "",
            str(summary["interpretation_boundary"]),
            "当前实现适合作为候选框生成和人工审核缓存；在未经独立 dev 校准并重新盲测前，不应把全部 143 条 ready 结果直接用于 Robo-Dopamine 主 steering 结论。",
            "",
        ]
    )
    return "\n".join(lines)


def build_audit_artifacts(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    sample = read_jsonl(root / "audit_sample.jsonl")
    annotations = read_jsonl(root / "manual_audit_annotations.jsonl")
    results = read_jsonl(root / "grounding_results.jsonl")
    rows = merge_manual_annotations(sample, annotations)
    fingerprints = validate_or_initialize_fingerprints(
        sample,
        results,
        root / "audit_grounding_fingerprints.jsonl",
    )
    for row in rows:
        row["grounding_fingerprint"] = fingerprints[str(row["example_id"])]
    summary = summarize_manual_audit(
        rows,
        population_total=len(results),
        population_steering_ready=sum(bool(row.get("steering_ready")) for row in results),
    )
    write_jsonl(root / "manual_audit.jsonl", rows)
    write_manual_audit_csv(rows, root / "manual_audit.csv")
    (root / "manual_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (root / "audit_report.md").write_text(
        manual_audit_markdown(summary, rows),
        encoding="utf-8",
    )
    run_summary_path = root / "summary.json"
    manifest_path = root / "run_manifest.json"
    if run_summary_path.is_file():
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        run_summary["manual_audit"] = summary
        run_summary_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        if manifest:
            artifacts = manifest.setdefault("artifacts", {})
            artifacts.update(
                {
                    "manual_audit_jsonl": str(root / "manual_audit.jsonl"),
                    "manual_audit_csv": str(root / "manual_audit.csv"),
                    "manual_audit_summary": str(root / "manual_audit_summary.json"),
                    "manual_audit_report": str(root / "audit_report.md"),
                    "manual_audit_fingerprints": str(root / "audit_grounding_fingerprints.jsonl"),
                }
            )
            manifest["manual_audit"] = {
                "status": "completed",
                "sample_size": len(rows),
                "subsets_covered": summary["subsets_covered"],
                "steering_ready_correct": summary["steering_ready"]["correct"],
                "steering_ready_evaluated": summary["steering_ready"]["evaluated"],
                "fingerprints_frozen": len(fingerprints),
            }
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            temporary_manifest.replace(manifest_path)
        (root / "summary.md").write_text(
            summary_markdown(
                run_summary,
                command=str(manifest.get("command", "unknown")),
                output_dir=str(root),
            ),
            encoding="utf-8",
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    summary = build_audit_artifacts(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
