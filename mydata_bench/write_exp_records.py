#!/usr/bin/env python3
"""Generate descriptive experiment records for completed mydata_bench runs.

The generated ``exp_record.md`` files are deliberately derived only from the
immutable JSON/JSONL result artifacts and the dataset metadata.  Re-running
this script therefore provides a cheap audit of all reported counts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence


GRM_ENDPOINT_THRESHOLDS = ((0.125, 0.875), (0.2, 0.8))
GRM_ORDINAL_THRESHOLDS = (0.125, 0.375, 0.625, 0.875)
GRM_DISTRIBUTION_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)
NATIVE_PAIR_BINS = ("<0", "0", "1", "2", "3", "4")
GRM_PAIR_BINS = (
    "<0",
    "0–10%",
    "10–20%",
    "20–30%",
    "30–50%",
    "50–60%",
    "60–70%",
    "70–80%",
    "80–90%",
    "≥90%",
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def latest_result_rows(rows: Sequence[dict], *, attention: bool) -> list[dict]:
    """Match the append-only resume semantics used by the evaluation code."""
    latest: dict[tuple[str, ...], dict] = {}
    for row in rows:
        example_id = str(row.get("example_id"))
        key = (example_id, str(row.get("condition"))) if attention else (example_id,)
        latest[key] = row
    return list(latest.values())


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def count_rate(count: int, total: int) -> str:
    return f"{count} ({pct(count / total if total else None)})"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def split_of(example_id: str) -> str:
    split = example_id.split("/", 1)[0]
    if split not in {"suc", "fail"}:
        raise ValueError(f"Cannot infer suc/fail label from example_id={example_id!r}")
    return split


def reward_of(example_id: str) -> int:
    return 5 if split_of(example_id) == "suc" else 1


def ordinal_prediction(progress: float) -> int:
    return 1 + sum(float(progress) >= threshold for threshold in GRM_ORDINAL_THRESHOLDS)


def endpoint_prediction(progress: float, low: float, high: float) -> str:
    if progress <= low:
        return "1"
    if progress >= high:
        return "5"
    return "uncertain"


def grm_distribution_label(progress: float) -> int:
    """Map continuous progress to five equal-width bins for distributions."""
    return 1 + sum(float(progress) >= threshold for threshold in GRM_DISTRIBUTION_THRESHOLDS)


def accuracy_cell(rows: Sequence[dict], prediction_key: str = "prediction") -> str:
    correct = sum(int(row[prediction_key]) == int(row["label"]) for row in rows)
    return count_rate(correct, len(rows))


def native_pair_bin(delta: int) -> str:
    if delta < 0:
        return "<0"
    if delta > 4:
        raise ValueError(f"Native pair delta outside expected range: {delta}")
    return str(delta)


def grm_pair_bin(delta: float) -> str:
    if delta < 0:
        return "<0"
    if delta < 0.1:
        return "0–10%"
    if delta < 0.2:
        return "10–20%"
    if delta < 0.3:
        return "20–30%"
    if delta < 0.5:
        return "30–50%"
    if delta < 0.6:
        return "50–60%"
    if delta < 0.7:
        return "60–70%"
    if delta < 0.8:
        return "70–80%"
    if delta < 0.9:
        return "80–90%"
    return "≥90%"


def enrich_rows(rows: Sequence[dict], metadata: Mapping[str, dict], *, native: bool) -> list[dict]:
    enriched = []
    for source in rows:
        if source.get("status") != "ok":
            continue
        example_id = str(source["example_id"])
        if example_id not in metadata:
            raise ValueError(f"Result is absent from metadata: {example_id}")
        row = dict(source)
        row["label"] = reward_of(example_id)
        row["split"] = split_of(example_id)
        row["task_id"] = str(metadata[example_id]["task_id"])
        row["source_suc_id"] = str(metadata[example_id]["source_suc_id"])
        if native:
            value = row.get("native_prediction")
            if not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"Invalid native_prediction for {example_id}: {value!r}")
            row["prediction"] = value
        else:
            progress = float(row["progress"])
            if not 0 <= progress <= 1:
                raise ValueError(f"Invalid progress for {example_id}: {progress}")
            row["progress"] = progress
            row["prediction"] = ordinal_prediction(progress)
        enriched.append(row)
    return enriched


def condition_groups(rows: Sequence[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("condition", "baseline"))].append(row)
    for condition, values in groups.items():
        ids = [row["example_id"] for row in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate example_id within condition {condition}")
    return dict(groups)


def ordered_conditions(groups: Mapping[str, Sequence[dict]], preferred: Sequence[str]) -> list[str]:
    seen = set(groups)
    result = [condition for condition in preferred if condition in seen]
    result.extend(sorted(seen - set(result)))
    return result


def native_summary(rows: Sequence[dict]) -> dict:
    errors = [abs(int(row["prediction"]) - int(row["label"])) for row in rows]
    suc = [row for row in rows if row["split"] == "suc"]
    fail = [row for row in rows if row["split"] == "fail"]
    return {
        "n": len(rows),
        "suc": len(suc),
        "fail": len(fail),
        "mae": mean(errors) if errors else None,
        "accuracy": mean(error == 0 for error in errors) if errors else None,
        "suc_accuracy": mean(row["prediction"] == row["label"] for row in suc) if suc else None,
        "fail_accuracy": mean(row["prediction"] == row["label"] for row in fail) if fail else None,
    }


def grm_summary(rows: Sequence[dict]) -> dict:
    ordinal_errors = [abs(int(row["prediction"]) - int(row["label"])) for row in rows]
    continuous_errors = [abs(1 + 4 * row["progress"] - int(row["label"])) for row in rows]
    result = {
        "n": len(rows),
        "suc": sum(row["split"] == "suc" for row in rows),
        "fail": sum(row["split"] == "fail" for row in rows),
        "mae": mean(ordinal_errors) if ordinal_errors else None,
        "continuous_mae": mean(continuous_errors) if continuous_errors else None,
    }
    for low, high in GRM_ENDPOINT_THRESHOLDS:
        key = f"{low:g}/{high:g}"
        for split, values in (
            ("overall", rows),
            ("suc", [row for row in rows if row["split"] == "suc"]),
            ("fail", [row for row in rows if row["split"] == "fail"]),
        ):
            result[f"{key}:{split}"] = (
                mean(endpoint_prediction(row["progress"], low, high) == str(row["label"]) for row in values)
                if values
                else None
            )
    return result


def assert_metric_matches(name: str, actual: float | int | None, expected: float | int | None) -> None:
    if actual is None or expected is None or abs(float(actual) - float(expected)) > 1e-12:
        raise ValueError(f"Metric mismatch for {name}: records={actual!r}, metrics={expected!r}")


def validate_existing_metrics(
    experiment: Path,
    groups: Mapping[str, Sequence[dict]],
    *,
    native: bool,
    attention: bool,
) -> None:
    """Cross-check report statistics against metrics already emitted by a run."""
    if attention and not native:
        # GRM attention_metrics contains causal estimands, not descriptive MAE.
        return
    if attention:
        existing = read_json(experiment / "steering_metrics.json").get("by_condition", {})
        for condition, rows in groups.items():
            summary = native_summary(rows)
            expected = existing[condition]
            assert_metric_matches(f"{experiment.name}/{condition}/n", summary["n"], expected["n"])
            assert_metric_matches(f"{experiment.name}/{condition}/mae", summary["mae"], expected["mae"])
            assert_metric_matches(
                f"{experiment.name}/{condition}/exact_accuracy", summary["accuracy"], expected["exact_accuracy"]
            )
        return
    existing = read_json(experiment / "metrics.json")["micro"]
    summary = native_summary(groups["baseline"]) if native else grm_summary(groups["baseline"])
    assert_metric_matches(f"{experiment.name}/n", summary["n"], existing["n"])
    assert_metric_matches(f"{experiment.name}/mae", summary["mae"], existing["mae"])
    if not native:
        assert_metric_matches(
            f"{experiment.name}/continuous_ordinal_mae",
            summary["continuous_mae"],
            existing["continuous_ordinal_mae"],
        )


def task_accuracy_native(rows: Sequence[dict]) -> str:
    table_rows = []
    for task_id in sorted({row["task_id"] for row in rows}):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        suc = [row for row in task_rows if row["split"] == "suc"]
        fail = [row for row in task_rows if row["split"] == "fail"]
        table_rows.append((task_id, len(task_rows), accuracy_cell(task_rows), accuracy_cell(suc), accuracy_cell(fail)))
    return markdown_table(("task", "n", "总准确率", "suc 准确率", "fail 准确率"), table_rows)


def endpoint_accuracy_cell(rows: Sequence[dict], low: float, high: float) -> str:
    correct = sum(endpoint_prediction(row["progress"], low, high) == str(row["label"]) for row in rows)
    return count_rate(correct, len(rows))


def task_accuracy_grm(rows: Sequence[dict], low: float, high: float) -> str:
    table_rows = []
    for task_id in sorted({row["task_id"] for row in rows}):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        suc = [row for row in task_rows if row["split"] == "suc"]
        fail = [row for row in task_rows if row["split"] == "fail"]
        table_rows.append(
            (task_id, len(task_rows), endpoint_accuracy_cell(task_rows, low, high), endpoint_accuracy_cell(suc, low, high), endpoint_accuracy_cell(fail, low, high))
        )
    return markdown_table(("task", "n", "总准确率", "suc 准确率", "fail 准确率"), table_rows)


def distribution_scopes(rows: Sequence[dict]) -> list[tuple[str, list[dict]]]:
    result = [
        ("split:suc", [row for row in rows if row["split"] == "suc"]),
        ("split:fail", [row for row in rows if row["split"] == "fail"]),
    ]
    result.extend(
        (f"task:{task_id}", [row for row in rows if row["task_id"] == task_id])
        for task_id in sorted({row["task_id"] for row in rows})
    )
    return result


def native_distribution(rows: Sequence[dict]) -> str:
    table_rows = []
    for scope, values in distribution_scopes(rows):
        counts = Counter(str(row["prediction"]) for row in values)
        table_rows.append((scope, len(values), *(count_rate(counts[str(label)], len(values)) for label in range(1, 6))))
    return markdown_table(("范围", "n", "label 1", "label 2", "label 3", "label 4", "label 5"), table_rows)


def grm_distribution(rows: Sequence[dict]) -> str:
    table_rows = []
    for scope, values in distribution_scopes(rows):
        counts = Counter(grm_distribution_label(row["progress"]) for row in values)
        table_rows.append(
            (
                scope,
                len(values),
                *(count_rate(counts[label], len(values)) for label in range(1, 6)),
            )
        )
    return markdown_table(
        (
            "范围",
            "n",
            "label 1 (0–20%)",
            "label 2 (20–40%)",
            "label 3 (40–60%)",
            "label 4 (60–80%)",
            "label 5 (80–100%)",
        ),
        table_rows,
    )


def pair_stats(rows: Sequence[dict], *, native: bool) -> dict:
    lookup = {str(row["example_id"]): row for row in rows}
    fail_rows = [row for row in rows if row["split"] == "fail"]
    missing_metadata = sum(not row.get("source_suc_id") for row in fail_rows)
    missing_source = 0
    counts: Counter[str] = Counter()
    for fail in fail_rows:
        source_id = fail.get("source_suc_id")
        if not source_id:
            continue
        suc = lookup.get(str(source_id))
        if suc is None:
            missing_source += 1
            continue
        if suc["split"] != "suc":
            raise ValueError(f"source_suc_id does not point to suc: {source_id}")
        if native:
            counts[native_pair_bin(int(suc["prediction"]) - int(fail["prediction"]))] += 1
        else:
            counts[grm_pair_bin(float(suc["progress"]) - float(fail["progress"]))] += 1
    bins = NATIVE_PAIR_BINS if native else GRM_PAIR_BINS
    valid = sum(counts.values())
    if valid + missing_source + missing_metadata != len(fail_rows):
        raise AssertionError("Pair accounting mismatch")
    return {
        "candidate": len(fail_rows),
        "valid": valid,
        "missing_metadata": missing_metadata,
        "missing_source": missing_source,
        "counts": {key: counts[key] for key in bins},
        "bins": bins,
    }


def pair_table(groups: Mapping[str, Sequence[dict]], conditions: Sequence[str], *, native: bool) -> str:
    stats = [(condition, pair_stats(groups[condition], native=native)) for condition in conditions]
    bins = NATIVE_PAIR_BINS if native else GRM_PAIR_BINS
    headers = ("condition", "候选", "有效", "缺 source suc", "缺配对元数据", *bins)
    table_rows = []
    for condition, item in stats:
        table_rows.append(
            (
                condition,
                item["candidate"],
                item["valid"],
                item["missing_source"],
                item["missing_metadata"],
                *(count_rate(item["counts"][key], item["valid"]) for key in bins),
            )
        )
    return markdown_table(headers, table_rows)


def dataset_overlap_note(rows: Sequence[dict], ranking_metadata: Path) -> str:
    if not ranking_metadata.exists():
        return "- 未找到 ranking metadata，无法审计 ranking/cohort ID 重叠。"
    ranking_ids = {str(row["id"]) for row in read_jsonl(ranking_metadata)}
    cohort_ids = {str(row["example_id"]) for row in rows}
    overlap = len(ranking_ids & cohort_ids)
    return f"- ranking 数据与评测 cohort 的 ID 重叠：**{overlap}/{len(ranking_ids)}**；这些样本并非独立 hold-out。"


def native_effect_summary(metrics: Mapping[str, object]) -> str:
    paired = metrics.get("paired_vs_baseline", {})
    if not isinstance(paired, dict) or not paired:
        return "现有 metrics 中没有 paired-vs-baseline 结果。"
    rows = []
    for condition, value in sorted(paired.items()):
        if not isinstance(value, dict):
            continue
        estimand = value.get("cluster_estimands", {}).get("absolute_error_change", {})
        rows.append(
            (
                condition,
                value.get("n", "—"),
                number(value.get("mean_score_delta_vs_baseline")),
                number(estimand.get("mean")),
                f"[{number((estimand.get('ci95') or [None, None])[0])}, {number((estimand.get('ci95') or [None, None])[1])}]",
                f"{value.get('corrected_count', '—')}/{value.get('harmed_count', '—')}",
            )
        )
    return markdown_table(("condition", "n", "均值分数变化", "绝对误差变化", "95% CI", "纠正/损害"), rows)


def grm_effect_summary(metrics: Mapping[str, object]) -> str:
    top_k = metrics.get("top_k_estimands", {})
    rows = []
    if isinstance(top_k, dict):
        for key in sorted(top_k, key=lambda value: int(value)):
            estimands = top_k[key].get("estimands", {})
            for name in ("target_shift", "spatial_specificity", "head_specificity"):
                value = estimands.get(name, {})
                ci = value.get("ci95") or [None, None]
                rows.append((key, name, number(value.get("mean"), 6), f"[{number(ci[0], 6)}, {number(ci[1], 6)}]"))
    gate = metrics.get("target_head_specific_causal_effect_supported")
    gate_text = f"正式 causal gate：`target_head_specific_causal_effect_supported={gate}`。"
    return gate_text + "\n\n" + markdown_table(("top-k", "estimand", "mean", "95% CI"), rows)


def attention_cohort_count(experiment: Path, rows: Sequence[dict]) -> int:
    """Return the declared cohort size, falling back to completed result IDs."""
    for filename, key in (
        ("cohort_manifest.json", "expected_count"),
        ("prepare_manifest.json", "eligible_count"),
    ):
        path = experiment / filename
        if path.is_file():
            value = read_json(path).get(key)
            if isinstance(value, int):
                return value
    return len(
        {
            str(row["example_id"])
            for row in rows
            if isinstance(row.get("example_id"), str)
        }
    )


def report_header(experiment: Path, source_file: str, rows: Sequence[dict], *, attention: bool, ranking_metadata: Path) -> list[str]:
    lines = [
        f"# {experiment.name} 实验记录",
        "",
        "> 本文件由 `mydata_bench/write_exp_records.py` 从结果 JSONL 自动生成；不要手工修改统计表。",
        "",
        "## 数据与统计口径",
        "",
        f"- 结果源：`{source_file}`；append-only 重跑记录按 key 取最后一条，再仅统计 `status=ok` 的记录。",
        "- 标签：`suc=5`，`fail=1`，由 `example_id` 前缀确定，并与 `metadata.jsonl` 交叉关联 task 和 `source_suc_id`。",
        "- 表内 `count (rate)` 的概率分母始终是该行的 `n`；准确率里的 uncertain 计为不正确。",
        "- pairwise 差值定义为 `同视频 suc 预测 - fail 预测`；一条 suc 对应多个 fail 时，每个 fail 各计一个 pair。",
    ]
    if attention:
        cohort_count = attention_cohort_count(experiment, rows)
        lines.extend(
            [
                f"- 本实验 cohort 是自动 grounding 双端点成功的 {cohort_count} 条样本，尚未经过人工审核；结论仅适用于该筛选 cohort。",
                dataset_overlap_note(rows, ranking_metadata),
            ]
        )
    return lines


def write_native_report(experiment: Path, rows: Sequence[dict], metadata: Mapping[str, dict], *, attention: bool, ranking_metadata: Path) -> None:
    source_file = "steering.jsonl" if attention else "records.shard-00.jsonl"
    enriched = enrich_rows(rows, metadata, native=True)
    groups = condition_groups(enriched)
    validate_existing_metrics(experiment, groups, native=True, attention=attention)
    preferred = []
    if attention:
        manifest = read_json(experiment / "steering_manifest.json")
        preferred = [str(value) for value in manifest.get("conditions", [])]
    conditions = ordered_conditions(groups, preferred)
    lines = report_header(experiment, source_file, enriched, attention=attention, ranking_metadata=ranking_metadata)
    lines.extend(["", "## 总览：MAE 与准确率", ""])
    summary_rows = []
    for condition in conditions:
        item = native_summary(groups[condition])
        summary_rows.append(
            (
                condition,
                item["n"],
                item["suc"],
                item["fail"],
                number(item["mae"]),
                pct(item["accuracy"]),
                pct(item["suc_accuracy"]),
                pct(item["fail_accuracy"]),
            )
        )
    lines.append(markdown_table(("condition", "n", "suc", "fail", "MAE", "总准确率", "suc 准确率", "fail 准确率"), summary_rows))
    lines.extend(
        [
            "",
            "MAE 按 RoboReward 离散输出原定义计算：`mean(abs(native_prediction-label))`；准确率要求预测值与 1/5 标签完全相同。",
            "",
            "## 各 task 准确率",
        ]
    )
    for condition in conditions:
        lines.extend(["", f"<details><summary>{condition}</summary>", "", task_accuracy_native(groups[condition]), "", "</details>"])
    lines.extend(["", "## 预测 label 分布"])
    for condition in conditions:
        lines.extend(["", f"<details><summary>{condition}</summary>", "", native_distribution(groups[condition]), "", "</details>"])
    lines.extend(
        [
            "",
            "## Pairwise 区分度",
            "",
            "离散差值分为 `<0 / 0 / 1 / 2 / 3 / 4`；负值表示相同视频下模型给 suc 的分数反而低于 fail。",
            "",
            pair_table(groups, conditions, native=True),
        ]
    )
    if attention:
        lines.extend(
            [
                "",
                "## Attention steering 的既有配对效应",
                "",
                "下表直接摘录 `steering_metrics.json` 的 paired-vs-baseline 聚类统计；误差变化为负表示相对 baseline 改善。",
                "",
                native_effect_summary(read_json(experiment / "steering_metrics.json")),
            ]
        )
    (experiment / "exp_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grm_report(experiment: Path, rows: Sequence[dict], metadata: Mapping[str, dict], *, attention: bool, ranking_metadata: Path) -> None:
    source_file = "steering.jsonl" if attention else "records.shard-00.jsonl"
    enriched = enrich_rows(rows, metadata, native=False)
    groups = condition_groups(enriched)
    validate_existing_metrics(experiment, groups, native=False, attention=attention)
    conditions = ordered_conditions(groups, [])
    lines = report_header(experiment, source_file, enriched, attention=attention, ranking_metadata=ranking_metadata)
    lines.extend(
        [
            "- GRM 原始离散 MAE：按项目既有 `0.125/0.375/0.625/0.875` 四边界把 progress 映射为 label 1–5，再计算 MAE。",
            "- 连续 ordinal MAE：`mean(abs(1+4*progress-label))`，作为不量化进度的补充指标。",
            "- 二分类阈值：`progress<=low → 1`、`progress>=high → 5`，中间区间为 `uncertain`。",
            "- 五档预测分布：按 `[0,20%) / [20%,40%) / [40%,60%) / [60%,80%) / [80%,100%]` 映射为 label 1–5；它与二分类准确率阈值相互独立。",
            "",
            "## 总览：MAE 与两套阈值准确率",
            "",
        ]
    )
    summary_rows = []
    threshold_rows = []
    for condition in conditions:
        item = grm_summary(groups[condition])
        summary_rows.append(
            (
                condition,
                item["n"],
                item["suc"],
                item["fail"],
                number(item["mae"]),
                number(item["continuous_mae"]),
            )
        )
        for low, high in GRM_ENDPOINT_THRESHOLDS:
            key = f"{low:g}/{high:g}"
            threshold_rows.append(
                (
                    condition,
                    key,
                    pct(item[f"{key}:overall"]),
                    pct(item[f"{key}:suc"]),
                    pct(item[f"{key}:fail"]),
                )
            )
    lines.append(markdown_table(("condition", "n", "suc", "fail", "离散 MAE", "连续 ordinal MAE"), summary_rows))
    lines.extend(
        [
            "",
            "### 两套阈值准确率总览",
            "",
            markdown_table(("condition", "阈值 low/high", "总准确率", "suc 准确率", "fail 准确率"), threshold_rows),
        ]
    )
    lines.extend(["", "## 各 task 准确率"])
    for condition in conditions:
        lines.extend(["", f"<details><summary>{condition}</summary>"])
        for low, high in GRM_ENDPOINT_THRESHOLDS:
            lines.extend(["", f"阈值 `{low:g}/{high:g}`：", "", task_accuracy_grm(groups[condition], low, high)])
        lines.extend(["", "</details>"])
    lines.extend(
        [
            "",
            "## 阈值化预测分布",
            "",
            "此处分布固定按每 20% 一档统计；20%、40%、60%、80% 边界归入后一档，100% 归入 label 5。",
        ]
    )
    for condition in conditions:
        lines.extend(
            [
                "",
                f"<details><summary>{condition}</summary>",
                "",
                grm_distribution(groups[condition]),
                "",
                "</details>",
            ]
        )
    lines.extend(
        [
            "",
            "## Pairwise 区分度",
            "",
            "差值使用连续 progress，分箱为左闭右开；`0–10%` 包含 0，`≥90%` 包含 100%。负值表示 suc 进度低于 fail。",
            "",
            pair_table(groups, conditions, native=False),
        ]
    )
    if attention:
        lines.extend(
            [
                "",
                "## Attention steering 的既有因果 estimand",
                "",
                "以下直接摘录 `attention_metrics.json`；详细显著性校正与 gate 定义见原指标文件。",
                "",
                grm_effect_summary(read_json(experiment / "attention_metrics.json")),
            ]
        )
    (experiment / "exp_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_metadata(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    metadata = {str(row["id"]): row for row in rows}
    if len(metadata) != len(rows):
        raise ValueError(f"Duplicate ids in {path}")
    return metadata


def generate(experiments_root: Path, metadata_file: Path, ranking_metadata: Path) -> list[Path]:
    metadata = load_metadata(metadata_file)
    written = []
    experiment_dirs = sorted(path for path in experiments_root.iterdir() if path.is_dir())
    for experiment in experiment_dirs:
        name = experiment.name
        if name.startswith("baseline_"):
            result_files = sorted(experiment.glob("records.shard-*.jsonl"))
            attention = False
        elif name.startswith("attention_"):
            result_files = [experiment / "steering.jsonl"]
            attention = True
        else:
            continue
        result_files = [path for path in result_files if path.exists()]
        if not result_files:
            continue
        result_rows = [row for path in result_files for row in read_jsonl(path)]
        rows = latest_result_rows(result_rows, attention=attention)
        is_grm = "_grm_" in name
        if is_grm:
            write_grm_report(experiment, rows, metadata, attention=attention, ranking_metadata=ranking_metadata)
        else:
            write_native_report(experiment, rows, metadata, attention=attention, ranking_metadata=ranking_metadata)
        written.append(experiment / "exp_record.md")
    return written


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    workspace = repo_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=repo_root / "results/mydata_bench/experiments")
    parser.add_argument("--metadata", type=Path, default=workspace / "data/ljx_lfz_task/new/metadata.jsonl")
    parser.add_argument("--ranking-metadata", type=Path, default=workspace / "data/ljx_lfz_task/new/ranking_data.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = generate(args.experiments_root, args.metadata, args.ranking_metadata)
    if not written:
        raise SystemExit(f"No completed experiments found under {args.experiments_root}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
