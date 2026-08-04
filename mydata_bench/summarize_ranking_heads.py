#!/usr/bin/env python3
"""Validate and summarize exploratory ranking-head artifacts.

The report intentionally treats a head as the global ``(layer, head)`` pair.
Comparing only the local head index would conflate different transformer
layers.  The script is dependency-free so the report can be reproduced in a
plain Python environment without loading any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


N_VALUES = (5, 10, 20)
K_VALUES = (8, 32, 64)
N_PAIRS = ((5, 10), (10, 20), (5, 20))

BENCH_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = BENCH_ROOT / "outputs/my_dataset/ljx_lfz_cf_v1"
DEFAULT_MATRIX_ROOT = DEFAULT_RESULT_ROOT / "exploratory_matrix"
DEFAULT_OUTPUT = DEFAULT_RESULT_ROOT / "ranking_summary.md"
DEFAULT_SELECTION_MANIFEST = (
    BENCH_ROOT
    / "artifacts/my_dataset/ljx_lfz_cf_v1/ranking_cohort/selection_manifest.json"
)


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    model: str
    model_family: str
    directory: str
    input_order: str


VARIANTS = (
    Variant(
        key="rr_t2v",
        label="RR-T→V",
        model="RoboReward-8B",
        model_family="roboreward",
        directory="roboreward_8b_native_front_text_then_video",
        input_order="text_then_video",
    ),
    Variant(
        key="rr_v2t",
        label="RR-V→T",
        model="RoboReward-8B",
        model_family="roboreward",
        directory="roboreward_8b_model_card_native_front_video_then_text",
        input_order="video_then_text",
    ),
    Variant(
        key="qwen_v2t",
        label="Qwen-V→T",
        model="Qwen3-VL-8B",
        model_family="qwen",
        directory="qwen3vl_8b_video_then_text_attention8",
        input_order="video_then_text",
    ),
    Variant(
        key="grm",
        label="GRM",
        model="GRM-8B",
        model_family="grm",
        directory="grm_8b_multiview_endpoints",
        input_order="multiview_endpoints",
    ),
)


Head = tuple[int, int]


@dataclass(frozen=True)
class Artifact:
    path: Path
    value: Mapping[str, Any]
    ranking: tuple[Mapping[str, Any], ...]

    def heads(self, k: int) -> tuple[Head, ...]:
        return tuple(
            (int(row["layer"]), int(row["head"])) for row in self.ranking[:k]
        )


@dataclass(frozen=True)
class CohortManifest:
    path: Path
    value: Mapping[str, Any]
    ids_by_n: Mapping[int, tuple[str, ...]]


@dataclass(frozen=True)
class InvariantAudit:
    variant_label: str
    left_n: int
    right_n: int
    example_count: int
    equal_fields: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    left: tuple[Head, ...]
    right: tuple[Head, ...]
    intersection: frozenset[Head]
    union: frozenset[Head]
    exited: tuple[Head, ...]
    entered: tuple[Head, ...]
    movements: tuple[tuple[Head, int, int], ...]

    @property
    def jaccard(self) -> float:
        return len(self.intersection) / len(self.union)

    @property
    def retention(self) -> float:
        return len(self.intersection) / len(self.left)

    @property
    def mean_absolute_rank_change(self) -> float:
        if not self.movements:
            return math.nan
        return sum(abs(old - new) for _, old, new in self.movements) / len(
            self.movements
        )

    @property
    def relation(self) -> str:
        if self.left == self.right:
            return "集合、顺序均相同"
        if set(self.left) == set(self.right):
            return "集合相同、顺序不同"
        return "集合有变化"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def object_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_auto_grounding(value: Mapping[str, Any], *, path: Path, n: int) -> None:
    _require(value["human_reviewed"] is False, f"{path}: human_reviewed must be false")
    _require(value["claim_status"] == "exploratory", f"{path}: claim_status mismatch")
    composition = value["grounding_composition"]
    _require(isinstance(composition, dict), f"{path}: grounding_composition must be an object")
    strict_count = composition.get("strict_count")
    proxy_count = composition.get("proxy_count")
    total = composition.get("total")
    for field, count in (
        ("strict_count", strict_count),
        ("proxy_count", proxy_count),
        ("total", total),
    ):
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0,
            f"{path}: grounding_composition.{field} must be a non-negative integer",
        )
    _require(total == n, f"{path}: grounding composition total must equal N")
    _require(
        strict_count + proxy_count == total,
        f"{path}: grounding counts do not sum to total",
    )
    ratio = composition.get("proxy_ratio")
    _require(
        isinstance(ratio, (int, float))
        and not isinstance(ratio, bool)
        and math.isfinite(float(ratio)),
        f"{path}: invalid grounding proxy_ratio",
    )
    expected_ratio = proxy_count / total
    _require(
        math.isclose(float(ratio), expected_ratio, rel_tol=0.0, abs_tol=1e-12),
        f"{path}: grounding proxy_ratio disagrees with counts",
    )
    if proxy_count == 0:
        expected = ("auto_assumed_unreviewed", "strict")
    elif strict_count == 0:
        expected = ("auto_proxy_unreviewed", "proxy")
    else:
        expected = ("mixed", "mixed")
    actual = (value["grounding_status"], value["grounding_resolution"])
    _require(
        actual == expected,
        f"{path}: illegal aggregated auto grounding pair {actual}; expected {expected}",
    )


def load_artifact(path: Path, *, expected_n: int, variant: Variant) -> Artifact:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"缺少 ranking 文件：{path}") from exc
    _require(isinstance(value, dict), f"{path}: root must be an object")

    required = {
        "schema_version",
        "variant_id",
        "model_family",
        "claim_status",
        "grounding_status",
        "grounding_resolution",
        "grounding_composition",
        "human_reviewed",
        "run_fingerprint",
        "ranking_n",
        "sample_count",
        "sample_example_ids",
        "method",
        "ranking_score_kind",
        "skip_early_layers",
        "ranking",
        "fingerprint",
    }
    missing = sorted(required - set(value))
    _require(not missing, f"{path}: missing fields: {missing}")
    _require(
        value["variant_id"] == f"{variant.directory}_matrix",
        f"{path}: variant_id mismatch",
    )
    _require(
        value["schema_version"] == "my_dataset.exploratory_matrix.v1",
        f"{path}: schema_version mismatch",
    )
    _require(value["model_family"] == variant.model_family, f"{path}: model_family mismatch")
    _require(
        value["method"] == "terminal_last_prompt_excess_mass_sample_mean_skip8",
        f"{path}: ranking method mismatch",
    )
    _require(
        value["ranking_score_kind"] == "excess_mass",
        f"{path}: ranking_score_kind mismatch",
    )
    _require(value["skip_early_layers"] == 8, f"{path}: skip_early_layers must be 8")
    _require(value["ranking_n"] == expected_n, f"{path}: ranking_n mismatch")
    _require(value["sample_count"] == expected_n, f"{path}: sample_count mismatch")
    _validate_auto_grounding(value, path=path, n=expected_n)
    sample_ids = value["sample_example_ids"]
    _require(isinstance(sample_ids, list), f"{path}: sample_example_ids must be a list")
    _require(len(sample_ids) == expected_n, f"{path}: sample ID count mismatch")
    _require(len(set(sample_ids)) == expected_n, f"{path}: duplicate sample IDs")

    ranking = value["ranking"]
    _require(isinstance(ranking, list), f"{path}: ranking must be a list")
    _require(len(ranking) == max(K_VALUES), f"{path}: expected exactly 64 ranked heads")
    seen: set[Head] = set()
    previous_score = math.inf
    skip = value["skip_early_layers"]
    _require(isinstance(skip, int) and not isinstance(skip, bool), f"{path}: bad skip")
    for position, row in enumerate(ranking, 1):
        _require(isinstance(row, dict), f"{path}: ranking[{position}] must be an object")
        _require(row.get("rank") == position, f"{path}: non-contiguous rank at {position}")
        layer, head, score = row.get("layer"), row.get("head"), row.get("score")
        _require(
            isinstance(layer, int) and not isinstance(layer, bool),
            f"{path}: invalid layer at rank {position}",
        )
        _require(
            isinstance(head, int) and not isinstance(head, bool),
            f"{path}: invalid head at rank {position}",
        )
        _require(layer >= skip and head >= 0, f"{path}: invalid global head at rank {position}")
        _require(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score)),
            f"{path}: invalid score at rank {position}",
        )
        _require(float(score) <= previous_score, f"{path}: scores are not descending")
        previous_score = float(score)
        coordinate = (layer, head)
        _require(coordinate not in seen, f"{path}: duplicate head {coordinate}")
        seen.add(coordinate)

    fingerprint_view = dict(value)
    recorded_fingerprint = fingerprint_view.pop("fingerprint")
    _require(
        recorded_fingerprint == object_fingerprint(fingerprint_view),
        f"{path}: corrupt artifact fingerprint",
    )
    return Artifact(path=path, value=value, ranking=tuple(ranking))


def load_selection_manifest(path: Path) -> CohortManifest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"缺少 ranking cohort selection manifest：{path}") from exc
    _require(isinstance(value, dict), f"{path}: root must be an object")
    _require(
        value.get("schema_version") == "my_dataset.external_ranking_cohort.v1",
        f"{path}: unexpected schema_version",
    )
    _require(value.get("claim_status") == "exploratory", f"{path}: claim_status mismatch")
    _require(value.get("cohort_role") == "external_ranking", f"{path}: cohort_role mismatch")
    fingerprint_view = dict(value)
    recorded_fingerprint = fingerprint_view.pop("fingerprint", None)
    _require(
        recorded_fingerprint == object_fingerprint(fingerprint_view),
        f"{path}: corrupt selection manifest fingerprint",
    )

    selection = value.get("selection")
    _require(isinstance(selection, dict), f"{path}: selection must be an object")
    _require(selection.get("nested_sizes") == list(N_VALUES), f"{path}: nested_sizes mismatch")
    _require(selection.get("labels_used_for_selection") is False, f"{path}: selection used labels")
    ordered = selection.get("ordered_max20")
    _require(isinstance(ordered, list) and len(ordered) == 20, f"{path}: ordered_max20 mismatch")
    _require(
        all(isinstance(row, dict) for row in ordered),
        f"{path}: ordered_max20 rows must be objects",
    )
    _require(
        selection.get("fingerprint") == object_fingerprint(ordered),
        f"{path}: corrupt selection fingerprint",
    )
    _require(
        [row.get("ranking_order") for row in ordered] == list(range(1, 21)),
        f"{path}: ranking_order is not contiguous",
    )
    ordered_ids = tuple(str(row.get("example_id", "")) for row in ordered)
    _require(all(ordered_ids), f"{path}: empty example_id in ordered_max20")
    _require(len(set(ordered_ids)) == 20, f"{path}: duplicate example IDs in ordered_max20")

    cohorts = value.get("cohorts")
    _require(isinstance(cohorts, dict), f"{path}: cohorts must be an object")
    ids_by_n: dict[int, tuple[str, ...]] = {}
    for n in N_VALUES:
        cohort = cohorts.get(str(n))
        _require(isinstance(cohort, dict), f"{path}: missing cohort N={n}")
        cohort_ids = cohort.get("example_ids")
        _require(isinstance(cohort_ids, list), f"{path}: N={n} example_ids must be a list")
        actual_ids = tuple(str(item) for item in cohort_ids)
        _require(cohort.get("size") == n, f"{path}: N={n} size mismatch")
        _require(cohort.get("is_prefix_of_max20") is True, f"{path}: N={n} is not marked prefix")
        _require(actual_ids == ordered_ids[:n], f"{path}: N={n} IDs/order differ from ordered_max20")
        ids_by_n[n] = actual_ids

    model_outputs = value.get("model_outputs")
    _require(isinstance(model_outputs, dict), f"{path}: model_outputs must be an object")
    for model in ("roboreward", "qwen", "grm"):
        output = model_outputs.get(model)
        _require(isinstance(output, dict), f"{path}: missing model output {model}")
        _require(output.get("count") == 20, f"{path}: {model} output count mismatch")
        actual_ids = tuple(str(item) for item in output.get("ordered_example_ids", ()))
        _require(actual_ids == ordered_ids, f"{path}: {model} output IDs/order mismatch")
    return CohortManifest(path=path, value=value, ids_by_n=ids_by_n)


def load_all(
    matrix_root: Path, cohort_manifest: CohortManifest
) -> dict[str, dict[int, Artifact]]:
    artifacts: dict[str, dict[int, Artifact]] = {}
    for variant in VARIANTS:
        artifacts[variant.key] = {}
        for n in N_VALUES:
            path = matrix_root / variant.directory / "ranking" / f"rank_n{n:03d}.json"
            artifacts[variant.key][n] = load_artifact(
                path, expected_n=n, variant=variant
            )
    validate_collection(artifacts, cohort_manifest)
    return artifacts


def validate_collection(
    artifacts: Mapping[str, Mapping[int, Artifact]], cohort_manifest: CohortManifest
) -> None:
    for variant in VARIANTS:
        by_n = artifacts[variant.key]
        ids = {
            n: tuple(str(value) for value in by_n[n].value["sample_example_ids"])
            for n in N_VALUES
        }
        _require(ids[5] == ids[10][:5], f"{variant.label}: S5 is not a prefix of S10")
        _require(ids[10] == ids[20][:10], f"{variant.label}: S10 is not a prefix of S20")
        reference = by_n[5].value
        for n in N_VALUES[1:]:
            value = by_n[n].value
            for field in (
                "schema_version",
                "variant_id",
                "model_family",
                "claim_status",
                "human_reviewed",
                "method",
                "ranking_score_kind",
                "skip_early_layers",
                "run_fingerprint",
            ):
                _require(
                    value[field] == reference[field],
                    f"{variant.label}: {field} differs between N artifacts",
                )

    for n in N_VALUES:
        reference_ids = tuple(
            str(value)
            for value in artifacts[VARIANTS[0].key][n].value["sample_example_ids"]
        )
        _require(
            reference_ids == cohort_manifest.ids_by_n[n],
            f"N={n}: artifact sample IDs/order differ from frozen selection manifest",
        )
        for variant in VARIANTS[1:]:
            actual = tuple(
                str(value)
                for value in artifacts[variant.key][n].value["sample_example_ids"]
            )
            _require(
                actual == reference_ids,
                f"N={n}: {variant.label} does not use the same ordered cohort",
            )


def validate_equal_set_steering_invariants(
    matrix_root: Path,
    artifacts: Mapping[str, Mapping[int, Artifact]],
) -> tuple[InvariantAudit, ...]:
    specs = (
        (
            "RR-T→V",
            "rr_t2v",
            ((5, 10),),
            ("native_prediction", "progress", "raw_output"),
        ),
        (
            "GRM",
            "grm",
            ((5, 10), (5, 20), (10, 20)),
            ("signed_score", "raw_output"),
        ),
    )
    audits: list[InvariantAudit] = []
    variants_by_key = {variant.key: variant for variant in VARIANTS}
    for label, key, pairs, equal_fields in specs:
        variant = variants_by_key[key]
        path = matrix_root / variant.directory / "steering/records.jsonl"
        conditions = {
            f"candidate_target__rank_n{n:03d}__top_k008"
            for pair in pairs
            for n in pair
        }
        latest: dict[tuple[str, str], Mapping[str, Any]] = {}
        try:
            handle = path.open("r", encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"缺少 steering records：{path}") from exc
        with handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                _require(isinstance(row, dict), f"{path}:{number}: row must be an object")
                condition = str(row.get("condition", ""))
                if condition not in conditions:
                    continue
                example_id = str(row.get("example_id", ""))
                _require(example_id, f"{path}:{number}: missing example_id")
                latest[(example_id, condition)] = row

        for left_n, right_n in pairs:
            _require(
                set(artifacts[key][left_n].heads(8))
                == set(artifacts[key][right_n].heads(8)),
                f"{label} N={left_n}/{right_n}: Top-8 artifact head sets differ",
            )
            left_condition = f"candidate_target__rank_n{left_n:03d}__top_k008"
            right_condition = f"candidate_target__rank_n{right_n:03d}__top_k008"
            left_ids = {example_id for example_id, condition in latest if condition == left_condition}
            right_ids = {example_id for example_id, condition in latest if condition == right_condition}
            _require(left_ids == right_ids, f"{label} N={left_n}/{right_n}: record IDs differ")
            _require(len(left_ids) == 755, f"{label} N={left_n}/{right_n}: expected 755 records")
            for n, condition in ((left_n, left_condition), (right_n, right_condition)):
                expected_heads = set(artifacts[key][n].heads(8))
                for example_id in left_ids:
                    row = latest[(example_id, condition)]
                    _require(row.get("status") == "ok", f"{path}/{example_id}/{condition}: status is not ok")
                    _require(row.get("bias") == 6.0, f"{path}/{example_id}/{condition}: bias mismatch")
                    _require(row.get("top_k") == 8, f"{path}/{example_id}/{condition}: top_k mismatch")
                    _require(row.get("ranking_n") == n, f"{path}/{example_id}/{condition}: ranking_n mismatch")
                    assertion = row.get("hook_assertion")
                    _require(
                        isinstance(assertion, dict) and assertion.get("passed") is True,
                        f"{path}/{example_id}/{condition}: hook assertion did not pass",
                    )
                    head_rows = row.get("heads")
                    _require(isinstance(head_rows, list) and len(head_rows) == 8, f"{path}/{example_id}/{condition}: heads mismatch")
                    actual_heads = {
                        (int(head["layer"]), int(head["head"])) for head in head_rows
                    }
                    _require(actual_heads == expected_heads, f"{path}/{example_id}/{condition}: head set differs from ranking")
            for example_id in left_ids:
                left = latest[(example_id, left_condition)]
                right = latest[(example_id, right_condition)]
                for field in equal_fields:
                    _require(
                        left.get(field) == right.get(field),
                        f"{label}/{example_id}: {field} differs between N={left_n} and N={right_n}",
                    )
            audits.append(
                InvariantAudit(
                    variant_label=label,
                    left_n=left_n,
                    right_n=right_n,
                    example_count=len(left_ids),
                    equal_fields=equal_fields,
                )
            )
    return tuple(audits)


def compare(left: Sequence[Head], right: Sequence[Head]) -> Comparison:
    _require(len(left) == len(right), "Top-K comparisons require equal K")
    _require(len(set(left)) == len(left), "Left head sequence has duplicates")
    _require(len(set(right)) == len(right), "Right head sequence has duplicates")
    left_tuple, right_tuple = tuple(left), tuple(right)
    left_set, right_set = set(left_tuple), set(right_tuple)
    right_rank = {head: rank for rank, head in enumerate(right_tuple, 1)}
    movements = tuple(
        (head, old_rank, right_rank[head])
        for old_rank, head in enumerate(left_tuple, 1)
        if head in right_rank
    )
    return Comparison(
        left=left_tuple,
        right=right_tuple,
        intersection=frozenset(left_set & right_set),
        union=frozenset(left_set | right_set),
        exited=tuple(head for head in left_tuple if head not in right_set),
        entered=tuple(head for head in right_tuple if head not in left_set),
        movements=movements,
    )


def head_name(head: Head) -> str:
    return f"L{head[0]}H{head[1]}"


def heads_text(heads: Iterable[Head], *, code: bool = True) -> str:
    values = [head_name(head) for head in heads]
    if not values:
        return "无"
    joined = "、".join(values)
    return f"`{joined}`" if code else joined


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def decimal(value: float) -> str:
    return f"{value:.3f}"


def rank_change_text(comparison: Comparison) -> str:
    value = comparison.mean_absolute_rank_change
    return "—" if math.isnan(value) else f"{value:.2f}"


def source_link(output: Path, artifact: Artifact) -> str:
    try:
        relative = artifact.path.resolve().relative_to(output.parent.resolve())
        target = relative.as_posix()
    except ValueError:
        target = artifact.path.resolve().as_posix()
    return f"[{artifact.path.name}]({target})"


def _append_table(lines: list[str], headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")


def _top8_cell(artifact: Artifact) -> str:
    values = []
    for row in artifact.ranking[:8]:
        coordinate = head_name((int(row["layer"]), int(row["head"])))
        values.append(f'{int(row["rank"])}. `{coordinate}` ({float(row["score"]):.6f})')
    return "<br>".join(values)


def _movement_text(comparison: Comparison) -> str:
    return "、".join(
        f"`{head_name(head)}` {old}→{new} (Δ{new - old:+d})"
        for head, old, new in comparison.movements
    )


def _internal_summary_rows(
    artifacts: Mapping[str, Mapping[int, Artifact]], variant: Variant
) -> list[list[str]]:
    rows: list[list[str]] = []
    for k in K_VALUES:
        for left_n, right_n in N_PAIRS:
            comparison = compare(
                artifacts[variant.key][left_n].heads(k),
                artifacts[variant.key][right_n].heads(k),
            )
            rows.append(
                [
                    str(k),
                    f"N={left_n}→{right_n}",
                    str(len(comparison.intersection)),
                    str(len(comparison.union)),
                    decimal(comparison.jaccard),
                    pct(comparison.retention),
                    rank_change_text(comparison),
                    comparison.relation,
                ]
            )
    return rows


def _pair_variants() -> tuple[tuple[Variant, Variant], ...]:
    return tuple(
        (left, right)
        for index, left in enumerate(VARIANTS)
        for right in VARIANTS[index + 1 :]
    )


def _multiway(sets: Sequence[set[Head]]) -> tuple[int, int, float]:
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    return len(intersection), len(union), len(intersection) / len(union)


def build_report(
    artifacts: Mapping[str, Mapping[int, Artifact]],
    *,
    output: Path,
    matrix_root: Path,
    cohort_manifest: CohortManifest,
    invariant_audits: Sequence[InvariantAudit],
) -> str:
    try:
        manifest_target = (
            cohort_manifest.path.resolve().relative_to(output.parent.resolve()).as_posix()
        )
    except ValueError:
        manifest_target = cohort_manifest.path.resolve().as_posix()
    manifest_link = f"[selection_manifest.json]({manifest_target})"
    lines: list[str] = [
        "# LJX/LFZ exploratory ranking head 统计",
        "",
        "> 结论边界：本报告统计的是当前 **未经人工审核** 的 exploratory ranking。所有 12 个 artifact "
        "均为 `human_reviewed=false`；它们适合用于流程诊断和提出待审核候选，不应作为最终科学结论。",
        "",
        "## 口径",
        "",
        "- `N` 是计算 sample-mean attention mass 时使用的外部成功 rollout 数量；不是 head 数量。"
        "S5、S10、S20 是同一个冻结序列的嵌套前缀。",
        "- `K` 是从 ranking 中截取、用于 steering 的 head 数量。本轮配置实际包含 "
        "`K=8/32/64`，因此三种 K 都统计。",
        "- 四个 variant 的 `ranking_score_kind` 都是 `excess_mass`，聚合方法均为 "
        "`terminal_last_prompt_excess_mass_sample_mean_skip8`。score 只用于各自 variant 内排序；"
        "不同模型、输入顺序或视图协议之间的 score 绝对值不应直接比较。",
        "- head 始终用全局坐标 `(layer, head)` 比较，记作 `LxHy`。例如 `L19H23` 与 "
        "`L20H23` 是两个不同 head。",
        "- Jaccard = `|A∩B| / |A∪B|`；保留率 = `|A∩B| / K`。两边 K 相等时，"
        "保留率也就是 top-K overlap。平均绝对名次变化只在双方共同 head 上计算。",
        "- **集合相同不等于 ranking 相同**：集合相同但顺序变化仍可能改变每个 head 对应的 score/"
        "名次；当前 steering 对同一 top-K 内各 head 使用相同系数时，纯顺序变化本身不会改变被干预的 "
        "head 集合。",
        "- 跨不同模型的 `(layer, head)` 重合只表示架构坐标编号碰巧相同，不代表参数相同、同一个"
        "神经元或功能等价。两个 RoboReward 条目使用同一模型但 content order 不同，因而其坐标"
        "重合具有更直接的可比性。",
        "",
        "### Variant 对照",
        "",
    ]
    _append_table(
        lines,
        ("简称", "模型", "输入/视图协议", "目录含义"),
        (
            (
                f"`{variant.label}`",
                variant.model,
                f"`{variant.input_order}`",
                f"`{variant.directory}`",
            )
            for variant in VARIANTS
        ),
    )
    lines.extend(
        [
            "`RR-T→V` 与 `RR-V→T` 是 **RoboReward-8B 的两个输入顺序 variant**，不能当作两个"
            "独立模型样本。",
            "",
            "## Artifact 与 cohort 完整性",
            "",
            "脚本已执行以下硬校验：每个 JSON 的 schema 必需字段存在；`ranking_n=sample_count=N`；"
            "sample ID 唯一；每份 ranking 恰有 64 个唯一 `(layer, head)`；rank 连续、score 降序、"
            "layer 不小于 `skip_early_layers`；artifact fingerprint 可复算；12 份 artifact 均严格"
            "满足 `human_reviewed=false`、`claim_status=exploratory`，且 grounding status/resolution/"
            "composition 是合法的 auto strict、auto proxy 或二者 mixed 聚合；模型内 S5⊂S10⊂S20；"
            "四个 variant 在相同 N 使用完全相同且顺序相同的 sample IDs；这些 IDs/顺序又逐 N 与"
            "冻结 selection manifest 的 S5/S10/S20 精确相等。所有检查通过。",
            "",
            f"冻结 cohort 来源：{manifest_link}；manifest fingerprint："
            f"`{cohort_manifest.value['fingerprint']}`。",
            "",
        ]
    )
    integrity_rows = []
    for variant in VARIANTS:
        by_n = artifacts[variant.key]
        reference = by_n[5].value
        input_links = " / ".join(source_link(output, by_n[n]) for n in N_VALUES)
        grounding = "<br>".join(
            (
                f"N={n}: `{by_n[n].value['grounding_status']}/"
                f"{by_n[n].value['grounding_resolution']}`；"
                f"strict={by_n[n].value['grounding_composition']['strict_count']}，"
                f"proxy={by_n[n].value['grounding_composition']['proxy_count']}"
            )
            for n in N_VALUES
        )
        integrity_rows.append(
            (
                f"`{variant.label}`",
                reference["model_family"],
                reference["schema_version"],
                reference["method"],
                str(reference["skip_early_layers"]),
                reference["claim_status"],
                str(reference["human_reviewed"]).lower(),
                grounding,
                input_links,
            )
        )
    _append_table(
        lines,
        (
            "Variant",
            "model_family",
            "schema",
            "ranking method",
            "skip layers",
            "claim",
            "reviewed",
            "N=5/10/20 grounding",
            "N=5/10/20 输入",
        ),
        integrity_rows,
    )

    lines.extend(["## 一眼看懂：Top-8 随 N 的稳定性", ""])
    summary_rows = []
    for variant in VARIANTS:
        row = [f"`{variant.label}`"]
        for left_n, right_n in N_PAIRS:
            comp = compare(
                artifacts[variant.key][left_n].heads(8),
                artifacts[variant.key][right_n].heads(8),
            )
            row.append(
                f"{len(comp.intersection)}/8；J={comp.jaccard:.3f}；{comp.relation}"
            )
        summary_rows.append(row)
    _append_table(
        lines,
        ("Variant", "N=5→10", "N=10→20", "N=5→20"),
        summary_rows,
    )
    lines.append(
        "这里的 `x/8` 是共同 head 数。若写为“集合相同、顺序不同”，说明 steering 的 top-8 "
        "集合没有变，变化仅发生在 ranking 次序；若写为“集合有变化”，才有 head 进入或退出。"
    )
    lines.append("")
    lines.extend(
        [
            "### 相同 Top-8 集合的 steering 不变量复核",
            "",
            "当前 matrix 对入选 head 使用相同 `bias=6.0`，hook 按 layer/head 集合工作，不使用"
            "集合内部的 ranking 顺序。因此，当两个条件的 Top-8 集合相同而只改变顺序时，逐例输出"
            "理应完全一致。脚本直接读取 steering records，并对 755 个样本执行了这一硬校验：",
            "",
        ]
    )
    _append_table(
        lines,
        ("Variant", "N 对", "样本数", "逐例完全相等字段", "结果"),
        (
            (
                f"`{audit.variant_label}`",
                f"N={audit.left_n} ↔ N={audit.right_n}",
                str(audit.example_count),
                "、".join(f"`{field}`" for field in audit.equal_fields),
                "全部一致",
            )
            for audit in invariant_audits
        ),
    )
    lines.append(
        "结论：RR-T→V 的 N=5/N=10 Top-8 虽然顺序不同，但 755 条 `native_prediction`、"
        "`progress` 和 `raw_output` 全部一致；GRM 的 N=5/10/20 Top-8 集合相同，三个 "
        "pair 的 755 条 `signed_score` 与 `raw_output` 也全部一致。这与当前等权 hook "
        "实现相符，没有观察到由纯排序变化引入的 nondeterminism。"
    )
    lines.append("")

    for variant in VARIANTS:
        by_n = artifacts[variant.key]
        lines.extend(
            [
                f"## {variant.label}：{variant.model}",
                "",
                "### Top-8 排名与 score",
                "",
            ]
        )
        _append_table(
            lines,
            ("N", "按 rank 排序的 Top-8（括号内为 ranking score）"),
            ((str(n), _top8_cell(by_n[n])) for n in N_VALUES),
        )
        lines.extend(["### N 变化统计（K=8/32/64）", ""])
        _append_table(
            lines,
            (
                "K",
                "N 变化",
                "交集",
                "并集",
                "Jaccard",
                "保留率",
                "共同 head 平均绝对名次变化",
                "集合/顺序判定",
            ),
            _internal_summary_rows(artifacts, variant),
        )
        lines.extend(["### Top-8 的具体进入、退出与名次移动", ""])
        for left_n, right_n in N_PAIRS:
            comp = compare(by_n[left_n].heads(8), by_n[right_n].heads(8))
            lines.extend(
                [
                    f"#### N={left_n}→{right_n}",
                    "",
                    f"- 判定：{comp.relation}。",
                    f"- 退出 Top-8：{heads_text(comp.exited)}。",
                    f"- 进入 Top-8：{heads_text(comp.entered)}。",
                    f"- 共同 head 名次：{_movement_text(comp)}。",
                    "",
                ]
            )

    lines.extend(["## Variant/模型之间的 head 重合", ""])
    lines.append(
        "以下把四个运行 variant 两两比较。请特别注意：`RR-T→V ↔ RR-V→T` 是同一"
        "RoboReward-8B 的 content-order 敏感性；其余 pair 是不同模型/权重之间的坐标重合，"
        "只能作描述性统计。"
    )
    lines.append("")
    lines.extend(["### Top-8 两两重合（含共同 head）", ""])
    cross_top8_rows = []
    for n in N_VALUES:
        for left, right in _pair_variants():
            comp = compare(
                artifacts[left.key][n].heads(8), artifacts[right.key][n].heads(8)
            )
            common_in_left_order = [
                head for head in comp.left if head in comp.intersection
            ]
            cross_top8_rows.append(
                (
                    str(n),
                    f"`{left.label}` ↔ `{right.label}`",
                    str(len(comp.intersection)),
                    str(len(comp.union)),
                    decimal(comp.jaccard),
                    pct(comp.retention),
                    rank_change_text(comp),
                    comp.relation,
                    heads_text(common_in_left_order),
                )
            )
    _append_table(
        lines,
        (
            "N",
            "Pair",
            "交集",
            "并集",
            "Jaccard",
            "重合率",
            "共同 head 平均绝对名次变化",
            "集合/顺序判定",
            "共同 head（按左侧 rank）",
        ),
        cross_top8_rows,
    )

    lines.extend(["### Top-32 / Top-64 两两重合", ""])
    cross_large_rows = []
    for k in (32, 64):
        for n in N_VALUES:
            for left, right in _pair_variants():
                comp = compare(
                    artifacts[left.key][n].heads(k),
                    artifacts[right.key][n].heads(k),
                )
                cross_large_rows.append(
                    (
                        str(k),
                        str(n),
                        f"`{left.label}` ↔ `{right.label}`",
                        str(len(comp.intersection)),
                        str(len(comp.union)),
                        decimal(comp.jaccard),
                        pct(comp.retention),
                        rank_change_text(comp),
                        comp.relation,
                    )
                )
    _append_table(
        lines,
        (
            "K",
            "N",
            "Pair",
            "交集",
            "并集",
            "Jaccard",
            "重合率",
            "共同 head 平均绝对名次变化",
            "集合/顺序判定",
        ),
        cross_large_rows,
    )

    lines.extend(["### 四个 variant 的多方交集", ""])
    multi_rows = []
    for k in K_VALUES:
        for n in N_VALUES:
            sets = [set(artifacts[variant.key][n].heads(k)) for variant in VARIANTS]
            intersection_size, union_size, jaccard = _multiway(sets)
            common = set.intersection(*sets)
            common_in_reference_order = [
                head
                for head in artifacts[VARIANTS[0].key][n].heads(k)
                if head in common
            ]
            multi_rows.append(
                (
                    str(k),
                    str(n),
                    str(intersection_size),
                    str(union_size),
                    decimal(jaccard),
                    heads_text(common_in_reference_order),
                )
            )
    _append_table(
        lines,
        ("K", "N", "四方交集", "四方并集", "广义 Jaccard", "四方共同坐标"),
        multi_rows,
    )

    lines.extend(
        [
            "## 附录：完整 Top-64 坐标顺序",
            "",
            "每行是完整的 rank 1→64；Top-8 和 Top-32 分别就是该行前 8、前 32 个坐标。"
            "精确 score 见对应 `rank_n*.json`，正文已列出最需要审阅的 Top-8 score。",
            "",
        ]
    )
    for variant in VARIANTS:
        lines.extend([f"### {variant.label}", ""])
        for n in N_VALUES:
            ordered = "、".join(
                f"{index}.`{head_name(head)}`"
                for index, head in enumerate(artifacts[variant.key][n].heads(64), 1)
            )
            lines.extend([f"- N={n}：{ordered}", ""])

    try:
        matrix_display = matrix_root.resolve().relative_to(BENCH_ROOT.resolve()).as_posix()
        output_display = output.resolve().relative_to(BENCH_ROOT.resolve()).as_posix()
        selection_display = (
            cohort_manifest.path.resolve().relative_to(BENCH_ROOT.resolve()).as_posix()
        )
    except ValueError:
        matrix_display = matrix_root.resolve().as_posix()
        output_display = output.resolve().as_posix()
        selection_display = cohort_manifest.path.resolve().as_posix()
    lines.extend(
        [
            "## 复算方式",
            "",
            "在 `Robo-Dopamine/mydata_bench` 下运行：",
            "",
            "```bash",
            "python summarize_ranking_heads.py \\",
            f"  --matrix-root {matrix_display} \\",
            f"  --selection-manifest {selection_display} \\",
            f"  --output {output_display}",
            "```",
            "",
            "只做 schema、fingerprint、nested cohort 与跨 variant cohort 一致性校验，不写文档：",
            "",
            "```bash",
            f"python summarize_ranking_heads.py --matrix-root {matrix_display} "
            f"--selection-manifest {selection_display} --check-only",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验四个 exploratory ranking variant，并生成中文 head 重合度报告。"
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=DEFAULT_MATRIX_ROOT,
        help=f"exploratory_matrix 根目录（默认：{DEFAULT_MATRIX_ROOT}）",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
        help=f"冻结 ranking cohort manifest（默认：{DEFAULT_SELECTION_MANIFEST}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown 输出路径（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只读取并校验 12 个 artifact，不写 Markdown。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_root = args.matrix_root.resolve()
    selection_manifest_path = args.selection_manifest.resolve()
    output = args.output.resolve()
    cohort_manifest = load_selection_manifest(selection_manifest_path)
    artifacts = load_all(matrix_root, cohort_manifest)
    invariant_audits = validate_equal_set_steering_invariants(matrix_root, artifacts)
    if args.check_only:
        print(
            "校验通过：12 个 unreviewed exploratory ranking artifact；auto grounding "
            "composition 合法；S5/S10/S20 与冻结 selection manifest 精确一致；"
            "RR/GRM 相同 Top-8 集合的四组逐例 steering 不变量全部通过。"
        )
        return 0
    report = build_report(
        artifacts,
        output=output,
        matrix_root=matrix_root,
        cohort_manifest=cohort_manifest,
        invariant_audits=invariant_audits,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"已写入：{output}")
    print("校验通过：ranking、cohort provenance 与同集合 steering 不变量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
