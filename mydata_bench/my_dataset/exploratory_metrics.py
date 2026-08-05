"""Strict label-joined metrics for the exploratory ranking-size matrix.

Inference artifacts consumed here are deliberately label-free.  This module is
the only stage of the exploratory matrix pipeline that opens scoring labels.
It treats the baseline as shared by all nine N x K conditions and reports both
the intentionally contaminated all-data view and group-disjoint evaluation
views derived from the frozen ranking-cohort selection manifest.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from ..io import read_jsonl, sha256_file, write_json, write_jsonl
from ..protocol import progress, progress_to_reward
from .data import load_labels
from .grounding_contract import (
    AUTO_UNREVIEWED,
    HUMAN_REVIEWED,
    grounding_composition,
    grounding_contract,
    infer_grounding_mode,
    validate_grounding_row,
)


METRIC_CONTRACT = "my_dataset.exploratory_matrix_metrics.v2"
BASELINE_CONDITION = "baseline"
BASELINE_EQUIVALENCE_FIELDS = (
    "model_family",
    "group_id",
    "task_id",
    "task_family",
    "raw_output",
    "native_prediction",
    "signed_score",
    "progress",
)
RANKING_SIZES = (5, 10, 20)
TOP_K_VALUES = (8, 32, 64)
SCOPE_NAMES = (
    "all_including_rank_sources",
    "common_unseen_s20",
    "n_specific_unseen",
    "ranking_source_groups_only",
)
GROUNDING_STATUS_BY_RESOLUTION = {
    "strict": "auto_assumed_unreviewed",
    "proxy": "auto_proxy_unreviewed",
}
GROUNDING_STRATA = {
    "strict_grounding": "strict",
    "proxy_grounding": "proxy",
}

CONDITION_KINDS = (
    "candidate_target",
    "candidate_wrong_region",
    "low_rank_target",
)
_CONDITION_RE = re.compile(
    r"\A(?:(?P<kind>candidate_target|candidate_wrong_region|low_rank_target)__)?"
    r"rank_n(?P<n>\d{3})(?:__|_)top_k(?P<k>\d{3})\Z"
)
_LABEL_FIELDS = frozenset(
    {
        "protocol_reward",
        "instruction_video_match",
        "reward",
        "gold_reward",
        "label",
        "gold",
        "is_success",
        "success",
        "prediction_correct",
    }
)
_PUBLISHED_COUNTS_755 = {
    "common_unseen_s20": 682,
    "n_specific_unseen": {5: 738, 10: 721, 20: 682},
    "ranking_source_groups_only": {5: 17, 10: 34, 20: 73},
}


def _canonical_condition(
    ranking_n: int, top_k: int, kind: str = "candidate_target"
) -> str:
    if kind not in CONDITION_KINDS:
        raise ValueError(f"Unknown condition kind: {kind!r}")
    return f"{kind}__rank_n{ranking_n:03d}__top_k{top_k:03d}"


GRID_CONDITIONS = tuple(
    _canonical_condition(ranking_n, top_k, kind)
    for kind in CONDITION_KINDS
    for ranking_n in RANKING_SIZES
    for top_k in TOP_K_VALUES
)
REQUIRED_CONDITIONS = (BASELINE_CONDITION, *GRID_CONDITIONS)


def _parse_condition(
    value: Any,
) -> tuple[str, str, int | None, int | None]:
    condition = str(value)
    if condition == BASELINE_CONDITION:
        return condition, "baseline", None, None
    match = _CONDITION_RE.fullmatch(condition)
    if match is None:
        raise ValueError(f"Unexpected exploratory condition: {condition!r}")
    ranking_n = int(match.group("n"))
    top_k = int(match.group("k"))
    if ranking_n not in RANKING_SIZES or top_k not in TOP_K_VALUES:
        raise ValueError(
            f"Condition is outside the frozen N x K grid: {condition!r}"
        )
    kind = str(match.group("kind") or "candidate_target")
    return _canonical_condition(ranking_n, top_k, kind), kind, ranking_n, top_k


def _finite_float(value: Any, *, field: str, identity: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{identity}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{identity}: {field} must be finite")
    return result


def _head_coordinates(
    value: Any,
    *,
    top_k: int,
    identity: str,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or len(value) != top_k:
        actual = len(value) if isinstance(value, list) else type(value).__name__
        raise ValueError(
            f"{identity}: heads must be a list with exactly top_k={top_k} "
            f"entries; found {actual}"
        )
    coordinates: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{identity}: heads[{index}] must be a mapping")
        pair: list[int] = []
        for field in ("layer", "head"):
            raw = item.get(field)
            minimum = 8 if field == "layer" else 0
            error = (
                f"{identity}: heads[{index}].layer must be >=8"
                if field == "layer"
                else f"{identity}: heads[{index}].head must be a non-negative integer"
            )
            if isinstance(raw, bool):
                raise ValueError(error)
            try:
                numeric = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(error) from exc
            integer = int(numeric)
            if (
                not math.isfinite(numeric)
                or numeric != integer
                or integer < minimum
            ):
                raise ValueError(error)
            pair.append(integer)
        coordinates.append((pair[0], pair[1]))
    if len(coordinates) != len(set(coordinates)):
        duplicates = sorted(
            coordinate
            for coordinate, count in Counter(coordinates).items()
            if count > 1
        )
        raise ValueError(
            f"{identity}: heads coordinates must be unique; duplicates: {duplicates[:5]}"
        )
    return tuple(coordinates)


def _position_coordinates(
    value: Any, *, field: str, identity: str
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{identity}: {field} must be a non-empty list")
    result: list[int] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(
                f"{identity}: {field}[{index}] must be a non-negative integer"
            )
        result.append(int(raw))
    if result != sorted(set(result)):
        raise ValueError(f"{identity}: {field} must be sorted and unique")
    return tuple(result)


def _prediction(row: Mapping[str, Any], *, identity: str) -> int:
    native = row.get("native_prediction")
    if native is not None:
        if isinstance(native, bool):
            raise ValueError(f"{identity}: native_prediction cannot be boolean")
        numeric = _finite_float(native, field="native_prediction", identity=identity)
        value = int(numeric)
        if numeric != value or not 1 <= value <= 5:
            raise ValueError(
                f"{identity}: native_prediction must be an integer in [1, 5]"
            )
        return value
    if row.get("signed_score") is None:
        raise ValueError(
            f"{identity}: expected native_prediction or signed_score"
        )
    signed = _finite_float(row["signed_score"], field="signed_score", identity=identity)
    # Keep GRM's signed-score fallback identical to the canonical protocol
    # conversion used by the rest of mydata_bench.  Reimplementing this with
    # round(progress * 4) is subtly different at threshold boundaries because
    # Python uses bankers' rounding.
    return progress_to_reward(progress(signed))


def _effect_progress(
    row: Mapping[str, Any], *, prediction: int, identity: str
) -> float:
    if row.get("progress") is not None:
        value = _finite_float(row["progress"], field="progress", identity=identity)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{identity}: progress must lie in [0, 1]")
        return value
    if row.get("signed_score") is not None:
        signed = _finite_float(
            row["signed_score"], field="signed_score", identity=identity
        )
        return progress(signed)
    # Native ordinal protocols have a canonical linear progress encoding.
    return (prediction - 1) / 4


def _grounding_metadata(
    row: Mapping[str, Any],
    *,
    identity: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    validate_grounding_row(row, identity=identity, contract=contract)
    return {
        "grounding_mode": contract["mode"],
        "grounding_resolution": row["grounding_resolution"],
        "grounding_status": row["grounding_status"],
        "grounding_selection": row.get("grounding_selection"),
        "human_reviewed": row["human_reviewed"],
        "claim_status": row["claim_status"],
    }


def _load_latest_records(
    path: str | Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    latest_physical: dict[tuple[str, str], dict[str, Any]] = {}
    record_count = 0
    for number, row in enumerate(read_jsonl(path), 1):
        record_count += 1
        example_id = str(row.get("example_id", "")).strip()
        condition = str(row.get("condition", "")).strip()
        if not example_id or not condition:
            raise ValueError(
                f"Record {number} must contain non-empty example_id and condition"
            )
        latest_physical[(example_id, condition)] = row
    if not latest_physical:
        raise ValueError(f"Exploratory records are empty: {path}")

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    source_names: dict[tuple[str, str], str] = {}
    for (example_id, source_condition), row in latest_physical.items():
        condition, condition_kind, ranking_n, top_k = _parse_condition(
            source_condition
        )
        key = (example_id, condition)
        if key in latest:
            raise ValueError(
                f"{example_id}: multiple spellings resolve to condition {condition!r}: "
                f"{source_names[key]!r} and {source_condition!r}"
            )
        normalized = dict(row)
        normalized["source_condition"] = source_condition
        # Preserve the physical record's metadata before installing the
        # condition-derived canonical values.  These fields are validated
        # below; comparing row["ranking_n"] after overwriting it only compares
        # a parsed value with itself and cannot catch a corrupt record.
        normalized["source_ranking_n"] = row.get("ranking_n")
        normalized["source_top_k"] = row.get("top_k")
        normalized["condition"] = condition
        normalized["parsed_condition_kind"] = condition_kind
        normalized["ranking_n"] = ranking_n
        normalized["top_k"] = top_k
        latest[key] = normalized
        source_names[key] = source_condition
    return latest, record_count


def _validate_latest_records(
    latest: dict[tuple[str, str], dict[str, Any]],
    *,
    expected_count: int,
) -> tuple[list[str], str, dict[int, str], dict[str, Any]]:
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    example_ids = sorted({example_id for example_id, _condition in latest})
    if len(example_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} exploratory examples, found {len(example_ids)}"
        )

    required = set(REQUIRED_CONDITIONS)
    incomplete: dict[str, dict[str, list[str]]] = {}
    for example_id in example_ids:
        present = {
            condition for candidate_id, condition in latest if candidate_id == example_id
        }
        missing = sorted(required - present)
        unexpected = sorted(present - required)
        if missing or unexpected:
            incomplete[example_id] = {
                "missing": missing,
                "unexpected": unexpected,
            }
    if incomplete:
        preview = dict(list(sorted(incomplete.items()))[:5])
        raise ValueError(f"Incomplete exploratory matrix: {preview}")

    invalid: list[str] = []
    fingerprints: set[str] = set()
    ranking_fingerprint_values: dict[int, set[str]] = {
        ranking_n: set() for ranking_n in RANKING_SIZES
    }
    reference_metadata: dict[str, tuple[str, str, str]] = {}
    reference_grounding: dict[str, dict[str, Any]] = {}
    grounding_modes: set[str] = set()
    record_bindings: set[tuple[str, str, str]] = set()
    heads_by_condition: dict[str, tuple[tuple[int, int], ...]] = {}
    positions_by_example: dict[
        str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ] = {}
    visual_scope_bindings: set[tuple[str, str]] = set()
    for (example_id, condition), row in sorted(latest.items()):
        identity = f"{example_id}/{condition}"
        if row.get("status") != "ok":
            invalid.append(identity)
        leaked = sorted(_LABEL_FIELDS & row.keys())
        if leaked:
            raise ValueError(f"{identity}: label fields leaked into inference record: {leaked}")
        fingerprint = str(row.get("run_fingerprint", "")).strip()
        if not fingerprint:
            raise ValueError(f"{identity}: missing run_fingerprint")
        fingerprints.add(fingerprint)
        variant_id = str(row.get("variant_id", "")).strip()
        raw_reference_variant_id = row.get("reference_variant_id")
        reference_variant_id = (
            ""
            if raw_reference_variant_id is None
            else str(raw_reference_variant_id).strip()
        )
        model_family = str(row.get("model_family", "")).strip()
        if not variant_id or not model_family:
            raise ValueError(
                f"{identity}: variant_id and model_family are required"
            )
        record_bindings.add((variant_id, reference_variant_id, model_family))
        mode = infer_grounding_mode(row, identity=identity)
        grounding_modes.add(mode)
        contract = grounding_contract(mode)
        grounding = _grounding_metadata(
            row,
            identity=identity,
            contract=contract,
        )
        old_grounding = reference_grounding.setdefault(example_id, grounding)
        if old_grounding != grounding:
            raise ValueError(
                f"{example_id}: grounding metadata vary across conditions"
            )

        metadata = tuple(
            str(row.get(field, "")).strip()
            for field in ("group_id", "task_id", "task_family")
        )
        if any(not value for value in metadata):
            raise ValueError(
                f"{identity}: group_id, task_id, and task_family are required"
            )
        old = reference_metadata.setdefault(example_id, metadata)
        if old != metadata:
            raise ValueError(
                f"{example_id}: group/task metadata vary across conditions"
            )

        ranking_n = row["ranking_n"]
        top_k = row["top_k"]
        parsed_kind = str(row["parsed_condition_kind"])
        target_positions = _position_coordinates(
            row.get("target_positions"),
            field="target_positions",
            identity=identity,
        )
        wrong_positions = _position_coordinates(
            row.get("wrong_region_positions"),
            field="wrong_region_positions",
            identity=identity,
        )
        selected_positions = _position_coordinates(
            row.get("selected_positions"),
            field="selected_positions",
            identity=identity,
        )
        visual_positions = _position_coordinates(
            row.get("visual_positions"),
            field="visual_positions",
            identity=identity,
        )
        if (
            len(target_positions) != len(wrong_positions)
            or set(target_positions) & set(wrong_positions)
            or not set(target_positions) < set(visual_positions)
            or not set(wrong_positions) < set(visual_positions)
        ):
            raise ValueError(
                f"{identity}: target/wrong positions must be equal-size, "
                "disjoint proper subsets of the visual universe"
            )
        expected_selected = (
            wrong_positions
            if parsed_kind == "candidate_wrong_region"
            else target_positions
        )
        if selected_positions != expected_selected:
            raise ValueError(
                f"{identity}: selected_positions disagree with condition kind"
            )
        position_contract = (target_positions, wrong_positions, visual_positions)
        old_position_contract = positions_by_example.setdefault(
            example_id, position_contract
        )
        if old_position_contract != position_contract:
            raise ValueError(
                f"{example_id}: target/wrong/visual positions vary across conditions"
            )
        ranking_visual_scope = str(row.get("ranking_visual_scope", ""))
        intervention_visual_scope = str(
            row.get("intervention_visual_scope", "")
        )
        if (
            ranking_visual_scope not in {"target_slot_only", "all_visual"}
            or intervention_visual_scope
            not in {"target_slot_only", "all_visual"}
        ):
            raise ValueError(f"{identity}: visual scope contract is invalid")
        visual_scope_bindings.add(
            (ranking_visual_scope, intervention_visual_scope)
        )
        if condition == BASELINE_CONDITION:
            if row.get("condition_kind") not in (None, "baseline"):
                raise ValueError(f"{identity}: invalid baseline condition_kind")
            if row["source_ranking_n"] is not None or row["source_top_k"] is not None:
                raise ValueError(
                    f"{identity}: baseline ranking_n and top_k must both be null"
                )
        else:
            if row.get("condition_kind") != parsed_kind:
                raise ValueError(
                    f"{identity}: condition_kind differs from condition name"
                )
            original_n = row["source_ranking_n"]
            original_k = row["source_top_k"]
            try:
                original_n_numeric = float(original_n)
                original_k_numeric = float(original_k)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{identity}: ranking_n/top_k must be explicit integers"
                ) from exc
            if (
                isinstance(original_n, bool)
                or isinstance(original_k, bool)
                or not math.isfinite(original_n_numeric)
                or not math.isfinite(original_k_numeric)
                or original_n_numeric != int(original_n_numeric)
                or original_k_numeric != int(original_k_numeric)
                or int(original_n_numeric) != ranking_n
                or int(original_k_numeric) != top_k
            ):
                raise ValueError(
                    f"{identity}: record ranking_n/top_k "
                    f"{original_n!r}/{original_k!r} disagree with condition "
                    f"{ranking_n}/{top_k}"
                )

            coordinates = _head_coordinates(
                row.get("heads"),
                top_k=top_k,
                identity=identity,
            )
            expected_coordinates = heads_by_condition.setdefault(
                condition, coordinates
            )
            if coordinates != expected_coordinates:
                raise ValueError(
                    f"{identity}: heads list varies across examples for "
                    f"condition {condition}"
                )
            bias = _finite_float(row.get("bias"), field="bias", identity=identity)
            if bias != 6.0:
                raise ValueError(f"{identity}: bias must equal the frozen value 6.0")
            if row.get("scope") != "all":
                raise ValueError(f"{identity}: scope must equal the frozen value 'all'")
            hook_assertion = row.get("hook_assertion")
            if (
                not isinstance(hook_assertion, Mapping)
                or hook_assertion.get("passed") is not True
            ):
                raise ValueError(f"{identity}: hook_assertion.passed must be true")
            ranking_fingerprint = str(row.get("ranking_fingerprint", "")).strip()
            if not ranking_fingerprint:
                raise ValueError(f"{identity}: missing ranking_fingerprint")
            ranking_fingerprint_values[ranking_n].add(ranking_fingerprint)
        prediction = _prediction(row, identity=identity)
        _effect_progress(row, prediction=prediction, identity=identity)

    if invalid:
        raise ValueError(
            f"Invalid latest exploratory records ({len(invalid)}): {invalid[:10]}"
        )
    if len(record_bindings) != 1:
        raise ValueError(
            "Exploratory records contain inconsistent "
            "variant_id/reference_variant_id/model_family bindings: "
            f"{sorted(record_bindings)}"
        )
    if len(fingerprints) != 1:
        raise ValueError(
            f"Exploratory records contain inconsistent run_fingerprint values: "
            f"{sorted(fingerprints)}"
        )
    if len(grounding_modes) != 1:
        raise ValueError(
            "Exploratory records mix grounding contracts: "
            f"{sorted(grounding_modes)}"
        )
    if len(visual_scope_bindings) != 1:
        raise ValueError(
            "Exploratory records mix ranking/intervention visual scopes: "
            f"{sorted(visual_scope_bindings)}"
        )
    inconsistent_rankings = {
        ranking_n: sorted(values)
        for ranking_n, values in ranking_fingerprint_values.items()
        if len(values) != 1
    }
    if inconsistent_rankings:
        raise ValueError(
            "Candidate records contain missing or inconsistent ranking_fingerprint "
            f"values by N: {inconsistent_rankings}"
        )
    for ranking_n in RANKING_SIZES:
        maximal_heads = heads_by_condition[
            _canonical_condition(ranking_n, 64)
        ]
        for top_k in (8, 32):
            condition = _canonical_condition(ranking_n, top_k)
            if heads_by_condition[condition] != maximal_heads[:top_k]:
                raise ValueError(
                    f"Candidate heads for N={ranking_n}, K={top_k} must equal "
                    "the ordered K=64 prefix"
                )
        for top_k in TOP_K_VALUES:
            candidate = heads_by_condition[
                _canonical_condition(ranking_n, top_k, "candidate_target")
            ]
            wrong = heads_by_condition[
                _canonical_condition(
                    ranking_n, top_k, "candidate_wrong_region"
                )
            ]
            low = heads_by_condition[
                _canonical_condition(ranking_n, top_k, "low_rank_target")
            ]
            if wrong != candidate:
                raise ValueError(
                    f"N={ranking_n}, K={top_k}: wrong-region heads must "
                    "exactly equal candidate-target heads"
                )
            if set(low) & set(candidate):
                raise ValueError(
                    f"N={ranking_n}, K={top_k}: low-rank heads overlap candidates"
                )
            candidate_layers = Counter(layer for layer, _head in candidate)
            low_layers = Counter(layer for layer, _head in low)
            if low_layers != candidate_layers:
                raise ValueError(
                    f"N={ranking_n}, K={top_k}: low-rank heads are not "
                    "layer matched to candidates"
                )
    ranking_fingerprints = {
        ranking_n: next(iter(ranking_fingerprint_values[ranking_n]))
        for ranking_n in RANKING_SIZES
    }
    return (
        example_ids,
        next(iter(fingerprints)),
        ranking_fingerprints,
        grounding_contract(next(iter(grounding_modes))),
    )


def _cohort_value(cohorts: Mapping[str, Any], ranking_n: int) -> Mapping[str, Any]:
    aliases = (
        str(ranking_n),
        f"n{ranking_n:03d}",
        f"N{ranking_n:03d}",
        f"s{ranking_n}",
        f"S{ranking_n}",
    )
    for key in aliases:
        value = cohorts.get(key)
        if isinstance(value, Mapping):
            return value
    for value in cohorts.values():
        if not isinstance(value, Mapping):
            continue
        size = value.get("size", value.get("ranking_n", value.get("n")))
        try:
            matches = int(size) == ranking_n
        except (TypeError, ValueError):
            matches = False
        if matches:
            return value
    raise ValueError(f"selection_manifest.cohorts has no N={ranking_n} cohort")


def _string_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, Mapping):
            candidate = item.get("group_id", item.get("example_id", item.get("id")))
        else:
            candidate = item
        text = str(candidate or "").strip()
        if text:
            result.append(text)
    return result


def _cohort_group_ids(
    cohort: Mapping[str, Any],
    labels: list[dict[str, Any]],
    *,
    ranking_n: int,
) -> set[str]:
    for field in (
        "group_ids",
        "ranking_source_group_ids",
        "selected_group_ids",
        "groups",
    ):
        values = _string_values(cohort.get(field))
        if values:
            if len(values) != len(set(values)):
                raise ValueError(f"N={ranking_n} cohort contains duplicate group IDs")
            return set(values)

    group_by_example = {
        str(row["example_id"]): str(row["group_id"]) for row in labels
    }
    example_ids = _string_values(
        cohort.get("example_ids", cohort.get("selected_example_ids"))
    )
    if example_ids:
        missing = sorted(set(example_ids) - set(group_by_example))
        if missing:
            raise ValueError(
                f"N={ranking_n} cohort example IDs are absent from labels: {missing[:5]}"
            )
        return {group_by_example[value] for value in example_ids}

    group_by_source: dict[str, set[str]] = defaultdict(set)
    for row in labels:
        source_id = str(row.get("source_group_id", "")).strip()
        if source_id:
            group_by_source[source_id].add(str(row["group_id"]))
    source_ids = _string_values(cohort.get("source_record_ids"))
    if source_ids and group_by_source:
        missing = sorted(set(source_ids) - set(group_by_source))
        ambiguous = sorted(
            source_id
            for source_id in source_ids
            if len(group_by_source.get(source_id, ())) != 1
        )
        if missing or ambiguous:
            raise ValueError(
                f"N={ranking_n} cohort source IDs cannot be mapped to one group: "
                f"missing={missing[:5]}, ambiguous={ambiguous[:5]}"
            )
        return {
            next(iter(group_by_source[source_id])) for source_id in source_ids
        }
    raise ValueError(
        f"N={ranking_n} cohort must provide group_ids or mappable example/source IDs"
    )


def _load_selection_groups(
    path: str | Path,
    labels: list[dict[str, Any]],
) -> tuple[dict[int, set[str]], dict[str, Any]]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid selection manifest JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("selection_manifest must contain a JSON object")
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, Mapping):
        raise ValueError("selection_manifest.cohorts must be a mapping")
    groups = {
        ranking_n: _cohort_group_ids(
            _cohort_value(cohorts, ranking_n), labels, ranking_n=ranking_n
        )
        for ranking_n in RANKING_SIZES
    }
    if not groups[5] <= groups[10] or not groups[10] <= groups[20]:
        raise ValueError("selection_manifest S5/S10/S20 group cohorts are not nested")
    known_groups = {str(row["group_id"]) for row in labels}
    foreign = sorted(groups[20] - known_groups)
    if foreign:
        raise ValueError(
            f"selection_manifest contains groups absent from labels: {foreign[:5]}"
        )
    return groups, manifest


def _optional_example_count(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    candidate = value.get("example_count", value.get("n"))
    if candidate is None:
        return None
    try:
        return int(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid manifest example count: {candidate!r}") from exc


def _validate_manifest_counts(
    manifest: Mapping[str, Any],
    expected_counts: Mapping[str, Any],
    *,
    record_ids: set[str],
    contract: Mapping[str, Any],
    allow_stale_population_counts: bool = False,
) -> None:
    population = manifest.get("evaluation_population")
    if isinstance(population, Mapping) and _string_values(
        population.get("example_ids")
    ):
        declared_ids = set(_string_values(population.get("example_ids")))
        if declared_ids != record_ids:
            raise ValueError(
                "selection_manifest reviewed evaluation population differs "
                "from scored records: "
                f"missing={sorted(record_ids - declared_ids)[:5]}, "
                f"extra={sorted(declared_ids - record_ids)[:5]}"
            )
        declared_count = _optional_example_count(population)
        if declared_count != len(record_ids):
            raise ValueError(
                "selection_manifest evaluation_population count differs "
                "from scored records"
            )
    elif contract["mode"] == HUMAN_REVIEWED or allow_stale_population_counts:
        # A legacy selection manifest declares all=755.  It may still be used
        # to rescore old unreviewed records on reviewed IDs, but its stale
        # population counts must not veto the explicit evaluation filter.
        return
    evaluation = manifest.get("evaluation_cohorts")
    if not isinstance(evaluation, Mapping):
        return
    checks: list[tuple[str, int | None, int]] = [
        (
            "all",
            _optional_example_count(
                evaluation.get("all", evaluation.get("all_including_rank_sources"))
            ),
            int(expected_counts["all_including_rank_sources"]),
        ),
        (
            "common_unseen_s20",
            _optional_example_count(evaluation.get("common_unseen_s20")),
            int(expected_counts["common_unseen_s20"]),
        ),
    ]
    by_size = evaluation.get("by_ranking_size")
    if isinstance(by_size, Mapping):
        for ranking_n in RANKING_SIZES:
            value = None
            for key in (str(ranking_n), f"n{ranking_n:03d}", f"N{ranking_n:03d}"):
                if isinstance(by_size.get(key), Mapping):
                    value = by_size[key]
                    break
            if value is None:
                continue
            checks.extend(
                (
                    (
                        f"N={ranking_n} n_specific_unseen",
                        _optional_example_count(value.get("n_specific_unseen")),
                        int(expected_counts["n_specific_unseen"][ranking_n]),
                    ),
                    (
                        f"N={ranking_n} ranking_source_groups_only",
                        _optional_example_count(
                            value.get(
                                "ranking_source_only",
                                value.get("ranking_source_groups_only"),
                            )
                        ),
                        int(expected_counts["ranking_source_groups_only"][ranking_n]),
                    ),
                )
            )
    for name, declared, computed in checks:
        if declared is not None and declared != computed:
            raise ValueError(
                f"selection_manifest {name} example_count={declared}, computed={computed}"
            )


def _validate_labels_and_metadata(
    labels: list[dict[str, Any]],
    example_ids: list[str],
    latest: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    expected_count: int,
    contract: Mapping[str, Any],
    allow_label_subset: bool = False,
) -> dict[str, dict[str, Any]]:
    all_label_by_id = {str(row["example_id"]): row for row in labels}
    if len(all_label_by_id) != len(labels):
        raise ValueError("Scoring labels contain duplicate example_id values")
    record_ids = set(example_ids)
    label_ids = set(all_label_by_id)
    exact_label_contract = (
        contract["mode"] == AUTO_UNREVIEWED and not allow_label_subset
    )
    if exact_label_contract and len(labels) != expected_count:
        raise ValueError(f"Expected {expected_count} labels, found {len(labels)}")
    if exact_label_contract and record_ids != label_ids:
        raise ValueError(
            "Exploratory record and label IDs differ: "
            f"missing_labels={sorted(record_ids - label_ids)[:5]}, "
            f"labels_without_records={sorted(label_ids - record_ids)[:5]}"
        )
    if not record_ids <= label_ids:
        raise ValueError(
            "Scored records contain IDs absent from the full labels file: "
            f"{sorted(record_ids - label_ids)[:5]}"
        )
    label_by_id = {
        example_id: all_label_by_id[example_id] for example_id in example_ids
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_id in example_ids:
        label = label_by_id[example_id]
        baseline = latest[(example_id, BASELINE_CONDITION)]
        for field in ("group_id", "task_id", "task_family"):
            if label.get(field) is not None and str(label[field]) != str(baseline[field]):
                raise ValueError(
                    f"{example_id}: label/record {field} mismatch: "
                    f"{label[field]!r} != {baseline[field]!r}"
                )
        groups[str(baseline["group_id"])].append(label)
    malformed = {}
    for group_id, rows in groups.items():
        rewards = [int(row["protocol_reward"]) for row in rows]
        if rewards.count(5) != 1 or rewards.count(1) < 1 or set(rewards) - {1, 5}:
            malformed[group_id] = Counter(rewards)
    if malformed:
        preview = {
            group_id: dict(counts)
            for group_id, counts in list(sorted(malformed.items()))[:5]
        }
        raise ValueError(
            "Every source group must contain one reward=5 and at least one reward=1: "
            f"{preview}"
        )
    return label_by_id


def _mean_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return mean(materialized) if materialized else None


def _class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "exact_accuracy": _mean_or_none(float(row["candidate_exact"]) for row in rows),
        "baseline_exact_accuracy": _mean_or_none(
            float(row["baseline_exact"]) for row in rows
        ),
        "exact_delta": _mean_or_none(float(row["exact_delta"]) for row in rows),
        "mean_progress": _mean_or_none(float(row["candidate_progress"]) for row in rows),
        "baseline_mean_progress": _mean_or_none(
            float(row["baseline_progress"]) for row in rows
        ),
        "mean_progress_shift": _mean_or_none(
            float(row["progress_shift"]) for row in rows
        ),
    }


def _group_ranking(rows: list[dict[str, Any]], progress_field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    group_pairwise: list[float] = []
    strict_top1: list[float] = []
    pair_count = 0
    for group_id, values in sorted(groups.items()):
        successes = [row for row in values if int(row["protocol_reward"]) == 5]
        failures = [row for row in values if int(row["protocol_reward"]) == 1]
        if len(successes) != 1 or not failures:
            raise ValueError(
                f"Scope contains malformed source group {group_id}: "
                f"reward5={len(successes)}, reward1={len(failures)}"
            )
        success_value = float(successes[0][progress_field])
        wins = [success_value > float(row[progress_field]) for row in failures]
        pair_count += len(wins)
        group_pairwise.append(mean(float(value) for value in wins))
        strict_top1.append(float(all(wins)))
    return {
        "group_count": len(group_pairwise),
        "pair_count": pair_count,
        "group_macro_pairwise_accuracy": _mean_or_none(group_pairwise),
        "strict_top1_accuracy": _mean_or_none(strict_top1),
    }


def _effect_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if int(row["protocol_reward"]) == 1]
    successes = [row for row in rows if int(row["protocol_reward"]) == 5]
    return {
        "exact_delta": _mean_or_none(float(row["exact_delta"]) for row in rows),
        "fail_correction_rate": _mean_or_none(
            float(row["fail_correction"]) for row in failures
        ),
        "suc_harm_rate": _mean_or_none(float(row["suc_harm"]) for row in successes),
        "mean_progress_shift": _mean_or_none(
            float(row["progress_shift"]) for row in rows
        ),
    }


def _lightweight_subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if int(row["protocol_reward"]) == 5]
    failures = [row for row in rows if int(row["protocol_reward"]) == 1]
    overall = _class_summary(rows)
    success = _class_summary(successes)
    failure = _class_summary(failures)
    effects = _effect_summary(rows)
    return {
        "n": len(rows),
        "overall": overall,
        "suc_reward5": success,
        "fail_reward1": failure,
        "versus_baseline": effects,
        "exact_accuracy": overall["exact_accuracy"],
        "suc_exact_accuracy": success["exact_accuracy"],
        "fail_exact_accuracy": failure["exact_accuracy"],
        **effects,
    }


def _grounding_aggregate(
    rows: Iterable[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    values = list(rows)
    resolutions = {str(row["grounding_resolution"]) for row in values}
    statuses = {str(row["grounding_status"]) for row in values}
    composition = grounding_composition(values, contract=contract)
    if not values:
        raise AssertionError("Internal grounding composition is empty")
    return {
        "grounding_resolution": (
            next(iter(resolutions)) if len(resolutions) == 1 else "mixed"
        ),
        "grounding_status": (
            next(iter(statuses)) if len(statuses) == 1 else "mixed"
        ),
        "grounding_composition": composition,
    }


def _subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if int(row["protocol_reward"]) == 5]
    failures = [row for row in rows if int(row["protocol_reward"]) == 1]
    overall = _class_summary(rows)
    success = _class_summary(successes)
    failure = _class_summary(failures)
    effects = _effect_summary(rows)
    candidate_ranking = _group_ranking(rows, "candidate_progress")
    baseline_ranking = _group_ranking(rows, "baseline_progress")
    return {
        "n": len(rows),
        "overall": overall,
        "suc_reward5": success,
        "fail_reward1": failure,
        "group_ranking": candidate_ranking,
        "baseline_group_ranking": baseline_ranking,
        "versus_baseline": effects,
        # Compact aliases keep the machine-readable summary convenient while
        # the nested fields retain explicit denominators.
        "exact_accuracy": overall["exact_accuracy"],
        "suc_exact_accuracy": success["exact_accuracy"],
        "fail_exact_accuracy": failure["exact_accuracy"],
        **effects,
    }


def _baseline_class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "exact_accuracy": _mean_or_none(float(row["baseline_exact"]) for row in rows),
        "mean_progress": _mean_or_none(
            float(row["baseline_progress"]) for row in rows
        ),
    }


def _baseline_subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if int(row["protocol_reward"]) == 5]
    failures = [row for row in rows if int(row["protocol_reward"]) == 1]
    return {
        "n": len(rows),
        "overall": _baseline_class_summary(rows),
        "suc_reward5": _baseline_class_summary(successes),
        "fail_reward1": _baseline_class_summary(failures),
        "group_ranking": _group_ranking(rows, "baseline_progress"),
    }


def _shared_baseline_metrics(
    joined: list[dict[str, Any]], expected_counts: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize the one baseline prediction shared by the full 9-grid.

    ``joined`` repeats that prediction once for every condition.  Select one
    canonical K=8 row per relevant N so no baseline example is accidentally
    counted nine times in the report.
    """
    rows_by_n: dict[int, list[dict[str, Any]]] = {}
    for ranking_n in RANKING_SIZES:
        condition = _canonical_condition(ranking_n, TOP_K_VALUES[0])
        values = [row for row in joined if row["condition"] == condition]
        if len(values) != int(expected_counts["all_including_rank_sources"]):
            raise AssertionError(
                f"Internal shared-baseline count mismatch for N={ranking_n}"
            )
        rows_by_n[ranking_n] = values

    canonical = rows_by_n[RANKING_SIZES[0]]

    def selected(
        rows: list[dict[str, Any]], scope: str, expected: int
    ) -> dict[str, Any]:
        values = [row for row in rows if bool(row["scope_membership"][scope])]
        if len(values) != expected:
            raise AssertionError(
                f"Internal shared-baseline scope mismatch for {scope}: "
                f"expected={expected}, actual={len(values)}"
            )
        return _baseline_subset_summary(values)

    return {
        "shared_across_all_grid_conditions": True,
        "all_including_rank_sources": selected(
            canonical,
            "all_including_rank_sources",
            int(expected_counts["all_including_rank_sources"]),
        ),
        "common_unseen_s20": selected(
            canonical,
            "common_unseen_s20",
            int(expected_counts["common_unseen_s20"]),
        ),
        "n_specific_unseen": {
            ranking_n: selected(
                rows_by_n[ranking_n],
                "n_specific_unseen",
                int(expected_counts["n_specific_unseen"][ranking_n]),
            )
            for ranking_n in RANKING_SIZES
        },
        "ranking_source_groups_only": {
            ranking_n: selected(
                rows_by_n[ranking_n],
                "ranking_source_groups_only",
                int(expected_counts["ranking_source_groups_only"][ranking_n]),
            )
            for ranking_n in RANKING_SIZES
        },
    }


def _by_field(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: _subset_summary(values) for key, values in sorted(grouped.items())}


def _scope_summary(
    rows: list[dict[str, Any]], *, expected_count: int
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise AssertionError(
            f"Internal scope count mismatch: expected={expected_count}, actual={len(rows)}"
        )
    summary = _subset_summary(rows)
    summary.update(
        {
            "expected_count": expected_count,
            "by_task_id": _by_field(rows, "task_id"),
            "by_task_family": _by_field(rows, "task_family"),
        }
    )
    return summary


def _scope_expected_counts(
    label_by_id: Mapping[str, Mapping[str, Any]],
    groups_by_n: Mapping[int, set[str]],
    *,
    expected_count: int,
) -> dict[str, Any]:
    group_by_example = {
        example_id: str(row["group_id"]) for example_id, row in label_by_id.items()
    }
    source_counts = {
        ranking_n: sum(
            group_id in groups_by_n[ranking_n]
            for group_id in group_by_example.values()
        )
        for ranking_n in RANKING_SIZES
    }
    result = {
        "all_including_rank_sources": expected_count,
        "common_unseen_s20": expected_count - source_counts[20],
        "n_specific_unseen": {
            ranking_n: expected_count - source_counts[ranking_n]
            for ranking_n in RANKING_SIZES
        },
        "ranking_source_groups_only": source_counts,
    }
    if expected_count == 755:
        if result["common_unseen_s20"] != _PUBLISHED_COUNTS_755["common_unseen_s20"]:
            raise ValueError(
                "Frozen common_unseen_s20 count must be 682 for the 755-example dataset"
            )
        for scope in ("n_specific_unseen", "ranking_source_groups_only"):
            if result[scope] != _PUBLISHED_COUNTS_755[scope]:
                raise ValueError(
                    f"Frozen {scope} counts differ from the 755-example contract: "
                    f"{result[scope]}"
                )
    return result


def _join_conditions(
    latest: Mapping[tuple[str, str], Mapping[str, Any]],
    label_by_id: Mapping[str, Mapping[str, Any]],
    groups_by_n: Mapping[int, set[str]],
    *,
    run_fingerprint: str,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    s20 = groups_by_n[20]
    for condition in GRID_CONDITIONS:
        _canonical, condition_kind, ranking_n, top_k = _parse_condition(
            condition
        )
        assert ranking_n is not None and top_k is not None
        for example_id in sorted(label_by_id):
            label = label_by_id[example_id]
            baseline = latest[(example_id, BASELINE_CONDITION)]
            candidate = latest[(example_id, condition)]
            baseline_identity = f"{example_id}/{BASELINE_CONDITION}"
            candidate_identity = f"{example_id}/{condition}"
            grounding = _grounding_metadata(
                candidate,
                identity=candidate_identity,
                contract=contract,
            )
            baseline_prediction = _prediction(baseline, identity=baseline_identity)
            candidate_prediction = _prediction(candidate, identity=candidate_identity)
            baseline_progress = _effect_progress(
                baseline, prediction=baseline_prediction, identity=baseline_identity
            )
            candidate_progress = _effect_progress(
                candidate, prediction=candidate_prediction, identity=candidate_identity
            )
            reward = int(label["protocol_reward"])
            group_id = str(baseline["group_id"])
            baseline_exact = baseline_prediction == reward
            candidate_exact = candidate_prediction == reward
            rows.append(
                {
                    "schema_version": METRIC_CONTRACT,
                    "example_id": example_id,
                    "group_id": group_id,
                    "task_id": str(baseline["task_id"]),
                    "task_family": str(baseline["task_family"]),
                    "protocol_reward": reward,
                    "instruction_video_match": reward == 5,
                    "condition": condition,
                    "source_condition": str(candidate["source_condition"]),
                    "condition_kind": condition_kind,
                    "ranking_n": ranking_n,
                    "top_k": top_k,
                    "ranking_fingerprint": candidate.get("ranking_fingerprint"),
                    "run_fingerprint": run_fingerprint,
                    "baseline_prediction": baseline_prediction,
                    "candidate_prediction": candidate_prediction,
                    "prediction": candidate_prediction,
                    "baseline_progress": baseline_progress,
                    "candidate_progress": candidate_progress,
                    "progress": candidate_progress,
                    "baseline_exact": baseline_exact,
                    "candidate_exact": candidate_exact,
                    "correct": candidate_exact,
                    "exact_delta": int(candidate_exact) - int(baseline_exact),
                    "fail_correction": bool(
                        reward == 1
                        and baseline_prediction != 1
                        and candidate_prediction == 1
                    ),
                    "suc_harm": bool(
                        reward == 5
                        and baseline_prediction == 5
                        and candidate_prediction != 5
                    ),
                    "progress_shift": candidate_progress - baseline_progress,
                    "scope_membership": {
                        "all_including_rank_sources": True,
                        "common_unseen_s20": group_id not in s20,
                        "n_specific_unseen": group_id not in groups_by_n[ranking_n],
                        "ranking_source_groups_only": group_id
                        in groups_by_n[ranking_n],
                    },
                    **grounding,
                }
            )
    return rows


def _condition_metrics(
    joined: list[dict[str, Any]],
    expected_counts: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_condition[str(row["condition"])].append(row)
    output = []
    for condition in GRID_CONDITIONS:
        rows = by_condition[condition]
        _canonical, condition_kind, ranking_n, top_k = _parse_condition(
            condition
        )
        assert ranking_n is not None and top_k is not None
        scopes = {}
        for scope in SCOPE_NAMES:
            selected = [
                row for row in rows if bool(row["scope_membership"][scope])
            ]
            if scope in ("all_including_rank_sources", "common_unseen_s20"):
                expected = int(expected_counts[scope])
            else:
                expected = int(expected_counts[scope][ranking_n])
            scopes[scope] = _scope_summary(selected, expected_count=expected)
        grounding_strata = {}
        for resolution, status in contract["status_by_resolution"].items():
            stratum = (
                next(
                    (
                        name
                        for name, candidate_resolution in GROUNDING_STRATA.items()
                        if candidate_resolution == resolution
                    ),
                    f"{resolution}_grounding",
                )
            )
            selected = [
                row
                for row in rows
                if row["grounding_resolution"] == resolution
            ]
            grounding_strata[stratum] = {
                "grounding_resolution": resolution,
                "grounding_status": status,
                **_lightweight_subset_summary(selected),
            }
        output.append(
            {
                "schema_version": METRIC_CONTRACT,
                "condition": condition,
                "condition_kind": condition_kind,
                "grid_condition": f"rank_n{ranking_n:03d}_top_k{top_k:03d}",
                "ranking_n": ranking_n,
                "top_k": top_k,
                "scopes": scopes,
                "grounding_strata": grounding_strata,
                **_grounding_aggregate(rows, contract=contract),
            }
        )
    return output


def _paired_control_summary(
    target_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    *,
    control_kind: str,
) -> dict[str, Any]:
    target_by_id = {str(row["example_id"]): row for row in target_rows}
    control_by_id = {str(row["example_id"]): row for row in control_rows}
    if target_by_id.keys() != control_by_id.keys():
        raise AssertionError(f"Internal {control_kind} paired IDs differ")
    progress_gaps: list[float] = []
    exact_gaps: list[float] = []
    lower: list[float] = []
    for example_id in sorted(target_by_id):
        target = target_by_id[example_id]
        control = control_by_id[example_id]
        if (
            target["group_id"] != control["group_id"]
            or target["protocol_reward"] != control["protocol_reward"]
            or target["baseline_progress"] != control["baseline_progress"]
        ):
            raise AssertionError(
                f"Internal {control_kind} pair metadata differ for {example_id}"
            )
        gap = float(target["candidate_progress"]) - float(
            control["candidate_progress"]
        )
        progress_gaps.append(gap)
        exact_gaps.append(
            float(target["candidate_exact"]) - float(control["candidate_exact"])
        )
        lower.append(float(gap < 0))
    return {
        "control_kind": control_kind,
        "n": len(progress_gaps),
        "mean_target_minus_control_progress": _mean_or_none(progress_gaps),
        "mean_target_effect_minus_control_effect": _mean_or_none(
            progress_gaps
        ),
        "target_lower_progress_fraction": _mean_or_none(lower),
        "mean_target_minus_control_exact": _mean_or_none(exact_gaps),
    }


def _specificity_metrics(
    joined: list[dict[str, Any]], expected_counts: Mapping[str, Any]
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_condition[str(row["condition"])].append(row)
    output: dict[str, Any] = {}
    for ranking_n in RANKING_SIZES:
        for top_k in TOP_K_VALUES:
            target_name = _canonical_condition(
                ranking_n, top_k, "candidate_target"
            )
            wrong_name = _canonical_condition(
                ranking_n, top_k, "candidate_wrong_region"
            )
            low_name = _canonical_condition(
                ranking_n, top_k, "low_rank_target"
            )
            target_all = by_condition[target_name]
            wrong_all = by_condition[wrong_name]
            low_all = by_condition[low_name]
            scopes: dict[str, Any] = {}
            for scope in SCOPE_NAMES:
                target = [
                    row
                    for row in target_all
                    if bool(row["scope_membership"][scope])
                ]
                target_ids = {str(row["example_id"]) for row in target}
                wrong = [
                    row
                    for row in wrong_all
                    if str(row["example_id"]) in target_ids
                ]
                low = [
                    row
                    for row in low_all
                    if str(row["example_id"]) in target_ids
                ]
                expected = (
                    int(expected_counts[scope])
                    if scope in {
                        "all_including_rank_sources",
                        "common_unseen_s20",
                    }
                    else int(expected_counts[scope][ranking_n])
                )
                if len(target) != expected:
                    raise AssertionError(
                        f"Internal specificity count mismatch for {target_name}/{scope}"
                    )
                scopes[scope] = {
                    "expected_count": expected,
                    "spatial_specificity": _paired_control_summary(
                        target,
                        wrong,
                        control_kind="candidate_wrong_region",
                    ),
                    "head_specificity": _paired_control_summary(
                        target,
                        low,
                        control_kind="layer_matched_low_rank_target",
                    ),
                }
            output[target_name] = {
                "ranking_n": ranking_n,
                "top_k": top_k,
                "scopes": scopes,
            }
    return output


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return str(value)


def _markdown_table(
    condition_rows: list[dict[str, Any]], scope: str
) -> list[str]:
    lines = [
        "| Kind | N | K | n | Overall exact | Suc reward5 exact | Fail reward1 exact | Exact delta | Fail correction | Suc harm | Mean progress shift | Pairwise | Strict top1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in condition_rows:
        metric = condition["scopes"][scope]
        lines.append(
            "| {kind} | {n_rank} | {top_k} | {n} | {overall} | {suc} | {fail} | "
            "{delta} | {correction} | {harm} | {shift} | {pairwise} | {top1} |".format(
                kind=condition["condition_kind"],
                n_rank=condition["ranking_n"],
                top_k=condition["top_k"],
                n=metric["n"],
                overall=_format_metric(metric["overall"]["exact_accuracy"]),
                suc=_format_metric(metric["suc_reward5"]["exact_accuracy"]),
                fail=_format_metric(metric["fail_reward1"]["exact_accuracy"]),
                delta=_format_metric(metric["versus_baseline"]["exact_delta"]),
                correction=_format_metric(
                    metric["versus_baseline"]["fail_correction_rate"]
                ),
                harm=_format_metric(metric["versus_baseline"]["suc_harm_rate"]),
                shift=_format_metric(
                    metric["versus_baseline"]["mean_progress_shift"]
                ),
                pairwise=_format_metric(
                    metric["group_ranking"]["group_macro_pairwise_accuracy"]
                ),
                top1=_format_metric(
                    metric["group_ranking"]["strict_top1_accuracy"]
                ),
            )
        )
    return lines


def _per_task_table(
    condition_rows: list[dict[str, Any]], field: str
) -> list[str]:
    lines = [
        f"| {field} | Kind | N | K | n | Overall exact | Suc exact | Fail exact | Exact delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in condition_rows:
        by_task = condition["scopes"]["common_unseen_s20"][f"by_{field}"]
        for task, metric in sorted(by_task.items()):
            lines.append(
                "| {task} | {kind} | {n_rank} | {top_k} | {n} | {overall} | {suc} | "
                "{fail} | {delta} |".format(
                    task=task.replace("|", "\\|"),
                    kind=condition["condition_kind"],
                    n_rank=condition["ranking_n"],
                    top_k=condition["top_k"],
                    n=metric["n"],
                    overall=_format_metric(metric["overall"]["exact_accuracy"]),
                    suc=_format_metric(metric["suc_reward5"]["exact_accuracy"]),
                    fail=_format_metric(metric["fail_reward1"]["exact_accuracy"]),
                    delta=_format_metric(metric["versus_baseline"]["exact_delta"]),
                )
            )
    return lines


def _specificity_markdown_table(
    specificity: Mapping[str, Any], scope: str
) -> list[str]:
    lines = [
        "| N | K | n | Target−wrong progress | Target<wrong | Target−low-rank progress | Target<low-rank |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in specificity.values():
        metrics = value["scopes"][scope]
        spatial = metrics["spatial_specificity"]
        head = metrics["head_specificity"]
        lines.append(
            "| {n_rank} | {top_k} | {n} | {spatial_gap} | {spatial_lower} | "
            "{head_gap} | {head_lower} |".format(
                n_rank=value["ranking_n"],
                top_k=value["top_k"],
                n=spatial["n"],
                spatial_gap=_format_metric(
                    spatial["mean_target_minus_control_progress"]
                ),
                spatial_lower=_format_metric(
                    spatial["target_lower_progress_fraction"]
                ),
                head_gap=_format_metric(
                    head["mean_target_minus_control_progress"]
                ),
                head_lower=_format_metric(
                    head["target_lower_progress_fraction"]
                ),
            )
        )
    return lines


def _baseline_markdown_table(shared_baseline: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Scope | N | n | Overall exact | Suc reward5 exact | Fail reward1 exact | Pairwise | Strict top1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def append(scope: str, ranking_n: str, metric: Mapping[str, Any]) -> None:
        lines.append(
            "| {scope} | {ranking_n} | {n} | {overall} | {suc} | {fail} | "
            "{pairwise} | {top1} |".format(
                scope=scope,
                ranking_n=ranking_n,
                n=metric["n"],
                overall=_format_metric(metric["overall"]["exact_accuracy"]),
                suc=_format_metric(metric["suc_reward5"]["exact_accuracy"]),
                fail=_format_metric(metric["fail_reward1"]["exact_accuracy"]),
                pairwise=_format_metric(
                    metric["group_ranking"]["group_macro_pairwise_accuracy"]
                ),
                top1=_format_metric(
                    metric["group_ranking"]["strict_top1_accuracy"]
                ),
            )
        )

    append(
        "all_including_rank_sources",
        "—",
        shared_baseline["all_including_rank_sources"],
    )
    append("common_unseen_s20", "—", shared_baseline["common_unseen_s20"])
    for scope in ("n_specific_unseen", "ranking_source_groups_only"):
        for ranking_n in RANKING_SIZES:
            append(scope, str(ranking_n), shared_baseline[scope][ranking_n])
    return lines


def _render_markdown(
    condition_rows: list[dict[str, Any]],
    expected_counts: Mapping[str, Any],
    shared_baseline: Mapping[str, Any],
    specificity: Mapping[str, Any],
    grounding_aggregate: Mapping[str, Any],
    *,
    run_fingerprint: str,
    contract: Mapping[str, Any],
) -> str:
    composition = grounding_aggregate["grounding_composition"]
    if contract["mode"] == AUTO_UNREVIEWED:
        provenance_lines = [
            "- 本报告的 claim status 为 `exploratory`；strict 行来自 "
            "`assumed_valid` 自动选择（`auto_assumed_unreviewed`），"
            f"grounding 汇总为 `{grounding_aggregate['grounding_resolution']}` / "
            f"`{grounding_aggregate['grounding_status']}`，且 "
            "`human_reviewed=false`，不是人工复核后的正式结论。",
            "",
            "## Grounding resolution 构成",
            "",
            f"- strict grounding：{composition['strict_count']}/{composition['total']}。",
            f"- proxy grounding（`auto_proxy_unreviewed`）："
            f"{composition['proxy_count']}/{composition['total']}（"
            f"{composition['proxy_ratio']:.2%}）。这些 proxy 行使用 fallback "
            "grounding（回退定位）；proxy 分层仅作敏感性披露，不作为 "
            "all-data primary result（全数据主结果）。",
        ]
    else:
        provenance_lines = [
            "- 本报告是 `human-reviewed exploratory robustness rerun`；"
            "它在未审核结果已经被观察后开展，不能标为 confirmatory/formal。",
            "- 所有推理行都满足 "
            "`human_audited / audited_eligible / human_reviewed=true`；"
            "评估仅使用人工审核后仍完整的 counterfactual groups。",
            "",
            "## Grounding resolution 构成",
            "",
            f"- human-audited grounding："
            f"{composition['human_audited_count']}/{composition['total']}（"
            f"{composition['human_audited_ratio']:.2%}）。",
        ]
    lines = [
        "# 探索性 N×K attention steering 结果",
        "",
        "## 结论边界与审计状态",
        "",
        *provenance_lines,
        "- 排名源组同时出现在部分评估 scope，存在明确的 `ranking/eval overlap`。",
        "- `all_including_rank_sources` 包含用于 head ranking 的同一批 source groups，因此是 in-sample contaminated，只能作探索性描述。",
        "- `common_unseen_s20` 排除全部 S20 ranking source groups，是 N=5/10/20 之间的 cross-N main comparison（主比较口径）。",
        "- 每个 N×K 均包含同 heads 的 equal-size/disjoint wrong-region control，以及同层数分布且不重叠的 low-rank-head target control。",
        "- controls 使 spatial/head specificity 可被估计；是否支持因果特异性仍取决于 paired effect CI，而不是仅看点估计。",
        "- 标签仅在本评分阶段 join；推理及 head ranking 阶段不读取评分标签。",
        f"- run fingerprint：`{run_fingerprint}`。",
        "",
        "## Scope 样本数",
        "",
        f"- all_including_rank_sources：{expected_counts['all_including_rank_sources']}",
        f"- common_unseen_s20：{expected_counts['common_unseen_s20']}",
        "- n_specific_unseen："
        + ", ".join(
            f"N={ranking_n}: {expected_counts['n_specific_unseen'][ranking_n]}"
            for ranking_n in RANKING_SIZES
        ),
        "- ranking_source_groups_only："
        + ", ".join(
            f"N={ranking_n}: {expected_counts['ranking_source_groups_only'][ranking_n]}"
            for ranking_n in RANKING_SIZES
        ),
        "",
        "## 共享 baseline 汇总（overall / suc / fail）",
        "",
        "以下 baseline 每个样本只统计一次；同一 baseline 被 27 个干预条件共享。",
        "",
        *_baseline_markdown_table(shared_baseline),
        "",
        "## 27-condition 主汇总：common_unseen_s20（overall / suc / fail）",
        "",
        *_markdown_table(condition_rows, "common_unseen_s20"),
        "",
        "## Paired causal specificity：common_unseen_s20",
        "",
        *_specificity_markdown_table(specificity, "common_unseen_s20"),
        "",
        "## 27-condition 敏感性汇总：all_including_rank_sources",
        "",
        *_markdown_table(condition_rows, "all_including_rank_sources"),
        "",
        "## 27-condition 敏感性汇总：n_specific_unseen",
        "",
        *_markdown_table(condition_rows, "n_specific_unseen"),
        "",
        "## 27-condition 诊断汇总：ranking_source_groups_only",
        "",
        *_markdown_table(condition_rows, "ranking_source_groups_only"),
        "",
        "## common_unseen_s20 按 task_id",
        "",
        *_per_task_table(condition_rows, "task_id"),
        "",
        "## common_unseen_s20 按 task_family",
        "",
        *_per_task_table(condition_rows, "task_family"),
        "",
    ]
    return "\n".join(lines)


def _evaluation_filter(path: str | Path) -> tuple[set[str], dict[str, Any]]:
    source = Path(path).resolve()
    rows = list(read_jsonl(source))
    ids = [str(row.get("example_id", "")).strip() for row in rows]
    if any(not value for value in ids):
        raise ValueError("Evaluation manifest contains an empty example_id")
    if len(ids) != len(set(ids)):
        duplicates = sorted(
            value for value, count in Counter(ids).items() if count > 1
        )
        raise ValueError(
            f"Evaluation manifest contains duplicate IDs: {duplicates[:5]}"
        )
    return set(ids), {
        "path": str(source),
        "sha256": sha256_file(source),
        "example_count": len(ids),
    }


def _filter_latest_records(
    latest: Mapping[tuple[str, str], dict[str, Any]],
    evaluation_ids: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    available = {example_id for example_id, _condition in latest}
    missing = sorted(evaluation_ids - available)
    if missing:
        raise ValueError(
            f"Evaluation manifest IDs are absent from records: {missing[:5]}"
        )
    return {
        key: row
        for key, row in latest.items()
        if key[0] in evaluation_ids
    }


def _baseline_reference_equivalence(
    latest: Mapping[tuple[str, str], Mapping[str, Any]],
    reference_records_path: str | Path,
    example_ids: Iterable[str],
) -> dict[str, Any]:
    materialized_ids = sorted(set(example_ids))
    reference, _physical_count = _load_latest_records(reference_records_path)
    mismatches = []
    reference_fingerprints = set()
    variant_bindings: set[tuple[str, str]] = set()
    for example_id in materialized_ids:
        key = (example_id, BASELINE_CONDITION)
        current = latest[key]
        current_variant_id = str(current.get("variant_id", "")).strip()
        reference_variant_id = str(
            current.get("reference_variant_id", "")
        ).strip()
        if not current_variant_id:
            mismatches.append(
                {
                    "example_id": example_id,
                    "reason": "missing_current_variant_id",
                }
            )
        if not reference_variant_id:
            mismatches.append(
                {
                    "example_id": example_id,
                    "reason": "missing_reference_variant_id",
                }
            )
        if current_variant_id and reference_variant_id:
            variant_bindings.add((current_variant_id, reference_variant_id))
        old = reference.get(key)
        if old is None or old.get("status") != "ok":
            mismatches.append(
                {"example_id": example_id, "reason": "missing_valid_reference"}
            )
            continue
        old_variant_id = str(old.get("variant_id", "")).strip()
        if reference_variant_id and old_variant_id != reference_variant_id:
            mismatches.append(
                {
                    "example_id": example_id,
                    "reason": "wrong_reference_variant",
                    "current_variant_id": current_variant_id,
                    "declared_reference_variant_id": reference_variant_id,
                    "actual_reference_variant_id": old_variant_id,
                }
            )
        old_fingerprint = str(old.get("run_fingerprint", "")).strip()
        if not old_fingerprint:
            mismatches.append(
                {"example_id": example_id, "reason": "missing_reference_run_fingerprint"}
            )
        else:
            reference_fingerprints.add(old_fingerprint)
        current_prediction = _prediction(
            current, identity=f"{example_id}/current_baseline"
        )
        reference_prediction = _prediction(
            old, identity=f"{example_id}/reference_baseline"
        )
        current_progress = _effect_progress(
            current,
            prediction=current_prediction,
            identity=f"{example_id}/current_baseline",
        )
        reference_progress = _effect_progress(
            old,
            prediction=reference_prediction,
            identity=f"{example_id}/reference_baseline",
        )
        output_field_mismatches = {
            field: {
                "current": current.get(field),
                "reference": old.get(field),
            }
            for field in BASELINE_EQUIVALENCE_FIELDS
            if current.get(field) != old.get(field)
        }
        if (
            current_prediction != reference_prediction
            or not math.isclose(
                current_progress,
                reference_progress,
                rel_tol=0,
                abs_tol=0,
            )
            or output_field_mismatches
        ):
            mismatches.append(
                {
                    "example_id": example_id,
                    "current_prediction": current_prediction,
                    "reference_prediction": reference_prediction,
                    "current_progress": current_progress,
                    "reference_progress": reference_progress,
                    "output_field_mismatches": output_field_mismatches,
                }
            )
    if len(variant_bindings) != 1:
        mismatches.append(
            {
                "reason": "non_unique_variant_binding",
                "variant_bindings": [
                    {
                        "current_variant_id": current_variant_id,
                        "reference_variant_id": reference_variant_id,
                    }
                    for current_variant_id, reference_variant_id in sorted(
                        variant_bindings
                    )
                ],
            }
        )
    if len(reference_fingerprints) != 1:
        mismatches.append(
            {
                "reason": "non_unique_reference_run_fingerprint",
                "reference_run_fingerprints": sorted(reference_fingerprints),
            }
        )
    if mismatches:
        raise ValueError(
            "Reviewed/current shared baseline differs from the unreviewed "
            f"reference on {len(mismatches)} examples: {mismatches[:5]}"
        )
    current_variant_id, reference_variant_id = next(iter(variant_bindings))
    source = Path(reference_records_path).resolve()
    return {
        "passed": True,
        "example_count": len(materialized_ids),
        "variant_binding": {
            "current_variant_id": current_variant_id,
            "reference_variant_id": reference_variant_id,
        },
        "reference_records_path": str(source),
        "reference_records_sha256": sha256_file(source),
        "reference_run_fingerprints": sorted(reference_fingerprints),
        "compared_fields": [
            *BASELINE_EQUIVALENCE_FIELDS,
            "derived_prediction",
            "derived_progress",
        ],
    }


def score_exploratory_matrix(
    records_path: str | Path,
    labels_path: str | Path,
    selection_manifest_path: str | Path,
    output_dir: str | Path,
    expected_count: int | None = 755,
    evaluation_manifest_path: str | Path | None = None,
    reference_records_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score baseline plus three N5/N10/N20 x K8/K32/K64 families.

    The last JSONL record for each physical ``(example_id, condition)`` key is
    authoritative, allowing an invalid attempt to be superseded by a successful
    retry.  All latest logical keys must nevertheless form an exact, valid
    matrix with a single run fingerprint before any metric artifact is written.
    """
    latest, input_record_count = _load_latest_records(records_path)
    evaluation_filter = None
    if evaluation_manifest_path is not None:
        evaluation_ids, evaluation_filter = _evaluation_filter(
            evaluation_manifest_path
        )
        latest = _filter_latest_records(latest, evaluation_ids)
    observed_count = len({example_id for example_id, _condition in latest})
    effective_expected_count = (
        observed_count if expected_count is None else int(expected_count)
    )
    (
        example_ids,
        run_fingerprint,
        ranking_fingerprints,
        contract,
    ) = _validate_latest_records(
        latest, expected_count=effective_expected_count
    )
    if contract["mode"] == HUMAN_REVIEWED:
        if evaluation_filter is None:
            raise ValueError(
                "Human-reviewed scoring requires --evaluation-manifest"
            )
        if reference_records_path is None:
            raise ValueError(
                "Human-reviewed scoring requires --reference-records for "
                "the shared-baseline parity gate"
            )
    labels = load_labels(labels_path)
    label_by_id = _validate_labels_and_metadata(
        labels,
        example_ids,
        latest,
        expected_count=effective_expected_count,
        contract=contract,
        allow_label_subset=evaluation_filter is not None,
    )
    groups_by_n, selection_manifest = _load_selection_groups(
        selection_manifest_path, labels
    )
    expected_counts = _scope_expected_counts(
        label_by_id,
        groups_by_n,
        expected_count=effective_expected_count,
    )
    _validate_manifest_counts(
        selection_manifest,
        expected_counts,
        record_ids=set(example_ids),
        contract=contract,
        allow_stale_population_counts=evaluation_filter is not None,
    )
    grounding_aggregate = _grounding_aggregate(
        [
            latest[(example_id, BASELINE_CONDITION)]
            for example_id in example_ids
        ],
        contract=contract,
    )
    joined = _join_conditions(
        latest,
        label_by_id,
        groups_by_n,
        run_fingerprint=run_fingerprint,
        contract=contract,
    )
    condition_rows = _condition_metrics(
        joined,
        expected_counts,
        contract=contract,
    )
    specificity = _specificity_metrics(joined, expected_counts)
    shared_baseline = _shared_baseline_metrics(joined, expected_counts)
    baseline_equivalence = (
        _baseline_reference_equivalence(
            latest,
            reference_records_path,
            example_ids,
        )
        if reference_records_path is not None
        else None
    )
    condition_counts = dict(
        sorted(Counter(row["condition"] for row in latest.values()).items())
    )
    result: dict[str, Any] = {
        "schema_version": METRIC_CONTRACT,
        "metric_contract": METRIC_CONTRACT,
        "claim_status": contract["required_claim_status"],
        "grounding_mode": contract["mode"],
        **grounding_aggregate,
        "human_reviewed": contract["required_human_reviewed"],
        "labels_joined_only_during_scoring": True,
        "run_fingerprint": run_fingerprint,
        "ranking_fingerprints": ranking_fingerprints,
        "selection_manifest_fingerprint": selection_manifest.get("fingerprint"),
        "primary_comparison_scope": "common_unseen_s20",
        "evaluation_filter": evaluation_filter,
        "baseline_reference_equivalence": baseline_equivalence,
        "completion": {
            "expected_example_count": effective_expected_count,
            "example_count": len(example_ids),
            "input_record_count": input_record_count,
            "latest_record_count": len(latest),
            "conditions_per_example": len(REQUIRED_CONDITIONS),
            "condition_counts": condition_counts,
            "invalid_latest_count": 0,
            "complete": True,
        },
        "scope_expected_counts": expected_counts,
        "ranking_source_group_counts": {
            ranking_n: len(groups_by_n[ranking_n]) for ranking_n in RANKING_SIZES
        },
        "shared_baseline": shared_baseline,
        "paired_specificity": specificity,
        "conditions": {
            row["condition"]: {
                key: value for key, value in row.items() if key != "schema_version"
            }
            for row in condition_rows
        },
        "limitations": {
            "ranking_eval_overlap": True,
            "all_scope_in_sample_contaminated": True,
            "common_unseen_s20_is_cross_n_main_comparison": True,
            "wrong_region_control": True,
            "low_rank_control": True,
            "low_rank_control_is_layer_matched": True,
            "layer_matched_random_control": False,
            "specificity_controls_complete": True,
            "target_or_head_specific_causal_claim_requires_effect_ci": True,
            "human_reviewed_is_robustness_rerun_not_confirmatory": (
                contract["mode"] == HUMAN_REVIEWED
            ),
            "wrong_region_is_equal_size_disjoint_same_target_span": True,
        },
    }

    destination = Path(output_dir)
    write_jsonl(destination / "joined_conditions.jsonl", joined)
    write_jsonl(destination / "condition_metrics.jsonl", condition_rows)
    write_json(destination / "metrics.json", result)
    destination.mkdir(parents=True, exist_ok=True)
    report = _render_markdown(
        condition_rows,
        expected_counts,
        shared_baseline,
        specificity,
        grounding_aggregate,
        run_fingerprint=run_fingerprint,
        contract=contract,
    )
    (destination / "exp_record.md").write_text(report, encoding="utf-8")
    if destination.name == "scoring":
        (destination.parent / "exp_record.md").write_text(
            report, encoding="utf-8"
        )
    return result
