"""Frozen exploratory ranking-size by steering-size attention matrix.

This runner intentionally has a narrow contract.  It ranks heads from one
ordered, externally selected cohort, then evaluates a shared baseline and the
nine candidate-target cells of the frozen N={5,10,20} by K={8,32,64} grid.
The input manifests are label-free and explicitly marked as automatically
grounded, unreviewed, exploratory data.
"""

from __future__ import annotations

import json
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..attention_eval.masking import Head
from ..config import section
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    object_fingerprint,
    read_jsonl,
    sha256_file,
    write_json,
)
from . import causal_runner
from .checkpoint_manifest import verify_checkpoint_content_manifest
from .grounding_contract import (
    AUTO_UNREVIEWED,
    HUMAN_REVIEWED,
    aggregate_grounding_value as _contract_aggregate_grounding_value,
    grounding_composition as _contract_grounding_composition,
    grounding_contract,
    grounding_fields as _contract_grounding_fields,
    validate_grounding_row,
)
from .review_provenance import (
    load_fingerprinted_manifest,
    validate_attention_review_manifest,
    validate_jsonl_artifact,
    validate_review_provenance,
    validate_tracking_review_provenance,
)
from .tracked_grounding import validate_processor_content_order_contract


MATRIX_SCHEMA_VERSION = "my_dataset.exploratory_matrix.v1"
STRICT_GROUNDING_STATUS = "auto_assumed_unreviewed"
PROXY_GROUNDING_STATUS = "auto_proxy_unreviewed"
# Backward-compatible public name for callers which construct strict fixtures.
GROUNDING_STATUS = STRICT_GROUNDING_STATUS
_GROUNDING_STATUS_BY_RESOLUTION = {
    "strict": STRICT_GROUNDING_STATUS,
    "proxy": PROXY_GROUNDING_STATUS,
}
CLAIM_STATUS = "exploratory"
RANKING_COHORT_SCHEMA = "my_dataset.external_ranking_cohort.v1"

_PREFIX_SIZES = (5, 10, 20)
_TOP_K_VALUES = (8, 32, 64)
_RANKING_SCORE_KIND = "excess_mass"
_SKIP_EARLY_LAYERS = 8
_SWAP_BIAS = 6.0
_QUERY_SCOPE = "all"
_CAPTURE_GENERATION_ATTENTIONS = False
_BASELINE = "baseline"


def _candidate_condition(ranking_n: int, top_k: int) -> str:
    return f"candidate_target__rank_n{ranking_n:03d}__top_k{top_k:03d}"


_GRID = tuple(
    (ranking_n, top_k, _candidate_condition(ranking_n, top_k))
    for ranking_n in _PREFIX_SIZES
    for top_k in _TOP_K_VALUES
)
_CONDITIONS = (_BASELINE, *(condition for _n, _k, condition in _GRID))


def _as_int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"my_dataset_exploratory_matrix.{field} must be a list")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"my_dataset_exploratory_matrix.{field} must contain integers"
        ) from exc
    return result


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    value = dict(section(config, "my_dataset_exploratory_matrix"))
    required = (
        "variant_id",
        "model_family",
        "model_path",
        "ranking_manifest",
        "evaluation_manifest",
        "output_dir",
        "protocol",
        "content_order",
        "attention_video_max_frames",
    )
    missing = [key for key in required if key not in value or value[key] in (None, "")]
    if missing:
        raise ValueError(
            "my_dataset_exploratory_matrix is missing: " + ", ".join(missing)
        )
    if str(value["model_family"]) not in {"qwen", "roboreward", "grm"}:
        raise ValueError("model_family must be qwen, roboreward, or grm")

    prefixes = _as_int_tuple(
        value.get("ranking_prefix_sizes", _PREFIX_SIZES),
        field="ranking_prefix_sizes",
    )
    top_k = _as_int_tuple(
        value.get("steering_top_k", _TOP_K_VALUES), field="steering_top_k"
    )
    if prefixes != _PREFIX_SIZES:
        raise ValueError(f"ranking_prefix_sizes is frozen to {list(_PREFIX_SIZES)}")
    if top_k != _TOP_K_VALUES:
        raise ValueError(f"steering_top_k is frozen to {list(_TOP_K_VALUES)}")
    if str(value.get("ranking_score_kind", _RANKING_SCORE_KIND)) != _RANKING_SCORE_KIND:
        raise ValueError(f"ranking_score_kind is frozen to {_RANKING_SCORE_KIND!r}")
    if int(value.get("skip_early_layers", _SKIP_EARLY_LAYERS)) != _SKIP_EARLY_LAYERS:
        raise ValueError(f"skip_early_layers is frozen to {_SKIP_EARLY_LAYERS}")
    if not math.isclose(
        float(value.get("swap_bias", _SWAP_BIAS)), _SWAP_BIAS, rel_tol=0, abs_tol=0
    ):
        raise ValueError(f"swap_bias is frozen to {_SWAP_BIAS:g}")
    if str(value.get("steering_query_scope", _QUERY_SCOPE)) != _QUERY_SCOPE:
        raise ValueError(f"steering_query_scope is frozen to {_QUERY_SCOPE!r}")
    if bool(
        value.get(
            "capture_generation_attentions", _CAPTURE_GENERATION_ATTENTIONS
        )
    ) is not _CAPTURE_GENERATION_ATTENTIONS:
        raise ValueError("capture_generation_attentions is frozen to false")

    expected_ranking = int(value.get("expected_ranking_count", 20))
    raw_expected_evaluation = value.get("expected_evaluation_count", 755)
    expected_evaluation = (
        None
        if raw_expected_evaluation in (None, "", "auto")
        else int(raw_expected_evaluation)
    )
    if expected_ranking < max(_PREFIX_SIZES):
        raise ValueError("expected_ranking_count must be at least 20")
    if expected_evaluation is not None and expected_evaluation < 1:
        raise ValueError("expected_evaluation_count must be positive")
    contract = grounding_contract(value.get("grounding_mode"))
    variant_id = str(value["variant_id"])
    raw_reference_variant = value.get("reference_variant_id")
    reference_variant_id = (
        None
        if raw_reference_variant in (None, "")
        else str(raw_reference_variant)
    )
    if contract["mode"] == HUMAN_REVIEWED:
        if reference_variant_id is None:
            raise ValueError(
                "human_reviewed matrix requires reference_variant_id"
            )
        if reference_variant_id == variant_id:
            raise ValueError(
                "reference_variant_id must name the distinct unreviewed variant"
            )
    elif reference_variant_id is not None:
        raise ValueError(
            "reference_variant_id is allowed only for human_reviewed matrices"
        )
    protocol_addendum = value.get("protocol_addendum")
    if contract["mode"] != AUTO_UNREVIEWED and protocol_addendum in (None, ""):
        raise ValueError(
            "human_reviewed matrix requires a protocol_addendum"
        )
    addendum_path = (
        Path(str(protocol_addendum)).resolve()
        if protocol_addendum not in (None, "")
        else None
    )
    if addendum_path is not None and not addendum_path.is_file():
        raise FileNotFoundError(addendum_path)
    raw_checkpoint_manifest = value.get("checkpoint_content_manifest")
    if contract["mode"] == HUMAN_REVIEWED and raw_checkpoint_manifest in (None, ""):
        raise ValueError(
            "human_reviewed matrix requires checkpoint_content_manifest"
        )
    checkpoint_manifest_path = (
        Path(str(raw_checkpoint_manifest)).resolve()
        if raw_checkpoint_manifest not in (None, "")
        else None
    )
    if checkpoint_manifest_path is not None and not checkpoint_manifest_path.is_file():
        raise FileNotFoundError(checkpoint_manifest_path)

    value.update(
        {
            "variant_id": variant_id,
            "reference_variant_id": reference_variant_id,
            "model_family": str(value["model_family"]),
            "model_path": str(Path(value["model_path"]).resolve()),
            "ranking_manifest": str(Path(value["ranking_manifest"]).resolve()),
            "evaluation_manifest": str(
                Path(value["evaluation_manifest"]).resolve()
            ),
            "output_dir": str(Path(value["output_dir"]).resolve()),
            "protocol": str(value["protocol"]),
            "content_order": str(value["content_order"]),
            "attention_video_max_frames": int(
                value["attention_video_max_frames"]
            ),
            "ranking_prefix_sizes": list(prefixes),
            "steering_top_k": list(top_k),
            "ranking_score_kind": _RANKING_SCORE_KIND,
            "skip_early_layers": _SKIP_EARLY_LAYERS,
            "swap_bias": _SWAP_BIAS,
            "steering_query_scope": _QUERY_SCOPE,
            "capture_generation_attentions": _CAPTURE_GENERATION_ATTENTIONS,
            "expected_ranking_count": expected_ranking,
            "expected_evaluation_count": expected_evaluation,
            "expected_evaluation_count_mode": (
                "auto" if expected_evaluation is None else "fixed"
            ),
            "grounding_mode": contract["mode"],
            "grounding_contract": contract,
            "protocol_addendum": (
                str(addendum_path) if addendum_path is not None else None
            ),
            "checkpoint_content_manifest": (
                str(checkpoint_manifest_path)
                if checkpoint_manifest_path is not None
                else None
            ),
        }
    )
    for key in ("ranking_manifest", "evaluation_manifest"):
        if not Path(value[key]).is_file():
            raise FileNotFoundError(value[key])
    return value


def _grounding_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the input grounding metadata without normalizing its values."""
    return _contract_grounding_fields(row)


def _grounding_composition(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _contract_grounding_composition(
        rows,
        contract=contract or grounding_contract(),
    )


def _aggregate_grounding_value(
    rows: Sequence[Mapping[str, Any]], field: str
) -> str:
    return _contract_aggregate_grounding_value(rows, field)


def _validate_input_row(
    row: Mapping[str, Any],
    *,
    source: str,
    number: int,
    expected_model_family: str,
    contract: Mapping[str, Any],
) -> None:
    identity = str(row.get("example_id", "")).strip()
    if not identity:
        raise ValueError(f"{source} row {number} has no example_id")
    actual_model_family = row.get("model_family")
    if actual_model_family != expected_model_family:
        raise ValueError(
            f"{source}/{identity}: model_family must equal configured "
            f"model_family {expected_model_family!r}; found {actual_model_family!r}"
        )
    for field in ("group_id", "task_id", "task_family"):
        if not str(row.get(field, "")).strip():
            raise ValueError(f"{source}/{identity}: missing {field}")
    validate_grounding_row(
        row,
        identity=f"{source}/{identity}",
        contract=contract,
    )


def _load_inputs(
    path: str | Path,
    *,
    source: str,
    expected_count: int | None,
    ordered_ranking: bool,
    expected_model_family: str,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"{source} manifest must contain at least one row: {path}")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {source} rows, found {len(rows)} in {path}"
        )
    for number, row in enumerate(rows, 1):
        _validate_input_row(
            row,
            source=source,
            number=number,
            expected_model_family=expected_model_family,
            contract=contract,
        )
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted(
            value for value, count in Counter(ids).items() if count > 1
        )
        raise ValueError(f"{source} example_id values are not unique: {duplicates[:5]}")
    if ordered_ranking:
        orders = [row.get("ranking_order") for row in rows]
        if orders != list(range(1, len(rows) + 1)):
            raise ValueError(
                "ranking_manifest ranking_order must be the contiguous input order 1..N"
            )
    return rows


def _manifest_reference(
    reference: Any,
    *,
    actual_path: Path,
    actual_manifest: Mapping[str, Any],
    identity: str,
) -> dict[str, str]:
    """Validate a child manifest's cryptographic reference to its parent."""
    if not isinstance(reference, Mapping):
        raise ValueError(f"{identity} reference is missing")
    source = actual_path.resolve()
    recorded_path = Path(str(reference.get("path", ""))).resolve()
    if recorded_path != source:
        raise ValueError(
            f"{identity} path differs: recorded={recorded_path}, actual={source}"
        )
    actual_sha256 = sha256_file(source)
    if reference.get("sha256") != actual_sha256:
        raise ValueError(f"{identity} SHA-256 differs")
    actual_fingerprint = str(actual_manifest.get("fingerprint", ""))
    if reference.get("fingerprint") != actual_fingerprint:
        raise ValueError(f"{identity} fingerprint differs")
    return {
        "path": str(source),
        "sha256": actual_sha256,
        "fingerprint": actual_fingerprint,
    }


def _reviewed_manifest_summary(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "fingerprint": str(manifest["fingerprint"]),
    }


def _review_provenance_key(manifest: Mapping[str, Any], *, identity: str) -> str:
    has_legacy = "review_provenance" in manifest
    has_tracking = "tracking_review_provenance" in manifest
    if has_legacy == has_tracking:
        raise ValueError(
            f"{identity} must contain exactly one reviewed provenance kind"
        )
    return "tracking_review_provenance" if has_tracking else "review_provenance"


def _validate_review_provenance_value(
    value: Any,
    *,
    key: str,
    identity: str,
) -> dict[str, Any]:
    if key == "tracking_review_provenance":
        return dict(
            validate_tracking_review_provenance(value, identity=identity)
        )
    if key == "review_provenance":
        return dict(validate_review_provenance(value, identity=identity))
    raise ValueError(f"Unsupported reviewed provenance key: {key!r}")


def _validate_tracking_review_contract(
    value: Mapping[str, Any], *, identity: str
) -> None:
    if "wrong_region_bbox" in value:
        raise ValueError(f"{identity}: tracking v2 forbids wrong_region_bbox")
    if value.get("target_grounding_scope") != "terminal_only":
        raise ValueError(f"{identity}: target_grounding_scope must be terminal_only")
    if value.get("control_region_policy") != "none":
        raise ValueError(f"{identity}: control_region_policy must be none")


def _propagate_review_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical reviewed fields for every downstream artifact."""
    has_legacy = cfg.get("review_provenance") is not None
    has_tracking = cfg.get("tracking_review_provenance") is not None
    if has_legacy and has_tracking:
        raise ValueError("matrix configuration mixes reviewed provenance kinds")
    if has_tracking:
        if cfg.get("review_source_kind") != "tracked_grounding_v2":
            raise ValueError("tracking provenance requires tracked_grounding_v2")
        if cfg.get("target_grounding_scope") != "terminal_only":
            raise ValueError("tracking matrix target scope must be terminal_only")
        if cfg.get("control_region_policy") != "none":
            raise ValueError("tracking matrix control-region policy must be none")
        return {
            "review_source_kind": "tracked_grounding_v2",
            "tracking_review_provenance": dict(
                validate_tracking_review_provenance(
                    cfg["tracking_review_provenance"],
                    identity="matrix tracking review provenance",
                )
            ),
            "target_grounding_scope": "terminal_only",
            "control_region_policy": "none",
        }
    if has_legacy:
        value: dict[str, Any] = {
            "review_provenance": dict(
                validate_review_provenance(
                    cfg["review_provenance"],
                    identity="matrix legacy review provenance",
                )
            )
        }
        if cfg.get("review_source_kind") is not None:
            value["review_source_kind"] = cfg["review_source_kind"]
        return value
    return {}


def _validate_reviewed_input_chain(
    cfg: Mapping[str, Any],
    ranking_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind reviewed matrix JSONL inputs to their fingerprinted parents.

    This is intentionally stricter than ID/grounding validation: a stale or
    hand-edited same-ID JSONL must not reach model loading.
    """
    ranking_path = Path(str(cfg["ranking_manifest"])).resolve()
    evaluation_path = Path(str(cfg["evaluation_manifest"])).resolve()
    selection_path = ranking_path.parent.parent / "selection_manifest.json"
    attention_path = evaluation_path.parent.parent / "manifest.json"

    selection = load_fingerprinted_manifest(
        selection_path,
        identity="reviewed ranking selection manifest",
    )
    attention = load_fingerprinted_manifest(
        attention_path,
        identity="reviewed attention manifest",
    )
    if selection.get("schema_version") != RANKING_COHORT_SCHEMA:
        raise ValueError("reviewed ranking selection manifest schema_version is invalid")
    if selection.get("grounding_mode") != HUMAN_REVIEWED:
        raise ValueError(
            "reviewed ranking selection manifest grounding_mode is not human_reviewed"
        )
    if selection.get("claim_status") != "reviewed_exploratory":
        raise ValueError(
            "reviewed ranking selection manifest claim_status is invalid"
        )

    attention_provenance = validate_attention_review_manifest(
        attention,
        identity="reviewed attention manifest",
    )
    provenance_key = _review_provenance_key(
        attention, identity="reviewed attention manifest"
    )
    selection_key = _review_provenance_key(
        selection, identity="reviewed ranking selection manifest"
    )
    if selection_key != provenance_key:
        raise ValueError(
            "reviewed ranking and attention manifests use different "
            "provenance kinds"
        )
    selection_provenance = _validate_review_provenance_value(
        selection.get(provenance_key),
        key=provenance_key,
        identity=(
            "reviewed ranking selection manifest."
            f"{provenance_key}"
        ),
    )
    if selection_provenance != attention_provenance:
        raise ValueError(
            "reviewed ranking and attention manifests have different provenance"
        )
    review_source_kind = str(
        attention.get("review_source_kind") or "legacy_grounding_v1"
    )
    if str(
        selection.get("review_source_kind") or "legacy_grounding_v1"
    ) != review_source_kind:
        raise ValueError(
            "reviewed ranking and attention manifests have different "
            "review_source_kind"
        )
    if provenance_key == "tracking_review_provenance":
        if review_source_kind != "tracked_grounding_v2":
            raise ValueError("tracking provenance has an invalid review_source_kind")
        _validate_tracking_review_contract(
            selection, identity="reviewed ranking selection manifest"
        )
    _manifest_reference(
        selection.get("attention_manifest"),
        actual_path=attention_path,
        actual_manifest=attention,
        identity="reviewed selection attention_manifest",
    )

    model_family = str(cfg["model_family"])
    if evaluation_path.name != "complete_groups.jsonl":
        raise ValueError(
            "reviewed evaluation manifest must be complete_groups.jsonl"
        )
    evaluation_population = selection.get("evaluation_population")
    if not isinstance(evaluation_population, Mapping):
        raise ValueError(
            "reviewed ranking selection manifest has no evaluation_population"
        )
    evaluation_filename = evaluation_population.get(
        "evaluation_attention_filename"
    )
    if evaluation_filename != evaluation_path.name:
        raise ValueError(
            "reviewed selection evaluation_attention_filename differs from "
            f"the configured evaluation JSONL: {evaluation_filename!r} != "
            f"{evaluation_path.name!r}"
        )
    ranking_filename = evaluation_population.get("ranking_attention_filename")
    attention_inputs = selection.get("attention_inputs")
    if not isinstance(attention_inputs, Mapping):
        raise ValueError(
            "reviewed ranking selection manifest has no attention_inputs"
        )
    if ranking_filename == evaluation_filename:
        selection_evaluation_artifacts = attention_inputs
    else:
        selection_evaluation_artifacts = attention_inputs.get("evaluation")
    selection_evaluation_artifact = (
        selection_evaluation_artifacts.get(model_family)
        if isinstance(selection_evaluation_artifacts, Mapping)
        else None
    )
    validated_selection_evaluation = validate_jsonl_artifact(
        selection_evaluation_artifact,
        actual_path=evaluation_path,
        rows=evaluation_rows,
        identity=f"reviewed selection evaluation {model_family}",
    )

    model_outputs = selection.get("model_outputs")
    ranking_artifact = (
        model_outputs.get(model_family)
        if isinstance(model_outputs, Mapping)
        else None
    )
    validated_ranking = validate_jsonl_artifact(
        ranking_artifact,
        actual_path=ranking_path,
        rows=ranking_rows,
        identity=f"reviewed ranking {model_family}/ordered_max20",
    )
    attention_artifacts = attention.get("artifacts")
    evaluation_key = f"{model_family}/{evaluation_path.stem}"
    evaluation_artifact = (
        attention_artifacts.get(evaluation_key)
        if isinstance(attention_artifacts, Mapping)
        else None
    )
    validated_evaluation = validate_jsonl_artifact(
        evaluation_artifact,
        actual_path=evaluation_path,
        rows=evaluation_rows,
        identity=f"reviewed evaluation {evaluation_key}",
    )
    if validated_selection_evaluation != validated_evaluation:
        raise ValueError(
            "reviewed selection and attention evaluation artifacts differ"
        )

    for source, rows in (
        ("ranking", ranking_rows),
        ("evaluation", evaluation_rows),
    ):
        for number, row in enumerate(rows, 1):
            row_provenance = _validate_review_provenance_value(
                row.get(provenance_key),
                key=provenance_key,
                identity=f"reviewed {source} row {number}.{provenance_key}",
            )
            if row_provenance != attention_provenance:
                raise ValueError(
                    f"reviewed {source} row {number} provenance differs from "
                    "the parent manifests"
                )
            other_key = (
                "review_provenance"
                if provenance_key == "tracking_review_provenance"
                else "tracking_review_provenance"
            )
            if other_key in row:
                raise ValueError(
                    f"reviewed {source} row {number} mixes provenance kinds"
                )
            if provenance_key == "tracking_review_provenance":
                _validate_tracking_review_contract(
                    row, identity=f"reviewed {source} row {number}"
                )
                if model_family in {"roboreward", "qwen"}:
                    source_order = str(row.get("content_order", ""))
                    runtime_order = str(cfg["content_order"])
                    if source_order not in {
                        "text_then_video",
                        "video_then_text",
                    }:
                        raise ValueError(
                            f"reviewed {source} row {number}: invalid "
                            "content_order"
                        )
                    indices = row.get("processor_frame_indices")
                    if (
                        not isinstance(indices, list)
                        or not indices
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            for value in indices
                        )
                    ):
                        raise ValueError(
                            f"reviewed {source} row {number}: invalid frozen "
                            "processor_frame_indices"
                        )
                    grid = row.get("processor_video_grid_thw")
                    if (
                        not isinstance(grid, list)
                        or not grid
                        or any(
                            not isinstance(grid_row, list)
                            or len(grid_row) != 3
                            or any(
                                isinstance(value, bool)
                                or not isinstance(value, int)
                                or value < 1
                                for value in grid_row
                            )
                            for grid_row in grid
                        )
                    ):
                        raise ValueError(
                            f"reviewed {source} row {number}: invalid frozen "
                            "processor_video_grid_thw"
                        )
                if model_family == "roboreward":
                    contract = validate_processor_content_order_contract(
                        row,
                        identity=f"reviewed {source} row {number}/roboreward",
                    )
                    if (
                        contract is not None
                        and source_order not in contract["validated_orders"]
                    ):
                        raise ValueError(
                            f"reviewed {source} row {number}: source "
                            "content_order is outside the frozen contract"
                        )
                    if source_order != runtime_order and (
                        contract is None
                        or runtime_order not in contract["validated_orders"]
                    ):
                        raise ValueError(
                            f"reviewed {source} row {number}: content_order "
                            f"{source_order!r} cannot be rebound to "
                            f"{runtime_order!r} without the frozen dual-order "
                            "processor contract"
                        )
                elif model_family == "qwen" and source_order != runtime_order:
                    raise ValueError(
                        f"reviewed {source} row {number}: Qwen content_order "
                        f"{source_order!r} differs from configured "
                        f"{runtime_order!r}"
                    )
                elif "processor_content_order_contract" in row:
                    raise ValueError(
                        f"reviewed {source} row {number}: processor "
                        "content-order contract is RoboReward-only"
                    )

    result: dict[str, Any] = {
        "provenance_key": provenance_key,
        provenance_key: attention_provenance,
        "review_source_kind": review_source_kind,
        "selection_manifest": _reviewed_manifest_summary(selection_path, selection),
        "attention_manifest": _reviewed_manifest_summary(attention_path, attention),
        "ranking_artifact": validated_ranking,
        "evaluation_artifact": validated_evaluation,
        "selection_evaluation_artifact": validated_selection_evaluation,
    }
    if provenance_key == "tracking_review_provenance":
        result["target_grounding_scope"] = "terminal_only"
        result["control_region_policy"] = "none"
        if model_family == "roboreward":
            result["processor_content_order_binding"] = {
                "runtime_content_order": str(cfg["content_order"]),
                "source_content_orders": sorted(
                    {
                        str(row.get("content_order", ""))
                        for row in (*ranking_rows, *evaluation_rows)
                    }
                ),
                "cross_order_requires_frozen_contract": True,
            }
    return result


def _optional_path_fingerprint(value: Any) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).resolve()
    return {
        "path": str(path),
        "fingerprint": artifact_fingerprint(path),
    }


def _implementation_fingerprint(cfg: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(__file__).resolve()
    package_root = source.parents[1]
    paths = [
        source,
        source.with_name("causal_runner.py"),
        source.with_name("checkpoint_manifest.py"),
        source.with_name("grounding_contract.py"),
        package_root / "config.py",
        package_root / "io.py",
        package_root / "protocol.py",
        package_root / "attention_eval" / "masking.py",
    ]
    if str(cfg["model_family"]) in {"qwen", "roboreward"}:
        paths.extend(
            [
                package_root / "qwen_eval" / "attention.py",
                package_root / "qwen_eval" / "protocols.py",
                package_root / "attention_eval" / "runtime.py",
                package_root / "roboreward_eval" / "runner.py",
            ]
        )
    else:
        paths.extend(
            [
                package_root / "attention_eval" / "runtime.py",
                package_root / "attention_eval" / "stats.py",
            ]
        )
    repository_root = package_root.parent
    return {
        "source_sha256": {
            path.relative_to(repository_root).as_posix(): sha256_file(path)
            for path in paths
        }
    }


def _runtime_fingerprint(cfg: Mapping[str, Any]) -> dict[str, Any]:
    family = str(cfg["model_family"])
    dtype = str(cfg.get("torch_dtype", cfg.get("dtype", "bfloat16")))
    if family == "grm":
        min_pixels = int(cfg.get("min_pixels", 12544))
        max_pixels = int(cfg.get("max_pixels", 76800))
        max_new_tokens = int(cfg.get("max_new_tokens", 16))
    else:
        min_pixels = cfg.get("min_pixels")
        max_pixels = cfg.get("max_pixels")
        max_new_tokens = int(cfg.get("max_new_tokens", 32))
    return {
        "max_new_tokens": max_new_tokens,
        "dtype": dtype,
        "device_map": cfg.get("device_map", "auto"),
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "prompt_mode": str(cfg.get("prompt_mode", "official")),
        "blank_goal": _optional_path_fingerprint(cfg.get("blank_goal")),
        "intervention_location": str(
            cfg.get("intervention_location", "after_cam_high")
        ),
        "capture_generation_attentions": bool(
            cfg["capture_generation_attentions"]
        ),
        "generation_contract": {
            "attn_implementation": "eager",
            "decoding": "greedy",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "use_cache": True,
            "output_attentions": False,
            "return_dict_in_generate": True,
        },
    }


def _fingerprint_components(
    cfg: Mapping[str, Any],
    ranking_samples: Sequence[Mapping[str, Any]],
    evaluation_samples: Sequence[Mapping[str, Any]],
    reviewed_input_chain: Mapping[str, Any] | None = None,
    checkpoint_content_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ranking_path = Path(str(cfg["ranking_manifest"]))
    evaluation_path = Path(str(cfg["evaluation_manifest"]))
    grounding = {
        "contract": dict(cfg["grounding_contract"]),
        "ranking": {
            "composition": _grounding_composition(
                ranking_samples,
                contract=cfg["grounding_contract"],
            ),
            "examples": [
                {
                    "example_id": str(row["example_id"]),
                    "grounding_resolution": row["grounding_resolution"],
                    "grounding_status": row["grounding_status"],
                    "human_reviewed": row["human_reviewed"],
                    "claim_status": row["claim_status"],
                }
                for row in ranking_samples
            ],
        },
        "evaluation": {
            "composition": _grounding_composition(
                evaluation_samples,
                contract=cfg["grounding_contract"],
            ),
            "examples": [
                {
                    "example_id": str(row["example_id"]),
                    "grounding_resolution": row["grounding_resolution"],
                    "grounding_status": row["grounding_status"],
                    "human_reviewed": row["human_reviewed"],
                    "claim_status": row["claim_status"],
                }
                for row in evaluation_samples
            ],
        },
    }
    components = {
        "implementation": _implementation_fingerprint(cfg),
        "model": {
            "variant_id": cfg["variant_id"],
            "reference_variant_id": cfg.get("reference_variant_id"),
            "model_family": cfg["model_family"],
            "model_path": cfg["model_path"],
            "model_fingerprint": artifact_fingerprint(cfg["model_path"]),
            "runtime": _runtime_fingerprint(cfg),
        },
        "input": {
            "ranking_manifest": str(ranking_path),
            "ranking_manifest_sha256": sha256_file(ranking_path),
            "evaluation_manifest": str(evaluation_path),
            "evaluation_manifest_sha256": sha256_file(evaluation_path),
            "ranking_count": len(ranking_samples),
            "evaluation_count": len(evaluation_samples),
            "expected_evaluation_count_mode": cfg[
                "expected_evaluation_count_mode"
            ],
            "protocol_addendum": _optional_path_fingerprint(
                cfg.get("protocol_addendum")
            ),
        },
        "order": {
            "ranking": [
                {
                    "ranking_order": int(row["ranking_order"]),
                    "example_id": str(row["example_id"]),
                }
                for row in ranking_samples
            ],
            "evaluation_example_ids": [
                str(row["example_id"]) for row in evaluation_samples
            ],
        },
        "grounding": grounding,
        "frame": {
            "protocol": cfg["protocol"],
            "content_order": cfg["content_order"],
            "attention_video_max_frames": cfg["attention_video_max_frames"],
            "intervention_location": cfg.get(
                "intervention_location", "after_cam_high"
            ),
        },
        "method": {
            "mass_collection": "runtime.collect_mass_terminal_last_prompt",
            "ranking_score_kind": cfg["ranking_score_kind"],
            "aggregation": "sample_mean",
            "skip_early_layers": cfg["skip_early_layers"],
        },
        "grid": {
            "ranking_prefix_sizes": cfg["ranking_prefix_sizes"],
            "steering_top_k": cfg["steering_top_k"],
            "swap_bias": cfg["swap_bias"],
            "steering_query_scope": cfg["steering_query_scope"],
            "conditions": list(_CONDITIONS),
        },
    }
    if reviewed_input_chain is not None:
        components["reviewed_input_chain"] = dict(reviewed_input_chain)
    if checkpoint_content_verification is not None:
        components["model"]["checkpoint_content_verification"] = dict(
            checkpoint_content_verification
        )
    return components


def _validate_existing_manifest(path: Path, run_fingerprint: str) -> None:
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise RuntimeError(f"Existing matrix manifest has an incompatible schema: {path}")
    if value.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(
            f"run fingerprint mismatch for existing matrix output: {path}"
        )


def _load_latest_mass(
    path: Path,
    *,
    expected_ids: set[str],
    run_fingerprint: str,
    retry_failed: bool,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    latest: dict[str, dict[str, Any]] = {}
    attempts: Counter[str] = Counter()
    if not path.is_file():
        return latest, attempts
    for number, row in enumerate(read_jsonl(path), 1):
        example_id = str(row.get("example_id", ""))
        if example_id not in expected_ids:
            raise RuntimeError(
                f"Unexpected ranking mass example_id at {path}:{number}: {example_id!r}"
            )
        if row.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"run fingerprint mismatch in {path}:{number}")
        latest[example_id] = row
        attempts[example_id] += 1
    invalid = sorted(
        example_id
        for example_id, row in latest.items()
        if row.get("status") != "ok"
    )
    if invalid and not retry_failed:
        raise RuntimeError(
            f"Latest ranking mass attempts are invalid for {invalid[:5]}; "
            "rerun with retry_failed=True"
        )
    return latest, attempts


def _load_latest_records(
    path: Path,
    *,
    expected_ids: set[str],
    run_fingerprint: str,
    retry_failed: bool,
) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[tuple[str, str]]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    attempts: Counter[tuple[str, str]] = Counter()
    if not path.is_file():
        return latest, attempts
    allowed = set(_CONDITIONS)
    for number, row in enumerate(read_jsonl(path), 1):
        example_id = str(row.get("example_id", ""))
        condition = str(row.get("condition", ""))
        if example_id not in expected_ids or condition not in allowed:
            raise RuntimeError(
                f"Unexpected steering key at {path}:{number}: "
                f"({example_id!r}, {condition!r})"
            )
        if row.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"run fingerprint mismatch in {path}:{number}")
        key = (example_id, condition)
        latest[key] = row
        attempts[key] += 1
    invalid = sorted(key for key, row in latest.items() if row.get("status") != "ok")
    if invalid and not retry_failed:
        raise RuntimeError(
            f"Latest steering attempts are invalid for {invalid[:5]}; "
            "rerun with retry_failed=True"
        )
    return latest, attempts


def _ranking_rows(
    mass_rows: Sequence[Mapping[str, Any]],
    *,
    score_kind: str,
    skip_early_layers: int,
    limit: int,
) -> list[dict[str, Any]]:
    arrays = []
    expected_shape: tuple[int, int] | None = None
    for row in mass_rows:
        identity = str(row.get("example_id", "<unknown>"))
        if score_kind not in row:
            raise ValueError(f"{identity}: ranking mass has no {score_kind}")
        array = np.asarray(row[score_kind], dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(
                f"{identity}: {score_kind} must be a layer-by-head matrix, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{identity}: {score_kind} contains non-finite values")
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in array.shape)
        elif tuple(array.shape) != expected_shape:
            raise ValueError(
                f"{identity}: {score_kind} shape {array.shape} differs from {expected_shape}"
            )
        arrays.append(array)
    if not arrays:
        raise ValueError("Cannot derive a ranking from no mass records")
    aggregate = np.stack(arrays, axis=0).mean(axis=0)
    if not 0 <= skip_early_layers < aggregate.shape[0]:
        raise ValueError(
            f"skip_early_layers={skip_early_layers} is outside {aggregate.shape[0]} layers"
        )
    rows = [
        {
            "layer": int(layer),
            "head": int(head),
            "score": float(aggregate[layer, head]),
        }
        for layer in range(skip_early_layers, aggregate.shape[0])
        for head in range(aggregate.shape[1])
    ]
    rows.sort(key=lambda row: (-row["score"], row["layer"], row["head"]))
    if len(rows) < limit:
        raise ValueError(
            f"Only {len(rows)} eligible ranked heads are available; need {limit}"
        )
    result = rows[:limit]
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def _ranking_artifact(
    cfg: Mapping[str, Any],
    mass_rows: Sequence[Mapping[str, Any]],
    *,
    ranking_n: int,
    run_fingerprint: str,
) -> dict[str, Any]:
    selected_mass_rows = list(mass_rows[:ranking_n])
    ranking = _ranking_rows(
        selected_mass_rows,
        score_kind=str(cfg["ranking_score_kind"]),
        skip_early_layers=int(cfg["skip_early_layers"]),
        limit=max(int(value) for value in cfg["steering_top_k"]),
    )
    artifact: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "variant_id": cfg["variant_id"],
        "reference_variant_id": cfg.get("reference_variant_id"),
        "model_family": cfg["model_family"],
        "claim_status": cfg["grounding_contract"]["required_claim_status"],
        "grounding_mode": cfg["grounding_mode"],
        "grounding_status": _aggregate_grounding_value(
            selected_mass_rows, "grounding_status"
        ),
        "grounding_resolution": _aggregate_grounding_value(
            selected_mass_rows, "grounding_resolution"
        ),
        "grounding_composition": _grounding_composition(
            selected_mass_rows,
            contract=cfg["grounding_contract"],
        ),
        "human_reviewed": cfg["grounding_contract"][
            "required_human_reviewed"
        ],
        "run_fingerprint": run_fingerprint,
        "ranking_n": ranking_n,
        "sample_count": ranking_n,
        "sample_example_ids": [
            str(row["example_id"]) for row in mass_rows[:ranking_n]
        ],
        "method": "terminal_last_prompt_excess_mass_sample_mean_skip8",
        "ranking_score_kind": cfg["ranking_score_kind"],
        "skip_early_layers": cfg["skip_early_layers"],
        "ranking": ranking,
    }
    artifact.update(_propagate_review_contract(cfg))
    artifact["fingerprint"] = object_fingerprint(artifact)
    return artifact


def _write_or_validate_ranking(path: Path, artifact: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != artifact["run_fingerprint"]:
            raise RuntimeError(f"run fingerprint mismatch in ranking artifact {path}")
        if existing.get("fingerprint") != artifact["fingerprint"]:
            raise RuntimeError(f"Derived ranking fingerprint mismatch for {path}")
        fingerprint_view = dict(existing)
        recorded = fingerprint_view.pop("fingerprint", None)
        if recorded != object_fingerprint(fingerprint_view):
            raise RuntimeError(f"Corrupt ranking artifact fingerprint: {path}")
        return
    write_json(path, artifact)


def _derive_rankings(
    cfg: Mapping[str, Any],
    ranking_samples: Sequence[Mapping[str, Any]],
    latest_mass: Mapping[str, Mapping[str, Any]],
    *,
    ranking_dir: Path,
    run_fingerprint: str,
) -> dict[int, dict[str, Any]]:
    ordered = [latest_mass[str(sample["example_id"])] for sample in ranking_samples]
    artifacts = {}
    for ranking_n in _PREFIX_SIZES:
        artifact = _ranking_artifact(
            cfg, ordered, ranking_n=ranking_n, run_fingerprint=run_fingerprint
        )
        path = ranking_dir / f"rank_n{ranking_n:03d}.json"
        _write_or_validate_ranking(path, artifact)
        artifacts[ranking_n] = artifact
    return artifacts


def _all_mass_complete(
    latest: Mapping[str, Mapping[str, Any]], expected_ids: Iterable[str]
) -> bool:
    return all(latest.get(value, {}).get("status") == "ok" for value in expected_ids)


def _all_steering_complete(
    latest: Mapping[tuple[str, str], Mapping[str, Any]],
    expected_ids: Iterable[str],
) -> bool:
    return all(
        latest.get((example_id, condition), {}).get("status") == "ok"
        for example_id in expected_ids
        for condition in _CONDITIONS
    )


def _validate_record_rankings(
    latest: Mapping[tuple[str, str], Mapping[str, Any]],
    rankings: Mapping[int, Mapping[str, Any]],
) -> None:
    for (_example_id, condition), row in latest.items():
        if row.get("status") != "ok" or condition == _BASELINE:
            continue
        matches = [item for item in _GRID if item[2] == condition]
        if len(matches) != 1:
            raise AssertionError(f"Unknown frozen condition {condition}")
        ranking_n, top_k, _condition = matches[0]
        if row.get("ranking_n") != ranking_n or row.get("top_k") != top_k:
            raise RuntimeError(
                f"Ranking coordinates mismatch in steering record for {condition}"
            )
        if row.get("ranking_fingerprint") != rankings[ranking_n]["fingerprint"]:
            raise RuntimeError(
                f"Ranking fingerprint mismatch in steering record for {condition}"
            )
        raw_heads = row.get("heads")
        if not isinstance(raw_heads, list) or len(raw_heads) != top_k:
            raise RuntimeError(
                f"Steering record for {condition} must contain exactly {top_k} heads"
            )
        actual: list[tuple[int, int]] = []
        for index, value in enumerate(raw_heads, 1):
            if not isinstance(value, Mapping):
                raise RuntimeError(
                    f"Steering record for {condition} head {index} is not a mapping"
                )
            layer = value.get("layer")
            head = value.get("head")
            if (
                isinstance(layer, bool)
                or isinstance(head, bool)
                or not isinstance(layer, (int, np.integer))
                or not isinstance(head, (int, np.integer))
                or int(layer) < 0
                or int(head) < 0
            ):
                raise RuntimeError(
                    f"Steering record for {condition} head {index} is invalid"
                )
            actual.append((int(layer), int(head)))
        if len(actual) != len(set(actual)):
            raise RuntimeError(
                f"Steering record for {condition} contains duplicate heads"
            )
        expected = [
            (int(value["layer"]), int(value["head"]))
            for value in rankings[ranking_n]["ranking"][:top_k]
        ]
        if actual != expected:
            raise RuntimeError(
                f"Ordered Top-{top_k} heads mismatch in steering record for {condition}"
            )


def _head_values(artifact: Mapping[str, Any], top_k: int) -> list[Head]:
    rows = artifact.get("ranking")
    if not isinstance(rows, list) or len(rows) < top_k:
        raise ValueError(f"Ranking artifact contains fewer than {top_k} heads")
    return [
        Head(layer=int(row["layer"]), head=int(row["head"]))
        for row in rows[:top_k]
    ]


def _runtime_contract(runtime: Any, cfg: Mapping[str, Any]) -> None:
    actual_protocol = getattr(runtime, "protocol", None)
    if actual_protocol is not None and str(actual_protocol) != str(cfg["protocol"]):
        raise RuntimeError(
            f"Runtime protocol {actual_protocol!r} differs from frozen {cfg['protocol']!r}"
        )
    actual_order = getattr(runtime, "content_order", None)
    if actual_order is not None and str(actual_order) != str(cfg["content_order"]):
        raise RuntimeError(
            f"Runtime content_order {actual_order!r} differs from frozen "
            f"{cfg['content_order']!r}"
        )
    # Only native-video Qwen/RoboReward consume video_processor.max_frames.
    # GRM's frozen value is an eight-image layout count; comparing it with an
    # unused processor video default (often 768) incorrectly aborts the run.
    if str(cfg["model_family"]) in {"qwen", "roboreward"}:
        processor = getattr(runtime, "processor", None)
        video_processor = getattr(processor, "video_processor", None)
        actual_cap = getattr(video_processor, "max_frames", None)
        if actual_cap is None:
            raise RuntimeError("Native-video runtime exposes no max_frames contract")
        if int(actual_cap) != int(cfg["attention_video_max_frames"]):
            raise RuntimeError(
                f"Runtime frame cap {actual_cap} differs from frozen "
                f"{cfg['attention_video_max_frames']}"
            )


def _prepared_contract(prepared: Any, cfg: Mapping[str, Any]) -> None:
    actual_protocol = getattr(prepared, "protocol", None)
    if actual_protocol is not None and str(actual_protocol) != str(cfg["protocol"]):
        raise RuntimeError("Prepared input protocol differs from the frozen protocol")
    metadata = getattr(prepared, "video_metadata", None)
    if isinstance(metadata, Mapping):
        indices = metadata.get("frames_indices")
        cap = int(cfg["attention_video_max_frames"])
        if isinstance(indices, list) and len(indices) > cap:
            raise RuntimeError(
                f"Prepared input sampled {len(indices)} frames above frozen cap {cap}"
            )


def _generation_context(
    runtime: Any, cfg: Mapping[str, Any], sample: dict[str, Any]
) -> tuple[Any, list[int], list[int], str]:
    prepared = runtime.prepare(sample)
    _prepared_contract(prepared, cfg)
    if str(cfg["model_family"]) in {"qwen", "roboreward"}:
        target = list(runtime.target_positions(sample, prepared))
        visual = list(prepared.visual_positions)
        return prepared, target, visual, "visual_positions"
    try:
        _inputs, spans = prepared
    except (TypeError, ValueError) as exc:
        raise RuntimeError("GRM prepare() must return (inputs, spans)") from exc
    target, visual, _target_spans = runtime.target_positions(
        sample,
        spans,
        str(cfg.get("intervention_location", "after_cam_high")),
    )
    return prepared, list(target), list(visual), "image_positions"


def _assert_hook_applied(
    result: Mapping[str, Any], heads: Sequence[Head]
) -> dict[str, Any]:
    diagnostics = result.get("hook_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise AssertionError("Candidate result has no structured hook_diagnostics")
    if diagnostics.get("hook_active") is not True:
        raise AssertionError("Candidate attention hook was not active")
    per_layer = diagnostics.get("per_layer")
    if not isinstance(per_layer, Mapping):
        raise AssertionError("Candidate hook diagnostics have no per_layer mapping")
    by_layer: dict[str, int] = {}
    expected_heads: dict[int, list[int]] = {}
    for head in heads:
        expected_heads.setdefault(int(head.layer), []).append(int(head.head))
    for layer in sorted(expected_heads):
        value = per_layer.get(str(layer), per_layer.get(layer))
        if not isinstance(value, Mapping):
            raise AssertionError(f"Candidate hook has no diagnostics for layer {layer}")
        selected_heads = value.get("selected_heads")
        if (
            not isinstance(selected_heads, list)
            or sorted(int(head) for head in selected_heads)
            != sorted(expected_heads[layer])
        ):
            raise AssertionError(
                f"Candidate hook layer {layer} selected_heads differ from "
                "the frozen condition"
            )
        integer_fields = {}
        for field in (
            "calls",
            "applied_calls",
            "skipped_calls",
            "missing_mask_calls",
            "selected_token_count",
            "other_visual_token_count",
        ):
            field_value = value.get(field)
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, (int, np.integer))
            ):
                raise AssertionError(
                    f"Candidate hook layer {layer} has invalid "
                    f"{field}={field_value!r}"
                )
            integer_fields[field] = int(field_value)
        if integer_fields["calls"] <= 0:
            raise AssertionError(
                f"Candidate hook layer {layer} calls must be positive"
            )
        if (
            value.get("query_scope") != _QUERY_SCOPE
            or integer_fields["applied_calls"] != integer_fields["calls"]
            or integer_fields["skipped_calls"] != 0
            or integer_fields["missing_mask_calls"] != 0
        ):
            raise AssertionError(
                f"Candidate hook layer {layer} did not apply on every "
                "scope=all invocation"
            )
        if (
            integer_fields["selected_token_count"] <= 0
            or integer_fields["other_visual_token_count"] <= 0
            or value.get("selected_other_disjoint") is not True
            or not math.isclose(
                float(value.get("swap_bias", float("nan"))),
                _SWAP_BIAS,
                rel_tol=0,
                abs_tol=0,
            )
        ):
            raise AssertionError(
                f"Candidate hook layer {layer} has invalid target/control evidence"
            )
        by_layer[str(layer)] = integer_fields["applied_calls"]
    return {
        "passed": True,
        "applied_calls_by_layer": by_layer,
        "applied_calls": sum(by_layer.values()),
    }


def _base_record(
    sample: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    condition: str,
    condition_kind: str,
    run_fingerprint: str,
    attempt: int,
) -> dict[str, Any]:
    value = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "variant_id": cfg["variant_id"],
        "reference_variant_id": cfg.get("reference_variant_id"),
        "model_family": cfg["model_family"],
        "example_id": sample["example_id"],
        "group_id": sample["group_id"],
        "task_id": sample["task_id"],
        "task_family": sample["task_family"],
        "partition": sample.get("partition"),
        "condition": condition,
        "condition_kind": condition_kind,
        "run_fingerprint": run_fingerprint,
        **_grounding_fields(sample),
        "attempt": attempt,
    }
    value.update(_propagate_review_contract(cfg))
    return value


def _append_invalid(
    path: Path,
    base: dict[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    row = {
        **base,
        "status": "invalid",
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    }
    append_jsonl(path, row)
    return row


def _manifest(
    cfg: Mapping[str, Any],
    components: Mapping[str, Any],
    *,
    run_fingerprint: str,
    status: str,
    mass_path: Path,
    records_path: Path,
    latest_mass: Mapping[str, Mapping[str, Any]],
    latest_records: Mapping[tuple[str, str], Mapping[str, Any]],
    rankings: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    ok_mass = sum(row.get("status") == "ok" for row in latest_mass.values())
    ok_records = sum(row.get("status") == "ok" for row in latest_records.values())
    grounding = components["grounding"]
    ranking_grounding = grounding["ranking"]
    evaluation_grounding = grounding["evaluation"]
    grounding_examples = [
        *ranking_grounding["examples"],
        *evaluation_grounding["examples"],
    ]
    value: dict[str, Any] = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "status": status,
        "variant_id": cfg["variant_id"],
        "reference_variant_id": cfg.get("reference_variant_id"),
        "model_family": cfg["model_family"],
        "claim_status": cfg["grounding_contract"]["required_claim_status"],
        "grounding_mode": cfg["grounding_mode"],
        "grounding_status": _aggregate_grounding_value(
            grounding_examples, "grounding_status"
        ),
        "grounding_resolution": _aggregate_grounding_value(
            grounding_examples, "grounding_resolution"
        ),
        "grounding_composition": {
            "ranking": ranking_grounding["composition"],
            "evaluation": evaluation_grounding["composition"],
        },
        "human_reviewed": cfg["grounding_contract"][
            "required_human_reviewed"
        ],
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_components": components,
        "ranking": {
            "mass_path": str(mass_path),
            "mass_ok_count": ok_mass,
            "expected_mass_count": cfg["expected_ranking_count"],
            "prefix_sizes": list(_PREFIX_SIZES),
            "score_kind": _RANKING_SCORE_KIND,
            "skip_early_layers": _SKIP_EARLY_LAYERS,
            "artifacts": {
                f"rank_n{ranking_n:03d}": {
                    "path": str(
                        mass_path.parent / f"rank_n{ranking_n:03d}.json"
                    ),
                    "fingerprint": artifact["fingerprint"],
                }
                for ranking_n, artifact in sorted(rankings.items())
            },
        },
        "steering": {
            "records_path": str(records_path),
            "record_ok_count": ok_records,
            "expected_record_count": int(
                components["input"]["evaluation_count"]
            )
            * len(_CONDITIONS),
            "condition_count": len(_CONDITIONS),
            "conditions": list(_CONDITIONS),
            "ranking_prefix_sizes": list(_PREFIX_SIZES),
            "top_k": list(_TOP_K_VALUES),
            "bias": _SWAP_BIAS,
            "scope": _QUERY_SCOPE,
            "controls": [],
        },
        "labels_opened": False,
        "protocol_addendum": components["input"]["protocol_addendum"],
    }
    review_contract = _propagate_review_contract(cfg)
    if review_contract:
        value.update(review_contract)
        value["reviewed_input_chain"] = dict(components["reviewed_input_chain"])
    checkpoint_verification = components["model"].get(
        "checkpoint_content_verification"
    )
    if checkpoint_verification is not None:
        value["checkpoint_content_verification"] = dict(checkpoint_verification)
    value["fingerprint"] = object_fingerprint(value)
    return value


def _write_manifest(
    path: Path,
    cfg: Mapping[str, Any],
    components: Mapping[str, Any],
    *,
    run_fingerprint: str,
    status: str,
    mass_path: Path,
    records_path: Path,
    latest_mass: Mapping[str, Mapping[str, Any]],
    latest_records: Mapping[tuple[str, str], Mapping[str, Any]],
    rankings: Mapping[int, Mapping[str, Any]],
) -> None:
    write_json(
        path,
        _manifest(
            cfg,
            components,
            run_fingerprint=run_fingerprint,
            status=status,
            mass_path=mass_path,
            records_path=records_path,
            latest_mass=latest_mass,
            latest_records=latest_records,
            rankings=rankings,
        ),
    )


def run_exploratory_matrix(
    config: dict[str, Any], *, retry_failed: bool = False
) -> Path:
    """Run or resume the frozen exploratory matrix and return its manifest."""
    cfg = _cfg(config)
    ranking_samples = _load_inputs(
        cfg["ranking_manifest"],
        source="ranking",
        expected_count=int(cfg["expected_ranking_count"]),
        ordered_ranking=True,
        expected_model_family=str(cfg["model_family"]),
        contract=cfg["grounding_contract"],
    )
    evaluation_samples = _load_inputs(
        cfg["evaluation_manifest"],
        source="evaluation",
        expected_count=cfg["expected_evaluation_count"],
        ordered_ranking=False,
        expected_model_family=str(cfg["model_family"]),
        contract=cfg["grounding_contract"],
    )
    reviewed_input_chain = None
    if cfg["grounding_mode"] == HUMAN_REVIEWED:
        reviewed_input_chain = _validate_reviewed_input_chain(
            cfg,
            ranking_samples,
            evaluation_samples,
        )
        provenance_key = str(reviewed_input_chain["provenance_key"])
        cfg[provenance_key] = dict(reviewed_input_chain[provenance_key])
        cfg["review_source_kind"] = reviewed_input_chain["review_source_kind"]
        if provenance_key == "tracking_review_provenance":
            cfg["target_grounding_scope"] = "terminal_only"
            cfg["control_region_policy"] = "none"
    checkpoint_content_verification = None
    if cfg["grounding_mode"] == HUMAN_REVIEWED:
        checkpoint_content_verification = verify_checkpoint_content_manifest(
            cfg["model_path"],
            cfg["checkpoint_content_manifest"],
        )
    components = _fingerprint_components(
        cfg,
        ranking_samples,
        evaluation_samples,
        reviewed_input_chain,
        checkpoint_content_verification,
    )
    run_fingerprint = object_fingerprint(components)

    output_dir = Path(cfg["output_dir"])
    ranking_dir = output_dir / "ranking"
    steering_dir = output_dir / "steering"
    mass_path = ranking_dir / "mass.jsonl"
    records_path = steering_dir / "records.jsonl"
    manifest_path = output_dir / "matrix_manifest.json"
    _validate_existing_manifest(manifest_path, run_fingerprint)

    ranking_ids = {str(sample["example_id"]) for sample in ranking_samples}
    evaluation_ids = {str(sample["example_id"]) for sample in evaluation_samples}
    latest_mass, mass_attempts = _load_latest_mass(
        mass_path,
        expected_ids=ranking_ids,
        run_fingerprint=run_fingerprint,
        retry_failed=retry_failed,
    )
    latest_records, record_attempts = _load_latest_records(
        records_path,
        expected_ids=evaluation_ids,
        run_fingerprint=run_fingerprint,
        retry_failed=retry_failed,
    )

    mass_complete = _all_mass_complete(latest_mass, ranking_ids)
    rankings: dict[int, dict[str, Any]] = {}
    if mass_complete:
        rankings = _derive_rankings(
            cfg,
            ranking_samples,
            latest_mass,
            ranking_dir=ranking_dir,
            run_fingerprint=run_fingerprint,
        )
        _validate_record_rankings(latest_records, rankings)
        if _all_steering_complete(latest_records, evaluation_ids):
            _write_manifest(
                manifest_path,
                cfg,
                components,
                run_fingerprint=run_fingerprint,
                status="complete",
                mass_path=mass_path,
                records_path=records_path,
                latest_mass=latest_mass,
                latest_records=latest_records,
                rankings=rankings,
            )
            return manifest_path

    _write_manifest(
        manifest_path,
        cfg,
        components,
        run_fingerprint=run_fingerprint,
        status="incomplete",
        mass_path=mass_path,
        records_path=records_path,
        latest_mass=latest_mass,
        latest_records=latest_records,
        rankings=rankings,
    )

    runtime_cfg = dict(cfg)
    runtime_cfg["capture_generation_attentions"] = _CAPTURE_GENERATION_ATTENTIONS
    runtime = causal_runner._runtime(runtime_cfg)
    _runtime_contract(runtime, cfg)

    for sample in ranking_samples:
        example_id = str(sample["example_id"])
        if latest_mass.get(example_id, {}).get("status") == "ok":
            continue
        attempt = int(mass_attempts[example_id]) + 1
        base = {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "variant_id": cfg["variant_id"],
            "reference_variant_id": cfg.get("reference_variant_id"),
            "model_family": cfg["model_family"],
            "example_id": example_id,
            "group_id": sample["group_id"],
            "task_id": sample["task_id"],
            "task_family": sample["task_family"],
            "partition": sample.get("partition"),
            "ranking_order": int(sample["ranking_order"]),
            "run_fingerprint": run_fingerprint,
            **_grounding_fields(sample),
            "attempt": attempt,
        }
        base.update(_propagate_review_contract(cfg))
        try:
            result = runtime.collect_mass(sample)
            if result.get("status", "ok") != "ok":
                raise RuntimeError(
                    f"collect_mass returned status={result.get('status')!r}"
                )
            row = {**result, **base, "status": "ok"}
            append_jsonl(mass_path, row)
        except Exception as exc:
            row = _append_invalid(mass_path, base, exc)
            latest_mass[example_id] = row
            raise
        latest_mass[example_id] = row
        mass_attempts[example_id] += 1

    rankings = _derive_rankings(
        cfg,
        ranking_samples,
        latest_mass,
        ranking_dir=ranking_dir,
        run_fingerprint=run_fingerprint,
    )
    _validate_record_rankings(latest_records, rankings)

    for sample in evaluation_samples:
        example_id = str(sample["example_id"])
        missing = [
            condition
            for condition in _CONDITIONS
            if latest_records.get((example_id, condition), {}).get("status") != "ok"
        ]
        if not missing:
            continue
        current_condition = missing[0]
        try:
            prepared, target, visual, visual_key = _generation_context(
                runtime, cfg, sample
            )
            if not target:
                raise ValueError("Target visual-token positions are empty")
            if not visual or not set(target) < set(visual):
                raise ValueError(
                    "Target positions must be a non-empty proper subset of "
                    "visual positions"
                )
        except Exception as exc:
            key = (example_id, current_condition)
            base = _base_record(
                sample,
                cfg,
                condition=current_condition,
                condition_kind=(
                    "baseline" if current_condition == _BASELINE else "candidate_target"
                ),
                run_fingerprint=run_fingerprint,
                attempt=int(record_attempts[key]) + 1,
            )
            row = _append_invalid(records_path, base, exc)
            latest_records[key] = row
            raise

        specs: list[tuple[str, str, int | None, int | None, list[Head], float, str | None]] = [
            (_BASELINE, "baseline", None, None, [], 0.0, None)
        ]
        for ranking_n, top_k, condition in _GRID:
            specs.append(
                (
                    condition,
                    "candidate_target",
                    ranking_n,
                    top_k,
                    _head_values(rankings[ranking_n], top_k),
                    _SWAP_BIAS,
                    str(rankings[ranking_n]["fingerprint"]),
                )
            )

        for (
            condition,
            condition_kind,
            ranking_n,
            top_k,
            heads,
            bias,
            ranking_fingerprint,
        ) in specs:
            key = (example_id, condition)
            if latest_records.get(key, {}).get("status") == "ok":
                continue
            base = _base_record(
                sample,
                cfg,
                condition=condition,
                condition_kind=condition_kind,
                run_fingerprint=run_fingerprint,
                attempt=int(record_attempts[key]) + 1,
            )
            try:
                generate_kwargs = {
                    "prepared": prepared,
                    "heads": heads,
                    "selected_positions": target,
                    visual_key: visual,
                    "bias": bias,
                    "query_scope": _QUERY_SCOPE,
                }
                result = runtime.generate(sample, **generate_kwargs)
                hook_assertion = (
                    _assert_hook_applied(result, heads)
                    if condition_kind == "candidate_target"
                    else None
                )
                row = {
                    **result,
                    **base,
                    "ranking_n": ranking_n,
                    "top_k": top_k,
                    "heads": [
                        {"layer": int(head.layer), "head": int(head.head)}
                        for head in heads
                    ],
                    "bias": float(bias),
                    "scope": _QUERY_SCOPE,
                    "ranking_fingerprint": ranking_fingerprint,
                    "target_positions": target,
                    "visual_positions": visual,
                    "hook_assertion": hook_assertion,
                    "status": "ok",
                }
                append_jsonl(records_path, row)
            except Exception as exc:
                invalid_base = {
                    **base,
                    "ranking_n": ranking_n,
                    "top_k": top_k,
                    "heads": [
                        {"layer": int(head.layer), "head": int(head.head)}
                        for head in heads
                    ],
                    "bias": float(bias),
                    "scope": _QUERY_SCOPE,
                    "ranking_fingerprint": ranking_fingerprint,
                }
                row = _append_invalid(records_path, invalid_base, exc)
                latest_records[key] = row
                raise
            latest_records[key] = row
            record_attempts[key] += 1

    if not _all_mass_complete(latest_mass, ranking_ids) or not _all_steering_complete(
        latest_records, evaluation_ids
    ):
        raise AssertionError("Exploratory matrix stopped without a complete latest state")
    _write_manifest(
        manifest_path,
        cfg,
        components,
        run_fingerprint=run_fingerprint,
        status="complete",
        mass_path=mass_path,
        records_path=records_path,
        latest_mass=latest_mass,
        latest_records=latest_records,
        rankings=rankings,
    )
    return manifest_path
