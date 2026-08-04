"""Freeze a label-free, nested external cohort for attention-head ranking.

``ranking_data.jsonl`` contains successful rollouts and therefore also carries
label-like bookkeeping fields.  This module opens those fields only for an
up-front integrity check.  Cohort selection itself receives a deliberately
small, label-free candidate record built from the instruction, task and source
record identifier.  Media are never loaded from the source ``suc`` tree: the
source identifier and instruction are mapped to the anonymous IDs frozen by
``my_dataset.data`` and all model inputs are copied from the prepared attention
manifests.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import section
from ..io import object_fingerprint, read_jsonl, sha256_file, write_json, write_jsonl
from .attention_manifest import MODELS
from .data import FORBIDDEN_MODEL_FIELDS, _anonymous_id, _task_family, load_model_inputs
from .grounding_contract import HUMAN_REVIEWED, grounding_contract, validate_grounding_row
from .review_provenance import (
    load_fingerprinted_manifest,
    validate_attention_review_manifest,
    validate_jsonl_artifact,
    validate_review_provenance,
    validate_tracking_review_provenance,
)
from .roles import parse_instruction


RANKING_COHORT_SCHEMA = "my_dataset.external_ranking_cohort.v1"
SELECTION_SEED = "ljx_lfz-ranking-v1"
COHORT_SIZES = (5, 10, 20)
FROZEN_LJX_LFZ_S20 = (
    "suc/ljx_lfz_task_1_1/2", "suc/ljx_lfz_task_1_3/1",
    "suc/ljx_lfz_task_2_1/1", "suc/ljx_lfz_task_2_2/2",
    "suc/ljx_lfz_task_2_3/2", "suc/ljx_lfz_task_1_1/3",
    "suc/ljx_lfz_task_1_3/4", "suc/ljx_lfz_task_2_1/4",
    "suc/ljx_lfz_task_2_2/3", "suc/ljx_lfz_task_2_3/1",
    "suc/ljx_lfz_task_1_1/1", "suc/ljx_lfz_task_1_3/22",
    "suc/ljx_lfz_task_2_1/3", "suc/ljx_lfz_task_2_2/1",
    "suc/ljx_lfz_task_1_1/9", "suc/ljx_lfz_task_1_3/3",
    "suc/ljx_lfz_task_2_1/2", "suc/ljx_lfz_task_1_1/10",
    "suc/ljx_lfz_task_1_3/21", "suc/ljx_lfz_task_2_1/13",
)

# These keys must not enter an artifact consumed by a model.  The first group
# comes from data.py; the remaining aliases occur in scoring manifests.
_LABEL_KEYS = frozenset(FORBIDDEN_MODEL_FIELDS) | {
    "protocol_reward",
    "native_reward",
    "gold",
    "is_success",
    "success",
    "prediction_correct",
}


def _required_path(cfg: dict[str, Any], key: str, *aliases: str) -> Path:
    value = cfg.get(key)
    if value is None:
        for alias in aliases:
            value = cfg.get(alias)
            if value is not None:
                break
    if value is None:
        names = ", ".join((key, *aliases))
        raise ValueError(f"my_dataset_ranking_cohort requires one of: {names}")
    path = Path(str(value)).resolve()
    if key != "output_dir" and not path.exists():
        raise FileNotFoundError(path)
    return path


def _normalize_phrase(value: str) -> str:
    """Normalize a target phrase without consulting target-label fields."""
    return " ".join(value.casefold().strip().rstrip(".!?,;:").split())


def _validate_source_integrity(rows: list[dict[str, Any]], expected_count: int) -> None:
    """Validate the supplied successful-rollout file before label-free selection."""
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} ranking rows, found {len(rows)}")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("ranking_data IDs must be present and unique")
    for row in rows:
        row_id = str(row["id"])
        source_id = str(row.get("source_suc_id", ""))
        if str(row.get("split", "")) != "suc":
            raise ValueError(f"{row_id}: external ranking row is not split=suc")
        if row.get("instruction_video_match") is not True:
            raise ValueError(f"{row_id}: external ranking row is not a matched rollout")
        if row_id != source_id:
            raise ValueError(f"{row_id}: id and source_suc_id differ")
        target_value = row.get("target_obj")
        correct_value = row.get("correct_target_obj")
        target = _normalize_phrase(target_value) if isinstance(target_value, str) else ""
        correct = _normalize_phrase(correct_value) if isinstance(correct_value, str) else ""
        if not target or target != correct:
            raise ValueError(f"{row_id}: target_obj and correct_target_obj differ")
        if not str(row.get("task_id", "")).strip():
            raise ValueError(f"{row_id}: missing task_id")
        if not str(row.get("instruction", "")).strip():
            raise ValueError(f"{row_id}: missing instruction")


def _label_free_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
) -> list[dict[str, str]]:
    """Project source rows to the only fields visible to cohort selection."""
    candidates: list[dict[str, str]] = []
    for row in rows:
        source_id = str(row["id"])
        task_id = str(row["task_id"])
        instruction = str(row["instruction"]).strip()
        role = parse_instruction(instruction, _task_family(task_id))
        if role["parse_status"] != "parsed" or not role.get("target_phrase"):
            raise ValueError(f"{source_id}: instruction target phrase could not be parsed")
        target_phrase = _normalize_phrase(str(role["target_phrase"]))
        candidates.append(
            {
                "source_record_id": source_id,
                "task_id": task_id,
                "instruction": instruction,
                "normalized_target_phrase": target_phrase,
                "example_id": _anonymous_id(
                    f"{dataset_name}-e", source_id, instruction
                ),
                "group_id": _anonymous_id(f"{dataset_name}-g", source_id),
            }
        )
    return candidates


def _validate_parsed_targets(
    source_rows: list[dict[str, Any]], candidates: list[dict[str, str]]
) -> None:
    """Use source labels only to audit, never to rank, parsed target phrases."""
    for source, candidate in zip(source_rows, candidates, strict=True):
        integrity_target = _normalize_phrase(str(source["target_obj"]))
        if candidate["normalized_target_phrase"] != integrity_target:
            raise ValueError(
                f"{candidate['source_record_id']}: instruction-derived target phrase "
                f"{candidate['normalized_target_phrase']!r} differs from integrity target"
            )


def _stable_rank(seed: str, candidate: dict[str, str]) -> str:
    material = "\0".join((seed, candidate["source_record_id"]))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _select_nested(
    candidates: list[dict[str, str]],
    *,
    seed: str = SELECTION_SEED,
    maximum: int = max(COHORT_SIZES),
) -> list[dict[str, str]]:
    """Task-sorted round robin, preferring a target phrase not yet selected.

    Each task queue has a deterministic SHA-256 order.  At a task's turn the
    first candidate in that order whose instruction-derived target phrase has
    not appeared globally is chosen.  Once no fresh phrase remains for that
    task, its first remaining candidate is used.  The resulting single ordered
    list makes S5, S10 and S20 exact prefixes rather than three resamples.
    """
    if maximum < 1 or maximum > len(candidates):
        raise ValueError(
            f"Requested a maximum cohort of {maximum} from {len(candidates)} candidates"
        )
    queues: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        queues[candidate["task_id"]].append(candidate)
    for task_id, values in queues.items():
        values.sort(key=lambda value: (_stable_rank(seed, value), value["source_record_id"]))
        if not values:
            raise AssertionError(f"Internal empty ranking queue for {task_id}")

    selected: list[dict[str, str]] = []
    used_phrases: set[str] = set()
    task_ids = sorted(queues)
    while len(selected) < maximum:
        made_progress = False
        for task_id in task_ids:
            queue = queues[task_id]
            if not queue:
                continue
            fresh_index = next(
                (
                    index
                    for index, value in enumerate(queue)
                    if value["normalized_target_phrase"] not in used_phrases
                ),
                None,
            )
            choice = queue.pop(0 if fresh_index is None else fresh_index)
            selected.append(choice)
            used_phrases.add(choice["normalized_target_phrase"])
            made_progress = True
            if len(selected) == maximum:
                break
        if not made_progress:
            raise AssertionError("Ranking round robin exhausted before reaching maximum")
    return selected


def _index_prepared(
    inputs_path: Path,
    candidates: list[dict[str, str]],
    *,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    prepared = load_model_inputs(inputs_path)
    if len(prepared) != expected_count:
        raise ValueError(f"Expected {expected_count} prepared inputs, found {len(prepared)}")
    by_id = {str(row["example_id"]): row for row in prepared}
    for candidate in candidates:
        row = by_id.get(candidate["example_id"])
        if row is None:
            raise ValueError(
                f"{candidate['source_record_id']}: anonymous example is absent from prepared inputs"
            )
        expected = {
            "group_id": candidate["group_id"],
            "task_id": candidate["task_id"],
            "instruction": candidate["instruction"],
        }
        actual = {key: str(row.get(key, "")) for key in expected}
        if actual != expected:
            raise ValueError(
                f"{candidate['source_record_id']}: anonymous prepared mapping mismatch: "
                f"expected={expected}, actual={actual}"
            )
    return prepared, by_id


def _partition_index(
    split_path: Path,
    prepared: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    examples = split.get("examples")
    if not isinstance(examples, dict) or not examples:
        raise ValueError("White-box split must contain a non-empty examples mapping")
    partition_by_example: dict[str, str] = {}
    for partition, values in examples.items():
        if not isinstance(values, list):
            raise ValueError(f"split.examples.{partition} must be a list")
        for value in values:
            example_id = str(value)
            if example_id in partition_by_example:
                raise ValueError(f"Example occurs in multiple split partitions: {example_id}")
            partition_by_example[example_id] = str(partition)
    prepared_ids = {str(row["example_id"]) for row in prepared}
    if set(partition_by_example) != prepared_ids:
        missing = sorted(prepared_ids - set(partition_by_example))[:5]
        extra = sorted(set(partition_by_example) - prepared_ids)[:5]
        raise ValueError(f"White-box split and prepared inputs differ: missing={missing}, extra={extra}")

    partitions_by_group: dict[str, set[str]] = defaultdict(set)
    for row in prepared:
        partitions_by_group[str(row["group_id"])].add(
            partition_by_example[str(row["example_id"])]
        )
    mixed = {key: sorted(value) for key, value in partitions_by_group.items() if len(value) != 1}
    if mixed:
        raise ValueError(f"Groups span white-box partitions: {dict(list(mixed.items())[:5])}")
    partition_by_group = {key: next(iter(value)) for key, value in partitions_by_group.items()}
    return partition_by_example, partition_by_group, split


def _nested_label_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in _LABEL_KEYS:
                found.add(str(key))
            found.update(_nested_label_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_nested_label_keys(child))
    return found


def _review_provenance_key(manifest: Mapping[str, Any]) -> str:
    """Return the one reviewed-provenance field allowed by the parent manifest."""
    has_legacy = "review_provenance" in manifest
    has_tracking = "tracking_review_provenance" in manifest
    if has_legacy == has_tracking:
        raise ValueError(
            "reviewed attention manifest must contain exactly one provenance kind"
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


def _validate_tracking_row_contract(
    row: Mapping[str, Any], *, identity: str
) -> None:
    if "wrong_region_bbox" in row:
        raise ValueError(f"{identity}: tracking v2 forbids wrong_region_bbox")
    if row.get("target_grounding_scope") != "terminal_only":
        raise ValueError(f"{identity}: tracking v2 target scope must be terminal_only")
    if row.get("control_region_policy") != "none":
        raise ValueError(f"{identity}: tracking v2 control-region policy must be none")


def _load_attention_inputs(
    attention_inputs_dir: Path,
    prepared_ids: set[str],
    *,
    expected_count: int | None,
    filename: str = "all.jsonl",
    coverage: str = "full",
    required_grounding_mode: str | None = None,
    review_manifest: Mapping[str, Any] | None = None,
    review_provenance: Mapping[str, Any] | None = None,
    review_provenance_key: str | None = None,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, str | int]],
    set[str],
]:
    if coverage not in {"full", "subset"}:
        raise ValueError("attention_coverage must be 'full' or 'subset'")
    required_contract = (
        grounding_contract(required_grounding_mode)
        if required_grounding_mode is not None
        else None
    )
    required_review_provenance = None
    if required_contract is not None and required_contract["mode"] == HUMAN_REVIEWED:
        if (
            review_manifest is None
            or review_provenance is None
            or review_provenance_key is None
        ):
            raise ValueError(
                "human_reviewed attention inputs require their fingerprinted manifest"
            )
        required_review_provenance = _validate_review_provenance_value(
            review_provenance,
            key=review_provenance_key,
            identity="reviewed attention provenance",
        )
    indices: dict[str, dict[str, dict[str, Any]]] = {}
    artifacts: dict[str, dict[str, str | int]] = {}
    reference_ids: set[str] | None = None
    for model in MODELS:
        path = attention_inputs_dir / model / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = list(read_jsonl(path))
        if expected_count is not None and len(rows) != expected_count:
            raise ValueError(f"{model}: expected {expected_count} attention rows, found {len(rows)}")
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            example_id = str(row.get("example_id", ""))
            if not example_id or example_id in by_id:
                raise ValueError(f"{model}: missing or duplicate attention ID {example_id!r}")
            forbidden = _nested_label_keys(row)
            if forbidden:
                raise ValueError(
                    f"{model}/{example_id}: label fields in attention input: {sorted(forbidden)}"
                )
            family = row.get("model_family")
            if family is not None and str(family) != model:
                raise ValueError(
                    f"{model}/{example_id}: model_family is {family!r}, expected {model!r}"
                )
            if required_contract is not None:
                validate_grounding_row(
                    row,
                    identity=f"{model}/{example_id}",
                    contract=required_contract,
                )
            if required_review_provenance is not None:
                row_provenance = _validate_review_provenance_value(
                    row.get(review_provenance_key),
                    key=review_provenance_key,
                    identity=f"{model}/{example_id}.{review_provenance_key}",
                )
                if row_provenance != required_review_provenance:
                    raise ValueError(
                        f"{model}/{example_id}: reviewed provenance differs "
                        "from the attention manifest"
                    )
                other_key = (
                    "review_provenance"
                    if review_provenance_key == "tracking_review_provenance"
                    else "tracking_review_provenance"
                )
                if other_key in row:
                    raise ValueError(
                        f"{model}/{example_id}: reviewed row mixes provenance kinds"
                    )
                if review_provenance_key == "tracking_review_provenance":
                    _validate_tracking_row_contract(
                        row, identity=f"{model}/{example_id}"
                    )
            by_id[example_id] = row
        ids = set(by_id)
        if coverage == "full" and ids != prepared_ids:
            missing = sorted(prepared_ids - ids)[:5]
            extra = sorted(ids - prepared_ids)[:5]
            raise ValueError(
                f"{model}: attention/prepared ID mismatch: missing={missing}, extra={extra}"
            )
        if coverage == "subset" and not ids <= prepared_ids:
            extra = sorted(ids - prepared_ids)[:5]
            raise ValueError(
                f"{model}: reviewed attention contains IDs absent from prepared inputs: "
                f"{extra}"
            )
        if reference_ids is not None and ids != reference_ids:
            raise ValueError(f"{model}: attention ID set differs across models")
        reference_ids = ids
        indices[model] = by_id
        if required_review_provenance is not None:
            manifest_artifacts = review_manifest.get("artifacts")
            artifact_key = f"{model}/{Path(filename).stem}"
            artifact = (
                manifest_artifacts.get(artifact_key)
                if isinstance(manifest_artifacts, Mapping)
                else None
            )
            artifacts[model] = validate_jsonl_artifact(
                artifact,
                actual_path=path,
                rows=rows,
                identity=f"reviewed attention {artifact_key}",
            )
        else:
            artifacts[model] = {
                "path": str(path),
                "count": len(rows),
                "sha256": sha256_file(path),
                "fingerprint": object_fingerprint(rows),
            }
    if reference_ids is None:
        raise AssertionError("No attention models were loaded")
    return indices, artifacts, reference_ids


def _validate_group_closed_evaluation_population(
    prepared: Iterable[dict[str, Any]],
    evaluation_ids: set[str],
) -> None:
    """Reject evaluation subsets that contain only part of a source group."""
    if not evaluation_ids:
        raise ValueError("Evaluation attention population must not be empty")
    prepared_ids_by_group: dict[str, set[str]] = defaultdict(set)
    for row in prepared:
        prepared_ids_by_group[str(row["group_id"])].add(str(row["example_id"]))
    partial_groups = {
        group_id: {
            "included": sorted(expected_ids & evaluation_ids),
            "missing": sorted(expected_ids - evaluation_ids),
        }
        for group_id, expected_ids in prepared_ids_by_group.items()
        if expected_ids & evaluation_ids and not expected_ids <= evaluation_ids
    }
    if partial_groups:
        preview = dict(list(sorted(partial_groups.items()))[:5])
        raise ValueError(
            "Evaluation attention population is not group-closed relative to "
            f"prepared inputs: {preview}"
        )


def _cohort_stats(
    rows: Iterable[dict[str, Any]],
    partition_by_example: dict[str, str],
) -> dict[str, Any]:
    values = list(rows)
    groups = {str(row["group_id"]) for row in values}
    group_partitions = {
        str(row["group_id"]): partition_by_example[str(row["example_id"])]
        for row in values
    }
    return {
        "example_count": len(values),
        "group_count": len(groups),
        "partition_example_counts": dict(
            sorted(
                Counter(
                    partition_by_example[str(row["example_id"])] for row in values
                ).items()
            )
        ),
        "partition_group_counts": dict(
            sorted(Counter(group_partitions.values()).items())
        ),
        "task_example_counts": dict(
            sorted(Counter(str(row["task_id"]) for row in values).items())
        ),
    }


def _evaluation_cohorts(
    prepared: list[dict[str, Any]],
    selected: list[dict[str, str]],
    partition_by_example: dict[str, str],
) -> dict[str, Any]:
    selected_groups = {
        size: {row["group_id"] for row in selected[:size]} for size in COHORT_SIZES
    }
    common_unseen_groups = selected_groups[max(COHORT_SIZES)]
    result: dict[str, Any] = {
        "all": _cohort_stats(prepared, partition_by_example),
        "common_unseen_s20": _cohort_stats(
            [row for row in prepared if str(row["group_id"]) not in common_unseen_groups],
            partition_by_example,
        ),
        "ranking_source_only_s20": _cohort_stats(
            [row for row in prepared if str(row["group_id"]) in common_unseen_groups],
            partition_by_example,
        ),
        "by_ranking_size": {},
    }
    for size in COHORT_SIZES:
        groups = selected_groups[size]
        result["by_ranking_size"][str(size)] = {
            "n_specific_unseen": _cohort_stats(
                [row for row in prepared if str(row["group_id"]) not in groups],
                partition_by_example,
            ),
            "ranking_source_only": _cohort_stats(
                [row for row in prepared if str(row["group_id"]) in groups],
                partition_by_example,
            ),
        }
    return result


def freeze_ranking_cohort(config: dict[str, Any]) -> Path:
    """Freeze nested S5/S10/S20 ranking inputs for all three model families."""
    cfg = section(config, "my_dataset_ranking_cohort")
    ranking_data_path = _required_path(
        cfg, "ranking_data_path", "source_path", "ranking_data"
    )
    inputs_path = _required_path(cfg, "inputs_path", "prepared_inputs_path")
    split_path = _required_path(cfg, "split_path", "whitebox_split_path")
    attention_inputs_dir = _required_path(cfg, "attention_inputs_dir")
    output_dir = _required_path(cfg, "output_dir")
    seed = str(cfg.get("seed", SELECTION_SEED))
    if seed != SELECTION_SEED:
        raise ValueError(f"Ranking cohort seed is frozen to {SELECTION_SEED!r}, got {seed!r}")
    expected_source_count = int(cfg.get("expected_source_count", 30))
    expected_input_count = int(cfg.get("expected_input_count", 755))
    attention_coverage = str(cfg.get("attention_coverage", "full"))
    ranking_attention_filename = str(
        cfg.get("ranking_attention_filename", "all.jsonl")
    )
    evaluation_attention_filename = str(
        cfg.get("evaluation_attention_filename", ranking_attention_filename)
    )
    raw_attention_count = cfg.get(
        "expected_attention_count",
        expected_input_count if attention_coverage == "full" else None,
    )
    expected_attention_count = (
        None
        if raw_attention_count in (None, "", "auto")
        else int(raw_attention_count)
    )
    required_grounding_mode = cfg.get("required_grounding_mode")
    if required_grounding_mode is not None:
        required_grounding_mode = str(required_grounding_mode)

    attention_manifest_path: Path | None = None
    attention_manifest: dict[str, Any] | None = None
    review_provenance: dict[str, Any] | None = None
    review_provenance_key: str | None = None
    review_source_kind: str | None = None
    if required_grounding_mode == HUMAN_REVIEWED:
        attention_manifest_path = attention_inputs_dir / "manifest.json"
        attention_manifest = load_fingerprinted_manifest(
            attention_manifest_path,
            identity="reviewed attention manifest",
        )
        review_provenance = validate_attention_review_manifest(
            attention_manifest,
            identity="reviewed attention manifest",
        )
        review_provenance_key = _review_provenance_key(attention_manifest)
        review_source_kind = str(
            attention_manifest.get("review_source_kind")
            or "legacy_grounding_v1"
        )

    source_rows = list(read_jsonl(ranking_data_path))
    _validate_source_integrity(source_rows, expected_source_count)

    # Infer dataset_name from prepared data before deriving its anonymous IDs.
    prepared_probe = list(read_jsonl(inputs_path))
    dataset_names = {str(row.get("dataset_name", "")) for row in prepared_probe}
    if len(dataset_names) != 1 or not next(iter(dataset_names)):
        raise ValueError(f"Prepared inputs must contain one non-empty dataset_name: {dataset_names}")
    dataset_name = next(iter(dataset_names))
    candidates = _label_free_candidates(source_rows, dataset_name=dataset_name)
    _validate_parsed_targets(source_rows, candidates)
    if len({row["example_id"] for row in candidates}) != len(candidates):
        raise ValueError("Ranking source maps to duplicate anonymous example IDs")
    if len({row["group_id"] for row in candidates}) != len(candidates):
        raise ValueError("Ranking source maps to duplicate anonymous group IDs")

    prepared, prepared_by_id = _index_prepared(
        inputs_path, candidates, expected_count=expected_input_count
    )
    partition_by_example, partition_by_group, split = _partition_index(
        split_path, prepared
    )
    selected = _select_nested(candidates, seed=seed)
    enforce_reference = bool(
        cfg.get("enforce_frozen_s20", expected_source_count == 30)
    )
    selected_source_ids = tuple(row["source_record_id"] for row in selected)
    frozen_reference_match = selected_source_ids == FROZEN_LJX_LFZ_S20
    if enforce_reference and not frozen_reference_match:
        raise ValueError(
            "Frozen LJX/LFZ S20 regression mismatch: "
            f"expected={list(FROZEN_LJX_LFZ_S20)}, actual={list(selected_source_ids)}"
        )
    attention, ranking_input_artifacts, ranking_attention_ids = _load_attention_inputs(
        attention_inputs_dir,
        set(prepared_by_id),
        expected_count=expected_attention_count,
        filename=ranking_attention_filename,
        coverage=attention_coverage,
        required_grounding_mode=required_grounding_mode,
        review_manifest=attention_manifest,
        review_provenance=review_provenance,
        review_provenance_key=review_provenance_key,
    )
    if evaluation_attention_filename == ranking_attention_filename:
        evaluation_ids = ranking_attention_ids
        evaluation_input_artifacts = ranking_input_artifacts
    else:
        (
            _evaluation_attention,
            evaluation_input_artifacts,
            evaluation_ids,
        ) = _load_attention_inputs(
            attention_inputs_dir,
            set(prepared_by_id),
            expected_count=None,
            filename=evaluation_attention_filename,
            coverage="subset",
            required_grounding_mode=required_grounding_mode,
            review_manifest=attention_manifest,
            review_provenance=review_provenance,
            review_provenance_key=review_provenance_key,
        )
        if not evaluation_ids <= ranking_attention_ids:
            extra = sorted(evaluation_ids - ranking_attention_ids)[:5]
            raise ValueError(
                "Evaluation attention population is not a subset of ranking "
                f"attention inputs: {extra}"
            )
    _validate_group_closed_evaluation_population(prepared, evaluation_ids)
    missing_selected = [
        {
            "ranking_order": order,
            "example_id": candidate["example_id"],
            "source_record_id": candidate["source_record_id"],
            "group_id": candidate["group_id"],
        }
        for order, candidate in enumerate(selected, 1)
        if candidate["example_id"] not in ranking_attention_ids
    ]
    if missing_selected:
        raise ValueError(
            "Reviewed attention excludes frozen S20 ranking sources. "
            "Complete/correct their human grounding instead of silently "
            f"replacing the preregistered cohort: {missing_selected[:5]}"
        )
    evaluation_prepared = [
        row for row in prepared if str(row["example_id"]) in evaluation_ids
    ]

    cohorts: dict[str, Any] = {}
    for size in COHORT_SIZES:
        values = selected[:size]
        cohorts[str(size)] = {
            "size": size,
            "is_prefix_of_max20": values == selected[:size],
            "source_record_ids": [row["source_record_id"] for row in values],
            "example_ids": [row["example_id"] for row in values],
            "group_ids": [row["group_id"] for row in values],
            "task_counts": dict(sorted(Counter(row["task_id"] for row in values).items())),
            "target_phrase_counts": dict(
                sorted(Counter(row["normalized_target_phrase"] for row in values).items())
            ),
            "unique_target_phrases": len(
                {row["normalized_target_phrase"] for row in values}
            ),
            "whitebox_partition_counts": dict(
                sorted(Counter(partition_by_group[row["group_id"]] for row in values).items())
            ),
            "fingerprint": object_fingerprint(values),
        }
    overlap = {
        first: {
            second: len(
                set(cohorts[first]["group_ids"]) & set(cohorts[second]["group_ids"])
            )
            for second in map(str, COHORT_SIZES)
        }
        for first in map(str, COHORT_SIZES)
    }
    if not (
        cohorts["5"]["source_record_ids"] == cohorts["10"]["source_record_ids"][:5]
        and cohorts["10"]["source_record_ids"] == cohorts["20"]["source_record_ids"][:10]
    ):
        raise AssertionError("S5/S10/S20 ranking cohorts are not nested prefixes")

    output_artifacts: dict[str, Any] = {}
    source_sha = sha256_file(ranking_data_path)
    for model in MODELS:
        ordered_rows = []
        for order, candidate in enumerate(selected, 1):
            row = deepcopy(attention[model][candidate["example_id"]])
            prepared_row = prepared_by_id[candidate["example_id"]]
            if str(row.get("group_id", "")) != candidate["group_id"]:
                raise ValueError(f"{model}/{candidate['example_id']}: attention group mismatch")
            if str(row.get("task_id", "")) != candidate["task_id"]:
                raise ValueError(f"{model}/{candidate['example_id']}: attention task mismatch")
            row.update(
                {
                    "cohort_role": "external_ranking",
                    "ranking_order": order,
                    "ranking_cohort_sizes": [
                        size for size in COHORT_SIZES if order <= size
                    ],
                    "source_provenance": {
                        "kind": "external_ranking_data_record",
                        "source_file_sha256": source_sha,
                        "source_record_id": candidate["source_record_id"],
                        "anonymous_mapping": "my_dataset.data._anonymous_id",
                        "canonical_media_origin": "prepared_model_input",
                        "prepared_group_media_sha256": str(
                            prepared_row["group_media_sha256"]
                        ),
                    },
                }
            )
            forbidden = _nested_label_keys(row)
            if forbidden:
                raise AssertionError(
                    f"{model}/{candidate['example_id']}: output contains label fields {sorted(forbidden)}"
                )
            if review_provenance_key == "tracking_review_provenance":
                _validate_tracking_row_contract(
                    row, identity=f"{model}/{candidate['example_id']}"
                )
            ordered_rows.append(row)
        if len(ordered_rows) != max(COHORT_SIZES):
            raise AssertionError(f"{model}: ordered_max20 has the wrong length")
        if [int(row["ranking_order"]) for row in ordered_rows] != list(
            range(1, max(COHORT_SIZES) + 1)
        ):
            raise AssertionError(f"{model}: ranking_order is not contiguous")
        path = output_dir / model / "ordered_max20.jsonl"
        write_jsonl(path, ordered_rows)
        output_artifacts[model] = {
            "path": str(path),
            "count": len(ordered_rows),
            "sha256": sha256_file(path),
            "fingerprint": object_fingerprint(ordered_rows),
            "ordered_example_ids": [row["example_id"] for row in ordered_rows],
        }

    ordered_id_sets = {
        tuple(value["ordered_example_ids"]) for value in output_artifacts.values()
    }
    if len(ordered_id_sets) != 1:
        raise AssertionError("The three ordered model cohorts do not have identical IDs/order")

    selection_view = [
        {
            "ranking_order": index,
            **candidate,
            "whitebox_partition": partition_by_group[candidate["group_id"]],
        }
        for index, candidate in enumerate(selected, 1)
    ]
    contract = (
        grounding_contract(required_grounding_mode)
        if required_grounding_mode is not None
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": RANKING_COHORT_SCHEMA,
        "claim_status": (
            contract["required_claim_status"] if contract else "exploratory"
        ),
        "grounding_mode": (
            contract["mode"] if contract is not None else "not_enforced"
        ),
        "cohort_role": "external_ranking",
        "selection": {
            "seed": seed,
            "method": "task_sorted_round_robin_source_id_sha256_prefer_new_instruction_target",
            "within_task_order": "sha256(seed + NUL + source_record_id)",
            "target_phrase_source": "instruction_parser_only",
            "labels_used_for_selection": False,
            "nested_sizes": list(COHORT_SIZES),
            "ordered_max20": selection_view,
            "fingerprint": object_fingerprint(selection_view),
        },
        "source": {
            "path": str(ranking_data_path),
            "sha256": source_sha,
            "row_count": len(source_rows),
            "label_free_candidate_fingerprint": object_fingerprint(candidates),
            "integrity_checks": {
                "all_split_suc": True,
                "all_instruction_video_match": True,
                "all_id_equals_source_suc_id": True,
                "all_target_equals_correct_target": True,
            },
        },
        "prepared_inputs": {
            "path": str(inputs_path),
            "sha256": sha256_file(inputs_path),
            "fingerprint": object_fingerprint(prepared),
            "count": len(prepared),
        },
        "whitebox_split": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "fingerprint": split.get("fingerprint", object_fingerprint(split)),
        },
        "cohorts": cohorts,
        "cohort_group_overlap": overlap,
        "evaluation_cohorts": _evaluation_cohorts(
            evaluation_prepared, selected, partition_by_example
        ),
        "evaluation_population": {
            "attention_coverage": attention_coverage,
            "ranking_attention_filename": ranking_attention_filename,
            "evaluation_attention_filename": evaluation_attention_filename,
            "example_count": len(evaluation_ids),
            "example_ids": sorted(evaluation_ids),
            "fingerprint": object_fingerprint(sorted(evaluation_ids)),
            "is_strict_subset_of_prepared": evaluation_ids < set(prepared_by_id),
        },
        "attention_inputs": (
            ranking_input_artifacts
            if evaluation_attention_filename == ranking_attention_filename
            else {
                "ranking": ranking_input_artifacts,
                "evaluation": evaluation_input_artifacts,
            }
        ),
        "model_outputs": output_artifacts,
        "strict_audit": {
            "expected_source_count": expected_source_count,
            "expected_prepared_count": expected_input_count,
            "expected_attention_count_per_model": expected_attention_count,
            "actual_attention_count_per_model": len(evaluation_ids),
            "actual_ranking_attention_count_per_model": len(
                ranking_attention_ids
            ),
            "attention_coverage": attention_coverage,
            "required_grounding_mode": required_grounding_mode,
            "review_source_kind": review_source_kind,
            "all_s20_sources_grounded": True,
            "source_ids_unique": True,
            "anonymous_ids_unique": True,
            "prepared_split_ids_identical": True,
            "three_model_input_ids_identical": True,
            "three_model_output_order_identical": True,
            "nested_prefixes": True,
            "model_outputs_label_free": True,
            "frozen_s20_reference_enforced": enforce_reference,
            "frozen_s20_reference_match": frozen_reference_match,
            "source_suc_media_paths_opened": False,
        },
    }
    if review_provenance is not None:
        if (
            attention_manifest_path is None
            or attention_manifest is None
            or review_provenance_key is None
            or review_source_kind is None
        ):
            raise AssertionError("Reviewed ranking lost its attention manifest")
        manifest["review_source_kind"] = review_source_kind
        manifest[review_provenance_key] = review_provenance
        if review_provenance_key == "tracking_review_provenance":
            manifest["target_grounding_scope"] = "terminal_only"
            manifest["control_region_policy"] = "none"
        manifest["attention_manifest"] = {
            "path": str(attention_manifest_path.resolve()),
            "sha256": sha256_file(attention_manifest_path),
            "fingerprint": attention_manifest["fingerprint"],
        }
    # Exclude the final fingerprint field itself from the digest.
    manifest["fingerprint"] = object_fingerprint(manifest)
    manifest_path = output_dir / "selection_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path
