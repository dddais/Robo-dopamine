"""Prepare ljx/lfz-style same-video counterfactual groups without label leakage.

The model-facing artifact and the scoring labels deliberately live in separate
directories.  Every instruction variant points to the canonical raw videos,
not to the ``suc``/``fail`` copies in the generated dataset tree.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from ..config import section
from ..io import (
    object_fingerprint,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


PREPARED_SCHEMA_VERSION = "my_dataset.counterfactual.v1"
VIEW_NAMES = ("front", "left_wrist", "right_wrist")
FORBIDDEN_MODEL_FIELDS = {
    "reward",
    "gold_reward",
    "label",
    "match",
    "instruction_video_match",
    "correct_target_obj",
    "source_suc_id",
    "source_split",
}


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = section(config, "my_dataset")
    required = ("dataset_name", "source_root", "prepared_dir")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"my_dataset is missing required keys: {', '.join(missing)}")
    return cfg


def _anonymous_id(prefix: str, *values: str, length: int = 20) -> str:
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


def _task_family(task_id: str) -> str:
    prefix = task_id.split("_", 1)[0]
    return {
        "task1": "object_identity",
        "task2": "attribute_color",
        "task3": "ordinal_position",
        "task4": "left_right_relation",
        "task5": "distance_relation",
    }.get(prefix, "other")


def _source_metadata_path(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["source_root"]).resolve()
    value = cfg.get("source_metadata", root / "metadata.jsonl")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _canonical_view_paths(source: dict[str, Any], source_root: Path) -> dict[str, str]:
    raw_dir_value = source.get("source_raw_video_dir")
    if raw_dir_value:
        raw_dir = Path(str(raw_dir_value)).resolve()
        paths = [raw_dir / "faceImg.mp4", raw_dir / "leftImg.mp4", raw_dir / "rightImg.mp4"]
    else:
        paths = [(source_root / str(value)).resolve() for value in source["video_paths"]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {name: str(path) for name, path in zip(VIEW_NAMES, paths)}


def _source_copy_paths(row: dict[str, Any], source_root: Path) -> list[Path]:
    values = row.get("video_paths")
    if not isinstance(values, list) or len(values) != len(VIEW_NAMES):
        raise ValueError(f"{row.get('id')}: video_paths must contain three views")
    paths = [(source_root / str(value)).resolve() for value in values]
    for path in paths:
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"Video path escapes source root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _video_properties(path: str | Path) -> dict[str, int | float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    value = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    if value["width"] <= 0 or value["height"] <= 0:
        raise RuntimeError(f"Video reports an invalid resolution: {path}")
    if value["fps"] <= 0 or value["frame_count"] <= 0:
        raise RuntimeError(f"Video reports invalid timing metadata: {path}")
    return value


def prepare_dataset(config: dict[str, Any]) -> Path:
    """Freeze label-free model inputs and a separate post-inference label join."""
    cfg = _cfg(config)
    source_root = Path(cfg["source_root"]).resolve()
    prepared_dir = Path(cfg["prepared_dir"]).resolve()
    dataset_name = str(cfg["dataset_name"])
    rows = list(read_jsonl(_source_metadata_path(cfg)))
    if not rows:
        raise ValueError("Source metadata is empty")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_source_ids: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_suc_id", ""))
        if not source_id:
            raise ValueError(f"Missing source_suc_id in row {row.get('id')}")
        grouped[source_id].append(row)
        seen_source_ids.add(source_id)

    expected_groups = int(cfg.get("expected_groups", 0))
    expected_examples = int(cfg.get("expected_examples", 0))
    if expected_groups and len(grouped) != expected_groups:
        raise ValueError(f"Expected {expected_groups} groups, found {len(grouped)}")
    if expected_examples and len(rows) != expected_examples:
        raise ValueError(f"Expected {expected_examples} examples, found {len(rows)}")

    model_inputs: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    media_audit: list[dict[str, Any]] = []
    byte_mismatches: list[dict[str, Any]] = []
    duplicate_instructions: list[dict[str, Any]] = []
    verify_copies = bool(cfg.get("verify_counterfactual_bytes", True))
    audit_video_metadata = bool(cfg.get("audit_video_metadata", True))
    hashed_bytes = 0

    for source_id in sorted(grouped):
        variants = grouped[source_id]
        originals = [row for row in variants if row.get("split") == "suc"]
        if len(originals) != 1:
            raise ValueError(
                f"Expected exactly one matched source row for {source_id}, found {len(originals)}"
            )
        source = originals[0]
        task_ids = {str(row.get("task_id")) for row in variants}
        if len(task_ids) != 1:
            raise ValueError(f"Task IDs vary within source group {source_id}: {sorted(task_ids)}")
        task_id = next(iter(task_ids))
        instructions = [str(row.get("instruction", "")) for row in variants]
        repeated = [value for value, count in Counter(instructions).items() if count > 1]
        if repeated:
            duplicate_instructions.append({"source_id": source_id, "instructions": repeated})
            continue

        group_id = _anonymous_id(f"{dataset_name}-g", source_id)
        canonical_paths = _canonical_view_paths(source, source_root)
        canonical_hashes = {
            name: sha256_file(path) for name, path in canonical_paths.items()
        }
        hashed_bytes += sum(Path(path).stat().st_size for path in canonical_paths.values())
        group_media_sha256 = object_fingerprint(
            [(name, canonical_hashes[name]) for name in VIEW_NAMES]
        )

        properties = {
            name: _video_properties(path) for name, path in canonical_paths.items()
        } if audit_video_metadata else {}
        if properties:
            timing = {(item["fps"], item["frame_count"]) for item in properties.values()}
            if len(timing) != 1:
                raise ValueError(f"The three views are not synchronized for {source_id}")
        media_audit.append(
            {
                "group_id": group_id,
                "group_media_sha256": group_media_sha256,
                "view_sha256": canonical_hashes,
                "video_properties": properties,
            }
        )

        for row in variants:
            source_copy_paths = _source_copy_paths(row, source_root)
            if verify_copies:
                for view_name, copy_path in zip(VIEW_NAMES, source_copy_paths):
                    copy_digest = sha256_file(copy_path)
                    hashed_bytes += copy_path.stat().st_size
                    if copy_digest != canonical_hashes[view_name]:
                        byte_mismatches.append(
                            {
                                "source_id": source_id,
                                "row_id": row.get("id"),
                                "view": view_name,
                                "canonical_sha256": canonical_hashes[view_name],
                                "copy_sha256": copy_digest,
                            }
                        )
            instruction = str(row.get("instruction", "")).strip()
            if not instruction:
                raise ValueError(f"Empty instruction in row {row.get('id')}")
            example_id = _anonymous_id(
                f"{dataset_name}-e", source_id, instruction
            )
            model_inputs.append(
                {
                    "schema_version": PREPARED_SCHEMA_VERSION,
                    "dataset_name": dataset_name,
                    "example_id": example_id,
                    "group_id": group_id,
                    "task_id": task_id,
                    "task_family": _task_family(task_id),
                    "instruction": instruction,
                    "evaluation_split": str(cfg.get("evaluation_split", "external_test")),
                    "video_paths": canonical_paths,
                    "view_sha256": canonical_hashes,
                    "group_media_sha256": group_media_sha256,
                }
            )
            matched = bool(row.get("instruction_video_match"))
            source_split = str(row.get("split"))
            if matched != (source_split == "suc"):
                raise ValueError(
                    f"Inconsistent split/match fields for source row {row.get('id')}"
                )
            labels.append(
                {
                    "schema_version": PREPARED_SCHEMA_VERSION,
                    "dataset_name": dataset_name,
                    "example_id": example_id,
                    "group_id": group_id,
                    "task_id": task_id,
                    "task_family": _task_family(task_id),
                    "instruction_video_match": matched,
                    "protocol_reward": 5 if matched else 1,
                    "source_row_id": str(row.get("id")),
                    "source_group_id": source_id,
                    "target_obj": row.get("target_obj"),
                    "correct_target_obj": row.get("correct_target_obj"),
                }
            )

    if duplicate_instructions:
        raise ValueError(f"Duplicate instructions within source groups: {duplicate_instructions}")
    if byte_mismatches:
        raise ValueError(f"Counterfactual video byte mismatches: {byte_mismatches[:10]}")

    model_inputs.sort(key=lambda row: str(row["example_id"]))
    labels.sort(key=lambda row: str(row["example_id"]))
    if len({row["example_id"] for row in model_inputs}) != len(model_inputs):
        raise ValueError("Anonymous example IDs are not unique")
    if {row["example_id"] for row in model_inputs} != {row["example_id"] for row in labels}:
        raise AssertionError("Model-input and label IDs differ")

    inputs_dir = prepared_dir / "model_inputs"
    scoring_dir = prepared_dir / "scoring"
    inputs_path = inputs_dir / "inputs.jsonl"
    labels_path = scoring_dir / "labels.jsonl"
    write_jsonl(inputs_path, model_inputs)
    write_jsonl(labels_path, labels)
    write_jsonl(prepared_dir / "media_audit.jsonl", media_audit)
    write_json(
        inputs_dir / "manifest.json",
        {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "evaluation_split": str(cfg.get("evaluation_split", "external_test")),
            "num_examples": len(model_inputs),
            "num_groups": len({row["group_id"] for row in model_inputs}),
            "task_counts": dict(sorted(Counter(row["task_id"] for row in model_inputs).items())),
            "inputs_sha256": sha256_file(inputs_path),
            "model_fields": sorted(model_inputs[0]),
            "label_file_not_required_for_inference": True,
        },
    )
    write_json(
        prepared_dir / "prepare_manifest.json",
        {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "source_metadata": str(_source_metadata_path(cfg)),
            "source_metadata_sha256": sha256_file(_source_metadata_path(cfg)),
            "inputs_path": str(inputs_path),
            "inputs_sha256": sha256_file(inputs_path),
            "labels_path": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "num_examples": len(model_inputs),
            "num_groups": len(grouped),
            "num_matched": sum(bool(row["instruction_video_match"]) for row in labels),
            "num_mismatched": sum(not bool(row["instruction_video_match"]) for row in labels),
            "hashed_bytes": hashed_bytes,
            "counterfactual_byte_mismatches": 0,
        },
    )
    audit = audit_prepared(inputs_path, labels_path)
    write_json(prepared_dir / "audit.json", audit)
    return prepared_dir


def load_model_inputs(path: str | Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Model input manifest is empty: {path}")
    seen: set[str] = set()
    for row in rows:
        forbidden = FORBIDDEN_MODEL_FIELDS & row.keys()
        if forbidden:
            raise ValueError(f"Label fields in model input {row.get('example_id')}: {sorted(forbidden)}")
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in seen:
            raise ValueError(f"Missing or duplicate model-input ID: {example_id!r}")
        seen.add(example_id)
        paths = row.get("video_paths")
        if not isinstance(paths, dict) or set(paths) != set(VIEW_NAMES):
            raise ValueError(f"{example_id}: expected exactly {VIEW_NAMES} video paths")
        for value in paths.values():
            if not Path(str(value)).is_file():
                raise FileNotFoundError(value)
    return rows


def load_labels(path: str | Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Label manifest is empty: {path}")
    seen: set[str] = set()
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in seen:
            raise ValueError(f"Missing or duplicate label ID: {example_id!r}")
        seen.add(example_id)
        matched = bool(row.get("instruction_video_match"))
        expected = 5 if matched else 1
        if int(row.get("protocol_reward", 0)) != expected:
            raise ValueError(f"Inconsistent protocol reward for {example_id}")
    return rows


def audit_prepared(inputs_path: str | Path, labels_path: str | Path) -> dict[str, Any]:
    inputs = load_model_inputs(inputs_path)
    labels = load_labels(labels_path)
    input_ids = {str(row["example_id"]) for row in inputs}
    label_ids = {str(row["example_id"]) for row in labels}
    label_coded_paths = []
    for row in inputs:
        for view, value in row["video_paths"].items():
            parts = {part.lower() for part in Path(str(value)).parts}
            if parts & {"suc", "fail", "success", "failure"}:
                label_coded_paths.append(
                    {"example_id": row["example_id"], "view": view, "path": value}
                )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        groups[str(row["group_id"])].append(row)
    invalid_group_labels = {
        group_id: {
            "matched": sum(bool(row["instruction_video_match"]) for row in values),
            "mismatched": sum(not bool(row["instruction_video_match"]) for row in values),
        }
        for group_id, values in groups.items()
        if sum(bool(row["instruction_video_match"]) for row in values) != 1
        or not any(not bool(row["instruction_video_match"]) for row in values)
    }
    result = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "inputs_path": str(Path(inputs_path).resolve()),
        "labels_path": str(Path(labels_path).resolve()),
        "num_model_inputs": len(inputs),
        "num_labels": len(labels),
        "num_groups": len(groups),
        "model_label_fields_found": 0,
        "label_coded_model_paths": label_coded_paths,
        "missing_label_ids": sorted(input_ids - label_ids),
        "labels_without_inputs": sorted(label_ids - input_ids),
        "invalid_group_labels": invalid_group_labels,
    }
    result["passed"] = bool(
        input_ids == label_ids
        and not label_coded_paths
        and not invalid_group_labels
    )
    return result
