"""Frozen group-level discovery/validation/test partitions."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..config import section
from ..io import object_fingerprint, read_jsonl, sha256_file, write_json, write_jsonl
from .data import load_model_inputs


SPLIT_SCHEMA_VERSION = "my_dataset.group_split.v1"
PARTITIONS = ("discovery", "validation", "test")


def _rank(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).hexdigest()


def _counts(size: int, discovery_fraction: float, validation_fraction: float) -> tuple[int, int, int]:
    if size < 1:
        return 0, 0, 0
    if size == 1:
        return 0, 0, 1
    if size == 2:
        return 0, 1, 1
    discovery = max(1, round(size * discovery_fraction))
    validation = max(1, round(size * validation_fraction))
    while discovery + validation >= size:
        if discovery >= validation and discovery > 1:
            discovery -= 1
        elif validation > 1:
            validation -= 1
        else:
            break
    return discovery, validation, size - discovery - validation


def grouped_three_way_split(
    rows: list[dict[str, Any]],
    *,
    seed: int = 20260803,
    discovery_fraction: float = 0.2,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    if not 0 < discovery_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("discovery_fraction and validation_fraction must be in (0, 1)")
    if discovery_fraction + validation_fraction >= 1:
        raise ValueError("discovery_fraction + validation_fraction must be below 1")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    strata: dict[str, list[str]] = defaultdict(list)
    group_meta: dict[str, dict[str, str]] = {}
    media_to_group: dict[str, str] = {}
    for group_id, values in by_group.items():
        task_ids = {str(row["task_id"]) for row in values}
        media = {str(row["group_media_sha256"]) for row in values}
        if len(task_ids) != 1 or len(media) != 1:
            raise ValueError(f"Inconsistent task/media fields in group {group_id}")
        task_id = next(iter(task_ids))
        digest = next(iter(media))
        previous = media_to_group.setdefault(digest, group_id)
        if previous != group_id:
            raise ValueError(f"Media hash {digest} occurs in groups {previous} and {group_id}")
        strata[task_id].append(group_id)
        group_meta[group_id] = {"task_id": task_id, "group_media_sha256": digest}

    assignment: dict[str, str] = {}
    stratum_counts: dict[str, dict[str, int]] = {}
    for task_id, group_ids in sorted(strata.items()):
        ordered = sorted(group_ids, key=lambda value: (_rank(seed, value), value))
        discovery_n, validation_n, test_n = _counts(
            len(ordered), discovery_fraction, validation_fraction
        )
        slices = {
            "discovery": ordered[:discovery_n],
            "validation": ordered[discovery_n : discovery_n + validation_n],
            "test": ordered[discovery_n + validation_n :],
        }
        if len(slices["test"]) != test_n:
            raise AssertionError("Internal split allocation mismatch")
        stratum_counts[task_id] = {key: len(value) for key, value in slices.items()}
        for partition, values in slices.items():
            for group_id in values:
                assignment[group_id] = partition

    groups = {
        partition: sorted(group_id for group_id, value in assignment.items() if value == partition)
        for partition in PARTITIONS
    }
    examples = {
        partition: sorted(
            str(row["example_id"])
            for row in rows
            if assignment[str(row["group_id"])] == partition
        )
        for partition in PARTITIONS
    }
    media = {
        partition: sorted(group_meta[group_id]["group_media_sha256"] for group_id in groups[partition])
        for partition in PARTITIONS
    }
    if any(set(groups[a]) & set(groups[b]) for index, a in enumerate(PARTITIONS) for b in PARTITIONS[index + 1 :]):
        raise AssertionError("Group leakage across partitions")
    if any(set(media[a]) & set(media[b]) for index, a in enumerate(PARTITIONS) for b in PARTITIONS[index + 1 :]):
        raise AssertionError("Media leakage across partitions")
    result: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "method": "group_id_task_id_stratified_sha256_order",
        "fractions": {
            "discovery": discovery_fraction,
            "validation": validation_fraction,
            "test": 1 - discovery_fraction - validation_fraction,
        },
        "groups": groups,
        "examples": examples,
        "media_sha256": media,
        "stratum_group_counts": stratum_counts,
        "group_counts": {key: len(value) for key, value in groups.items()},
        "example_counts": {key: len(value) for key, value in examples.items()},
        "labels_opened": False,
    }
    result["fingerprint"] = object_fingerprint(result)
    return result


def build_split(config: dict[str, Any]) -> Path:
    cfg = section(config, "my_dataset_split")
    inputs_path = Path(cfg["inputs_path"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    rows = load_model_inputs(inputs_path)
    split = grouped_three_way_split(
        rows,
        seed=int(cfg.get("seed", 20260803)),
        discovery_fraction=float(cfg.get("discovery_fraction", 0.2)),
        validation_fraction=float(cfg.get("validation_fraction", 0.2)),
    )
    split["inputs_path"] = str(inputs_path)
    split["inputs_sha256"] = sha256_file(inputs_path)
    split["fingerprint"] = object_fingerprint(
        {key: value for key, value in split.items() if key != "fingerprint"}
    )
    path = output_dir / "split.json"
    write_json(path, split)
    assignments = []
    assignment_by_group = {
        group_id: partition
        for partition, group_ids in split["groups"].items()
        for group_id in group_ids
    }
    for row in sorted(rows, key=lambda value: value["example_id"]):
        assignments.append(
            {
                "schema_version": SPLIT_SCHEMA_VERSION,
                "example_id": str(row["example_id"]),
                "group_id": str(row["group_id"]),
                "group_media_sha256": str(row["group_media_sha256"]),
                "task_id": str(row["task_id"]),
                "partition": assignment_by_group[str(row["group_id"])],
            }
        )
    write_jsonl(output_dir / "assignments.jsonl", assignments)
    for partition in PARTITIONS:
        write_jsonl(
            output_dir / "model_inputs" / f"{partition}.jsonl",
            [row for row in rows if assignment_by_group[str(row["group_id"])] == partition],
        )
    write_json(
        output_dir / "audit.json",
        {
            "passed": True,
            "group_leakage": False,
            "media_leakage": False,
            "task_counts": dict(sorted(Counter(row["task_id"] for row in assignments).items())),
            "split_fingerprint": split["fingerprint"],
        },
    )
    return path

