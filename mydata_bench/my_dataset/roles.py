"""Instruction-side semantic roles for the LJX/LFZ counterfactual dataset.

This module only reads the label-free model input manifest.  The produced role
manifest is annotation-side metadata for grounding and never exposes whether an
instruction matches its video.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import section
from ..io import object_fingerprint, sha256_file, write_json, write_jsonl
from .data import FORBIDDEN_MODEL_FIELDS, load_model_inputs


ROLE_SCHEMA_VERSION = "my_dataset.semantic_roles.v1"

_SIMPLE = re.compile(
    r"^Pick up the (?P<object>.+?) and place it in the (?P<destination>.+?)\.?$",
    re.IGNORECASE,
)
_ORDINAL = re.compile(
    r"^Pick up the (?P<ordinal>first|second|third|fourth) "
    r"(?P<object>.+?) from the (?P<direction>left|right) and place it in the "
    r"(?P<destination>.+?)\.?$",
    re.IGNORECASE,
)
_RELATION = re.compile(
    r"^Pick up the (?P<object>.+?) to the (?P<relation>left|right) of the "
    r"(?P<reference>.+?) and place it in the (?P<destination>.+?)\.?$",
    re.IGNORECASE,
)
_DISTANCE = re.compile(
    r"^Pick up the (?P<object>.+?) (?P<relation>closest to|farthest from) the "
    r"(?P<reference>.+?) and place it in the (?P<destination>.+?)\.?$",
    re.IGNORECASE,
)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.lower().strip().split())


def parse_instruction(instruction: str, task_family: str) -> dict[str, Any]:
    """Parse the frozen task templates into explicit semantic roles."""
    match = None
    strategy = task_family
    if task_family == "ordinal_position":
        match = _ORDINAL.match(instruction)
    elif task_family == "left_right_relation":
        match = _RELATION.match(instruction)
    elif task_family == "distance_relation":
        match = _DISTANCE.match(instruction)
    else:
        match = _SIMPLE.match(instruction)
    if match is None:
        return {
            "grounding_strategy": "manual_unparsed",
            "manipulated_object": None,
            "attribute": None,
            "reference_object": None,
            "destination": None,
            "relation": None,
            "ordinal": None,
            "direction": None,
            "target_phrase": instruction,
            "target_instance": instruction,
            "requires_instance_review": True,
            "parse_status": "unparsed",
        }

    values = {key: _clean(value) for key, value in match.groupdict().items()}
    surface_object = values.get("object")
    attribute = None
    manipulated = surface_object
    if task_family == "attribute_color" and surface_object and " " in surface_object:
        attribute, manipulated = surface_object.split(" ", 1)
    relation = values.get("relation")
    ordinal = values.get("ordinal")
    direction = values.get("direction")
    reference = values.get("reference")
    if ordinal:
        target_instance = f"{ordinal} {manipulated} from the {direction}"
    elif relation:
        connector = "of" if relation in {"left", "right"} else ""
        target_instance = " ".join(
            value for value in (manipulated, relation, connector, reference) if value
        )
    else:
        target_instance = surface_object
    requires_review = task_family in {
        "ordinal_position",
        "left_right_relation",
        "distance_relation",
    }
    return {
        "grounding_strategy": strategy,
        "manipulated_object": manipulated,
        "attribute": attribute,
        "reference_object": reference,
        "destination": values.get("destination"),
        "relation": relation,
        "ordinal": ordinal,
        "direction": direction,
        "target_phrase": surface_object,
        "target_instance": target_instance,
        "requires_instance_review": requires_review,
        "parse_status": "parsed",
    }


def build_roles(config: dict[str, Any]) -> Path:
    cfg = section(config, "my_dataset_roles")
    inputs_path = Path(cfg["inputs_path"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    rows = load_model_inputs(inputs_path)
    roles = []
    for row in rows:
        role = {
            "schema_version": ROLE_SCHEMA_VERSION,
            "example_id": str(row["example_id"]),
            "group_id": str(row["group_id"]),
            "task_id": str(row["task_id"]),
            "task_family": str(row["task_family"]),
            "instruction": str(row["instruction"]),
            **parse_instruction(str(row["instruction"]), str(row["task_family"])),
        }
        if FORBIDDEN_MODEL_FIELDS & role.keys():
            raise AssertionError("Label field entered semantic role manifest")
        roles.append(role)
    roles.sort(key=lambda row: row["example_id"])
    path = output_dir / "roles.jsonl"
    write_jsonl(path, roles)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": ROLE_SCHEMA_VERSION,
            "inputs_path": str(inputs_path),
            "inputs_sha256": sha256_file(inputs_path),
            "roles_path": str(path),
            "roles_fingerprint": object_fingerprint(roles),
            "num_examples": len(roles),
            "parse_status": dict(sorted(Counter(row["parse_status"] for row in roles).items())),
            "grounding_strategies": dict(
                sorted(Counter(row["grounding_strategy"] for row in roles).items())
            ),
            "labels_opened": False,
        },
    )
    return path
