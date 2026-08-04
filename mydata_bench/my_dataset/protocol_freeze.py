"""Machine-readable preregistration for the custom-dataset white-box track."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import load_config, section
from ..io import artifact_fingerprint, object_fingerprint, sha256_file, write_json


PROTOCOL_SCHEMA_VERSION = "my_dataset.protocol_freeze.v1"


def _file_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def freeze_protocol(config: dict[str, Any]) -> Path:
    cfg = section(config, "my_dataset_protocol")
    output_dir = Path(cfg["output_dir"]).resolve()
    inputs = _file_record(cfg["inputs_path"])
    split = _file_record(cfg["split_path"])
    roles = _file_record(cfg["roles_path"])
    split_value = json.loads(Path(split["path"]).read_text(encoding="utf-8"))
    models = {}
    for name, value in sorted(dict(cfg.get("models", {})).items()):
        if not isinstance(value, dict):
            raise ValueError(f"models.{name} must be a mapping")
        config_path = Path(value["config_path"]).resolve()
        model_config = load_config(config_path)
        section_value = model_config.get("my_dataset_eval", model_config)
        model_path = Path(str(value.get("model_path", section_value["model_path"]))).resolve()
        models[name] = {
            "config": _file_record(config_path),
            "model_path": str(model_path),
            "model_fingerprint": artifact_fingerprint(model_path),
            "input_protocol": section_value.get("input_protocol"),
            "content_order": section_value.get("content_order"),
            "prompt_mode": section_value.get("prompt_mode"),
            "decoding": {
                key: section_value.get(key)
                for key in ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens", "max_tokens")
                if key in section_value
            },
        }
    primary = dict(cfg.get("primary", {}))
    required_primary = {
        "ranking_query",
        "spatial_scope",
        "skip_early_layers",
        "top_k",
        "steering_query_scope",
        "validation_biases",
        "validation_top_k",
        "primary_metrics",
    }
    missing = sorted(required_primary - primary.keys())
    if missing:
        raise ValueError(f"my_dataset_protocol.primary is missing: {', '.join(missing)}")
    package_root = Path(__file__).resolve().parents[1]
    source_paths = [
        package_root / "my_dataset" / name
        for name in (
            "splits.py",
            "roles.py",
            "grounding_manifest.py",
            "attention_manifest.py",
            "causal_runner.py",
            "causal_metrics.py",
        )
    ] + [
        package_root / "qwen_eval" / "protocols.py",
        package_root / "qwen_eval" / "runner.py",
        package_root / "qwen_eval" / "attention.py",
        package_root / "attention_eval" / "runtime.py",
        package_root / "attention_eval" / "masking.py",
    ]
    source_files = {
        str(path.relative_to(package_root)): sha256_file(path) for path in source_paths
    }
    record: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "status": "frozen_before_attention_results",
        "inputs": inputs,
        "split": split,
        "roles": roles,
        "split_fingerprint": split_value.get("fingerprint"),
        "models": models,
        "primary": primary,
        "source_files": source_files,
        "label_access": {
            "inference": False,
            "ranking": False,
            "grounding": False,
            "steering": False,
            "scoring_only": True,
        },
        "claim_boundary": cfg.get(
            "claim_boundary",
            "exploratory_until at least 90 audited test groups and 15 task subsets",
        ),
    }
    record["fingerprint"] = object_fingerprint(record)
    path = output_dir / "preregistration.json"
    write_json(path, record)
    return path
