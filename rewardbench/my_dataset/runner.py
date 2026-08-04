"""Label-blind baseline runners for the isolated custom-dataset manifests."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from ..config import section
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    latest_by_id,
    provenance,
    read_jsonl,
    sha256_file,
    stable_shard,
    write_json,
)
from ..protocol import IMAGE_LABELS, parse_score, progress
from ..qwen_eval.protocols import ROBOREWARDBENCH_NATIVE
from ..roboreward_eval.runner import ROBOREWARD_PROMPT, parse_native_score
from ..schemas import SCHEMA_VERSION
from ..video import extract_endpoints
from .data import FORBIDDEN_MODEL_FIELDS, VIEW_NAMES, load_model_inputs


MODEL_FAMILIES = {"roboreward", "qwen", "grm"}
NATIVE_FRONT_PROTOCOL = "checkpoint_native_front_video_v1"
GRM_MULTIVIEW_PROTOCOL = "grm_native_three_view_endpoints_v1"


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = section(config, "my_dataset_eval")
    required = ("model_family", "inputs_path", "output_dir", "model_path")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"my_dataset_eval is missing required keys: {', '.join(missing)}")
    family = str(cfg["model_family"])
    if family not in MODEL_FAMILIES:
        raise ValueError(f"model_family must be one of {sorted(MODEL_FAMILIES)}, got {family!r}")
    expected_protocol = (
        GRM_MULTIVIEW_PROTOCOL if family == "grm" else NATIVE_FRONT_PROTOCOL
    )
    actual_protocol = str(cfg.get("input_protocol", expected_protocol))
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"{family} baseline requires input_protocol={expected_protocol!r}, "
            f"got {actual_protocol!r}"
        )
    return cfg


def _selected_inputs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_model_inputs(cfg["inputs_path"])
    shard_id = int(cfg.get("shard_id", 0))
    num_shards = int(cfg.get("num_shards", 1))
    if not 0 <= shard_id < num_shards:
        raise ValueError("my_dataset_eval.shard_id must be in [0, num_shards)")
    rows = [
        row
        for row in rows
        if stable_shard(str(row["group_media_sha256"]), num_shards) == shard_id
    ]
    limit_groups = int(cfg.get("limit_groups", 0))
    if limit_groups:
        group_ids = sorted({str(row["group_id"]) for row in rows})[:limit_groups]
        rows = [row for row in rows if str(row["group_id"]) in set(group_ids)]
    return rows


def _native_front_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "example_id": str(row["example_id"]),
        "group_id": str(row["group_id"]),
        "task": str(row["instruction"]),
        "video_path": str(Path(row["video_paths"]["front"]).resolve()),
        "video_sha256": str(row["view_sha256"]["front"]),
    }
    if FORBIDDEN_MODEL_FIELDS & payload.keys():
        raise AssertionError("Label field entered native model payload")
    return payload


def _load_cached_multiview_frames(
    row: dict[str, Any], frames_root: Path
) -> dict[str, dict[str, Any]]:
    group_id = str(row["group_id"])
    group_dir = frames_root / group_id
    cache_path = group_dir / "endpoints.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("group_media_sha256") == row["group_media_sha256"]
            and cached.get("view_sha256") == row["view_sha256"]
        ):
            records = cached.get("frames", {})
            paths = [
                value
                for record in records.values()
                for value in (record.get("first_path"), record.get("last_path"))
            ]
            if set(records) == set(VIEW_NAMES) and all(
                value and Path(str(value)).is_file() for value in paths
            ):
                return records

    records: dict[str, dict[str, Any]] = {}
    for view in VIEW_NAMES:
        record = extract_endpoints(
            f"{group_id}:{view}",
            str(row["view_sha256"][view]),
            str(row["video_paths"][view]),
            group_dir / view,
        )
        records[view] = record.to_dict()
    write_json(
        cache_path,
        {
            "group_id": group_id,
            "group_media_sha256": row["group_media_sha256"],
            "view_sha256": row["view_sha256"],
            "frames": records,
        },
    )
    return records


def _grm_multiview_payload(
    row: dict[str, Any], frames: dict[str, dict[str, Any]], cfg: dict[str, Any]
) -> dict[str, Any]:
    blank = Path(str(cfg["blank_goal"])).resolve()
    if not blank.is_file():
        raise FileNotFoundError(blank)
    front = frames["front"]
    left = frames["left_wrist"]
    right = frames["right_wrist"]
    images = [
        front["first_path"],
        str(blank),
        front["first_path"],
        left["first_path"],
        right["first_path"],
        front["last_path"],
        left["last_path"],
        right["last_path"],
    ]
    payload = {
        "protocol": GRM_MULTIVIEW_PROTOCOL,
        "example_id": str(row["example_id"]),
        "group_id": str(row["group_id"]),
        "task": str(row["instruction"]),
        "image": images,
        "image_labels": list(IMAGE_LABELS),
        "prompt_mode": str(cfg.get("prompt_mode", "official")),
    }
    if FORBIDDEN_MODEL_FIELDS & payload.keys():
        raise AssertionError("Label field entered GRM model payload")
    return payload


def _engine(cfg: dict[str, Any], dry_run: bool) -> Any:
    if dry_run:
        return None
    family = str(cfg["model_family"])
    if family == "roboreward":
        from ..roboreward_eval.runner import NativeRoboReward

        return NativeRoboReward(cfg)
    if family == "qwen":
        from ..qwen_eval.runner import Qwen3VLBaseline

        return Qwen3VLBaseline(cfg, ROBOREWARDBENCH_NATIVE)
    from ..raw_eval.runner import VLLMGRM

    return VLLMGRM(cfg)


def run_baseline(
    config: dict[str, Any], *, dry_run: bool = False, retry_failed: bool = False
) -> Path:
    """Run one model family without opening or recording the scoring labels."""
    cfg = _cfg(config)
    family = str(cfg["model_family"])
    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_id = int(cfg.get("shard_id", 0))
    num_shards = int(cfg.get("num_shards", 1))
    records_path = output_dir / f"records.shard-{shard_id:02d}.jsonl"
    previous = latest_by_id(read_jsonl(records_path)) if records_path.exists() else {}
    rows = _selected_inputs(cfg)

    manifest = provenance(sys.argv, config, Path(__file__).resolve().parents[2])
    manifest.update(
        {
            "model_family": family,
            "model_fingerprint": artifact_fingerprint(cfg["model_path"]),
            "inputs_path": str(Path(cfg["inputs_path"]).resolve()),
            "inputs_sha256": sha256_file(cfg["inputs_path"]),
            "labels_opened_by_inference": False,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "selected_input_count": len(rows),
            "selected_group_count": len({row["group_id"] for row in rows}),
            "input_protocol": str(cfg.get("input_protocol")),
            "source_fingerprints": {
                "rewardbench/my_dataset/data.py": sha256_file(
                    Path(__file__).resolve().parent / "data.py"
                ),
                "rewardbench/my_dataset/runner.py": sha256_file(Path(__file__).resolve()),
            },
        }
    )
    manifest_path = output_dir / (
        "manifest.json" if num_shards == 1 else f"manifest.shard-{shard_id:02d}.json"
    )
    write_json(manifest_path, manifest)
    engine = _engine(cfg, dry_run)
    frames_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for row in rows:
        example_id = str(row["example_id"])
        old = previous.get(example_id)
        if old and (
            old.get("status") == "ok"
            or (old.get("status") != "dry_run" and not retry_failed)
        ):
            continue
        attempt = int(old.get("attempt", 0)) + 1 if old else 1
        base = {
            "schema_version": SCHEMA_VERSION,
            "example_id": example_id,
            "group_id": row["group_id"],
            "group_media_sha256": row["group_media_sha256"],
            "task_id": row["task_id"],
            "task_family": row["task_family"],
            "model_family": family,
            "attempt": attempt,
        }
        try:
            if family in {"roboreward", "qwen"}:
                payload = _native_front_payload(row)
                if dry_run:
                    raw_output = "ANSWER: 1"
                    diagnostics = {"dry_run": True}
                elif family == "roboreward":
                    from ..roboreward_eval.runner import _use_checkpoint_native_video

                    frame_paths, video_record = _use_checkpoint_native_video(
                        payload["video_path"]
                    )
                    raw_output, diagnostics = engine.infer(
                        payload["task"], frame_paths, video_record
                    )
                    diagnostics = {**diagnostics, "video_record": video_record}
                else:
                    qwen_payload = {
                        "protocol": ROBOREWARDBENCH_NATIVE,
                        "task": payload["task"],
                        "video_path": payload["video_path"],
                        "prompt": ROBOREWARD_PROMPT.format(task=payload["task"]),
                    }
                    raw_output, diagnostics = engine.infer(qwen_payload)
                prediction = parse_native_score(raw_output)
                parsed = {
                    "native_prediction": prediction,
                    "progress": (prediction - 1) / 4,
                }
                protocol = NATIVE_FRONT_PROTOCOL
                frame_record = None
            else:
                group_id = str(row["group_id"])
                frames = frames_cache.get(group_id)
                if frames is None:
                    frames = _load_cached_multiview_frames(
                        row, output_dir / "frames" / "multiview_endpoints"
                    )
                    frames_cache[group_id] = frames
                payload = _grm_multiview_payload(row, frames, cfg)
                raw_output = "<score>0%</score>" if dry_run else engine.infer(payload)
                signed = parse_score(raw_output)
                parsed = {"signed_score": signed, "progress": progress(signed)}
                diagnostics = {
                    "dry_run": dry_run,
                    "image_count": len(payload["image"]),
                    "image_labels": payload["image_labels"],
                    "real_three_view_input": True,
                }
                protocol = GRM_MULTIVIEW_PROTOCOL
                frame_record = frames
            append_jsonl(
                records_path,
                {
                    **base,
                    "instruction": row["instruction"],
                    "protocol": protocol,
                    "raw_output": raw_output,
                    "input_diagnostics": diagnostics,
                    "frame_record": frame_record,
                    **parsed,
                    "status": "dry_run" if dry_run else "ok",
                },
            )
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    **base,
                    "protocol": str(cfg.get("input_protocol")),
                    "status": "invalid",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    return records_path
