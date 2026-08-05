"""Auditable first-frame grounding followed by official SAM3 video tracking.

The contract in this module is intentionally fail closed.  Text queries are
used only to propose target/reference instances on the canonical first frame.
Every accepted proposal is then converted to a visual bounding-box prompt and
the locked SAM3 object id is propagated through the complete front-view video.
No frame is independently re-detected and no text-only tracking is allowed.
"""

from __future__ import annotations

import copy
import gc
import importlib
import json
import math
import sys
from collections import Counter
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from ..config import section
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    object_fingerprint,
    read_jsonl,
    sha256_file,
    stable_shard,
    write_json,
    write_jsonl,
)
from .data import load_model_inputs


TRACKED_GROUNDING_REQUEST_SCHEMA = "mydata_bench.tracked_grounding.request.v2"
TRACKED_GROUNDING_PROPOSAL_SCHEMA = "mydata_bench.tracked_grounding.proposal.v2"
TRACKED_GROUNDING_TRACK_SCHEMA = "mydata_bench.tracked_grounding.track.v2"
TRACKED_GROUNDING_ARTIFACT_SCHEMA = "mydata_bench.tracked_grounding.artifact.v2"
TRACKED_GROUNDING_MANUAL_ANCHOR_SCHEMA = (
    "mydata_bench.tracked_grounding.manual_anchor.v2"
)
TRACKED_GROUNDING_MANIFEST_SCHEMA = "mydata_bench.tracked_grounding.manifest.v2"
TRACKED_GROUNDING_CACHE_SCHEMA = "mydata_bench.tracked_grounding.cache.v2"
PROCESSOR_CONTENT_ORDER_CONTRACT_SCHEMA = (
    "my_dataset.processor_content_order_contract.v1"
)
ROBOREWARD_CONTENT_ORDERS = ("text_then_video", "video_then_text")

MODELS = ("roboreward", "qwen", "grm")
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def derive_manual_anchor_id(
    example_id: str, image_sha256: str, bbox: Sequence[float]
) -> str:
    return "manual-" + object_fingerprint(
        {
            "example_id": str(example_id),
            "image_sha256": str(image_sha256),
            "bbox": [float(value) for value in bbox],
        }
    )[:24]


def _without(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(key, None)
    return result


def _fingerprint_row(value: Mapping[str, Any], key: str = "fingerprint") -> str:
    return object_fingerprint(_without(value, key))


def _orchestrator_provenance() -> dict[str, str]:
    """Bind cached/proposed tracks to the local orchestration implementation."""

    path = Path(__file__).resolve()
    return {
        "orchestrator_source_path": str(path),
        "orchestrator_source_sha256": sha256_file(path),
    }


def _cfg(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = section(config, "my_dataset_tracked_grounding")
    required = ("inputs_path", "roles_path", "split_path", "output_dir", "baseline_runs")
    missing = [name for name in required if not cfg.get(name)]
    if missing:
        raise ValueError(
            "my_dataset_tracked_grounding is missing required keys: "
            + ", ".join(missing)
        )
    runs = cfg.get("baseline_runs")
    if not isinstance(runs, dict) or set(runs) != set(MODELS):
        raise ValueError(f"baseline_runs must contain exactly {MODELS}")
    nested = cfg.get("sam3")
    if nested is None:
        nested = config.get("sam3", {})
    if not isinstance(nested, dict):
        raise ValueError("sam3 configuration must be a mapping")
    return cfg, nested


def _resolved_file(path: str | Path, what: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{what}: {result}")
    return result


def _image_size(path: str | Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid image dimensions: {path}")
    return int(width), int(height)


def _record_files(run: str | Path) -> list[Path]:
    path = Path(run).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("records*.jsonl"))
    if not files:
        files = sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No baseline record JSONL found under {path}")
    return files


def _load_baseline_run(run: str | Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    file_hashes: dict[Path, str] = {}
    for path in _record_files(run):
        digest = file_hashes.setdefault(path, sha256_file(path))
        for row in read_jsonl(path):
            example_id = str(row.get("example_id", ""))
            if not example_id:
                raise ValueError(f"Missing example_id in {path}")
            wrapped = {
                "row": row,
                "path": str(path),
                "file_sha256": digest,
            }
            old = latest.get(example_id)
            attempt = int(row.get("attempt", 0))
            if old is not None and attempt == int(old["row"].get("attempt", 0)):
                raise ValueError(
                    f"Duplicate latest baseline attempt for {example_id} in {path}"
                )
            if old is None or attempt > int(old["row"].get("attempt", 0)):
                latest[example_id] = wrapped
    return latest


def _load_content_order_run(
    run: str | Path,
    content_order: str,
    *,
    inputs_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    run_dir = Path(run).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest_path = _resolved_file(run_dir / "manifest.json", "baseline manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        raise ValueError(f"{content_order}: malformed baseline manifest")
    config_fingerprint = str(manifest.get("config_fingerprint", ""))
    if config_fingerprint != object_fingerprint(manifest["config"]):
        raise ValueError(f"{content_order}: baseline config fingerprint is invalid")
    evaluation = manifest["config"].get("my_dataset_eval")
    if not isinstance(evaluation, dict):
        raise ValueError(f"{content_order}: missing my_dataset_eval")
    declared_order = str(
        manifest.get("content_order", evaluation.get("content_order", ""))
    )
    expected_inputs_sha = sha256_file(inputs_path)
    expected = {
        "model_family": "roboreward",
        "inputs_sha256": expected_inputs_sha,
        "selected_input_count": expected_count,
        "labels_opened_by_inference": False,
    }
    mismatch = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise ValueError(
            f"{content_order}: baseline manifest contract differs: {mismatch}"
        )
    if declared_order != content_order:
        raise ValueError(
            f"{content_order}: baseline declares content_order={declared_order!r}"
        )
    if str(evaluation.get("model_family")) != "roboreward":
        raise ValueError(f"{content_order}: evaluation model_family is not roboreward")
    if Path(str(evaluation.get("inputs_path", ""))).resolve() != inputs_path:
        raise ValueError(f"{content_order}: evaluation inputs_path differs")
    files = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _record_files(run_dir)
    ]
    rows = _load_baseline_run(run_dir)
    if len(rows) != expected_count:
        raise ValueError(
            f"{content_order}: expected {expected_count} latest rows, found {len(rows)}"
        )
    provenance = {
        "run_dir": str(run_dir),
        "content_order": content_order,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_config_fingerprint": config_fingerprint,
        "record_files": files,
        "record_files_fingerprint": object_fingerprint(files),
    }
    return {"rows": rows, "provenance": provenance}


def _load_roboreward_content_order_runs(
    cfg: Mapping[str, Any],
    *,
    inputs_path: Path,
    expected_count: int,
) -> dict[str, dict[str, Any]]:
    configured = cfg.get("roboreward_content_order_runs")
    if configured is None:
        return {}
    if not isinstance(configured, Mapping) or set(configured) != set(
        ROBOREWARD_CONTENT_ORDERS
    ):
        raise ValueError(
            "roboreward_content_order_runs must contain exactly "
            f"{ROBOREWARD_CONTENT_ORDERS}"
        )
    canonical = Path(str(cfg["baseline_runs"]["roboreward"])).resolve()
    configured_paths = {
        Path(str(value)).expanduser().resolve() for value in configured.values()
    }
    if canonical not in configured_paths:
        raise ValueError(
            "baseline_runs.roboreward must equal one frozen content-order run"
        )
    return {
        order: _load_content_order_run(
            configured[order],
            order,
            inputs_path=inputs_path,
            expected_count=expected_count,
        )
        for order in ROBOREWARD_CONTENT_ORDERS
    }


def _role_rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in result:
            raise ValueError(f"Missing or duplicate role example_id: {example_id!r}")
        result[example_id] = row
    return result


def _finite_int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"{label} contains a non-integer frame index")
        result.append(int(item))
    if result != sorted(set(result)) or result[0] != 0:
        raise ValueError(f"{label} must be sorted, unique, and start at frame 0")
    return result


def _native_binding(
    model: str,
    row: Mapping[str, Any],
    canonical_path: Path,
    *,
    expected_content_order: str | None = None,
) -> dict[str, Any]:
    diagnostics = row.get("input_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{model}: missing input_diagnostics")
    metadata = diagnostics.get("video_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{model}: missing video_metadata")
    indices = _finite_int_list(metadata.get("frames_indices"), f"{model}.frames_indices")
    count = int(metadata.get("total_num_frames", 0))
    width = int(metadata.get("width", 0))
    height = int(metadata.get("height", 0))
    fps = float(metadata.get("fps", 0.0))
    if count <= 0 or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{model}: invalid video metadata")
    if indices[-1] != count - 1:
        raise ValueError(
            f"{model}: processor terminal {indices[-1]} != decoded terminal {count - 1}"
        )
    record = diagnostics.get("video_record")
    if isinstance(record, dict) and record.get("source_video_path"):
        recorded = Path(str(record["source_video_path"])).resolve()
        if recorded != canonical_path:
            raise ValueError(f"{model}: source video differs from model input")
    content_order = diagnostics.get("content_order")
    content_order_source = "row_input_diagnostics"
    if content_order is None and expected_content_order is not None:
        content_order = expected_content_order
        content_order_source = "run_manifest"
    if content_order not in {"text_then_video", "video_then_text"}:
        raise ValueError(f"{model}: unsupported or missing content_order")
    if (
        expected_content_order is not None
        and content_order != expected_content_order
    ):
        raise ValueError(
            f"{model}: row content_order differs from its frozen run manifest"
        )
    grid = diagnostics.get("video_grid_thw")
    if (
        not isinstance(grid, list)
        or not grid
        or any(
            not isinstance(row, (list, tuple))
            or len(row) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
                for value in row
            )
            for row in grid
        )
    ):
        raise ValueError(f"{model}: video_grid_thw must be positive integer [T,H,W] rows")
    return {
        "sampled_frame_indices": indices,
        "video_grid_thw": [[int(value) for value in row] for row in grid],
        "content_order": content_order,
        "content_order_source": content_order_source,
        "frame_count": count,
        "width": width,
        "height": height,
        "fps": fps,
    }


def _roboreward_content_order_contract(
    runs: Mapping[str, Mapping[str, Any]],
    *,
    example_id: str,
    input_row: Mapping[str, Any],
    source_video: Path,
    canonical_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not runs:
        return None
    bindings: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for order in ROBOREWARD_CONTENT_ORDERS:
        run = runs[order]
        wrapped = run["rows"][example_id]
        row = wrapped["row"]
        if row.get("status") != "ok":
            raise ValueError(f"{example_id}: roboreward/{order} baseline is not ok")
        expected_metadata = (
            str(input_row.get("group_id")),
            str(input_row.get("instruction")),
            input_row.get("group_media_sha256"),
        )
        actual_metadata = (
            str(row.get("group_id")),
            str(row.get("instruction")),
            row.get("group_media_sha256"),
        )
        if actual_metadata != expected_metadata:
            raise ValueError(
                f"{example_id}: roboreward/{order} input binding differs"
            )
        binding = _native_binding(
            f"roboreward/{order}",
            row,
            source_video,
            expected_content_order=order,
        )
        if binding["content_order"] != order:
            raise ValueError(f"{example_id}: roboreward/{order} row order differs")
        bindings[order] = binding
        records[order] = {
            **copy.deepcopy(run["provenance"]),
            "baseline_record": {
                "path": wrapped["path"],
                "file_sha256": wrapped["file_sha256"],
                "row_fingerprint": object_fingerprint(row),
                "content_order_source": binding["content_order_source"],
            },
        }
    invariant_fields = (
        "sampled_frame_indices",
        "video_grid_thw",
        "frame_count",
        "width",
        "height",
        "fps",
    )
    first = bindings[ROBOREWARD_CONTENT_ORDERS[0]]
    second = bindings[ROBOREWARD_CONTENT_ORDERS[1]]
    differences = {
        field: {
            ROBOREWARD_CONTENT_ORDERS[0]: first[field],
            ROBOREWARD_CONTENT_ORDERS[1]: second[field],
        }
        for field in invariant_fields
        if first[field] != second[field]
    }
    if differences:
        raise ValueError(
            "RoboReward content-order processor bindings differ: "
            f"{differences}"
        )
    canonical_differences = {
        field: {"canonical": canonical_binding[field], "shared": first[field]}
        for field in invariant_fields
        if canonical_binding[field] != first[field]
    }
    if (
        canonical_binding.get("content_order") not in ROBOREWARD_CONTENT_ORDERS
        or canonical_differences
    ):
        raise ValueError(
            f"{example_id}: canonical RoboReward binding differs from the "
            f"dual-order contract: {canonical_differences}"
        )
    contract = {
        "schema_version": PROCESSOR_CONTENT_ORDER_CONTRACT_SCHEMA,
        "model_family": "roboreward",
        "validated_orders": list(ROBOREWARD_CONTENT_ORDERS),
        "shared_processor_frame_indices": copy.deepcopy(
            first["sampled_frame_indices"]
        ),
        "shared_processor_video_grid_thw": copy.deepcopy(
            first["video_grid_thw"]
        ),
        "shared_video_metadata": {
            field: first[field]
            for field in ("frame_count", "width", "height", "fps")
        },
        "runs": records,
    }
    contract["fingerprint"] = object_fingerprint(contract)
    return contract


def validate_processor_content_order_contract(
    binding: Mapping[str, Any],
    *,
    identity: str,
    verify_file: Any | None = None,
) -> dict[str, Any] | None:
    """Validate and optionally re-open every source frozen by the RR order contract."""
    raw = binding.get("processor_content_order_contract")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{identity}: processor content-order contract is malformed")
    contract = copy.deepcopy(dict(raw))
    fingerprint = str(contract.pop("fingerprint", ""))
    if not fingerprint or fingerprint != object_fingerprint(contract):
        raise ValueError(f"{identity}: processor content-order fingerprint is invalid")
    if (
        contract.get("schema_version")
        != PROCESSOR_CONTENT_ORDER_CONTRACT_SCHEMA
        or contract.get("model_family") != "roboreward"
        or contract.get("validated_orders") != list(ROBOREWARD_CONTENT_ORDERS)
    ):
        raise ValueError(f"{identity}: processor content-order contract is invalid")
    actual_indices = binding.get(
        "sampled_frame_indices", binding.get("processor_frame_indices")
    )
    actual_grid = binding.get(
        "video_grid_thw", binding.get("processor_video_grid_thw")
    )
    if contract.get("shared_processor_frame_indices") != actual_indices:
        raise ValueError(f"{identity}: shared processor frame indices differ")
    if contract.get("shared_processor_video_grid_thw") != actual_grid:
        raise ValueError(f"{identity}: shared processor video grid differs")
    runs = contract.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(
        ROBOREWARD_CONTENT_ORDERS
    ):
        raise ValueError(f"{identity}: processor contract run coverage differs")
    for order in ROBOREWARD_CONTENT_ORDERS:
        run = runs[order]
        if not isinstance(run, Mapping) or run.get("content_order") != order:
            raise ValueError(f"{identity}/{order}: processor run binding is malformed")
        files = run.get("record_files")
        if (
            not isinstance(files, list)
            or not files
            or run.get("record_files_fingerprint") != object_fingerprint(files)
        ):
            raise ValueError(f"{identity}/{order}: record-file binding is invalid")
        baseline = run.get("baseline_record")
        if not isinstance(baseline, Mapping):
            raise ValueError(f"{identity}/{order}: baseline record binding is missing")
        sha_values = (
            run.get("manifest_sha256"),
            run.get("manifest_config_fingerprint"),
            run.get("record_files_fingerprint"),
            baseline.get("file_sha256"),
            baseline.get("row_fingerprint"),
        )
        if any(
            len(str(value or "")) != 64
            or any(ch not in "0123456789abcdef" for ch in str(value))
            for value in sha_values
        ):
            raise ValueError(f"{identity}/{order}: invalid SHA/fingerprint field")
        if baseline.get("content_order_source") not in {
            "row_input_diagnostics",
            "run_manifest",
        }:
            raise ValueError(f"{identity}/{order}: content_order_source is invalid")
        if verify_file is not None:
            manifest_path = verify_file(
                run.get("manifest_path"),
                run.get("manifest_sha256"),
                identity=f"{identity}/{order}/manifest",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config") if isinstance(manifest, Mapping) else None
            if (
                not isinstance(config, Mapping)
                or object_fingerprint(config)
                != run.get("manifest_config_fingerprint")
                or str(
                    config.get("my_dataset_eval", {}).get("content_order", "")
                )
                != order
            ):
                raise ValueError(
                    f"{identity}/{order}: current manifest config differs"
                )
            for number, value in enumerate(files):
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"{identity}/{order}: malformed record file entry"
                    )
                verify_file(
                    value.get("path"),
                    value.get("sha256"),
                    identity=f"{identity}/{order}/records/{number}",
                )
            verify_file(
                baseline.get("path"),
                baseline.get("file_sha256"),
                identity=f"{identity}/{order}/selected-record-file",
            )
    result = dict(contract)
    result["fingerprint"] = fingerprint
    return result


def _checked_record_image(value: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    path = _resolved_file(str(value.get(f"{prefix}_path", "")), f"GRM {prefix} frame")
    declared = str(value.get(f"{prefix}_sha256", ""))
    actual = sha256_file(path)
    if not declared or declared != actual:
        raise ValueError(f"GRM {prefix} frame SHA does not match the record")
    width, height = _image_size(path)
    return {
        "source_frame_index": int(value.get(f"{prefix}_index", -1)),
        "image_path": str(path),
        "image_sha256": actual,
        "width": width,
        "height": height,
    }


def _grm_binding(row: Mapping[str, Any], blank_goal: Path) -> dict[str, Any]:
    records = row.get("frame_record")
    if not isinstance(records, dict) or set(records) != {
        "front",
        "left_wrist",
        "right_wrist",
    }:
        raise ValueError("grm: frame_record must contain the three canonical views")
    frames: dict[str, dict[str, Any]] = {}
    counts: set[int] = set()
    dimensions: set[tuple[int, int]] = set()
    for view in ("front", "left_wrist", "right_wrist"):
        value = records[view]
        if not isinstance(value, dict):
            raise ValueError(f"grm: malformed {view} frame record")
        first = _checked_record_image(value, "first")
        last = _checked_record_image(value, "last")
        if first["source_frame_index"] != 0:
            raise ValueError(f"grm: {view} first frame is not frame 0")
        count = int(value.get("reported_frame_count", 0))
        if count <= 0 or last["source_frame_index"] != count - 1:
            raise ValueError(f"grm: invalid {view} terminal binding")
        declared_video_sha = str(value.get("video_sha256", ""))
        if not declared_video_sha:
            raise ValueError(f"grm: missing {view} video SHA")
        frames[view] = {
            "first": first,
            "last": last,
            "video_sha256": declared_video_sha,
        }
        counts.add(count)
        dimensions.add((int(value.get("width", 0)), int(value.get("height", 0))))
        dimensions.add((first["width"], first["height"]))
        dimensions.add((last["width"], last["height"]))
    if len(counts) != 1 or len(dimensions) != 1:
        raise ValueError("grm: the three endpoint views disagree on count or dimensions")
    count = next(iter(counts))
    width, height = next(iter(dimensions))
    image_paths = [
        frames["front"]["first"]["image_path"],
        str(blank_goal),
        frames["front"]["first"]["image_path"],
        frames["left_wrist"]["first"]["image_path"],
        frames["right_wrist"]["first"]["image_path"],
        frames["front"]["last"]["image_path"],
        frames["left_wrist"]["last"]["image_path"],
        frames["right_wrist"]["last"]["image_path"],
    ]
    return {
        "sampled_frame_indices": [0, count - 1],
        "image_paths": image_paths,
        "primary_target_slot": "after_cam_high",
        "primary_target_view": "front",
        "frame_count": count,
        "width": width,
        "height": height,
        "fps": None,
        "front": frames["front"],
    }


def _decode_key_frames(
    video_path: Path,
    frame_indices: Sequence[int],
    frame_count: int,
    output_dir: Path,
) -> dict[int, dict[str, Any]]:
    wanted = set(int(value) for value in frame_indices)
    if not wanted or min(wanted) < 0 or max(wanted) >= frame_count:
        raise ValueError("Requested key frame falls outside the source video")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
    result: dict[int, dict[str, Any]] = {}
    try:
        for index in range(frame_count):
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Video decode stopped before frame {index}: {video_path}")
            if index not in wanted:
                continue
            path = output_dir / f"frame_{index:06d}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Cannot write canonical frame: {path}")
            height, width = image.shape[:2]
            result[index] = {
                "source_frame_index": index,
                "image_path": str(path.resolve()),
                "image_sha256": sha256_file(path),
                "width": int(width),
                "height": int(height),
            }
        extra_ok, extra_image = capture.read()
        if extra_ok and extra_image is not None:
            raise RuntimeError(
                f"Video contains frames beyond frozen frame_count={frame_count}: {video_path}"
            )
    finally:
        capture.release()
    if set(result) != wanted:
        raise RuntimeError("Canonical key-frame extraction is incomplete")
    return result


def _manifest_base(
    cfg: Mapping[str, Any],
    inputs_path: Path,
    roles_path: Path,
    split_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": TRACKED_GROUNDING_MANIFEST_SCHEMA,
        "status": "requests_built",
        "requests_path": None,
        "requests_sha256": None,
        "request_count": 0,
        "tracks_path": None,
        "tracks_sha256": None,
        "artifact_count": 0,
        "status_counts": {},
        "coverage_complete": False,
        "request_bindings_current": False,
        "inputs_path": str(inputs_path),
        "inputs_sha256": sha256_file(inputs_path),
        "roles_path": str(roles_path),
        "roles_sha256": sha256_file(roles_path),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "baseline_runs": {
            model: str(Path(str(cfg["baseline_runs"][model])).resolve())
            for model in MODELS
        },
        "tracker": None,
        "proposal_backend": None,
        "labels_opened": False,
    }


def build_tracked_grounding_requests(config: dict[str, Any]) -> Path:
    """Freeze label-free model/frame bindings for first-frame tracking."""
    cfg, _ = _cfg(config)
    inputs_path = _resolved_file(cfg["inputs_path"], "model input manifest")
    roles_path = _resolved_file(cfg["roles_path"], "semantic role manifest")
    split_path = _resolved_file(cfg["split_path"], "frozen whitebox split")
    output_dir = Path(str(cfg["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_model_inputs(inputs_path)
    roles = _role_rows(roles_path)
    if set(roles) != {str(row["example_id"]) for row in inputs}:
        raise ValueError("Role manifest IDs do not exactly match model input IDs")
    split_value = json.loads(split_path.read_text(encoding="utf-8"))
    split_examples = split_value.get("examples") if isinstance(split_value, dict) else None
    if not isinstance(split_examples, dict) or set(split_examples) != {
        "discovery",
        "validation",
        "test",
    }:
        raise ValueError("Frozen split must contain discovery/validation/test examples")
    partitions: dict[str, str] = {}
    for partition, values in split_examples.items():
        if not isinstance(values, list):
            raise ValueError(f"Frozen split partition {partition} is not a list")
        for value in values:
            example_id = str(value)
            if not example_id or example_id in partitions:
                raise ValueError(f"Missing or duplicate split example_id: {example_id!r}")
            partitions[example_id] = str(partition)
    if set(partitions) != {str(row["example_id"]) for row in inputs}:
        raise ValueError("Frozen split IDs do not exactly match model input IDs")
    baselines = {
        model: _load_baseline_run(cfg["baseline_runs"][model]) for model in MODELS
    }
    input_ids = {str(row["example_id"]) for row in inputs}
    roboreward_order_runs = _load_roboreward_content_order_runs(
        cfg,
        inputs_path=inputs_path,
        expected_count=len(inputs),
    )
    for model in MODELS:
        missing = sorted(input_ids - set(baselines[model]))
        if missing:
            raise ValueError(f"{model}: missing {len(missing)} baseline records")
    for order, run in roboreward_order_runs.items():
        if set(run["rows"]) != input_ids:
            raise ValueError(
                f"roboreward/{order}: baseline IDs differ from model inputs"
            )

    default_blank = Path(__file__).resolve().parents[2] / "examples" / "blank_goal.png"
    blank_goal = _resolved_file(cfg.get("blank_goal", default_blank), "blank goal image")
    inputs_sha = sha256_file(inputs_path)
    roles_sha = sha256_file(roles_path)
    split_sha = sha256_file(split_path)
    frame_cache: dict[str, dict[int, dict[str, Any]]] = {}
    video_hash_cache: dict[str, str] = {}
    requests: list[dict[str, Any]] = []

    for input_row in sorted(inputs, key=lambda value: str(value["example_id"])):
        example_id = str(input_row["example_id"])
        role = roles[example_id]
        if str(role.get("group_id")) != str(input_row.get("group_id")):
            raise ValueError(f"{example_id}: role group_id differs from model input")
        if str(role.get("instruction")) != str(input_row.get("instruction")):
            raise ValueError(f"{example_id}: role instruction differs from model input")
        source_video = _resolved_file(
            input_row["video_paths"]["front"], f"{example_id} front video"
        )
        video_sha = video_hash_cache.setdefault(
            str(source_video), sha256_file(source_video)
        )
        declared_video_sha = str(input_row.get("view_sha256", {}).get("front", ""))
        if declared_video_sha and declared_video_sha != video_sha:
            raise ValueError(f"{example_id}: model-input front video SHA mismatch")

        wrapped = {model: baselines[model][example_id] for model in MODELS}
        for model, value in wrapped.items():
            baseline = value["row"]
            if baseline.get("status") != "ok":
                raise ValueError(f"{example_id}: latest {model} baseline is not ok")
            if str(baseline.get("group_id")) != str(input_row.get("group_id")):
                raise ValueError(f"{example_id}: {model} baseline group mismatch")
            if str(baseline.get("instruction")) != str(input_row.get("instruction")):
                raise ValueError(f"{example_id}: {model} baseline instruction mismatch")
            if baseline.get("group_media_sha256") != input_row.get("group_media_sha256"):
                raise ValueError(f"{example_id}: {model} baseline media provenance mismatch")

        native = {
            model: _native_binding(model, wrapped[model]["row"], source_video)
            for model in ("roboreward", "qwen")
        }
        processor_content_order_contract = _roboreward_content_order_contract(
            roboreward_order_runs,
            example_id=example_id,
            input_row=input_row,
            source_video=source_video,
            canonical_binding=native["roboreward"],
        )
        grm = _grm_binding(wrapped["grm"]["row"], blank_goal)
        if grm["front"]["video_sha256"] != video_sha:
            raise ValueError(f"{example_id}: GRM front video SHA differs from canonical video")
        count_dim = {
            (value["frame_count"], value["width"], value["height"])
            for value in (*native.values(), grm)
        }
        if len(count_dim) != 1:
            raise ValueError(f"{example_id}: baseline frame count/dimensions disagree")
        frame_count, width, height = next(iter(count_dim))
        terminal = frame_count - 1
        fps_values = {float(value["fps"]) for value in native.values()}
        if len(fps_values) != 1:
            raise ValueError(f"{example_id}: native processors disagree on FPS")
        fps = next(iter(fps_values))
        key_indices = sorted(
            {0, terminal}
            | set(native["roboreward"]["sampled_frame_indices"])
            | set(native["qwen"]["sampled_frame_indices"])
        )
        if video_sha not in frame_cache:
            frame_cache[video_sha] = _decode_key_frames(
                source_video,
                key_indices,
                frame_count,
                output_dir / "frames" / video_sha,
            )
        frames = frame_cache[video_sha]
        if not set(key_indices).issubset(frames):
            raise ValueError("Cached key-frame set is incompatible with this baseline binding")
        for frame in frames.values():
            if (frame["width"], frame["height"]) != (width, height):
                raise ValueError(f"{example_id}: decoded key-frame dimensions disagree")

        model_bindings: dict[str, dict[str, Any]] = {}
        for model in ("roboreward", "qwen"):
            value = native[model]
            model_bindings[model] = {
                "sampled_frame_indices": value["sampled_frame_indices"],
                "video_grid_thw": value["video_grid_thw"],
                "content_order": value["content_order"],
                "terminal": copy.deepcopy(frames[terminal]),
            }
        if processor_content_order_contract is not None:
            model_bindings["roboreward"][
                "processor_content_order_contract"
            ] = processor_content_order_contract
        grm_terminal = copy.deepcopy(grm["front"]["last"])
        model_bindings["grm"] = {
            "sampled_frame_indices": grm["sampled_frame_indices"],
            "image_paths": grm["image_paths"],
            "primary_target_slot": grm["primary_target_slot"],
            "primary_target_view": grm["primary_target_view"],
            "terminal": grm_terminal,
        }
        source = {
            "inputs": {
                "path": str(inputs_path),
                "sha256": inputs_sha,
                "row_fingerprint": object_fingerprint(input_row),
            },
            "roles": {
                "path": str(roles_path),
                "sha256": roles_sha,
                "row_fingerprint": object_fingerprint(role),
            },
            "split": {
                "path": str(split_path),
                "sha256": split_sha,
                "partition": partitions[example_id],
                "assignment_fingerprint": object_fingerprint(
                    {"example_id": example_id, "partition": partitions[example_id]}
                ),
            },
            "baseline_records": {
                model: {
                    "path": wrapped[model]["path"],
                    "file_sha256": wrapped[model]["file_sha256"],
                    "row_fingerprint": object_fingerprint(wrapped[model]["row"]),
                }
                for model in MODELS
            },
        }
        request = {
            "schema_version": TRACKED_GROUNDING_REQUEST_SCHEMA,
            "example_id": example_id,
            "group_id": str(input_row["group_id"]),
            "partition": partitions[example_id],
            "task_id": str(input_row.get("task_id", role.get("task_id", ""))),
            "instruction": str(input_row["instruction"]),
            "roles": {
                key: copy.deepcopy(role.get(key))
                for key in (
                    "grounding_strategy",
                    "target_phrase",
                    "reference_object",
                    "relation",
                    "ordinal",
                    "direction",
                    "requires_instance_review",
                )
            },
            "status": "ok",
            "error": None,
            "source": source,
            "video": {
                "path": str(source_video),
                "sha256": video_sha,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "fps": fps,
            },
            "first_frame": copy.deepcopy(frames[0]),
            "key_frames": [copy.deepcopy(frames[index]) for index in key_indices],
            "model_frame_bindings": model_bindings,
        }
        request["request_fingerprint"] = _fingerprint_row(
            request, "request_fingerprint"
        )
        requests.append(request)

    requests_path = output_dir / "requests.jsonl"
    write_jsonl(requests_path, requests)
    manifest = _manifest_base(cfg, inputs_path, roles_path, split_path)
    manifest.update(
        {
            "requests_path": str(requests_path),
            "requests_sha256": sha256_file(requests_path),
            "request_count": len(requests),
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    return requests_path


def _valid_bbox(value: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        return None
    return bbox


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (float(left[2]) - float(left[0])) * (
        float(left[3]) - float(left[1])
    )
    right_area = (float(right[2]) - float(right[0])) * (
        float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _center(bbox: Sequence[float]) -> list[float]:
    return [
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ]


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.hypot(
        float(left["center_xy"][0]) - float(right["center_xy"][0]),
        float(left["center_xy"][1]) - float(right["center_xy"][1]),
    )


def _candidate_provider_from_config(sam3_cfg: Mapping[str, Any]) -> Any:
    injected = sam3_cfg.get("_candidate_provider")
    if injected is not None:
        return injected
    from ..grounding.sam3 import SAM3Grounder

    return SAM3Grounder(dict(sam3_cfg))


def _candidate_provider_provenance(
    provider: Any,
    sam3_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    injected = sam3_cfg.get("_candidate_provider") is not None
    model_value = sam3_cfg.get("model_path")
    model_path = (
        Path(str(model_value)).expanduser().resolve() if model_value else None
    )
    config_path = model_path / "config.json" if model_path else None
    try:
        transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        transformers_version = "not-installed"
    weight_entries: list[dict[str, Any]] = []
    if not injected:
        if model_path is None or not model_path.is_dir():
            raise FileNotFoundError(
                f"Transformers SAM3 model_path is not a directory: {model_path}"
            )
        weight_paths = sorted(
            {
                path
                for pattern in (
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "pytorch_model*.bin",
                    "pytorch_model*.bin.index.json",
                )
                for path in model_path.glob(pattern)
                if path.is_file()
            }
        )
        if not any(
            path.suffix == ".safetensors" or path.suffix == ".bin"
            for path in weight_paths
        ):
            raise FileNotFoundError(
                f"No Transformers SAM3 safetensors/bin weights under {model_path}"
            )
        weight_entries = [
            {
                "path": str(path.relative_to(model_path)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weight_paths
        ]
    provenance = {
        "backend": "injected_test_double" if injected else "transformers_sam3_image",
        "provider_class": (
            f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"
        ),
        "model_path": str(model_path) if model_path else None,
        "model_artifact_fingerprint": (
            artifact_fingerprint(model_path) if model_path else None
        ),
        "model_config_sha256": (
            sha256_file(config_path) if config_path and config_path.is_file() else None
        ),
        "model_weight_files": weight_entries,
        "model_weights_fingerprint": (
            object_fingerprint(weight_entries) if weight_entries else None
        ),
        "grounder_source_path": str(
            (Path(__file__).resolve().parents[1] / "grounding" / "sam3.py").resolve()
        ),
        "grounder_source_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "grounding" / "sam3.py"
        ),
        **_orchestrator_provenance(),
        "transformers_version": transformers_version,
        "provider_reported_fingerprint": str(
            getattr(provider, "fingerprint", "unspecified")
        ),
        "settings": {
            "threshold": float(sam3_cfg.get("threshold", 0.3)),
            "mask_threshold": float(sam3_cfg.get("mask_threshold", 0.5)),
            "top_n": int(sam3_cfg.get("top_n", 10)),
            "top_n_per_query": (
                int(sam3_cfg["top_n_per_query"])
                if sam3_cfg.get("top_n_per_query") is not None
                else None
            ),
            "proposal_max_per_query": int(
                sam3_cfg.get("proposal_max_per_query", 20)
            ),
            "proposal_nms_iou": float(sam3_cfg.get("proposal_nms_iou", 0.8)),
            "reference_exclusion_iou": float(
                sam3_cfg.get("reference_exclusion_iou", 0.5)
            ),
        },
    }
    provenance["proposer_fingerprint"] = object_fingerprint(provenance)
    return provenance


def _official_source_tree_provenance(source: Path) -> dict[str, Any]:
    sam_root = source / "sam3"
    files = sorted(
        path
        for path in sam_root.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    )
    bpe_candidates = sorted(
        path
        for path in source.rglob("bpe_simple_vocab_16e6.txt.gz")
        if path.is_file()
    )
    files.extend(path for path in bpe_candidates if path not in files)
    entries = [
        {
            "path": str(path.relative_to(source)),
            "sha256": sha256_file(path),
        }
        for path in files
    ]

    def hashes_named(name: str) -> list[dict[str, str]]:
        return [entry for entry in entries if Path(entry["path"]).name == name]

    return {
        "source_tree_fingerprint": object_fingerprint(entries),
        "source_tree_file_count": len(entries),
        "sam3_video_inference_files": hashes_named("sam3_video_inference.py"),
        "tracking_predictor_files": hashes_named("tracking_predictor.py"),
        "bpe_vocab_files": hashes_named("bpe_simple_vocab_16e6.txt.gz"),
    }


def _save_proposal_mask(
    mask: Any,
    path: Path,
    width: int,
    height: int,
) -> tuple[str | None, str | None]:
    if mask is None:
        return None, None
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape != (height, width):
        raise ValueError("Proposal mask does not match the canonical first frame")
    binary = array.astype(bool)
    if not binary.any():
        raise ValueError("Proposal mask is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), binary.astype(np.uint8) * 255):
        raise RuntimeError(f"Cannot write proposal mask: {path}")
    return str(path.resolve()), sha256_file(path)


def _normalize_candidates(
    raw: Iterable[Mapping[str, Any]],
    *,
    query: str,
    query_role: str,
    width: int,
    height: int,
    output_dir: Path,
    nms_iou: float,
    max_candidates: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for value in raw:
        if str(value.get("query", value.get("label", ""))) != query:
            continue
        bbox = _valid_bbox(value.get("bbox_xyxy", value.get("bbox")), width, height)
        try:
            score = float(value.get("score"))
        except (TypeError, ValueError):
            continue
        if bbox is None or not math.isfinite(score):
            continue
        candidate_id = "candidate-" + object_fingerprint(
            {
                "query_role": query_role,
                "query": query,
                "bbox_xyxy": bbox,
                "score": score,
            }
        )[:24]
        normalized.append(
            {
                "candidate_id": candidate_id,
                "query_role": query_role,
                "query": query,
                "bbox_xyxy": bbox,
                "score": score,
                "center_xy": _center(bbox),
                "_mask": value.get("_mask"),
                "_source_mask_path": value.get("mask_path"),
            }
        )
    normalized.sort(
        key=lambda item: (
            -float(item["score"]),
            tuple(float(value) for value in item["bbox_xyxy"]),
            str(item["candidate_id"]),
        )
    )
    kept: list[dict[str, Any]] = []
    for candidate in normalized:
        if any(
            _bbox_iou(candidate["bbox_xyxy"], other["bbox_xyxy"]) > nms_iou
            for other in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= max_candidates:
            break
    serializable: list[dict[str, Any]] = []
    for candidate in kept:
        value = dict(candidate)
        mask = value.pop("_mask")
        source_mask_path = value.pop("_source_mask_path")
        mask_path: str | None = None
        mask_sha: str | None = None
        if mask is not None:
            mask_path, mask_sha = _save_proposal_mask(
                mask,
                output_dir / f"{candidate['candidate_id']}.png",
                width,
                height,
            )
        elif source_mask_path:
            path = _resolved_file(str(source_mask_path), "proposal candidate mask")
            mask_width, mask_height = _image_size(path)
            if (mask_width, mask_height) != (width, height):
                raise ValueError("Proposal mask file has the wrong dimensions")
            mask_path = str(path)
            mask_sha = sha256_file(path)
        value["mask_path"] = mask_path
        value["mask_sha256"] = mask_sha
        serializable.append(value)
    return serializable


def _extreme_tied(
    values: Sequence[float],
    *,
    choose_maximum: bool,
    tolerance: float = 1e-9,
) -> bool:
    if not values:
        return False
    extreme = (max if choose_maximum else min)(float(value) for value in values)
    return sum(abs(float(value) - extreme) <= tolerance for value in values) > 1


def _select_algorithmic_default(
    roles: Mapping[str, Any],
    targets: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not targets:
        return None, ["target_missing"]
    strategy = str(roles.get("grounding_strategy", ""))
    if strategy in {"simple", "object_identity", "attribute_color"}:
        return targets[0], []
    if strategy == "ordinal_position":
        ordinal = _ORDINALS.get(str(roles.get("ordinal", "")).lower())
        direction = str(roles.get("direction", "")).lower()
        if ordinal is None or direction not in {"left", "right"}:
            return None, ["ordinal_metadata_invalid"]
        ordered = sorted(
            targets,
            key=lambda item: (
                float(item["center_xy"][0]),
                str(item["candidate_id"]),
            ),
            reverse=direction == "right",
        )
        if ordinal > len(ordered):
            return None, ["ordinal_candidate_count_insufficient"]
        chosen_x = float(ordered[ordinal - 1]["center_xy"][0])
        if (
            sum(
                abs(float(item["center_xy"][0]) - chosen_x) <= 1e-9
                for item in targets
            )
            > 1
        ):
            return None, ["ordinal_geometry_tie"]
        return ordered[ordinal - 1], []
    if strategy in {"left_right_relation", "distance_relation"}:
        if not references:
            return None, ["reference_missing"]
        if len(references) != 1:
            return None, ["reference_ambiguous"]
        reference = references[0]
        if strategy == "left_right_relation":
            relation = str(roles.get("relation", "")).lower()
            if relation not in {"left", "right"}:
                return None, ["side_relation_invalid"]
            reference_x = float(reference["center_xy"][0])
            eligible = [
                item
                for item in targets
                if (
                    float(item["center_xy"][0]) < reference_x
                    if relation == "left"
                    else float(item["center_xy"][0]) > reference_x
                )
            ]
            if not eligible:
                return None, ["no_target_on_requested_side"]
            distances = [_distance(item, reference) for item in eligible]
            if _extreme_tied(distances, choose_maximum=False):
                return None, ["side_geometry_tie"]
            return min(eligible, key=lambda item: _distance(item, reference)), []
        relation = str(roles.get("relation", "")).lower()
        if relation not in {"closest to", "farthest from", "closest", "farthest"}:
            return None, ["distance_relation_invalid"]
        distances = [_distance(item, reference) for item in targets]
        choose_maximum = relation.startswith("farthest")
        if _extreme_tied(distances, choose_maximum=choose_maximum):
            return None, ["distance_geometry_tie"]
        chooser = max if choose_maximum else min
        return chooser(targets, key=lambda item: _distance(item, reference)), []
    return None, ["strategy_requires_manual_review"]


def _proposal_from_candidates(
    request: Mapping[str, Any],
    targets: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
    reference_exclusion_iou: float = 0.5,
) -> dict[str, Any]:
    roles = request["roles"]
    if not 0 <= reference_exclusion_iou <= 1:
        raise ValueError("reference_exclusion_iou must be in [0, 1]")
    eligible_targets = list(targets)
    exclusions: list[dict[str, Any]] = []
    if (
        str(roles.get("grounding_strategy"))
        in {"left_right_relation", "distance_relation"}
        and len(references) == 1
    ):
        reference = references[0]
        eligible_targets = []
        for target in targets:
            overlap = _bbox_iou(target["bbox_xyxy"], reference["bbox_xyxy"])
            if overlap >= reference_exclusion_iou:
                excluded = copy.deepcopy(target)
                excluded.update(
                    {
                        "exclusion_reason": "overlaps_unique_reference_instance",
                        "reference_candidate_id": reference["candidate_id"],
                        "reference_iou": overlap,
                        "reference_exclusion_iou": reference_exclusion_iou,
                    }
                )
                exclusions.append(excluded)
            else:
                eligible_targets.append(target)
    default, reasons = _select_algorithmic_default(
        roles, eligible_targets, references
    )
    status = (
        "invalid"
        if not eligible_targets
        else ("needs_review" if default is None else "ok")
    )
    options: list[dict[str, Any]] = []
    if default is not None:
        item = copy.deepcopy(default)
        item["selection"] = "algorithmic_default"
        options.append(item)
        alternatives = [
            value
            for value in eligible_targets
            if value["candidate_id"] != default["candidate_id"]
        ][:2]
    else:
        # No instance is auto-accepted.  The first score-ordered option is only
        # a UI review default, followed by at most two other candidates.
        alternatives = list(eligible_targets[:3])
    for candidate in alternatives:
        item = copy.deepcopy(candidate)
        item["selection"] = "alternative"
        options.append(item)
    proposal = {
        "schema_version": TRACKED_GROUNDING_PROPOSAL_SCHEMA,
        "status": status,
        "strategy": str(roles.get("grounding_strategy", "")),
        "target_query": roles.get("target_phrase"),
        "reference_query": roles.get("reference_object"),
        "target_candidates": copy.deepcopy(list(targets)),
        "reference_candidates": copy.deepcopy(list(references)),
        "excluded_target_candidates": exclusions,
        "reference_exclusion_iou": reference_exclusion_iou,
        "algorithmic_default": copy.deepcopy(default),
        "options": options,
        "review_reasons": reasons,
    }
    proposal["fingerprint"] = _fingerprint_row(proposal)
    return proposal


def _propose(
    request: Mapping[str, Any],
    provider: Any,
    provider_provenance: Mapping[str, Any],
    output_dir: Path,
    sam3_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    target_query = str(request["roles"].get("target_phrase") or "").strip()
    if not target_query:
        proposal = _proposal_from_candidates(request, [], [])
        proposal["provider_provenance"] = copy.deepcopy(dict(provider_provenance))
        proposal["fingerprint"] = _fingerprint_row(proposal)
        return proposal
    reference_value = request["roles"].get("reference_object")
    reference_query = str(reference_value).strip() if reference_value else None
    image_path = request["first_frame"]["image_path"]
    width = int(request["first_frame"]["width"])
    height = int(request["first_frame"]["height"])
    nms_iou = float(sam3_cfg.get("proposal_nms_iou", 0.8))
    if not 0 <= nms_iou <= 1:
        raise ValueError("proposal_nms_iou must be in [0, 1]")
    maximum = int(sam3_cfg.get("proposal_max_per_query", 20))
    if maximum < 1:
        raise ValueError("proposal_max_per_query must be positive")

    # These calls remain deliberately separate.  A reference/destination result
    # can never leak into the target alternatives.
    target_raw = provider.candidates(image_path, [target_query])
    targets = _normalize_candidates(
        target_raw,
        query=target_query,
        query_role="target",
        width=width,
        height=height,
        output_dir=output_dir / "proposal_masks" / request["video"]["sha256"],
        nms_iou=nms_iou,
        max_candidates=maximum,
    )
    references: list[dict[str, Any]] = []
    if reference_query:
        reference_raw = provider.candidates(image_path, [reference_query])
        references = _normalize_candidates(
            reference_raw,
            query=reference_query,
            query_role="reference",
            width=width,
            height=height,
            output_dir=output_dir / "proposal_masks" / request["video"]["sha256"],
            nms_iou=nms_iou,
            max_candidates=maximum,
        )
    proposal = _proposal_from_candidates(
        request,
        targets,
        references,
        reference_exclusion_iou=float(sam3_cfg.get("reference_exclusion_iou", 0.5)),
    )
    proposal["provider_provenance"] = copy.deepcopy(dict(provider_provenance))
    proposal["fingerprint"] = _fingerprint_row(proposal)
    return proposal


def _module_under(module: Any, root: Path) -> bool:
    value = getattr(module, "__file__", None)
    if not value:
        return False
    try:
        Path(value).resolve().relative_to(root)
    except ValueError:
        return False
    return True


class _OfficialSam3InstanceTracker:
    """Own the official SAM3 model while exposing its SAM2-style tracker."""

    def __init__(self, model: Any):
        self._model = model
        self._tracker = model.tracker
        # Required by Meta's official sam3_for_sam2_video_task_example.ipynb.
        self._tracker.backbone = model.detector.backbone

    def init_state(self, **kwargs: Any) -> Any:
        return self._tracker.init_state(**kwargs)

    def add_new_points_or_box(self, **kwargs: Any) -> Any:
        return self._tracker.add_new_points_or_box(**kwargs)

    def propagate_in_video(self, *args: Any, **kwargs: Any) -> Any:
        return self._tracker.propagate_in_video(*args, **kwargs)

    def release_state(self, inference_state: Any) -> None:
        # The official instance API has no close_session operation. Its state is
        # caller-owned, so dropping the per-video tensors is the lifecycle boundary.
        if isinstance(inference_state, dict):
            inference_state.clear()
        gc.collect()

    def shutdown(self) -> None:
        tracker = self._tracker
        context = getattr(tracker, "bf16_context", None)
        if context is not None:
            context.__exit__(None, None, None)
        self._tracker = None
        self._model = None
        gc.collect()
        torch_module = sys.modules.get("torch")
        cuda = getattr(torch_module, "cuda", None) if torch_module else None
        if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
            cuda.empty_cache()


def _to_numpy(value: Any, *, force_float: bool = False) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if force_float and callable(getattr(value, "float", None)):
        value = value.float()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value)


def _predictor_from_config(
    sam3_cfg: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    injected = sam3_cfg.get("_predictor")
    if injected is not None:
        provenance = copy.deepcopy(sam3_cfg.get("_predictor_provenance", {}))
        provenance.setdefault("official_source_path", "injected-test-double")
        provenance.setdefault("model_builder_sha256", object_fingerprint("injected"))
        provenance.setdefault("video_predictor_sha256", object_fingerprint("injected"))
        provenance.setdefault("tracking_predictor_sha256", object_fingerprint("injected"))
        provenance.setdefault("checkpoint_path", "injected-test-double")
        provenance.setdefault("checkpoint_sha256", object_fingerprint("injected"))
        provenance.setdefault("source_tree_fingerprint", object_fingerprint("injected"))
        provenance.setdefault("source_tree_file_count", 0)
        provenance.setdefault("sam3_video_inference_files", [])
        provenance.setdefault("tracking_predictor_files", [])
        provenance.setdefault("bpe_vocab_files", [])
        for key, value in _orchestrator_provenance().items():
            provenance.setdefault(key, value)
        provenance.setdefault(
            "tracker_fingerprint",
            object_fingerprint(
                {
                    key: value
                    for key, value in provenance.items()
                    if key != "tracker_fingerprint"
                }
            ),
        )
        return injected, provenance

    source_value = sam3_cfg.get("official_source_path")
    if not source_value:
        raise RuntimeError(
            "sam3.official_source_path is required; Transformers SAM3 has no video tracker"
        )
    source = Path(str(source_value)).expanduser().resolve()
    builder_path = source / "sam3" / "model_builder.py"
    dense_predictor_path = source / "sam3" / "model" / "sam3_video_predictor.py"
    tracking_predictor_path = (
        source / "sam3" / "model" / "sam3_tracking_predictor.py"
    )
    if not all(
        path.is_file()
        for path in (builder_path, dense_predictor_path, tracking_predictor_path)
    ):
        raise RuntimeError(f"Official SAM3 video source is incomplete: {source}")
    checkpoint_value = sam3_cfg.get("checkpoint_path")
    if not checkpoint_value and sam3_cfg.get("model_path"):
        checkpoint_value = Path(str(sam3_cfg["model_path"])) / "sam3.pt"
    if not checkpoint_value:
        raise RuntimeError("sam3.checkpoint_path is required")
    checkpoint = _resolved_file(checkpoint_value, "SAM3 video checkpoint")

    existing = sys.modules.get("sam3")
    if existing is not None and not _module_under(existing, source):
        raise RuntimeError(
            "A non-official or different SAM3 package is already imported; refusing ambiguity"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    builder_module = importlib.import_module("sam3.model_builder")
    tracking_module = importlib.import_module(
        "sam3.model.sam3_tracking_predictor"
    )
    if not _module_under(builder_module, source) or not _module_under(
        tracking_module, source
    ):
        raise RuntimeError(
            "Imported SAM3 instance-tracking modules do not originate from "
            "official_source_path"
        )
    builder = getattr(builder_module, "build_sam3_video_model", None)
    if not callable(builder):
        raise RuntimeError("Official SAM3 source lacks build_sam3_video_model")
    provenance = {
        "backend": "official_sam3_sam2_style_instance_tracker",
        "tracker_api": "init_state/add_new_points_or_box/propagate_in_video",
        "official_source_path": str(source),
        "model_builder_sha256": sha256_file(builder_path),
        "video_predictor_sha256": sha256_file(dense_predictor_path),
        "tracking_predictor_sha256": sha256_file(tracking_predictor_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(sam3_cfg.get("device", "cuda")),
        "configured_gpus_to_use": copy.deepcopy(sam3_cfg.get("gpus_to_use")),
        **_official_source_tree_provenance(source),
        **_orchestrator_provenance(),
    }
    provenance["tracker_fingerprint"] = object_fingerprint(provenance)
    model = builder(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=str(sam3_cfg.get("device", "cuda")),
    )
    predictor = _OfficialSam3InstanceTracker(model)
    return predictor, provenance


def _parse_tracker_frame(
    frame_value: Any,
    obj_ids: Any,
    video_res_masks: Any,
    obj_score_logits: Any,
    width: int,
    height: int,
) -> tuple[int, dict[int, dict[str, Any]]]:
    """Parse the official SAM2-style tracker tuple, never dense detector rows."""

    if isinstance(frame_value, bool) or not isinstance(
        frame_value, (int, np.integer)
    ):
        raise ValueError("SAM3 tracker frame index is malformed")
    frame_index = int(frame_value)
    ids = _to_numpy(obj_ids)
    if ids.ndim != 1:
        raise ValueError("SAM3 tracker object ids must have shape [N]")
    count = int(ids.shape[0])
    if count < 1:
        raise ValueError("SAM3 instance tracker returned no locked object id")

    masks = _to_numpy(video_res_masks, force_float=True)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2 and count == 1:
        masks = masks[None, :, :]
    if masks.shape != (count, height, width):
        raise ValueError(
            "SAM3 instance tracker masks do not match [N, video_height, video_width]"
        )

    scores: np.ndarray | None = None
    if obj_score_logits is not None:
        scores = _to_numpy(obj_score_logits, force_float=True)
        if scores.size != count:
            raise ValueError("SAM3 instance tracker scores do not match object ids")
        scores = scores.reshape(count)

    parsed_ids: list[int] = []
    result: dict[int, dict[str, Any]] = {}
    for index in range(count):
        obj_value = ids[index]
        if isinstance(obj_value, (bool, np.bool_)):
            raise ValueError("SAM3 tracker object id is boolean")
        try:
            obj_id = int(obj_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("SAM3 tracker object id is not integral") from exc
        if float(obj_id) != float(obj_value):
            raise ValueError("SAM3 tracker object id is not integral")
        parsed_ids.append(obj_id)

        if scores is None:
            score = 1.0
        else:
            logit = float(scores[index])
            if not math.isfinite(logit):
                raise ValueError("SAM3 tracker object score contains NaN or infinity")
            score = float(1.0 / (1.0 + np.exp(-np.clip(logit, -80.0, 80.0))))

        mask = np.asarray(masks[index] > 0.0, dtype=bool)
        ys, xs = np.nonzero(mask)
        visible = bool(xs.size and ys.size)
        bbox = (
            [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ]
            if visible
            else None
        )
        result[obj_id] = {
            "obj_id": obj_id,
            "score": score,
            "bbox_xyxy": bbox,
            "visible": visible,
            "_mask": mask,
        }
    if len(set(parsed_ids)) != len(parsed_ids):
        raise ValueError("SAM3 tracker returned duplicate object ids in one frame")
    return frame_index, result


def _cache_key(
    request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tracker_fingerprint: str,
) -> str:
    return object_fingerprint(
        {
            "video_sha256": request["video"]["sha256"],
            "first_frame_index": 0,
            "first_frame_image_sha256": request["first_frame"]["image_sha256"],
            "anchor_bbox_xyxy": candidate["bbox_xyxy"],
            "anchor_mask_sha256": candidate.get("mask_sha256"),
            "tracker_fingerprint": tracker_fingerprint,
        }
    )


def _load_track_cache(
    path: Path,
    cache_key: str,
    required_indices: set[int],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != TRACKED_GROUNDING_CACHE_SCHEMA
        or value.get("cache_key") != cache_key
    ):
        return None
    expected = value.get("fingerprint")
    if expected != _fingerprint_row(value):
        return None
    frames = value.get("frames")
    if not isinstance(frames, list):
        return None
    available = {int(frame.get("source_frame_index", -1)) for frame in frames}
    if not required_indices.issubset(available):
        return None
    for frame in frames:
        mask_path = frame.get("mask_path")
        mask_sha = frame.get("mask_sha256")
        if not mask_path or not Path(str(mask_path)).is_file():
            return None
        if not mask_sha or sha256_file(mask_path) != mask_sha:
            return None
    return value


def _write_track_mask(path: Path, mask: np.ndarray) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Cannot write propagated SAM3 mask: {path}")
    return str(path.resolve()), sha256_file(path)


def _run_visual_propagation(
    predictor: Any,
    request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    predictor_provenance: Mapping[str, Any],
    cache_dir: Path,
    cache_key: str,
    sam3_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    width = int(request["video"]["width"])
    height = int(request["video"]["height"])
    frame_count = int(request["video"]["frame_count"])
    terminal = frame_count - 1
    required_indices = {
        int(frame["source_frame_index"]) for frame in request["key_frames"]
    }
    terminal_indices: set[int] = set()
    for binding in request["model_frame_bindings"].values():
        index = int(binding["terminal"]["source_frame_index"])
        terminal_indices.add(index)
        required_indices.add(index)

    bbox = [float(value) for value in candidate["bbox_xyxy"]]
    x1, y1, x2, y2 = bbox
    normalized_xyxy = np.asarray(
        [[x1 / width, y1 / height, x2 / width, y2 / height]],
        dtype=np.float32,
    )
    locked_obj_id = 1
    inference_state: Any = None
    stream: Any = None
    pending_error: Exception | None = None
    result: dict[str, Any] | None = None
    try:
        inference_state = predictor.init_state(
            video_path=request["video"]["path"],
            offload_video_to_cpu=bool(
                sam3_cfg.get("offload_video_to_cpu", True)
            ),
            offload_state_to_cpu=bool(
                sam3_cfg.get("offload_state_to_cpu", False)
            ),
            async_loading_frames=bool(
                sam3_cfg.get("async_loading_frames", False)
            ),
        )
        if not isinstance(inference_state, Mapping):
            raise RuntimeError("SAM3 instance tracker returned a malformed state")
        if int(inference_state.get("num_frames", -1)) != frame_count:
            raise ValueError(
                "SAM3 instance tracker frame count differs from the frozen request"
            )
        if (
            int(inference_state.get("video_width", -1)) != width
            or int(inference_state.get("video_height", -1)) != height
        ):
            raise ValueError(
                "SAM3 instance tracker video dimensions differ from the frozen request"
            )

        anchor_output = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=locked_obj_id,
            box=normalized_xyxy,
        )
        if not isinstance(anchor_output, tuple) or len(anchor_output) != 4:
            raise RuntimeError(
                "SAM3 add_new_points_or_box returned an unexpected structure"
            )
        anchor_index, anchor_ids, _, anchor_masks = anchor_output
        anchor_index, anchor_objects = _parse_tracker_frame(
            anchor_index,
            anchor_ids,
            anchor_masks,
            None,
            width,
            height,
        )
        if anchor_index != 0 or set(anchor_objects) != {locked_obj_id}:
            raise ValueError(
                "SAM3 bbox prompt did not lock exactly one object id on frame 0"
            )
        if not anchor_objects[locked_obj_id]["visible"]:
            raise ValueError("SAM3 bbox prompt returned an empty anchor mask")
        threshold = float(sam3_cfg.get("anchor_match_iou", 0.1))
        if (
            _bbox_iou(anchor_objects[locked_obj_id]["bbox_xyxy"], bbox)
            < threshold
        ):
            raise ValueError("SAM3 anchor mask does not overlap the proposal bbox")

        observed: list[int] = []
        duplicate: list[int] = []
        locked_missing: list[int] = []
        empty_masks: list[int] = []
        seen: set[int] = set()
        frame_states: dict[int, dict[str, Any]] = {}
        stream = predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=frame_count,
            reverse=False,
            tqdm_disable=True,
            propagate_preflight=True,
        )
        for frame_output in stream:
            if not isinstance(frame_output, tuple) or len(frame_output) != 5:
                raise ValueError(
                    "SAM3 propagate_in_video returned an unexpected structure"
                )
            frame_index, obj_ids, _, video_res_masks, obj_scores = frame_output
            frame_index, objects = _parse_tracker_frame(
                frame_index,
                obj_ids,
                video_res_masks,
                obj_scores,
                width,
                height,
            )
            if frame_index < 0 or frame_index > terminal:
                raise ValueError("SAM3 propagated an out-of-range frame index")
            if frame_index in seen:
                duplicate.append(frame_index)
                continue
            seen.add(frame_index)
            observed.append(frame_index)
            if set(objects) != {locked_obj_id}:
                locked_missing.append(frame_index)
                continue
            state = objects[locked_obj_id]
            if not state["visible"]:
                empty_masks.append(frame_index)
            if frame_index in required_indices:
                mask_path, mask_sha = _write_track_mask(
                    cache_dir / "masks" / f"frame_{frame_index:06d}.png",
                    state["_mask"],
                )
                frame_states[frame_index] = {
                    "source_frame_index": frame_index,
                    "bbox_xyxy": state["bbox_xyxy"],
                    "mask_path": mask_path,
                    "mask_sha256": mask_sha,
                    "obj_id": locked_obj_id,
                    "score": state["score"],
                    "visible": bool(state["visible"]),
                }

        expected_indices = list(range(frame_count))
        observed_sorted = sorted(observed)
        missing = sorted(set(expected_indices) - set(observed_sorted))
        if duplicate:
            raise ValueError(f"SAM3 emitted duplicate frames: {sorted(set(duplicate))}")
        if missing:
            raise ValueError(f"SAM3 propagation omitted frames: {missing[:10]}")
        if locked_missing:
            raise ValueError(
                f"Locked SAM3 object id changed at frames: {locked_missing[:10]}"
            )
        invisible_terminals = sorted(terminal_indices.intersection(empty_masks))
        if invisible_terminals:
            raise ValueError(
                "SAM3 terminal mask is empty at frames: "
                f"{invisible_terminals}"
            )
        absent_keyframes = sorted(required_indices - set(frame_states))
        if absent_keyframes:
            raise ValueError(f"SAM3 key/terminal frames are missing: {absent_keyframes}")
        result = {
            "schema_version": TRACKED_GROUNDING_CACHE_SCHEMA,
            "cache_key": cache_key,
            "anchor_bbox_xyxy": bbox,
            "locked_obj_id": locked_obj_id,
            "frames": [frame_states[index] for index in sorted(frame_states)],
            "continuity": {
                "expected_frame_indices": expected_indices,
                "observed_frame_indices": observed_sorted,
                "missing_frame_indices": [],
                "duplicate_frame_indices": [],
                "locked_obj_id": locked_obj_id,
                "locked_id_missing_frame_indices": [],
                "empty_mask_frame_indices": empty_masks,
                "id_switch_detected": False,
                "frame_coverage_complete": True,
            },
            "predictor_provenance": copy.deepcopy(dict(predictor_provenance)),
        }
        result["fingerprint"] = _fingerprint_row(result)
    except Exception as exc:
        pending_error = exc
    finally:
        if stream is not None and callable(getattr(stream, "close", None)):
            try:
                stream.close()
            except Exception as exc:
                if pending_error is None:
                    pending_error = RuntimeError(
                        f"SAM3 propagation stream close failed: {exc}"
                    )
        if inference_state is not None:
            try:
                release = getattr(predictor, "release_state", None)
                if callable(release):
                    release(inference_state)
                elif isinstance(inference_state, dict):
                    inference_state.clear()
            except Exception as exc:
                if pending_error is None:
                    pending_error = RuntimeError(
                        f"SAM3 instance state release failed: {exc}"
                    )
    if pending_error is not None:
        raise pending_error
    if result is None:
        raise AssertionError("SAM3 propagation returned no result")
    return result


def _track_candidate(
    predictor: Any,
    predictor_provenance: Mapping[str, Any],
    request: Mapping[str, Any],
    proposal_fingerprint: str,
    candidate: Mapping[str, Any],
    source: str,
    output_dir: Path,
    sam3_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    tracker_fingerprint = str(predictor_provenance["tracker_fingerprint"])
    cache_key = _cache_key(request, candidate, tracker_fingerprint)
    cache_dir = output_dir / "track_cache" / cache_key
    cache_path = cache_dir / "track.json"
    required_indices = {
        int(frame["source_frame_index"]) for frame in request["key_frames"]
    }
    for binding in request["model_frame_bindings"].values():
        required_indices.add(int(binding["terminal"]["source_frame_index"]))
    cached = _load_track_cache(cache_path, cache_key, required_indices)
    cache_hit = cached is not None
    if cached is None:
        cached = _run_visual_propagation(
            predictor,
            request,
            candidate,
            predictor_provenance,
            cache_dir,
            cache_key,
            sam3_cfg,
        )
        write_json(cache_path, cached)
    frame_states = {
        int(frame["source_frame_index"]): frame for frame in cached["frames"]
    }
    key_frames = {
        int(frame["source_frame_index"]): frame for frame in request["key_frames"]
    }
    frames: list[dict[str, Any]] = []
    for frame_index in sorted(required_indices):
        if frame_index not in frame_states:
            raise ValueError(f"Cached SAM3 track lacks required frame {frame_index}")
        source_frame = key_frames.get(frame_index)
        if source_frame is None:
            # Model-only terminal images (notably GRM) are attached below.
            continue
        state = frame_states[frame_index]
        frames.append(
            {
                "source_frame_index": frame_index,
                "image_path": source_frame["image_path"],
                "image_sha256": source_frame["image_sha256"],
                "bbox_xyxy": copy.deepcopy(state["bbox_xyxy"]),
                "mask_path": state["mask_path"],
                "mask_sha256": state["mask_sha256"],
                "obj_id": int(state["obj_id"]),
                "score": float(state["score"]),
                "visible": bool(state["visible"]),
            }
        )
    terminal_by_model: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        binding = request["model_frame_bindings"][model]["terminal"]
        frame_index = int(binding["source_frame_index"])
        state = frame_states.get(frame_index)
        if state is None:
            raise ValueError(f"SAM3 track lacks the {model} terminal frame")
        image_path = _resolved_file(binding["image_path"], f"{model} terminal image")
        if sha256_file(image_path) != binding["image_sha256"]:
            raise ValueError(f"{model} terminal image SHA changed after request freeze")
        terminal_by_model[model] = {
            "source_frame_index": frame_index,
            "image_path": str(image_path),
            "image_sha256": binding["image_sha256"],
            "bbox_xyxy": copy.deepcopy(state["bbox_xyxy"]),
            "mask_path": state["mask_path"],
            "mask_sha256": state["mask_sha256"],
            "obj_id": int(state["obj_id"]),
            "score": float(state["score"]),
            "visible": bool(state["visible"]),
        }
    provenance = copy.deepcopy(dict(predictor_provenance))
    provenance.update(
        {
            "source_video_path": request["video"]["path"],
            "source_video_sha256": request["video"]["sha256"],
            "cache_key": cache_key,
            "cache_path": str(cache_path.resolve()),
            "cache_hit": cache_hit,
        }
    )
    track = {
        "schema_version": TRACKED_GROUNDING_TRACK_SCHEMA,
        "status": "ok",
        "error": None,
        "example_id": request["example_id"],
        "candidate_id": candidate["candidate_id"],
        "source": source,
        "request_fingerprint": request["request_fingerprint"],
        "proposal_fingerprint": proposal_fingerprint,
        "anchor": {
            "source_frame_index": 0,
            "image_path": request["first_frame"]["image_path"],
            "image_sha256": request["first_frame"]["image_sha256"],
            "bbox_xyxy": copy.deepcopy(candidate["bbox_xyxy"]),
            "mask_path": candidate.get("mask_path"),
            "mask_sha256": candidate.get("mask_sha256"),
        },
        "locked_obj_id": int(cached["locked_obj_id"]),
        "frames": frames,
        "continuity": copy.deepcopy(cached["continuity"]),
        "terminal_by_model": terminal_by_model,
        "predictor_provenance": provenance,
    }
    track["fingerprint"] = _fingerprint_row(track)
    return track


def _invalid_track(
    request: Mapping[str, Any],
    proposal_fingerprint: str,
    candidate: Mapping[str, Any],
    source: str,
    error: Exception | str,
    predictor_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    provenance = copy.deepcopy(dict(predictor_provenance or {}))
    provenance.setdefault("source_video_path", request["video"]["path"])
    provenance.setdefault("source_video_sha256", request["video"]["sha256"])
    track = {
        "schema_version": TRACKED_GROUNDING_TRACK_SCHEMA,
        "status": "invalid",
        "error": str(error),
        "example_id": request["example_id"],
        "candidate_id": candidate["candidate_id"],
        "source": source,
        "request_fingerprint": request["request_fingerprint"],
        "proposal_fingerprint": proposal_fingerprint,
        "anchor": {
            "source_frame_index": 0,
            "image_path": request["first_frame"]["image_path"],
            "image_sha256": request["first_frame"]["image_sha256"],
            "bbox_xyxy": copy.deepcopy(candidate["bbox_xyxy"]),
            "mask_path": candidate.get("mask_path"),
            "mask_sha256": candidate.get("mask_sha256"),
        },
        "locked_obj_id": None,
        "frames": [],
        "continuity": {
            "expected_frame_indices": list(range(int(request["video"]["frame_count"]))),
            "observed_frame_indices": [],
            "missing_frame_indices": list(range(int(request["video"]["frame_count"]))),
            "duplicate_frame_indices": [],
            "locked_obj_id": None,
            "locked_id_missing_frame_indices": [],
            "id_switch_detected": False,
            "frame_coverage_complete": False,
        },
        "terminal_by_model": {model: None for model in MODELS},
        "predictor_provenance": provenance,
    }
    track["fingerprint"] = _fingerprint_row(track)
    return track


def _artifact_source(request: Mapping[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(request["source"])
    source["video"] = copy.deepcopy(request["video"])
    return source


def _artifact_base(
    request: Mapping[str, Any],
    proposal: Mapping[str, Any],
    attempt: int,
) -> dict[str, Any]:
    return {
        "schema_version": TRACKED_GROUNDING_ARTIFACT_SCHEMA,
        "example_id": request["example_id"],
        "group_id": request["group_id"],
        "partition": request["partition"],
        "task_id": request["task_id"],
        "attempt": int(attempt),
        "status": "invalid",
        "error": None,
        "request_fingerprint": request["request_fingerprint"],
        "proposal_fingerprint": proposal["fingerprint"],
        "source": _artifact_source(request),
        "first_frame": copy.deepcopy(request["first_frame"]),
        "queries": {
            "target_phrase": request["roles"].get("target_phrase"),
            "reference_object": request["roles"].get("reference_object"),
        },
        "proposal": copy.deepcopy(dict(proposal)),
        "candidate_tracks": [],
        "terminal_by_model": {model: None for model in MODELS},
        "selected_candidate_id": None,
        "selection_source": None,
    }


def _invalid_proposal(
    request: Mapping[str, Any],
    error: Exception | str,
    provider_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = {
        "schema_version": TRACKED_GROUNDING_PROPOSAL_SCHEMA,
        "status": "invalid",
        "strategy": str(request["roles"].get("grounding_strategy", "")),
        "target_query": request["roles"].get("target_phrase"),
        "reference_query": request["roles"].get("reference_object"),
        "target_candidates": [],
        "reference_candidates": [],
        "excluded_target_candidates": [],
        "reference_exclusion_iou": None,
        "algorithmic_default": None,
        "options": [],
        "review_reasons": ["proposal_backend_failed"],
        "error": str(error),
        "provider_provenance": copy.deepcopy(dict(provider_provenance or {})),
    }
    proposal["fingerprint"] = _fingerprint_row(proposal)
    return proposal


def _automated_artifact(
    request: Mapping[str, Any],
    proposal: Mapping[str, Any],
    attempt: int,
    predictor: Any | None,
    predictor_provenance: Mapping[str, Any] | None,
    predictor_error: Exception | None,
    output_dir: Path,
    sam3_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = _artifact_base(request, proposal, attempt)
    if proposal["status"] == "invalid":
        artifact["error"] = proposal.get("error", "No valid target proposal")
        artifact["fingerprint"] = _fingerprint_row(artifact)
        return artifact
    tracks: list[dict[str, Any]] = []
    for candidate in proposal["options"]:
        if predictor_error is not None or predictor is None:
            track = _invalid_track(
                request,
                proposal["fingerprint"],
                candidate,
                "sam3_candidate",
                predictor_error or "SAM3 predictor unavailable",
                predictor_provenance,
            )
        else:
            try:
                track = _track_candidate(
                    predictor,
                    predictor_provenance or {},
                    request,
                    proposal["fingerprint"],
                    candidate,
                    "sam3_candidate",
                    output_dir,
                    sam3_cfg,
                )
            except Exception as exc:
                track = _invalid_track(
                    request,
                    proposal["fingerprint"],
                    candidate,
                    "sam3_candidate",
                    exc,
                    predictor_provenance,
                )
        tracks.append(track)
    artifact["candidate_tracks"] = tracks
    valid = {track["candidate_id"]: track for track in tracks if track["status"] == "ok"}
    default = proposal.get("algorithmic_default")
    default_id = default.get("candidate_id") if isinstance(default, dict) else None
    if default_id and default_id in valid:
        chosen = valid[default_id]
        artifact["status"] = "ok"
        artifact["selected_candidate_id"] = default_id
        artifact["selection_source"] = "algorithmic_default"
        artifact["terminal_by_model"] = copy.deepcopy(chosen["terminal_by_model"])
    elif valid:
        artifact["status"] = "needs_review"
        artifact["error"] = (
            "Algorithmic default unavailable or invalid; a reviewed alternative is required"
        )
    else:
        artifact["status"] = "invalid"
        artifact["error"] = "All candidate tracks failed closed"
    artifact["fingerprint"] = _fingerprint_row(artifact)
    return artifact


def _latest_artifacts(path: Path, key: str = "example_id") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for row in read_jsonl(path):
        identity = str(row.get(key, ""))
        if not identity:
            raise ValueError(f"Artifact in {path} lacks {key}")
        if row.get("fingerprint") != _fingerprint_row(row):
            raise ValueError(f"Artifact fingerprint mismatch for {identity}")
        result[identity] = row
    return result


def _validate_frozen_request(
    request: Mapping[str, Any],
    file_hash_cache: dict[str, str] | None = None,
) -> None:
    cache = file_hash_cache if file_hash_cache is not None else {}

    def digest(path: str | Path) -> str:
        resolved = str(Path(path).resolve())
        if resolved not in cache:
            cache[resolved] = sha256_file(resolved)
        return cache[resolved]

    if request.get("schema_version") != TRACKED_GROUNDING_REQUEST_SCHEMA:
        raise ValueError("Unsupported tracked-grounding request schema")
    if request.get("request_fingerprint") != _fingerprint_row(
        request, "request_fingerprint"
    ):
        raise ValueError(f"{request.get('example_id')}: request fingerprint mismatch")
    video_path = _resolved_file(request["video"]["path"], "frozen source video")
    if digest(video_path) != request["video"]["sha256"]:
        raise ValueError(f"{request.get('example_id')}: source video changed after freeze")
    first = request["first_frame"]
    first_path = _resolved_file(first["image_path"], "frozen first frame")
    if digest(first_path) != first["image_sha256"]:
        raise ValueError(f"{request.get('example_id')}: first frame changed after freeze")
    split = request.get("source", {}).get("split")
    if not isinstance(split, dict):
        raise ValueError(f"{request.get('example_id')}: split provenance is missing")
    split_path = _resolved_file(split["path"], "frozen whitebox split")
    if digest(split_path) != split["sha256"]:
        raise ValueError(f"{request.get('example_id')}: frozen split changed")
    if split.get("partition") != request.get("partition"):
        raise ValueError(f"{request.get('example_id')}: split partition binding mismatch")
    if split.get("assignment_fingerprint") != object_fingerprint(
        {"example_id": request["example_id"], "partition": request["partition"]}
    ):
        raise ValueError(f"{request.get('example_id')}: split assignment fingerprint mismatch")
    for model in MODELS:
        binding = request.get("model_frame_bindings", {}).get(model)
        if not isinstance(binding, dict):
            raise ValueError(f"{request.get('example_id')}: missing {model} frame binding")
        terminal = binding.get("terminal")
        if not isinstance(terminal, dict):
            raise ValueError(f"{request.get('example_id')}: missing {model} terminal")
        if int(terminal.get("source_frame_index", -1)) != int(
            request["video"]["frame_count"]
        ) - 1:
            raise ValueError(f"{request.get('example_id')}: {model} terminal index drift")
        terminal_path = _resolved_file(terminal["image_path"], f"{model} terminal")
        if digest(terminal_path) != terminal["image_sha256"]:
            raise ValueError(f"{request.get('example_id')}: {model} terminal SHA drift")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest root must be an object: {path}")
    return value


def run_tracked_grounding(
    config: dict[str, Any],
    retry_failed: bool = False,
    *,
    shard_id: int | None = None,
    num_shards: int | None = None,
) -> Path:
    """Propose on frame zero and propagate every retained target candidate."""
    if (shard_id is None) != (num_shards is None):
        raise ValueError("shard_id and num_shards must be provided together")
    if shard_id is not None and (
        num_shards is None
        or num_shards < 1
        or shard_id < 0
        or shard_id >= num_shards
    ):
        raise ValueError("tracking shard must satisfy 0 <= shard_id < num_shards")
    cfg, sam3_cfg = _cfg(config)
    output_dir = Path(str(cfg["output_dir"])).expanduser().resolve()
    requests_path = output_dir / "requests.jsonl"
    if not requests_path.is_file():
        requests_path = build_tracked_grounding_requests(config)
    requests = list(read_jsonl(requests_path))
    if not requests:
        raise ValueError("Tracked-grounding request manifest is empty")
    validation_hashes: dict[str, str] = {}
    for request in requests:
        _validate_frozen_request(request, validation_hashes)
    tracks_path = output_dir / "tracks.jsonl"
    previous = _latest_artifacts(tracks_path)
    pending: list[tuple[dict[str, Any], int]] = []
    for request in requests:
        if shard_id is not None and stable_shard(
            str(request["video"]["sha256"]),
            int(num_shards),
        ) != shard_id:
            continue
        old = previous.get(str(request["example_id"]))
        if old and old.get("request_fingerprint") == request["request_fingerprint"]:
            if old.get("status") in {"ok", "needs_review"}:
                continue
            if old.get("status") == "invalid" and not retry_failed:
                continue
        pending.append((request, int(old.get("attempt", 0)) + 1 if old else 1))

    provider: Any | None = None
    predictor: Any | None = None
    provider_error: Exception | None = None
    provider_shutdown_error: Exception | None = None
    provider_provenance: dict[str, Any] | None = None
    predictor_error: Exception | None = None
    predictor_provenance: dict[str, Any] | None = None
    prepared: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    if pending:
        try:
            provider = _candidate_provider_from_config(sam3_cfg)
            provider_provenance = _candidate_provider_provenance(provider, sam3_cfg)
        except Exception as exc:
            provider_error = exc
        try:
            for request, attempt in pending:
                if provider_error is not None or provider is None:
                    proposal = _invalid_proposal(
                        request,
                        provider_error or "SAM3 candidate provider unavailable",
                        provider_provenance,
                    )
                else:
                    try:
                        proposal = _propose(
                            request,
                            provider,
                            provider_provenance or {},
                            output_dir,
                            sam3_cfg,
                        )
                    except Exception as exc:
                        proposal = _invalid_proposal(
                            request, exc, provider_provenance
                        )
                prepared.append((request, attempt, proposal))
        finally:
            if provider is not None and callable(getattr(provider, "shutdown", None)):
                try:
                    provider.shutdown()
                except Exception as exc:
                    provider_shutdown_error = exc
            # Do not keep the image proposal model resident while constructing
            # the official video predictor.
            provider = None
            gc.collect()
            torch_module = sys.modules.get("torch")
            cuda = getattr(torch_module, "cuda", None) if torch_module else None
            if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
                cuda.empty_cache()
        if any(proposal.get("options") for _, _, proposal in prepared):
            try:
                predictor, predictor_provenance = _predictor_from_config(sam3_cfg)
            except Exception as exc:
                predictor_error = exc
    shutdown_error: Exception | None = None
    try:
        for request, attempt, proposal in prepared:
            artifact = _automated_artifact(
                request,
                proposal,
                attempt,
                predictor,
                predictor_provenance,
                predictor_error,
                output_dir,
                sam3_cfg,
            )
            append_jsonl(tracks_path, artifact)
    finally:
        if predictor is not None and callable(getattr(predictor, "shutdown", None)):
            try:
                predictor.shutdown()
            except Exception as exc:
                shutdown_error = exc

    if shard_id is not None:
        if shutdown_error is not None:
            raise RuntimeError(
                f"SAM3 predictor shutdown failed: {shutdown_error}"
            )
        if provider_shutdown_error is not None:
            raise RuntimeError(
                "SAM3 candidate provider shutdown failed: "
                f"{provider_shutdown_error}"
            )
        return tracks_path
    latest = _latest_artifacts(tracks_path)
    counts = Counter(str(row.get("status")) for row in latest.values())
    expected_ids = {str(request["example_id"]) for request in requests}
    coverage_complete = set(latest) == expected_ids
    request_bindings_current = coverage_complete and all(
        latest[str(request["example_id"])].get("request_fingerprint")
        == request["request_fingerprint"]
        for request in requests
    )
    manifest_path = output_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    if not manifest:
        inputs_path = _resolved_file(cfg["inputs_path"], "model input manifest")
        roles_path = _resolved_file(cfg["roles_path"], "semantic role manifest")
        split_path = _resolved_file(cfg["split_path"], "frozen whitebox split")
        manifest = _manifest_base(cfg, inputs_path, roles_path, split_path)
    tracker = None
    if predictor_provenance is not None:
        tracker = {
            "backend": "official_sam3_sam2_style_instance_tracker",
            "official_source_path": predictor_provenance["official_source_path"],
            "checkpoint_path": predictor_provenance["checkpoint_path"],
            "tracker_fingerprint": predictor_provenance["tracker_fingerprint"],
        }
    elif isinstance(manifest.get("tracker"), dict):
        tracker = manifest["tracker"]
    if predictor_error is not None:
        tracker = {"backend": "official_sam3_sam2_style_instance_tracker", "error": str(predictor_error)}
    manifest.update(
        {
            "status": "complete"
            if coverage_complete
            and request_bindings_current
            and provider_error is None
            and predictor_error is None
            and shutdown_error is None
            and provider_shutdown_error is None
            else "invalid",
            "requests_path": str(requests_path.resolve()),
            "requests_sha256": sha256_file(requests_path),
            "request_count": len(requests),
            "tracks_path": str(tracks_path.resolve()) if tracks_path.is_file() else None,
            "tracks_sha256": sha256_file(tracks_path) if tracks_path.is_file() else None,
            "artifact_count": len(latest),
            "artifact_row_count": sum(1 for _ in read_jsonl(tracks_path))
            if tracks_path.is_file()
            else 0,
            "status_counts": dict(sorted(counts.items())),
            "coverage_complete": coverage_complete,
            "request_bindings_current": request_bindings_current,
            "tracker": tracker,
            "proposal_backend": provider_provenance
            if provider_provenance is not None
            else manifest.get("proposal_backend"),
            "labels_opened": False,
        }
    )
    for stale_error in (
        "proposal_backend_error",
        "proposal_backend_shutdown_error",
        "tracker_shutdown_error",
    ):
        manifest.pop(stale_error, None)
    if provider_error is not None:
        manifest["proposal_backend_error"] = str(provider_error)
    if provider_shutdown_error is not None:
        manifest["proposal_backend_shutdown_error"] = str(provider_shutdown_error)
    if shutdown_error is not None:
        manifest["tracker_shutdown_error"] = str(shutdown_error)
    write_json(manifest_path, manifest)
    if shutdown_error is not None:
        raise RuntimeError(f"SAM3 predictor shutdown failed: {shutdown_error}")
    if provider_shutdown_error is not None:
        raise RuntimeError(
            f"SAM3 candidate provider shutdown failed: {provider_shutdown_error}"
        )
    return tracks_path


def _manual_anchors(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("schema_version") != TRACKED_GROUNDING_MANUAL_ANCHOR_SCHEMA:
            raise ValueError("Unsupported manual-anchor schema")
        anchor_id = str(row.get("manual_anchor_id", "")).strip()
        if not anchor_id:
            raise ValueError("Manual anchor lacks manual_anchor_id")
        if row.get("fingerprint") != _fingerprint_row(row):
            raise ValueError(f"Manual anchor fingerprint mismatch: {anchor_id}")
        latest[anchor_id] = row
    return [latest[key] for key in sorted(latest)]


def _validate_manual_anchor(
    anchor: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if str(anchor.get("example_id")) != str(request["example_id"]):
        raise ValueError("Manual anchor example_id does not bind to the request")
    if anchor.get("request_fingerprint") != request["request_fingerprint"]:
        raise ValueError("Manual anchor request fingerprint is stale")
    if int(anchor.get("first_frame_index", -1)) != 0:
        raise ValueError("Manual anchor must be drawn on source frame 0")
    if str(Path(str(anchor.get("first_image_path", ""))).resolve()) != str(
        Path(request["first_frame"]["image_path"]).resolve()
    ):
        raise ValueError("Manual anchor first-frame path differs from the request")
    if anchor.get("first_image_sha256") != request["first_frame"]["image_sha256"]:
        raise ValueError("Manual anchor first-frame SHA differs from the request")
    if sha256_file(anchor["first_image_path"]) != anchor["first_image_sha256"]:
        raise ValueError("Manual anchor first-frame bytes changed")
    reviewer = str(anchor.get("reviewer_id", "")).strip()
    if not reviewer:
        raise ValueError("Manual anchor requires a non-empty reviewer_id")
    bbox = _valid_bbox(
        anchor.get("bbox_xyxy"),
        int(request["first_frame"]["width"]),
        int(request["first_frame"]["height"]),
    )
    if bbox is None:
        raise ValueError("Manual anchor bbox is invalid or out of bounds")
    expected_anchor_id = derive_manual_anchor_id(
        str(request["example_id"]),
        str(request["first_frame"]["image_sha256"]),
        bbox,
    )
    if anchor.get("manual_anchor_id") != expected_anchor_id:
        raise ValueError("Manual anchor ID does not match its image and bbox")
    return {
        "candidate_id": str(anchor["manual_anchor_id"]),
        "bbox_xyxy": bbox,
        "mask_path": None,
        "mask_sha256": None,
    }


def run_manual_retracks(
    config: dict[str, Any],
    anchors_path: str | Path,
    output_path: str | Path,
    retry_failed: bool = False,
) -> Path:
    """Propagate reviewed frame-zero boxes; never copy them to the terminal."""
    cfg, sam3_cfg = _cfg(config)
    output_dir = Path(str(cfg["output_dir"])).expanduser().resolve()
    requests_path = output_dir / "requests.jsonl"
    tracks_path = output_dir / "tracks.jsonl"
    anchors_path = _resolved_file(anchors_path, "manual anchor queue")
    output_path = Path(output_path).expanduser().resolve()
    if not requests_path.is_file() or not tracks_path.is_file():
        raise FileNotFoundError("Automated requests/tracks must exist before manual retracking")
    requests = {str(row["example_id"]): row for row in read_jsonl(requests_path)}
    automated = _latest_artifacts(tracks_path)
    anchors = _manual_anchors(anchors_path)
    previous = _latest_artifacts(output_path, key="selected_candidate_id")
    pending: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]] = []
    validation_hashes: dict[str, str] = {}
    for anchor in anchors:
        example_id = str(anchor["example_id"])
        request = requests.get(example_id)
        artifact = automated.get(example_id)
        if request is None or artifact is None:
            raise ValueError(f"Manual anchor has no automated request/artifact: {example_id}")
        _validate_frozen_request(request, validation_hashes)
        candidate = _validate_manual_anchor(anchor, request)
        old = previous.get(candidate["candidate_id"])
        if old and old.get("request_fingerprint") == request["request_fingerprint"]:
            if old.get("status") == "ok":
                continue
            if old.get("status") == "invalid" and not retry_failed:
                continue
        pending.append(
            (
                anchor,
                request,
                candidate,
                int(old.get("attempt", 0)) + 1 if old else 1,
            )
        )

    predictor: Any | None = None
    predictor_provenance: dict[str, Any] | None = None
    predictor_error: Exception | None = None
    if pending:
        try:
            predictor, predictor_provenance = _predictor_from_config(sam3_cfg)
        except Exception as exc:
            predictor_error = exc
    shutdown_error: Exception | None = None
    try:
        for anchor, request, candidate, attempt in pending:
            original = automated[str(request["example_id"])]
            proposal = original["proposal"]
            artifact = _artifact_base(request, proposal, attempt)
            artifact["manual_anchor"] = copy.deepcopy(anchor)
            artifact["selected_candidate_id"] = candidate["candidate_id"]
            artifact["selection_source"] = "manual_bbox"
            if predictor_error is not None or predictor is None:
                track = _invalid_track(
                    request,
                    proposal["fingerprint"],
                    candidate,
                    "manual_bbox",
                    predictor_error or "SAM3 predictor unavailable",
                    predictor_provenance,
                )
            else:
                try:
                    track = _track_candidate(
                        predictor,
                        predictor_provenance or {},
                        request,
                        proposal["fingerprint"],
                        candidate,
                        "manual_bbox",
                        output_dir,
                        sam3_cfg,
                    )
                except Exception as exc:
                    track = _invalid_track(
                        request,
                        proposal["fingerprint"],
                        candidate,
                        "manual_bbox",
                        exc,
                        predictor_provenance,
                    )
            track["anchor"]["manual_anchor_id"] = anchor["manual_anchor_id"]
            track["anchor"]["reviewer_id"] = anchor["reviewer_id"]
            track["fingerprint"] = _fingerprint_row(track)
            artifact["candidate_tracks"] = [track]
            if track["status"] == "ok":
                artifact["status"] = "ok"
                artifact["terminal_by_model"] = copy.deepcopy(track["terminal_by_model"])
            else:
                artifact["status"] = "invalid"
                artifact["error"] = "Manual visual anchor failed SAM3 propagation"
            artifact["fingerprint"] = _fingerprint_row(artifact)
            append_jsonl(output_path, artifact)
    finally:
        if predictor is not None and callable(getattr(predictor, "shutdown", None)):
            try:
                predictor.shutdown()
            except Exception as exc:
                shutdown_error = exc
    if shutdown_error is not None:
        raise RuntimeError(f"SAM3 predictor shutdown failed: {shutdown_error}")
    return output_path


__all__ = [
    "TRACKED_GROUNDING_REQUEST_SCHEMA",
    "TRACKED_GROUNDING_PROPOSAL_SCHEMA",
    "TRACKED_GROUNDING_TRACK_SCHEMA",
    "TRACKED_GROUNDING_ARTIFACT_SCHEMA",
    "TRACKED_GROUNDING_MANUAL_ANCHOR_SCHEMA",
    "build_tracked_grounding_requests",
    "run_tracked_grounding",
    "run_manual_retracks",
]
