"""Build and audit label-free, processor-aligned grounding requests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2

from ..config import section
from ..io import (
    append_jsonl,
    latest_by_id,
    object_fingerprint,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .data import FORBIDDEN_MODEL_FIELDS, load_model_inputs
from .media import grm_multiview_image_paths


GROUNDING_REQUEST_SCHEMA = "my_dataset.grounding_request.v1"
GROUNDING_REVIEW_SCHEMA = "my_dataset.grounding_review.v1"


def _run_records(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir).resolve()
    paths = sorted(root.glob("records.shard-*.jsonl"))
    if not paths:
        path = root / "records.jsonl"
        paths = [path] if path.is_file() else []
    if not paths:
        raise FileNotFoundError(f"No baseline records in {root}")
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return list(latest_by_id(rows).values())


def _by_group(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in records:
        if row.get("status") != "ok":
            continue
        group_id = str(row["group_id"])
        previous = result.get(group_id)
        if previous is not None:
            first = previous.get("input_diagnostics", {})
            current = row.get("input_diagnostics", {})
            if first != current or previous.get("frame_record") != row.get("frame_record"):
                raise ValueError(f"Input diagnostics vary within group {group_id}")
        result[group_id] = row
    return result


def _extract_frame(video_path: str | Path, index: int, destination: Path) -> Path:
    if destination.is_file():
        return destination
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {index} from {video_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Cannot write extracted frame {destination}")
    return destination


def _native_frames(
    model: str,
    row: dict[str, Any],
    input_row: dict[str, Any],
    frames_root: Path,
) -> dict[str, Any]:
    diagnostics = row.get("input_diagnostics", {})
    metadata = diagnostics.get("video_metadata", {})
    indices = metadata.get("frames_indices")
    total = metadata.get("total_num_frames")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"{model} did not record processor frames_indices")
    terminal = int(indices[-1])
    if int(total or 0) < 1 or terminal != int(total) - 1:
        raise ValueError(f"{model} native processor did not include the terminal frame")
    video = Path(input_row["video_paths"]["front"]).resolve()
    image = _extract_frame(
        video,
        terminal,
        frames_root / model / str(input_row["group_id"]) / f"frame_{terminal:06d}.png",
    )
    return {
        "input_layout": "native_front_video",
        "video_path": str(video),
        "view": "front",
        "sampled_frame_indices": [int(value) for value in indices],
        "source_frame_index": terminal,
        "image_path": str(image),
        "image_sha256": sha256_file(image),
        "video_grid_thw": diagnostics.get("video_grid_thw"),
        "content_order": diagnostics.get("content_order"),
    }


def _grm_frames(row: dict[str, Any], blank_goal: str | Path) -> dict[str, Any]:
    frames = row.get("frame_record")
    if not isinstance(frames, dict):
        raise ValueError("GRM baseline record has no multiview frame_record")
    paths = grm_multiview_image_paths(frames, blank_goal)
    terminal = {
        view: {
            "view": view,
            "source_frame_index": int(frames[view]["last_index"]),
            "image_path": str(Path(frames[view]["last_path"]).resolve()),
            "image_sha256": str(frames[view]["last_sha256"]),
        }
        for view in ("front", "left_wrist", "right_wrist")
    }
    return {
        "input_layout": "grm_native_three_view_endpoints_v1",
        "image_paths": paths,
        "terminal_views": terminal,
        "primary_target_view": "front",
        "primary_target_slot": "after_cam_high",
    }


def build_grounding_manifest(config: dict[str, Any]) -> Path:
    cfg = section(config, "my_dataset_grounding")
    inputs_path = Path(cfg["inputs_path"]).resolve()
    roles_path = Path(cfg["roles_path"]).resolve()
    split_path = Path(cfg["split_path"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    rows = load_model_inputs(inputs_path)
    roles = {str(row["example_id"]): row for row in read_jsonl(roles_path)}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    partition = {
        example_id: name
        for name, example_ids in split["examples"].items()
        for example_id in example_ids
    }
    baseline_runs = dict(cfg.get("baseline_runs", {}))
    required = {"roboreward", "qwen", "grm"}
    if set(baseline_runs) != required:
        raise ValueError(f"baseline_runs must contain exactly {sorted(required)}")
    run_rows = {
        model: _by_group(_run_records(path)) for model, path in baseline_runs.items()
    }
    representative: dict[str, dict[str, Any]] = {}
    for row in rows:
        representative.setdefault(str(row["group_id"]), row)
    frames_by_group = {}
    frames_root = output_dir / "frames"
    for group_id, input_row in sorted(representative.items()):
        missing = [model for model in required if group_id not in run_rows[model]]
        if missing:
            raise ValueError(f"Missing baseline records for {group_id}: {missing}")
        frames_by_group[group_id] = {
            "roboreward": _native_frames(
                "roboreward", run_rows["roboreward"][group_id], input_row, frames_root
            ),
            "qwen": _native_frames(
                "qwen", run_rows["qwen"][group_id], input_row, frames_root
            ),
            "grm": _grm_frames(run_rows["grm"][group_id], cfg["blank_goal"]),
        }
    requests = []
    for row in rows:
        example_id = str(row["example_id"])
        role = roles.get(example_id)
        if role is None:
            raise ValueError(f"Missing semantic role for {example_id}")
        request = {
            "schema_version": GROUNDING_REQUEST_SCHEMA,
            "example_id": example_id,
            "group_id": str(row["group_id"]),
            "partition": partition[example_id],
            "task_id": str(row["task_id"]),
            "instruction": str(row["instruction"]),
            "roles": {
                key: role.get(key)
                for key in (
                    "grounding_strategy",
                    "manipulated_object",
                    "attribute",
                    "reference_object",
                    "destination",
                    "relation",
                    "ordinal",
                    "direction",
                    "target_phrase",
                    "target_instance",
                    "requires_instance_review",
                )
            },
            "model_frames": frames_by_group[str(row["group_id"])],
            "requested_regions": ["manipulated_object", "wrong_object_or_background"],
        }
        if FORBIDDEN_MODEL_FIELDS & request.keys():
            raise AssertionError("Label field entered grounding request")
        requests.append(request)
    requests.sort(key=lambda row: row["example_id"])
    path = output_dir / "requests.jsonl"
    write_jsonl(path, requests)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": GROUNDING_REQUEST_SCHEMA,
            "inputs_sha256": sha256_file(inputs_path),
            "roles_sha256": sha256_file(roles_path),
            "split_sha256": sha256_file(split_path),
            "request_count": len(requests),
            "group_count": len(representative),
            "request_fingerprint": object_fingerprint(requests),
            "baseline_runs": {key: str(Path(value).resolve()) for key, value in baseline_runs.items()},
            "labels_opened": False,
        },
    )
    return path


def propose_grounding(config: dict[str, Any], *, retry_failed: bool = False) -> Path:
    """Run SAM3 proposals without auto-accepting any instance-level target."""
    cfg = section(config, "my_dataset_grounding")
    sam3_cfg = section(config, "sam3")
    output_dir = Path(cfg["output_dir"]).resolve()
    requests_path = output_dir / "requests.jsonl"
    if not requests_path.is_file():
        raise FileNotFoundError("Run ground-prepare before ground-propose")
    from ..grounding.sam3 import SAM3Grounder

    grounder = SAM3Grounder(sam3_cfg)
    proposals_path = output_dir / "proposals.jsonl"
    previous = (
        {
            (str(row["example_id"]), str(row["model_family"])): row
            for row in read_jsonl(proposals_path)
        }
        if proposals_path.is_file()
        else {}
    )
    cache: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    rows = []
    for request in read_jsonl(requests_path):
        roles = request["roles"]
        queries = []
        for role, value in (
            ("manipulated_object", roles.get("target_phrase")),
            ("reference_object", roles.get("reference_object")),
            ("destination", roles.get("destination")),
        ):
            if value and value not in {query["text"] for query in queries}:
                queries.append({"role": role, "text": str(value)})
        query_text = [query["text"] for query in queries]
        for model in ("roboreward", "qwen", "grm"):
            key = (str(request["example_id"]), model)
            old = previous.get(key)
            if old and old.get("status") == "ok":
                rows.append(old)
                continue
            if old and not retry_failed:
                raise RuntimeError(
                    f"Previous SAM3 proposal failed for {key}; inspect {proposals_path} "
                    "before using --retry-failed"
                )
            frame = request["model_frames"][model]
            image_path = (
                frame["image_path"]
                if model in {"roboreward", "qwen"}
                else frame["terminal_views"]["front"]["image_path"]
            )
            image_digest = sha256_file(image_path)
            cache_key = (image_digest, tuple(query_text))
            try:
                if cache_key not in cache:
                    candidates = grounder.candidates(image_path, query_text)
                    serializable = []
                    query_digest = object_fingerprint(query_text)
                    for index, candidate in enumerate(candidates):
                        value = dict(candidate)
                        mask = value.pop("_mask", None)
                        mask_path = None
                        if mask is not None:
                            mask_file = (
                                output_dir
                                / "proposal_masks"
                                / image_digest
                                / query_digest
                                / f"candidate_{index:02d}.png"
                            )
                            mask_file.parent.mkdir(parents=True, exist_ok=True)
                            if not cv2.imwrite(str(mask_file), mask.astype("uint8") * 255):
                                raise RuntimeError(f"Cannot write SAM3 mask {mask_file}")
                            mask_path = str(mask_file.resolve())
                        value["mask_path"] = mask_path
                        value["candidate_index"] = index
                        serializable.append(value)
                    cache[cache_key] = serializable
                candidates = cache[cache_key]
                row = {
                    "schema_version": GROUNDING_REQUEST_SCHEMA,
                    "example_id": request["example_id"],
                    "group_id": request["group_id"],
                    "partition": request["partition"],
                    "model_family": model,
                    "image_path": str(Path(image_path).resolve()),
                    "image_sha256": image_digest,
                    "queries": queries,
                    "candidates": candidates,
                    "requires_instance_review": bool(roles["requires_instance_review"]),
                    "auto_accepted": False,
                    "status": "ok",
                }
            except Exception as exc:
                row = {
                    "schema_version": GROUNDING_REQUEST_SCHEMA,
                    "example_id": request["example_id"],
                    "group_id": request["group_id"],
                    "partition": request["partition"],
                    "model_family": model,
                    "status": "invalid",
                    "error": str(exc),
                }
            append_jsonl(proposals_path, row)
            if row["status"] != "ok":
                raise RuntimeError(f"SAM3 proposal failed for {key}: {row['error']}")
            rows.append(row)
    by_example: dict[str, dict[str, Any]] = {}
    for row in rows:
        template = by_example.setdefault(
            str(row["example_id"]),
            {
                "schema_version": GROUNDING_REVIEW_SCHEMA,
                "example_id": row["example_id"],
                "status": "pending",
                "review_id": None,
                "models": {},
            },
        )
        template["models"][row["model_family"]] = {
            "proposal_image_path": row["image_path"],
            "target": None,
            "wrong_region": None,
        }
    write_jsonl(output_dir / "review_template.jsonl", by_example.values())
    write_json(
        output_dir / "proposal_manifest.json",
        {
            "backend": "sam3",
            "backend_fingerprint": grounder.fingerprint,
            "request_sha256": sha256_file(requests_path),
            "proposal_count": len(rows),
            "example_count": len(by_example),
            "auto_accepted": False,
            "labels_opened": False,
        },
    )
    return proposals_path


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
        and value[2] > value[0]
        and value[3] > value[1]
    )


def _legacy_audit_grounding_review(requests_path: str | Path, reviews_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    requests = {str(row["example_id"]): row for row in read_jsonl(requests_path)}
    reviews = {str(row["example_id"]): row for row in read_jsonl(reviews_path)}
    unknown = sorted(set(reviews) - set(requests))
    missing = sorted(set(requests) - set(reviews))
    invalid = []
    dispositions = Counter()
    for example_id, review in reviews.items():
        status = str(review.get("status", ""))
        dispositions[status] += 1
        if status not in {"eligible", "ineligible"}:
            invalid.append({"example_id": example_id, "reason": "invalid_status"})
            continue
        if status == "eligible":
            models = review.get("models")
            if not isinstance(models, dict) or set(models) != {"roboreward", "qwen", "grm"}:
                invalid.append({"example_id": example_id, "reason": "missing_model_reviews"})
                continue
            for model, value in models.items():
                target = value.get("target", {}) if isinstance(value, dict) else {}
                if not _valid_bbox(target.get("bbox")) or not Path(str(target.get("image_path", ""))).is_file():
                    invalid.append({"example_id": example_id, "model": model, "reason": "invalid_target"})
                wrong = value.get("wrong_region", {}) if isinstance(value, dict) else {}
                if not _valid_bbox(wrong.get("bbox")) or not Path(
                    str(wrong.get("image_path", ""))
                ).is_file():
                    invalid.append(
                        {"example_id": example_id, "model": model, "reason": "invalid_wrong_region"}
                    )
    result = {
        "schema_version": GROUNDING_REVIEW_SCHEMA,
        "passed": not unknown and not missing and not invalid,
        "request_count": len(requests),
        "review_count": len(reviews),
        "unknown_example_ids": unknown,
        "missing_example_ids": missing,
        "invalid": invalid,
        "dispositions": dict(sorted(dispositions.items())),
        "review_sha256": sha256_file(reviews_path),
    }
    write_json(Path(output_dir) / "review_audit.json", result)
    return result


def audit_grounding_review(
    requests_path: str | Path,
    reviews_path: str | Path,
    output_dir: str | Path,
    *,
    proposals_path: str | Path | None = None,
    tracking_artifact_path: str | Path | None = None,
    manual_tracking_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the strict source-bound human-review audit gate."""
    if tracking_artifact_path is not None:
        if proposals_path is not None:
            raise ValueError(
                "proposals_path and tracking_artifact_path are mutually exclusive"
            )
        from .review_audit import audit_tracked_grounding_review

        return audit_tracked_grounding_review(
            requests_path,
            reviews_path,
            output_dir,
            tracking_artifact_path=tracking_artifact_path,
            manual_tracking_artifact_path=manual_tracking_artifact_path,
        )
    if manual_tracking_artifact_path is not None:
        raise ValueError(
            "manual_tracking_artifact_path requires tracking_artifact_path"
        )
    from .review_audit import audit_grounding_review as strict_audit

    return strict_audit(
        requests_path,
        reviews_path,
        output_dir,
        proposals_path=proposals_path,
    )
