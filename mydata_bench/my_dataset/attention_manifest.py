"""Convert accepted grounding reviews into model-specific causal manifests."""

from __future__ import annotations

import json
import math
from collections import Counter
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from ..config import section
from ..io import object_fingerprint, read_jsonl, sha256_file, write_json, write_jsonl
from .data import FORBIDDEN_MODEL_FIELDS, load_model_inputs
from .grounding_contract import AUTO_UNREVIEWED, HUMAN_REVIEWED
from .review_provenance import (
    TRACKED_REVIEW_SOURCE_KINDS,
    build_review_provenance,
    build_tracking_review_provenance,
)
from .tracked_grounding import (
    TRACKED_GROUNDING_REQUEST_SCHEMA,
    validate_processor_content_order_contract,
)


ATTENTION_INPUT_SCHEMA = "my_dataset.attention_input.v1"
GROUNDING_REQUEST_SCHEMA = "my_dataset.grounding_request.v1"
GROUNDING_REVIEW_SCHEMA = "my_dataset.grounding_review.v1"
GROUNDING_REVIEW_AUDIT_SCHEMA = "my_dataset.grounding_review_audit.v2"
MODELS = ("roboreward", "qwen", "grm")
PARTITIONS = ("discovery", "validation", "test")
SUPPORTED_REVIEW_STATUSES = ("eligible", "assumed_valid", "assumed_proxy")
ASSUMED_STATUS_CONTRACT = {
    "assumed_valid": ("strict", "auto_assumed_unreviewed"),
    "assumed_proxy": ("proxy", "auto_proxy_unreviewed"),
}


def _accepted_statuses(cfg: dict[str, Any]) -> list[str]:
    value = cfg.get("accepted_review_statuses", ["eligible"])
    if not isinstance(value, list) or not value:
        raise ValueError("accepted_review_statuses must be a non-empty list")
    statuses = [str(item) for item in value]
    if len(set(statuses)) != len(statuses):
        raise ValueError("accepted_review_statuses must not contain duplicates")
    unsupported = sorted(set(statuses) - set(SUPPORTED_REVIEW_STATUSES))
    if unsupported:
        raise ValueError(f"Unsupported accepted review statuses: {unsupported}")
    return statuses


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _bbox(value: Any, identity: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(_finite_number(item) for item in value)
    ):
        raise ValueError(f"{identity}: target bbox must contain four finite numbers")
    bbox = [float(item) for item in value]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{identity}: target bbox has non-positive area")
    return bbox


def _configured_sha256(
    cfg: Mapping[str, Any],
    field: str,
    *,
    required: bool,
    audit: Mapping[str, Any] | None = None,
    audit_field: str | None = None,
) -> str | None:
    raw = cfg.get(field)
    if raw == "from_audit":
        if audit is None:
            raise ValueError(f"{field}=from_audit requires a loaded review audit")
        raw = audit.get(audit_field or field)
    if raw in (None, ""):
        if required:
            raise ValueError(f"{field} is required for audited human grounding")
        return None
    value = str(raw).lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256")
    return value


def _bbox_intersection_area(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _strict_audit_id_list(audit: Mapping[str, Any], field: str) -> list[str]:
    raw = audit.get(field)
    if not isinstance(raw, list) or any(not str(value).strip() for value in raw):
        raise ValueError(f"Grounding review audit {field} must be a string list")
    values = [str(value) for value in raw]
    if values != sorted(set(values)):
        raise ValueError(
            f"Grounding review audit {field} must be sorted and duplicate-free"
        )
    return values


def _validate_strict_review_audit(
    audit: Mapping[str, Any],
    *,
    requests: Mapping[str, Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, Any]],
    expected_count: int,
    requests_sha256: str,
    reviews_sha256: str,
    expected_proposals_sha256: str,
) -> None:
    expected_scalars = {
        "audit_schema_version": GROUNDING_REVIEW_AUDIT_SCHEMA,
        "review_schema_version": GROUNDING_REVIEW_SCHEMA,
        "grounding_mode": HUMAN_REVIEWED,
        "passed": True,
        "human_reviewed": True,
        "expected_count": expected_count,
        "request_count": expected_count,
        "review_count": expected_count,
        "requests_sha256": requests_sha256,
        "reviews_sha256": reviews_sha256,
        "review_sha256": reviews_sha256,
        "proposals_sha256": expected_proposals_sha256,
    }
    mismatched = {
        field: {"expected": expected, "actual": audit.get(field)}
        for field, expected in expected_scalars.items()
        if audit.get(field) != expected
    }
    if mismatched:
        raise ValueError(f"Grounding review audit contract mismatch: {mismatched}")
    for field in (
        "unknown_example_ids",
        "missing_example_ids",
        "duplicate_request_ids",
        "duplicate_review_ids",
        "invalid",
    ):
        if audit.get(field) != []:
            raise ValueError(f"Grounding review audit {field} must be empty")
    recorded_fingerprint = str(audit.get("fingerprint", ""))
    fingerprint_payload = dict(audit)
    fingerprint_payload.pop("fingerprint", None)
    if not recorded_fingerprint or recorded_fingerprint != object_fingerprint(
        fingerprint_payload
    ):
        raise ValueError("Grounding review audit fingerprint is invalid")

    bad_request_schema = sorted(
        example_id
        for example_id, request in requests.items()
        if request.get("schema_version") != GROUNDING_REQUEST_SCHEMA
    )
    if bad_request_schema:
        raise ValueError(
            f"Grounding requests have incompatible schema: {bad_request_schema[:5]}"
        )
    reviewer_ids: set[str] = set()
    eligible_ids: list[str] = []
    dispositions: Counter[str] = Counter()
    for example_id, review in reviews.items():
        status = str(review.get("status", ""))
        dispositions[status] += 1
        if status not in {"eligible", "ineligible"}:
            raise ValueError(f"{example_id}: reviewed status is invalid")
        if review.get("schema_version") != GROUNDING_REVIEW_SCHEMA:
            raise ValueError(f"{example_id}: reviewed schema is invalid")
        if review.get("human_reviewed") is not True:
            raise ValueError(f"{example_id}: human_reviewed must be true")
        reviewer_id = str(review.get("review_id", "")).strip()
        if not reviewer_id:
            raise ValueError(f"{example_id}: review_id must be non-empty")
        reviewer_ids.add(reviewer_id)
        if review.get("request_sha256") != requests_sha256:
            raise ValueError(f"{example_id}: request_sha256 differs from requests")
        if review.get("proposals_sha256") != expected_proposals_sha256:
            raise ValueError(
                f"{example_id}: proposals_sha256 differs from frozen source"
            )
        if not str(review.get("reviewed_at", "")).strip():
            raise ValueError(f"{example_id}: reviewed_at must be non-empty")
        if status == "eligible":
            eligible_ids.append(example_id)
    if len(reviewer_ids) != 1:
        raise ValueError(
            "Grounding review output mixes reviewer IDs within one review session"
        )
    expected_dispositions = dict(sorted(dispositions.items()))
    if audit.get("dispositions") != expected_dispositions:
        raise ValueError("Grounding review audit dispositions differ from reviews")
    if _strict_audit_id_list(audit, "eligible_example_ids") != sorted(eligible_ids):
        raise ValueError("Grounding review audit eligible IDs differ from reviews")
    if audit.get("eligible_example_count") != len(eligible_ids):
        raise ValueError("Grounding review audit eligible count differs from reviews")

def _expected_target_images(request: dict[str, Any], model: str) -> tuple[Path, ...]:
    frames = request.get("model_frames")
    frame = frames.get(model) if isinstance(frames, dict) else None
    if not isinstance(frame, dict):
        return ()
    values: list[Any] = []
    if model in {"roboreward", "qwen"}:
        values.append(frame.get("image_path"))
    else:
        terminal_views = frame.get("terminal_views")
        front = terminal_views.get("front") if isinstance(terminal_views, dict) else None
        if isinstance(front, dict):
            values.append(front.get("image_path"))
        image_paths = frame.get("image_paths")
        if isinstance(image_paths, list) and len(image_paths) > 5:
            values.append(image_paths[5])
    return tuple(
        Path(str(value)).resolve()
        for value in values
        if isinstance(value, (str, Path)) and str(value)
    )


def _expected_target_sha256(request: dict[str, Any], model: str) -> str:
    frames = request.get("model_frames")
    frame = frames.get(model) if isinstance(frames, dict) else None
    if not isinstance(frame, dict):
        return ""
    if model in {"roboreward", "qwen"}:
        return str(frame.get("image_sha256", ""))
    terminal_views = frame.get("terminal_views")
    front = terminal_views.get("front") if isinstance(terminal_views, dict) else None
    if not isinstance(front, dict):
        return ""
    image_paths = frame.get("image_paths")
    if (
        frame.get("primary_target_slot") not in (None, "after_cam_high")
        or frame.get("primary_target_view") not in (None, "front")
        or not isinstance(image_paths, list)
        or len(image_paths) != 8
        or Path(str(image_paths[5])).resolve()
        != Path(str(front.get("image_path", ""))).resolve()
    ):
        raise ValueError("GRM frozen after_cam_high/front slot is inconsistent")
    return str(front.get("image_sha256", ""))


def _assumed_review_contract(
    review: dict[str, Any],
    *,
    example_id: str,
    review_status: str,
) -> None:
    resolution, grounding_status = ASSUMED_STATUS_CONTRACT[review_status]
    if review.get("human_reviewed") is not False:
        raise ValueError(f"{example_id}: assumed review must be human_reviewed=false")
    if review.get("claim_status") != "exploratory":
        raise ValueError(f"{example_id}: assumed review must be exploratory")
    if review.get("grounding_resolution") != resolution:
        raise ValueError(f"{example_id}: assumed grounding_resolution mismatch")
    if review.get("grounding_status") != grounding_status:
        raise ValueError(f"{example_id}: assumed grounding_status mismatch")


def _validate_request_metadata(
    request: dict[str, Any],
    input_row: dict[str, Any],
    part: str,
    *,
    example_id: str,
    require_all: bool,
) -> None:
    expected = {
        "group_id": input_row["group_id"],
        "task_id": input_row["task_id"],
        "partition": part,
    }
    for field, expected_value in expected.items():
        if field not in request:
            if require_all:
                raise ValueError(
                    f"{example_id}: assumed request is missing {field} metadata"
                )
            continue
        if str(request[field]) != str(expected_value):
            raise ValueError(
                f"{example_id}: request metadata differs from frozen input/split"
            )


def _partition_map(split: Any) -> dict[str, str]:
    examples = split.get("examples") if isinstance(split, dict) else None
    if not isinstance(examples, dict):
        raise ValueError("Split manifest must contain an examples mapping")
    result: dict[str, str] = {}
    for name in PARTITIONS:
        values = examples.get(name, [])
        if not isinstance(values, list):
            raise ValueError(f"Split examples.{name} must be a list")
        for raw_id in values:
            example_id = str(raw_id)
            if example_id in result:
                raise ValueError(f"Split assigns {example_id} more than once")
            result[example_id] = name
    return result


def _unique_index(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError(f"{kind} row has no example_id")
        if example_id in result:
            raise ValueError(f"Duplicate {kind} example_id: {example_id}")
        result[example_id] = row
    return result


def _tracked_artifacts_by_fingerprint(
    *paths: Path | None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path is None:
            continue
        for row in read_jsonl(path):
            if not isinstance(row, dict):
                raise ValueError(f"Tracked artifact in {path} is not an object")
            fingerprint = str(row.get("fingerprint", ""))
            view = dict(row)
            view.pop("fingerprint", None)
            if not fingerprint or fingerprint != object_fingerprint(view):
                raise ValueError(f"Tracked artifact fingerprint is invalid in {path}")
            if fingerprint in result and result[fingerprint] != row:
                raise ValueError(f"Conflicting tracked artifact fingerprint {fingerprint}")
            result[fingerprint] = row
    return result


def _review_selected_track(
    review: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    decision = review.get("decision")
    provenance = (
        decision.get("track_provenance")
        if isinstance(decision, Mapping)
        else None
    )
    if not isinstance(provenance, Mapping):
        raise ValueError("Eligible tracked review has no track provenance")
    artifact_fp = str(provenance.get("artifact_fingerprint", ""))
    track_fp = str(provenance.get("track_fingerprint", ""))
    artifact = artifacts.get(artifact_fp)
    if not isinstance(artifact, Mapping):
        raise ValueError("Reviewed tracking artifact fingerprint is unavailable")
    if artifact.get("example_id") != review.get("example_id"):
        raise ValueError("Reviewed tracking artifact belongs to another example")
    tracks = artifact.get("candidate_tracks")
    matches = [
        track
        for track in tracks
        if isinstance(track, Mapping)
        and track.get("fingerprint") == track_fp
        and track.get("candidate_id") == review.get("selected_candidate_id")
        and track.get("status") == "ok"
    ] if isinstance(tracks, list) else []
    if len(matches) != 1:
        raise ValueError("Reviewed selected track is missing or ambiguous")
    return matches[0]



def _build_tracked_attention_manifests(cfg: dict[str, Any]) -> Path:
    review_source_kind = str(cfg.get("review_source_kind", ""))
    if review_source_kind not in TRACKED_REVIEW_SOURCE_KINDS:
        raise ValueError(
            "review_source_kind must be one of "
            f"{sorted(TRACKED_REVIEW_SOURCE_KINDS)}"
        )
    if _accepted_statuses(cfg) != ["eligible"]:
        raise ValueError("tracked grounding accepts only status=eligible")
    if not bool(cfg.get("require_review_audit", False)):
        raise ValueError("tracked grounding requires the strict review audit")
    if not bool(cfg.get("include_all", False)) or not bool(
        cfg.get("complete_groups_only", False)
    ):
        raise ValueError(
            "tracked grounding requires include_all and complete_groups_only"
        )

    native_temporal_patch_size = int(
        cfg.get("native_video_temporal_patch_size", 0)
    )
    if native_temporal_patch_size < 1:
        raise ValueError("native_video_temporal_patch_size must be positive")
    inputs_path = Path(cfg["inputs_path"]).resolve()
    split_path = Path(cfg["split_path"]).resolve()
    requests_path = Path(cfg["grounding_requests_path"]).resolve()
    reviews_path = Path(cfg["grounding_reviews_path"]).resolve()
    tracking_path = Path(cfg["tracking_artifact_path"]).resolve()
    tracking_manifest_path = Path(cfg["tracking_manifest_path"]).resolve()
    audit_path = Path(cfg["review_audit_path"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    manual_value = cfg.get("manual_tracking_artifact_path")
    manual_path = (
        Path(str(manual_value)).resolve()
        if manual_value not in (None, "")
        else None
    )
    for path in (
        inputs_path,
        split_path,
        requests_path,
        reviews_path,
        tracking_path,
        tracking_manifest_path,
        audit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if manual_path is not None and not manual_path.is_file():
        raise FileNotFoundError(manual_path)
    if tracking_manifest_path != tracking_path.parent / "manifest.json":
        raise ValueError("tracking_manifest_path must bind the tracking run directory")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise ValueError("Tracked review audit must be a JSON object")

    expected_requests = _configured_sha256(
        cfg,
        "expected_requests_sha256",
        required=True,
        audit=audit,
        audit_field="requests_sha256",
    )
    expected_tracking = _configured_sha256(
        cfg,
        "expected_tracking_artifact_sha256",
        required=True,
        audit=audit,
        audit_field="tracking_artifact_sha256",
    )
    expected_manifest = _configured_sha256(
        cfg,
        "expected_tracking_manifest_sha256",
        required=True,
        audit=audit,
        audit_field="tracking_manifest_sha256",
    )
    expected_manual = _configured_sha256(
        cfg,
        "expected_manual_tracking_artifact_sha256",
        required=False,
        audit=audit,
        audit_field="manual_tracking_artifact_sha256",
    )
    requests_sha = sha256_file(requests_path)
    reviews_sha = sha256_file(reviews_path)
    tracking_sha = sha256_file(tracking_path)
    tracking_manifest_sha = sha256_file(tracking_manifest_path)
    manual_sha = sha256_file(manual_path) if manual_path is not None else None
    sha_pairs = {
        "expected_requests_sha256": (expected_requests, requests_sha),
        "expected_tracking_artifact_sha256": (
            expected_tracking,
            tracking_sha,
        ),
        "expected_tracking_manifest_sha256": (
            expected_manifest,
            tracking_manifest_sha,
        ),
        "expected_manual_tracking_artifact_sha256": (
            expected_manual,
            manual_sha,
        ),
    }
    mismatched = {
        key: {"expected": expected, "actual": actual}
        for key, (expected, actual) in sha_pairs.items()
        if expected != actual
    }
    if mismatched:
        raise ValueError(f"tracked grounding SHA mismatch: {mismatched}")

    inputs = {
        str(row["example_id"]): row for row in load_model_inputs(inputs_path)
    }
    requests = _unique_index(requests_path, "tracked grounding request")
    reviews = _unique_index(reviews_path, "tracked grounding review")
    tracked_artifacts = _tracked_artifacts_by_fingerprint(
        tracking_path, manual_path
    )
    partition = _partition_map(
        json.loads(split_path.read_text(encoding="utf-8"))
    )
    configured_count = cfg.get("expected_input_count")
    expected_count = (
        len(inputs)
        if configured_count in (None, "auto")
        else int(configured_count)
    )
    if len(inputs) != expected_count:
        raise ValueError(f"Expected {expected_count} inputs, found {len(inputs)}")
    expected_ids = set(inputs)
    for name, ids in (
        ("requests", set(requests)),
        ("reviews", set(reviews)),
        ("split", set(partition)),
    ):
        if ids != expected_ids:
            raise ValueError(f"tracked {name} IDs differ from model inputs")

    fingerprint_view = dict(audit) if isinstance(audit, dict) else {}
    fingerprint = str(fingerprint_view.pop("fingerprint", ""))
    if not fingerprint or fingerprint != object_fingerprint(fingerprint_view):
        raise ValueError("tracked review audit fingerprint is invalid")
    expected_audit = {
        "audit_schema_version": (
            "my_dataset.tracked_grounding_review_audit.v2"
        ),
        "review_schema_version": "my_dataset.tracked_grounding_review.v2",
        "grounding_mode": HUMAN_REVIEWED,
        "passed": True,
        "human_reviewed": True,
        "expected_count": len(inputs),
        "request_count": len(inputs),
        "review_count": len(inputs),
        "requests_sha256": requests_sha,
        "reviews_sha256": reviews_sha,
        "review_sha256": reviews_sha,
        "tracking_artifact_sha256": tracking_sha,
        "tracking_manifest_sha256": tracking_manifest_sha,
        "manual_tracking_artifact_sha256": manual_sha,
        "target_grounding_scope": "terminal_only",
        "control_region_policy": "none",
        "tracking_continuity_verified": True,
        "labels_opened": False,
    }
    bad_audit = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in expected_audit.items()
        if audit.get(key) != expected
    }
    if bad_audit:
        raise ValueError(f"tracked review audit contract mismatch: {bad_audit}")
    expected_audit_paths = {
        "requests_path": str(requests_path),
        "reviews_path": str(reviews_path),
        "tracking_artifact_path": str(tracking_path),
        "tracking_manifest_path": str(tracking_manifest_path),
        "manual_tracking_artifact_path": (
            str(manual_path) if manual_path is not None else None
        ),
    }
    if any(
        audit.get(key) != value for key, value in expected_audit_paths.items()
    ):
        raise ValueError("tracked review audit paths differ from configured sources")
    for key in (
        "unknown_example_ids",
        "missing_example_ids",
        "duplicate_request_ids",
        "duplicate_review_ids",
        "invalid",
    ):
        if audit.get(key) != []:
            raise ValueError(f"tracked review audit {key} must be empty")

    provenance = build_tracking_review_provenance(
        requests_sha256=requests_sha,
        tracking_artifact_sha256=tracking_sha,
        tracking_manifest_sha256=tracking_manifest_sha,
        manual_tracking_artifact_sha256=manual_sha,
        reviews_sha256=reviews_sha,
        review_audit_sha256=sha256_file(audit_path),
        review_audit_fingerprint=fingerprint,
    )
    eligible = {
        example_id
        for example_id, row in reviews.items()
        if row.get("status") == "eligible"
    }
    if bool(cfg.get("require_all_inputs", False)) and eligible != expected_ids:
        raise ValueError(
            f"require_all_inputs rejected {len(expected_ids - eligible)} "
            "non-eligible inputs"
        )
    if _strict_audit_id_list(audit, "eligible_example_ids") != sorted(eligible):
        raise ValueError("audit and attention eligible ID sets differ")

    groups: dict[str, set[str]] = {}
    for example_id, row in inputs.items():
        groups.setdefault(str(row["group_id"]), set()).add(example_id)
    complete_group_ids = sorted(
        group_id
        for group_id, members in groups.items()
        if members <= eligible
    )
    complete_ids = sorted(
        example_id
        for group_id in complete_group_ids
        for example_id in groups[group_id]
    )
    if (
        _strict_audit_id_list(audit, "complete_group_ids")
        != complete_group_ids
        or _strict_audit_id_list(audit, "complete_group_example_ids")
        != complete_ids
    ):
        raise ValueError("audit and attention complete-group IDs differ")
    counts = {
        "eligible_example_count": len(eligible),
        "skipped_example_count": len(inputs) - len(eligible),
        "complete_group_count": len(complete_group_ids),
        "complete_group_example_count": len(complete_ids),
        "incomplete_or_skipped_group_count": (
            len(groups) - len(complete_group_ids)
        ),
    }
    bad_counts = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in counts.items()
        if audit.get(key) != expected
    }
    if bad_counts:
        raise ValueError(f"audit and attention counts differ: {bad_counts}")

    output: dict[tuple[str, str], list[dict[str, Any]]] = {
        (model, part): [] for model in MODELS for part in PARTITIONS
    }
    sha_cache: dict[Path, str] = {}
    dispositions = Counter(str(row.get("status")) for row in reviews.values())
    for example_id in sorted(eligible):
        input_row = inputs[example_id]
        request = requests[example_id]
        review = reviews[example_id]
        part = partition[example_id]
        if part not in PARTITIONS or request.get("partition") != part:
            raise ValueError(f"{example_id}: partition binding mismatch")
        if request.get("schema_version") != TRACKED_GROUNDING_REQUEST_SCHEMA:
            raise ValueError(f"{example_id}: tracked request schema mismatch")
        request_view = dict(request)
        request_fp = str(request_view.pop("request_fingerprint", ""))
        if not request_fp or request_fp != object_fingerprint(request_view):
            raise ValueError(f"{example_id}: tracked request fingerprint mismatch")
        if (
            str(request.get("group_id")) != str(input_row.get("group_id"))
            or str(request.get("instruction"))
            != str(input_row.get("instruction"))
        ):
            raise ValueError(f"{example_id}: request/input metadata mismatch")
        if (
            review.get("schema_version")
            != "my_dataset.tracked_grounding_review.v2"
            or review.get("human_reviewed") is not True
        ):
            raise ValueError(f"{example_id}: tracked review contract mismatch")
        selected_track = _review_selected_track(review, tracked_artifacts)
        raw_track_frames = selected_track.get("frames")
        if not isinstance(raw_track_frames, list):
            raise ValueError(f"{example_id}: selected track frames are missing")
        track_frames_by_index: dict[int, Mapping[str, Any]] = {}
        for track_frame in raw_track_frames:
            index = (
                track_frame.get("source_frame_index")
                if isinstance(track_frame, Mapping)
                else None
            )
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index in track_frames_by_index
            ):
                raise ValueError(
                    f"{example_id}: selected track frame index is invalid/duplicate"
                )
            track_frames_by_index[index] = track_frame

        models = review.get("models")
        bindings = request.get("model_frame_bindings")
        if (
            not isinstance(models, Mapping)
            or set(models) != set(MODELS)
            or not isinstance(bindings, Mapping)
            or set(bindings) != set(MODELS)
        ):
            raise ValueError(f"{example_id}: model coverage mismatch")
        for model in MODELS:
            model_review = models[model]
            binding = bindings[model]
            target = (
                model_review.get("target")
                if isinstance(model_review, Mapping)
                else None
            )
            terminal = (
                binding.get("terminal")
                if isinstance(binding, Mapping)
                else None
            )
            if not isinstance(target, Mapping) or not isinstance(
                terminal, Mapping
            ):
                raise ValueError(f"{example_id}/{model}: terminal target missing")
            target_path = Path(str(target.get("image_path", ""))).resolve()
            terminal_path = Path(str(terminal.get("image_path", ""))).resolve()
            terminal_sha = str(terminal.get("image_sha256", ""))
            if (
                target_path != terminal_path
                or target.get("image_sha256") != terminal_sha
                or target.get("source_frame_index")
                != terminal.get("source_frame_index")
                or target.get("source") != "tracked_sam3"
            ):
                raise ValueError(f"{example_id}/{model}: terminal binding mismatch")
            if target_path not in sha_cache:
                sha_cache[target_path] = sha256_file(target_path)
            if sha_cache[target_path] != terminal_sha:
                raise ValueError(f"{example_id}/{model}: terminal image SHA mismatch")
            bbox = _bbox(target.get("bbox"), f"{example_id}/{model}")
            with Image.open(target_path) as image:
                width, height = image.size
            if (
                bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] > width
                or bbox[3] > height
            ):
                raise ValueError(f"{example_id}/{model}: terminal bbox out of bounds")
            base: dict[str, Any] = {
                "schema_version": ATTENTION_INPUT_SCHEMA,
                "example_id": example_id,
                "group_id": str(input_row["group_id"]),
                "group_media_sha256": str(input_row["group_media_sha256"]),
                "task_id": str(input_row["task_id"]),
                "task_family": str(input_row["task_family"]),
                "partition": part,
                "model_family": model,
                "task": str(input_row["instruction"]),
                "last_image_path": str(target_path),
                "last_image_sha256": terminal_sha,
                "last_bbox": bbox,
                "target_source_frame_index": int(
                    terminal["source_frame_index"]
                ),
                "grounding_review_id": str(review["review_id"]),
                "grounding_status": "audited_eligible",
                "grounding_mode": HUMAN_REVIEWED,
                "grounding_resolution": "human_audited",
                "claim_status": "reviewed_exploratory",
                "human_reviewed": True,
                "target_grounding_scope": "terminal_only",
                "control_region_policy": "none",
                "tracking_review_provenance": dict(provenance),
                "grounding_selection": {
                    "decision_source": review["decision"]["source"],
                    "selected_candidate_id": review[
                        "selected_candidate_id"
                    ],
                    "track_fingerprint": review["decision"][
                        "track_provenance"
                    ]["track_fingerprint"],
                    "locked_obj_id": review["decision"]["track_provenance"][
                        "locked_obj_id"
                    ],
                },
            }
            if "wrong_region" in model_review:
                raise ValueError(
                    f"{example_id}/{model}: tracking v2 forbids wrong-region"
                )
            if model in {"roboreward", "qwen"}:
                sampled = binding.get("sampled_frame_indices")
                content_order = str(binding.get("content_order", ""))
                grid = binding.get("video_grid_thw")
                if not isinstance(sampled, list) or not sampled:
                    raise ValueError(f"{example_id}/{model}: sampled frames missing")
                if content_order not in {"text_then_video", "video_then_text"}:
                    raise ValueError(f"{example_id}/{model}: content_order invalid")
                if (
                    not isinstance(grid, list)
                    or len(grid) != 1
                    or any(
                        not isinstance(item, (list, tuple))
                        or len(item) != 3
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 1
                            for value in item
                        )
                        for item in grid
                    )
                ):
                    raise ValueError(f"{example_id}/{model}: video_grid_thw invalid")
                temporal_grid_size = int(grid[0][0])
                expected_temporal_grid_size = (
                    len(sampled) + native_temporal_patch_size - 1
                ) // native_temporal_patch_size
                if temporal_grid_size != expected_temporal_grid_size:
                    raise ValueError(
                        f"{example_id}/{model}: sampled frames, temporal patch size, "
                        "and video_grid_thw disagree"
                    )
                tracked_processor_frames: list[dict[str, Any]] = []
                for raw_index in sampled:
                    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                        raise ValueError(
                            f"{example_id}/{model}: sampled frame index is invalid"
                        )
                    frame_index = int(raw_index)
                    track_frame = track_frames_by_index.get(frame_index)
                    if track_frame is None:
                        raise ValueError(
                            f"{example_id}/{model}: reviewed track lacks sampled frame "
                            f"{frame_index}"
                        )
                    visible = track_frame.get("visible") is True
                    raw_bbox = track_frame.get("bbox_xyxy")
                    tracked_bbox = (
                        _bbox(
                            raw_bbox,
                            f"{example_id}/{model}/tracked_frame/{frame_index}",
                        )
                        if visible
                        else None
                    )
                    if not visible and raw_bbox is not None:
                        raise ValueError(
                            f"{example_id}/{model}: invisible tracked frame has bbox"
                        )
                    tracked_processor_frames.append(
                        {
                            "source_frame_index": frame_index,
                            "image_path": str(track_frame["image_path"]),
                            "image_sha256": str(track_frame["image_sha256"]),
                            "bbox_xyxy": tracked_bbox,
                            "mask_path": str(track_frame["mask_path"]),
                            "mask_sha256": str(track_frame["mask_sha256"]),
                            "score": float(track_frame["score"]),
                            "visible": visible,
                            "obj_id": int(track_frame["obj_id"]),
                        }
                    )

                base.update(
                    {
                        "input_layout": "native_front_video",
                        "video_path": str(
                            Path(input_row["video_paths"]["front"]).resolve()
                        ),
                        "video_sha256": str(input_row["view_sha256"]["front"]),
                        "processor_frame_indices": list(sampled),
                        "tracked_processor_frames": tracked_processor_frames,
                        "target_token_grounding_scope": (
                            "terminal_temporal_patch_tracked_bbox_union"
                        ),
                        "content_order": content_order,
                        "processor_video_grid_thw": [
                            list(item) for item in grid
                        ],
                        "processor_temporal_patch_size": (
                            native_temporal_patch_size
                        ),
                    }
                )
                processor_contract = validate_processor_content_order_contract(
                    binding,
                    identity=f"{example_id}/{model}",
                )
                if model == "qwen" and processor_contract is not None:
                    raise ValueError(
                        f"{example_id}/qwen: RoboReward order contract is forbidden"
                    )
                if model == "roboreward" and processor_contract is not None:
                    base["processor_content_order_contract"] = processor_contract
            else:
                image_paths = binding.get("image_paths")
                if (
                    not isinstance(image_paths, list)
                    or len(image_paths) != 8
                    or binding.get("primary_target_slot") != "after_cam_high"
                ):
                    raise ValueError(
                        f"{example_id}/grm: canonical eight-image binding invalid"
                    )
                if Path(str(image_paths[5])).resolve() != target_path:
                    raise ValueError(
                        f"{example_id}/grm: terminal is not after_cam_high"
                    )
                base.update(
                    {
                        "input_layout": "grm_native_three_view_endpoints_v1",
                        "video_sha256": str(input_row["group_media_sha256"]),
                        "image_paths": list(image_paths),
                        "target_token_grounding_scope": (
                            "after_cam_high_terminal_frame"
                        ),
                        "first": {
                            "provenance": {"image_path": image_paths[2]}
                        },
                        "last": {
                            "provenance": {"image_path": str(target_path)},
                            "bbox": bbox,
                        },
                    }
                )
            if FORBIDDEN_MODEL_FIELDS & base.keys():
                raise AssertionError("Label field entered attention manifest")
            output[(model, part)].append(base)

    artifacts: dict[str, dict[str, Any]] = {}
    complete_set = set(complete_ids)
    for model in MODELS:
        for part in PARTITIONS:
            rows = sorted(
                output[(model, part)], key=lambda row: row["example_id"]
            )
            path = output_dir / model / f"{part}.jsonl"
            write_jsonl(path, rows)
            artifacts[f"{model}/{part}"] = {
                "path": str(path),
                "count": len(rows),
                "sha256": sha256_file(path),
                "fingerprint": object_fingerprint(rows),
            }
        all_rows = sorted(
            (
                row
                for part in PARTITIONS
                for row in output[(model, part)]
            ),
            key=lambda row: row["example_id"],
        )
        for filename, rows in (
            ("all", all_rows),
            (
                "complete_groups",
                [row for row in all_rows if row["example_id"] in complete_set],
            ),
        ):
            path = output_dir / model / f"{filename}.jsonl"
            write_jsonl(path, rows)
            artifacts[f"{model}/{filename}"] = {
                "path": str(path),
                "count": len(rows),
                "sha256": sha256_file(path),
                "fingerprint": object_fingerprint(rows),
            }

    included = len(eligible)
    manifest: dict[str, Any] = {
        "schema_version": ATTENTION_INPUT_SCHEMA,
        "review_source_kind": review_source_kind,
        "inputs_sha256": sha256_file(inputs_path),
        "split_sha256": sha256_file(split_path),
        "grounding_requests_sha256": requests_sha,
        "expected_requests_sha256": expected_requests,
        "tracking_artifact_sha256": tracking_sha,
        "expected_tracking_artifact_sha256": expected_tracking,
        "tracking_manifest_sha256": tracking_manifest_sha,
        "expected_tracking_manifest_sha256": expected_manifest,
        "manual_tracking_artifact_sha256": manual_sha,
        "expected_manual_tracking_artifact_sha256": expected_manual,
        "grounding_reviews_sha256": reviews_sha,
        "review_audit_path": str(audit_path),
        "review_audit_sha256": sha256_file(audit_path),
        "review_audit_schema_version": audit["audit_schema_version"],
        "review_audit_fingerprint": fingerprint,
        "tracking_review_provenance": provenance,
        "target_grounding_scope": "terminal_only",
        "control_region_policy": "none",
        "input_count": len(inputs),
        "expected_input_count": expected_count,
        "accepted_review_statuses": ["eligible"],
        "include_all": True,
        "require_all_inputs": bool(cfg.get("require_all_inputs", False)),
        "complete_groups_only": True,
        "require_review_audit": True,
        "artifacts": artifacts,
        "dispositions": dict(sorted(dispositions.items())),
        "grounding_resolution_counts": {
            model: {"human_audited": included} for model in MODELS
        },
        "omitted_input_model_count": (len(inputs) - included) * len(MODELS),
        "included_example_count": included,
        "complete_group_example_count": len(complete_ids),
        "dropped_incomplete_group_count": (
            len(groups) - len(complete_group_ids)
        ),
        "dropped_incomplete_example_count": len(inputs) - len(complete_ids),
        "dropped_incomplete_groups": {
            group_id: sorted(members - eligible)
            for group_id, members in sorted(groups.items())
            if not members <= eligible
        },
        "config": {
            key: cfg.get(key)
            for key in (
                "review_source_kind",
                "accepted_review_statuses",
                "include_all",
                "require_all_inputs",
                "complete_groups_only",
                "require_review_audit",
                "expected_input_count",
                "expected_requests_sha256",
                "expected_tracking_artifact_sha256",
                "expected_tracking_manifest_sha256",
                "expected_manual_tracking_artifact_sha256",
            )
        },
        "labels_opened": False,
    }
    manifest["fingerprint"] = object_fingerprint(manifest)
    path = output_dir / "manifest.json"
    write_json(path, manifest)
    return path


def build_attention_manifests(config: dict[str, Any]) -> Path:
    cfg = section(config, "my_dataset_attention")
    if cfg.get("review_source_kind") in TRACKED_REVIEW_SOURCE_KINDS:
        return _build_tracked_attention_manifests(cfg)
    inputs_path = Path(cfg["inputs_path"]).resolve()
    requests_path = Path(cfg["grounding_requests_path"]).resolve()
    reviews_path = Path(cfg["grounding_reviews_path"]).resolve()
    split_path = Path(cfg["split_path"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    accepted_statuses = _accepted_statuses(cfg)
    include_all = bool(cfg.get("include_all", False))
    require_all_inputs = bool(cfg.get("require_all_inputs", False))
    complete_groups_only = bool(cfg.get("complete_groups_only", False))
    require_review_audit = bool(cfg.get("require_review_audit", False))
    configured_audit_path = cfg.get("review_audit_path")
    review_audit_path = (
        Path(str(configured_audit_path)).resolve()
        if configured_audit_path not in (None, "")
        else None
    )
    if require_review_audit and review_audit_path is None:
        raise ValueError(
            "require_review_audit=true requires review_audit_path"
        )
    expected_requests_sha256 = _configured_sha256(
        cfg,
        "expected_requests_sha256",
        required=require_review_audit,
    )
    expected_proposals_sha256 = _configured_sha256(
        cfg,
        "expected_proposals_sha256",
        required=require_review_audit,
    )
    if require_review_audit:
        if accepted_statuses != ["eligible"]:
            raise ValueError("Audited review accepts only status=eligible")
        if not include_all or not complete_groups_only:
            raise ValueError(
                "Audited review requires include_all=true and complete_groups_only=true"
            )
    configured_expected_count = cfg.get("expected_input_count")
    expected_input_count = (
        int(configured_expected_count)
        if configured_expected_count is not None
        else None
    )

    inputs = {
        str(row["example_id"]): row for row in load_model_inputs(inputs_path)
    }
    requests = _unique_index(requests_path, "grounding request")
    reviews = _unique_index(reviews_path, "grounding review")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    partition = _partition_map(split)
    requests_sha256 = sha256_file(requests_path)
    reviews_sha256 = sha256_file(reviews_path)
    if (
        expected_requests_sha256 is not None
        and requests_sha256 != expected_requests_sha256
    ):
        raise ValueError(
            "grounding_requests_path SHA-256 differs from expected_requests_sha256"
        )

    review_audit: dict[str, Any] | None = None
    review_audit_sha256: str | None = None
    if review_audit_path is not None:
        if not review_audit_path.is_file():
            raise FileNotFoundError(review_audit_path)
        loaded_audit = json.loads(review_audit_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_audit, dict):
            raise ValueError("Grounding review audit must be a JSON object")
        review_audit = loaded_audit
        review_audit_sha256 = sha256_file(review_audit_path)
    if expected_input_count is not None and len(inputs) != expected_input_count:
        raise ValueError(
            f"Expected {expected_input_count} model inputs, found {len(inputs)}"
        )

    expected_ids = set(inputs)
    if require_review_audit or require_all_inputs:
        for name, ids in (
            ("grounding requests", set(requests)),
            ("grounding reviews", set(reviews)),
            ("split", set(partition)),
        ):
            if ids != expected_ids:
                missing = sorted(expected_ids - ids)[:10]
                extra = sorted(ids - expected_ids)[:10]
                raise ValueError(
                    f"{name} IDs differ from inputs: missing={missing}, extra={extra}"
                )
    review_provenance: dict[str, str] | None = None
    if require_review_audit:
        if review_audit is None:
            raise AssertionError("Audited review unexpectedly has no audit object")
        _validate_strict_review_audit(
            review_audit,
            requests=requests,
            reviews=reviews,
            expected_count=len(inputs),
            requests_sha256=requests_sha256,
            reviews_sha256=reviews_sha256,
            expected_proposals_sha256=str(expected_proposals_sha256),
        )
        if review_audit_sha256 is None:
            raise AssertionError("Audited review unexpectedly has no audit SHA-256")
        review_provenance = build_review_provenance(
            requests_sha256=requests_sha256,
            proposals_sha256=str(expected_proposals_sha256),
            reviews_sha256=reviews_sha256,
            review_audit_sha256=review_audit_sha256,
            review_audit_fingerprint=str(review_audit["fingerprint"]),
        )
        for example_id in sorted(expected_ids):
            _validate_request_metadata(
                requests[example_id],
                inputs[example_id],
                partition[example_id],
                example_id=example_id,
                require_all=True,
            )
    elif review_audit is not None:
        if review_audit.get("passed") is not True:
            raise ValueError("Grounding review audit did not pass")
        if review_audit.get("review_sha256") != reviews_sha256:
            raise ValueError(
                "Grounding review audit SHA does not match reviews.jsonl"
            )

    # Human review dispositions are per example, whereas source-group ranking
    # metrics require an intact counterfactual group. Decide membership before
    # writing model rows so all three model populations remain identical.
    groups: dict[str, set[str]] = {}
    for example_id, input_row in inputs.items():
        groups.setdefault(str(input_row["group_id"]), set()).add(example_id)
    initially_accepted = {
        example_id
        for example_id in inputs
        if example_id in requests
        and example_id in reviews
        and str(reviews[example_id].get("status", "unreviewed"))
        in accepted_statuses
        and partition.get(example_id) in PARTITIONS
    }
    dropped_incomplete_groups = {
        group_id: sorted(example_ids - initially_accepted)
        for group_id, example_ids in groups.items()
        if complete_groups_only and not example_ids <= initially_accepted
    }
    complete_group_ids = sorted(
        group_id
        for group_id, example_ids in groups.items()
        if example_ids <= initially_accepted
    )
    complete_group_example_ids = sorted(
        example_id
        for group_id in complete_group_ids
        for example_id in groups[group_id]
    )
    if require_review_audit:
        if review_audit is None:
            raise AssertionError("Audited review unexpectedly has no audit object")
        if _strict_audit_id_list(
            review_audit, "eligible_example_ids"
        ) != sorted(initially_accepted):
            raise ValueError("Audit and attention eligible ID sets differ")
        if _strict_audit_id_list(
            review_audit, "complete_group_ids"
        ) != complete_group_ids:
            raise ValueError("Audit and attention complete group ID sets differ")
        if _strict_audit_id_list(
            review_audit, "complete_group_example_ids"
        ) != complete_group_example_ids:
            raise ValueError("Audit and attention complete-group example IDs differ")
        expected_audit_counts = {
            "eligible_example_count": len(initially_accepted),
            "ineligible_example_count": len(inputs) - len(initially_accepted),
            "complete_group_count": len(complete_group_ids),
            "complete_group_example_count": len(complete_group_example_ids),
            "incomplete_or_ineligible_group_count": (
                len(groups) - len(complete_group_ids)
            ),
        }
        mismatched_audit_counts = {
            field: {"expected": expected, "actual": review_audit.get(field)}
            for field, expected in expected_audit_counts.items()
            if review_audit.get(field) != expected
        }
        if mismatched_audit_counts:
            raise ValueError(
                "Audit and attention eligible/complete counts differ: "
                f"{mismatched_audit_counts}"
            )
    output: dict[tuple[str, str], list[dict[str, Any]]] = {
        (model, part): [] for model in MODELS for part in PARTITIONS
    }
    disposition: Counter[str] = Counter()
    image_sha_cache: dict[Path, str] = {}
    image_size_cache: dict[Path, tuple[int, int]] = {}

    for example_id, input_row in sorted(inputs.items()):
        request = requests.get(example_id)
        review = reviews.get(example_id)
        if request is None or review is None:
            disposition["missing_request_or_review"] += 1
            continue
        review_status = str(review.get("status", "unreviewed"))
        if review_status not in accepted_statuses:
            disposition[review_status] += 1
            continue
        is_assumed_review = review_status in ASSUMED_STATUS_CONTRACT
        if is_assumed_review:
            _assumed_review_contract(
                review,
                example_id=example_id,
                review_status=review_status,
            )
        part = partition.get(example_id)
        if part not in PARTITIONS:
            disposition["missing_partition"] += 1
            continue
        _validate_request_metadata(
            request,
            input_row,
            part,
            example_id=example_id,
            require_all=is_assumed_review,
        )

        model_reviews = review.get("models")
        if not isinstance(model_reviews, dict):
            raise ValueError(f"{example_id}: accepted review has no models mapping")
        if is_assumed_review:
            fallback_flags = [
                bool(
                    isinstance(value, dict)
                    and isinstance(value.get("target"), dict)
                    and value["target"].get("fallback_used") is True
                )
                for value in model_reviews.values()
            ]
            if review_status == "assumed_valid" and any(fallback_flags):
                raise ValueError(f"{example_id}: strict assumed review contains proxy target")
            if review_status == "assumed_proxy" and not any(fallback_flags):
                raise ValueError(f"{example_id}: proxy assumed review has no proxy target")
        model_frames = request.get("model_frames")
        if not isinstance(model_frames, dict):
            raise ValueError(f"{example_id}: grounding request has no model_frames")

        for model in MODELS:
            model_review = model_reviews.get(model)
            if not isinstance(model_review, dict):
                raise ValueError(f"{example_id}/{model}: accepted model review is missing")
            if is_assumed_review:
                if model_review.get("valid") is not True:
                    raise ValueError(
                        f"{example_id}/{model}: assumed model review is not valid"
                    )
                if model_review.get("invalid_reasons") != []:
                    raise ValueError(
                        f"{example_id}/{model}: assumed model review has invalid reasons"
                    )
            target = model_review.get("target")
            if not isinstance(target, dict):
                raise ValueError(f"{example_id}/{model}: accepted target is missing")
            expected_target_images = _expected_target_images(request, model)
            if not expected_target_images:
                raise ValueError(
                    f"{example_id}/{model}: frozen terminal target image is missing"
                )
            target_image = Path(str(target.get("image_path", ""))).resolve()
            if target_image not in expected_target_images:
                raise ValueError(
                    f"{example_id}/{model}: reviewed target image does not match the "
                    "frozen terminal target slot"
                )
            recorded_sha256 = str(target.get("image_sha256", ""))
            last_bbox = _bbox(target.get("bbox"), f"{example_id}/{model}")
            if is_assumed_review:
                expected_sha256 = _expected_target_sha256(request, model)
                if not expected_sha256 or recorded_sha256 != expected_sha256:
                    raise ValueError(
                        f"{example_id}/{model}: target image SHA differs from frozen request"
                    )
                if target_image not in image_sha_cache:
                    image_sha_cache[target_image] = sha256_file(target_image)
                if image_sha_cache[target_image] != expected_sha256:
                    raise ValueError(
                        f"{example_id}/{model}: target image file SHA mismatch"
                    )
                if target_image not in image_size_cache:
                    with Image.open(target_image) as image:
                        image_size_cache[target_image] = tuple(image.size)
                width, height = image_size_cache[target_image]
                if (
                    last_bbox[0] < 0
                    or last_bbox[1] < 0
                    or last_bbox[2] > width
                    or last_bbox[3] > height
                ):
                    raise ValueError(
                        f"{example_id}/{model}: target bbox exceeds image bounds"
                    )
                last_image_sha256 = expected_sha256
            else:
                if require_review_audit:
                    expected_sha256 = _expected_target_sha256(request, model)
                    if (
                        not expected_sha256
                        or recorded_sha256 != expected_sha256
                    ):
                        raise ValueError(
                            f"{example_id}/{model}: reviewed target recorded SHA "
                            "differs from the frozen request"
                        )
                    if target_image not in image_sha_cache:
                        image_sha_cache[target_image] = sha256_file(target_image)
                    actual_sha256 = image_sha_cache[target_image]
                    if actual_sha256 != expected_sha256:
                        raise ValueError(
                            f"{example_id}/{model}: reviewed target image file SHA mismatch"
                        )
                    if target_image not in image_size_cache:
                        with Image.open(target_image) as image:
                            image_size_cache[target_image] = tuple(image.size)
                    width, height = image_size_cache[target_image]
                    if (
                        last_bbox[0] < 0
                        or last_bbox[1] < 0
                        or last_bbox[2] > width
                        or last_bbox[3] > height
                    ):
                        raise ValueError(
                            f"{example_id}/{model}: target bbox exceeds image bounds"
                        )
                    last_image_sha256 = actual_sha256
                else:
                    # Compatibility only for legacy human manifests. Reviewed
                    # production configs must set require_review_audit=true.
                    last_image_sha256 = recorded_sha256
            frame = model_frames.get(model)
            if not isinstance(frame, dict):
                raise ValueError(f"{example_id}/{model}: model frame is missing")
            if is_assumed_review and model in {"roboreward", "qwen"}:
                sampled = frame.get("sampled_frame_indices")
                source_index = frame.get("source_frame_index")
                expected_video = Path(
                    str(input_row["video_paths"]["front"])
                ).resolve()
                if (
                    frame.get("input_layout") != "native_front_video"
                    or Path(str(frame.get("video_path", ""))).resolve()
                    != expected_video
                    or not isinstance(sampled, list)
                    or not sampled
                    or source_index != sampled[-1]
                ):
                    raise ValueError(
                        f"{example_id}/{model}: native video frame provenance is inconsistent"
                    )
            elif is_assumed_review:
                image_paths = frame.get("image_paths")
                if (
                    frame.get("input_layout")
                    != "grm_native_three_view_endpoints_v1"
                    or not isinstance(image_paths, list)
                    or len(image_paths) != 8
                    or Path(str(image_paths[5])).resolve() != target_image
                ):
                    raise ValueError(
                        f"{example_id}/grm: official eight-image slot contract is inconsistent"
                    )
            fallback_used = target.get("fallback_used") is True
            grounding_resolution = "proxy" if fallback_used else "strict"
            grounding_status = (
                "auto_proxy_unreviewed"
                if fallback_used
                else "auto_assumed_unreviewed"
            )
            base = {
                "schema_version": ATTENTION_INPUT_SCHEMA,
                "example_id": example_id,
                "group_id": str(input_row["group_id"]),
                "group_media_sha256": str(input_row["group_media_sha256"]),
                "task_id": str(input_row["task_id"]),
                "task_family": str(input_row["task_family"]),
                "partition": part,
                "model_family": model,
                "task": str(input_row["instruction"]),
                "last_image_path": str(target_image),
                "last_image_sha256": last_image_sha256,
                "last_bbox": last_bbox,
                "grounding_review_id": str(review.get("review_id") or example_id),
                "grounding_status": (
                    "audited_eligible"
                    if review_status == "eligible"
                    else grounding_status
                ),
                "human_reviewed": review_status == "eligible",
            }
            if is_assumed_review:
                score = target.get("score")
                query = target.get("query")
                policy = target.get("selection_method")
                if not _finite_number(score):
                    raise ValueError(
                        f"{example_id}/{model}: assumed target score must be finite"
                    )
                if not isinstance(query, str) or not query:
                    raise ValueError(
                        f"{example_id}/{model}: assumed target query is required"
                    )
                if not isinstance(policy, str) or not policy:
                    raise ValueError(
                        f"{example_id}/{model}: assumed target selection_method is required"
                    )
                base["claim_status"] = "exploratory"
                base["grounding_resolution"] = grounding_resolution
                base["grounding_selection"] = {
                    "proposal_score": float(score),
                    "proposal_query": query,
                    "selection_policy": policy,
                    "fallback_used": fallback_used,
                    "proposal_source_path": target.get("proposal_source_path"),
                    "proposal_source_sha256": target.get(
                        "proposal_source_sha256"
                    ),
                }
                base["grounding_mode"] = AUTO_UNREVIEWED
            elif require_review_audit:
                if review.get("human_reviewed") is not True:
                    raise ValueError(
                        f"{example_id}: eligible review must set human_reviewed=true"
                    )
                if review.get("claim_status") not in (
                    None,
                    "reviewed_exploratory",
                ):
                    raise ValueError(
                        f"{example_id}: eligible review has an invalid claim_status"
                    )
                base.update(
                    {
                        "grounding_mode": HUMAN_REVIEWED,
                        "grounding_resolution": "human_audited",
                        "claim_status": "reviewed_exploratory",
                        "review_provenance": dict(review_provenance or {}),
                    }
                )

            wrong = model_review.get("wrong_region")
            if require_review_audit and (
                not isinstance(wrong, dict)
                or not isinstance(wrong.get("bbox"), list)
            ):
                raise ValueError(
                    f"{example_id}/{model}: audited wrong-region control is required"
                )
            if isinstance(wrong, dict) and isinstance(wrong.get("bbox"), list):
                wrong_image = Path(str(wrong.get("image_path", ""))).resolve()
                if wrong_image != target_image:
                    raise ValueError(
                        f"{example_id}/{model}: wrong-region control must use the "
                        "same frozen target image"
                    )
                wrong_bbox = _bbox(
                    wrong["bbox"], f"{example_id}/{model}/wrong_region"
                )
                if require_review_audit:
                    if str(wrong.get("image_sha256", "")) != last_image_sha256:
                        raise ValueError(
                            f"{example_id}/{model}: wrong-region recorded SHA mismatch"
                        )
                    width, height = image_size_cache[target_image]
                    if (
                        wrong_bbox[0] < 0
                        or wrong_bbox[1] < 0
                        or wrong_bbox[2] > width
                        or wrong_bbox[3] > height
                    ):
                        raise ValueError(
                            f"{example_id}/{model}: wrong-region bbox exceeds image bounds"
                        )
                    if _bbox_intersection_area(last_bbox, wrong_bbox) > 0:
                        raise ValueError(
                            f"{example_id}/{model}: target and wrong-region bboxes overlap"
                        )
                base["wrong_region_bbox"] = wrong_bbox
            if model in {"roboreward", "qwen"}:
                base.update(
                    {
                        "input_layout": "native_front_video",
                        "video_path": str(
                            Path(input_row["video_paths"]["front"]).resolve()
                        ),
                        "video_sha256": str(input_row["view_sha256"]["front"]),
                        "processor_frame_indices": frame["sampled_frame_indices"],
                        "target_source_frame_index": frame["source_frame_index"],
                    }
                )
            else:
                base.update(
                    {
                        "input_layout": "grm_native_three_view_endpoints_v1",
                        "video_sha256": str(input_row["group_media_sha256"]),
                        "image_paths": list(frame["image_paths"]),
                        "first": {
                            "provenance": {"image_path": frame["image_paths"][2]},
                            "bbox": base["last_bbox"],
                        },
                        "last": {
                            "provenance": {"image_path": base["last_image_path"]},
                            "bbox": base["last_bbox"],
                        },
                    }
                )
            if FORBIDDEN_MODEL_FIELDS & base.keys():
                raise AssertionError("Label field entered attention manifest")
            output[(model, part)].append(base)
            disposition[f"{review_status}_{model}"] += 1

    included = {
        (str(row["example_id"]), model)
        for (model, _part), rows in output.items()
        for row in rows
    }
    expected_coverage = {
        (example_id, model) for example_id in inputs for model in MODELS
    }
    omitted = sorted(expected_coverage - included)
    if require_all_inputs and omitted:
        raise ValueError(
            f"require_all_inputs rejected {len(omitted)} omitted input/model pairs: "
            f"{omitted[:10]}"
        )

    artifacts: dict[str, dict[str, Any]] = {}
    for (model, part), rows in output.items():
        rows.sort(key=lambda row: row["example_id"])
        path = output_dir / model / f"{part}.jsonl"
        write_jsonl(path, rows)
        artifacts[f"{model}/{part}"] = {
            "path": str(path),
            "count": len(rows),
            "sha256": sha256_file(path),
            "fingerprint": object_fingerprint(rows),
        }
    if include_all:
        for model in MODELS:
            rows = sorted(
                (
                    row
                    for part in PARTITIONS
                    for row in output[(model, part)]
                ),
                key=lambda row: row["example_id"],
            )
            path = output_dir / model / "all.jsonl"
            write_jsonl(path, rows)
            artifacts[f"{model}/all"] = {
                "path": str(path),
                "count": len(rows),
                "sha256": sha256_file(path),
                "fingerprint": object_fingerprint(rows),
            }
            if complete_groups_only:
                complete_rows = [
                    row
                    for row in rows
                    if str(row["group_id"]) not in dropped_incomplete_groups
                ]
                complete_path = output_dir / model / "complete_groups.jsonl"
                write_jsonl(complete_path, complete_rows)
                artifacts[f"{model}/complete_groups"] = {
                    "path": str(complete_path),
                    "count": len(complete_rows),
                    "sha256": sha256_file(complete_path),
                    "fingerprint": object_fingerprint(complete_rows),
                }

    recorded_config = {
        "accepted_review_statuses": accepted_statuses,
        "include_all": include_all,
        "require_all_inputs": require_all_inputs,
    }
    if "complete_groups_only" in cfg:
        recorded_config["complete_groups_only"] = complete_groups_only
    if "require_review_audit" in cfg:
        recorded_config["require_review_audit"] = require_review_audit
    if review_audit_path is not None:
        recorded_config["review_audit_path"] = str(review_audit_path)
    if expected_requests_sha256 is not None:
        recorded_config["expected_requests_sha256"] = expected_requests_sha256
    if expected_proposals_sha256 is not None:
        recorded_config["expected_proposals_sha256"] = expected_proposals_sha256
    if expected_input_count is not None:
        recorded_config["expected_input_count"] = expected_input_count
    manifest = {
        "schema_version": ATTENTION_INPUT_SCHEMA,
        "inputs_sha256": sha256_file(inputs_path),
        "grounding_requests_sha256": requests_sha256,
        "grounding_reviews_sha256": reviews_sha256,
        "expected_requests_sha256": expected_requests_sha256,
        "expected_proposals_sha256": expected_proposals_sha256,
        "grounding_proposals_sha256": (
            review_audit.get("proposals_sha256") if review_audit else None
        ),
        "split_sha256": sha256_file(split_path),
        "input_count": len(inputs),
        "expected_input_count": expected_input_count,
        "accepted_review_statuses": accepted_statuses,
        "include_all": include_all,
        "require_all_inputs": require_all_inputs,
        "complete_groups_only": complete_groups_only,
        "require_review_audit": require_review_audit,
        "review_audit_path": (
            str(review_audit_path) if review_audit_path is not None else None
        ),
        "review_audit_sha256": review_audit_sha256,
        "review_audit_schema_version": (
            review_audit.get("audit_schema_version") if review_audit else None
        ),
        "review_audit_fingerprint": (
            review_audit.get("fingerprint") if review_audit else None
        ),
        "config": recorded_config,
        "artifacts": artifacts,
        "dispositions": dict(sorted(disposition.items())),
        "grounding_resolution_counts": {
            model: dict(
                sorted(
                    Counter(
                        row.get("grounding_resolution", "human_audited")
                        for part in PARTITIONS
                        for row in output[(model, part)]
                    ).items()
                )
            )
            for model in MODELS
        },
        "omitted_input_model_count": len(omitted),
        "included_example_count": len(included) // len(MODELS),
        "complete_group_example_count": (
            sum(
                len(example_ids)
                for group_id, example_ids in groups.items()
                if group_id not in dropped_incomplete_groups
            )
            if complete_groups_only
            else None
        ),
        "dropped_incomplete_group_count": len(dropped_incomplete_groups),
        "dropped_incomplete_example_count": sum(
            len(groups[group_id]) for group_id in dropped_incomplete_groups
        ),
        "dropped_incomplete_groups": dict(
            sorted(dropped_incomplete_groups.items())
        ),
        "labels_opened": False,
    }
    if review_provenance is not None:
        manifest["review_provenance"] = review_provenance
    manifest["fingerprint"] = object_fingerprint(manifest)
    path = output_dir / "manifest.json"
    write_json(path, manifest)
    return path
