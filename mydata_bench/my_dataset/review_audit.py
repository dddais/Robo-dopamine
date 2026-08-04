"""Strict audit gate for human-reviewed SAM3 grounding decisions."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..io import object_fingerprint, read_jsonl, sha256_file, write_json
from .grounding_manifest import GROUNDING_REQUEST_SCHEMA, GROUNDING_REVIEW_SCHEMA
from .tracked_grounding import (
    TRACKED_GROUNDING_ARTIFACT_SCHEMA,
    TRACKED_GROUNDING_MANIFEST_SCHEMA,
    TRACKED_GROUNDING_MANUAL_ANCHOR_SCHEMA,
    TRACKED_GROUNDING_PROPOSAL_SCHEMA,
    TRACKED_GROUNDING_REQUEST_SCHEMA,
    TRACKED_GROUNDING_TRACK_SCHEMA,
    derive_manual_anchor_id,
    validate_processor_content_order_contract,
)


AUDIT_SCHEMA_VERSION = "my_dataset.grounding_review_audit.v2"
SESSION_SCHEMA_VERSION = "my_dataset.grounding_review_session.v1"
MODELS = ("roboreward", "qwen", "grm")
TRACKING_AUDIT_SCHEMA_VERSION = "my_dataset.tracked_grounding_review_audit.v2"
TRACKING_REVIEW_SCHEMA_VERSION = "my_dataset.tracked_grounding_review.v2"
TRACKING_SESSION_SCHEMA_VERSION = "my_dataset.tracked_grounding_review_session.v2"
TRACKING_FINAL_STATUSES = frozenset({"eligible", "skipped"})
TRACKING_SKIP_CODES = frozenset({"reviewer_skip"})


def _index_unique(
    path: Path, *, kind: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for number, row in enumerate(read_jsonl(path), 1):
        example_id = str(row.get("example_id", "")).strip()
        if not example_id:
            raise ValueError(f"{path}:{number}: {kind} row has no example_id")
        if example_id in result:
            duplicates.append(example_id)
        result[example_id] = row
    return result, sorted(set(duplicates))


def _request_image(request: Mapping[str, Any], model: str) -> tuple[Path, str]:
    frames = request.get("model_frames")
    frame = frames.get(model) if isinstance(frames, Mapping) else None
    if not isinstance(frame, Mapping):
        raise ValueError("missing_model_frame")
    if model in {"roboreward", "qwen"}:
        path_value = frame.get("image_path")
        digest = str(frame.get("image_sha256", ""))
    else:
        terminal = frame.get("terminal_views")
        front = terminal.get("front") if isinstance(terminal, Mapping) else None
        if not isinstance(front, Mapping):
            raise ValueError("missing_grm_front_terminal")
        path_value = front.get("image_path")
        digest = str(front.get("image_sha256", ""))
    path = Path(str(path_value or "")).resolve()
    if not path.is_file() or not digest:
        raise ValueError("missing_frozen_image_or_sha")
    return path, digest


def _bbox(value: Any) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError("bbox_not_four_finite_numbers")
    result = [float(item) for item in value]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("bbox_non_positive_area")
    return result


def _clip_bbox(value: Any, size: tuple[int, int]) -> tuple[list[float], list[float], bool]:
    raw = _bbox(value)
    width, height = size
    clipped = [
        min(max(raw[0], 0.0), float(width)),
        min(max(raw[1], 0.0), float(height)),
        min(max(raw[2], 0.0), float(width)),
        min(max(raw[3], 0.0), float(height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("bbox_outside_image")
    changed = any(abs(a - b) > 1e-9 for a, b in zip(raw, clipped, strict=True))
    return raw, clipped, changed


def _same_bbox(left: list[float], right: list[float], tolerance: float = 1e-4) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right, strict=True))


def _intersection_area(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _latest_proposals(
    path: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    # ``proposals.jsonl`` is append-only and resumable. A failed attempt may
    # legitimately precede a successful retry for the same key. Validate the
    # latest indexable attempt; malformed keys remain fatal because they cannot
    # participate in the latest-wins contract.
    latest_with_rows: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    invalid: list[dict[str, Any]] = []
    for number, row in enumerate(read_jsonl(path), 1):
        example_id = str(row.get("example_id", "")).strip()
        model = str(row.get("model_family", "")).strip()
        if not example_id or model not in MODELS:
            invalid.append({"row": number, "reason": "invalid_proposal_key"})
            continue
        latest_with_rows[(example_id, model)] = (number, row)

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for key, (number, row) in latest_with_rows.items():
        latest[key] = row
        if row.get("schema_version") != GROUNDING_REQUEST_SCHEMA:
            invalid.append({"row": number, "reason": "proposal_schema_mismatch"})
        if row.get("status") != "ok":
            invalid.append({"row": number, "reason": "proposal_not_ok"})
        if not isinstance(row.get("candidates"), list):
            invalid.append({"row": number, "reason": "proposal_candidates_invalid"})
    return latest, invalid


def _candidate_by_index(
    proposal: Mapping[str, Any], candidate_index: Any
) -> Mapping[str, Any]:
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise ValueError("candidate_index_missing_or_invalid")
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("proposal_candidates_invalid")
    matches = [
        candidate
        for index, candidate in enumerate(candidates)
        if int(candidate.get("candidate_index", index)) == candidate_index
    ]
    if len(matches) != 1:
        raise ValueError("candidate_index_not_unique")
    return matches[0]


def _audit_region(
    value: Any,
    *,
    expected_image: Path, expected_image_sha256: str,
    image_size: tuple[int, int],
    proposal: Mapping[str, Any],
) -> tuple[list[float], str, bool]:
    if not isinstance(value, Mapping):
        raise ValueError("region_not_mapping")
    if Path(str(value.get("image_path", ""))).resolve() != expected_image:
        raise ValueError("region_image_not_frozen_target")
    if value.get("image_sha256") != expected_image_sha256:
        raise ValueError("region_image_sha_mismatch")
    bbox = _bbox(value.get("bbox"))
    width, height = image_size
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
        raise ValueError("region_bbox_out_of_bounds")
    source = str(value.get("source", ""))
    if source == "manual_bbox":
        return bbox, source, False
    if source != "sam3_candidate":
        raise ValueError("region_source_invalid")
    candidate = _candidate_by_index(proposal, value.get("candidate_index"))
    raw, expected_bbox, clipped = _clip_bbox(candidate.get("bbox"), image_size)
    if not _same_bbox(bbox, expected_bbox):
        raise ValueError("candidate_bbox_differs_after_clip")
    recorded_raw = value.get("raw_proposal_bbox")
    if recorded_raw is None or not _same_bbox(_bbox(recorded_raw), raw):
        raise ValueError("raw_proposal_bbox_mismatch")
    if value.get("bbox_clipped") is not clipped:
        raise ValueError("bbox_clipped_flag_mismatch")
    return bbox, source, clipped


def audit_grounding_review(
    requests_path: str | Path,
    reviews_path: str | Path,
    output_dir: str | Path,
    *,
    proposals_path: str | Path | None = None,
) -> dict[str, Any]:
    requests_path = Path(requests_path).resolve()
    reviews_path = Path(reviews_path).resolve()
    output_dir = Path(output_dir).resolve()
    proposals = Path(proposals_path).resolve() if proposals_path is not None else None
    requests, duplicate_requests = _index_unique(requests_path, kind="request")
    reviews, duplicate_reviews = _index_unique(reviews_path, kind="review")
    request_digest = sha256_file(requests_path)
    reviews_digest = sha256_file(reviews_path)
    proposals_digest = sha256_file(proposals) if proposals is not None else None
    invalid: list[dict[str, Any]] = []
    if duplicate_requests:
        invalid.append(
            {
                "reason": "duplicate_request_ids",
                "example_ids": duplicate_requests,
            }
        )
    bad_request_schema = sorted(
        example_id
        for example_id, row in requests.items()
        if row.get("schema_version") != GROUNDING_REQUEST_SCHEMA
    )
    if bad_request_schema:
        invalid.append(
            {
                "reason": "request_schema_mismatch",
                "example_ids": bad_request_schema,
            }
        )
    if duplicate_reviews:
        invalid.append({"reason": "duplicate_review_ids", "example_ids": duplicate_reviews})
    if proposals is None or not proposals.is_file():
        invalid.append({"reason": "proposals_path_required"})
        proposal_rows: dict[tuple[str, str], dict[str, Any]] = {}
    else:
        proposal_rows, proposal_invalid = _latest_proposals(proposals)
        invalid.extend(proposal_invalid)

    session_path = reviews_path.parent / "review_session.json"
    session: dict[str, Any] | None = None
    if not session_path.is_file():
        invalid.append({"reason": "review_session_missing"})
    else:
        try:
            loaded = json.loads(session_path.read_text(encoding="utf-8"))
            session = loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            session = None
        if session is None:
            invalid.append({"reason": "review_session_invalid"})
        else:
            expected_session = {
                "schema_version": SESSION_SCHEMA_VERSION,
                "requests_path": str(requests_path),
                "requests_sha256": request_digest,
                "proposals_path": str(proposals) if proposals is not None else "",
                "proposals_sha256": proposals_digest,
                "review_schema_version": GROUNDING_REVIEW_SCHEMA,
                "reviewer_id": session.get("reviewer_id"),
                "frozen_images_verified": True,
            }
            if session != expected_session:
                invalid.append({"reason": "review_session_source_mismatch"})

    unknown = sorted(set(reviews) - set(requests))
    missing = sorted(set(requests) - set(reviews))
    expected_pairs = {(example_id, model) for example_id in requests for model in MODELS}
    if proposal_rows and set(proposal_rows) != expected_pairs:
        invalid.append(
            {
                "reason": "proposal_coverage_mismatch",
                "missing_count": len(expected_pairs - set(proposal_rows)),
                "extra_count": len(set(proposal_rows) - expected_pairs),
            }
        )

    dispositions: Counter[str] = Counter()
    region_sources: Counter[str] = Counter()
    clipped_regions = 0
    image_sha_cache: dict[Path, str] = {}
    image_size_cache: dict[Path, tuple[int, int]] = {}
    eligible_ids: set[str] = set()
    for example_id, review in sorted(reviews.items()):
        status = str(review.get("status", ""))
        dispositions[status] += 1
        reasons: list[str] = []
        if review.get("schema_version") != GROUNDING_REVIEW_SCHEMA:
            reasons.append("review_schema_mismatch")
        if review.get("human_reviewed") is not True:
            reasons.append("human_reviewed_not_true")
        review_id = str(review.get("review_id", "")).strip()
        if not review_id:
            reasons.append("review_id_missing")
        if session is not None and review_id != str(session.get("reviewer_id", "")):
            reasons.append("reviewer_differs_from_session")
        if review.get("request_sha256") != request_digest:
            reasons.append("request_sha256_mismatch")
        if review.get("proposals_sha256") != proposals_digest:
            reasons.append("proposals_sha256_mismatch")
        if not str(review.get("reviewed_at", "")).strip():
            reasons.append("reviewed_at_missing")
        if status not in {"eligible", "ineligible"}:
            reasons.append("invalid_status")
        elif status == "ineligible":
            if not str(review.get("ineligible_reason", "")).strip():
                reasons.append("ineligible_reason_missing")
            if review.get("models") not in ({}, None):
                reasons.append("ineligible_models_must_be_empty")
        else:
            models = review.get("models")
            if not isinstance(models, Mapping) or set(models) != set(MODELS):
                reasons.append("eligible_models_not_exact")
            else:
                for model in MODELS:
                    try:
                        request = requests[example_id]
                        expected_image, expected_sha = _request_image(request, model)
                        if expected_image not in image_sha_cache:
                            image_sha_cache[expected_image] = sha256_file(expected_image)
                            with Image.open(expected_image) as image:
                                image_size_cache[expected_image] = tuple(map(int, image.size))
                        if image_sha_cache[expected_image] != expected_sha:
                            raise ValueError("frozen_image_sha_mismatch")
                        proposal = proposal_rows[(example_id, model)]
                        if proposal.get("status") != "ok":
                            raise ValueError("proposal_not_ok")
                        if Path(str(proposal.get("image_path", ""))).resolve() != expected_image:
                            raise ValueError("proposal_image_path_mismatch")
                        if str(proposal.get("image_sha256", "")) != expected_sha:
                            raise ValueError("proposal_image_sha_mismatch")
                        model_review = models[model]
                        if not isinstance(model_review, Mapping):
                            raise ValueError("model_review_not_mapping")
                        if Path(str(model_review.get("proposal_image_path", ""))).resolve() != expected_image:
                            raise ValueError("recorded_proposal_image_path_mismatch")
                        target, target_source, target_clipped = _audit_region(
                            model_review.get("target"), expected_image=expected_image,
                            expected_image_sha256=expected_sha,
                            image_size=image_size_cache[expected_image], proposal=proposal,
                        )
                        wrong, wrong_source, wrong_clipped = _audit_region(
                            model_review.get("wrong_region"), expected_image=expected_image,
                            expected_image_sha256=expected_sha,
                            image_size=image_size_cache[expected_image], proposal=proposal,
                        )
                        if _intersection_area(target, wrong) > 0:
                            raise ValueError("target_wrong_bbox_overlap")
                        region_sources[f"target/{target_source}"] += 1
                        region_sources[f"wrong_region/{wrong_source}"] += 1
                        clipped_regions += int(target_clipped) + int(wrong_clipped)
                    except (KeyError, OSError, ValueError) as exc:
                        reasons.append(f"{model}:{exc}")
        if reasons:
            invalid.append({"example_id": example_id, "reasons": reasons})
        elif status == "eligible":
            eligible_ids.add(example_id)

    group_members: dict[str, set[str]] = defaultdict(set)
    for example_id, request in requests.items():
        group_members[str(request.get("group_id", ""))].add(example_id)
    complete_groups = {
        group_id for group_id, members in group_members.items() if members <= eligible_ids
    }
    complete_ids = {
        example_id
        for group_id in complete_groups
        for example_id in group_members[group_id]
    }
    eligible_rows = [requests[value] for value in sorted(eligible_ids)]
    result: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "review_schema_version": GROUNDING_REVIEW_SCHEMA,
        "grounding_mode": "human_reviewed",
        "passed": not unknown and not missing and not invalid,
        "expected_count": len(requests),
        "request_count": len(requests),
        "review_count": len(reviews),
        "requests_path": str(requests_path),
        "requests_sha256": request_digest,
        "reviews_path": str(reviews_path),
        "reviews_sha256": reviews_digest,
        "review_sha256": reviews_digest,
        "proposals_path": str(proposals) if proposals is not None else None,
        "proposals_sha256": proposals_digest,
        "session_path": str(session_path),
        "unknown_example_ids": unknown,
        "missing_example_ids": missing,
        "duplicate_request_ids": duplicate_requests,
        "duplicate_review_ids": duplicate_reviews,
        "invalid": invalid,
        "dispositions": dict(sorted(dispositions.items())),
        "human_reviewed": True,
        "eligible_example_count": len(eligible_ids),
        "ineligible_example_count": dispositions["ineligible"],
        "complete_group_count": len(complete_groups),
        "complete_group_example_count": len(complete_ids),
        "incomplete_or_ineligible_group_count": len(group_members) - len(complete_groups),
        "region_source_counts": dict(sorted(region_sources.items())),
        "bbox_clipped_region_count": clipped_regions,
        "eligible_partition_counts": dict(
            sorted(Counter(str(row.get("partition", "")) for row in eligible_rows).items())
        ),
        "eligible_task_counts": dict(
            sorted(Counter(str(row.get("task_id", "")) for row in eligible_rows).items())
        ),
        "eligible_example_ids": sorted(eligible_ids),
        "complete_group_ids": sorted(complete_groups),
        "complete_group_example_ids": sorted(complete_ids),
        "wrong_region_token_preflight_passed": False,
        "wrong_region_token_preflight_note": (
            "Pixel-level non-overlap is audited here; equal-size/disjoint processor-token "
            "preflight is still required before candidate-wrong causal controls."
        ),
    }
    result["fingerprint"] = object_fingerprint(result)
    write_json(output_dir / "review_audit.json", result)
    write_json(output_dir / "eligible_example_ids.json", sorted(eligible_ids))
    write_json(output_dir / "complete_group_example_ids.json", sorted(complete_ids))
    return result


def _tracking_fingerprint_valid(value: Any, field: str = "fingerprint") -> bool:
    if not isinstance(value, Mapping):
        return False
    payload = dict(value)
    recorded = str(payload.pop(field, ""))
    return bool(recorded) and recorded == object_fingerprint(payload)


def _tracking_expected_source(request: Mapping[str, Any]) -> dict[str, Any]:
    source = request.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("request source is missing")
    result = dict(source)
    result["video"] = request.get("video")
    return result


def _tracking_artifact_index(
    path: Path,
    *,
    requests: Mapping[str, Mapping[str, Any]],
    manual: bool,
) -> tuple[dict[Any, dict[str, Any]], int, list[dict[str, Any]]]:
    """Validate every physical append-only row and return latest valid keys."""

    latest: dict[Any, dict[str, Any]] = {}
    attempts: dict[Any, int] = {}
    invalid: list[dict[str, Any]] = []
    row_count = 0
    for number, row in enumerate(read_jsonl(path), 1):
        row_count += 1
        example_id = str(row.get("example_id", "")).strip()
        candidate_id = str(row.get("selected_candidate_id", "")).strip()
        key: Any = (example_id, candidate_id) if manual else example_id
        reasons: list[str] = []
        if not example_id:
            reasons.append("example_id_missing")
        request = requests.get(example_id)
        if request is None:
            reasons.append("unknown_example_id")
        if manual and not candidate_id:
            reasons.append("manual_candidate_id_missing")
        attempt = row.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            reasons.append("attempt_invalid")
        elif attempt != attempts.get(key, 0) + 1:
            reasons.append("attempt_not_append_only")
        else:
            attempts[key] = attempt
        if row.get("schema_version") != TRACKED_GROUNDING_ARTIFACT_SCHEMA:
            reasons.append("artifact_schema_mismatch")
        if not _tracking_fingerprint_valid(row):
            reasons.append("artifact_fingerprint_invalid")
        if request is not None:
            if row.get("request_fingerprint") != request.get("request_fingerprint"):
                reasons.append("artifact_request_fingerprint_mismatch")
            try:
                if row.get("source") != _tracking_expected_source(request):
                    reasons.append("artifact_source_mismatch")
            except ValueError as exc:
                reasons.append(str(exc))
            for field in ("group_id", "partition", "task_id"):
                if row.get(field) != request.get(field):
                    reasons.append(f"artifact_{field}_mismatch")
        proposal = row.get("proposal")
        proposal_fingerprint = row.get("proposal_fingerprint")
        if (
            not isinstance(proposal, Mapping)
            or proposal.get("schema_version") != TRACKED_GROUNDING_PROPOSAL_SCHEMA
            or not _tracking_fingerprint_valid(proposal)
            or proposal.get("fingerprint") != proposal_fingerprint
        ):
            reasons.append("proposal_binding_invalid")
        tracks = row.get("candidate_tracks")
        seen_candidates: set[str] = set()
        if not isinstance(tracks, list):
            reasons.append("candidate_tracks_invalid")
        else:
            for track_index, track in enumerate(tracks):
                identity = f"track_{track_index}"
                track_candidate = (
                    str(track.get("candidate_id", "")).strip()
                    if isinstance(track, Mapping)
                    else ""
                )
                if (
                    not isinstance(track, Mapping)
                    or track.get("schema_version") != TRACKED_GROUNDING_TRACK_SCHEMA
                    or not _tracking_fingerprint_valid(track)
                ):
                    reasons.append(f"{identity}_schema_or_fingerprint_invalid")
                    continue
                if not track_candidate or track_candidate in seen_candidates:
                    reasons.append(f"{identity}_candidate_identity_invalid")
                seen_candidates.add(track_candidate)
                if (
                    track.get("example_id") != example_id
                    or (
                        request is not None
                        and track.get("request_fingerprint")
                        != request.get("request_fingerprint")
                    )
                    or track.get("proposal_fingerprint") != proposal_fingerprint
                ):
                    reasons.append(f"{identity}_source_binding_invalid")
                expected_source = "manual_bbox" if manual else "sam3_candidate"
                if track.get("source") != expected_source:
                    reasons.append(f"{identity}_source_kind_invalid")
        if manual:
            anchor = row.get("manual_anchor")
            if (
                row.get("selection_source") != "manual_bbox"
                or not isinstance(anchor, Mapping)
                or anchor.get("schema_version")
                != TRACKED_GROUNDING_MANUAL_ANCHOR_SCHEMA
                or anchor.get("manual_anchor_id") != candidate_id
                or derive_manual_anchor_id(
                    example_id,
                    str(anchor.get("first_image_sha256", "")),
                    anchor.get("bbox_xyxy", []),
                )
                != candidate_id
                or not _tracking_fingerprint_valid(anchor)
            ):
                reasons.append("manual_anchor_binding_invalid")
        if reasons:
            invalid.append(
                {
                    "source": "manual_tracking_artifact" if manual else "tracking_artifact",
                    "row": number,
                    "example_id": example_id or None,
                    "reasons": sorted(set(reasons)),
                }
            )
        latest[key] = row
    if row_count == 0:
        invalid.append(
            {
                "source": "manual_tracking_artifact" if manual else "tracking_artifact",
                "reason": "artifact_is_empty",
            }
        )
    return latest, row_count, invalid


def _tracking_verify_file(
    value: Any,
    digest: Any,
    *,
    identity: str,
    cache: dict[Path, str],
) -> Path:
    path = Path(str(value or "")).resolve()
    recorded = str(digest or "")
    if not path.is_file() or len(recorded) != 64:
        raise ValueError(f"{identity}: frozen file/path SHA is missing")
    if path not in cache:
        cache[path] = sha256_file(path)
    if cache[path] != recorded:
        raise ValueError(f"{identity}: frozen file SHA changed")
    return path


def _tracking_bbox(value: Any, *, identity: str) -> list[float]:
    result = _bbox(value)
    return [float(item) for item in result]


def _tracking_asset(
    value: Any,
    expected: Mapping[str, Any],
    *,
    identity: str,
    bbox_field: str,
    locked_obj_id: int,
    file_cache: dict[Path, str],
    image_size_cache: dict[Path, tuple[int, int]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity}: asset is missing")
    expected_path = Path(str(expected.get("image_path", ""))).resolve()
    actual_path = Path(str(value.get("image_path", ""))).resolve()
    expected_sha = str(expected.get("image_sha256", ""))
    if (
        actual_path != expected_path
        or value.get("image_sha256") != expected_sha
        or value.get("source_frame_index") != expected.get("source_frame_index")
    ):
        raise ValueError(f"{identity}: asset differs from frozen request")
    _tracking_verify_file(
        actual_path,
        expected_sha,
        identity=f"{identity}/image",
        cache=file_cache,
    )
    if actual_path not in image_size_cache:
        with Image.open(actual_path) as image:
            image_size_cache[actual_path] = tuple(map(int, image.size))
    width, height = image_size_cache[actual_path]
    bbox = _tracking_bbox(value.get(bbox_field), identity=f"{identity}/bbox")
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height:
        raise ValueError(f"{identity}: bbox exceeds frozen image")
    obj_id = value.get("obj_id")
    if isinstance(obj_id, bool) or not isinstance(obj_id, int) or obj_id != locked_obj_id:
        raise ValueError(f"{identity}: locked obj_id mismatch")
    mask_path = value.get("mask_path")
    mask_sha = value.get("mask_sha256")
    if (mask_path is None) != (mask_sha is None):
        raise ValueError(f"{identity}: mask path/SHA mismatch")
    if mask_path is not None:
        _tracking_verify_file(
            mask_path,
            mask_sha,
            identity=f"{identity}/mask",
            cache=file_cache,
        )
    return {"path": str(actual_path), "sha256": expected_sha, "bbox": bbox}


def _tracking_validate_selected_track(
    track: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    proposal_fingerprint: str,
    candidate_id: str,
    expected_source: str,
    manifest_tracker_fingerprint: str,
    file_cache: dict[Path, str],
    image_size_cache: dict[Path, tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    example_id = str(request["example_id"])
    if (
        track.get("schema_version") != TRACKED_GROUNDING_TRACK_SCHEMA
        or track.get("status") != "ok"
        or not _tracking_fingerprint_valid(track)
        or track.get("example_id") != example_id
        or track.get("candidate_id") != candidate_id
        or track.get("source") != expected_source
        or track.get("request_fingerprint") != request.get("request_fingerprint")
        or track.get("proposal_fingerprint") != proposal_fingerprint
    ):
        raise ValueError("selected track identity/schema/fingerprint is invalid")
    locked_obj_id = track.get("locked_obj_id")
    if isinstance(locked_obj_id, bool) or not isinstance(locked_obj_id, int):
        raise ValueError("selected track has no integer locked_obj_id")

    video = request.get("video")
    if not isinstance(video, Mapping):
        raise ValueError("request video provenance is missing")
    _tracking_verify_file(
        video.get("path"),
        video.get("sha256"),
        identity=f"{example_id}/video",
        cache=file_cache,
    )
    predictor = track.get("predictor_provenance")
    if (
        not isinstance(predictor, Mapping)
        or Path(str(predictor.get("source_video_path", ""))).resolve()
        != Path(str(video.get("path", ""))).resolve()
        or predictor.get("source_video_sha256") != video.get("sha256")
        or predictor.get("tracker_fingerprint") != manifest_tracker_fingerprint
    ):
        raise ValueError("selected track predictor provenance is stale")

    first = request.get("first_frame")
    anchor = track.get("anchor")
    if not isinstance(first, Mapping) or not isinstance(anchor, Mapping):
        raise ValueError("selected track anchor is missing")
    if (
        anchor.get("source_frame_index") != 0
        or Path(str(anchor.get("image_path", ""))).resolve()
        != Path(str(first.get("image_path", ""))).resolve()
        or anchor.get("image_sha256") != first.get("image_sha256")
    ):
        raise ValueError("selected track anchor differs from frozen first frame")
    _tracking_verify_file(
        first.get("image_path"),
        first.get("image_sha256"),
        identity=f"{example_id}/first_frame",
        cache=file_cache,
    )
    _tracking_bbox(anchor.get("bbox_xyxy"), identity=f"{example_id}/anchor")
    if (anchor.get("mask_path") is None) != (anchor.get("mask_sha256") is None):
        raise ValueError("selected track anchor mask path/SHA mismatch")
    if anchor.get("mask_path") is not None:
        _tracking_verify_file(
            anchor.get("mask_path"),
            anchor.get("mask_sha256"),
            identity=f"{example_id}/anchor_mask",
            cache=file_cache,
        )

    frame_count = int(video.get("frame_count", 0))
    expected_indices = list(range(frame_count))
    continuity = track.get("continuity")
    if (
        frame_count < 1
        or not isinstance(continuity, Mapping)
        or continuity.get("expected_frame_indices") != expected_indices
        or continuity.get("observed_frame_indices") != expected_indices
        or continuity.get("missing_frame_indices") != []
        or continuity.get("duplicate_frame_indices") != []
        or continuity.get("locked_id_missing_frame_indices") != []
        or continuity.get("locked_obj_id") != locked_obj_id
        or continuity.get("id_switch_detected") is not False
        or continuity.get("frame_coverage_complete") is not True
    ):
        raise ValueError("selected track continuity contract failed")

    key_rows = request.get("key_frames")
    if not isinstance(key_rows, list):
        raise ValueError("request key frames are missing")
    expected_keyframes: dict[int, Mapping[str, Any]] = {}
    for value in key_rows:
        if not isinstance(value, Mapping):
            raise ValueError("request key frame is malformed")
        index = value.get("source_frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in expected_keyframes:
            raise ValueError("request key frame index is invalid/duplicate")
        expected_keyframes[index] = value
    frames = track.get("frames")
    if not isinstance(frames, list):
        raise ValueError("selected track key frames are missing")
    actual_indices = [
        value.get("source_frame_index") if isinstance(value, Mapping) else None
        for value in frames
    ]
    if actual_indices != sorted(expected_keyframes):
        raise ValueError("selected track key-frame coverage/order differs from request")
    for value in frames:
        index = int(value["source_frame_index"])
        _tracking_asset(
            value,
            expected_keyframes[index],
            identity=f"{example_id}/track_frame/{index}",
            bbox_field="bbox_xyxy",
            locked_obj_id=locked_obj_id,
            file_cache=file_cache,
            image_size_cache=image_size_cache,
        )

    terminals = track.get("terminal_by_model")
    bindings = request.get("model_frame_bindings")
    if (
        not isinstance(terminals, Mapping)
        or set(terminals) != set(MODELS)
        or not isinstance(bindings, Mapping)
        or set(bindings) != set(MODELS)
    ):
        raise ValueError("selected track terminal coverage is incomplete")
    result: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        binding = bindings[model]
        expected = binding.get("terminal") if isinstance(binding, Mapping) else None
        if not isinstance(expected, Mapping):
            raise ValueError(f"{model}: frozen terminal binding is missing")
        result[model] = _tracking_asset(
            terminals[model],
            expected,
            identity=f"{example_id}/terminal/{model}",
            bbox_field="bbox_xyxy",
            locked_obj_id=locked_obj_id,
            file_cache=file_cache,
            image_size_cache=image_size_cache,
        )
    return result


def _tracking_review_bindings(
    row: Mapping[str, Any],
    *,
    example_id: str,
    reviewer_id: str,
    requests_sha256: str,
    tracking_artifact_sha256: str,
    tracking_manifest_sha256: str,
) -> list[str]:
    reasons: list[str] = []
    if row.get("schema_version") != TRACKING_REVIEW_SCHEMA_VERSION:
        reasons.append("review_schema_mismatch")
    if row.get("example_id") != example_id:
        reasons.append("review_example_id_mismatch")
    if row.get("human_reviewed") is not True:
        reasons.append("human_reviewed_not_true")
    if row.get("review_id") != reviewer_id:
        reasons.append("reviewer_differs_from_session")
    if row.get("requests_sha256") != requests_sha256:
        reasons.append("requests_sha256_mismatch")
    if row.get("tracking_artifact_sha256") != tracking_artifact_sha256:
        reasons.append("tracking_artifact_sha256_mismatch")
    if row.get("manifest_sha256") != tracking_manifest_sha256:
        reasons.append("manifest_sha256_mismatch")
    if not str(row.get("reviewed_at", "")).strip():
        reasons.append("reviewed_at_missing")
    if not _tracking_fingerprint_valid(row):
        reasons.append("review_fingerprint_invalid")
    return reasons


def audit_tracked_grounding_review(
    requests_path: str | Path,
    reviews_path: str | Path,
    output_dir: str | Path,
    *,
    tracking_artifact_path: str | Path,
    manual_tracking_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the final tracked-grounding decisions without opening labels."""

    requests_path = Path(requests_path).resolve()
    reviews_path = Path(reviews_path).resolve()
    tracking_path = Path(tracking_artifact_path).resolve()
    manual_path = (
        Path(manual_tracking_artifact_path).resolve()
        if manual_tracking_artifact_path is not None
        else None
    )
    output_dir = Path(output_dir).resolve()
    for source in (requests_path, reviews_path, tracking_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if manual_path is not None and not manual_path.is_file():
        raise FileNotFoundError(manual_path)
    manifest_path = tracking_path.parent / "manifest.json"
    session_path = reviews_path.parent / "review_session.json"
    history_path = reviews_path.parent / "review_history.jsonl"
    for source in (manifest_path, session_path, history_path):
        if not source.is_file():
            raise FileNotFoundError(source)

    requests, duplicate_requests = _index_unique(requests_path, kind="request")
    reviews, duplicate_reviews = _index_unique(reviews_path, kind="review")
    requests_sha256 = sha256_file(requests_path)
    reviews_sha256 = sha256_file(reviews_path)
    tracking_sha256 = sha256_file(tracking_path)
    manifest_sha256 = sha256_file(manifest_path)
    supplied_manual_sha256 = sha256_file(manual_path) if manual_path is not None else None
    invalid: list[dict[str, Any]] = []
    if duplicate_requests:
        invalid.append({"reason": "duplicate_request_ids", "example_ids": duplicate_requests})
    if duplicate_reviews:
        invalid.append({"reason": "duplicate_review_ids", "example_ids": duplicate_reviews})
    request_source_sha_cache: dict[Path, str] = {}

    def verify_request_source(
        path_value: Any, digest_value: Any, *, identity: str
    ) -> Path:
        path = Path(str(path_value or "")).resolve()
        digest = str(digest_value or "")
        if not path.is_file() or len(digest) != 64:
            raise ValueError(f"{identity}: missing frozen file or SHA")
        actual = request_source_sha_cache.setdefault(path, sha256_file(path))
        if actual != digest:
            raise ValueError(f"{identity}: frozen file content changed")
        return path

    for example_id, request in requests.items():
        reasons = []
        if request.get("schema_version") != TRACKED_GROUNDING_REQUEST_SCHEMA:
            reasons.append("request_schema_mismatch")
        if request.get("status") != "ok":
            reasons.append("request_status_not_ok")
        payload = dict(request)
        recorded = str(payload.pop("request_fingerprint", ""))
        if not recorded or recorded != object_fingerprint(payload):
            reasons.append("request_fingerprint_invalid")
        bindings = request.get("model_frame_bindings")
        if not isinstance(bindings, Mapping):
            reasons.append("request_model_bindings_missing")
        else:
            try:
                rr_contract = validate_processor_content_order_contract(
                    bindings.get("roboreward", {}),
                    identity=f"{example_id}/roboreward",
                    verify_file=verify_request_source,
                )
                if (
                    rr_contract is not None
                    and "processor_content_order_contract"
                    in bindings.get("qwen", {})
                ):
                    raise ValueError("qwen carries a RoboReward order contract")
            except (OSError, TypeError, ValueError) as exc:
                reasons.append(f"processor_content_order_contract_invalid:{exc}")
        if reasons:
            invalid.append({"example_id": example_id, "reasons": reasons})

    auto_latest, auto_row_count, auto_invalid = _tracking_artifact_index(
        tracking_path, requests=requests, manual=False
    )
    invalid.extend(auto_invalid)
    manual_latest: dict[Any, dict[str, Any]] = {}
    manual_row_count = 0
    if manual_path is not None:
        manual_latest, manual_row_count, manual_invalid = _tracking_artifact_index(
            manual_path, requests=requests, manual=True
        )
        invalid.extend(manual_invalid)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tracking manifest JSON: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Tracking manifest must be a JSON object")
    manifest_reasons: list[str] = []
    if manifest.get("schema_version") != TRACKED_GROUNDING_MANIFEST_SCHEMA:
        manifest_reasons.append("manifest_schema_mismatch")
    expected_manifest = {
        "status": "complete",
        "coverage_complete": True,
        "request_bindings_current": True,
        "labels_opened": False,
        "requests_sha256": requests_sha256,
        "tracks_sha256": tracking_sha256,
        "request_count": len(requests),
        "artifact_count": len(auto_latest),
        "artifact_row_count": auto_row_count,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            manifest_reasons.append(f"manifest_{field}_mismatch")
    if Path(str(manifest.get("requests_path", ""))).resolve() != requests_path:
        manifest_reasons.append("manifest_requests_path_mismatch")
    if Path(str(manifest.get("tracks_path", ""))).resolve() != tracking_path:
        manifest_reasons.append("manifest_tracks_path_mismatch")
    expected_status_counts = dict(
        sorted(Counter(str(value.get("status", "")) for value in auto_latest.values()).items())
    )
    if manifest.get("status_counts") != expected_status_counts:
        manifest_reasons.append("manifest_status_counts_mismatch")
    proposal_backend = manifest.get("proposal_backend")
    if (
        not isinstance(proposal_backend, Mapping)
        or not _tracking_fingerprint_valid(proposal_backend, "proposer_fingerprint")
    ):
        manifest_reasons.append("manifest_proposal_backend_invalid")
    tracker = manifest.get("tracker")
    tracker_fingerprint = (
        str(tracker.get("tracker_fingerprint", ""))
        if isinstance(tracker, Mapping)
        else ""
    )
    if (
        not isinstance(tracker, Mapping)
        or tracker.get("backend") != "official_sam3_video_predictor"
        or not tracker_fingerprint
    ):
        manifest_reasons.append("manifest_tracker_invalid")
    if set(auto_latest) != set(requests):
        manifest_reasons.append("tracking_artifact_coverage_mismatch")
    if manifest_reasons:
        invalid.append({"source": "tracking_manifest", "reasons": manifest_reasons})

    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tracking review session JSON: {session_path}") from exc
    if not isinstance(session, Mapping):
        raise ValueError("Tracking review session must be a JSON object")
    reviewer_id = str(session.get("reviewer_id", "")).strip()
    expected_session = {
        "schema_version": TRACKING_SESSION_SCHEMA_VERSION,
        "requests_path": str(requests_path),
        "requests_sha256": requests_sha256,
        "tracking_artifact_path": str(tracking_path),
        "tracking_artifact_sha256": tracking_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "review_schema_version": TRACKING_REVIEW_SCHEMA_VERSION,
        "reviewer_id": reviewer_id,
        "frozen_sources_verified": True,
    }
    if not reviewer_id or dict(session) != expected_session:
        invalid.append({"source": "review_session", "reason": "session_source_mismatch"})

    history_latest: dict[str, dict[str, Any]] = {}
    frozen_manual_anchors: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(read_jsonl(history_path), 1):
        example_id = str(row.get("example_id", "")).strip()
        reasons = _tracking_review_bindings(
            row,
            example_id=example_id,
            reviewer_id=reviewer_id,
            requests_sha256=requests_sha256,
            tracking_artifact_sha256=tracking_sha256,
            tracking_manifest_sha256=manifest_sha256,
        )
        if not example_id or example_id not in requests:
            reasons.append("history_example_id_unknown")
        history_status = row.get("status")
        if history_status not in {"eligible", "needs_retrack", "skipped"}:
            reasons.append("history_status_invalid")
        elif history_status == "needs_retrack":
            decision = row.get("decision")
            try:
                bbox = _bbox(
                    decision.get("first_bbox")
                    if isinstance(decision, Mapping)
                    else None
                )
                image_sha = str(
                    decision.get("first_image_sha256", "")
                    if isinstance(decision, Mapping)
                    else ""
                )
                anchor_id = derive_manual_anchor_id(
                    example_id, image_sha, bbox
                )
                value = {
                    "manual_anchor_id": anchor_id,
                    "bbox_xyxy": bbox,
                    "first_image_sha256": image_sha,
                    "reviewer_id": reviewer_id,
                }
                if (
                    row.get("manual_anchor_id") != anchor_id
                    or not isinstance(decision, Mapping)
                    or decision.get("source") != "manual_first_bbox"
                ):
                    raise ValueError("history manual anchor identity differs")
                frozen = frozen_manual_anchors.get(example_id)
                if frozen is not None and frozen != value:
                    raise ValueError("history manual anchor changed after first freeze")
                frozen_manual_anchors[example_id] = value
            except (TypeError, ValueError) as exc:
                reasons.append(f"history_manual_anchor_invalid:{exc}")
        elif history_status == "skipped":
            disposition = row.get("disposition")
            code = (
                str(disposition.get("code", ""))
                if isinstance(disposition, Mapping)
                else ""
            )
            if code not in TRACKING_SKIP_CODES:
                reasons.append("skipped_disposition_code_invalid")
            if row.get("models") not in ({}, None):
                reasons.append("skipped_models_must_be_empty")
            if any(field in row for field in ("reason", "note", "wrong_region")):
                reasons.append("skipped_free_text_or_control_forbidden")
        if reasons:
            invalid.append(
                {
                    "source": "review_history",
                    "row": number,
                    "example_id": example_id or None,
                    "reasons": sorted(set(reasons)),
                }
            )
        if example_id:
            history_latest[example_id] = row
    if history_latest != reviews:
        invalid.append({"source": "review_history", "reason": "materialized_reviews_not_latest"})

    unknown = sorted(set(reviews) - set(requests))
    missing = sorted(set(requests) - set(reviews))
    dispositions: Counter[str] = Counter()
    eligible_ids: set[str] = set()
    used_manual = False
    file_cache: dict[Path, str] = {}
    image_size_cache: dict[Path, tuple[int, int]] = {}
    for example_id, review in sorted(reviews.items()):
        status = str(review.get("status", ""))
        dispositions[status] += 1
        reasons = _tracking_review_bindings(
            review,
            example_id=example_id,
            reviewer_id=reviewer_id,
            requests_sha256=requests_sha256,
            tracking_artifact_sha256=tracking_sha256,
            tracking_manifest_sha256=manifest_sha256,
        )
        if status not in TRACKING_FINAL_STATUSES:
            reasons.append("final_status_must_be_eligible_or_skipped")
        request = requests.get(example_id)
        if request is None:
            reasons.append("review_has_unknown_request")
        elif status == "eligible":
            decision = review.get("decision")
            candidate_id = str(review.get("selected_candidate_id", "")).strip()
            source = str(decision.get("source", "")) if isinstance(decision, Mapping) else ""
            is_manual = source == "accept_manual_track"
            selected_artifact = (
                manual_latest.get((example_id, candidate_id))
                if is_manual
                else auto_latest.get(example_id)
            )
            if not candidate_id or source not in {
                "accept_default",
                "select_alternative",
                "accept_manual_track",
            }:
                reasons.append("eligible_decision_identity_invalid")
            if selected_artifact is None:
                reasons.append("selected_tracking_artifact_missing")
            else:
                proposal = selected_artifact.get("proposal")
                proposal_fp = str(selected_artifact.get("proposal_fingerprint", ""))
                tracks = selected_artifact.get("candidate_tracks")
                matches = [
                    value
                    for value in tracks
                    if isinstance(value, Mapping)
                    and value.get("candidate_id") == candidate_id
                ] if isinstance(tracks, list) else []
                if len(matches) != 1:
                    reasons.append("selected_track_missing_or_ambiguous")
                else:
                    try:
                        terminals = _tracking_validate_selected_track(
                            matches[0],
                            request=request,
                            proposal_fingerprint=proposal_fp,
                            candidate_id=candidate_id,
                            expected_source="manual_bbox" if is_manual else "sam3_candidate",
                            manifest_tracker_fingerprint=tracker_fingerprint,
                            file_cache=file_cache,
                            image_size_cache=image_size_cache,
                        )
                        if not isinstance(decision, Mapping):
                            raise ValueError("eligible decision mapping is missing")
                        if decision.get("selected_candidate_id") != candidate_id:
                            raise ValueError("decision candidate differs from review")
                        candidate_provenance = decision.get("candidate_provenance")
                        if not isinstance(candidate_provenance, Mapping):
                            raise ValueError("candidate provenance is missing")
                        if is_manual:
                            anchor = selected_artifact.get("manual_anchor")
                            expected_selection = "manual_track"
                            expected_bbox = anchor.get("bbox_xyxy") if isinstance(anchor, Mapping) else None
                            frozen_anchor = frozen_manual_anchors.get(example_id)
                            if (
                                not isinstance(anchor, Mapping)
                                or frozen_anchor is None
                                or anchor.get("reviewer_id") != reviewer_id
                                or anchor.get("manual_anchor_id")
                                != frozen_anchor["manual_anchor_id"]
                                or anchor.get("first_image_sha256")
                                != frozen_anchor["first_image_sha256"]
                                or not _same_bbox(
                                    _bbox(anchor.get("bbox_xyxy")),
                                    frozen_anchor["bbox_xyxy"],
                                )
                            ):
                                raise ValueError(
                                    "manual track is not bound to the frozen "
                                    "first-review anchor/reviewer"
                                )
                            if review.get("manual_tracking_artifact_sha256") != supplied_manual_sha256:
                                raise ValueError("manual artifact SHA is absent or stale")
                            used_manual = True
                        else:
                            options = proposal.get("options") if isinstance(proposal, Mapping) else None
                            candidates = [
                                value for value in options
                                if isinstance(value, Mapping)
                                and value.get("candidate_id") == candidate_id
                            ] if isinstance(options, list) else []
                            if len(candidates) != 1:
                                raise ValueError("selected candidate is not one frozen option")
                            option = candidates[0]
                            expected_selection = str(option.get("selection", ""))
                            expected_bbox = option.get("bbox_xyxy")
                            expected_decision = (
                                "accept_default"
                                if expected_selection == "algorithmic_default"
                                else "select_alternative"
                            )
                            if source != expected_decision:
                                raise ValueError("decision source differs from option provenance")
                            if review.get("manual_tracking_artifact_sha256") is not None:
                                raise ValueError("automatic decision carries manual artifact SHA")
                        if candidate_provenance.get("selection") != expected_selection:
                            raise ValueError("candidate selection provenance mismatch")
                        if not _same_bbox(
                            _tracking_bbox(candidate_provenance.get("first_bbox"), identity="candidate first bbox"),
                            _tracking_bbox(expected_bbox, identity="frozen candidate bbox"),
                        ):
                            raise ValueError("candidate first bbox changed")
                        track_provenance = decision.get("track_provenance")
                        expected_track_provenance = {
                            "artifact_fingerprint": selected_artifact.get("fingerprint"),
                            "track_fingerprint": matches[0].get("fingerprint"),
                            "track_source": matches[0].get("source"),
                            "request_fingerprint": matches[0].get("request_fingerprint"),
                            "proposal_fingerprint": matches[0].get("proposal_fingerprint"),
                            "locked_obj_id": matches[0].get("locked_obj_id"),
                            "continuity": matches[0].get("continuity"),
                            "predictor_provenance": matches[0].get("predictor_provenance"),
                        }
                        if track_provenance != expected_track_provenance:
                            raise ValueError("review track provenance differs from artifact")
                        models = review.get("models")
                        if not isinstance(models, Mapping) or set(models) != set(MODELS):
                            raise ValueError("eligible review model coverage is incomplete")
                        for model in MODELS:
                            model_row = models[model]
                            target = model_row.get("target") if isinstance(model_row, Mapping) else None
                            terminal = matches[0]["terminal_by_model"][model]
                            if not isinstance(target, Mapping):
                                raise ValueError(f"{model}: reviewed terminal target is missing")
                            if (
                                target.get("source") != "tracked_sam3"
                                or target.get("candidate_id") != candidate_id
                                or target.get("obj_id") != matches[0].get("locked_obj_id")
                                or target.get("source_frame_index") != terminal.get("source_frame_index")
                                or Path(str(target.get("image_path", ""))).resolve()
                                != Path(str(terminal.get("image_path", ""))).resolve()
                                or target.get("image_sha256") != terminal.get("image_sha256")
                                or not _same_bbox(
                                    _tracking_bbox(target.get("bbox"), identity=f"{model} review bbox"),
                                    terminals[model]["bbox"],
                                )
                            ):
                                raise ValueError(f"{model}: reviewed target differs from locked terminal")
                    except (KeyError, OSError, ValueError) as exc:
                        reasons.append(f"eligible_track_invalid:{exc}")
            if not reasons:
                eligible_ids.add(example_id)
        elif status == "skipped":
            disposition = review.get("disposition")
            code = str(disposition.get("code", "")) if isinstance(disposition, Mapping) else ""
            if code not in TRACKING_SKIP_CODES:
                reasons.append("skipped_disposition_code_invalid")
            if review.get("models") not in ({}, None):
                reasons.append("skipped_models_must_be_empty")
            if any(field in review for field in ("reason", "note", "wrong_region")):
                reasons.append("skipped_free_text_or_control_forbidden")
        if reasons:
            invalid.append({"example_id": example_id, "reasons": sorted(set(reasons))})

    group_members: dict[str, set[str]] = defaultdict(set)
    for example_id, request in requests.items():
        group_members[str(request.get("group_id", ""))].add(example_id)
    complete_groups = {
        group_id for group_id, members in group_members.items() if members <= eligible_ids
    }
    complete_ids = {
        example_id for group_id in complete_groups for example_id in group_members[group_id]
    }
    manual_sha256 = supplied_manual_sha256 if used_manual else None
    if used_manual and manual_path is None:
        invalid.append({"reason": "accepted_manual_track_without_manual_artifact"})
    passed = not unknown and not missing and not invalid
    incomplete_groups = len(group_members) - len(complete_groups)
    result: dict[str, Any] = {
        "audit_schema_version": TRACKING_AUDIT_SCHEMA_VERSION,
        "review_schema_version": TRACKING_REVIEW_SCHEMA_VERSION,
        "grounding_mode": "human_reviewed",
        "passed": passed,
        "human_reviewed": True,
        "expected_count": len(requests),
        "request_count": len(requests),
        "review_count": len(reviews),
        "requests_path": str(requests_path),
        "requests_sha256": requests_sha256,
        "reviews_path": str(reviews_path),
        "reviews_sha256": reviews_sha256,
        "review_sha256": reviews_sha256,
        "review_history_path": str(history_path),
        "review_history_sha256": sha256_file(history_path),
        "tracking_artifact_path": str(tracking_path),
        "tracking_artifact_sha256": tracking_sha256,
        "tracking_manifest_path": str(manifest_path),
        "tracking_manifest_sha256": manifest_sha256,
        "manual_tracking_artifact_path": str(manual_path) if used_manual and manual_path else None,
        "manual_tracking_artifact_sha256": manual_sha256,
        "tracking_artifact_row_count": auto_row_count,
        "manual_tracking_artifact_row_count": manual_row_count if used_manual else 0,
        "session_path": str(session_path),
        "unknown_example_ids": unknown,
        "missing_example_ids": missing,
        "duplicate_request_ids": duplicate_requests,
        "duplicate_review_ids": duplicate_reviews,
        "invalid": invalid,
        "dispositions": dict(sorted(dispositions.items())),
        "eligible_example_count": len(eligible_ids),
        "skipped_example_count": dispositions["skipped"],
        "ineligible_example_count": dispositions["skipped"],
        "complete_group_count": len(complete_groups),
        "complete_group_example_count": len(complete_ids),
        "incomplete_or_skipped_group_count": incomplete_groups,
        "incomplete_or_ineligible_group_count": incomplete_groups,
        "eligible_example_ids": sorted(eligible_ids),
        "complete_group_ids": sorted(complete_groups),
        "complete_group_example_ids": sorted(complete_ids),
        "target_grounding_scope": "terminal_only",
        "control_region_policy": "none",
        "tracking_continuity_verified": passed,
        "labels_opened": False,
    }
    result["fingerprint"] = object_fingerprint(result)
    write_json(output_dir / "review_audit.json", result)
    write_json(output_dir / "eligible_example_ids.json", sorted(eligible_ids))
    write_json(output_dir / "complete_group_example_ids.json", sorted(complete_ids))
    return result
