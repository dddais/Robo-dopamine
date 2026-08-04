"""Build explicitly unreviewed, exploratory grounding decisions.

Selections here are deterministic assumptions over frozen proposals.  They are
deliberately distinguished from human-audited grounding.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from numbers import Real
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import section
from ..io import object_fingerprint, read_jsonl, sha256_file, write_json, write_jsonl


ASSUMED_REVIEW_SCHEMA = "my_dataset.assumed_grounding.v1"
MODELS = ("roboreward", "qwen", "grm")
STRICT_STATUS = "assumed_valid"
PROXY_STATUS = "assumed_proxy"
STRICT_GROUNDING_STATUS = "auto_assumed_unreviewed"
PROXY_GROUNDING_STATUS = "auto_proxy_unreviewed"
PROPOSAL_SCHEMAS = {
    "my_dataset.grounding_request.v1",
    "my_dataset.grounding_supplement.v1",
}
SIMPLE_STRATEGIES = {"object_identity", "attribute_color", "simple"}
ORDINAL_INDEX = {"first": 0, "second": 1, "third": 2, "fourth": 3}
FALLBACK_SELECTION_METHODS = {
    "destination_proxy_for_missing_target",
    "ordinal_clamped_to_available_candidates",
    "highest_score_target_without_reference",
    "nearest_target_without_requested_side",
}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _candidate_index_key(value: Any) -> tuple[int, int | str]:
    """Return a comparable tie key even when candidate_index is malformed."""
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value)
    return (1, f"{type(value).__name__}:{value!r}")


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -candidate["score"],
        *candidate["bbox"],
        _candidate_index_key(candidate.get("candidate_index")),
        candidate["source_index"],
    )


def _normalise_candidate(
    value: Any, *, source_index: int, query: str
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, dict):
        return None, "candidate_not_object"
    bbox = value.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(_finite_number(item) for item in bbox)
    ):
        return None, "invalid_bbox"
    normalised_bbox = [float(item) for item in bbox]
    if (
        normalised_bbox[2] <= normalised_bbox[0]
        or normalised_bbox[3] <= normalised_bbox[1]
    ):
        return None, "invalid_bbox"
    score = value.get("score")
    if not _finite_number(score):
        return None, "invalid_score"
    return (
        {
            "bbox": normalised_bbox,
            "score": float(score),
            "query": query,
            "candidate_index": value.get("candidate_index"),
            "source_index": source_index,
            "proposal_source_path": value.get("proposal_source_path"),
            "proposal_source_sha256": value.get("proposal_source_sha256"),
        },
        None,
    )


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _query_candidates(
    raw_candidates: list[Any], query: str, iou_threshold: float
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter by exact query, validate, then suppress query-local overlap."""
    candidates: list[dict[str, Any]] = []
    reasons: list[str] = []
    for source_index, value in enumerate(raw_candidates):
        if not isinstance(value, dict) or value.get("query") != query:
            continue
        candidate, reason = _normalise_candidate(
            value, source_index=source_index, query=query
        )
        if reason is not None:
            reasons.append(reason)
        elif candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=_candidate_sort_key)
    deduplicated: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            _iou(candidate["bbox"], kept["bbox"]) >= iou_threshold
            for kept in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    return deduplicated, sorted(set(reasons))


def _center(candidate: dict[str, Any]) -> tuple[float, float]:
    bbox = candidate["bbox"]
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)
    return math.hypot(first_x - second_x, first_y - second_y)


def _select_candidate(
    request: dict[str, Any],
    proposal: dict[str, Any],
    *,
    iou_threshold: float,
    allow_exploratory_fallbacks: bool = False,
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    roles = request.get("roles")
    if not isinstance(roles, dict):
        return None, ["missing_roles"], None
    strategy = roles.get("grounding_strategy")
    if not isinstance(strategy, str) or not strategy:
        return None, ["missing_grounding_strategy"], None
    target_query = roles.get("target_phrase")
    if not isinstance(target_query, str) or not target_query:
        return None, ["missing_target_phrase"], None
    raw_candidates = proposal.get("candidates")
    if not isinstance(raw_candidates, list):
        return None, ["missing_candidates"], None

    cache: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}

    def candidates_for(query: str) -> tuple[list[dict[str, Any]], list[str]]:
        if query not in cache:
            cache[query] = _query_candidates(raw_candidates, query, iou_threshold)
        return cache[query]

    targets, target_errors = candidates_for(target_query)
    if not targets:
        if target_errors:
            return None, [f"target_{reason}" for reason in target_errors], None
        destination_query = roles.get("destination")
        if allow_exploratory_fallbacks and isinstance(destination_query, str):
            destinations, destination_errors = candidates_for(destination_query)
            if destination_errors:
                return (
                    None,
                    [f"destination_{reason}" for reason in destination_errors],
                    None,
                )
            if destinations:
                return (
                    destinations[0],
                    [],
                    "destination_proxy_for_missing_target",
                )
        return None, ["missing_target_candidate"], None

    if strategy in SIMPLE_STRATEGIES:
        return targets[0], [], "highest_score_exact_query"

    if strategy == "ordinal_position":
        ordinal = roles.get("ordinal")
        direction = roles.get("direction")
        if ordinal not in ORDINAL_INDEX:
            return None, ["missing_or_invalid_ordinal"], None
        if direction not in {"left", "right"}:
            return None, ["missing_or_invalid_direction"], None
        required_index = ORDINAL_INDEX[str(ordinal)]
        ordered = sorted(
            targets,
            key=lambda candidate: (
                _center(candidate)[0]
                if direction == "left"
                else -_center(candidate)[0],
                _center(candidate)[1],
                *_candidate_sort_key(candidate),
            ),
        )
        if len(ordered) <= required_index:
            if allow_exploratory_fallbacks:
                return (
                    ordered[-1],
                    [],
                    "ordinal_clamped_to_available_candidates",
                )
            return None, ["insufficient_target_candidates"], None
        return ordered[required_index], [], "ordinal_x_position_exact_query"

    if strategy not in {"left_right_relation", "distance_relation"}:
        return None, ["unsupported_grounding_strategy"], None
    reference_query = roles.get("reference_object")
    if not isinstance(reference_query, str) or not reference_query:
        return None, ["missing_reference_object"], None
    references, reference_errors = candidates_for(reference_query)
    if not references:
        if reference_errors:
            return None, [f"reference_{reason}" for reason in reference_errors], None
        if allow_exploratory_fallbacks:
            return (
                targets[0],
                [],
                "highest_score_target_without_reference",
            )
        return None, ["missing_reference_candidate"], None
    reference = references[0]

    if strategy == "left_right_relation":
        side = roles.get("relation")
        if side not in {"left", "right"}:
            return None, ["missing_or_invalid_side"], None
        reference_x, _reference_y = _center(reference)
        side_targets = [
            candidate
            for candidate in targets
            if (
                _center(candidate)[0] < reference_x
                if side == "left"
                else _center(candidate)[0] > reference_x
            )
        ]
        if not side_targets:
            if allow_exploratory_fallbacks:
                return (
                    min(
                        targets,
                        key=lambda candidate: (
                            _distance(candidate, reference),
                            *_candidate_sort_key(candidate),
                        ),
                    ),
                    [],
                    "nearest_target_without_requested_side",
                )
            return None, ["missing_target_on_requested_side"], None
        return (
            min(
                side_targets,
                key=lambda candidate: (
                    _distance(candidate, reference),
                    *_candidate_sort_key(candidate),
                ),
            ),
            [],
            "nearest_center_on_requested_reference_side",
        )

    relation = roles.get("relation")
    if not isinstance(relation, str):
        return None, ["missing_or_invalid_distance_relation"], None
    if relation.startswith("closest"):
        distance_direction = 1.0
        selection_method = "closest_euclidean_center_to_reference"
    elif relation.startswith("farthest"):
        distance_direction = -1.0
        selection_method = "farthest_euclidean_center_from_reference"
    else:
        return None, ["missing_or_invalid_distance_relation"], None
    return (
        min(
            targets,
            key=lambda candidate: (
                distance_direction * _distance(candidate, reference),
                *_candidate_sort_key(candidate),
            ),
        ),
        [],
        selection_method,
    )


def _resolved_path(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    return Path(value).resolve()


def _expected_images(request: dict[str, Any], model: str) -> tuple[Path, ...]:
    frames = request.get("model_frames")
    frame = frames.get(model) if isinstance(frames, dict) else None
    if not isinstance(frame, dict):
        return ()
    values: list[Any] = []
    if model in {"roboreward", "qwen"}:
        values.append(frame.get("image_path"))
    else:
        if frame.get("primary_target_slot") not in (None, "after_cam_high"):
            raise ValueError("GRM primary_target_slot must be after_cam_high")
        if frame.get("primary_target_view") not in (None, "front"):
            raise ValueError("GRM primary_target_view must be front")
        terminal_views = frame.get("terminal_views")
        front = terminal_views.get("front") if isinstance(terminal_views, dict) else None
        if isinstance(front, dict):
            values.append(front.get("image_path"))
        image_paths = frame.get("image_paths")
        if isinstance(image_paths, list) and len(image_paths) > 5:
            values.append(image_paths[5])
    resolved = [_resolved_path(value) for value in values]
    present = tuple(value for value in resolved if value is not None)
    if model == "grm" and len(set(present)) > 1:
        raise ValueError("GRM terminal_views.front and image_paths[5] differ")
    return present


def _model_review(
    request: dict[str, Any],
    proposal: dict[str, Any],
    model: str,
    *,
    iou_threshold: float,
    allow_exploratory_fallbacks: bool,
    image_size_cache: dict[Path, tuple[int, int]],
) -> dict[str, Any]:
    proposal_image = _resolved_path(proposal.get("image_path"))
    reasons: list[str] = []
    if proposal.get("status") != "ok":
        reasons.append("proposal_status_not_ok")
    expected_images = _expected_images(request, model)
    if not expected_images:
        reasons.append("missing_request_terminal_image")
    if proposal_image is None:
        reasons.append("missing_proposal_image")
    elif expected_images and proposal_image not in expected_images:
        reasons.append("proposal_image_mismatch")

    selected: dict[str, Any] | None = None
    selection_method: str | None = None
    if not reasons:
        selected, selection_reasons, selection_method = _select_candidate(
            request,
            proposal,
            iou_threshold=iou_threshold,
            allow_exploratory_fallbacks=allow_exploratory_fallbacks,
        )
        reasons.extend(selection_reasons)

    target = None
    if (
        not reasons
        and selected is not None
        and selection_method is not None
        and proposal_image is not None
    ):
        if proposal_image not in image_size_cache:
            with Image.open(proposal_image) as image:
                image_size_cache[proposal_image] = tuple(image.size)
        width, height = image_size_cache[proposal_image]
        original_bbox = list(selected["bbox"])
        clipped_bbox = [
            min(max(original_bbox[0], 0.0), float(width)),
            min(max(original_bbox[1], 0.0), float(height)),
            min(max(original_bbox[2], 0.0), float(width)),
            min(max(original_bbox[3], 0.0), float(height)),
        ]
        if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
            reasons.append("bbox_invalid_after_image_bounds_clip")
            clipped_bbox = []
    if (
        not reasons
        and selected is not None
        and selection_method is not None
        and proposal_image is not None
    ):
        target = {
            "image_path": str(proposal_image),
            "image_sha256": str(proposal.get("image_sha256", "")),
            "bbox": clipped_bbox,
            "bbox_original": original_bbox,
            "bbox_clipped_to_image": clipped_bbox != original_bbox,
            "image_size": [width, height],
            "score": selected["score"],
            "query": selected["query"],
            "selection_method": selection_method,
            "fallback_used": selection_method in FALLBACK_SELECTION_METHODS,
            "proposal_source_path": selected.get("proposal_source_path"),
            "proposal_source_sha256": selected.get("proposal_source_sha256"),
        }
    reasons = sorted(set(reasons))
    return {
        "proposal_image_path": (
            str(proposal_image) if proposal_image is not None else None
        ),
        "target": target,
        "wrong_region": None,
        "valid": target is not None,
        "invalid_reasons": reasons,
    }


def _validate_structure(
    request_rows: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]],
    expected_count: int,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    request_ids = [str(row.get("example_id", "")) for row in request_rows]
    unique_nonempty = set(request_ids) - {""}
    if (
        len(request_rows) != expected_count
        or len(unique_nonempty) != expected_count
        or "" in request_ids
    ):
        counts = Counter(request_ids)
        duplicates = sorted(
            key for key, count in counts.items() if key and count > 1
        )
        raise ValueError(
            f"Expected exactly {expected_count} unique requests; found "
            f"{len(request_rows)} rows, {len(unique_nonempty)} unique non-empty IDs, "
            f"duplicates={duplicates[:10]}"
        )
    requests = dict(zip(request_ids, request_rows))

    expected_proposals = expected_count * len(MODELS)
    if len(proposal_rows) != expected_proposals:
        raise ValueError(
            f"Expected exactly {expected_proposals} proposal rows; "
            f"found {len(proposal_rows)}"
        )
    proposals: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, str]] = []
    unexpected: list[tuple[str, str]] = []
    for row in proposal_rows:
        key = (
            str(row.get("example_id", "")),
            str(row.get("model_family", "")),
        )
        if key in proposals:
            duplicates.append(key)
        proposals[key] = row
        if key[0] not in requests or key[1] not in MODELS:
            unexpected.append(key)
    expected_keys = {
        (example_id, model) for example_id in requests for model in MODELS
    }
    missing = sorted(expected_keys - set(proposals))
    if duplicates or unexpected or missing or set(proposals) != expected_keys:
        raise ValueError(
            "Expected exactly one proposal per request/model; "
            f"duplicates={sorted(set(duplicates))[:10]}, "
            f"unexpected={sorted(set(unexpected))[:10]}, missing={missing[:10]}"
        )
    return requests, proposals


def _requested_queries(
    request: dict[str, Any], proposal: dict[str, Any]
) -> list[str]:
    values: list[Any] = []
    roles = request.get("roles")
    if isinstance(roles, dict):
        values.extend(
            (
                roles.get("target_phrase"),
                roles.get("reference_object"),
                roles.get("destination"),
            )
        )
    proposal_queries = proposal.get("queries")
    if isinstance(proposal_queries, list):
        values.extend(
            value.get("text") if isinstance(value, dict) else None
            for value in proposal_queries
        )
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and value and value not in result:
            result.append(value)
    return result


def _merge_candidate_sources(
    requests: dict[str, dict[str, Any]],
    proposals: dict[tuple[str, str], dict[str, Any]],
    sources: list[tuple[Path, str, list[dict[str, Any]]]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    """Pool only validated candidates by frozen model/image/query."""
    pool: defaultdict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    source_stats: list[dict[str, Any]] = []
    image_sha_cache: dict[Path, str] = {}
    for source_index, (path, source_sha256, rows) in enumerate(sources):
        status_counts = Counter(str(row.get("status", "missing")) for row in rows)
        candidate_count = 0
        for row_number, row in enumerate(rows, 1):
            identity = f"{path}:{row_number}"
            schema = str(row.get("schema_version", ""))
            if schema not in PROPOSAL_SCHEMAS:
                raise ValueError(
                    f"{identity}: unsupported proposal schema {schema!r}"
                )
            if row.get("auto_accepted") is not False:
                raise ValueError(f"{identity}: auto_accepted must be false")
            example_id = str(row.get("example_id", ""))
            model = str(row.get("model_family", ""))
            request = requests.get(example_id)
            if request is None or model not in MODELS:
                raise ValueError(f"{identity}: unknown example/model")
            if str(row.get("group_id", "")) != str(request.get("group_id", "")):
                raise ValueError(f"{identity}: group_id differs from frozen request")
            if str(row.get("partition", "")) != str(
                request.get("partition", "")
            ):
                raise ValueError(
                    f"{identity}: partition differs from frozen request"
                )
            image_path = _resolved_path(row.get("image_path"))
            expected_images = _expected_images(request, model)
            if image_path is None or image_path not in expected_images:
                raise ValueError(
                    f"{identity}: image_path differs from frozen request"
                )
            if image_path not in image_sha_cache:
                image_sha_cache[image_path] = sha256_file(image_path)
            image_sha256 = str(row.get("image_sha256", ""))
            if image_sha256 != image_sha_cache[image_path]:
                raise ValueError(f"{identity}: image_sha256 mismatch")
            raw_candidates = row.get("candidates", [])
            if not isinstance(raw_candidates, list):
                raise ValueError(f"{identity}: candidates must be a list")
            status = str(row.get("status", ""))
            if status == "no_candidate" and raw_candidates:
                raise ValueError(f"{identity}: no_candidate row has candidates")
            if raw_candidates and status != "ok":
                raise ValueError(
                    f"{identity}: candidate-bearing row must be status=ok"
                )
            candidate_count += len(raw_candidates)
            if not raw_candidates:
                continue
            row_query = row.get("query")
            allowed_queries = set(_requested_queries(request, row))
            for candidate_number, raw_candidate in enumerate(raw_candidates):
                if not isinstance(raw_candidate, dict):
                    raise ValueError(
                        f"{identity}: candidate {candidate_number} "
                        "must be an object"
                    )
                query = raw_candidate.get("query", row_query)
                if not isinstance(query, str) or not query:
                    raise ValueError(
                        f"{identity}: candidate {candidate_number} "
                        "has no query"
                    )
                if isinstance(row_query, str) and query != row_query:
                    raise ValueError(
                        f"{identity}: supplement candidate query differs from row query"
                    )
                if query not in allowed_queries:
                    raise ValueError(
                        f"{identity}: candidate query {query!r} was not requested"
                    )
                candidate = dict(raw_candidate)
                candidate["query"] = query
                candidate["proposal_source_path"] = str(path)
                candidate["proposal_source_sha256"] = source_sha256
                pool[(model, image_sha256, query)].append(candidate)
        source_stats.append(
            {
                "path": str(path),
                "sha256": source_sha256,
                "source_role": "primary" if source_index == 0 else "supplemental",
                "row_count": len(rows),
                "candidate_count": candidate_count,
                "status_counts": dict(sorted(status_counts.items())),
            }
        )

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for key, proposal in proposals.items():
        example_id, model = key
        image_sha256 = str(proposal.get("image_sha256", ""))
        combined = dict(proposal)
        combined["candidates"] = [
            candidate
            for query in _requested_queries(requests[example_id], proposal)
            for candidate in pool.get((model, image_sha256, query), [])
        ]
        merged[key] = combined
    return merged, source_stats


def _summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row["status"]) for row in reviews)
    strategy_counts = Counter(
        str(row.get("grounding_strategy", "missing")) for row in reviews
    )
    model_valid_counts = {
        model: sum(bool(row["models"][model]["valid"]) for row in reviews)
        for model in MODELS
    }
    invalid_reason_counts: Counter[str] = Counter()
    model_reason_counts: dict[str, Counter[str]] = {
        model: Counter() for model in MODELS
    }
    fallback_method_counts: Counter[str] = Counter()
    model_fallback_counts: Counter[str] = Counter()
    fallback_examples: set[str] = set()
    clipped_model_target_count = 0
    clipped_examples: set[str] = set()
    for review in reviews:
        for model in MODELS:
            for reason in review["models"][model]["invalid_reasons"]:
                invalid_reason_counts[reason] += 1
                model_reason_counts[model][reason] += 1
            target = review["models"][model].get("target")
            if isinstance(target, dict) and target.get("fallback_used") is True:
                fallback_method_counts[str(target["selection_method"])] += 1
                model_fallback_counts[model] += 1
                fallback_examples.add(str(review["example_id"]))
            if (
                isinstance(target, dict)
                and target.get("bbox_clipped_to_image") is True
            ):
                clipped_model_target_count += 1
                clipped_examples.add(str(review["example_id"]))
    invalid_examples = [
        {
            "example_id": row["example_id"],
            "grounding_strategy": row.get("grounding_strategy"),
            "invalid_reasons": row["invalid_reasons"],
        }
        for row in reviews
        if row["status"] == "invalid"
    ]
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "strict_example_count": status_counts[STRICT_STATUS],
        "proxy_example_count": status_counts[PROXY_STATUS],
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "model_valid_counts": model_valid_counts,
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "model_invalid_reason_counts": {
            model: dict(sorted(counts.items()))
            for model, counts in model_reason_counts.items()
        },
        "fallback_model_target_count": sum(fallback_method_counts.values()),
        "fallback_example_count": len(fallback_examples),
        "fallback_method_counts": dict(sorted(fallback_method_counts.items())),
        "model_fallback_counts": dict(sorted(model_fallback_counts.items())),
        "bbox_clipped_model_target_count": clipped_model_target_count,
        "bbox_clipped_example_count": len(clipped_examples),
        "invalid_example_count": len(invalid_examples),
        "invalid_examples": invalid_examples,
    }


def build_assumed_grounding_reviews(
    requests_path: str | Path,
    proposals_path: str | Path,
    output_dir: str | Path,
    supplemental_proposals_paths: list[str | Path] | tuple[str | Path, ...] = (),
    expected_count: int = 755,
    iou_threshold: float = 0.8,
    allow_exploratory_fallbacks: bool = False,
) -> Path:
    """Select assumed targets from structurally complete frozen proposals."""
    requests_path = Path(requests_path).resolve()
    proposals_path = Path(proposals_path).resolve()
    supplemental_paths = [
        Path(value).resolve() for value in supplemental_proposals_paths
    ]
    output_dir = Path(output_dir).resolve()
    expected_count = int(expected_count)
    iou_threshold = float(iou_threshold)
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and between 0 and 1")

    request_rows = list(read_jsonl(requests_path))
    proposal_rows = list(read_jsonl(proposals_path))
    requests, proposals = _validate_structure(
        request_rows, proposal_rows, expected_count
    )
    proposal_sources: list[tuple[Path, str, list[dict[str, Any]]]] = [
        (proposals_path, sha256_file(proposals_path), proposal_rows)
    ]
    for path in supplemental_paths:
        proposal_sources.append((path, sha256_file(path), list(read_jsonl(path))))
    proposals, proposal_source_stats = _merge_candidate_sources(
        requests, proposals, proposal_sources
    )

    reviews: list[dict[str, Any]] = []
    image_size_cache: dict[Path, tuple[int, int]] = {}
    for example_id, request in sorted(requests.items()):
        model_reviews = {
            model: _model_review(
                request,
                proposals[(example_id, model)],
                model,
                iou_threshold=iou_threshold,
                allow_exploratory_fallbacks=allow_exploratory_fallbacks,
                image_size_cache=image_size_cache,
            )
            for model in MODELS
        }
        invalid_reasons = {
            model: value["invalid_reasons"]
            for model, value in model_reviews.items()
            if value["invalid_reasons"]
        }
        roles = request.get("roles")
        strategy = (
            roles.get("grounding_strategy")
            if isinstance(roles, dict)
            else None
        )
        fallback_used = any(
            isinstance(value.get("target"), dict)
            and value["target"].get("fallback_used") is True
            for value in model_reviews.values()
        )
        status = (
            "invalid"
            if invalid_reasons
            else PROXY_STATUS
            if fallback_used
            else STRICT_STATUS
        )
        grounding_status = (
            PROXY_GROUNDING_STATUS
            if status == PROXY_STATUS
            else STRICT_GROUNDING_STATUS
        )
        review_id = "assumed-" + object_fingerprint(
            {
                "schema_version": ASSUMED_REVIEW_SCHEMA,
                "example_id": example_id,
                "request_fingerprint": object_fingerprint(request),
                "iou_threshold": iou_threshold,
                "allow_exploratory_fallbacks": allow_exploratory_fallbacks,
                "models": model_reviews,
            }
        )[:24]
        reviews.append(
            {
                "schema_version": ASSUMED_REVIEW_SCHEMA,
                "example_id": example_id,
                "group_id": request.get("group_id"),
                "partition": request.get("partition"),
                "grounding_strategy": strategy,
                "status": status,
                "grounding_resolution": (
                    "invalid"
                    if status == "invalid"
                    else "proxy"
                    if status == PROXY_STATUS
                    else "strict"
                ),
                "review_id": review_id,
                "human_reviewed": False,
                "grounding_status": grounding_status,
                "claim_status": "exploratory",
                "models": model_reviews,
                "invalid_reasons": invalid_reasons,
            }
        )

    reviews_path = output_dir / "reviews_assumed.jsonl"
    write_jsonl(reviews_path, reviews)
    summary = _summary(reviews)
    requests_sha256 = sha256_file(requests_path)
    proposals_sha256 = sha256_file(proposals_path)
    provenance = {
        "schema_version": ASSUMED_REVIEW_SCHEMA,
        "expected_count": expected_count,
        "iou_threshold": iou_threshold,
        "allow_exploratory_fallbacks": allow_exploratory_fallbacks,
        "requests_path": str(requests_path),
        "proposals_path": str(proposals_path),
        "supplemental_proposals_paths": [
            str(path) for path in supplemental_paths
        ],
        "proposal_sources": proposal_source_stats,
        "reviews_path": str(reviews_path),
        "request_count": len(request_rows),
        "proposal_count": len(proposal_rows),
        "supplemental_proposal_count": sum(
            source["row_count"] for source in proposal_source_stats[1:]
        ),
        "review_count": len(reviews),
        "requests_sha256": requests_sha256,
        "proposals_sha256": proposals_sha256,
        "reviews_sha256": sha256_file(reviews_path),
        "source_sha256s": {
            "requests": requests_sha256,
            "primary_proposals": proposals_sha256,
            "supplemental_proposals": [
                source["sha256"] for source in proposal_source_stats[1:]
            ],
        },
        "object_fingerprints": {
            "requests": object_fingerprint(request_rows),
            "proposal_sources": object_fingerprint(proposal_source_stats),
            "reviews": object_fingerprint(reviews),
        },
        "labels_opened": False,
        "human_reviewed": False,
        "grounding_status": (
            "mixed_auto_assumed_and_proxy_unreviewed"
            if summary["proxy_example_count"]
            else STRICT_GROUNDING_STATUS
        ),
        "claim_status": "exploratory",
        **summary,
    }
    manifest = dict(provenance)
    manifest["fingerprint"] = object_fingerprint(manifest)
    write_json(output_dir / "assumed_grounding_manifest.json", manifest)
    # Compatibility aliases for the first public version of this pipeline.
    write_json(output_dir / "assumed_review_manifest.json", manifest)
    audit = {
        **provenance,
        "passed": summary["invalid_example_count"] == 0,
        "all_valid": summary["invalid_example_count"] == 0,
        "structurally_complete": True,
    }
    audit["fingerprint"] = object_fingerprint(audit)
    write_json(output_dir / "assumed_grounding_audit.json", audit)
    write_json(output_dir / "assumed_review_audit.json", audit)
    return reviews_path


def build_assumed_grounding(config: dict[str, Any]) -> Path:
    """Config wrapper for :func:`build_assumed_grounding_reviews`."""
    cfg = section(config, "my_dataset_assumed_grounding")
    required = ("requests_path", "proposals_path", "output_dir")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(
            "my_dataset_assumed_grounding is missing: " + ", ".join(missing)
        )
    supplemental = cfg.get("supplemental_proposals_paths", [])
    if not isinstance(supplemental, list):
        raise ValueError("supplemental_proposals_paths must be a list")
    path = build_assumed_grounding_reviews(
        cfg["requests_path"],
        cfg["proposals_path"],
        cfg["output_dir"],
        supplemental_proposals_paths=supplemental,
        expected_count=int(cfg.get("expected_count", 755)),
        iou_threshold=float(cfg.get("iou_threshold", 0.8)),
        allow_exploratory_fallbacks=bool(
            cfg.get("allow_exploratory_fallbacks", False)
        ),
    )
    if bool(cfg.get("require_all_valid", False)):
        invalid = [
            str(row["example_id"])
            for row in read_jsonl(path)
            if row.get("status") == "invalid"
        ]
        if invalid:
            raise ValueError(
                f"require_all_valid rejected {len(invalid)} invalid assumed "
                f"reviews: {invalid[:10]}"
            )
    return path


__all__ = [
    "MODELS",
    "build_assumed_grounding",
    "build_assumed_grounding_reviews",
]
