"""Build an explicitly unreviewed 755-example policy from frozen tracking."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..config import section
from ..io import object_fingerprint, read_jsonl, sha256_file, write_json, write_jsonl
from .assumed_grounding import ASSUMED_REVIEW_SCHEMA, MODELS


def _file(cfg: Mapping[str, Any], key: str) -> Path:
    value = cfg.get(key)
    if value in (None, ""):
        raise ValueError(f"my_dataset_tracking_assumed requires {key}")
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _frozen_sha(cfg: Mapping[str, Any], key: str, path: Path) -> str:
    expected = str(cfg.get(key, "")).lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(f"{key} must be a frozen lowercase SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path} SHA-256 differs from {key}: {actual} != {expected}")
    return actual


def _unique(path: Path, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(read_jsonl(path), 1):
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in result:
            raise ValueError(f"{path}:{number}: invalid/duplicate {name} example_id")
        result[example_id] = row
    return result


def _latest_tracks(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    result: dict[str, dict[str, Any]] = {}
    row_count = 0
    for number, row in enumerate(read_jsonl(path), 1):
        row_count += 1
        example_id = str(row.get("example_id", ""))
        payload = dict(row)
        fingerprint = str(payload.pop("fingerprint", ""))
        if not example_id or fingerprint != object_fingerprint(payload):
            raise ValueError(f"{path}:{number}: invalid track identity/fingerprint")
        result[example_id] = row
    return result, row_count


def _valid_tracks(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = artifact.get("candidate_tracks")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("candidate_id")): row
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "ok"
        and str(row.get("candidate_id", ""))
    }


def _select(artifact: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    status = str(artifact.get("status", ""))
    valid = _valid_tracks(artifact)
    if status == "ok":
        candidate_id = str(artifact.get("selected_candidate_id", ""))
        if artifact.get("selection_source") != "algorithmic_default":
            raise ValueError("status=ok is not bound to algorithmic_default")
        if candidate_id not in valid:
            raise ValueError("status=ok selected track is missing/invalid")
        return valid[candidate_id], "algorithmic_default_tracked"
    if status == "needs_review":
        proposal = artifact.get("proposal")
        options = proposal.get("options") if isinstance(proposal, Mapping) else None
        if not isinstance(options, list):
            raise ValueError("needs_review proposal options are missing")
        for option in options:
            candidate_id = (
                str(option.get("candidate_id", ""))
                if isinstance(option, Mapping)
                else ""
            )
            if candidate_id in valid:
                return valid[candidate_id], "first_valid_proposal_option_unreviewed"
        raise ValueError("needs_review artifact has no valid option track")
    if status == "invalid":
        return None, "legacy_assumed_terminal_for_invalid_tracking"
    raise ValueError(f"unsupported latest tracking status: {status!r}")


def _candidate(artifact: Mapping[str, Any], candidate_id: str) -> tuple[float, str]:
    proposal = artifact.get("proposal")
    options = proposal.get("options") if isinstance(proposal, Mapping) else None
    if not isinstance(options, list):
        raise ValueError("tracked proposal options are missing")
    for option in options:
        if (
            isinstance(option, Mapping)
            and str(option.get("candidate_id", "")) == candidate_id
        ):
            score, query = option.get("score"), option.get("query")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{candidate_id}: invalid score")
            if not isinstance(query, str) or not query:
                raise ValueError(f"{candidate_id}: invalid query")
            return float(score), query
    raise ValueError(f"{candidate_id}: proposal metadata is missing")


def _tracked_models(
    fallback: Mapping[str, Any],
    artifact: Mapping[str, Any],
    selected: Mapping[str, Any],
    policy: str,
    tracks_path: Path,
    tracks_sha: str,
) -> dict[str, dict[str, Any]]:
    fallback_models = fallback.get("models")
    terminals = selected.get("terminal_by_model")
    candidate_id = str(selected.get("candidate_id", ""))
    if not isinstance(fallback_models, Mapping) or not isinstance(terminals, Mapping):
        raise ValueError("tracked/fallback model mappings are incomplete")
    score, query = _candidate(artifact, candidate_id)
    result: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        fallback_model = fallback_models.get(model)
        terminal = terminals.get(model)
        if not isinstance(fallback_model, Mapping) or not isinstance(terminal, Mapping):
            raise ValueError(f"{model}: tracked/fallback terminal is missing")
        target = copy.deepcopy(fallback_model.get("target"))
        bbox = terminal.get("bbox_xyxy")
        if not isinstance(target, dict) or not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{model}: tracked terminal bbox is invalid")
        if terminal.get("visible") is not True:
            raise ValueError(f"{model}: tracked terminal is not visible")
        if str(target.get("image_sha256", "")) != str(terminal.get("image_sha256", "")):
            raise ValueError(f"{model}: tracked and frozen terminal images differ")
        target.update(
            {
                "bbox": [float(value) for value in bbox],
                "bbox_original": [float(value) for value in bbox],
                "bbox_clipped_to_image": False,
                "score": score,
                "query": query,
                "selection_method": policy,
                "fallback_used": policy != "algorithmic_default_tracked",
                "proposal_source_path": str(tracks_path),
                "proposal_source_sha256": tracks_sha,
                "selected_candidate_id": candidate_id,
                "track_fingerprint": selected.get("fingerprint"),
            }
        )
        result[model] = {
            "proposal_image_path": target["image_path"],
            "target": target,
            "wrong_region": None,
            "valid": True,
            "invalid_reasons": [],
        }
    return result


def _fallback_models(
    fallback: Mapping[str, Any], policy: str
) -> dict[str, dict[str, Any]]:
    raw = fallback.get("models")
    if not isinstance(raw, Mapping):
        raise ValueError("fallback review has no models mapping")
    result: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_row = copy.deepcopy(raw.get(model))
        target = model_row.get("target") if isinstance(model_row, dict) else None
        if not isinstance(target, dict) or model_row.get("valid") is not True:
            raise ValueError(f"{model}: legacy fallback target is invalid")
        target["fallback_used"] = True
        target["selection_method"] = policy
        model_row["invalid_reasons"] = []
        model_row["wrong_region"] = None
        result[model] = model_row
    return result


def build_tracking_assumed(config: dict[str, Any]) -> Path:
    cfg = section(config, "my_dataset_tracking_assumed")
    requests_path = _file(cfg, "tracking_requests_path")
    tracks_path = _file(cfg, "tracking_artifact_path")
    tracking_manifest_path = _file(cfg, "tracking_manifest_path")
    fallback_path = _file(cfg, "fallback_reviews_path")
    output_dir = Path(str(cfg["output_dir"])).resolve()
    expected_count = int(cfg.get("expected_count", 755))

    requests_sha = _frozen_sha(
        cfg, "expected_tracking_requests_sha256", requests_path
    )
    tracks_sha = _frozen_sha(
        cfg, "expected_tracking_artifact_sha256", tracks_path
    )
    manifest_sha = _frozen_sha(
        cfg, "expected_tracking_manifest_sha256", tracking_manifest_path
    )
    fallback_sha = _frozen_sha(
        cfg, "expected_fallback_reviews_sha256", fallback_path
    )
    tracking_manifest = json.loads(
        tracking_manifest_path.read_text(encoding="utf-8")
    )
    if (
        tracking_manifest.get("status") != "complete"
        or tracking_manifest.get("coverage_complete") is not True
        or tracking_manifest.get("labels_opened") is not False
        or tracking_manifest.get("requests_sha256") != requests_sha
        or tracking_manifest.get("tracks_sha256") != tracks_sha
    ):
        raise ValueError("tracking manifest is not the frozen complete artifact")

    requests = _unique(requests_path, "tracking request")
    fallbacks = _unique(fallback_path, "fallback review")
    tracks, track_row_count = _latest_tracks(tracks_path)
    expected_ids = set(requests)
    for name, rows in (("latest tracks", tracks), ("fallback reviews", fallbacks)):
        if set(rows) != expected_ids:
            raise ValueError(f"{name} IDs differ from tracking requests")
    if len(expected_ids) != expected_count:
        raise ValueError(f"expected {expected_count} examples, found {len(expected_ids)}")

    reviews: list[dict[str, Any]] = []
    policies: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for example_id in sorted(expected_ids):
        request, artifact, fallback = (
            requests[example_id],
            tracks[example_id],
            fallbacks[example_id],
        )
        for field in ("group_id", "partition"):
            if str(request.get(field, "")) != str(fallback.get(field, "")):
                raise ValueError(f"{example_id}: fallback {field} differs")
        selected, policy = _select(artifact)
        source_status = str(artifact["status"])
        statuses[source_status] += 1
        policies[policy] += 1
        models = (
            _fallback_models(fallback, policy)
            if selected is None
            else _tracked_models(
                fallback, artifact, selected, policy, tracks_path, tracks_sha
            )
        )
        strict = policy == "algorithmic_default_tracked"
        review = {
            "schema_version": ASSUMED_REVIEW_SCHEMA,
            "example_id": example_id,
            "group_id": request["group_id"],
            "partition": request["partition"],
            "grounding_strategy": request.get("roles", {}).get(
                "grounding_strategy"
            ),
            "status": "assumed_valid" if strict else "assumed_proxy",
            "grounding_resolution": "strict" if strict else "proxy",
            "human_reviewed": False,
            "grounding_status": (
                "auto_assumed_unreviewed"
                if strict
                else "auto_proxy_unreviewed"
            ),
            "claim_status": "exploratory",
            "models": models,
            "invalid_reasons": {},
            "tracking_assumption": {
                "source_tracking_status": source_status,
                "selection_policy": policy,
                "selected_candidate_id": (
                    selected.get("candidate_id") if selected else None
                ),
                "tracking_artifact_sha256": tracks_sha,
                "fallback_reviews_sha256": (
                    fallback_sha if selected is None else None
                ),
            },
        }
        review["review_id"] = "tracking-assumed-" + object_fingerprint(review)[:24]
        reviews.append(review)

    output_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = output_dir / "reviews_assumed.jsonl"
    write_jsonl(reviews_path, reviews)
    manifest = {
        "schema_version": "my_dataset.tracking_assumed_manifest.v1",
        "claim_status": "exploratory",
        "human_reviewed": False,
        "expected_count": expected_count,
        "review_count": len(reviews),
        "tracking_source_status_counts": dict(sorted(statuses.items())),
        "selection_policy_counts": dict(sorted(policies.items())),
        "tracking_requests_path": str(requests_path),
        "tracking_requests_sha256": requests_sha,
        "tracking_artifact_path": str(tracks_path),
        "tracking_artifact_sha256": tracks_sha,
        "tracking_artifact_row_count": track_row_count,
        "tracking_manifest_path": str(tracking_manifest_path),
        "tracking_manifest_sha256": manifest_sha,
        "fallback_reviews_path": str(fallback_path),
        "fallback_reviews_sha256": fallback_sha,
        "reviews_path": str(reviews_path),
        "reviews_sha256": sha256_file(reviews_path),
        "labels_opened": False,
    }
    manifest["fingerprint"] = object_fingerprint(manifest)
    destination = output_dir / "tracking_assumed_manifest.json"
    write_json(destination, manifest)
    return destination
