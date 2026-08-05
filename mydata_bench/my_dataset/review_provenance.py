"""Cryptographic provenance helpers for audited human-grounding artifacts.

The reviewed pipeline has an explicit trust chain::

    review sources/audit -> attention manifest + JSONL -> ranking manifest
    -> matrix inputs

These helpers deliberately do not apply to the legacy ``auto_unreviewed``
pipeline.  They protect against accidental edits and stale same-ID JSONL files
by binding every consumed file to a fingerprinted parent manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..io import object_fingerprint, sha256_file


REVIEW_PROVENANCE_SCHEMA = "my_dataset.review_provenance.v1"
TRACKING_REVIEW_PROVENANCE_SCHEMA = "my_dataset.tracking_review_provenance.v2"
ATTENTION_MANIFEST_SCHEMA = "my_dataset.attention_input.v1"
TRACKED_REVIEW_SOURCE_KINDS = frozenset(
    {"tracked_grounding_v2", "tracked_grounding_v3"}
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "requests_sha256",
        "proposals_sha256",
        "reviews_sha256",
        "review_audit_sha256",
        "review_audit_fingerprint",
    }
)
_TRACKING_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "requests_sha256",
        "tracking_artifact_sha256",
        "tracking_manifest_sha256",
        "manual_tracking_artifact_sha256",
        "reviews_sha256",
        "review_audit_sha256",
        "review_audit_fingerprint",
    }
)


def _sha256(value: Any, *, identity: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{identity} must be a lowercase SHA-256")
    return digest


def build_review_provenance(
    *,
    requests_sha256: str,
    proposals_sha256: str,
    reviews_sha256: str,
    review_audit_sha256: str,
    review_audit_fingerprint: str,
) -> dict[str, str]:
    """Build and validate the canonical reviewed-source provenance bundle."""
    return validate_review_provenance(
        {
            "schema_version": REVIEW_PROVENANCE_SCHEMA,
            "requests_sha256": requests_sha256,
            "proposals_sha256": proposals_sha256,
            "reviews_sha256": reviews_sha256,
            "review_audit_sha256": review_audit_sha256,
            "review_audit_fingerprint": review_audit_fingerprint,
        },
        identity="review provenance",
    )


def validate_review_provenance(
    value: Any,
    *,
    identity: str,
) -> dict[str, str]:
    """Return a canonical copy or reject an incomplete/malformed bundle."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    if set(value) != _PROVENANCE_FIELDS:
        missing = sorted(_PROVENANCE_FIELDS - set(value))
        extra = sorted(set(value) - _PROVENANCE_FIELDS)
        raise ValueError(
            f"{identity} fields differ from the reviewed contract: "
            f"missing={missing}, extra={extra}"
        )
    if value.get("schema_version") != REVIEW_PROVENANCE_SCHEMA:
        raise ValueError(f"{identity} schema_version is invalid")
    result = {"schema_version": REVIEW_PROVENANCE_SCHEMA}
    for field in sorted(_PROVENANCE_FIELDS - {"schema_version"}):
        result[field] = _sha256(value.get(field), identity=f"{identity}.{field}")
    return result


def build_tracking_review_provenance(
    *,
    requests_sha256: str,
    tracking_artifact_sha256: str,
    tracking_manifest_sha256: str,
    manual_tracking_artifact_sha256: str | None,
    reviews_sha256: str,
    review_audit_sha256: str,
    review_audit_fingerprint: str,
) -> dict[str, str | None]:
    return validate_tracking_review_provenance(
        {
            "schema_version": TRACKING_REVIEW_PROVENANCE_SCHEMA,
            "requests_sha256": requests_sha256,
            "tracking_artifact_sha256": tracking_artifact_sha256,
            "tracking_manifest_sha256": tracking_manifest_sha256,
            "manual_tracking_artifact_sha256": manual_tracking_artifact_sha256,
            "reviews_sha256": reviews_sha256,
            "review_audit_sha256": review_audit_sha256,
            "review_audit_fingerprint": review_audit_fingerprint,
        },
        identity="tracking review provenance",
    )


def validate_tracking_review_provenance(
    value: Any,
    *,
    identity: str,
) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    if set(value) != _TRACKING_PROVENANCE_FIELDS:
        raise ValueError(
            f"{identity} fields differ from the tracked-reviewed contract: "
            f"missing={sorted(_TRACKING_PROVENANCE_FIELDS - set(value))}, "
            f"extra={sorted(set(value) - _TRACKING_PROVENANCE_FIELDS)}"
        )
    if value.get("schema_version") != TRACKING_REVIEW_PROVENANCE_SCHEMA:
        raise ValueError(f"{identity} schema_version is invalid")
    result: dict[str, str | None] = {
        "schema_version": TRACKING_REVIEW_PROVENANCE_SCHEMA
    }
    required = _TRACKING_PROVENANCE_FIELDS - {
        "schema_version",
        "manual_tracking_artifact_sha256",
    }
    for field in sorted(required):
        result[field] = _sha256(
            value.get(field), identity=f"{identity}.{field}"
        )
    manual = value.get("manual_tracking_artifact_sha256")
    result["manual_tracking_artifact_sha256"] = (
        None
        if manual in (None, "")
        else _sha256(
            manual,
            identity=f"{identity}.manual_tracking_artifact_sha256",
        )
    )
    return result


def load_fingerprinted_manifest(
    path: str | Path,
    *,
    identity: str,
) -> dict[str, Any]:
    """Load a JSON object and verify its self-fingerprint."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{identity} is not valid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{identity} must be a JSON object")
    fingerprint_view = dict(value)
    recorded = str(fingerprint_view.pop("fingerprint", ""))
    if not recorded or recorded != object_fingerprint(fingerprint_view):
        raise ValueError(f"{identity} fingerprint is invalid")
    return value


def validate_attention_review_manifest(
    manifest: Mapping[str, Any],
    *,
    identity: str,
) -> dict[str, str | None]:
    """Validate the reviewed source anchors in an attention manifest."""
    if manifest.get("schema_version") != ATTENTION_MANIFEST_SCHEMA:
        raise ValueError(f"{identity} schema_version is invalid")
    if manifest.get("require_review_audit") is not True:
        raise ValueError(f"{identity} must require the strict review audit")
    if manifest.get("accepted_review_statuses") != ["eligible"]:
        raise ValueError(f"{identity} must accept only reviewed eligible rows")
    has_legacy = "review_provenance" in manifest
    has_tracking = "tracking_review_provenance" in manifest
    if has_legacy == has_tracking:
        raise ValueError(
            f"{identity} must contain exactly one of review_provenance or "
            "tracking_review_provenance"
        )
    if has_tracking:
        if manifest.get("review_source_kind") not in TRACKED_REVIEW_SOURCE_KINDS:
            raise ValueError(f"{identity}.review_source_kind is invalid")
        provenance = validate_tracking_review_provenance(
            manifest.get("tracking_review_provenance"),
            identity=f"{identity}.tracking_review_provenance",
        )
        expected = {
            "grounding_requests_sha256": provenance["requests_sha256"],
            "expected_requests_sha256": provenance["requests_sha256"],
            "tracking_artifact_sha256": provenance[
                "tracking_artifact_sha256"
            ],
            "expected_tracking_artifact_sha256": provenance[
                "tracking_artifact_sha256"
            ],
            "tracking_manifest_sha256": provenance[
                "tracking_manifest_sha256"
            ],
            "expected_tracking_manifest_sha256": provenance[
                "tracking_manifest_sha256"
            ],
            "manual_tracking_artifact_sha256": provenance[
                "manual_tracking_artifact_sha256"
            ],
            "expected_manual_tracking_artifact_sha256": provenance[
                "manual_tracking_artifact_sha256"
            ],
            "grounding_reviews_sha256": provenance["reviews_sha256"],
            "review_audit_sha256": provenance["review_audit_sha256"],
            "review_audit_fingerprint": provenance[
                "review_audit_fingerprint"
            ],
            "review_audit_schema_version": (
                "my_dataset.tracked_grounding_review_audit.v2"
            ),
            "target_grounding_scope": "terminal_only",
            "control_region_policy": "none",
        }
        mismatched = {
            field: {"expected": expected_value, "actual": manifest.get(field)}
            for field, expected_value in expected.items()
            if manifest.get(field) != expected_value
        }
        if mismatched:
            raise ValueError(
                f"{identity} tracked-reviewed source anchors differ: {mismatched}"
            )
        if (
            "grounding_proposals_sha256" in manifest
            or "expected_proposals_sha256" in manifest
        ):
            raise ValueError(
                f"{identity} tracked review must not carry proposal SHA fields"
            )
        if not isinstance(manifest.get("artifacts"), Mapping):
            raise ValueError(f"{identity}.artifacts must be a mapping")
        return provenance

    if manifest.get("review_source_kind") not in (None, "legacy_grounding_v1"):
        raise ValueError(f"{identity}.review_source_kind is invalid")
    provenance = validate_review_provenance(
        manifest.get("review_provenance"),
        identity=f"{identity}.review_provenance",
    )
    expected = {
        "grounding_requests_sha256": provenance["requests_sha256"],
        "expected_requests_sha256": provenance["requests_sha256"],
        "grounding_proposals_sha256": provenance["proposals_sha256"],
        "expected_proposals_sha256": provenance["proposals_sha256"],
        "grounding_reviews_sha256": provenance["reviews_sha256"],
        "review_audit_sha256": provenance["review_audit_sha256"],
        "review_audit_fingerprint": provenance["review_audit_fingerprint"],
        "review_audit_schema_version": "my_dataset.grounding_review_audit.v2",
    }
    mismatched = {
        field: {"expected": expected_value, "actual": manifest.get(field)}
        for field, expected_value in expected.items()
        if manifest.get(field) != expected_value
    }
    if mismatched:
        raise ValueError(f"{identity} reviewed source anchors differ: {mismatched}")
    if not isinstance(manifest.get("artifacts"), Mapping):
        raise ValueError(f"{identity}.artifacts must be a mapping")
    return provenance


def validate_jsonl_artifact(
    artifact: Any,
    *,
    actual_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    identity: str,
) -> dict[str, Any]:
    """Bind an already-loaded JSONL file to its parent-manifest record."""
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{identity} artifact record is missing")
    source = Path(actual_path).resolve()
    recorded_path = Path(str(artifact.get("path", ""))).resolve()
    if recorded_path != source:
        raise ValueError(
            f"{identity} path differs from parent manifest: "
            f"recorded={recorded_path}, actual={source}"
        )
    if artifact.get("count") != len(rows):
        raise ValueError(f"{identity} count differs from parent manifest")
    recorded_sha256 = _sha256(
        artifact.get("sha256"), identity=f"{identity}.sha256"
    )
    actual_sha256 = sha256_file(source)
    if recorded_sha256 != actual_sha256:
        raise ValueError(f"{identity} SHA-256 differs from parent manifest")
    recorded_fingerprint = _sha256(
        artifact.get("fingerprint"), identity=f"{identity}.fingerprint"
    )
    actual_fingerprint = object_fingerprint(list(rows))
    if recorded_fingerprint != actual_fingerprint:
        raise ValueError(f"{identity} row fingerprint differs from parent manifest")
    return {
        "path": str(source),
        "count": len(rows),
        "sha256": actual_sha256,
        "fingerprint": actual_fingerprint,
    }
