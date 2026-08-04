"""Shared provenance contracts for attention-grounded experiments.

The exploratory matrix has two deliberately disjoint input regimes.  Legacy
automatic SAM3 decisions are unreviewed, while the reviewed rerun only accepts
audited human decisions.  Keeping the contracts here prevents the attention
builder, ranking cohort, matrix runner and scorer from silently disagreeing
about the meaning of the same fields.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


AUTO_UNREVIEWED = "auto_unreviewed"
HUMAN_REVIEWED = "human_reviewed"

_CONTRACTS: dict[str, dict[str, Any]] = {
    AUTO_UNREVIEWED: {
        "mode": AUTO_UNREVIEWED,
        "status_by_resolution": {
            "strict": "auto_assumed_unreviewed",
            "proxy": "auto_proxy_unreviewed",
        },
        "required_human_reviewed": False,
        "required_claim_status": "exploratory",
        "proxy_resolution": "proxy",
    },
    HUMAN_REVIEWED: {
        "mode": HUMAN_REVIEWED,
        "status_by_resolution": {
            "human_audited": "audited_eligible",
        },
        "required_human_reviewed": True,
        "required_claim_status": "reviewed_exploratory",
        "proxy_resolution": None,
    },
}


def grounding_contract(mode: Any = None) -> dict[str, Any]:
    """Return a normalized, JSON-serializable grounding contract."""
    name = AUTO_UNREVIEWED if mode in (None, "") else str(mode)
    if name not in _CONTRACTS:
        raise ValueError(
            f"grounding_mode must be one of {sorted(_CONTRACTS)}, got {name!r}"
        )
    value = _CONTRACTS[name]
    return {
        **value,
        "status_by_resolution": dict(value["status_by_resolution"]),
    }


def infer_grounding_mode(row: Mapping[str, Any], *, identity: str) -> str:
    """Infer the contract for legacy records which predate ``grounding_mode``."""
    declared = row.get("grounding_mode")
    if declared not in (None, ""):
        name = str(declared)
        grounding_contract(name)
        return name
    if row.get("human_reviewed") is True or row.get("grounding_status") == "audited_eligible":
        return HUMAN_REVIEWED
    return AUTO_UNREVIEWED


def validate_grounding_row(
    row: Mapping[str, Any],
    *,
    identity: str,
    contract: Mapping[str, Any],
) -> None:
    """Validate one manifest/record against an explicit provenance contract."""
    resolution = row.get("grounding_resolution")
    status = row.get("grounding_status")
    status_by_resolution = contract["status_by_resolution"]
    expected_status = (
        status_by_resolution.get(resolution)
        if isinstance(resolution, str)
        else None
    )
    if expected_status is None or status != expected_status:
        allowed = ", ".join(
            f"{candidate_resolution}/{candidate_status}"
            for candidate_resolution, candidate_status in status_by_resolution.items()
        )
        raise ValueError(
            f"{identity}: grounding_resolution/grounding_status must be one of "
            f"{allowed}; found {resolution!r}/{status!r}"
        )

    required_reviewed = bool(contract["required_human_reviewed"])
    if row.get("human_reviewed") is not required_reviewed:
        expected = str(required_reviewed).lower()
        raise ValueError(f"{identity}: human_reviewed must be {expected}")
    required_claim = str(contract["required_claim_status"])
    if row.get("claim_status") != required_claim:
        raise ValueError(
            f"{identity}: claim_status must be {required_claim!r}"
        )
    declared_mode = row.get("grounding_mode")
    if declared_mode not in (None, contract["mode"]):
        raise ValueError(
            f"{identity}: grounding_mode {declared_mode!r} differs from "
            f"configured {contract['mode']!r}"
        )

    proxy_resolution = contract.get("proxy_resolution")
    if proxy_resolution is not None and resolution == proxy_resolution:
        selection = row.get("grounding_selection")
        if not isinstance(selection, Mapping):
            raise ValueError(
                f"{identity}: proxy grounding_selection must be a structured mapping"
            )
        if selection.get("fallback_used") is not True:
            raise ValueError(
                f"{identity}: proxy grounding_selection.fallback_used must be true"
            )


def grounding_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only provenance fields which may be absent in older artifacts."""
    return {
        field: row[field]
        for field in (
            "grounding_mode",
            "grounding_status",
            "grounding_resolution",
            "grounding_selection",
            "human_reviewed",
            "claim_status",
        )
        if field in row
    }


def aggregate_grounding_value(
    rows: Sequence[Mapping[str, Any]], field: str
) -> str:
    values = {str(row[field]) for row in rows}
    return next(iter(values)) if len(values) == 1 else "mixed"


def grounding_composition(
    rows: Sequence[Mapping[str, Any]], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize resolutions while preserving the legacy auto JSON shape."""
    counts = Counter(str(row.get("grounding_resolution", "")) for row in rows)
    total = len(rows)
    if contract["mode"] == AUTO_UNREVIEWED:
        proxy_count = int(counts["proxy"])
        return {
            "strict_count": int(counts["strict"]),
            "proxy_count": proxy_count,
            "total": total,
            "proxy_ratio": proxy_count / total if total else 0.0,
        }
    audited_count = int(counts["human_audited"])
    return {
        "human_audited_count": audited_count,
        "total": total,
        "human_audited_ratio": audited_count / total if total else 0.0,
    }
