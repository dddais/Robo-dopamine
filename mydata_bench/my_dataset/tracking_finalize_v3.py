"""Reconcile sharded tracking provenance and freeze the final run manifest.

This module deliberately lives outside tracked_grounding.py. Tracking
artifacts bind that orchestrator file by SHA, so changing it after a long run
would make otherwise valid artifacts unverifiable.
"""

from __future__ import annotations

import copy
import gc
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..io import append_jsonl, object_fingerprint, read_jsonl, sha256_file, write_json
from . import tracked_grounding_v3 as tg


_TRACK_VOLATILE_PROVENANCE_KEYS = {
    "cache_hit", "cache_key", "cache_path",
    "source_video_path", "source_video_sha256",
}


def _paths(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    cfg, sam3_cfg = tg._cfg(config)
    return cfg, sam3_cfg, Path(str(cfg["output_dir"])).expanduser().resolve()


def _requests_and_latest(
    output_dir: Path,
) -> tuple[Path, Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    requests_path = output_dir / "requests.jsonl"
    tracks_path = output_dir / "tracks.jsonl"
    if not requests_path.is_file() or not tracks_path.is_file():
        raise FileNotFoundError("requests.jsonl and tracks.jsonl must both exist")
    requests = list(read_jsonl(requests_path))
    if not requests:
        raise ValueError("Tracked-grounding requests are empty")
    validation_hashes: dict[str, str] = {}
    for request in requests:
        tg._validate_frozen_request(request, validation_hashes)
    return requests_path, tracks_path, requests, tg._latest_artifacts(tracks_path)


def _current_orchestrator_sha() -> str:
    return sha256_file(Path(tg.__file__).resolve())


def _artifact_is_stale(artifact: Mapping[str, Any], current_sha: str) -> bool:
    proposal = artifact.get("proposal")
    provider = proposal.get("provider_provenance") if isinstance(proposal, Mapping) else None
    if not isinstance(provider, Mapping):
        return True
    if provider.get("orchestrator_source_sha256") != current_sha:
        return True
    for track in artifact.get("candidate_tracks", []):
        provenance = track.get("predictor_provenance") if isinstance(track, Mapping) else None
        if not isinstance(provenance, Mapping):
            return True
        if provenance.get("orchestrator_source_sha256") != current_sha:
            return True
    return False


def _release_provider(provider: Any) -> None:
    if provider is not None and callable(getattr(provider, "shutdown", None)):
        provider.shutdown()
    gc.collect()
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None) if torch_module else None
    if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
        cuda.empty_cache()


def reconcile_stale_tracking_provenance(config: dict[str, Any]) -> Path:
    """Append fresh attempts for artifacts bound to an older orchestrator SHA."""
    _, sam3_cfg, output_dir = _paths(config)
    _, tracks_path, requests, latest = _requests_and_latest(output_dir)
    request_by_id = {str(row["example_id"]): row for row in requests}
    if set(latest) != set(request_by_id):
        raise ValueError("Cannot reconcile provenance before tracking coverage is complete")
    current_sha = _current_orchestrator_sha()
    stale_ids = sorted(
        example_id for example_id, artifact in latest.items()
        if _artifact_is_stale(artifact, current_sha)
    )
    if not stale_ids:
        return tracks_path

    provider = tg._candidate_provider_from_config(sam3_cfg)
    provider_provenance = tg._candidate_provider_provenance(provider, sam3_cfg)
    if provider_provenance.get("orchestrator_source_sha256") != current_sha:
        raise RuntimeError("Fresh proposer provenance does not bind the current orchestrator")
    prepared: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    try:
        for example_id in stale_ids:
            request = request_by_id[example_id]
            old = latest[example_id]
            try:
                proposal = tg._propose(
                    request, provider, provider_provenance, output_dir, sam3_cfg
                )
            except Exception as exc:
                proposal = tg._invalid_proposal(request, exc, provider_provenance)
            prepared.append((request, int(old.get("attempt", 0)) + 1, proposal))
    finally:
        _release_provider(provider)

    predictor: Any | None = None
    predictor_provenance: dict[str, Any] | None = None
    if any(proposal.get("options") for _, _, proposal in prepared):
        predictor, predictor_provenance = tg._predictor_from_config(sam3_cfg)
        if predictor_provenance.get("orchestrator_source_sha256") != current_sha:
            raise RuntimeError("Fresh tracker provenance does not bind the current orchestrator")
    try:
        for request, attempt, proposal in prepared:
            artifact = tg._automated_artifact(
                request, proposal, attempt, predictor, predictor_provenance,
                None, output_dir, sam3_cfg,
            )
            append_jsonl(tracks_path, artifact)
    finally:
        if predictor is not None and callable(getattr(predictor, "shutdown", None)):
            predictor.shutdown()
    return tracks_path


def _verified_provider(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Artifact lacks proposal provider provenance")
    result = copy.deepcopy(dict(value))
    fingerprint = result.pop("proposer_fingerprint", None)
    if not fingerprint or fingerprint != object_fingerprint(result):
        raise ValueError("Proposal provider provenance fingerprint mismatch")
    result["proposer_fingerprint"] = fingerprint
    return result


def _verified_tracker_run(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Candidate track lacks predictor provenance")
    run = copy.deepcopy(dict(value))
    for key in _TRACK_VOLATILE_PROVENANCE_KEYS:
        run.pop(key, None)
    fingerprint = run.pop("tracker_fingerprint", None)
    if not fingerprint or fingerprint != object_fingerprint(run):
        raise ValueError("Tracker provenance fingerprint mismatch")
    run["tracker_fingerprint"] = fingerprint
    return run


def finalize_sharded_tracking_manifest(config: dict[str, Any]) -> Path:
    """Fail closed unless latest artifacts form one complete, coherent run."""
    cfg, _, output_dir = _paths(config)
    requests_path, tracks_path, requests, latest = _requests_and_latest(output_dir)
    request_by_id = {str(row["example_id"]): row for row in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("Duplicate example_id in requests.jsonl")
    if set(latest) != set(request_by_id):
        raise ValueError("Tracking artifact coverage differs from frozen requests")

    providers: dict[str, dict[str, Any]] = {}
    tracker_runs: dict[str, dict[str, Any]] = {}
    for example_id, artifact in latest.items():
        if artifact.get("schema_version") != tg.TRACKED_GROUNDING_ARTIFACT_SCHEMA:
            raise ValueError(f"{example_id}: tracking artifact schema mismatch")
        if artifact.get("fingerprint") != tg._fingerprint_row(artifact):
            raise ValueError(f"{example_id}: tracking artifact fingerprint mismatch")
        if artifact.get("request_fingerprint") != request_by_id[example_id].get("request_fingerprint"):
            raise ValueError(f"{example_id}: request fingerprint mismatch")
        proposal = artifact.get("proposal")
        provider = _verified_provider(
            proposal.get("provider_provenance") if isinstance(proposal, Mapping) else None
        )
        providers[object_fingerprint(provider)] = provider
        for track in artifact.get("candidate_tracks", []):
            run = _verified_tracker_run(track.get("predictor_provenance"))
            tracker_runs[object_fingerprint(run)] = run
    if len(providers) != 1:
        raise ValueError(f"Latest artifacts contain {len(providers)} proposer provenances")
    if len(tracker_runs) != 1:
        raise ValueError(f"Latest artifacts contain {len(tracker_runs)} tracker provenances")
    provider = next(iter(providers.values()))
    tracker_run = next(iter(tracker_runs.values()))
    current_sha = _current_orchestrator_sha()
    if provider.get("orchestrator_source_sha256") != current_sha:
        raise ValueError("Proposer provenance is stale relative to tracked_grounding.py")
    if tracker_run.get("orchestrator_source_sha256") != current_sha:
        raise ValueError("Tracker provenance is stale relative to tracked_grounding.py")

    manifest_path = output_dir / "manifest.json"
    manifest = tg._read_manifest(manifest_path)
    if not manifest:
        inputs_path = tg._resolved_file(cfg["inputs_path"], "model input manifest")
        roles_path = tg._resolved_file(cfg["roles_path"], "semantic role manifest")
        split_path = tg._resolved_file(cfg["split_path"], "frozen whitebox split")
        manifest = tg._manifest_base(cfg, inputs_path, roles_path, split_path)
    counts = Counter(str(row.get("status")) for row in latest.values())
    manifest.update({
        "status": "complete",
        "requests_path": str(requests_path),
        "requests_sha256": sha256_file(requests_path),
        "request_count": len(requests),
        "tracks_path": str(tracks_path),
        "tracks_sha256": sha256_file(tracks_path),
        "artifact_count": len(latest),
        "artifact_row_count": sum(1 for _ in read_jsonl(tracks_path)),
        "status_counts": dict(sorted(counts.items())),
        "coverage_complete": True,
        "request_bindings_current": True,
        "proposal_backend": provider,
        "tracker": {
            "backend": tracker_run["backend"],
            "official_source_path": tracker_run["official_source_path"],
            "checkpoint_path": tracker_run["checkpoint_path"],
            "tracker_fingerprint": tracker_run["tracker_fingerprint"],
        },
        "labels_opened": False,
    })
    for key in (
        "proposal_backend_error", "proposal_backend_shutdown_error",
        "tracker_shutdown_error",
    ):
        manifest.pop(key, None)
    write_json(manifest_path, manifest)
    return manifest_path


__all__ = [
    "finalize_sharded_tracking_manifest",
    "reconcile_stale_tracking_provenance",
]
