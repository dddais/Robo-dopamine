"""Freeze and verify complete-content manifests for local model checkpoints.

The generic :func:`mydata_bench.io.artifact_fingerprint` intentionally avoids
reading multi-gigabyte weight shards.  That trade-off is appropriate for most
diagnostics, but it is too weak for a resumable reviewed experiment: replacing
a shard with different bytes of the same size would otherwise leave the run
fingerprint unchanged.  This module provides the stricter, opt-in contract used
by the human-reviewed matrices.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from ..io import object_fingerprint, sha256_file, write_json


CHECKPOINT_CONTENT_MANIFEST_SCHEMA = "my_dataset.checkpoint_content_manifest.v1"
CHECKPOINT_CONTENT_VERIFICATION_SCHEMA = (
    "my_dataset.checkpoint_content_verification.v1"
)
_ENTRY_FIELDS = frozenset({"path", "size", "sha256"})
_EXCLUDED_PATH_PREFIXES = (".cache/",)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "model_path",
        "path_kind",
        "hash_algorithm",
        "file_scope",
        "excluded_path_prefixes",
        "file_count",
        "total_size_bytes",
        "files",
        "content_fingerprint",
        "fingerprint",
    }
)


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _relative_path(value: Any) -> str:
    text = str(value)
    candidate = PurePosixPath(text)
    if (
        not text
        or candidate.is_absolute()
        or text != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"Invalid checkpoint manifest relative path: {text!r}")
    return text


def _checkpoint_files(root: Path) -> list[tuple[str, Path]]:
    """Return every regular file below ``root`` in stable relative-path order."""
    if not root.is_dir():
        raise ValueError(f"Checkpoint model_path must be a directory: {root}")
    values: list[tuple[str, Path]] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        # Hugging Face may create downloader metadata and transient ``.lock``
        # files below a local model directory. They are not consumed by local
        # ``from_pretrained`` and must not make a frozen checkpoint identity
        # depend on unrelated cache housekeeping. Every checkpoint payload
        # file outside this one explicit volatile prefix remains in scope.
        if any(
            relative == prefix.rstrip("/") or relative.startswith(prefix)
            for prefix in _EXCLUDED_PATH_PREFIXES
        ):
            continue
        if candidate.is_symlink():
            raise ValueError(
                f"Checkpoint directories must not contain symlinks: {relative}"
            )
        if candidate.is_dir():
            continue
        try:
            mode = candidate.stat(follow_symlinks=False).st_mode
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Checkpoint changed while enumerating files: {relative}"
            ) from exc
        if not stat.S_ISREG(mode):
            raise ValueError(
                f"Checkpoint contains a non-regular filesystem entry: {relative}"
            )
        values.append((_relative_path(relative), candidate))
    values.sort(key=lambda item: item[0])
    if not values:
        raise ValueError(f"Checkpoint directory contains no regular files: {root}")
    return values


def _sha256_stable_file(path: Path, *, chunk_size: int = 8 << 20) -> tuple[int, str]:
    """Hash one regular file and reject an in-place mutation during the read."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Checkpoint entry is not a regular file: {path}")
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"Checkpoint file changed while hashing: {path}")
    return int(after.st_size), digest.hexdigest()


def _entries(root: Path) -> list[dict[str, Any]]:
    result = []
    for relative, path in _checkpoint_files(root):
        size, digest = _sha256_stable_file(path)
        result.append({"path": relative, "size": size, "sha256": digest})
    return result


def _payload(model_path: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_CONTENT_MANIFEST_SCHEMA,
        "model_path": str(model_path),
        "path_kind": "directory",
        "hash_algorithm": "sha256",
        "file_scope": "all_regular_checkpoint_files_except_declared_volatile_cache",
        "excluded_path_prefixes": list(_EXCLUDED_PATH_PREFIXES),
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size"]) for entry in entries),
        "files": entries,
        "content_fingerprint": object_fingerprint(entries),
    }


def freeze_checkpoint_content_manifest(
    model_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Hash every checkpoint file and write a deterministic frozen manifest."""
    root = Path(model_path).resolve()
    destination = Path(output_path).resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Checkpoint content manifest must be stored outside the checkpoint directory"
        )
    payload = _payload(root, _entries(root))
    payload["fingerprint"] = object_fingerprint(payload)
    write_json(destination, payload)
    return destination


def _validated_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Checkpoint manifest files must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _ENTRY_FIELDS:
            raise ValueError(
                f"Checkpoint manifest file entry {index} has an invalid schema"
            )
        relative = _relative_path(raw["path"])
        if relative in seen:
            raise ValueError(f"Checkpoint manifest duplicates file path: {relative}")
        seen.add(relative)
        size = raw["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"Checkpoint manifest has invalid file size for {relative}: {size!r}"
            )
        digest = str(raw["sha256"])
        if not _is_sha256(digest):
            raise ValueError(
                f"Checkpoint manifest has invalid SHA-256 for {relative}: {digest!r}"
            )
        result.append({"path": relative, "size": size, "sha256": digest})
    if result != sorted(result, key=lambda entry: entry["path"]):
        raise ValueError("Checkpoint manifest files must be sorted by relative path")
    return result


def load_checkpoint_content_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest and validate its complete internal fingerprint contract."""
    source = Path(path).resolve()
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint content manifest JSON: {source}") from exc
    if not isinstance(loaded, dict) or set(loaded) != _MANIFEST_FIELDS:
        raise ValueError("Checkpoint content manifest has an invalid top-level schema")
    if loaded.get("schema_version") != CHECKPOINT_CONTENT_MANIFEST_SCHEMA:
        raise ValueError("Checkpoint content manifest schema_version mismatch")
    if loaded.get("path_kind") != "directory":
        raise ValueError("Checkpoint content manifest path_kind must be 'directory'")
    if loaded.get("hash_algorithm") != "sha256":
        raise ValueError("Checkpoint content manifest hash_algorithm must be 'sha256'")
    if (
        loaded.get("file_scope")
        != "all_regular_checkpoint_files_except_declared_volatile_cache"
        or loaded.get("excluded_path_prefixes") != list(_EXCLUDED_PATH_PREFIXES)
    ):
        raise ValueError("Checkpoint content manifest file-scope policy mismatch")
    model_path = str(loaded.get("model_path", "")).strip()
    if not model_path or Path(model_path) != Path(model_path).resolve():
        raise ValueError("Checkpoint content manifest model_path must be absolute")
    entries = _validated_entries(loaded.get("files"))
    if loaded.get("file_count") != len(entries):
        raise ValueError("Checkpoint content manifest file_count mismatch")
    if loaded.get("total_size_bytes") != sum(entry["size"] for entry in entries):
        raise ValueError("Checkpoint content manifest total_size_bytes mismatch")
    if loaded.get("content_fingerprint") != object_fingerprint(entries):
        raise ValueError("Checkpoint content manifest content_fingerprint mismatch")
    fingerprint_payload = dict(loaded)
    recorded_fingerprint = fingerprint_payload.pop("fingerprint")
    if not _is_sha256(recorded_fingerprint) or recorded_fingerprint != object_fingerprint(
        fingerprint_payload
    ):
        raise ValueError("Checkpoint content manifest fingerprint mismatch")
    result = dict(loaded)
    result["files"] = entries
    return result


def _paths(values: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(value["path"]) for value in values}


def verify_checkpoint_content_manifest(
    model_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify the exact file set, sizes and SHA-256 values before model loading."""
    root = Path(model_path).resolve()
    source = Path(manifest_path).resolve()
    manifest = load_checkpoint_content_manifest(source)
    if Path(str(manifest["model_path"])).resolve() != root:
        raise ValueError(
            "Checkpoint model_path differs from the frozen content manifest: "
            f"configured={root}, manifest={manifest['model_path']}"
        )
    actual_files = _checkpoint_files(root)
    actual_by_path = {relative: path for relative, path in actual_files}
    expected_by_path = {str(entry["path"]): entry for entry in manifest["files"]}
    missing = sorted(_paths(manifest["files"]) - set(actual_by_path))
    extra = sorted(set(actual_by_path) - _paths(manifest["files"]))
    if missing or extra:
        raise ValueError(
            "Checkpoint file set differs from the frozen content manifest: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    for relative in sorted(expected_by_path):
        expected = expected_by_path[relative]
        path = actual_by_path[relative]
        observed_size = path.stat(follow_symlinks=False).st_size
        if observed_size != expected["size"]:
            raise ValueError(
                f"Checkpoint file size mismatch for {relative}: "
                f"expected={expected['size']}, actual={observed_size}"
            )
        stable_size, observed_digest = _sha256_stable_file(path)
        if stable_size != expected["size"] or observed_digest != expected["sha256"]:
            raise ValueError(
                f"Checkpoint file SHA-256 mismatch for {relative}: "
                f"expected={expected['sha256']}, actual={observed_digest}"
            )
    return {
        "schema_version": CHECKPOINT_CONTENT_VERIFICATION_SCHEMA,
        "passed": True,
        "model_path": str(root),
        "manifest_path": str(source),
        "manifest_sha256": sha256_file(source),
        "manifest_fingerprint": manifest["fingerprint"],
        "content_fingerprint": manifest["content_fingerprint"],
        "file_count": manifest["file_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "hash_algorithm": "sha256",
        "all_checkpoint_file_bytes_hashed": True,
    }
