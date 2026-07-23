"""Strict JSON, provenance, and resumability helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_dump(value: Any, destination: str | Path) -> None:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def strict_jsonl_append(destination: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if not source.is_file():
        return rows
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            rows.append(value)
    return rows


def file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {"path": str(source), "exists": False}
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(source),
        "exists": True,
        "size": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def model_identity(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).expanduser().resolve()
    config = root / "config.json"
    inventory = []
    if root.is_dir():
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            stat = path.stat()
            inventory.append(
                {
                    "name": str(path.relative_to(root)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "path": str(root),
        "config": file_identity(config),
        "inventory_fingerprint": object_fingerprint(inventory),
        "file_count": len(inventory),
    }


def initialize_manifest(
    destination: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a manifest or verify exact signature agreement for resume."""

    path = Path(destination).expanduser().resolve()
    normalized = dict(payload)
    signature_payload = {
        key: value
        for key, value in normalized.items()
        if key not in {"run_signature", "created_at"}
    }
    normalized["run_signature"] = object_fingerprint(signature_payload)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_signature") != normalized["run_signature"]:
            raise ValueError(
                f"Existing run manifest does not match requested experiment: {path}"
            )
        return existing
    strict_dump(normalized, path)
    return normalized


def assert_unique(values: Iterable[Any], *, what: str) -> None:
    seen: set[Any] = set()
    duplicate: list[Any] = []
    for value in values:
        if value in seen:
            duplicate.append(value)
        seen.add(value)
    if duplicate:
        raise ValueError(f"Duplicate {what}: {duplicate[:10]}")
