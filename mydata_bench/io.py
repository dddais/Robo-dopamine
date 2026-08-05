from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def object_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: str | Path) -> str:
    """Fingerprint a file or model directory without rereading multi-GB weights.

    Small configuration/tokenizer files are content-hashed. Large shards are
    represented by relative path and byte size; this catches local checkpoint
    swaps while keeping every run startup practical.
    """
    path = Path(path).resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        return object_fingerprint({"missing": str(path)})
    entries = []
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = str(item.relative_to(path))
        size = item.stat().st_size
        entry = {"path": relative, "size": size}
        if size <= 10 * 1024 * 1024:
            entry["sha256"] = sha256_file(item)
        entries.append(entry)
    return object_fingerprint(entries)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{number}")
            yield value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def latest_by_id(
    rows: Iterable[dict[str, Any]], key: str = "example_id"
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row[key])] = row
    return latest


def stable_shard(video_sha256: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return int(video_sha256[:16], 16) % num_shards


def provenance(command: list[str], config: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": config,
        "config_fingerprint": object_fingerprint(config),
        "git_revision": revision,
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu,
    }


def deterministic_merge(paths: Iterable[str | Path], destination: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if Path(path).exists():
            rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: (str(row.get("example_id", "")), int(row.get("attempt", 0))))
    write_jsonl(destination, rows)
