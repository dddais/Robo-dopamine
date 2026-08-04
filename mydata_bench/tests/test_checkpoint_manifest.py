from __future__ import annotations

import json
from pathlib import Path

import pytest

from mydata_bench.io import object_fingerprint, write_json
from mydata_bench.my_dataset.checkpoint_manifest import (
    CHECKPOINT_CONTENT_MANIFEST_SCHEMA,
    freeze_checkpoint_content_manifest,
    load_checkpoint_content_manifest,
    verify_checkpoint_content_manifest,
)


def _checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"A" * 4096)
    manifest = tmp_path / "checkpoint_manifest.json"
    freeze_checkpoint_content_manifest(model, manifest)
    return model, manifest


def test_checkpoint_content_manifest_hashes_every_file_and_verifies(
    tmp_path: Path,
) -> None:
    model, manifest_path = _checkpoint(tmp_path)
    manifest = load_checkpoint_content_manifest(manifest_path)
    assert manifest["schema_version"] == CHECKPOINT_CONTENT_MANIFEST_SCHEMA
    assert manifest["file_count"] == 2
    assert [row["path"] for row in manifest["files"]] == [
        "config.json",
        "model-00001-of-00001.safetensors",
    ]
    result = verify_checkpoint_content_manifest(model, manifest_path)
    assert result["passed"] is True
    assert result["all_checkpoint_file_bytes_hashed"] is True
    assert result["content_fingerprint"] == manifest["content_fingerprint"]


def test_same_size_weight_shard_byte_change_is_rejected(tmp_path: Path) -> None:
    model, manifest_path = _checkpoint(tmp_path)
    shard = model / "model-00001-of-00001.safetensors"
    original_size = shard.stat().st_size
    shard.write_bytes(b"B" * original_size)
    assert shard.stat().st_size == original_size
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint_content_manifest(model, manifest_path)


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_checkpoint_file_set_change_is_rejected(
    tmp_path: Path, operation: str
) -> None:
    model, manifest_path = _checkpoint(tmp_path)
    if operation == "add":
        (model / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    else:
        (model / "config.json").unlink()
    with pytest.raises(ValueError, match="file set differs"):
        verify_checkpoint_content_manifest(model, manifest_path)


def test_manifest_internal_fingerprint_is_verified(tmp_path: Path) -> None:
    model, manifest_path = _checkpoint(tmp_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["files"][0]["sha256"] = "0" * 64
    write_json(manifest_path, value)
    with pytest.raises(ValueError, match="content_fingerprint mismatch"):
        verify_checkpoint_content_manifest(model, manifest_path)


def test_manifest_model_path_binding_is_verified(tmp_path: Path) -> None:
    model, manifest_path = _checkpoint(tmp_path)
    other = tmp_path / "other-model"
    other.mkdir()
    for source in model.iterdir():
        (other / source.name).write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="model_path differs"):
        verify_checkpoint_content_manifest(other, manifest_path)


def test_manifest_rejects_symlinked_checkpoint_entry(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    (model / "linked.bin").symlink_to(target)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        freeze_checkpoint_content_manifest(model, tmp_path / "manifest.json")


def test_manifest_fingerprint_covers_all_top_level_fields(tmp_path: Path) -> None:
    _model, manifest_path = _checkpoint(tmp_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded = payload.pop("fingerprint")
    assert recorded == object_fingerprint(payload)


def test_volatile_huggingface_cache_is_outside_checkpoint_identity(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    cache = model / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    lock = cache / "model.safetensors.lock"
    lock.write_text("first transient state", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    freeze_checkpoint_content_manifest(model, manifest_path)
    manifest = load_checkpoint_content_manifest(manifest_path)
    assert manifest["excluded_path_prefixes"] == [".cache/"]
    assert [entry["path"] for entry in manifest["files"]] == ["config.json"]
    lock.write_text("a later transient state", encoding="utf-8")
    assert verify_checkpoint_content_manifest(model, manifest_path)["passed"] is True
