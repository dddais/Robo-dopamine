from __future__ import annotations

import sys
import traceback
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import section
from ..data import load_configured_episodes
from ..io import (
    append_jsonl,
    deterministic_merge,
    latest_by_id,
    object_fingerprint,
    provenance,
    read_jsonl,
    sha256_file,
    stable_shard,
    write_json,
)
from ..schemas import SCHEMA_VERSION, GroundingRecord, TargetSpec
from ..video import extract_endpoints
from .dino import GroundingDINOGrounder
from .base import select_temporal_pair
from .audit import wilson_interval
from .parser import InstructionParser, build_queries
from .sam3 import SAM3Grounder


def _requested_example_ids(grounding: dict[str, Any]) -> set[str]:
    """Read a label-free, frozen ID list for parser and grounding.

    ``example_ids_file`` is intentionally an offline cohort artifact.  It is
    not a reward-bearing metadata file and is passed only as an ID allow-list.
    """
    inline = grounding.get("example_ids", [])
    path_value = grounding.get("example_ids_file")
    if inline and path_value:
        raise ValueError("Use only one of grounding.example_ids or example_ids_file")
    if path_value:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("example_ids_file must be a JSON array of strings")
        inline = value
    requested = {str(value) for value in inline}
    if len(requested) != len(inline):
        raise ValueError("Duplicate example IDs in grounding allow-list")
    return requested


def _target_from_row(row: dict) -> TargetSpec:
    value = dict(row)
    value.pop("schema_version", None)
    value["attributes"] = tuple(value.get("attributes", []))
    value["targets"] = tuple(value.get("targets", []))
    return TargetSpec(**value)


def _visualize(image_path: str, bbox, label: str, output: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline="red", width=max(2, image.width // 300))
        draw.text((bbox[0], max(0, bbox[1] - 14)), label, fill="red")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def run_parser(config: dict[str, Any], *, dry_run: bool = False) -> Path:
    grounding = section(config, "grounding")
    output_dir = Path(grounding["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "targets.jsonl"
    previous = latest_by_id(read_jsonl(path)) if path.exists() else {}
    parser = InstructionParser(
        grounding.get("parser_model_path"),
        use_model=bool(grounding.get("use_parser_model", True) and not dry_run),
    )
    episodes, _ = load_configured_episodes(grounding)
    requested_ids = _requested_example_ids(grounding)
    if requested_ids:
        episodes = [row for row in episodes if row.example_id in requested_ids]
    write_json(
        output_dir / "parse_manifest.json",
        provenance(sys.argv, config, Path(__file__).resolve().parents[2]),
    )
    limit = int(grounding.get("limit", 0))
    for index, episode in enumerate(episodes):
        if limit and index >= limit:
            break
        if episode.example_id in previous:
            continue
        # Deliberately pass only task and id: reward/check cannot leak.
        target = parser.parse(episode.task, episode.example_id)
        append_jsonl(path, target.to_dict())
    return path


def _save_mask(candidate: dict, path: Path) -> str | None:
    mask = candidate.pop("_mask", None)
    if mask is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)
    return str(path.resolve())


def run_grounding(
    config: dict[str, Any], backend: str, *, dry_run: bool = False, retry_failed: bool = False
) -> Path:
    grounding = section(config, "grounding")
    output_root = Path(grounding["output_dir"]).resolve()
    targets_path = output_root / "targets.jsonl"
    if not targets_path.exists():
        raise FileNotFoundError("Run `grounding parse` first")
    targets = {row["example_id"]: _target_from_row(row) for row in read_jsonl(targets_path)}
    backend_config = section(config, backend)
    grounder = (
        GroundingDINOGrounder(backend_config)
        if backend == "grounding_dino"
        else SAM3Grounder(backend_config)
    )
    run_dir = output_root / backend
    run_dir.mkdir(parents=True, exist_ok=True)
    shard_id = int(grounding.get("shard_id", 0))
    num_shards = int(grounding.get("num_shards", 1))
    records_path = (
        run_dir / "grounding.jsonl"
        if num_shards == 1
        else run_dir / f"grounding.shard-{shard_id:02d}.jsonl"
    )
    previous = latest_by_id(
        (
            {**row, "example_id": f"{row['example_id']}::{row['frame']}"}
            for row in read_jsonl(records_path)
        )
    ) if records_path.exists() else {}
    episodes, _ = load_configured_episodes(grounding)
    requested_ids = _requested_example_ids(grounding)
    if requested_ids:
        episodes = [row for row in episodes if row.example_id in requested_ids]
    all_episode_count = len(episodes)
    episodes = [
        row for row in episodes if stable_shard(row.video_sha256, num_shards) == shard_id
    ]
    write_json(
        run_dir / f"manifest.shard-{shard_id:02d}.json",
        {
            **provenance(sys.argv, config, Path(__file__).resolve().parents[2]),
            "backend": backend,
            "backend_fingerprint": grounder.fingerprint,
            "shard_id": shard_id,
            "num_shards": num_shards,
        },
    )
    limit = int(grounding.get("limit", 0))
    for index, episode in enumerate(episodes):
        if limit and index >= limit:
            break
        target = targets.get(episode.example_id)
        if target is None:
            continue
        queries = build_queries(target)
        try:
            frames = extract_endpoints(
                episode.example_id,
                episode.video_sha256,
                episode.video_path,
                run_dir / "frames" / episode.video_sha256,
            )
        except Exception as exc:
            for frame_name in ("first", "last"):
                append_jsonl(
                    records_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "example_id": episode.example_id,
                        "video_sha256": episode.video_sha256,
                        "backend": backend,
                        "frame": frame_name,
                        "status": "invalid",
                        "error": str(exc),
                    },
                )
            continue
        endpoint_specs = (
            ("first", frames.first_index, frames.first_path),
            ("last", frames.last_index, frames.last_path),
        )
        old_by_frame = {
            frame_name: previous.get(f"{episode.example_id}::{frame_name}")
            for frame_name, _, _ in endpoint_specs
        }
        if all(
            old
            and (old.get("status") == "ok" or not retry_failed)
            for old in old_by_frame.values()
        ):
            continue
        try:
            if dry_run:
                candidates_by_frame = {"first": [], "last": []}
                selected_by_frame = {"first": None, "last": None}
                selection_reason = "dry_run"
            else:
                candidates_by_frame = {
                    frame_name: grounder.candidates(image_path, queries)
                    for frame_name, _, image_path in endpoint_specs
                }
                first_selected, last_selected, selection_reason = select_temporal_pair(
                    frames.first_path,
                    frames.last_path,
                    candidates_by_frame["first"],
                    candidates_by_frame["last"],
                    query_count=len(queries),
                )
                selected_by_frame = {
                    "first": first_selected,
                    "last": last_selected,
                }
        except Exception as exc:
            for frame_name, _, _ in endpoint_specs:
                append_jsonl(
                    records_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "example_id": episode.example_id,
                        "video_sha256": episode.video_sha256,
                        "backend": backend,
                        "frame": frame_name,
                        "status": "invalid",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            continue
        for frame_name, frame_index, image_path in endpoint_specs:
            composite_id = f"{episode.example_id}::{frame_name}"
            old = previous.get(composite_id)
            if old and (old.get("status") == "ok" or not retry_failed):
                continue
            try:
                candidates = candidates_by_frame[frame_name]
                selected = selected_by_frame[frame_name]
                mask_path = None
                if selected is not None:
                    mask_path = _save_mask(
                        selected,
                        run_dir / "masks" / episode.video_sha256 / f"{frame_name}.png",
                    )
                record_payload = {
                    "example_id": episode.example_id,
                    "video_sha256": episode.video_sha256,
                    "backend": backend,
                    "query": queries,
                    "frame": frame_name,
                    "frame_index": frame_index,
                    "bbox": tuple(selected["bbox"]) if selected else None,
                    "mask_path": mask_path,
                    "score": float(selected["score"]) if selected else None,
                    "candidates": tuple(
                        {key: value for key, value in row.items() if key != "_mask"}
                        for row in candidates
                    ),
                    "selection_reason": selection_reason,
                    "audit_status": "pending",
                    "provenance": {
                        "backend_fingerprint": grounder.fingerprint,
                        "target_fingerprint": target.parser_fingerprint,
                        "frame_sha256": sha256_file(image_path),
                        "query": queries,
                        "image_path": image_path,
                        "task": episode.task,
                        "video_path": episode.video_path,
                        "target_phrase": target.target_phrase,
                    },
                    "status": "dry_run" if dry_run else ("ok" if selected else "no_detection"),
                }
                record_payload["grounding_fingerprint"] = object_fingerprint(record_payload)
                record = GroundingRecord(**record_payload)
                append_jsonl(records_path, record.to_dict())
                _visualize(
                    image_path,
                    record.bbox,
                    target.target_phrase,
                    run_dir / "visualizations" / episode.video_sha256 / f"{frame_name}.jpg",
                )
            except Exception as exc:
                append_jsonl(
                    records_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "example_id": episode.example_id,
                        "video_sha256": episode.video_sha256,
                        "backend": backend,
                        "frame": frame_name,
                        "status": "invalid",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
    latest = {}
    for row in read_jsonl(records_path):
        latest[(row["example_id"], row.get("frame"))] = row
    both = 0
    for episode in episodes:
        if all(
            latest.get((episode.example_id, frame), {}).get("status") == "ok"
            for frame in ("first", "last")
        ):
            both += 1
    write_json(
        run_dir
        / (
            "grounding_summary.json"
            if num_shards == 1
            else f"grounding_summary.shard-{shard_id:02d}.json"
        ),
        {
            "backend": backend,
            "population": len(episodes),
            "dual_endpoint_coverage_count": both,
            "dual_endpoint_coverage": both / len(episodes) if episodes else None,
            "dual_endpoint_wilson_ci95": wilson_interval(both, len(episodes)),
            "status_counts": {
                status: sum(row.get("status") == status for row in latest.values())
                for status in sorted({str(row.get("status")) for row in latest.values()})
            },
            "backend_fingerprint": grounder.fingerprint,
        },
    )
    if num_shards > 1:
        shard_paths = [
            run_dir / f"grounding.shard-{index:02d}.jsonl"
            for index in range(num_shards)
        ]
        if all(path.exists() for path in shard_paths):
            deterministic_merge(shard_paths, run_dir / "grounding.jsonl")
            merged_latest = {}
            for row in read_jsonl(run_dir / "grounding.jsonl"):
                merged_latest[(row["example_id"], row.get("frame"))] = row
            successful = {
                example_id
                for example_id, _ in merged_latest
                if all(
                    merged_latest.get((example_id, frame), {}).get("status") == "ok"
                    for frame in ("first", "last")
                )
            }
            write_json(
                run_dir / "grounding_summary.json",
                {
                    "backend": backend,
                    "population": all_episode_count,
                    "dual_endpoint_coverage_count": len(successful),
                    "dual_endpoint_coverage": len(successful) / all_episode_count
                    if all_episode_count
                    else None,
                    "dual_endpoint_wilson_ci95": wilson_interval(
                        len(successful), all_episode_count
                    ),
                    "backend_fingerprint": grounder.fingerprint,
                    "num_shards": num_shards,
                },
            )
    return records_path
