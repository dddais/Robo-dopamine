from __future__ import annotations

import sys
import traceback
import json
from dataclasses import replace
from uuid import uuid4
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
from .base import select_relational_candidate, select_temporal_pair, select_unambiguous_candidate
from .audit import wilson_interval
from .parser import SPATIAL_RELATIONS, InstructionParser, build_queries, reference_queries, spatial_description
from .sam3 import SAM3Grounder


def _select_initial_candidates(image_path, target, candidates, references, config):
    if target.relation:
        return select_relational_candidate(
            image_path, candidates, references, target.relation,
            ordinal_index=target.ordinal_index,
            ambiguity_margin=float(config.get("ambiguity_score_margin", 0.05)),
            geometry_margin=float(config.get("geometry_margin", 0.01)),
            reference_consensus=bool(config.get("reference_consensus", False)),
        )
    selected, reason = select_unambiguous_candidate(
        candidates, score_margin=float(config.get("ambiguity_score_margin", 0.05)),
    )
    return selected, None, reason


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
    # Also support old parse files: explicit spatial syntax must not silently
    # revert to maximum confidence after upgrading the grounder.
    description = spatial_description(str(value.get("target_phrase", "")))
    if description:
        value.update(description)
    relation = value.get("relation")
    if relation not in SPATIAL_RELATIONS:
        value["relation"] = None
        value["reference_object"] = None
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
    task_cache: dict[str, TargetSpec] = {}
    for index, episode in enumerate(episodes):
        if limit and index >= limit:
            break
        target_fingerprint = object_fingerprint({"parser": parser.fingerprint, "task": episode.task})
        if previous.get(episode.example_id, {}).get("parser_fingerprint") == target_fingerprint:
            task_cache[episode.task] = _target_from_row(previous[episode.example_id])
            continue
        # Deliberately pass only task and id: reward/check cannot leak.
        if episode.task not in task_cache:
            task_cache[episode.task] = parser.parse(episode.task, episode.example_id)
        target = replace(task_cache[episode.task], example_id=episode.example_id,
                         parser_fingerprint=target_fingerprint)
        append_jsonl(path, target.to_dict())
    return path


def _save_mask(candidate: dict, path: Path) -> str | None:
    mask = candidate.pop("_mask", None)
    if mask is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)
    return str(path.resolve())


def _artifact_key(example_id: str) -> str:
    """Filesystem-safe instruction key for one-to-many counterfactual videos."""
    return object_fingerprint({"example_id": example_id})[:20]


def _render_tracking_artifacts(
    video_path: str,
    tracks: list[dict[str, Any]],
    output_dir: Path,
    label: str,
    *,
    max_width: int = 960,
) -> tuple[str, str]:
    """Write a boxed tracking MP4 and a six-timepoint contact sheet."""
    if not tracks:
        raise ValueError("Cannot render an empty track")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "tracking_preview.mp4"
    contact_path = output_dir / "tracking_contact_sheet.jpg"
    by_frame = {int(row["frame_index"]): row for row in tracks}
    tracked_indices = sorted(by_frame)
    wanted = {
        int(round(value))
        for value in np.linspace(tracked_indices[0], tracked_indices[-1], num=min(6, len(tracked_indices)))
    }
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode tracking preview source: {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    scale = min(1.0, max_width / max(1, source_width))
    output_size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 20.0,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create tracking preview: {preview_path}")
    contacts: list[np.ndarray] = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            row = by_frame.get(index)
            if row is not None:
                x1, y1, x2, y2 = (round(float(value)) for value in row["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f"{label}  frame={index}",
                    (max(4, x1), max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            resized = cv2.resize(frame, output_size) if scale != 1.0 else frame
            writer.write(resized)
            if index in wanted:
                contacts.append(resized.copy())
            index += 1
    finally:
        capture.release()
        writer.release()
    if not contacts:
        raise RuntimeError("No frames were decoded for tracking contact sheet")
    cell_width = min(480, output_size[0])
    cell_height = max(1, round(output_size[1] * cell_width / output_size[0]))
    cells = [cv2.resize(frame, (cell_width, cell_height)) for frame in contacts]
    while len(cells) < 6:
        cells.append(np.full_like(cells[0], 245))
    sheet = np.vstack((np.hstack(cells[:3]), np.hstack(cells[3:6])))
    if not cv2.imwrite(str(contact_path), sheet):
        raise RuntimeError(f"Cannot write tracking contact sheet: {contact_path}")
    return str(preview_path.resolve()), str(contact_path.resolve())


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
    backend_fingerprint = grounder.fingerprint
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
            "backend_fingerprint": backend_fingerprint,
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
        ref_queries = reference_queries(target)
        artifact_key = _artifact_key(episode.example_id)
        track_path = None
        tracking_preview_path = None
        tracking_contact_sheet_path = None
        tracking_preview_error = None
        reference_selected = None
        tracking_diagnostics = None
        tracking_error = None
        visual_grounding = None
        reference_candidates = []
        input_fingerprint = object_fingerprint({
            "backend": backend_fingerprint, "target": target.to_dict(),
            "video_sha256": episode.video_sha256, "task": episode.task, "dry_run": dry_run,
        })
        try:
            frames = extract_endpoints(
                episode.example_id,
                episode.video_sha256,
                episode.video_path,
                run_dir / "frames" / episode.video_sha256,
            )
            view_endpoints = {
                "front": {
                    "first": frames.first_path,
                    "last": frames.last_path,
                    "first_index": frames.first_index,
                    "last_index": frames.last_index,
                }
            }
            for camera, view_video_path in episode.views.items():
                if camera == "front":
                    continue
                view_frames = extract_endpoints(
                    episode.example_id,
                    episode.video_sha256,
                    view_video_path,
                    run_dir / "frames" / episode.video_sha256 / camera,
                )
                view_endpoints[camera] = {
                    "first": view_frames.first_path,
                    "last": view_frames.last_path,
                    "first_index": view_frames.first_index,
                    "last_index": view_frames.last_index,
                }
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
        def current(old):
            return old and old.get("provenance", {}).get("input_fingerprint") == input_fingerprint

        if all(
            old
            and current(old)
            and (old.get("status") == "ok" or not retry_failed)
            for old in old_by_frame.values()
        ):
            continue
        try:
            if dry_run:
                candidates_by_frame = {"first": [], "last": []}
                selected_by_frame = {"first": None, "last": None}
                selection_reason = "dry_run"
            elif backend == "sam3" and bool(backend_config.get("tracking", True)):
                first_candidates = grounder.candidates(frames.first_path, queries)
                if target.relation:
                    reference_candidates = grounder.candidates(
                        frames.first_path, ref_queries
                    )
                else:
                    reference_candidates = []
                first_selected, reference_selected, relation_reason = _select_initial_candidates(
                    frames.first_path, target, first_candidates, reference_candidates, backend_config,
                )
                visual_config = backend_config.get("visual_resolver", {})
                force_visual = any(query in visual_config.get("always_verify_queries", []) for query in queries)
                if visual_config.get("enabled") and (first_selected is None or force_visual):
                    original_reason = relation_reason
                    first_candidates, target_visual = grounder.visual_candidates(
                        frames.first_path, queries, first_candidates,
                    )
                    reference_visual = None
                    if ref_queries and (not reference_candidates or "reference" in original_reason):
                        reference_candidates, reference_visual = grounder.visual_candidates(
                            frames.first_path, ref_queries, reference_candidates,
                        )
                    visual_grounding = {"original_reason": original_reason, "target": target_visual,
                                        "reference": reference_visual}
                    if force_visual:
                        visual_grounding["trigger"] = "configured_category_semantic_verification"
                    first_selected, reference_selected, relation_reason = _select_initial_candidates(
                        frames.first_path, target, first_candidates, reference_candidates, backend_config,
                    )
                    relation_reason += "_with_visual_proposals"
                tracks = []
                if first_selected is not None:
                    try:
                        tracks = grounder.track(
                            episode.video_path,
                            first_selected["bbox"],
                            frames.first_index,
                            anchor_mask=first_selected.get("_mask"),
                            terminal_index=frames.last_index,
                        )
                        visual_config = backend_config.get("visual_resolver", {})
                        if visual_config.get("enabled") and visual_config.get("verify_on_tracking_failure") and (
                            visual_grounding is None and not any(row["frame_index"] == frames.last_index for row in tracks)
                        ):
                            original_selection = first_selected
                            first_candidates, target_visual = grounder.visual_candidates(
                                frames.first_path, queries, first_candidates,
                            )
                            visual_grounding = {
                                "original_reason": relation_reason, "target": target_visual,
                                "trigger": "forward_tracking_missing_terminal", "reference": None,
                            }
                            first_selected, reference_selected, relation_reason = _select_initial_candidates(
                                frames.first_path, target, first_candidates, reference_candidates, backend_config,
                            )
                            relation_reason += "_with_visual_verification"
                            if first_selected is None:
                                tracks = []
                            elif first_selected["bbox"] != original_selection["bbox"] or (
                                first_selected.get("_mask") is not original_selection.get("_mask")
                            ):
                                # A subtype correction can change the intended instance.
                                # Rebuild a full track for that anchor before recovery.
                                tracks = grounder.track(
                                    episode.video_path, first_selected["bbox"], frames.first_index,
                                    anchor_mask=first_selected.get("_mask"), terminal_index=frames.last_index,
                                )
                        if first_selected is not None and backend_config.get("terminal_recovery", {}).get("enabled") and not any(
                            row["frame_index"] == frames.last_index for row in tracks
                        ):
                            from .recovery import recover_terminal_track
                            tracks = recover_terminal_track(
                                grounder, episode.video_path, first_selected, frames.last_path, queries,
                                frames.first_index, frames.last_index, tracks,
                            )
                    except Exception as tracking_exc:
                        # Detection and video propagation are separate failure
                        # stages; retain the valid first-frame observation.
                        tracking_error = {
                            "error_type": type(tracking_exc).__name__,
                            "error": str(tracking_exc), "traceback": traceback.format_exc(),
                        }
                track_by_frame = {int(row["frame_index"]): row for row in tracks}
                tracking_diagnostics = dict(grounder.last_tracking_diagnostics) if first_selected else None
                last_selected = track_by_frame.get(frames.last_index)
                candidates_by_frame = {
                    "first": first_candidates,
                    "last": [last_selected] if last_selected is not None else [],
                }
                selected_by_frame = {
                    "first": first_selected,
                    "last": last_selected,
                }
                selection_reason = f"{relation_reason}_then_sam3_box_tracking"
                # Every attempt gets its own immutable media paths. A retry
                # must not mutate artifacts referenced by older JSONL rows.
                attempt_key = f"{input_fingerprint[:12]}-{uuid4().hex[:12]}"
                track_dir = run_dir / "tracks" / episode.video_sha256 / artifact_key / attempt_key
                serializable_tracks = [
                    {key: value for key, value in row.items() if key != "_mask"}
                    for row in tracks
                ]
                track_path_obj = track_dir / "track.json"
                write_json(
                    track_path_obj,
                    {
                        "example_id": episode.example_id,
                        "video_sha256": episode.video_sha256,
                        "anchor_bbox": first_selected.get("bbox") if first_selected else None,
                        "anchor_frame_index": frames.first_index,
                        "terminal_frame_index": frames.last_index,
                        "relation": target.relation,
                        "reference_queries": ref_queries,
                        "input_fingerprint": input_fingerprint,
                        "bbox_frame_policy": "exact",
                        "tracking_diagnostics": tracking_diagnostics,
                        "tracking_error": tracking_error,
                        "reference_candidate": {
                            key: value
                            for key, value in (reference_selected or {}).items()
                            if key != "_mask"
                        }
                        or None,
                        "frames": serializable_tracks,
                    },
                )
                track_path = str(track_path_obj.resolve())
                if tracks and bool(backend_config.get("tracking_preview", True)):
                    try:
                        tracking_preview_path, tracking_contact_sheet_path = (
                            _render_tracking_artifacts(
                                episode.video_path,
                                tracks,
                                track_dir,
                                target.target_phrase,
                                max_width=int(
                                    backend_config.get(
                                        "tracking_preview_max_width", 960
                                    )
                                ),
                            )
                        )
                    except Exception as preview_exc:
                        # A missing MP4 codec must not turn a valid bbox track
                        # into an invalid grounding sample. The JSON track and
                        # endpoint visualizations remain reviewable.
                        tracking_preview_error = (
                            f"{type(preview_exc).__name__}: {preview_exc}"
                        )
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
                        "provenance": {"input_fingerprint": input_fingerprint},
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            continue
        for frame_name, frame_index, image_path in endpoint_specs:
            # Rerunning one video replaces both endpoint records coherently;
            # keeping an old successful first endpoint could bind a new last
            # endpoint to a different anchor/track.
            try:
                candidates = candidates_by_frame[frame_name]
                selected = selected_by_frame[frame_name]
                mask_path = None
                if selected is not None:
                    mask_path = _save_mask(
                        selected,
                        run_dir
                        / "masks"
                        / episode.video_sha256
                        / artifact_key
                        / (attempt_key if track_path else input_fingerprint[:12])
                        / f"{frame_name}.png",
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
                        "backend_fingerprint": backend_fingerprint,
                        "input_fingerprint": input_fingerprint,
                        "target_fingerprint": target.parser_fingerprint,
                        "frame_sha256": sha256_file(image_path),
                        "query": queries,
                        "image_path": image_path,
                        "task": episode.task,
                        "video_path": episode.video_path,
                        "view_paths": episode.views,
                        "view_endpoint_paths": view_endpoints,
                        "target_phrase": target.target_phrase,
                        "relation": target.relation,
                        "reference_object": target.reference_object,
                        "reference_queries": ref_queries,
                        "reference_candidates": [
                            {key: value for key, value in row.items() if key != "_mask"}
                            for row in reference_candidates
                        ],
                        "ordinal_index": target.ordinal_index,
                        "subject_phrase": target.subject_phrase,
                        "tracking_diagnostics": tracking_diagnostics,
                        "tracking_error": tracking_error,
                        "reference_candidate": {
                            key: value
                            for key, value in (reference_selected or {}).items()
                            if key != "_mask"
                        }
                        or None,
                        "tracking_path": track_path,
                        "visual_grounding": visual_grounding,
                        "tracking_preview_path": tracking_preview_path,
                        "tracking_contact_sheet_path": tracking_contact_sheet_path,
                        "tracking_preview_error": tracking_preview_error,
                    },
                    "status": (
                        "dry_run" if dry_run else "invalid" if frame_name == "last" and tracking_error
                        else "ok" if selected else "no_detection"
                    ),
                }
                record_payload["grounding_fingerprint"] = object_fingerprint(record_payload)
                record = GroundingRecord(**record_payload)
                append_jsonl(records_path, record.to_dict())
                _visualize(
                    image_path,
                    record.bbox,
                    target.target_phrase,
                    run_dir
                    / "visualizations"
                    / episode.video_sha256
                    / artifact_key
                    / f"{frame_name}.jpg",
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
                        "provenance": {"input_fingerprint": input_fingerprint},
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
            "backend_fingerprint": backend_fingerprint,
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
                    "backend_fingerprint": backend_fingerprint,
                    "num_shards": num_shards,
                },
            )
    return records_path
