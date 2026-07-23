#!/usr/bin/env python3
"""Run the complete local-LLM + GroundingDINO grounding pipeline.

The default ``all`` stage is resumable.  Raw per-model parses and per-frame
detector outputs are appended batch-by-batch; deterministic merged artifacts
are regenerated from those caches.  Runtime model inputs never include reward
labels or ``gpt5_mini_check`` fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from roborewardbench.data import iter_metadata_rows, normalize_subset, sha256_file
from roborewardbench.run_benchmark import extract_frames

from .grounding import (
    GroundingDinoGrounder,
    pair_consistency,
    select_temporal_candidate_pair,
)
from .instruction_parser import (
    LocalInstructionParser,
    build_grounding_queries,
    compare_parses,
    heuristic_parse,
    iter_batches,
    model_slug,
)
from .report import (
    append_jsonl,
    create_contact_sheets,
    read_jsonl,
    render_sample_visualization,
    summarize_run,
    summary_markdown,
    write_flat_csv,
    write_jsonl,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("/home/dais/workspace/data/RoboRewardBench_counterfactual_reward1")
DEFAULT_QWEN3 = Path("/home/dais/workspace/model/Qwen3-4B-Instruct-2507")
DEFAULT_QWEN25 = Path("/home/dais/workspace/model/Qwen2.5-7B-Instruct")
DEFAULT_GROUNDING_DINO = Path("/home/dais/workspace/model/grounding-dino-base")
DEFAULT_OUTPUT = PACKAGE_DIR / "outputs" / "counterfactual_reward1"


@dataclass(frozen=True)
class DatasetExample:
    index: int
    example_id: str
    video_path: Path
    task: str
    subset: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(example_id: str) -> str:
    return hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:12]


def load_examples(dataset_root: str | Path, split: str, max_samples: int | None) -> list[DatasetExample]:
    root = Path(dataset_root).expanduser().resolve()
    split_root = root / split
    metadata_path = split_root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    examples: list[DatasetExample] = []
    seen: set[str] = set()
    for row in iter_metadata_rows(metadata_path):
        example_id = str(row["file_name"])
        if example_id in seen:
            raise ValueError(f"duplicate example id: {example_id}")
        seen.add(example_id)
        video_path = (split_root / example_id).resolve()
        try:
            video_path.relative_to(split_root)
        except ValueError as exc:
            raise ValueError(f"video path escapes split root: {example_id}") from exc
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")
        examples.append(
            DatasetExample(
                index=len(examples),
                example_id=example_id,
                video_path=video_path,
                task=str(row["task"]),
                subset=normalize_subset(example_id),
            )
        )
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def _model_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    files = []
    for file_path in sorted(root.iterdir()):
        if file_path.is_file():
            stat = file_path.stat()
            files.append(
                {
                    "name": file_path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    identity_payload = json.dumps(files, sort_keys=True).encode("utf-8")
    config_path = root / "config.json"
    return {
        "path": str(root),
        "files": files,
        "file_inventory_sha256": hashlib.sha256(identity_payload).hexdigest(),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
    }


def _environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "opencv-python", "pillow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu: dict[str, Any] = {"available": False, "devices": []}
    try:
        import torch

        gpu = {
            "available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        gpu["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "gpu": gpu,
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _raw_parse_path(output_dir: Path, model_path: str | Path) -> Path:
    return output_dir / "raw_parses" / f"{model_slug(model_path)}.jsonl"


def _parse_cache_matches(
    row: Mapping[str, Any],
    example: DatasetExample,
    *,
    model_path: str,
    model_cache_signature: str,
) -> bool:
    return bool(
        str(row.get("example_id")) == example.example_id
        and str(row.get("task")) == example.task
        and str(row.get("model_path")) == model_path
        and str(row.get("model_cache_signature")) == model_cache_signature
    )


def run_one_parser(
    examples: Sequence[DatasetExample],
    *,
    model_path: str | Path,
    output_path: Path,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    resolved_model_path = str(Path(model_path).expanduser().resolve())
    model_cache_signature = str(_model_identity(model_path)["file_inventory_sha256"])
    existing_rows = read_jsonl(output_path)
    existing = {str(row["example_id"]): row for row in existing_rows}
    pending = [
        example
        for example in examples
        if not _parse_cache_matches(
            existing.get(example.example_id, {}),
            example,
            model_path=resolved_model_path,
            model_cache_signature=model_cache_signature,
        )
    ]
    if pending:
        print(f"[parse] loading {model_path} on {device}; pending={len(pending)}", flush=True)
        with LocalInstructionParser(model_path, device=device) as parser:
            completed = 0
            for batch in iter_batches(pending, batch_size):
                outputs = parser.parse_batch([example.task for example in batch])
                records = []
                for example, output in zip(batch, outputs):
                    records.append(
                        {
                            "example_id": example.example_id,
                            "index": example.index,
                            "subset": example.subset,
                            "task": example.task,
                            "model_path": resolved_model_path,
                            "model_cache_signature": model_cache_signature,
                            **output,
                        }
                    )
                append_jsonl(output_path, records)
                for record in records:
                    existing[str(record["example_id"])] = record
                completed += len(records)
                print(f"[parse] {Path(model_path).name}: {completed}/{len(pending)}", flush=True)
    canonical = {example.example_id: existing[example.example_id] for example in examples}
    write_jsonl(output_path, canonical.values())
    return canonical


def merge_parses(
    examples: Sequence[DatasetExample],
    primary: Mapping[str, Mapping[str, Any]],
    secondary: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for example in examples:
        first = dict(primary.get(example.example_id, {}))
        second = dict(secondary.get(example.example_id, {}))
        first_valid = bool(first.get("valid") and first.get("parsed"))
        second_valid = bool(second.get("valid") and second.get("parsed"))
        diagnostics = {
            "agreement_level": "unavailable",
            "head_exact": False,
            "head_compatible": False,
            "phrase_jaccard": None,
            "type_agreement": False,
        }
        if first_valid and second_valid:
            diagnostics = compare_parses(first["parsed"], second["parsed"])
        if first_valid:
            selected = first["parsed"]
            selected_source = "primary"
        elif second_valid:
            selected = second["parsed"]
            selected_source = "secondary"
        else:
            selected = heuristic_parse(example.task).to_dict()
            selected_source = "heuristic_fallback"
        merged.append(
            {
                "example_id": example.example_id,
                "index": example.index,
                "subset": example.subset,
                "task": example.task,
                "primary": {
                    "valid": first_valid,
                    "parsed": first.get("parsed"),
                    "raw": first.get("raw"),
                    "error": first.get("error"),
                    "model_path": first.get("model_path"),
                },
                "secondary": {
                    "valid": second_valid,
                    "parsed": second.get("parsed"),
                    "raw": second.get("raw"),
                    "error": second.get("error"),
                    "model_path": second.get("model_path"),
                },
                **diagnostics,
                "selected_source": selected_source,
                "selected_parse": selected,
                "grounding_queries": build_grounding_queries(selected),
            }
        )
    return merged


def parse_stage(
    examples: Sequence[DatasetExample],
    *,
    output_dir: Path,
    primary_model: str | Path,
    secondary_model: str | Path,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = run_one_parser(
        examples,
        model_path=primary_model,
        output_path=_raw_parse_path(output_dir, primary_model),
        device=device,
        batch_size=batch_size,
    )
    secondary = run_one_parser(
        examples,
        model_path=secondary_model,
        output_path=_raw_parse_path(output_dir, secondary_model),
        device=device,
        batch_size=batch_size,
    )
    merged = merge_parses(examples, primary, secondary)
    write_jsonl(output_dir / "instruction_parses.jsonl", merged)
    return merged


def extract_endpoint_manifest(
    examples: Sequence[DatasetExample],
    *,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    manifest_path = output_dir / "frame_manifest.jsonl"
    existing_rows = read_jsonl(manifest_path)
    existing = {str(row["example_id"]): row for row in existing_rows}
    for example in examples:
        row = existing.get(example.example_id)
        video_stat = example.video_path.stat()
        if row and row.get("status") == "ok" and all(
            Path(row[role]["image_path"]).is_file() for role in ("before", "after")
        ) and (
            str(row.get("video_path")) == str(example.video_path)
            and int(row.get("video_size", -1)) == video_stat.st_size
            and int(row.get("video_mtime_ns", -1)) == video_stat.st_mtime_ns
            and str(row.get("task")) == example.task
        ):
            continue
        frame_dir = output_dir / "frames" / f"{example.index:04d}_{stable_id(example.example_id)}"
        try:
            paths, indices, fps = extract_frames(
                example.video_path,
                frame_dir,
                sampling="uniform",
                max_states=2,
            )
            record = {
                "example_id": example.example_id,
                "index": example.index,
                "subset": example.subset,
                "task": example.task,
                "video_path": str(example.video_path),
                "video_size": video_stat.st_size,
                "video_mtime_ns": video_stat.st_mtime_ns,
                "status": "ok",
                "source_fps": fps,
                "before": {"image_path": str(paths[0]), "frame_index": indices[0]},
                "after": {"image_path": str(paths[-1]), "frame_index": indices[-1]},
                "error": None,
            }
        except Exception as exc:
            record = {
                "example_id": example.example_id,
                "index": example.index,
                "subset": example.subset,
                "task": example.task,
                "video_path": str(example.video_path),
                "video_size": video_stat.st_size,
                "video_mtime_ns": video_stat.st_mtime_ns,
                "status": "error",
                "before": None,
                "after": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        existing[example.example_id] = record
        # Regenerate rather than append duplicate records when an old frame
        # manifest entry was incomplete.
        write_jsonl(manifest_path, [existing[key] for key in sorted(existing)])
        print(f"[frames] {example.index + 1}/{len(examples)} {record['status']}", flush=True)
    canonical = {example.example_id: existing[example.example_id] for example in examples}
    write_jsonl(manifest_path, canonical.values())
    return canonical


def _frame_key(example_id: str, role: str) -> str:
    return f"{example_id}\n{role}"


def _detector_provenance(
    item: Mapping[str, Any],
    *,
    model_path: str,
    model_cache_signature: str,
    detection_threshold: float,
    text_threshold: float,
    accept_threshold: float,
    top_k: int,
) -> dict[str, Any]:
    image_path = Path(str(item["image_path"])).expanduser().resolve()
    image_stat = image_path.stat()
    return {
        "model_path": model_path,
        "model_cache_signature": model_cache_signature,
        "image_path": str(image_path),
        "image_size_bytes": image_stat.st_size,
        "image_mtime_ns": image_stat.st_mtime_ns,
        "detection_threshold": float(detection_threshold),
        "text_threshold": float(text_threshold),
        "accept_threshold": float(accept_threshold),
        "top_k": int(top_k),
    }


def _detector_cache_matches(
    row: Mapping[str, Any],
    item: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> bool:
    try:
        return bool(
            str(row.get("example_id")) == str(item["example_id"])
            and str(row.get("frame_role")) == str(item["frame_role"])
            and list(row.get("queries") or []) == list(item["queries"])
            and str(row.get("model_path")) == str(provenance["model_path"])
            and str(row.get("model_cache_signature"))
            == str(provenance["model_cache_signature"])
            and str(Path(str(row.get("image_path"))).expanduser().resolve())
            == str(provenance["image_path"])
            and int(row.get("image_size_bytes", -1)) == int(provenance["image_size_bytes"])
            and int(row.get("image_mtime_ns", -1)) == int(provenance["image_mtime_ns"])
            and float(row.get("detection_threshold"))
            == float(provenance["detection_threshold"])
            and float(row.get("text_threshold")) == float(provenance["text_threshold"])
            and float(row.get("accept_threshold")) == float(provenance["accept_threshold"])
            and int(row.get("top_k")) == int(provenance["top_k"])
        )
    except (TypeError, ValueError):
        return False


def run_detector(
    pending: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    model_path: str | Path,
    device: str,
    batch_size: int,
    detection_threshold: float,
    text_threshold: float,
    accept_threshold: float,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    resolved_model_path = str(Path(model_path).expanduser().resolve())
    model_cache_signature = str(_model_identity(model_path)["file_inventory_sha256"])
    existing_rows = read_jsonl(output_path)
    existing = {_frame_key(str(row["example_id"]), str(row["frame_role"])): row for row in existing_rows}
    provenance_by_key = {
        _frame_key(str(item["example_id"]), str(item["frame_role"])): _detector_provenance(
            item,
            model_path=resolved_model_path,
            model_cache_signature=model_cache_signature,
            detection_threshold=detection_threshold,
            text_threshold=text_threshold,
            accept_threshold=accept_threshold,
            top_k=top_k,
        )
        for item in pending
    }
    queue = [
        item
        for item in pending
        if not _detector_cache_matches(
            existing.get(_frame_key(str(item["example_id"]), str(item["frame_role"])), {}),
            item,
            provenance_by_key[_frame_key(str(item["example_id"]), str(item["frame_role"]))],
        )
    ]
    if queue:
        print(f"[ground] loading {model_path} on {device}; pending_frames={len(queue)}", flush=True)
        with GroundingDinoGrounder(
            model_path,
            device=device,
            detection_threshold=detection_threshold,
            text_threshold=text_threshold,
            accept_threshold=accept_threshold,
            top_k=top_k,
        ) as grounder:
            completed = 0
            for batch in iter_batches(queue, batch_size):
                try:
                    records = grounder.detect_batch(batch)
                except Exception as batch_exc:
                    # Preserve the batch-level failure, then isolate individual
                    # items so one corrupt frame cannot erase the other results.
                    records = []
                    for item in batch:
                        try:
                            records.extend(grounder.detect_batch([item]))
                        except Exception as item_exc:
                            records.append(
                                {
                                    "example_id": item["example_id"],
                                    "frame_role": item["frame_role"],
                                    "image_path": str(item["image_path"]),
                                    "queries": list(item["queries"]),
                                    "query_text": grounder.make_query_text(item["queries"]),
                                    "candidates": [],
                                    "selected": None,
                                    "detected": False,
                                    "accepted": False,
                                    "error": (
                                        f"batch={type(batch_exc).__name__}: {batch_exc}; "
                                        f"item={type(item_exc).__name__}: {item_exc}"
                                    ),
                                }
                            )
                for record in records:
                    key = _frame_key(str(record["example_id"]), str(record["frame_role"]))
                    record.update(provenance_by_key[key])
                append_jsonl(output_path, records)
                for record in records:
                    existing[_frame_key(str(record["example_id"]), str(record["frame_role"]))] = record
                completed += len(records)
                print(f"[ground] {completed}/{len(queue)}", flush=True)
    expected_keys = [
        _frame_key(str(item["example_id"]), str(item["frame_role"])) for item in pending
    ]
    canonical = {key: existing[key] for key in expected_keys}
    write_jsonl(output_path, canonical.values())
    return canonical


def aggregate_grounding(
    examples: Sequence[DatasetExample],
    parses: Sequence[Mapping[str, Any]],
    frame_manifest: Mapping[str, Mapping[str, Any]],
    frame_outputs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parse_by_id = {str(row["example_id"]): row for row in parses}
    results: list[dict[str, Any]] = []
    for example in examples:
        parse = parse_by_id[example.example_id]
        frames = frame_manifest.get(example.example_id)
        raw_before = frame_outputs.get(_frame_key(example.example_id, "before"))
        raw_after = frame_outputs.get(_frame_key(example.example_id, "after"))
        before = dict(raw_before) if raw_before else None
        after = dict(raw_after) if raw_after else None
        pair_selection: dict[str, Any] = {"available": False}
        selected_parse = parse["selected_parse"]
        if before and after and before.get("candidates") and after.get("candidates"):
            image_size = before.get("image_size") or after.get("image_size") or [1, 1]
            pair_selection = select_temporal_candidate_pair(
                before["candidates"],
                after["candidates"],
                before_image_path=before["image_path"],
                after_image_path=after["image_path"],
                image_size=image_size,
                attributes=selected_parse.get("attributes", []),
                target_type=str(selected_parse.get("target_type", "object")),
            )
            if pair_selection.get("available"):
                before["raw_selected"] = before.get("selected")
                after["raw_selected"] = after.get("selected")
                before["selected"] = pair_selection["before"]
                after["selected"] = pair_selection["after"]
                before["selection_method"] = "temporal_pair"
                after["selection_method"] = "temporal_pair"
                before["detected"] = True
                after["detected"] = True
                before["accepted"] = bool(
                    float(before["selected"]["score"]) >= float(before.get("accept_threshold", 0.25))
                )
                after["accepted"] = bool(
                    float(after["selected"]["score"]) >= float(after.get("accept_threshold", 0.25))
                )
        if not frames or frames.get("status") != "ok":
            status = "frame_error"
            consistency = {
                "available": False,
                "consistent": False,
                "iou": None,
                "center_distance": None,
                "area_ratio": None,
            }
        else:
            image_size = (before or after or {}).get("image_size", [1, 1])
            consistency = pair_consistency(
                (before or {}).get("selected"),
                (after or {}).get("selected"),
                image_size=image_size,
            )
            detected_count = sum(bool(frame and frame.get("detected")) for frame in (before, after))
            accepted_count = sum(bool(frame and frame.get("accepted")) for frame in (before, after))
            if detected_count == 0:
                status = "no_detection"
            elif accepted_count == 2:
                status = "accepted_both"
            elif accepted_count == 1:
                status = "accepted_partial"
            else:
                status = "low_confidence"
        pair_selection_summary = {
            key: value
            for key, value in pair_selection.items()
            if key not in {"before", "after"}
        }
        pair_margin = pair_selection_summary.get("pair_margin")
        steering_ready = bool(
            before
            and after
            and before.get("accepted")
            and after.get("accepted")
            and consistency.get("consistent")
            and not selected_parse.get("ambiguous")
            and parse.get("agreement_level") != "disagree"
            and float(pair_selection_summary.get("area_fraction_mean", 0.0)) < 0.85
            and (pair_margin is None or float(pair_margin) >= 0.02)
            and (
                not pair_selection_summary.get("color_attributes")
                or float(pair_selection_summary.get("color_match_mean") or 0.0) >= 0.10
            )
        )
        visualization_file = f"{example.index:04d}_{stable_id(example.example_id)}.jpg"
        results.append(
            {
                "example_id": example.example_id,
                "index": example.index,
                "subset": example.subset,
                "task": example.task,
                "selected_parse": selected_parse,
                "selected_source": parse["selected_source"],
                "agreement_level": parse["agreement_level"],
                "grounding_queries": parse["grounding_queries"],
                "before": before,
                "after": after,
                "pair_consistency": consistency,
                "pair_selection": pair_selection_summary,
                "steering_ready": steering_ready,
                "status": status,
                "frame_error": (frames or {}).get("error"),
                "visualization_file": visualization_file,
            }
        )
    return results


def grounding_stage(
    examples: Sequence[DatasetExample],
    parses: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    model_path: str | Path,
    device: str,
    batch_size: int,
    detection_threshold: float,
    text_threshold: float,
    accept_threshold: float,
    top_k: int,
    write_visualizations: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_manifest = extract_endpoint_manifest(examples, output_dir=output_dir)
    parse_by_id = {str(row["example_id"]): row for row in parses}
    detector_items: list[dict[str, Any]] = []
    for example in examples:
        frames = frame_manifest.get(example.example_id)
        if not frames or frames.get("status") != "ok":
            continue
        queries = parse_by_id[example.example_id]["grounding_queries"]
        for role in ("before", "after"):
            detector_items.append(
                {
                    "example_id": example.example_id,
                    "frame_role": role,
                    "image_path": frames[role]["image_path"],
                    "queries": queries,
                }
            )
    frame_outputs = run_detector(
        detector_items,
        output_path=output_dir / "grounding_frames.jsonl",
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        detection_threshold=detection_threshold,
        text_threshold=text_threshold,
        accept_threshold=accept_threshold,
        top_k=top_k,
    )
    results = aggregate_grounding(examples, parses, frame_manifest, frame_outputs)
    write_jsonl(output_dir / "grounding_results.jsonl", results)
    write_flat_csv(results, output_dir / "grounding_results.csv")

    if write_visualizations:
        visualization_dir = output_dir / "visualizations"
        for position, result in enumerate(results, 1):
            destination = visualization_dir / str(result["visualization_file"])
            render_sample_visualization(result, destination)
            if position % 25 == 0 or position == len(results):
                print(f"[viz] {position}/{len(results)}", flush=True)
        create_contact_sheets(results, visualization_dir, output_dir / "contact_sheets")

    summary = summarize_run(parses, results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return results, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "parse", "ground", "summarize"), nargs="?", default="all")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--primary-llm", default=str(DEFAULT_QWEN3))
    parser.add_argument("--secondary-llm", default=str(DEFAULT_QWEN25))
    parser.add_argument("--grounding-model", default=str(DEFAULT_GROUNDING_DINO))
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--grounding-device", default="cuda:0")
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--grounding-batch-size", type=int, default=8)
    parser.add_argument("--detection-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--accept-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means the full split")
    parser.add_argument("--skip-visualizations", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.llm_batch_size <= 0 or args.grounding_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    max_samples = args.max_samples or None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, "-m", "roborewardbench.dopamine_eval.pipeline", *(argv or sys.argv[1:])])
    examples = load_examples(args.dataset_root, args.split, max_samples)
    metadata_path = Path(args.dataset_root).expanduser().resolve() / args.split / "metadata.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "command": command,
        "cwd": os.getcwd(),
        "stage": args.stage,
        "dataset": {
            "root": str(Path(args.dataset_root).expanduser().resolve()),
            "split": args.split,
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "selected_samples": len(examples),
        },
        "models": {
            "primary_llm": _model_identity(args.primary_llm),
            "secondary_llm": _model_identity(args.secondary_llm),
            "grounding_dino": _model_identity(args.grounding_model),
        },
        "parameters": {
            "llm_device": args.llm_device,
            "grounding_device": args.grounding_device,
            "llm_batch_size": args.llm_batch_size,
            "grounding_batch_size": args.grounding_batch_size,
            "detection_threshold": args.detection_threshold,
            "text_threshold": args.text_threshold,
            "accept_threshold": args.accept_threshold,
            "top_k": args.top_k,
            "runtime_label_fields_provided_to_models": [],
        },
        "environment": _environment_manifest(),
    }
    _write_manifest(manifest_path, manifest)
    started = time.monotonic()
    try:
        if args.stage in {"all", "parse"}:
            parses = parse_stage(
                examples,
                output_dir=output_dir,
                primary_model=args.primary_llm,
                secondary_model=args.secondary_llm,
                device=args.llm_device,
                batch_size=args.llm_batch_size,
            )
        else:
            parses = read_jsonl(output_dir / "instruction_parses.jsonl")
            expected = {example.example_id for example in examples}
            parses = [row for row in parses if str(row.get("example_id")) in expected]
            if len(parses) != len(examples):
                raise RuntimeError(
                    f"instruction_parses.jsonl has {len(parses)} selected records, expected {len(examples)}"
                )

        if args.stage in {"all", "ground"}:
            results, summary = grounding_stage(
                examples,
                parses,
                output_dir=output_dir,
                model_path=args.grounding_model,
                device=args.grounding_device,
                batch_size=args.grounding_batch_size,
                detection_threshold=args.detection_threshold,
                text_threshold=args.text_threshold,
                accept_threshold=args.accept_threshold,
                top_k=args.top_k,
                write_visualizations=not args.skip_visualizations,
            )
        elif args.stage == "summarize":
            results = read_jsonl(output_dir / "grounding_results.jsonl")
            expected = {example.example_id for example in examples}
            results = [row for row in results if str(row.get("example_id")) in expected]
            summary = summarize_run(parses, results)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
        else:
            results = []
            summary = None

        if summary is not None:
            (output_dir / "summary.md").write_text(
                summary_markdown(summary, command=command, output_dir=str(output_dir)),
                encoding="utf-8",
            )
        manifest.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "artifacts": {
                    "instruction_parses": str(output_dir / "instruction_parses.jsonl"),
                    "grounding_results": str(output_dir / "grounding_results.jsonl"),
                    "summary_json": str(output_dir / "summary.json"),
                    "summary_markdown": str(output_dir / "summary.md"),
                },
                "counts": {
                    "examples": len(examples),
                    "parses": len(parses),
                    "grounding_results": len(results),
                },
            }
        )
        _write_manifest(manifest_path, manifest)
        print(f"[done] output={output_dir}", flush=True)
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
