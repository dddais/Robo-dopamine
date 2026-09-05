"""File-backed ranking, steering, and scoring for cross-model attention runs.

This runner keeps ranking stimuli independent from RoboRewardBench evaluation
cohorts.  Its model-facing JSONL samples intentionally omit reward labels;
labels are recovered only by :func:`score` after all conditions are generated.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from ..attention_eval.masking import (
    Head,
    bbox_to_token_positions,
    select_low_ranked_heads,
)
from ..attention_eval.ranking import consensus_ranking
from ..attention_eval.runtime import find_contiguous_spans
from ..attention_eval.dataset import prepare_grounded_ranking_samples
from ..attention_eval.stats import (
    exact_mcnemar_pvalue,
    paired_cluster_bootstrap,
    paired_sign_flip_pvalue,
)
from ..data import load_episodes
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    deterministic_merge,
    object_fingerprint,
    provenance,
    read_jsonl,
    sha256_file,
    stable_shard,
    write_json,
    write_jsonl,
)
from ..protocol import progress_to_reward
from ..schemas import SCHEMA_VERSION
from ..video import extract_uniform_image_sequence
from .attention import (
    QwenAttentionRuntime,
    _native_video_message,
    _spatial_merge_size,
    build_forward_image_spans,
    prepare_native_video_processor_input,
)
from .protocols import (
    DISCRETE_PROTOCOLS,
    IMAGE_SEQUENCE_PROTOCOLS,
    ROBO_DOPAMINE_FORWARD,
    ROBOREWARDBENCH_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_NATIVE,
    dopamine_forward_messages,
    image_sequence_messages,
    interleaved_image_sequence_messages,
    validate_protocol,
)


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("attention_steer")
    if not isinstance(value, dict):
        raise ValueError("Expected an attention_steer configuration section")
    return value


def _ids(path: str | Path) -> list[str]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("example_ids_file must contain a JSON string list")
    if len(values) != len(set(values)):
        raise ValueError("example_ids_file contains duplicate IDs")
    return values


def _latest_endpoints(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        example_id, frame = row.get("example_id"), row.get("frame")
        if isinstance(example_id, str) and frame in {"first", "last"}:
            latest[(example_id, frame)] = row
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (example_id, frame), row in latest.items():
        if row.get("status") == "ok":
            grouped[example_id][frame] = row
    return grouped


def _progressive_indices(length: int, count: int) -> list[int]:
    """Match ``rank_heads_by_bbox.py --num-samples`` exactly."""
    if length < 1 or count < 1:
        raise ValueError("progressive ranking requires non-empty samples and count >= 1")
    step = max(1, length // count)
    return list(range(0, length, step))[:count]


def _frame_index(path: str | Path) -> int:
    match = re.fullmatch(r"frame_(\d+)\.[^.]+", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot infer aligned source frame index from {path}")
    return int(match.group(1))


def _probe_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe did not report a frame count for {path}") from exc


def _ensure_prefix_clip(source: Path, output: Path, terminal_frame: int) -> Path:
    """Create a deterministic prefix whose last decoded frame is the target."""
    expected = terminal_frame + 1
    if output.is_file() and _probe_frame_count(output) == expected:
        return output
    source_count = _probe_frame_count(source)
    if not 0 <= terminal_frame < source_count:
        raise ValueError(
            f"Terminal frame {terminal_frame} is outside {source} ({source_count} frames)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + f".{os.getpid()}.tmp.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-frames:v",
                str(expected),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv420p",
                str(temporary),
            ],
            check=True,
        )
        actual = _probe_frame_count(temporary)
        if actual != expected:
            raise RuntimeError(
                f"Prefix clip has {actual} frames; expected exactly {expected}"
            )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def build_aligned_ranking_manifest(config: dict[str, Any]) -> Path:
    """Build GRM-matched progressive stimuli for three aligned trajectories."""
    attention = _section(config)
    output = Path(attention["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = str(attention.get("protocol", ""))
    sample_count = int(attention.get("ranking_num_samples", 12))
    sources = attention.get("aligned_ranking_sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("aligned_ranking_sources must contain carrot, bottle, and cube")
    rows = []
    seen = set()
    hashes: dict[str, str] = {}

    def digest(path: str | Path) -> str:
        resolved = str(Path(path).resolve())
        if resolved not in hashes:
            hashes[resolved] = sha256_file(resolved)
        return hashes[resolved]

    for source in sources:
        name = str(source["name"])
        if name in seen:
            raise ValueError(f"Duplicate aligned ranking source {name!r}")
        seen.add(name)
        sequence = json.loads(Path(source["bbox_sequence_path"]).read_text(encoding="utf-8"))
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"No bbox sequence for ranking source {name}")
        sample_json_path = source.get("sample_json_path")
        if sample_json_path is not None:
            sample_json_path = str(Path(sample_json_path).resolve())
            progressive = json.loads(Path(sample_json_path).read_text(encoding="utf-8"))
            if not isinstance(progressive, list) or len(progressive) != len(sequence):
                raise ValueError(
                    f"Sample/bbox sequence length mismatch for {name}: "
                    f"{len(progressive) if isinstance(progressive, list) else 'invalid'} "
                    f"vs {len(sequence)}"
                )
            source_video = Path(source["video_path"]).resolve()
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            for original_index in _progressive_indices(len(progressive), sample_count):
                sample = progressive[original_index]
                bbox_row = sequence[original_index]
                images = sample.get("image")
                bbox = (bbox_row.get("chosen") or {}).get("bbox")
                if not isinstance(images, list) or len(images) != 8:
                    raise ValueError(f"Aligned sample {name}[{original_index}] is not eight-image")
                image_paths = [str(Path(path).resolve()) for path in images]
                missing = [path for path in image_paths if not Path(path).is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing aligned images: {missing}")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Aligned bbox missing for {name}[{original_index}]")
                bbox_image = str(Path(bbox_row["image"]).resolve())
                if bbox_image != image_paths[5]:
                    raise ValueError(
                        f"after_cam_high/bbox mismatch for {name}[{original_index}]"
                    )
                terminal_frame = _frame_index(image_paths[5])
                video_path = str(source_video)
                media_representation = "exact_grm_eight_image_sample"
                if protocol == ROBOREWARDBENCH_NATIVE:
                    clip_root = Path(
                        attention.get("ranking_clip_dir", output / "ranking_clips")
                    ).resolve()
                    clip = (
                        clip_root
                        / name
                        / f"through_frame_{terminal_frame:06d}.mp4"
                    )
                    video_path = str(
                        _ensure_prefix_clip(source_video, clip, terminal_frame)
                    )
                    media_representation = "source_video_prefix_through_matched_endpoint"
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": f"aligned_ranking/{name}/{original_index:04d}",
                    "ranking_source": name,
                    "ranking_source_sample_index": original_index,
                    "grm_sample_id": sample.get("id"),
                    "task": str(source["task"]),
                    "video_path": video_path,
                    "source_video_path": str(source_video),
                    "first_image_path": image_paths[0],
                    "last_image_path": image_paths[5],
                    "image_paths": image_paths,
                    "last_bbox": [float(value) for value in bbox],
                    "bbox_sequence_path": str(
                        Path(source["bbox_sequence_path"]).resolve()
                    ),
                    "bbox_sequence_index": original_index,
                    "target_source_frame_index": terminal_frame,
                    "sample_json_path": sample_json_path,
                    "media_representation": media_representation,
                    "source_video_sha256": digest(source_video),
                    "video_sha256": digest(video_path),
                    "image_sha256": [digest(path) for path in image_paths],
                }
                row["sample_fingerprint"] = object_fingerprint(row)
                rows.append(row)
            continue
        # Backward-compatible single-endpoint construction for old manifests.
        first, last = sequence[0], sequence[-1]
        bbox = (last.get("chosen") or {}).get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Last aligned ranking bbox missing for {name}")
        first_image = str(Path(first["image"]).resolve())
        last_image = str(Path(last["image"]).resolve())
        video_path = str(Path(source["video_path"]).resolve())
        for path in (first_image, last_image, video_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "example_id": f"aligned_ranking/{name}",
                "ranking_source": name,
                "task": str(source["task"]),
                "video_path": video_path,
                "first_image_path": first_image,
                "last_image_path": last_image,
                "last_bbox": [float(value) for value in bbox],
                "bbox_sequence_path": str(Path(source["bbox_sequence_path"]).resolve()),
                "bbox_sequence_index": int(last.get("index", len(sequence) - 1)),
            }
        )
    path = output / "aligned_ranking_samples.jsonl"
    write_jsonl(path, rows)
    write_json(
        output / "aligned_ranking_manifest.json",
        {
            "sample_count": len(rows),
            "sources": [row["ranking_source"] for row in rows],
            "protocol": protocol or None,
            "ranking_num_samples_per_source": sample_count,
            "method": (
                "grm_identical_progressive_index_selection"
                if all(source.get("sample_json_path") for source in sources)
                else "legacy_final_annotated_endpoint_per_trajectory"
            ),
            "fingerprint": object_fingerprint(rows),
        },
    )
    return path


def build_ranking_manifest(config: dict[str, Any]) -> Path:
    attention = _section(config)
    if attention.get("ranking_grounding_run"):
        output = Path(attention["output_dir"]).resolve()
        return prepare_grounded_ranking_samples(
            attention, output / "grounded_ranking_inputs.jsonl"
        )
    if str(attention.get("protocol")) in IMAGE_SEQUENCE_PROTOCOLS:
        raise ValueError(
            "Image-sequence ranking requires ranking_grounding_run so sampled "
            "source-frame indices and tracking boxes remain aligned"
        )
    return build_aligned_ranking_manifest(config)


def build_cohort_manifest(config: dict[str, Any]) -> Path:
    """Materialize audited or explicitly auto-grounded label-free inputs."""
    attention = _section(config)
    output = Path(attention["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested = _ids(attention["example_ids_file"])
    grounding = Path(attention["grounding_run"]).resolve()
    eligibility_mode = str(attention.get("eligibility_mode", "audited"))
    endpoints = _latest_endpoints(grounding / "grounding.jsonl")
    audit_path: Path | None = None
    if eligibility_mode == "audited":
        audit_path = grounding / "audit_final.jsonl"
        formal = {
            row["example_id"]
            for row in read_jsonl(audit_path)
            if row.get("formal_eligible") is True and isinstance(row.get("example_id"), str)
        }
    elif eligibility_mode == "auto_valid_grounding":
        formal = set(endpoints)
    else:
        raise ValueError("Unknown attention eligibility_mode")
    missing_audit = set(requested) - formal
    if missing_audit:
        raise ValueError(
            "Frozen cohort contains ineligible grounding examples: "
            f"{sorted(missing_audit)[:5]}"
        )
    episodes = {
        row.example_id: row
        for row in load_episodes(attention["dataset_root"], attention.get("split", "test"))
    }
    rows = []
    for example_id in requested:
        episode = episodes.get(example_id)
        endpoint = endpoints.get(example_id, {})
        if episode is None or {"first", "last"} - endpoint.keys():
            raise ValueError(f"Missing dataset or endpoint grounding for {example_id}")
        first, last = endpoint["first"], endpoint["last"]
        first_path = first.get("provenance", {}).get("image_path")
        last_path = last.get("provenance", {}).get("image_path")
        first_bbox = first.get("bbox")
        bbox = last.get("bbox")
        tracking_path = last.get("provenance", {}).get("tracking_path")
        if not isinstance(first_path, str) or not isinstance(last_path, str):
            raise ValueError(f"Missing endpoint images for {example_id}")
        if not isinstance(first_bbox, list) or len(first_bbox) != 4:
            raise ValueError(f"Missing first-frame bbox for {example_id}")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Missing endpoint bbox for {example_id}")
        if not isinstance(tracking_path, str) or not Path(tracking_path).is_file():
            raise ValueError(f"Missing tracking path for {example_id}")
        row = {
            "schema_version": SCHEMA_VERSION,
            "example_id": example_id,
            "video_sha256": episode.video_sha256,
            "subset": episode.subset,
            "task": episode.task,
            "video_path": episode.video_path,
            "first_image_path": first_path,
            "last_image_path": last_path,
            "first_bbox": [float(value) for value in first_bbox],
            "last_bbox": [float(value) for value in bbox],
            "tracking_path": str(Path(tracking_path).resolve()),
            "grounding_fingerprint": last.get("grounding_fingerprint"),
        }
        views = last.get("provenance", {}).get("view_endpoint_paths", {})
        if (
            str(attention["protocol"]) == ROBO_DOPAMINE_FORWARD
            and {"front", "left_wrist", "right_wrist"} <= set(views)
        ):
            blank = str(Path(attention["blank_goal"]).resolve())
            row["image_paths"] = [
                views["front"]["first"],
                blank,
                views["front"]["first"],
                views["left_wrist"]["first"],
                views["right_wrist"]["first"],
                views["front"]["last"],
                views["left_wrist"]["last"],
                views["right_wrist"]["last"],
            ]
        elif str(attention["protocol"]) in IMAGE_SEQUENCE_PROTOCOLS:
            image_paths, sampling_record = extract_uniform_image_sequence(
                episode.video_path,
                output / "image_sequences" / "cohort" / episode.video_sha256,
                count=int(attention.get("num_images", 8)),
            )
            row["image_paths"] = image_paths
            row["image_source_indices"] = sampling_record[
                "selected_source_indices"
            ]
            row["image_sampling_record"] = sampling_record
        rows.append(row)
    path = output / "cohort_inputs.jsonl"
    write_jsonl(path, rows)
    write_json(
        output / "cohort_manifest.json",
        {
            "expected_count": len(requested),
            "model_facing_labels_omitted": True,
            "example_ids_file": str(Path(attention["example_ids_file"]).resolve()),
            "example_ids_sha256": sha256_file(attention["example_ids_file"]),
            "eligibility_mode": eligibility_mode,
            "grounding_audit": str(audit_path) if audit_path is not None else None,
            "automatic_tracking_assumed_correct": eligibility_mode == "auto_valid_grounding",
            "input_representation": (
                "uniform_interleaved_images_v1"
                if str(attention["protocol"])
                == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE
                else "uniform_independent_images_v1"
                if str(attention["protocol"]) == ROBOREWARDBENCH_IMAGE_SEQUENCE
                else None
            ),
            "num_images": attention.get("num_images"),
            "fingerprint": object_fingerprint(rows),
        },
    )
    return path


def validate_ranking_inputs(config: dict[str, Any]) -> Path:
    """Run the real processor contract and verify every target-token alignment."""
    from transformers import AutoConfig, AutoProcessor

    attention = _section(config)
    protocol = validate_protocol(str(attention["protocol"]))
    output = Path(attention["output_dir"]).resolve()
    samples_path = build_ranking_manifest(config)
    samples = list(read_jsonl(samples_path))
    processor = AutoProcessor.from_pretrained(
        attention["model_path"], trust_remote_code=True
    )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if "min_pixels" in attention:
            image_processor.min_pixels = int(attention["min_pixels"])
        if "max_pixels" in attention:
            image_processor.max_pixels = int(attention["max_pixels"])
    if protocol == ROBOREWARDBENCH_NATIVE:
        cap = attention.get("attention_video_max_frames")
        if cap is not None:
            processor.video_processor.max_frames = int(cap)
    model_config = AutoConfig.from_pretrained(
        attention["model_path"], trust_remote_code=True
    )
    merge_size = _spatial_merge_size(model_config)
    records = []
    for sample in samples:
        video_metadata = None
        if protocol == ROBO_DOPAMINE_FORWARD:
            paths = [str(Path(path).resolve()) for path in sample["image_paths"]]
            raw = processor.apply_chat_template(
                dopamine_forward_messages(
                    {
                        "task": sample["task"],
                        "prompt_mode": str(attention.get("prompt_mode", "official")),
                        "image": paths,
                    }
                ),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = raw["input_ids"][0].tolist()
            token_id = int(getattr(model_config, "image_token_id", 151655))
            token_spans = find_contiguous_spans(ids, token_id)
            grids = [
                tuple(int(value) for value in row)
                for row in raw["image_grid_thw"].tolist()
            ]
            spans = build_forward_image_spans(paths, token_spans, grids)
            target = next(span for span in spans if span.label == "after_cam_high")
            if target.path != str(Path(sample["last_image_path"]).resolve()):
                raise RuntimeError(
                    f"Processor target path mismatch for {sample['example_id']}"
                )
        elif protocol in IMAGE_SEQUENCE_PROTOCOLS:
            paths = [str(Path(path).resolve()) for path in sample["image_paths"]]
            messages = (
                interleaved_image_sequence_messages(sample["task"], paths)
                if protocol == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE
                else image_sequence_messages(
                    sample["task"],
                    paths,
                    content_order=str(
                        attention.get("content_order", "text_then_images")
                    ),
                )
            )
            raw = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = raw["input_ids"][0].tolist()
            token_id = int(getattr(model_config, "image_token_id", 151655))
            token_spans = find_contiguous_spans(ids, token_id)
            grids = raw.get("image_grid_thw")
            grid_count = int(grids.shape[0]) if grids is not None else 0
            if len(token_spans) != len(paths) or grid_count != len(paths):
                raise RuntimeError(
                    f"Image-sequence span mismatch for {sample['example_id']}: "
                    f"images={len(paths)}, spans={len(token_spans)}, grids={grid_count}"
                )
            from ..attention_eval.masking import ImageSpan

            spans = [
                ImageSpan(
                    f"image_t{index}",
                    path,
                    int(start),
                    int(end),
                    tuple(int(value) for value in grid),
                )
                for index, (path, (start, end), grid) in enumerate(
                    zip(paths, token_spans, grids.tolist())
                )
            ]
            target = spans[-1]
            source_indices = [
                int(value) for value in sample["image_source_indices"]
            ]
            sampling_record = dict(sample["image_sampling_record"])
            if (
                len(source_indices) != len(spans)
                or source_indices[-1]
                != int(sampling_record["terminal_source_index"])
            ):
                raise RuntimeError(
                    f"Terminal image alignment failed for {sample['example_id']}"
                )
            video_metadata = {
                **sampling_record,
                "span_source_frame_indices": [
                    [value] for value in source_indices
                ],
                "target_source_frame_indices": [source_indices[-1]],
                "independent_image_spans": True,
            }
        else:
            raw, temporal_record = prepare_native_video_processor_input(
                processor,
                sample["task"],
                sample["video_path"],
                content_order=str(attention.get("content_order", "text_then_video")),
                temporal_span_mode=str(
                    attention.get("temporal_span_mode", "native_pairs")
                ),
            )
            metadata = raw["video_metadata"]
            metadata = metadata[0] if isinstance(metadata, (list, tuple)) else metadata
            ids_value = raw["input_ids"]
            ids = ids_value[0].tolist() if hasattr(ids_value[0], "tolist") else list(ids_value[0])
            token_id = int(getattr(model_config, "video_token_id", 151656))
            token_spans = find_contiguous_spans(ids, token_id)
            temporal, height, width = (
                int(value) for value in raw["video_grid_thw"][0].tolist()
            )
            if len(token_spans) != temporal:
                raise RuntimeError(
                    f"Native temporal-span mismatch for {sample['example_id']}: "
                    f"{len(token_spans)} vs {temporal}"
                )
            from ..attention_eval.masking import ImageSpan

            spans = [
                ImageSpan(
                    f"video_t{index}",
                    sample["video_path"],
                    start,
                    end,
                    (1, height, width),
                )
                for index, (start, end) in enumerate(token_spans)
            ]
            target = spans[-1]
            frame_indices_value = getattr(metadata, "frames_indices", None)
            frame_indices = (
                frame_indices_value.tolist()
                if hasattr(frame_indices_value, "tolist")
                else list(frame_indices_value or [])
            )
            total_frames = int(getattr(metadata, "total_num_frames", 0))
            if not frame_indices or int(frame_indices[-1]) != total_frames - 1:
                raise RuntimeError(
                    f"Native processor omitted terminal frame for {sample['example_id']}"
                )
            frames_per_span = (
                len(frame_indices) // temporal
                if temporal and len(frame_indices) % temporal == 0
                else None
            )
            video_metadata = {
                **temporal_record,
                "total_num_frames": total_frames,
                "frames_indices": frame_indices,
                "temporal_grid": temporal,
                "spatial_grid": [height, width],
                "span_source_frame_indices": (
                    [
                        frame_indices[
                            index * frames_per_span : (index + 1) * frames_per_span
                        ]
                        for index in range(temporal)
                    ]
                    if frames_per_span
                    else [None] * (temporal - 1) + [[frame_indices[-1]]]
                ),
                "target_source_frame_indices": (
                    frame_indices[-frames_per_span:]
                    if frames_per_span
                    else [frame_indices[-1]]
                ),
                "terminal_frame_in_target_span": True,
            }
        with Image.open(target.path) as image:
            image_size = image.size
        target_positions = bbox_to_token_positions(
            target,
            sample["last_bbox"],
            image_size,
            merge_size,
        )
        if not target_positions or any(
            position < target.start or position >= target.end
            for position in target_positions
        ):
            raise RuntimeError(
                f"Target bbox escaped its processor span for {sample['example_id']}"
            )
        records.append(
            {
                "example_id": sample["example_id"],
                "ranking_source": sample["ranking_source"],
                "protocol": protocol,
                "input_token_count": len(ids),
                "visual_span_count": len(spans),
                "target_span": target.__dict__,
                "target_bbox": sample["last_bbox"],
                "target_token_count": len(target_positions),
                "target_positions_within_target_span": True,
                "video_metadata": video_metadata,
                "status": "ok",
            }
        )
    path = output / "processor_alignment_diagnostics.jsonl"
    write_jsonl(path, records)
    write_json(
        output / "processor_alignment_manifest.json",
        {
            "protocol": protocol,
            "sample_count": len(records),
            "all_valid": len(records) == len(samples),
            "processor_model_path": str(Path(attention["model_path"]).resolve()),
            "attention_video_max_frames": attention.get("attention_video_max_frames"),
            "num_images": attention.get("num_images"),
            "temporal_span_mode": attention.get(
                "temporal_span_mode", "native_pairs"
            ),
            "temporal_bbox_reduce": attention.get(
                "temporal_bbox_reduce", "last"
            ),
            "fingerprint": object_fingerprint(records),
        },
    )
    return path


RANKING_SCORE_KINDS = ("raw_mass", "excess_mass", "visual_enrichment")


def _ranking_artifact(
    mass_rows: list[dict[str, Any]],
    runtime: QwenAttentionRuntime,
    source: str,
    score_kind: str,
) -> dict[str, Any]:
    if score_kind not in RANKING_SCORE_KINDS:
        raise ValueError(f"Unknown ranking_score_kind {score_kind!r}")
    arrays = {
        kind: np.asarray([row[kind] for row in mass_rows], dtype=np.float64)
        for kind in RANKING_SCORE_KINDS
    }
    expected = (len(mass_rows), runtime.num_layers, runtime.num_heads)
    if any(value.shape != expected for value in arrays.values()):
        raise ValueError(
            f"Unexpected attention shapes "
            f"{ {kind: value.shape for kind, value in arrays.items()} }"
        )
    aggregated = {kind: value.mean(axis=0) for kind, value in arrays.items()}
    rows = [
        {
            "layer": layer,
            "head": head,
            "score": float(aggregated[score_kind][layer, head]),
            "mean_raw_mass": float(aggregated["raw_mass"][layer, head]),
            "mean_excess_mass": float(aggregated["excess_mass"][layer, head]),
            "mean_visual_enrichment": float(
                aggregated["visual_enrichment"][layer, head]
            ),
        }
        for layer in range(runtime.num_layers)
        for head in range(runtime.num_heads)
    ]
    rows.sort(
        key=lambda row: (
            -row["score"],
            -row["mean_raw_mass"],
            row["layer"],
            row["head"],
        )
    )
    return {
        "ranking_source": "aligned_success_progressive_trajectory",
        "aligned_source": source,
        "method": f"last_prompt_bbox_{score_kind}_arithmetic_mean",
        "score_kind": score_kind,
        "sample_count": len(mass_rows),
        "sample_ids": [row["example_id"] for row in mass_rows],
        "num_layers": runtime.num_layers,
        "num_heads": runtime.num_heads,
        "rankings": {"mean": rows},
        "fingerprint": object_fingerprint(rows),
    }


def rank(config: dict[str, Any], *, retry_failed: bool = False) -> Path:
    attention = _section(config)
    output = Path(attention["output_dir"]).resolve()
    samples_path = build_ranking_manifest(config)
    samples = list(read_jsonl(samples_path))
    records_path = output / "aligned_ranking_mass.jsonl"
    previous = {row.get("example_id"): row for row in read_jsonl(records_path)} if records_path.exists() else {}
    runtime = QwenAttentionRuntime(attention)
    score_kind = str(attention.get("ranking_score_kind", "raw_mass"))
    if score_kind not in RANKING_SCORE_KINDS:
        raise ValueError(
            f"ranking_score_kind must be one of {', '.join(RANKING_SCORE_KINDS)}"
        )
    successful: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        old = previous.get(sample["example_id"])
        if (
            old
            and old.get("status") == "ok"
            and old.get("sample_fingerprint") == sample.get("sample_fingerprint")
        ):
            row = old
        elif old and not retry_failed:
            raise RuntimeError(
                f"Previous ranking attempt failed for {sample['example_id']}; "
                "rerun with --retry-failed after inspecting aligned_ranking_mass.jsonl"
            )
        else:
            try:
                row = {**sample, **runtime.collect_mass(sample)}
            except Exception as exc:
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": sample["example_id"],
                    "ranking_source": sample["ranking_source"],
                    "status": "invalid",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            append_jsonl(records_path, row)
        if row.get("status") != "ok":
            raise RuntimeError(f"Ranking failed for {sample['example_id']}: {row.get('error')}")
        successful[str(sample["ranking_source"])].append(row)
    source_names = (
        ["mydata_ranking_data"]
        if attention.get("ranking_grounding_run")
        else [str(source["name"]) for source in attention["aligned_ranking_sources"]]
    )
    if set(successful) != set(source_names):
        raise RuntimeError("Ranking did not produce all configured aligned sources")
    artifact_paths: dict[str, list[Path]] = {kind: [] for kind in RANKING_SCORE_KINDS}
    for source in source_names:
        for kind in RANKING_SCORE_KINDS:
            artifact = _ranking_artifact(successful[source], runtime, source, kind)
            directory = output / "aligned_rankings" / source
            metric_path = directory / f"head_ranking_{kind}.json"
            write_json(metric_path, artifact)
            artifact_paths[kind].append(metric_path)
            if kind == score_kind:
                write_json(directory / "head_ranking.json", artifact)
    consensuses = {}
    for kind in RANKING_SCORE_KINDS:
        value = consensus_ranking(
            artifact_paths[kind],
            expected_layers=runtime.num_layers,
            expected_heads=runtime.num_heads,
            skip_early_layers=int(attention.get("skip_early_layers", 2)),
        )
        value.update(
            {
                "ranking_protocol": str(attention["protocol"]),
                "ranking_score_kind": kind,
                "ranking_method_detail": (
                    "three_trajectory_normalized_borda_of_per_trajectory_"
                    "progressive_sample_arithmetic_means"
                ),
                "aligned_manifest": str(samples_path),
                "attention_video_max_frames": attention.get(
                    "attention_video_max_frames"
                ),
                "num_images": attention.get("num_images"),
                "temporal_span_mode": attention.get(
                    "temporal_span_mode", "native_pairs"
                ),
                "temporal_bbox_reduce": attention.get(
                    "temporal_bbox_reduce", "last"
                ),
            }
        )
        write_json(output / f"consensus_ranking_{kind}.json", value)
        consensuses[kind] = value
    consensus = consensuses[score_kind]
    path = output / "consensus_ranking.json"
    write_json(path, consensus)
    return path


def _load_ranking(path: Path, top_k: int) -> tuple[list[Head], list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("ranking")
    if not isinstance(rows, list):
        raise ValueError(f"No consensus ranking list in {path}")
    heads = [Head(int(row["layer"]), int(row["head"])) for row in rows[:top_k]]
    if len(heads) != top_k:
        raise ValueError("Ranking has fewer heads than top_k")
    return heads, rows, data


def _steering_condition_kinds(attention: dict[str, Any]) -> tuple[str, ...]:
    """Return target plus optional negative controls under a stable contract."""
    allowed_controls = {
        "candidate_wrong",
        "candidate_target_only",
        "low_rank_target",
    }
    configured = attention.get(
        "control_conditions", ["candidate_wrong", "low_rank_target"]
    )
    if not isinstance(configured, list) or any(
        str(value) not in allowed_controls for value in configured
    ):
        raise ValueError(
            "control_conditions must contain candidate_wrong, "
            "candidate_target_only, and/or low_rank_target"
        )
    controls = tuple(
        name
        for name in ("candidate_wrong", "candidate_target_only", "low_rank_target")
        if name in {str(value) for value in configured}
    )
    return ("candidate_target", *controls)


def _steering_biases(
    attention: dict[str, Any], top_values: Sequence[int]
) -> dict[int, float]:
    """Resolve an optional K-aware dose map while preserving scalar defaults."""
    default = float(attention.get("swap_bias", 6))
    configured = attention.get("swap_bias_by_top_k", {})
    if not isinstance(configured, dict):
        raise ValueError("swap_bias_by_top_k must be a mapping")
    result = {}
    for top_k in top_values:
        value = configured.get(top_k, configured.get(str(top_k), default))
        result[top_k] = float(value)
        if not math.isfinite(result[top_k]) or result[top_k] < 0:
            raise ValueError("Steering biases must be finite and non-negative")
    return result


def _record(
    sample: dict[str, Any],
    condition: str,
    heads: Iterable[Head],
    bias: float,
    result: dict[str, Any],
    ranking_fingerprint: str,
    control_region: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": sample["example_id"],
        "video_sha256": sample["video_sha256"],
        "grounding_fingerprint": sample.get("grounding_fingerprint"),
        "subset": sample["subset"],
        "condition": condition,
        "heads": [{"layer": head.layer, "head": head.head} for head in heads],
        "bias": float(bias),
        "ranking_fingerprint": ranking_fingerprint,
        "control_region": control_region,
        "status": "ok",
        **result,
    }


def _completed_conditions(
    records_path: Path, *, ranking_fingerprint: str
) -> dict[tuple[str, str], set[str]]:
    done: dict[tuple[str, str], set[str]] = defaultdict(set)
    if not records_path.exists():
        return done
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(records_path):
        example_id = row.get("example_id")
        grounding_fingerprint = row.get("grounding_fingerprint")
        condition = row.get("condition")
        if not isinstance(example_id, str) or not example_id:
            continue
        if not isinstance(grounding_fingerprint, str) or not grounding_fingerprint:
            continue
        if row.get("ranking_fingerprint") != ranking_fingerprint:
            continue
        if not isinstance(condition, str) or not condition:
            continue
        latest[(example_id, grounding_fingerprint, condition)] = row
    for (example_id, grounding_fingerprint, condition), row in latest.items():
        if row.get("status") == "ok":
            done[(example_id, grounding_fingerprint)].add(condition)
    return done


def steer(config: dict[str, Any], *, retry_failed: bool = False) -> Path:
    attention = _section(config)
    output = Path(attention["output_dir"]).resolve()
    samples = list(read_jsonl(build_cohort_manifest(config)))
    shard_id = int(attention.get("shard_id", 0))
    num_shards = int(attention.get("num_shards", 1))
    if num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    samples = [
        sample
        for sample in samples
        if stable_shard(str(sample["video_sha256"]), num_shards) == shard_id
    ]
    ranking_path = Path(attention.get("ranking_path", output / "consensus_ranking.json")).resolve()
    top_values = sorted(
        {int(value) for value in attention.get("top_k_values", [8, 32, 64])}
    )
    if not top_values or top_values[0] < 1:
        raise ValueError("top_k_values must contain positive integers")
    _largest, ranking_rows, ranking_data = _load_ranking(ranking_path, max(top_values))
    ranking_fingerprint = str(ranking_data["fingerprint"])
    head_sets = {}
    for top_k in top_values:
        candidate = [
            Head(int(row["layer"]), int(row["head"]))
            for row in ranking_rows[:top_k]
        ]
        head_sets[top_k] = (
            candidate,
            select_low_ranked_heads(ranking_rows, top_k, candidate),
        )
    bias_by_top_k = _steering_biases(attention, top_values)
    scope = str(attention.get("steering_query_scope", "last_prompt"))
    negative_scope = str(attention.get("negative_scope", "other_spans"))
    condition_kinds = _steering_condition_kinds(attention)
    condition_names = [
        f"{kind}_k{top_k}"
        for top_k in top_values
        for kind in condition_kinds
    ]
    records_path = (
        output / "steering.jsonl"
        if num_shards == 1
        else output / f"steering.shard-{shard_id:02d}.jsonl"
    )
    done = _completed_conditions(records_path, ranking_fingerprint=ranking_fingerprint)
    runtime = None
    for sample in samples:
        grounding_fingerprint = sample.get("grounding_fingerprint")
        if not isinstance(grounding_fingerprint, str) or not grounding_fingerprint:
            raise ValueError(
                f"Missing grounding_fingerprint for {sample.get('example_id')}"
            )
        completion_key = (str(sample["example_id"]), grounding_fingerprint)
        required = {"baseline", *condition_names}
        if required <= done[completion_key]:
            continue
        try:
            if runtime is None:
                runtime = QwenAttentionRuntime(attention)
            prepared = runtime.prepare(sample)
            target = runtime.target_positions(sample, prepared)
            wrong, wrong_mode = runtime.wrong_control_positions(prepared, target)
            target_only = (
                runtime.target_positions(sample, prepared, include_reference=False)
                if "candidate_target_only" in condition_kinds
                else []
            )
            specs = [("baseline", (), 0.0, target, "target_reference")]
            for top_k in top_values:
                candidate, low = head_sets[top_k]
                bias = bias_by_top_k[top_k]
                specs.append(
                    (f"candidate_target_k{top_k}", candidate, bias, target, "target_reference")
                )
                if "candidate_wrong" in condition_kinds:
                    specs.append(
                        (f"candidate_wrong_k{top_k}", candidate, bias, wrong, wrong_mode)
                    )
                if "candidate_target_only" in condition_kinds:
                    specs.append(
                        (
                            f"candidate_target_only_k{top_k}",
                            candidate,
                            bias,
                            target_only,
                            "target_only",
                        )
                    )
                if "low_rank_target" in condition_kinds:
                    specs.append(
                        (f"low_rank_target_k{top_k}", low, bias, target, "target_reference")
                    )
            for condition, heads, magnitude, positions, control_region in specs:
                if condition in done[completion_key]:
                    continue
                result = runtime.generate(
                    sample,
                    prepared=prepared,
                    heads=heads,
                    selected_positions=positions,
                    visual_positions=prepared.visual_positions,
                    bias=magnitude,
                    query_scope=scope,
                    negative_scope=negative_scope,
                )
                append_jsonl(
                    records_path,
                    _record(
                        sample,
                        condition,
                        heads,
                        magnitude,
                        result,
                        ranking_fingerprint,
                        control_region,
                    ),
                )
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": sample["example_id"],
                    "video_sha256": sample["video_sha256"],
                    "condition": "sample_failure",
                    "status": "invalid",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            if not retry_failed:
                continue
    manifest = {
            **provenance(sys.argv, config, Path(__file__).resolve().parents[2]),
            "model_fingerprint": artifact_fingerprint(attention["model_path"]),
            "ranking_path": str(ranking_path),
            "ranking_sha256": sha256_file(ranking_path),
            "protocol": attention["protocol"],
            "attention_video_max_frames": attention.get("attention_video_max_frames"),
            "num_images": attention.get("num_images"),
            "content_order": attention.get("content_order", "text_then_video"),
            "temporal_intervention_scope": attention.get(
                "temporal_intervention_scope", "last_frame"
            ),
            "temporal_span_mode": attention.get(
                "temporal_span_mode", "native_pairs"
            ),
            "temporal_bbox_reduce": attention.get(
                "temporal_bbox_reduce", "last"
            ),
            "steering_query_scope": scope,
            "negative_scope": negative_scope,
            "spatial_target_mode": attention.get(
                "spatial_target_mode", "target_only"
            ),
            "reference_grounding_path": attention.get(
                "reference_grounding_path"
            ),
            "reference_grounding_sha256": (
                sha256_file(attention["reference_grounding_path"])
                if attention.get("reference_grounding_path")
                else None
            ),
            "swap_bias_by_top_k": {
                str(top_k): bias_by_top_k[top_k] for top_k in top_values
            },
            "control_conditions": list(condition_kinds[1:]),
            "record_discrete_label_logprobs": bool(
                attention.get("record_discrete_label_logprobs", False)
            ),
            "constrain_discrete_output": bool(
                attention.get("constrain_discrete_output", False)
            ),
            "top_k_values": top_values,
            "conditions": ["baseline", *condition_names],
            "labels_model_facing": False,
            "shard_id": shard_id,
            "num_shards": num_shards,
        }
    manifest_path = (
        output / "steering_manifest.json"
        if num_shards == 1
        else output / f"steering_manifest.shard-{shard_id:02d}.json"
    )
    write_json(manifest_path, manifest)
    if num_shards > 1:
        shard_paths = [
            output / f"steering.shard-{index:02d}.jsonl"
            for index in range(num_shards)
        ]
        manifest_paths = [
            output / f"steering_manifest.shard-{index:02d}.json"
            for index in range(num_shards)
        ]
        if all(path.exists() for path in [*shard_paths, *manifest_paths]):
            deterministic_merge(shard_paths, output / "steering.jsonl")
            manifests = [json.loads(path.read_text()) for path in manifest_paths]
            write_json(
                output / "steering_manifest.json",
                {
                    **manifests[0],
                    "shards": [
                        {
                            "shard_id": item["shard_id"],
                            "config_fingerprint": item["config_fingerprint"],
                        }
                        for item in manifests
                    ],
                },
            )
    return records_path


def _summary(rows: list[dict[str, Any]], labels: dict[str, int], protocol: str) -> dict[str, Any]:
    valid = []
    for row in rows:
        example_id = row.get("example_id")
        if row.get("status") != "ok" or example_id not in labels:
            continue
        prediction = (
            int(row["native_prediction"])
            if protocol in DISCRETE_PROTOCOLS
            else progress_to_reward(float(row["progress"]))
        )
        valid.append((str(example_id), prediction, labels[str(example_id)]))
    if not valid:
        return {"n": 0}
    errors = [abs(prediction - label) for _, prediction, label in valid]
    return {
        "n": len(valid),
        "exact_accuracy": mean(error == 0 for error in errors),
        "within_one_accuracy": mean(error <= 1 for error in errors),
        "mae": mean(errors),
        "mean_signed_error": mean(prediction - label for _, prediction, label in valid),
        "prediction_counts": dict(sorted(Counter(prediction for _, prediction, _ in valid).items())),
    }


def score(config: dict[str, Any]) -> Path:
    """Join labels after inference and summarize all steering conditions."""
    attention = _section(config)
    output = Path(attention["output_dir"]).resolve()
    protocol = validate_protocol(str(attention["protocol"]))
    top_values = sorted(
        {int(value) for value in attention.get("top_k_values", [8, 32, 64])}
    )
    condition_kinds = _steering_condition_kinds(attention)
    top_conditions = [
        f"{kind}_k{top_k}"
        for top_k in top_values
        for kind in condition_kinds
    ]
    expected_ids = _ids(attention["example_ids_file"])
    expected_id_set = set(expected_ids)
    labels = {
        row.example_id: row.reward
        for row in load_episodes(attention["dataset_root"], attention.get("split", "test"))
        if row.example_id in expected_id_set
    }
    if set(labels) != expected_id_set:
        raise ValueError("Configured cohort IDs do not exactly resolve to dataset labels")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(output / "steering.jsonl"):
        key = (str(row.get("example_id")), str(row.get("condition")))
        latest[key] = row
    required_conditions = ("baseline", *top_conditions)
    succeeded_by_id = {
        example_id
        for example_id in expected_id_set
        if all(latest.get((example_id, condition), {}).get("status") == "ok" for condition in required_conditions)
    }
    invalid = [
        row for (example_id, _condition), row in latest.items()
        if row.get("status") != "ok" and example_id not in succeeded_by_id
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_example_id, condition), row in latest.items():
        if condition != "sample_failure":
            grouped[condition].append(row)
    summaries = {condition: _summary(rows, labels, protocol) for condition, rows in grouped.items()}
    baseline = {row["example_id"]: row for row in grouped.get("baseline", []) if row.get("status") == "ok"}
    paired = {}
    bootstrap_samples = int(attention.get("bootstrap_samples", 10_000))

    def continuous_value(row: dict[str, Any]) -> float:
        return (
            float(row["native_prediction"])
            if protocol in DISCRETE_PROTOCOLS
            else float(row["progress"])
        )

    def discrete_prediction(row: dict[str, Any]) -> int:
        return (
            int(row["native_prediction"])
            if protocol in DISCRETE_PROTOCOLS
            else progress_to_reward(float(row["progress"]))
        )

    for condition, rows in grouped.items():
        if condition == "baseline":
            continue
        current = {row["example_id"]: row for row in rows if row.get("status") == "ok"}
        shared = sorted(set(baseline) & set(current) & set(labels))
        paired_rows = []
        for example_id in shared:
            baseline_prediction = discrete_prediction(baseline[example_id])
            candidate_prediction = discrete_prediction(current[example_id])
            label = labels[example_id]
            paired_rows.append(
                {
                    "example_id": example_id,
                    "video_sha256": str(
                        baseline[example_id].get("video_sha256", example_id)
                    ),
                    "subset": baseline[example_id].get("subset"),
                    "continuous_delta": continuous_value(current[example_id])
                    - continuous_value(baseline[example_id]),
                    "prediction_delta": candidate_prediction - baseline_prediction,
                    "absolute_error_change": abs(candidate_prediction - label)
                    - abs(baseline_prediction - label),
                    "baseline_correct": baseline_prediction == label,
                    "candidate_correct": candidate_prediction == label,
                    "corrected": baseline_prediction != label
                    and candidate_prediction == label,
                    "harmed": baseline_prediction == label
                    and candidate_prediction != label,
                }
            )
        deltas = [row["continuous_delta"] for row in paired_rows]
        paired[condition] = {
            "n": len(shared),
            "mean_score_delta_vs_baseline": mean(deltas) if deltas else None,
            "increased_count": sum(delta > 0 for delta in deltas),
            "decreased_count": sum(delta < 0 for delta in deltas),
            "unchanged_count": sum(delta == 0 for delta in deltas),
            "corrected_count": sum(row["corrected"] for row in paired_rows),
            "harmed_count": sum(row["harmed"] for row in paired_rows),
            "cluster_estimands": {
                field: paired_cluster_bootstrap(
                    paired_rows, field, samples=bootstrap_samples
                )
                for field in (
                    "continuous_delta",
                    "prediction_delta",
                    "absolute_error_change",
                )
            },
            "continuous_delta_two_sided_cluster_sign_flip_pvalue": (
                paired_sign_flip_pvalue(
                    paired_rows,
                    "continuous_delta",
                    samples=bootstrap_samples,
                )
            ),
            "exact_mcnemar_pvalue_record_level": exact_mcnemar_pvalue(
                paired_rows, "baseline_correct", "candidate_correct"
            ),
        }
    condition_ids = {
        condition: {str(row.get("example_id")) for row in grouped.get(condition, [])}
        for condition in required_conditions
    }
    formal_scoring_ready = (
        not invalid
        and all(condition_ids[condition] == expected_id_set for condition in required_conditions)
        and all(
            summaries.get(condition, {}).get("n") == len(expected_ids)
            for condition in required_conditions
        )
    )
    result = {
        "protocol": protocol,
        "top_k_values": top_values,
        "by_top_k": {
            str(top_k): {
                "by_condition": {
                    kind: summaries.get(f"{kind}_k{top_k}", {})
                    for kind in condition_kinds
                },
                "paired_vs_baseline": {
                    kind: paired.get(f"{kind}_k{top_k}", {})
                    for kind in condition_kinds
                },
            }
            for top_k in top_values
        },
        "discrete_output": protocol in DISCRETE_PROTOCOLS,
        "official_native_discrete_output": protocol == ROBOREWARDBENCH_NATIVE,
        "adapter_metric": protocol != ROBOREWARDBENCH_NATIVE,
        "completion": {
            "expected_count": len(expected_ids),
            "condition_counts": {name: len(ids) for name, ids in condition_ids.items()},
            "invalid_count": len(invalid),
            "formal_scoring_ready": formal_scoring_ready,
        },
        "by_condition": summaries,
        "paired_vs_baseline": paired,
    }
    path = output / "steering_metrics.json"
    write_json(path, result)
    write_json(output / "steering_invalid.json", invalid)
    return path
