#!/usr/bin/env python3
"""Evaluate Robo-Dopamine on the robomimic can/lift/square datasets.

The datasets contain one external RGB video (``image``) and one wrist RGB
video (``wrist_image``) per episode. Robo-Dopamine expects three camera
streams. The ``image-wrist`` view mode maps the external camera to
``cam_high`` and uses the single wrist stream for both wrist inputs; the
``image-only`` mode fills all three model inputs with the external camera.

The source videos are AV1. The OpenCV build used by Robo-Dopamine can inspect
their containers but cannot decode their frames on this machine. This script
therefore samples frames with FFmpeg and passes image directories to
``GRMInference.run_pipeline``. Source datasets are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/mnt/public1/dais/workspace/data/xrz")
DEFAULT_MODEL = SCRIPT_DIR / "pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/robomimic_abs_eval_fixed16_newinstruction"
VALID_DATASETS = ("can", "lift", "square")
VALID_QUALITIES = ("better", "okay", "worse")
VALID_MODES = ("forward", "incremental", "backward")
VALID_VIEW_MODES = ("image-wrist", "image-only")
SCORE_RE = re.compile(r"<score>\s*[+-]?\d+(?:\.\d+)?%\s*</score>")


@dataclass(frozen=True)
class Episode:
    dataset: str
    root: Path
    index: int
    length: int
    quality: str
    task: str
    fps: int
    front_video: Path
    wrist_video: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Robo-Dopamine on LeRobot-v2.1-style robomimic absolute-action "
            "datasets. By default, one episode from each quality level is "
            "selected per dataset as a safe smoke evaluation."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Parent directory of robomimic_*_abs (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=VALID_DATASETS,
        default=list(VALID_DATASETS),
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Robo-Dopamine checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Evaluation output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        choices=VALID_QUALITIES,
        default=list(VALID_QUALITIES),
        help="Episode quality labels to include.",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        metavar="SPEC",
        help="Explicit episode IDs, e.g. '0,5,50-54'. Overrides stratified limiting.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=3,
        help="Maximum episodes per dataset after filtering; 0 means all (default: 3).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=VALID_MODES,
        default=list(VALID_MODES),
        help="Robo-Dopamine evaluation modes.",
    )
    parser.add_argument(
        "--view-mode",
        choices=VALID_VIEW_MODES,
        default="image-wrist",
        help=(
            "Camera mapping: image-wrist uses image plus duplicated wrist_image; "
            "image-only fills all three inputs with image (default: image-wrist)."
        ),
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        help=(
            "Sample every N original frames. Used when --fixed-samples is omitted; "
            "the default interval is 20."
        ),
    )
    sampling_group.add_argument(
        "--fixed-samples",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Uniformly sample exactly K comparisons (K+1 frames including the "
            "episode start and end), independent of episode length."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--goal-mode",
        choices=("reference-end", "episode-end", "blank"),
        default="blank",
        help=(
            "Goal anchor: final front frame of one better reference episode, "
            "each evaluated episode's final frame, or blank image."
        ),
    )
    parser.add_argument(
        "--reference-episode",
        type=int,
        default=None,
        help=(
            "Reference episode ID for reference-end; default is a median-length "
            "better episode."
        ),
    )
    parser.add_argument(
        "--goal-image",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Custom goal image for a dataset; repeat for multiple datasets.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Ask run_pipeline to render its per-mode reward video.",
    )
    parser.add_argument(
        "--keep-staged-frames",
        action="store_true",
        help="Keep FFmpeg-sampled PNGs under output-root/_staged for reuse.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed dataset/episode/mode/config combinations (default: true).",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata/files and print the evaluation plan without loading the model.",
    )
    args = parser.parse_args(argv)

    if args.max_episodes < 0:
        parser.error("--max-episodes must be >= 0")
    if args.frame_interval is None and args.fixed_samples is None:
        args.frame_interval = 20
    if args.frame_interval is not None and args.frame_interval <= 0:
        parser.error("--frame-interval must be > 0")
    if args.fixed_samples is not None and args.fixed_samples <= 0:
        parser.error("--fixed-samples must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    return args


def parse_episode_spec(spec: str | None) -> set[int] | None:
    if spec is None:
        return None
    result: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid episode range: {token!r}") from exc
            if start < 0 or end < start:
                raise ValueError(f"Invalid episode range: {token!r}")
            result.update(range(start, end + 1))
        else:
            try:
                episode = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid episode ID: {token!r}") from exc
            if episode < 0:
                raise ValueError(f"Episode ID must be non-negative: {episode}")
            result.add(episode)
    if not result:
        raise ValueError("--episodes did not contain any episode IDs")
    return result


def parse_goal_images(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=PATH for --goal-image, got {value!r}")
        dataset, raw_path = value.split("=", 1)
        if dataset not in VALID_DATASETS:
            raise ValueError(f"Unknown goal-image dataset {dataset!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Goal image does not exist: {path}")
        result[dataset] = path
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_episode_quality(parquet_path: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read episode quality labels. Install it with "
            "`python -m pip install pyarrow`."
        ) from exc

    table = pq.read_table(parquet_path, columns=["quality"])
    if table.num_rows == 0:
        raise ValueError(f"Empty trajectory file: {parquet_path}")
    values = table.column("quality").unique().to_pylist()
    if len(values) != 1 or values[0] not in VALID_QUALITIES:
        raise ValueError(f"Unexpected quality values in {parquet_path}: {values}")
    return str(values[0])


def discover_dataset(data_root: Path, dataset: str) -> list[Episode]:
    root = (data_root / f"robomimic_{dataset}_abs").resolve()
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    tasks_path = root / "meta/tasks.jsonl"
    for path in (info_path, episodes_path, tasks_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset metadata: {path}")

    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    episode_rows = read_jsonl(episodes_path)
    task_rows = read_jsonl(tasks_path)
    if len(task_rows) != 1:
        raise ValueError(f"Expected exactly one task in {tasks_path}, found {len(task_rows)}")

    total_episodes = int(info["total_episodes"])
    if len(episode_rows) != total_episodes:
        raise ValueError(
            f"Episode metadata count mismatch for {dataset}: "
            f"info={total_episodes}, jsonl={len(episode_rows)}"
        )
    if int(info["fps"]) <= 0:
        raise ValueError(f"Invalid FPS in {info_path}: {info['fps']}")
    features = info.get("features", {})
    for feature in ("image", "wrist_image"):
        if features.get(feature, {}).get("dtype") != "video":
            raise ValueError(f"Missing video feature {feature!r} in {info_path}")

    chunk_size = int(info["chunks_size"])
    data_template = info["data_path"]
    video_template = info["video_path"]
    task = str(task_rows[0]["task"]).strip()
    result: list[Episode] = []
    for expected_index, row in enumerate(episode_rows):
        index = int(row["episode_index"])
        if index != expected_index:
            raise ValueError(
                f"Non-contiguous episode index for {dataset}: expected {expected_index}, got {index}"
            )
        chunk = index // chunk_size
        parquet_path = root / data_template.format(
            episode_chunk=chunk, episode_index=index
        )
        front_path = root / video_template.format(
            episode_chunk=chunk, episode_index=index, video_key="image"
        )
        wrist_path = root / video_template.format(
            episode_chunk=chunk, episode_index=index, video_key="wrist_image"
        )
        for path in (parquet_path, front_path, wrist_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing episode file: {path}")
        result.append(
            Episode(
                dataset=dataset,
                root=root,
                index=index,
                length=int(row["length"]),
                quality=read_episode_quality(parquet_path),
                task=task,
                fps=int(info["fps"]),
                front_video=front_path,
                wrist_video=wrist_path,
            )
        )
    return result


def stratified_limit(
    episodes: Sequence[Episode], max_episodes: int, quality_order: Sequence[str]
) -> list[Episode]:
    ordered = sorted(episodes, key=lambda episode: episode.index)
    if max_episodes == 0 or len(ordered) <= max_episodes:
        return ordered
    groups: dict[str, list[Episode]] = defaultdict(list)
    for episode in ordered:
        groups[episode.quality].append(episode)

    selected: list[Episode] = []
    positions = {quality: 0 for quality in quality_order}
    while len(selected) < max_episodes:
        made_progress = False
        for quality in quality_order:
            position = positions[quality]
            group = groups.get(quality, [])
            if position < len(group):
                selected.append(group[position])
                positions[quality] += 1
                made_progress = True
                if len(selected) == max_episodes:
                    break
        if not made_progress:
            break
    return selected


def select_episodes(
    all_episodes: Sequence[Episode],
    qualities: Sequence[str],
    explicit_ids: set[int] | None,
    max_episodes: int,
) -> list[Episode]:
    available_ids = {episode.index for episode in all_episodes}
    if explicit_ids is not None:
        unknown = sorted(explicit_ids - available_ids)
        if unknown:
            raise ValueError(f"Unknown episode IDs: {unknown}")
        selected = [episode for episode in all_episodes if episode.index in explicit_ids]
        rejected = [episode.index for episode in selected if episode.quality not in qualities]
        if rejected:
            raise ValueError(
                f"Explicit episodes excluded by --qualities: {rejected}. "
                "Include their labels or remove the IDs."
            )
        return sorted(selected, key=lambda episode: episode.index)

    filtered = [episode for episode in all_episodes if episode.quality in qualities]
    return stratified_limit(filtered, max_episodes, qualities)


def original_sample_indices(length: int, interval: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"Episode length must be positive, got {length}")
    indices = list(range(0, length, interval))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def fixed_sample_indices(length: int, comparisons: int) -> list[int]:
    """Return K+1 uniformly spaced frame indices for exactly K comparisons."""
    if length <= 0:
        raise ValueError(f"Episode length must be positive, got {length}")
    if comparisons <= 0:
        raise ValueError(f"Fixed comparisons must be positive, got {comparisons}")
    if length <= comparisons:
        raise ValueError(
            f"Episode length {length} is too short for {comparisons} fixed "
            f"comparisons; at least {comparisons + 1} frames are required"
        )

    last = length - 1
    # Integer rounding avoids floating-point drift while keeping both endpoints.
    indices = [
        (position * last + comparisons // 2) // comparisons
        for position in range(comparisons + 1)
    ]
    if len(set(indices)) != comparisons + 1:
        raise RuntimeError(
            f"Fixed sampling produced duplicate indices for length={length}, "
            f"comparisons={comparisons}: {indices}"
        )
    return indices


def source_sample_indices(
    length: int, interval: int | None, fixed_samples: int | None
) -> list[int]:
    if fixed_samples is not None:
        return fixed_sample_indices(length, fixed_samples)
    if interval is None:
        raise ValueError("frame interval is required when fixed sampling is disabled")
    return original_sample_indices(length, interval)


def run_command(command: Sequence[str], description: str) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found while {description}: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed while {description}: {' '.join(command)}") from exc


def clear_pngs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


def extract_sampled_frames(
    video: Path,
    output_dir: Path,
    indices: Sequence[int],
) -> list[Path]:
    if not indices:
        raise ValueError("At least one source frame index is required")
    if any(index < 0 for index in indices):
        raise ValueError(f"Frame indices must be non-negative: {indices}")
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError(f"Frame indices must be strictly increasing: {indices}")

    clear_pngs(output_dir)
    # FFmpeg's n is zero-based. Enumerating the selected indices supports both
    # fixed-interval sampling and fixed-K sampling with normalized time points.
    select_expr = "select=" + "+".join(f"eq(n\\,{index})" for index in indices)
    output_pattern = output_dir / "frame_%06d.png"
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        select_expr,
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(output_pattern),
    ]
    run_command(command, f"sampling {video}")
    paths = sorted(output_dir.glob("*.png"))
    if len(paths) != len(indices):
        raise RuntimeError(
            f"Sample count mismatch for {video}: expected {len(indices)}, "
            f"got {len(paths)}"
        )
    return paths


def extract_single_frame(video: Path, frame_index: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        str(output_path),
    ]
    run_command(command, f"extracting frame {frame_index} from {video}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create goal image: {output_path}")


def stage_episode(
    episode: Episode,
    stage_root: Path,
    interval: int | None,
    fixed_samples: int | None,
    view_mode: str = "image-wrist",
    reuse_existing: bool = False,
) -> tuple[Path, Path, list[int]]:
    front_dir = stage_root / "front"
    wrist_dir = stage_root / "wrist"
    indices = source_sample_indices(episode.length, interval, fixed_samples)
    sampling_strategy = "fixed" if fixed_samples is not None else "interval"
    metadata_path = stage_root / "stage_metadata.json"
    if reuse_existing and metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            matches = (
                existing.get("dataset") == episode.dataset
                and existing.get("episode_index") == episode.index
                and existing.get("source_length") == episode.length
                and existing.get("sampling_strategy", "interval") == sampling_strategy
                and existing.get("source_frame_interval") == interval
                and existing.get("fixed_samples") == fixed_samples
                and existing.get("source_indices") == indices
                and existing.get("view_mode") == view_mode
                and len(list(front_dir.glob("*.png"))) == len(indices)
                and (
                    view_mode == "image-only"
                    or len(list(wrist_dir.glob("*.png"))) == len(indices)
                )
            )
            if matches:
                print(f"  Reusing {len(indices)} staged frames per view from {stage_root}")
                return front_dir, front_dir if view_mode == "image-only" else wrist_dir, indices
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    front_frames = extract_sampled_frames(episode.front_video, front_dir, indices)
    if view_mode == "image-only":
        wrist_frames = front_frames
        wrist_dir = front_dir
    else:
        wrist_frames = extract_sampled_frames(episode.wrist_video, wrist_dir, indices)
        if len(front_frames) != len(wrist_frames):
            raise RuntimeError(
                f"Staged camera mismatch for {episode.dataset}/{episode.index}: "
                f"front={len(front_frames)}, wrist={len(wrist_frames)}"
            )
    stage_metadata = {
        "dataset": episode.dataset,
        "episode_index": episode.index,
        "source_length": episode.length,
        "source_fps": episode.fps,
        "sampling_strategy": sampling_strategy,
        "source_frame_interval": interval,
        "fixed_samples": fixed_samples,
        "source_indices": indices,
        "view_mode": view_mode,
        "view_mapping": (
            {
                "cam_high": "image",
                "cam_left_wrist": "image (duplicated)",
                "cam_right_wrist": "image (duplicated)",
            }
            if view_mode == "image-only"
            else {
                "cam_high": "image",
                "cam_left_wrist": "wrist_image",
                "cam_right_wrist": "wrist_image (duplicated)",
            }
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(stage_metadata, handle, indent=2, ensure_ascii=False)
    return front_dir, wrist_dir, indices


def safe_tag(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return (value.strip("_") or "unnamed")[:96]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    # Records created before fixed-K support implicitly used interval sampling.
    sampling_strategy = record.get("sampling_strategy", "interval")
    return (
        record.get("model_path"),
        record.get("dataset"),
        record.get("episode_index"),
        record.get("quality"),
        record.get("task"),
        record.get("view_mode"),
        record.get("mode"),
        sampling_strategy,
        record.get("source_frame_interval"),
        record.get("fixed_samples"),
        record.get("goal_id"),
    )


def completed_record_usable(record: dict[str, Any]) -> bool:
    if record.get("status") != "completed" or not record.get("output_dir"):
        return False
    return (Path(record["output_dir"]) / "pred_vllm.json").is_file()


def upsert_record(records: list[dict[str, Any]], new_record: dict[str, Any]) -> None:
    key = record_key(new_record)
    for position, existing in enumerate(records):
        if record_key(existing) == key:
            records[position] = new_record
            return
    records.append(new_record)


def result_metrics(output_dir: Path) -> dict[str, Any]:
    prediction_path = output_dir / "pred_vllm.json"
    with prediction_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    parse_failures = sum(
        1 for item in predictions if not SCORE_RE.search(str(item.get("pred", "")))
    )
    progress = [float(item["progress"]) for item in predictions]
    return {
        "num_predictions": len(predictions),
        "parse_failures": parse_failures,
        "final_progress": progress[-1] if progress else None,
        "min_progress": min(progress) if progress else None,
        "max_progress": max(progress) if progress else None,
    }


def plot_episode_progress(
    mode_paths: dict[str, Path],
    output_path: Path,
    source_indices: Sequence[int],
    fps: int,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves: dict[str, list[float]] = {}
    for mode, path in mode_paths.items():
        prediction_path = path / "pred_vllm.json"
        if not prediction_path.is_file():
            continue
        with prediction_path.open("r", encoding="utf-8") as handle:
            predictions = json.load(handle)
        curves[mode] = [float(item["progress"]) * 100.0 for item in predictions]
    if not curves:
        return

    fig, axis = plt.subplots(figsize=(10, 5))
    colors = {"forward": "#2196F3", "incremental": "#FF9800", "backward": "#4CAF50"}
    for mode in VALID_MODES:
        values = curves.get(mode)
        if not values:
            continue
        times = [index / fps for index in source_indices[1 : len(values) + 1]]
        axis.plot(times, values, label=f"{mode} ({values[-1]:.1f}%)", color=colors[mode])

    common_length = min((len(values) for values in curves.values()), default=0)
    if common_length:
        fused = [
            mean(values[position] for values in curves.values())
            for position in range(common_length)
        ]
        times = [index / fps for index in source_indices[1 : common_length + 1]]
        axis.plot(times, fused, "--", color="#E91E63", linewidth=2.5, label=f"mean ({fused[-1]:.1f}%)")
    axis.set_title(title)
    axis.set_xlabel("Original episode time (s)")
    axis.set_ylabel("Progress (%)")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "std": pstdev(finite),
        "min": min(finite),
        "max": max(finite),
    }


def build_summary(records: Sequence[dict[str, Any]], run_config: dict[str, Any]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    by_group: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    by_episode: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for record in completed:
        final = record.get("final_progress")
        if final is None:
            continue
        by_group[
            (record["dataset"], record["quality"], record["view_mode"], record["mode"])
        ].append(float(final))
        by_episode[
            (
                record["dataset"],
                int(record["episode_index"]),
                record["quality"],
                record["view_mode"],
            )
        ].append(float(final))

    aggregates = []
    for (dataset, quality, view_mode, mode), values in sorted(by_group.items()):
        aggregates.append(
            {
                "dataset": dataset,
                "quality": quality,
                "view_mode": view_mode,
                "mode": mode,
                **numeric_summary(values),
            }
        )

    fused_by_quality: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    fused_episodes = []
    for (dataset, episode_index, quality, view_mode), values in sorted(by_episode.items()):
        fused = mean(values)
        fused_episodes.append(
            {
                "dataset": dataset,
                "episode_index": episode_index,
                "quality": quality,
                "view_mode": view_mode,
                "completed_modes": len(values),
                "mean_final_progress": fused,
            }
        )
        fused_by_quality[(dataset, quality, view_mode)].append(fused)

    fused_aggregates = []
    for (dataset, quality, view_mode), values in sorted(fused_by_quality.items()):
        fused_aggregates.append(
            {
                "dataset": dataset,
                "quality": quality,
                "view_mode": view_mode,
                **numeric_summary(values),
            }
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": run_config,
        "completed_mode_runs": len(completed),
        "failed_mode_runs": sum(record.get("status") == "failed" for record in records),
        "mode_aggregates": aggregates,
        "fused_episodes": fused_episodes,
        "fused_aggregates": fused_aggregates,
    }


def write_records_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = [
        "status",
        "dataset",
        "episode_index",
        "quality",
        "task",
        "view_mode",
        "mode",
        "sampling_strategy",
        "source_length",
        "source_frame_interval",
        "fixed_samples",
        "num_predictions",
        "parse_failures",
        "final_progress",
        "elapsed_seconds",
        "goal_id",
        "output_dir",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def resolve_goal(
    episode: Episode,
    all_by_dataset: dict[str, list[Episode]],
    args: argparse.Namespace,
    custom_goals: dict[str, Path],
) -> tuple[Path, str]:
    if episode.dataset in custom_goals:
        path = custom_goals[episode.dataset]
        return path, f"custom:{path}"
    if args.goal_mode == "blank":
        path = SCRIPT_DIR / "examples/blank_goal.png"
        if not path.is_file():
            raise FileNotFoundError(f"Blank goal image not found: {path}")
        return path, "blank"

    goals_dir = args.output_root.resolve() / "_goals"
    if args.goal_mode == "episode-end":
        reference = episode
        goal_id = f"episode-end:{episode.index}"
    else:
        candidates = all_by_dataset[episode.dataset]
        if args.reference_episode is None:
            better = [item for item in candidates if item.quality == "better"]
            if better:
                median_length = median(item.length for item in better)
                reference = min(
                    better,
                    key=lambda item: (abs(item.length - median_length), item.index),
                )
            else:
                reference = candidates[0]
        else:
            matches = [item for item in candidates if item.index == args.reference_episode]
            if not matches:
                raise ValueError(
                    f"Reference episode {args.reference_episode} does not exist in {episode.dataset}"
                )
            reference = matches[0]
        goal_id = f"reference-end:{reference.index}"

    goal_path = goals_dir / f"{episode.dataset}_episode_{reference.index:06d}_end.png"
    if not goal_path.is_file():
        extract_single_frame(reference.front_video, reference.length - 1, goal_path)
    return goal_path, goal_id


def print_plan(selected: dict[str, list[Episode]], args: argparse.Namespace) -> None:
    total_episodes = sum(len(episodes) for episodes in selected.values())
    total_mode_runs = total_episodes * len(args.modes)
    print("Robo-Dopamine robomimic evaluation plan")
    print(f"  model: {args.model_path.resolve()}")
    print(f"  output: {args.output_root.resolve()}")
    print(f"  modes: {', '.join(args.modes)}")
    print(f"  view mode: {args.view_mode}")
    if args.fixed_samples is not None:
        print(f"  sampling: {args.fixed_samples} fixed comparisons per episode")
    else:
        print(f"  source frame interval: {args.frame_interval}")
    print(f"  goal mode: {args.goal_mode}")
    if args.view_mode == "image-only":
        print("  view mapping: image -> high/front + both wrist inputs")
    else:
        print("  view mapping: image -> high/front; wrist_image -> both wrist inputs")
    for dataset, episodes in selected.items():
        quality_counts: dict[str, int] = defaultdict(int)
        for episode in episodes:
            quality_counts[episode.quality] += 1
        ids = ",".join(str(episode.index) for episode in episodes)
        counts = ", ".join(f"{quality}={quality_counts[quality]}" for quality in VALID_QUALITIES if quality_counts[quality])
        print(f"  {dataset}: {len(episodes)} episodes [{ids}] ({counts})")
    print(f"  total: {total_episodes} episodes, {total_mode_runs} mode runs")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    explicit_ids = parse_episode_spec(args.episodes)
    custom_goals = parse_goal_images(args.goal_image)
    args.data_root = args.data_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    all_by_dataset: dict[str, list[Episode]] = {}
    selected: dict[str, list[Episode]] = {}
    for dataset in args.datasets:
        all_episodes = discover_dataset(args.data_root, dataset)
        all_by_dataset[dataset] = all_episodes
        selected[dataset] = select_episodes(
            all_episodes, args.qualities, explicit_ids, args.max_episodes
        )
        if not selected[dataset]:
            raise ValueError(f"No episodes selected for dataset {dataset}")

    # Validate that every selected episode can satisfy the requested sampling
    # strategy before loading the model or writing any outputs.
    for episodes in selected.values():
        for episode in episodes:
            source_sample_indices(
                episode.length, args.frame_interval, args.fixed_samples
            )

    print_plan(selected, args)
    if args.dry_run:
        print("Dry run complete; model was not loaded and no outputs were written.")
        return 0

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if args.goal_mode == "blank" and "backward" in args.modes:
        print("[WARN] backward mode with a blank goal is poorly calibrated; reference-end is recommended.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode the AV1 dataset videos")

    # Lazy import keeps --help and --dry-run usable without vLLM/GPU dependencies.
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from examples.inference import GRMInference
    except ModuleNotFoundError as exc:
        if exc.name == "vllm":
            raise RuntimeError(
                "vLLM is not installed in the active Python environment. "
                "Run `conda activate robo-dopamine` before launching this script."
            ) from exc
        raise

    model_tag = safe_tag(args.model_path.name)
    output_model_root = args.output_root / model_tag
    output_model_root.mkdir(parents=True, exist_ok=True)
    records_path = output_model_root / "results.json"
    records = load_records(records_path)
    sampling_strategy = "fixed" if args.fixed_samples is not None else "interval"
    sampling_tag = (
        f"fixed{args.fixed_samples}"
        if args.fixed_samples is not None
        else f"inter{args.frame_interval}"
    )

    run_config = {
        "model_path": str(args.model_path),
        "data_root": str(args.data_root),
        "datasets": list(args.datasets),
        "qualities": list(args.qualities),
        "episodes": args.episodes,
        "max_episodes": args.max_episodes,
        "modes": list(args.modes),
        "view_mode": args.view_mode,
        "sampling_strategy": sampling_strategy,
        "source_frame_interval": args.frame_interval,
        "fixed_samples": args.fixed_samples,
        "batch_size": args.batch_size,
        "goal_mode": args.goal_mode,
        "reference_episode": args.reference_episode,
        "custom_goals": {key: str(value) for key, value in custom_goals.items()},
        "view_mapping": (
            {
                "cam_high": "image",
                "cam_left_wrist": "image (duplicated)",
                "cam_right_wrist": "image (duplicated)",
            }
            if args.view_mode == "image-only"
            else {
                "cam_high": "image",
                "cam_left_wrist": "wrist_image",
                "cam_right_wrist": "wrist_image (duplicated)",
            }
        ),
    }

    # Resolve all goal anchors before occupying GPU memory.
    goal_cache: dict[tuple[str, int], tuple[Path, str]] = {}
    for episodes in selected.values():
        for episode in episodes:
            goal_cache[(episode.dataset, episode.index)] = resolve_goal(
                episode, all_by_dataset, args, custom_goals
            )

    print(f"Loading model: {args.model_path}")
    model = GRMInference(str(args.model_path))
    print("Model loaded successfully.")

    total = sum(len(episodes) for episodes in selected.values())
    sequence = 0
    for dataset in args.datasets:
        for episode in selected[dataset]:
            sequence += 1
            goal_path, goal_id = goal_cache[(dataset, episode.index)]
            episode_output = (
                output_model_root
                / dataset
                / args.view_mode
                / f"episode_{episode.index:06d}_{episode.quality}"
                / f"{sampling_tag}_{safe_tag(goal_id)}"
            )
            episode_output.mkdir(parents=True, exist_ok=True)
            print(
                f"\n[{sequence}/{total}] {dataset} episode={episode.index} "
                f"quality={episode.quality} length={episode.length} goal={goal_id}"
            )

            mode_paths: dict[str, Path] = {}
            pending_modes: list[str] = []
            for mode in args.modes:
                probe = {
                    "model_path": str(args.model_path),
                    "dataset": dataset,
                    "episode_index": episode.index,
                    "quality": episode.quality,
                    "task": episode.task,
                    "view_mode": args.view_mode,
                    "mode": mode,
                    "sampling_strategy": sampling_strategy,
                    "source_frame_interval": args.frame_interval,
                    "fixed_samples": args.fixed_samples,
                    "goal_id": goal_id,
                }
                key = record_key(probe)
                existing = next(
                    (
                        record
                        for record in records
                        if record_key(record) == key and completed_record_usable(record)
                    ),
                    None,
                )
                if args.resume and existing is not None:
                    print(f"  [SKIP] {mode}: already completed")
                    mode_paths[mode] = Path(existing["output_dir"])
                else:
                    pending_modes.append(mode)

            source_indices = source_sample_indices(
                episode.length, args.frame_interval, args.fixed_samples
            )
            if not pending_modes:
                plot_episode_progress(
                    mode_paths,
                    episode_output / "progress_curve.png",
                    source_indices,
                    episode.fps,
                    f"{dataset} episode {episode.index} ({episode.quality})",
                )
                continue

            if args.keep_staged_frames:
                stage_root = (
                    args.output_root
                    / "_staged"
                    / dataset
                    / f"episode_{episode.index:06d}"
                    / args.view_mode
                    / sampling_tag
                )
                temporary_context = None
            else:
                temporary_context = tempfile.TemporaryDirectory(
                    prefix=f"robo-dopamine-{dataset}-{episode.index:06d}-"
                )
                stage_root = Path(temporary_context.name)

            try:
                front_dir, wrist_dir, source_indices = stage_episode(
                    episode,
                    stage_root,
                    interval=args.frame_interval,
                    fixed_samples=args.fixed_samples,
                    view_mode=args.view_mode,
                    reuse_existing=args.keep_staged_frames,
                )
                for mode in pending_modes:
                    key_probe = {
                        "model_path": str(args.model_path),
                        "dataset": dataset,
                        "episode_index": episode.index,
                        "quality": episode.quality,
                        "task": episode.task,
                        "view_mode": args.view_mode,
                        "mode": mode,
                        "sampling_strategy": sampling_strategy,
                        "source_frame_interval": args.frame_interval,
                        "fixed_samples": args.fixed_samples,
                        "goal_id": goal_id,
                    }
                    key = record_key(key_probe)
                    print(f"  [{mode}] running {len(source_indices) - 1} comparisons")
                    started = time.time()
                    base_record: dict[str, Any] = {
                        **key_probe,
                        "source_length": episode.length,
                        "source_fps": episode.fps,
                        "source_indices": source_indices,
                        "goal_image": str(goal_path),
                        "view_mapping": run_config["view_mapping"],
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    try:
                        # The staged directories already contain only the desired
                        # original frames, so use interval=1 inside run_pipeline.
                        secondary_dir = (
                            front_dir if args.view_mode == "image-only" else wrist_dir
                        )
                        output_dir = Path(
                            model.run_pipeline(
                                cam_high_path=str(front_dir),
                                cam_left_path=str(secondary_dir),
                                cam_right_path=str(secondary_dir),
                                out_root=str(episode_output),
                                task=episode.task,
                                frame_interval=1,
                                batch_size=args.batch_size,
                                goal_image=str(goal_path),
                                eval_mode=mode,
                                visualize=args.visualize,
                            )
                        )
                        metrics = result_metrics(output_dir)
                        record = {
                            **base_record,
                            **metrics,
                            "status": "completed",
                            "elapsed_seconds": round(time.time() - started, 3),
                            "output_dir": str(output_dir),
                            "error": None,
                        }
                        mode_paths[mode] = output_dir
                        final_text = (
                            "n/a" if metrics["final_progress"] is None else f"{metrics['final_progress'] * 100:.1f}%"
                        )
                        print(
                            f"  [{mode}] completed: final={final_text}, "
                            f"parse_failures={metrics['parse_failures']}, "
                            f"elapsed={record['elapsed_seconds']:.1f}s"
                        )
                    except Exception as exc:
                        record = {
                            **base_record,
                            "status": "failed",
                            "elapsed_seconds": round(time.time() - started, 3),
                            "output_dir": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        print(f"  [{mode}] FAILED: {record['error']}")
                        traceback.print_exc()
                    upsert_record(records, record)
                    atomic_write_json(records_path, records)
                    write_records_csv(output_model_root / "results.csv", records)
                    if record["status"] == "failed" and args.fail_fast:
                        raise RuntimeError(record["error"])

                plot_episode_progress(
                    mode_paths,
                    episode_output / "progress_curve.png",
                    source_indices,
                    episode.fps,
                    f"{dataset} episode {episode.index} ({episode.quality})",
                )
            finally:
                if temporary_context is not None:
                    temporary_context.cleanup()

    current_keys = {
        record_key(
            {
                "model_path": str(args.model_path),
                "dataset": episode.dataset,
                "episode_index": episode.index,
                "quality": episode.quality,
                "task": episode.task,
                "view_mode": args.view_mode,
                "mode": mode,
                "sampling_strategy": sampling_strategy,
                "source_frame_interval": args.frame_interval,
                "fixed_samples": args.fixed_samples,
                "goal_id": goal_cache[(episode.dataset, episode.index)][1],
            }
        )
        for episodes in selected.values()
        for episode in episodes
        for mode in args.modes
    }
    current_records = [record for record in records if record_key(record) in current_keys]
    summary = build_summary(current_records, run_config)
    summary_path = output_model_root / "summary.json"
    atomic_write_json(summary_path, summary)
    failures = sum(record.get("status") == "failed" for record in current_records)
    completed = sum(record.get("status") == "completed" for record in current_records)
    print(f"\nEvaluation finished: completed={completed}, failed={failures}")
    print(f"Records: {records_path}")
    print(f"Summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("ROBO_DOPAMINE_DEBUG") == "1":
            traceback.print_exc()
        raise SystemExit(2)
