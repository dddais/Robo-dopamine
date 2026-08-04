#!/usr/bin/env python3
"""Evaluate Robo-Dopamine on ``data/ljx_lfz_task/new/suc``.

Each trajectory has a metadata-provided instruction and three synchronized
H.264 videos: ``faceImg.mp4``, ``leftImg.mp4``, and ``rightImg.mp4``. The
default ``three-view`` mode maps them to Robo-Dopamine's high/left/right
inputs. ``face-only`` fills all three inputs with ``faceImg.mp4``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/dais/workspace/data/ljx_lfz_task/new/suc")
DEFAULT_MODEL = SCRIPT_DIR / "pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview"
DEFAULT_OUTPUT = SCRIPT_DIR / "results/ljx_lfz_suc_eval"
DATASET_DISPLAY_NAME = "ljx_lfz successful-set"
DATASET_SPLIT = "suc"
VALID_MODES = ("forward", "incremental", "backward")
VALID_VIEW_MODES = ("three-view", "face-only")
SCORE_RE = re.compile(r"<score>\s*[+-]?\d+(?:\.\d+)?%\s*</score>")


@dataclass(frozen=True)
class Trajectory:
    sample_id: str
    task_id: str
    trajectory_index: int
    instruction: str
    target_obj: str
    face_video: Path
    left_video: Path
    right_video: Path
    dataset_split: str = "suc"
    correct_target_obj: str | None = None
    instruction_video_match: bool = True
    source_suc_id: str | None = None
    source_trajectory_index: int | None = None


@dataclass(frozen=True)
class VideoInfo:
    frames: int
    fps: float
    width: int
    height: int
    codec: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Run Robo-Dopamine on the {DATASET_DISPLAY_NAME} trajectories. "
            "By default, one trajectory per task directory is evaluated."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Directory containing metadata.jsonl (default: {DEFAULT_DATA_ROOT})",
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
        "--task-ids",
        nargs="+",
        default=None,
        metavar="TASK",
        help="Task IDs such as task1_1 task4_2; default is all tasks.",
    )
    parser.add_argument(
        "--trajectory-indices",
        default=None,
        metavar="SPEC",
        help=(
            "Trajectory indices applied within every selected task, e.g. "
            "'1,3,5-8'. Overrides per-task limiting."
        ),
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            f"Exact metadata IDs, e.g. {DATASET_SPLIT}/ljx_lfz_task_1_1/1. "
            "Overrides task/index filtering."
        ),
    )
    parser.add_argument(
        "--max-trajectories-per-task",
        type=int,
        default=1,
        help="Maximum per task after filtering; 0 means all (default: 1).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=VALID_MODES,
        default=list(VALID_MODES),
    )
    parser.add_argument(
        "--view-mode",
        choices=VALID_VIEW_MODES,
        default="three-view",
        help=(
            "three-view uses face/left/right videos; face-only fills all three "
            "inputs with faceImg.mp4 (default: three-view)."
        ),
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=20,
        help="Sample every N frames; 20 is one second for this dataset (default: 20).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--goal-mode",
        choices=("blank", "episode-end", "reference-end"),
        default="blank",
        help=(
            "Goal anchor. reference-end uses a canonical successful trajectory "
            "with the same task ID and exact instruction (default: blank)."
        ),
    )
    parser.add_argument(
        "--goal-image",
        type=Path,
        default=None,
        help="Custom goal image applied to all selected trajectories.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed trajectory/mode/config combinations (default: true).",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate metadata/media and print the plan without loading the model.",
    )
    args = parser.parse_args(argv)
    if args.max_trajectories_per_task < 0:
        parser.error("--max-trajectories-per-task must be >= 0")
    if args.frame_interval <= 0:
        parser.error("--frame-interval must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_index_spec(spec: str | None) -> set[int] | None:
    if spec is None:
        return None
    result: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            raw_start, raw_end = token.split("-", 1)
            try:
                start, end = int(raw_start), int(raw_end)
            except ValueError as exc:
                raise ValueError(f"Invalid trajectory range: {token!r}") from exc
            if start <= 0 or end < start:
                raise ValueError(f"Invalid trajectory range: {token!r}")
            result.update(range(start, end + 1))
        else:
            try:
                index = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid trajectory index: {token!r}") from exc
            if index <= 0:
                raise ValueError(f"Trajectory indices are one-based, got {index}")
            result.add(index)
    if not result:
        raise ValueError("--trajectory-indices did not contain any indices")
    return result


def resolve_metadata_video(data_root: Path, metadata_path: str) -> Path:
    relative = Path(metadata_path)
    candidates = [data_root / relative, data_root.parent / relative]
    if relative.parts and relative.parts[0] == data_root.name:
        candidates.append(data_root / Path(*relative.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve metadata video path {metadata_path!r}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def discover_trajectories(data_root: Path) -> list[Trajectory]:
    metadata_path = data_root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    rows = read_jsonl(metadata_path)
    if not rows:
        raise ValueError(f"No records found in {metadata_path}")

    required = {
        "id",
        "split",
        "task_id",
        "trajectory_index",
        "instruction",
        "target_obj",
        "instruction_video_match",
        "video_paths",
    }
    result: list[Trajectory] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Metadata row {row_number} is missing fields: {sorted(missing)}")
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate metadata ID: {sample_id}")
        seen.add(sample_id)
        if row["split"] != "suc":
            raise ValueError(f"Unexpected split for {sample_id}: {row['split']!r}")
        if row["instruction_video_match"] is not True:
            raise ValueError(f"Instruction/video mismatch in successful dataset: {sample_id}")
        video_paths = list(row["video_paths"])
        if len(video_paths) != 3:
            raise ValueError(f"Expected three videos for {sample_id}, got {video_paths}")
        by_name = {
            Path(path).name: resolve_metadata_video(data_root, str(path))
            for path in video_paths
        }
        expected_names = {"faceImg.mp4", "leftImg.mp4", "rightImg.mp4"}
        if set(by_name) != expected_names:
            raise ValueError(f"Unexpected video names for {sample_id}: {sorted(by_name)}")
        result.append(
            Trajectory(
                sample_id=sample_id,
                task_id=str(row["task_id"]),
                trajectory_index=int(row["trajectory_index"]),
                instruction=str(row["instruction"]).strip(),
                target_obj=str(row["target_obj"]).strip(),
                face_video=by_name["faceImg.mp4"],
                left_video=by_name["leftImg.mp4"],
                right_video=by_name["rightImg.mp4"],
                dataset_split="suc",
                correct_target_obj=str(row.get("correct_target_obj", row["target_obj"])).strip(),
                instruction_video_match=True,
                source_suc_id=str(row.get("source_suc_id", sample_id)),
                source_trajectory_index=int(row["trajectory_index"]),
            )
        )
    return sorted(result, key=lambda item: (natural_key(item.task_id), item.trajectory_index))


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def round_robin_by_instruction(
    trajectories: Sequence[Trajectory], limit: int
) -> list[Trajectory]:
    ordered = sorted(trajectories, key=lambda item: item.trajectory_index)
    if limit == 0 or len(ordered) <= limit:
        return ordered
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    instruction_order: list[str] = []
    for trajectory in ordered:
        if trajectory.instruction not in groups:
            instruction_order.append(trajectory.instruction)
        groups[trajectory.instruction].append(trajectory)
    positions = {instruction: 0 for instruction in instruction_order}
    selected: list[Trajectory] = []
    while len(selected) < limit:
        changed = False
        for instruction in instruction_order:
            position = positions[instruction]
            if position < len(groups[instruction]):
                selected.append(groups[instruction][position])
                positions[instruction] += 1
                changed = True
                if len(selected) == limit:
                    break
        if not changed:
            break
    return selected


def select_trajectories(
    trajectories: Sequence[Trajectory], args: argparse.Namespace
) -> list[Trajectory]:
    if args.sample_ids:
        requested = set(args.sample_ids)
        available = {item.sample_id for item in trajectories}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Unknown sample IDs: {missing}")
        return [item for item in trajectories if item.sample_id in requested]

    known_tasks = {item.task_id for item in trajectories}
    requested_tasks = set(args.task_ids) if args.task_ids else known_tasks
    unknown_tasks = sorted(requested_tasks - known_tasks)
    if unknown_tasks:
        raise ValueError(f"Unknown task IDs: {unknown_tasks}")
    index_filter = parse_index_spec(args.trajectory_indices)
    filtered = [
        item
        for item in trajectories
        if item.task_id in requested_tasks
        and (index_filter is None or item.trajectory_index in index_filter)
    ]
    if index_filter is not None:
        missing_by_task = {}
        for task_id in sorted(requested_tasks, key=natural_key):
            present = {
                item.trajectory_index for item in trajectories if item.task_id == task_id
            }
            missing = sorted(index_filter - present)
            if missing:
                missing_by_task[task_id] = missing
        if missing_by_task:
            raise ValueError(
                "Requested trajectory indices do not exist in every selected task: "
                f"{missing_by_task}"
            )
        return filtered

    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for item in filtered:
        by_task[item.task_id].append(item)
    selected = []
    for task_id in sorted(by_task, key=natural_key):
        selected.extend(
            round_robin_by_instruction(
                by_task[task_id], args.max_trajectories_per_task
            )
        )
    return selected


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        raw = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=30)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required to validate the dataset videos") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {path}: {exc.output.decode(errors='replace')}") from exc
    stream = json.loads(raw)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    fps = float(numerator) / float(denominator)
    return VideoInfo(
        frames=int(stream["nb_frames"]),
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
        codec=str(stream["codec_name"]),
    )


def validate_media(trajectory: Trajectory) -> VideoInfo:
    infos = [
        probe_video(trajectory.face_video),
        probe_video(trajectory.left_video),
        probe_video(trajectory.right_video),
    ]
    signatures = {
        (info.frames, info.fps, info.width, info.height, info.codec) for info in infos
    }
    if len(signatures) != 1:
        raise ValueError(
            f"Camera media mismatch for {trajectory.sample_id}: {infos}"
        )
    info = infos[0]
    if info.frames < 2 or info.fps <= 0:
        raise ValueError(f"Invalid video metadata for {trajectory.sample_id}: {info}")
    return info


def sample_indices(frame_count: int, interval: int) -> list[int]:
    indices = list(range(0, frame_count, interval))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def safe_tag(value: str, maximum: int = 96) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return (tag or "unnamed")[:maximum]


def extract_frame(video: Path, frame_index: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
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
        str(output),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to extract goal images") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not extract frame {frame_index} from {video}"
        ) from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Goal image was not created: {output}")


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
    return (
        record.get("model_path"),
        record.get("data_root"),
        record.get("sample_id"),
        record.get("instruction"),
        record.get("view_mode"),
        record.get("mode"),
        record.get("frame_interval"),
        record.get("goal_id"),
    )


def usable_completed_record(record: dict[str, Any]) -> bool:
    if record.get("status") != "completed" or not record.get("output_dir"):
        return False
    return (Path(record["output_dir"]) / "pred_vllm.json").is_file()


def upsert_record(records: list[dict[str, Any]], new_record: dict[str, Any]) -> None:
    key = record_key(new_record)
    for index, existing in enumerate(records):
        if record_key(existing) == key:
            records[index] = new_record
            return
    records.append(new_record)


def result_metrics(output_dir: Path) -> dict[str, Any]:
    prediction_path = output_dir / "pred_vllm.json"
    with prediction_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    progress = [float(item["progress"]) for item in predictions]
    parse_failures = sum(
        not SCORE_RE.search(str(item.get("pred", ""))) for item in predictions
    )
    return {
        "num_predictions": len(predictions),
        "parse_failures": int(parse_failures),
        "final_progress": progress[-1] if progress else None,
        "min_progress": min(progress) if progress else None,
        "max_progress": max(progress) if progress else None,
    }


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": mean(finite),
        "median": median(finite),
        "std": pstdev(finite),
        "min": min(finite),
        "max": max(finite),
    }


def build_summary(
    records: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "completed"]
    mode_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    episode_groups: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(list)
    for record in completed:
        final = record.get("final_progress")
        if final is None:
            continue
        mode_groups[
            (record["task_id"], record["view_mode"], record["mode"])
        ].append(float(final))
        episode_groups[
            (
                record["sample_id"],
                record["task_id"],
                int(record["trajectory_index"]),
                record["instruction"],
                record["view_mode"],
            )
        ].append(float(final))

    mode_aggregates = [
        {
            "task_id": task_id,
            "view_mode": view_mode,
            "mode": mode,
            **numeric_summary(values),
        }
        for (task_id, view_mode, mode), values in sorted(mode_groups.items())
    ]
    fused_trajectories = []
    fused_by_task: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, values in sorted(episode_groups.items()):
        sample_id, task_id, trajectory_index, instruction, view_mode = key
        fused = mean(values)
        fused_trajectories.append(
            {
                "sample_id": sample_id,
                "task_id": task_id,
                "trajectory_index": trajectory_index,
                "instruction": instruction,
                "view_mode": view_mode,
                "completed_modes": len(values),
                "mean_final_progress": fused,
            }
        )
        fused_by_task[(task_id, view_mode)].append(fused)
    fused_task_aggregates = [
        {
            "task_id": task_id,
            "view_mode": view_mode,
            **numeric_summary(values),
        }
        for (task_id, view_mode), values in sorted(fused_by_task.items())
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "completed_mode_runs": len(completed),
        "failed_mode_runs": sum(record.get("status") == "failed" for record in records),
        "mode_aggregates": mode_aggregates,
        "fused_trajectories": fused_trajectories,
        "fused_task_aggregates": fused_task_aggregates,
    }


def write_records_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = [
        "status",
        "sample_id",
        "task_id",
        "trajectory_index",
        "instruction",
        "target_obj",
        "correct_target_obj",
        "instruction_video_match",
        "source_suc_id",
        "source_trajectory_index",
        "view_mode",
        "mode",
        "frame_count",
        "frame_interval",
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


def plot_progress(
    mode_paths: dict[str, Path],
    output_path: Path,
    indices: Sequence[int],
    fps: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves: dict[str, list[float]] = {}
    for mode, directory in mode_paths.items():
        path = directory / "pred_vllm.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            predictions = json.load(handle)
        curves[mode] = [float(item["progress"]) * 100 for item in predictions]
    if not curves:
        return
    colors = {
        "forward": "#2196F3",
        "incremental": "#FF9800",
        "backward": "#4CAF50",
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    for mode in VALID_MODES:
        values = curves.get(mode)
        if not values:
            continue
        times = [frame / fps for frame in indices[1 : len(values) + 1]]
        axis.plot(times, values, color=colors[mode], label=f"{mode} ({values[-1]:.1f}%)")
    common = min((len(values) for values in curves.values()), default=0)
    if common:
        fused = [mean(values[i] for values in curves.values()) for i in range(common)]
        times = [frame / fps for frame in indices[1 : common + 1]]
        axis.plot(
            times,
            fused,
            "--",
            color="#E91E63",
            linewidth=2.5,
            label=f"mean ({fused[-1]:.1f}%)",
        )
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Progress (%)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def canonical_reference(
    trajectory: Trajectory, all_trajectories: Sequence[Trajectory]
) -> Trajectory:
    matches = [
        item
        for item in all_trajectories
        if item.task_id == trajectory.task_id
        and item.instruction == trajectory.instruction
    ]
    if not matches:
        raise ValueError(f"No reference candidate for {trajectory.sample_id}")
    ordered = sorted(matches, key=lambda item: item.trajectory_index)
    return ordered[(len(ordered) - 1) // 2]


def resolve_goal(
    trajectory: Trajectory,
    media: VideoInfo,
    all_trajectories: Sequence[Trajectory],
    media_cache: dict[str, VideoInfo],
    args: argparse.Namespace,
) -> tuple[Path, str]:
    if args.goal_image is not None:
        path = args.goal_image.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Goal image does not exist: {path}")
        return path, f"custom:{path}"
    if args.goal_mode == "blank":
        path = SCRIPT_DIR / "examples/blank_goal.png"
        if not path.is_file():
            raise FileNotFoundError(f"Blank goal image not found: {path}")
        return path, "blank"
    if args.goal_mode == "episode-end":
        reference = trajectory
        reference_media = media
        goal_id = f"episode-end:{trajectory.sample_id}"
    else:
        reference = canonical_reference(trajectory, all_trajectories)
        if reference.sample_id not in media_cache:
            media_cache[reference.sample_id] = validate_media(reference)
        reference_media = media_cache[reference.sample_id]
        goal_id = f"reference-end:{reference.sample_id}"
    goal_path = (
        args.output_root
        / "_goals"
        / safe_tag(reference.task_id)
        / f"trajectory_{reference.trajectory_index:03d}_{safe_tag(reference.instruction, 48)}.png"
    )
    if not goal_path.is_file():
        extract_frame(reference.face_video, reference_media.frames - 1, goal_path)
    return goal_path, goal_id


def print_plan(
    all_trajectories: Sequence[Trajectory],
    selected: Sequence[Trajectory],
    media_cache: dict[str, VideoInfo],
    args: argparse.Namespace,
) -> None:
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for item in selected:
        by_task[item.task_id].append(item)
    print(f"Robo-Dopamine {DATASET_DISPLAY_NAME} evaluation plan")
    print(f"  data: {args.data_root}")
    print(f"  model: {args.model_path}")
    print(f"  output: {args.output_root}")
    print(f"  dataset trajectories: {len(all_trajectories)}")
    print(f"  modes: {', '.join(args.modes)}")
    print(f"  view mode: {args.view_mode}")
    print(f"  frame interval: {args.frame_interval}")
    print(f"  goal mode: {args.goal_mode}")
    if args.view_mode == "face-only":
        print("  view mapping: faceImg -> high + left wrist + right wrist")
    else:
        print("  view mapping: faceImg -> high; leftImg/rightImg -> wrists")
    for task_id in sorted(by_task, key=natural_key):
        values = by_task[task_id]
        ids = ",".join(str(item.trajectory_index) for item in values)
        instructions = len({item.instruction for item in values})
        print(f"  {task_id}: {len(values)} trajectories [{ids}], {instructions} instructions")
    prompts = sum(
        len(sample_indices(media_cache[item.sample_id].frames, args.frame_interval)) - 1
        for item in selected
    )
    print(
        f"  total: {len(selected)} trajectories, "
        f"{len(selected) * len(args.modes)} mode runs, "
        f"{prompts * len(args.modes)} model comparisons"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    trajectories = discover_trajectories(args.data_root)
    selected = select_trajectories(trajectories, args)
    if not selected:
        raise ValueError("No trajectories selected")

    media_cache = {item.sample_id: validate_media(item) for item in selected}
    print_plan(trajectories, selected, media_cache, args)
    if args.dry_run:
        print("Dry run complete; model was not loaded and no outputs were written.")
        return 0
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from examples.inference import GRMInference
    except ModuleNotFoundError as exc:
        if exc.name == "vllm":
            raise RuntimeError(
                "vLLM is not installed in the active environment. Run "
                "`conda activate robo-dopamine` before launching this script."
            ) from exc
        raise

    model_tag = safe_tag(args.model_path.name)
    model_output = args.output_root / model_tag
    model_output.mkdir(parents=True, exist_ok=True)
    records_path = model_output / "results.json"
    records = load_records(records_path)
    config = {
        "data_root": str(args.data_root),
        "dataset_split": DATASET_SPLIT,
        "model_path": str(args.model_path),
        "task_ids": args.task_ids,
        "trajectory_indices": args.trajectory_indices,
        "sample_ids": args.sample_ids,
        "max_trajectories_per_task": args.max_trajectories_per_task,
        "modes": list(args.modes),
        "view_mode": args.view_mode,
        "frame_interval": args.frame_interval,
        "batch_size": args.batch_size,
        "goal_mode": args.goal_mode,
        "goal_image": str(args.goal_image) if args.goal_image else None,
    }

    goal_cache: dict[str, tuple[Path, str]] = {}
    for item in selected:
        goal_cache[item.sample_id] = resolve_goal(
            item, media_cache[item.sample_id], trajectories, media_cache, args
        )

    print(f"Loading model: {args.model_path}")
    model = GRMInference(str(args.model_path))
    print("Model loaded successfully.")

    total = len(selected)
    for sequence, item in enumerate(selected, 1):
        media = media_cache[item.sample_id]
        goal_path, goal_id = goal_cache[item.sample_id]
        indices = sample_indices(media.frames, args.frame_interval)
        trajectory_output = (
            model_output
            / args.view_mode
            / item.task_id
            / f"trajectory_{item.trajectory_index:03d}"
            / f"inter{args.frame_interval}_{safe_tag(goal_id, 64)}"
        )
        trajectory_output.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[{sequence}/{total}] {item.sample_id} frames={media.frames} "
            f"instruction={item.instruction!r} goal={goal_id}"
        )

        mode_paths: dict[str, Path] = {}
        pending_modes: list[str] = []
        for mode in args.modes:
            probe = {
                "model_path": str(args.model_path),
                "data_root": str(args.data_root),
                "sample_id": item.sample_id,
                "instruction": item.instruction,
                "view_mode": args.view_mode,
                "mode": mode,
                "frame_interval": args.frame_interval,
                "goal_id": goal_id,
            }
            existing = next(
                (
                    record
                    for record in records
                    if record_key(record) == record_key(probe)
                    and usable_completed_record(record)
                ),
                None,
            )
            if args.resume and existing is not None:
                print(f"  [SKIP] {mode}: already completed")
                mode_paths[mode] = Path(existing["output_dir"])
            else:
                pending_modes.append(mode)

        for mode in pending_modes:
            base_record: dict[str, Any] = {
                "model_path": str(args.model_path),
                "data_root": str(args.data_root),
                "sample_id": item.sample_id,
                "task_id": item.task_id,
                "trajectory_index": item.trajectory_index,
                "instruction": item.instruction,
                "target_obj": item.target_obj,
                "correct_target_obj": item.correct_target_obj,
                "instruction_video_match": item.instruction_video_match,
                "source_suc_id": item.source_suc_id,
                "source_trajectory_index": item.source_trajectory_index,
                "view_mode": args.view_mode,
                "mode": mode,
                "frame_count": media.frames,
                "fps": media.fps,
                "frame_interval": args.frame_interval,
                "goal_id": goal_id,
                "goal_image": str(goal_path),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            if args.view_mode == "face-only":
                high = left = right = item.face_video
            else:
                high, left, right = item.face_video, item.left_video, item.right_video
            print(f"  [{mode}] running {len(indices) - 1} comparisons")
            started = time.time()
            try:
                output_dir = Path(
                    model.run_pipeline(
                        cam_high_path=str(high),
                        cam_left_path=str(left),
                        cam_right_path=str(right),
                        out_root=str(trajectory_output),
                        task=item.instruction,
                        frame_interval=args.frame_interval,
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
                final = metrics["final_progress"]
                final_text = "n/a" if final is None else f"{final * 100:.1f}%"
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
            write_records_csv(model_output / "results.csv", records)
            if record["status"] == "failed" and args.fail_fast:
                raise RuntimeError(record["error"])

        plot_progress(
            mode_paths,
            trajectory_output / "progress_curve.png",
            indices,
            media.fps,
            f"{item.task_id}/{item.trajectory_index}: {item.instruction}",
        )

    current_keys = {
        (
            str(args.model_path),
            str(args.data_root),
            item.sample_id,
            item.instruction,
            args.view_mode,
            mode,
            args.frame_interval,
            goal_cache[item.sample_id][1],
        )
        for item in selected
        for mode in args.modes
    }
    current_records = [record for record in records if record_key(record) in current_keys]
    summary_path = model_output / "summary.json"
    atomic_write_json(summary_path, build_summary(current_records, config))
    completed = sum(record.get("status") == "completed" for record in current_records)
    failed = sum(record.get("status") == "failed" for record in current_records)
    print(f"\nEvaluation finished: completed={completed}, failed={failed}")
    print(f"Records: {records_path}")
    print(f"Summary: {summary_path}")
    return 1 if failed else 0


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
