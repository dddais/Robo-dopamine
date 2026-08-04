#!/usr/bin/env python3
"""Evaluate Robo-Dopamine on ``data/ljx_lfz_task/new/fail``.

The failure set contains counterfactual instruction/video pairs: each video is
a successful execution for ``correct_target_obj``, while the evaluation
instruction requests a different ``target_obj``. A high progress score is
therefore a false positive. The script reuses the synchronized three-view
evaluation core from ``auto_eval_ljx_lfz_suc.py`` and preserves failure/source
pairing metadata in every result record.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import auto_eval_ljx_lfz_suc as core


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/dais/workspace/data/ljx_lfz_task/new/fail")
DEFAULT_OUTPUT = SCRIPT_DIR / "results/ljx_lfz_fail_eval"

_discover_success_trajectories = core.discover_trajectories
_resolve_standard_goal = core.resolve_goal
_success_cache: dict[Path, list[core.Trajectory]] = {}


def discover_fail_trajectories(data_root: Path) -> list[core.Trajectory]:
    """Read and strictly validate the counterfactual failure metadata."""
    metadata_path = data_root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    rows = core.read_jsonl(metadata_path)
    if not rows:
        raise ValueError(f"No records found in {metadata_path}")

    required = {
        "id",
        "split",
        "task_id",
        "trajectory_index",
        "instruction",
        "target_obj",
        "correct_target_obj",
        "instruction_video_match",
        "source_suc_id",
        "video_paths",
    }
    result: list[core.Trajectory] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"Failure metadata row {row_number} is missing fields: {sorted(missing)}"
            )
        sample_id = str(row["id"])
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate failure metadata ID: {sample_id}")
        seen_ids.add(sample_id)
        if row["split"] != "fail":
            raise ValueError(f"Unexpected split for {sample_id}: {row['split']!r}")
        if row["instruction_video_match"] is not False:
            raise ValueError(
                f"Failure sample is not marked as an instruction/video mismatch: {sample_id}"
            )

        target_obj = str(row["target_obj"]).strip()
        correct_target_obj = str(row["correct_target_obj"]).strip()
        if target_obj == correct_target_obj:
            raise ValueError(
                f"Failure target unexpectedly equals the video's true target for {sample_id}"
            )
        source_suc_id = str(row["source_suc_id"])
        if not source_suc_id.startswith("suc/"):
            raise ValueError(f"Invalid source_suc_id for {sample_id}: {source_suc_id}")

        try:
            failure_index = int(Path(sample_id).name)
        except ValueError as exc:
            raise ValueError(f"Failure ID must end in a numeric index: {sample_id}") from exc
        if failure_index <= 0:
            raise ValueError(f"Failure index must be positive: {sample_id}")

        video_paths = list(row["video_paths"])
        if len(video_paths) != 3:
            raise ValueError(f"Expected three videos for {sample_id}, got {video_paths}")
        by_name = {
            Path(path).name: core.resolve_metadata_video(data_root, str(path))
            for path in video_paths
        }
        expected_names = {"faceImg.mp4", "leftImg.mp4", "rightImg.mp4"}
        if set(by_name) != expected_names:
            raise ValueError(f"Unexpected video names for {sample_id}: {sorted(by_name)}")

        result.append(
            core.Trajectory(
                sample_id=sample_id,
                task_id=str(row["task_id"]),
                # For failure data, the final path component is the unique
                # evaluation index. metadata.trajectory_index identifies the
                # source successful trajectory and is stored separately.
                trajectory_index=failure_index,
                instruction=str(row["instruction"]).strip(),
                target_obj=target_obj,
                face_video=by_name["faceImg.mp4"],
                left_video=by_name["leftImg.mp4"],
                right_video=by_name["rightImg.mp4"],
                dataset_split="fail",
                correct_target_obj=correct_target_obj,
                instruction_video_match=False,
                source_suc_id=source_suc_id,
                source_trajectory_index=int(row["trajectory_index"]),
            )
        )

    by_task: dict[str, list[int]] = defaultdict(list)
    for item in result:
        by_task[item.task_id].append(item.trajectory_index)
    for task_id, indices in by_task.items():
        expected = list(range(1, len(indices) + 1))
        if sorted(indices) != expected:
            raise ValueError(
                f"Failure indices are not contiguous for {task_id}: "
                f"expected 1..{len(indices)}"
            )
    return sorted(
        result,
        key=lambda item: (core.natural_key(item.task_id), item.trajectory_index),
    )


def resolve_fail_goal(
    trajectory: core.Trajectory,
    media: core.VideoInfo,
    all_trajectories: Sequence[core.Trajectory],
    media_cache: dict[str, core.VideoInfo],
    args: Any,
) -> tuple[Path, str]:
    """Use success-set references for reference-end failure evaluation."""
    if args.goal_mode != "reference-end" or args.goal_image is not None:
        return _resolve_standard_goal(
            trajectory, media, all_trajectories, media_cache, args
        )

    success_root = (args.data_root.parent / "suc").resolve()
    if success_root not in _success_cache:
        _success_cache[success_root] = _discover_success_trajectories(success_root)
    candidates = [
        item
        for item in _success_cache[success_root]
        if item.task_id == trajectory.task_id
        and item.instruction == trajectory.instruction
    ]
    if not candidates:
        raise ValueError(
            f"No successful reference with the same task/instruction as {trajectory.sample_id}"
        )
    candidates.sort(key=lambda item: item.trajectory_index)
    reference = candidates[(len(candidates) - 1) // 2]
    if reference.sample_id not in media_cache:
        media_cache[reference.sample_id] = core.validate_media(reference)
    reference_media = media_cache[reference.sample_id]
    goal_path = (
        args.output_root
        / "_goals"
        / core.safe_tag(reference.task_id)
        / (
            f"success_trajectory_{reference.trajectory_index:03d}_"
            f"{core.safe_tag(reference.instruction, 48)}.png"
        )
    )
    if not goal_path.is_file():
        core.extract_frame(
            reference.face_video, reference_media.frames - 1, goal_path
        )
    return goal_path, f"reference-suc:{reference.sample_id}"


def configure_shared_core() -> None:
    core.DEFAULT_DATA_ROOT = DEFAULT_DATA_ROOT
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.DATASET_DISPLAY_NAME = "ljx_lfz failure-set"
    core.DATASET_SPLIT = "fail"
    core.discover_trajectories = discover_fail_trajectories
    core.resolve_goal = resolve_fail_goal


def main(argv: Sequence[str] | None = None) -> int:
    configure_shared_core()
    return core.main(argv)


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
