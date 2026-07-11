"""Build GRM sample.json files from one aligned trajectory without running inference.

This is a lightweight companion to examples/inference.py for attention/steering
experiments: it extracts the same cached frames and writes sample.json, but does
not load vLLM or generate predictions. It is useful when the downstream scripts
only need the prompt/image structure.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from scan_localization_heads_best import (
    build_samples_json,
    ensure_dir,
    get_frame_count,
    make_sample_indices_by_interval,
    save_frames,
)


TASKS = {
    "carrot": "pick the carrot and put it on yellow plate",
    "cube": "pick the white cube and put it on yellow plate",
    "bottle": "pick the bottle and put it on yellow plate",
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def build_one(args: argparse.Namespace, task_key: str, task_text: str) -> Path:
    run_root = Path(args.output_root) / task_key / f"{args.eval_mode}_mode_{safe_name(task_text)}"
    cache_root = run_root / ".cache"
    cam_dirs = {
        "cam_high": cache_root / "cam_high",
        "cam_left_wrist": cache_root / "cam_left_wrist",
        "cam_right_wrist": cache_root / "cam_right_wrist",
    }
    for path in cam_dirs.values():
        ensure_dir(path)

    data_dir = Path(args.data_dir)
    paths = [
        data_dir / "cam_high.mp4",
        data_dir / "cam_left_wrist.mp4",
        data_dir / "cam_right_wrist.mp4",
    ]
    types_counts = [get_frame_count(path) for path in paths]
    counts = [count for _, count in types_counts]
    if len(set(counts)) != 1:
        raise ValueError(f"Frame count mismatch among cameras: {counts}")

    indices = make_sample_indices_by_interval(counts[0], args.frame_interval)
    if args.max_steps is not None:
        # Need one more frame than steps because each sample uses before/after.
        indices = indices[: max(2, int(args.max_steps) + 1)]

    for src, key, (src_type, _) in zip(paths, cam_dirs.keys(), types_counts):
        save_frames(src, cam_dirs[key], indices, src_type)

    ref_end_path = cache_root / "ref_end.png"
    shutil.copyfile(args.goal_image, ref_end_path)

    samples = build_samples_json(run_root, task_text, indices, str(ref_end_path), mode=args.eval_mode)
    sample_path = run_root / "sample.json"
    sample_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False))

    meta_path = run_root / "sample_meta.json"
    meta_path.write_text(json.dumps({
        "data_dir": args.data_dir,
        "task_key": task_key,
        "task": task_text,
        "eval_mode": args.eval_mode,
        "frame_interval": args.frame_interval,
        "indices": indices,
        "num_samples": len(samples),
        "goal_image": args.goal_image,
    }, indent=2, ensure_ascii=False))
    return sample_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Create sample.json files from an aligned success trajectory")
    ap.add_argument("--data-dir", default="/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot")
    ap.add_argument("--output-root", default="./results/attention/success_pick3suc_1_carrot_samples")
    ap.add_argument("--goal-image", default="./examples/blank_goal.png")
    ap.add_argument("--frame-interval", type=int, default=20)
    ap.add_argument("--eval-mode", default="forward", choices=["forward", "incremental", "backward"])
    ap.add_argument("--tasks", nargs="+", default=["carrot", "cube", "bottle"], choices=sorted(TASKS))
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    out = []
    for task_key in args.tasks:
        sample_path = build_one(args, task_key, TASKS[task_key])
        out.append(str(sample_path))
        print(f"[make-success-samples] {task_key}: {sample_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
