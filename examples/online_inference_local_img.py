#!/usr/bin/env python3
"""Online GRM inference from local image directory — periodically reads
fixed-name camera images from ``/tmp/img/`` and performs real-time progress
estimation.

Expected files under ``--img-dir`` (default ``/tmp/img/``)::

    /tmp/img/
        cam_high.png
        cam_left.png
        cam_right.png

The external system continuously overwrites these three files in-place.
This script reads them at a fixed interval and runs GRM inference each cycle.

Usage:
    # Start with defaults:
    python online_inference_local_img.py

    # Custom task, model, and interval:
    python online_inference_local_img.py \
        --task "pick the bottle and put it into the bag" \
        --model-path /path/to/GRM/model \
        --interval 1.0
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = (
    "/home/ubuntu/dais/Robo-dopamine/pretrained_models/"
    "Robo-Dopamine-GRM-2.0-4B-Preview"
)
DEFAULT_TASK = "pick the bowl and put it into the basket"
DEFAULT_GOAL_IMAGE = str(REPO_ROOT / "examples" / "blank_goal.png")
DEFAULT_OUT_ROOT = "./results/online_local_img"
DEFAULT_IMG_DIR = "/tmp/img"
DEFAULT_INTERVAL = 1.0
# Source filenames in /tmp/img/  →  GRM camera keys used in cache & prompts
IMG_FILES = {
    "cam_high": "extra_view_images_0.jpg",
    "cam_left_wrist": "main_images.jpg",
    "cam_right_wrist": "extra_view_images_1.jpg",
}
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
VALID_MODES = ("forward", "incremental", "backward")


# ---------------------------------------------------------------------------
# Progress tracking (same logic as online_inference_ray.py)
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ProgressTracker:
    def __init__(self):
        self.prev_progress = {mode: 0.0 for mode in VALID_MODES}
        self.counts = {mode: 0 for mode in VALID_MODES}

    def update(self, mode: str, score: float) -> Dict[str, float]:
        prev = self.prev_progress[mode]

        if mode == "incremental":
            if self.counts[mode] == 0:
                progress = score
            elif score >= 0:
                progress = prev + (1.0 - prev) * score
            else:
                progress = prev + prev * score
            hop = score
        elif mode == "forward":
            progress = score
            hop = progress - prev
        elif mode == "backward":
            progress = clamp(1.0 + score, 0.0, 1.0)
            hop = progress - prev
        else:
            raise ValueError(f"Unknown eval mode: {mode}")

        self.prev_progress[mode] = progress
        self.counts[mode] += 1
        return {"score": score, "hop": hop, "progress": progress}


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def sanitize_task(task: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", task).strip("_")
    return safe[:80] or "task"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def append_jsonl(path: Path, data: object) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def finalize_jsonl(jsonl_path: Path, out_path: Path) -> None:
    if not jsonl_path.exists():
        return
    items = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    write_json(out_path, items)
    print(f"[LOCAL-IMG] Wrote summary JSON: {out_path}")


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def parse_score(pred_text: str) -> float:
    try:
        match = re.search(r"<score>(.*?)</score>", pred_text)
        if match:
            value = match.group(1).replace("%", "").strip()
        else:
            matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", pred_text)
            value = matches[-1] if matches else "0"
        return clamp(float(value), -100.0, 100.0) / 100.0
    except Exception:
        return 0.0


def copy_goal_image(goal_image: str, cache_root: Path) -> Path:
    src = Path(goal_image).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Goal image not found: {src}")
    dst = cache_root / "ref_end.png"
    shutil.copyfile(src, dst)
    return dst


def create_run_dirs(out_root: str, task: str) -> Dict[str, Path]:
    ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run_root = Path(out_root).expanduser().resolve() / f"{ts}_img_{sanitize_task(task)}"
    cache_root = run_root / ".cache"
    dirs = {"run_root": run_root, "cache_root": cache_root}
    for key in CAMERA_KEYS:
        dirs[key] = cache_root / key
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def snapshot_current(img_dir: Path, dirs: Dict[str, Path], step: int) -> Dict[str, str]:
    """Copy current images from img_dir into cache, return saved paths."""
    saved = {}
    for key in CAMERA_KEYS:
        src = img_dir / IMG_FILES[key]
        out_path = dirs[key] / f"frame_{step:06d}.png"
        shutil.copyfile(str(src), str(out_path))
        saved[key] = str(out_path)
    return saved


def build_online_samples(
    task: str,
    step: int,
    ref_start: Dict[str, str],
    ref_end_path: Path,
    previous: Dict[str, str],
    current: Dict[str, str],
) -> List[Dict]:
    samples = []
    for mode in VALID_MODES:
        if mode == "incremental":
            before = previous
            before_id = f"prev_{step - 1:06d}"
        elif mode == "forward":
            before = ref_start
            before_id = "start_000000"
        elif mode == "backward":
            before = {
                "cam_high": str(ref_end_path),
                "cam_left_wrist": str(ref_end_path),
                "cam_right_wrist": str(ref_end_path),
            }
            before_id = "goal"
        else:
            raise ValueError(f"Unknown eval mode: {mode}")

        samples.append(
            {
                "id": f"local-img-{mode}-step_{step:06d}-{before_id}-af_{step:06d}",
                "task": task,
                "eval_mode": mode,
                "image": [
                    ref_start["cam_high"],
                    str(ref_end_path),
                    before["cam_high"],
                    before["cam_left_wrist"],
                    before["cam_right_wrist"],
                    current["cam_high"],
                    current["cam_left_wrist"],
                    current["cam_right_wrist"],
                ],
            }
        )
    return samples


def make_mode_result(
    mode: str, pred_text: str, tracker: ProgressTracker
) -> Dict[str, object]:
    score = parse_score(pred_text)
    stats = tracker.update(mode, score)
    return {
        "pred": pred_text,
        "score": stats["score"],
        "hop": stats["hop"],
        "progress": stats["progress"],
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online Robo-Dopamine GRM inference from local image directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img-dir",
        default=DEFAULT_IMG_DIR,
        help="Directory containing cam_high.png, cam_left.png, cam_right.png.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="GRM checkpoint or HF model path.",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task instruction used in the GRM prompt.",
    )
    parser.add_argument(
        "--goal-image",
        default=DEFAULT_GOAL_IMAGE,
        help="Goal/reference image path for ref_end and backward mode.",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help="Root directory for inference outputs.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="Seconds between inference steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum inference steps; 0 means run until interrupted.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_online(args: argparse.Namespace) -> None:
    img_dir = Path(args.img_dir).resolve()

    # Verify source images exist
    for key, fname in IMG_FILES.items():
        p = img_dir / fname
        if not p.exists():
            print(f"[LOCAL-IMG] Missing: {p}")
            print(f"[LOCAL-IMG] Please ensure {fname} exists in {img_dir}")
            raise SystemExit(1)

    # --- Load GRM model ---
    print(f"[LOCAL-IMG] Loading GRM model: {args.model_path}")
    from examples.inference import GRMInference

    model = GRMInference(args.model_path)
    print("[LOCAL-IMG] Model loaded.")

    # --- Prepare output directories ---
    dirs = create_run_dirs(args.out_root, args.task)
    ref_end_path = copy_goal_image(args.goal_image, dirs["cache_root"])
    jsonl_path = dirs["run_root"] / "online_pred.jsonl"
    latest_path = dirs["run_root"] / "latest_progress.json"
    summary_path = dirs["run_root"] / "pred_vllm_online_local_img.json"

    metadata = {
        "data_source": "local_img",
        "img_dir": str(img_dir),
        "model_path": args.model_path,
        "task": args.task,
        "goal_image": str(Path(args.goal_image).expanduser().resolve()),
        "ref_end": str(ref_end_path),
        "interval": args.interval,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(dirs["run_root"] / "metadata.json", metadata)

    tracker = ProgressTracker()

    # --- First snapshot as reference start ---
    print("[LOCAL-IMG] Reading initial frames ...")
    ref_start = snapshot_current(img_dir, dirs, step=0)
    previous = ref_start

    initial = {
        "step": 0,
        "wall_time": time.time(),
        "latency_s": 0.0,
        "task": args.task,
        "progress": 0.0,
        "progress_percent": 0.0,
        "modes": {
            mode: {"pred": "<score>0%</score>", "score": 0.0, "hop": 0.0, "progress": 0.0}
            for mode in VALID_MODES
        },
        "frames": ref_start,
    }
    append_jsonl(jsonl_path, initial)
    write_json(latest_path, initial)
    print("[LOCAL-IMG] Step 0 saved as reference start.")

    # --- Main inference loop ---
    step = 1
    try:
        while True:
            if args.max_steps > 0 and step > args.max_steps:
                print(f"[LOCAL-IMG] Reached max steps ({args.max_steps}). Stopping.")
                break

            time.sleep(args.interval)
            infer_start = time.time()

            current = snapshot_current(img_dir, dirs, step=step)

            samples = build_online_samples(
                task=args.task,
                step=step,
                ref_start=ref_start,
                ref_end_path=ref_end_path,
                previous=previous,
                current=current,
            )

            outputs = model.inference_batch(samples)
            mode_results = {}
            for item in outputs:
                mode = item["eval_mode"]
                mode_results[mode] = make_mode_result(
                    mode, item.get("pred", ""), tracker
                )

            fused = clamp(
                sum(float(mode_results[m]["progress"]) for m in VALID_MODES)
                / len(VALID_MODES),
                0.0,
                1.0,
            )

            latency = time.time() - infer_start
            record = {
                "step": step,
                "wall_time": time.time(),
                "latency_s": latency,
                "task": args.task,
                "progress": fused,
                "progress_percent": fused * 100.0,
                "modes": mode_results,
                "frames": current,
            }
            append_jsonl(jsonl_path, record)
            write_json(latest_path, record)

            print(
                "[LOCAL-IMG] Step {step:06d} "
                "fused={fused:6.2f}% "
                "forward={forward:6.2f}% incremental={incremental:6.2f}% "
                "backward={backward:6.2f}% latency={latency:.2f}s".format(
                    step=step,
                    fused=fused * 100.0,
                    forward=float(mode_results["forward"]["progress"]) * 100.0,
                    incremental=float(mode_results["incremental"]["progress"]) * 100.0,
                    backward=float(mode_results["backward"]["progress"]) * 100.0,
                    latency=latency,
                )
            )

            previous = current
            step += 1

    except KeyboardInterrupt:
        print("\n[LOCAL-IMG] Interrupted by user.")
    finally:
        finalize_jsonl(jsonl_path, summary_path)
        print(f"[LOCAL-IMG] Output directory: {dirs['run_root']}")


def main() -> None:
    args = parse_args()
    run_online(args)


if __name__ == "__main__":
    main()
