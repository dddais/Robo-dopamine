#!/usr/bin/env python3
"""Online GRM inference via Ray — replaces ROS1 camera topics with
``DataCollector.get_obs()`` calls to a running Ray cluster.

Usage:
    # On any machine that can reach the Ray head node:
    python online_inference_ray.py --address 192.168.120.143:6379

    # Running on the head node itself:
    python online_inference_ray.py

    # Custom task and model:
    python online_inference_ray.py \\
        --task "pick the bottle and put it into the bag" \\
        --model-path /path/to/GRM/model
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
from typing import Dict, List, Optional

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
DEFAULT_ADDRESS = "192.168.120.143:6379"
DEFAULT_ACTOR_NAME = "EnvGroup:0"
DEFAULT_TASK = "pick the bowl and put it into the bag"
DEFAULT_GOAL_IMAGE = "./blank_goal.png"
DEFAULT_OUT_ROOT = "./results/online_ray"
DEFAULT_SAMPLE_PERIOD = 1.0

# Mapping from DualFrankaJointEnv observation keys → GRM camera keys.
# The env returns {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
# while the GRM prompt expects {"cam_high", "cam_left_wrist", "cam_right_wrist"}.
FRAME_KEY_MAP = {
    "base_0_rgb": "cam_high",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
VALID_MODES = ("forward", "incremental", "backward")


# ---------------------------------------------------------------------------
# Ray helpers
# ---------------------------------------------------------------------------

def connect_ray(address: Optional[str], actor_name: str):
    """Connect to Ray and return a handle to the named DataCollector actor."""
    import ray

    init_kwargs = {}
    if address:
        init_kwargs["address"] = address
    ray.init(**init_kwargs)

    try:
        collector = ray.get_actor(actor_name)
    except ValueError:
        available = ray.util.list_named_actors()
        print(
            f"Actor '{actor_name}' not found. "
            f"Is the DataCollector running? "
            f"Available actors: {available}"
        )
        ray.shutdown()
        raise SystemExit(1)

    print(f"[RAY] Connected to actor '{actor_name}'.")
    return ray, collector


def get_obs_once(collector_handle) -> dict:
    """Call ``get_obs.remote()`` and return the raw observation dict."""
    obs = ray.get(collector_handle.get_obs.remote())
    if obs is None:
        raise RuntimeError(
            "get_obs() returned None — the environment may not be ready yet."
        )
    return obs


def extract_frames(obs: dict) -> Dict[str, np.ndarray]:
    """Extract and rename camera frames from a raw observation dict.

    Returns a dict keyed by GRM camera names (cam_high, cam_left_wrist,
    cam_right_wrist) with BGR uint8 ndarrays ready for ``cv2.imwrite``.
    """
    raw_frames = obs.get("frames", {})
    frames = {}
    for env_key, grm_key in FRAME_KEY_MAP.items():
        arr = raw_frames.get(env_key)
        if arr is None:
            raise KeyError(
                f"Frame key '{env_key}' not found in observation. "
                f"Available keys: {list(raw_frames.keys())}"
            )
        # Env returns RGB; convert to BGR for OpenCV.
        frames[grm_key] = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return frames


# ---------------------------------------------------------------------------
# Progress tracking (identical to online_inference_ros1.py)
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ProgressTracker:
    """Applies the same per-mode progress formulas as ``inference.py``."""

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
    print(f"[ONLINE-RAY] Wrote summary JSON: {out_path}")


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def parse_score(pred_text: str) -> float:
    """Return score in [-1, 1] from a model response."""
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
    run_root = Path(out_root).expanduser().resolve() / f"{ts}_ray_{sanitize_task(task)}"
    cache_root = run_root / ".cache"
    dirs = {
        "run_root": run_root,
        "cache_root": cache_root,
        "cam_high": cache_root / "cam_high",
        "cam_left_wrist": cache_root / "cam_left_wrist",
        "cam_right_wrist": cache_root / "cam_right_wrist",
    }
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def save_frame_triplet(
    frames: Dict[str, np.ndarray], dirs: Dict[str, Path], step: int
) -> Dict[str, str]:
    saved = {}
    for key in CAMERA_KEYS:
        out_path = dirs[key] / f"frame_{step:06d}.png"
        ok = cv2.imwrite(str(out_path), frames[key], [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        if not ok:
            raise RuntimeError(f"Failed to write image: {out_path}")
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
                "id": f"online-ray-{mode}-step_{step:06d}-{before_id}-af_{step:06d}",
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
# Main loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online Robo-Dopamine GRM inference via Ray DataCollector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--address",
        default=DEFAULT_ADDRESS,
        help="Ray cluster address (e.g. 192.168.120.143:6379). "
        "Omit to connect to a local Ray instance.",
    )
    parser.add_argument(
        "--actor-name",
        default=DEFAULT_ACTOR_NAME,
        help="Name of the DataCollector Ray actor.",
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
        help="Root directory for online inference outputs.",
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        default=DEFAULT_SAMPLE_PERIOD,
        help="Minimum seconds between inference samples.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum online inference steps; 0 means run forever.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the first observation from the Ray actor.",
    )
    return parser.parse_args()


def run_online(args: argparse.Namespace) -> None:
    if args.sample_period <= 0:
        raise ValueError("--sample-period must be > 0")

    # --- Load GRM model (before Ray init to avoid GPU conflicts) -----------
    print(f"[ONLINE-RAY] Loading GRM model: {args.model_path}")
    from examples.inference import GRMInference

    model = GRMInference(args.model_path)
    print("[ONLINE-RAY] Model loaded.")

    # --- Connect to Ray cluster --------------------------------------------
    global ray
    ray_module, collector = connect_ray(args.address, args.actor_name)

    # Monkey-patch module-level name used by ``get_obs_once``.
    globals()["ray"] = ray_module

    # --- Prepare output directories ----------------------------------------
    dirs = create_run_dirs(args.out_root, args.task)
    ref_end_path = copy_goal_image(args.goal_image, dirs["cache_root"])
    jsonl_path = dirs["run_root"] / "online_pred.jsonl"
    latest_path = dirs["run_root"] / "latest_progress.json"
    summary_path = dirs["run_root"] / "pred_vllm_online_ray.json"

    metadata = {
        "data_source": "ray",
        "ray_address": args.address,
        "actor_name": args.actor_name,
        "model_path": args.model_path,
        "task": args.task,
        "goal_image": str(Path(args.goal_image).expanduser().resolve()),
        "ref_end": str(ref_end_path),
        "sample_period": args.sample_period,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(dirs["run_root"] / "metadata.json", metadata)

    tracker = ProgressTracker()

    # --- Fetch first observation as reference start -------------------------
    print("[ONLINE-RAY] Waiting for first observation from DataCollector...")
    t0 = time.time()
    while True:
        try:
            obs = get_obs_once(collector)
            break
        except RuntimeError:
            if time.time() - t0 > args.wait_timeout:
                print("[ONLINE-RAY] Timed out waiting for first observation.")
                ray_module.shutdown()
                raise SystemExit(1)
            time.sleep(0.5)

    first_frames = extract_frames(obs)
    ref_start = save_frame_triplet(first_frames, dirs, step=0)
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
    print("[ONLINE-RAY] Step 000000 progress=0.00% saved as reference start.")

    # --- Main inference loop ------------------------------------------------
    step = 1
    try:
        while True:
            if args.max_steps > 0 and step > args.max_steps:
                break

            time.sleep(args.sample_period)
            infer_start = time.time()

            obs = get_obs_once(collector)
            current_frames = extract_frames(obs)
            current = save_frame_triplet(current_frames, dirs, step=step)

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
                sum(
                    float(mode_results[m]["progress"]) for m in VALID_MODES
                )
                / len(VALID_MODES),
                0.0,
                1.0,
            )
            record = {
                "step": step,
                "wall_time": time.time(),
                "latency_s": time.time() - infer_start,
                "task": args.task,
                "progress": fused,
                "progress_percent": fused * 100.0,
                "modes": mode_results,
                "frames": current,
            }
            append_jsonl(jsonl_path, record)
            write_json(latest_path, record)

            print(
                "[ONLINE-RAY] Step {step:06d} fused={fused:6.2f}% "
                "forward={forward:6.2f}% incremental={incremental:6.2f}% "
                "backward={backward:6.2f}% latency={latency:.2f}s".format(
                    step=step,
                    fused=fused * 100.0,
                    forward=float(mode_results["forward"]["progress"]) * 100.0,
                    incremental=float(mode_results["incremental"]["progress"]) * 100.0,
                    backward=float(mode_results["backward"]["progress"]) * 100.0,
                    latency=record["latency_s"],
                )
            )

            previous = current
            step += 1

    except KeyboardInterrupt:
        print("\n[ONLINE-RAY] Interrupted by user.")
    finally:
        finalize_jsonl(jsonl_path, summary_path)
        print(f"[ONLINE-RAY] Output directory: {dirs['run_root']}")
        ray_module.shutdown()


def main() -> None:
    args = parse_args()
    run_online(args)


if __name__ == "__main__":
    main()
