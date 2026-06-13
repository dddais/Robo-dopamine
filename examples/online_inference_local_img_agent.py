#!/usr/bin/env python3
"""Agent-style online GRM inference with fisheye undistortion (vLLM).

Based on ``online_inference_local_img_multi_fisheye.py`` with the following
agent-style changes:

1. **Subtask source** — reads the current subtask from a file
   (``/tmp/subtask.txt`` by default).  When the file content changes, the
   agent automatically switches to the new subtask and resets all tracking
   state.

2. **Automatic success/fail detection** — monitors fused progress and writes
   the current status (``running`` / ``success`` / ``failed``) to
   ``/tmp/monitor_result.txt``.  Once ``success`` or ``fail`` is triggered,
   inference for the current subtask stops and the agent waits for a new
   subtask to appear in the file.

   * **success** — progress exceeds a threshold and stays stable (no
     significant change) for N consecutive steps.
   * **fail** — progress stays below the success threshold and plateaus
     (no meaningful improvement) for M consecutive steps.

Usage:
    python online_inference_local_img_agent.py

    python online_inference_local_img_agent.py \\
        --subtask-file /tmp/subtask.txt \\
        --result-file /tmp/monitor_result.txt \\
        --subtask-check-interval 0.5 \\
        --success-threshold 0.90 \\
        --success-stable-steps 10 \\
        --success-max-drift 0.02 \\
        --fail-stable-steps 20 \\
        --fail-min-progress 0.01
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
from typing import Dict, List, Optional, Tuple

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
DEFAULT_GOAL_IMAGE = str(REPO_ROOT / "examples" / "blank_goal.png")
DEFAULT_OUT_ROOT = "./results/online_local_img_agent"
DEFAULT_IMG_DIR = "/tmp/img"
DEFAULT_INTERVAL = 1.0
DEFAULT_FISHEYE_CONFIG = "/home/ubuntu/dais/fisheye_process/config.yaml"

DEFAULT_SUBTASK_FILE = "/tmp/subtask.txt"
DEFAULT_RESULT_FILE = "/tmp/monitor_result.txt"
DEFAULT_SUBTASK_CHECK_INTERVAL = 0.5

# Success / fail detection defaults
DEFAULT_SUCCESS_THRESHOLD = 0.60
DEFAULT_SUCCESS_STABLE_STEPS = 5
DEFAULT_SUCCESS_MAX_DRIFT = 0.02
DEFAULT_FAIL_STABLE_STEPS = 8
DEFAULT_FAIL_MIN_PROGRESS = 0.01

IMG_FILES = {
    "cam_high": "base_0_rgb.jpg",
    "cam_left_wrist": "left_wrist_0_rgb.jpg",
    "cam_right_wrist": "right_wrist_0_rgb.jpg",
}
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FISHEYE_KEYS = ("cam_left_wrist", "cam_right_wrist")
VALID_MODES = ("forward", "incremental", "backward")

MONITOR_STATUS_RUNNING = "running"
MONITOR_STATUS_SUCCESS = "success"
MONITOR_STATUS_FAIL = "failed"


# ---------------------------------------------------------------------------
# Fisheye undistortion (reuses fisheye_process/convert.py)
# ---------------------------------------------------------------------------

def init_fisheye_remap(config_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build the fisheye→pinhole remap table once."""
    sys.path.insert(0, str(Path(config_path).resolve().parent))
    from convert import (
        build_remap_table,
        compute_extrinsics,
        get_border_flag,
        get_interp_flag,
        load_config,
        load_pinhole_intrinsics,
    )

    cfg = load_config(config_path)
    config_dir = str(Path(config_path).resolve().parent)
    load_pinhole_intrinsics(cfg, config_dir)
    compute_extrinsics(cfg, config_dir)

    cfg.setdefault("depth", {})
    cfg["depth"]["enabled"] = False

    map_x, map_y = build_remap_table(cfg)
    out_w = cfg["pinhole"]["image_width"]
    out_h = cfg["pinhole"]["image_height"]

    proc = cfg.get("processing", {})
    interp = get_interp_flag(proc.get("interpolation", "LINEAR"))
    border_mode = get_border_flag(proc.get("border_mode", "CONSTANT"))
    border_value = proc.get("border_value", 0)

    print(f"[AGENT] Fisheye remap table built: {out_w}x{out_h}, "
          f"interp={proc.get('interpolation', 'LINEAR')}, "
          f"border={proc.get('border_mode', 'CONSTANT')}")

    return map_x, map_y, out_w, out_h, interp, border_mode, border_value


def undistort_fisheye(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interp: int,
    border_mode: int,
    border_value: int,
) -> np.ndarray:
    """Apply pre-computed fisheye→pinhole remap to an image."""
    return cv2.remap(
        img, map_x, map_y, interp,
        borderMode=border_mode,
        borderValue=(border_value, border_value, border_value),
    )


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ProgressTracker:
    def __init__(self):
        self.prev_progress = {mode: 0.0 for mode in VALID_MODES}
        self.counts = {mode: 0 for mode in VALID_MODES}

    def reset(self):
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
# Monitor — success / fail detection
# ---------------------------------------------------------------------------

class MonitorState:
    """Tracks progress history and determines success / fail / running."""

    def __init__(
        self,
        success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
        success_stable_steps: int = DEFAULT_SUCCESS_STABLE_STEPS,
        success_max_drift: float = DEFAULT_SUCCESS_MAX_DRIFT,
        fail_stable_steps: int = DEFAULT_FAIL_STABLE_STEPS,
        fail_min_progress: float = DEFAULT_FAIL_MIN_PROGRESS,
    ):
        self.success_threshold = success_threshold
        self.success_stable_steps = success_stable_steps
        self.success_max_drift = success_max_drift
        self.fail_stable_steps = fail_stable_steps
        self.fail_min_progress = fail_min_progress
        self.reset()

    def reset(self):
        self.status = MONITOR_STATUS_RUNNING
        self.progress_history: List[float] = []
        self._success_counter = 0
        self._fail_counter = 0

    def update(self, fused_progress: float) -> str:
        """Feed a new fused progress value and return the current status."""
        if self.status != MONITOR_STATUS_RUNNING:
            return self.status

        self.progress_history.append(fused_progress)

        # --- Success check ---
        if fused_progress >= self.success_threshold:
            recent = self.progress_history[-self.success_stable_steps:]
            if len(recent) >= self.success_stable_steps:
                max_val = max(recent)
                min_val = min(recent)
                drift = max_val - min_val
                if drift <= self.success_max_drift:
                    self.status = MONITOR_STATUS_SUCCESS
                    self._success_counter = len(recent)
                    return self.status
            self._success_counter += 1
            self._fail_counter = 0
        else:
            self._success_counter = 0

            # --- Fail check (plateau below success threshold) ---
            recent = self.progress_history[-self.fail_stable_steps:]
            if len(recent) >= self.fail_stable_steps:
                has_improvement = False
                for i in range(1, len(recent)):
                    if recent[i] - recent[i - 1] >= self.fail_min_progress:
                        has_improvement = True
                        break
                if not has_improvement:
                    self.status = MONITOR_STATUS_FAIL
                    self._fail_counter = len(recent)
                    return self.status
            self._fail_counter += 1

        return self.status

    @property
    def is_finished(self) -> bool:
        return self.status in (MONITOR_STATUS_SUCCESS, MONITOR_STATUS_FAIL)


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
    print(f"[AGENT] Wrote summary JSON: {out_path}")


# ---------------------------------------------------------------------------
# Subtask file reader
# ---------------------------------------------------------------------------

def read_subtask_file(path: str) -> Optional[str]:
    """Read the current subtask from file.  Returns None if empty / missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content if content else None
    except FileNotFoundError:
        return None


def write_monitor_result(path: str, status: str) -> None:
    """Atomically write the monitor status to the result file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(status + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


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


def snapshot_current_with_undistort(
    img_dir: Path,
    dirs: Dict[str, Path],
    step: int,
    fisheye_map_x: np.ndarray,
    fisheye_map_y: np.ndarray,
    fisheye_interp: int,
    fisheye_border_mode: int,
    fisheye_border_value: int,
) -> Dict[str, str]:
    """Read images from img_dir, undistort fisheye cameras, save to cache."""
    saved = {}
    for key in CAMERA_KEYS:
        src_path = img_dir / IMG_FILES[key]
        img = cv2.imread(str(src_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {src_path}")

        if key in FISHEYE_KEYS:
            img = undistort_fisheye(
                img, fisheye_map_x, fisheye_map_y,
                fisheye_interp, fisheye_border_mode, fisheye_border_value,
            )

        out_path = dirs[key] / f"frame_{step:06d}.png"
        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        saved[key] = str(out_path)
    return saved


def build_online_samples(
    task: str,
    step: int,
    ref_start: Dict[str, str],
    ref_end_path: Path,
    previous: Dict[str, str],
    current: Dict[str, str],
    modes: List[str] = None,
) -> List[Dict]:
    if modes is None:
        modes = list(VALID_MODES)
    samples = []
    for mode in modes:
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
                "id": f"agent-{mode}-step_{step:06d}-{before_id}-af_{step:06d}",
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
        description="Agent-style online GRM inference with fisheye undistortion "
                    "and automatic success/fail detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Original arguments ---
    parser.add_argument(
        "--img-dir",
        default=DEFAULT_IMG_DIR,
        help="Directory containing the three camera image files.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="GRM checkpoint or HF model path.",
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
        "--fisheye-config",
        default=DEFAULT_FISHEYE_CONFIG,
        help="Path to fisheye_process/config.yaml for undistortion parameters.",
    )
    parser.add_argument(
        "--no-backward",
        action="store_true",
        help="Exclude backward mode from inference and fused progress calculation.",
    )

    # --- Agent-specific arguments ---
    parser.add_argument(
        "--subtask-file",
        default=DEFAULT_SUBTASK_FILE,
        help="Path to the file containing the current subtask description.",
    )
    parser.add_argument(
        "--result-file",
        default=DEFAULT_RESULT_FILE,
        help="Path to write the monitor status (running/success/failed).",
    )
    parser.add_argument(
        "--subtask-check-interval",
        type=float,
        default=DEFAULT_SUBTASK_CHECK_INTERVAL,
        help="Seconds between checks for subtask file changes.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=DEFAULT_SUCCESS_THRESHOLD,
        help="Fused progress must exceed this value to be considered for success.",
    )
    parser.add_argument(
        "--success-stable-steps",
        type=int,
        default=DEFAULT_SUCCESS_STABLE_STEPS,
        help="Number of consecutive steps progress must stay stable for success.",
    )
    parser.add_argument(
        "--success-max-drift",
        type=float,
        default=DEFAULT_SUCCESS_MAX_DRIFT,
        help="Maximum allowed progress drift (max-min) over stable window for success.",
    )
    parser.add_argument(
        "--fail-stable-steps",
        type=int,
        default=DEFAULT_FAIL_STABLE_STEPS,
        help="Number of consecutive steps with no improvement to trigger fail.",
    )
    parser.add_argument(
        "--fail-min-progress",
        type=float,
        default=DEFAULT_FAIL_MIN_PROGRESS,
        help="Minimum per-step progress increase to count as improvement.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_online(args: argparse.Namespace) -> None:
    img_dir = Path(args.img_dir).resolve()
    active_modes = [m for m in VALID_MODES if not (args.no_backward and m == "backward")]
    if args.no_backward:
        print(f"[AGENT] Backward mode disabled. Active modes: {active_modes}")

    # Verify source images
    for key, fname in IMG_FILES.items():
        p = img_dir / fname
        if not p.exists():
            print(f"[AGENT] Missing: {p}")
            print(f"[AGENT] Please ensure {fname} exists in {img_dir}")
            raise SystemExit(1)

    # --- Build fisheye remap table (once) ---
    print(f"[AGENT] Loading fisheye config: {args.fisheye_config}")
    map_x, map_y, out_w, out_h, f_interp, f_border_mode, f_border_value = \
        init_fisheye_remap(args.fisheye_config)

    # --- Load GRM model (once) ---
    print(f"[AGENT] Loading GRM model: {args.model_path}")
    from examples.inference import GRMInference

    model = GRMInference(args.model_path)
    print("[AGENT] Model loaded.")

    # --- Output dirs ---
    ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run_root = Path(args.out_root).expanduser().resolve() / f"{ts}_agent_session"
    ensure_dir(run_root)
    ensure_dir(run_root / ".cache")
    ref_end_path = copy_goal_image(args.goal_image, run_root / ".cache")

    jsonl_path = run_root / "online_pred.jsonl"
    latest_path = run_root / "latest_progress.json"
    summary_path = run_root / "pred_vllm_agent.json"

    metadata = {
        "data_source": "local_img_agent",
        "img_dir": str(img_dir),
        "model_path": args.model_path,
        "goal_image": str(Path(args.goal_image).expanduser().resolve()),
        "fisheye_config": str(Path(args.fisheye_config).resolve()),
        "pinhole_output_size": [out_w, out_h],
        "undistorted_cameras": list(FISHEYE_KEYS),
        "active_modes": active_modes,
        "interval": args.interval,
        "subtask_file": args.subtask_file,
        "result_file": args.result_file,
        "subtask_check_interval": args.subtask_check_interval,
        "success_threshold": args.success_threshold,
        "success_stable_steps": args.success_stable_steps,
        "success_max_drift": args.success_max_drift,
        "fail_stable_steps": args.fail_stable_steps,
        "fail_min_progress": args.fail_min_progress,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(run_root / "metadata.json", metadata)

    # --- Monitor state ---
    monitor = MonitorState(
        success_threshold=args.success_threshold,
        success_stable_steps=args.success_stable_steps,
        success_max_drift=args.success_max_drift,
        fail_stable_steps=args.fail_stable_steps,
        fail_min_progress=args.fail_min_progress,
    )

    # --- State for the current subtask ---
    tracker = ProgressTracker()
    step = 0
    previous: Optional[Dict[str, str]] = None
    ref_start: Optional[Dict[str, str]] = None
    current_subtask: Optional[str] = None

    def _cam_dirs_for_subtask(task_name: str) -> Dict[str, Path]:
        cache_root = run_root / ".cache" / sanitize_task(task_name)
        dirs = {}
        for key in CAMERA_KEYS:
            dirs[key] = cache_root / key
            ensure_dir(dirs[key])
        return dirs

    def init_subtask(task_name: str):
        nonlocal step, previous, ref_start, current_subtask
        current_subtask = task_name
        step = 0
        cam_dirs = _cam_dirs_for_subtask(task_name)

        print(f"\n{'=' * 60}")
        print(f"[AGENT] === New subtask: \"{task_name}\" ===")
        print(f"{'=' * 60}")

        saved = snapshot_current_with_undistort(
            img_dir, cam_dirs, step,
            map_x, map_y, f_interp, f_border_mode, f_border_value,
        )
        ref_start = saved
        previous = saved
        tracker.reset()
        monitor.reset()

        write_monitor_result(args.result_file, MONITOR_STATUS_RUNNING)

        record = {
            "subtask": task_name,
            "step": 0,
            "wall_time": time.time(),
            "latency_s": 0.0,
            "progress": 0.0,
            "progress_percent": 0.0,
            "monitor_status": MONITOR_STATUS_RUNNING,
            "modes": {
                mode: {"pred": "<score>0%</score>", "score": 0.0, "hop": 0.0, "progress": 0.0}
                for mode in VALID_MODES
            },
            "frames": saved,
            "event": "subtask_start",
        }
        append_jsonl(jsonl_path, record)
        write_json(latest_path, record)
        print(f"[AGENT] Subtask step 0 — reference start captured, status=running.")

    def do_inference_step():
        """Run one inference step for the current subtask."""
        nonlocal step, previous

        if current_subtask is None:
            return

        cam_dirs = _cam_dirs_for_subtask(current_subtask)
        current = snapshot_current_with_undistort(
            img_dir, cam_dirs, step,
            map_x, map_y, f_interp, f_border_mode, f_border_value,
        )

        infer_start = time.time()

        samples = build_online_samples(
            task=current_subtask,
            step=step,
            ref_start=ref_start,
            ref_end_path=ref_end_path,
            previous=previous,
            current=current,
            modes=active_modes,
        )

        outputs = model.inference_batch(samples)
        mode_results = {}
        for item in outputs:
            mode = item["eval_mode"]
            mode_results[mode] = make_mode_result(
                mode, item.get("pred", ""), tracker
            )

        fused = clamp(
            sum(float(mode_results[m]["progress"]) for m in active_modes)
            / len(active_modes),
            0.0,
            1.0,
        )

        # Update monitor
        status = monitor.update(fused)
        write_monitor_result(args.result_file, status)

        latency = time.time() - infer_start
        record = {
            "subtask": current_subtask,
            "step": step,
            "wall_time": time.time(),
            "latency_s": latency,
            "progress": fused,
            "progress_percent": fused * 100.0,
            "monitor_status": status,
            "modes": mode_results,
            "frames": current,
        }
        append_jsonl(jsonl_path, record)
        write_json(latest_path, record)

        parts = [
            "[AGENT] \"{task}\" step={step:06d} "
            "fused={fused:6.2f}%".format(
                task=current_subtask,
                step=step,
                fused=fused * 100.0,
            )
        ]
        if "forward" in mode_results:
            parts.append("fwd={:.2f}%".format(float(mode_results["forward"]["progress"]) * 100.0))
        if "incremental" in mode_results:
            parts.append("inc={:.2f}%".format(float(mode_results["incremental"]["progress"]) * 100.0))
        if "backward" in mode_results:
            parts.append("bwd={:.2f}%".format(float(mode_results["backward"]["progress"]) * 100.0))
        parts.append("lat={:.2f}s".format(latency))
        parts.append("[{}]".format(status))
        print(" ".join(parts))

        previous = current
        step += 1

    # -----------------------------------------------------------------
    # Main loop: wait for subtask, run inference, detect success/fail
    # -----------------------------------------------------------------
    try:
        # Wait for the first subtask to appear
        print(f"[AGENT] Waiting for subtask in {args.subtask_file} ...")
        while True:
            subtask_text = read_subtask_file(args.subtask_file)
            if subtask_text is not None:
                break
            print(f"[AGENT] Subtask file is empty or missing ({args.subtask_file}). "
                  f"Waiting...")
            time.sleep(args.subtask_check_interval)

        init_subtask(subtask_text)
        step = 1

        while True:
            # --- Check for subtask change ---
            subtask_text = read_subtask_file(args.subtask_file)
            if subtask_text is not None and subtask_text != current_subtask:
                print(f"\n[AGENT] Subtask changed: \"{current_subtask}\" -> \"{subtask_text}\"")
                init_subtask(subtask_text)
                step = 1

            # --- If subtask file is empty, warn and skip ---
            if subtask_text is None:
                print(f"[AGENT] Subtask file is empty ({args.subtask_file}). "
                      f"Waiting for new subtask...")
                time.sleep(args.subtask_check_interval)
                continue

            # --- If monitor has already finished, just wait for new subtask ---
            if monitor.is_finished:
                time.sleep(args.subtask_check_interval)
                continue

            # --- Run inference ---
            do_inference_step()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[AGENT] Interrupted by user.")
    finally:
        finalize_jsonl(jsonl_path, summary_path)
        print(f"[AGENT] Output directory: {run_root}")


def main() -> None:
    args = parse_args()
    run_online(args)


if __name__ == "__main__":
    main()
