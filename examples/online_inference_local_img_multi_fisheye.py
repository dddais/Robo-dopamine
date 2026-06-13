#!/usr/bin/env python3
"""Multi-subtask online GRM inference with fisheye undistortion (vLLM).

Based on ``online_inference_local_img_multi.py`` with added fisheye
undistortion for wrist camera images.  The left/right wrist images
are remapped from fisheye to pinhole using the SEUCM model before
being saved and sent to the GRM model.

The fisheye remap table is built **once** at startup from the config
at ``--fisheye-config`` (default: ``/home/ubuntu/dais/fisheye_process/config.yaml``).

Expected files under ``--img-dir`` (default ``/tmp/img/``)::

    /tmp/img/
        base_0_rgb.jpg              (cam_high, no undistortion)
        left_wrist_0_rgb.jpg        (cam_left_wrist, fisheye → pinhole)
        right_wrist_0_rgb.jpg       (cam_right_wrist, fisheye → pinhole)

Usage:
    python online_inference_local_img_multi_fisheye.py

    python online_inference_local_img_multi_fisheye.py \\
        --fisheye-config /home/ubuntu/dais/fisheye_process/config.yaml \\
        --interval 0.5
"""

import argparse
import json
import os
import re
import select
import shutil
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
DEFAULT_OUT_ROOT = "./results/online_local_img_multi_fisheye"
DEFAULT_IMG_DIR = "/tmp/img"
DEFAULT_INTERVAL = 1.0
DEFAULT_FISHEYE_CONFIG = "/home/ubuntu/dais/fisheye_process/config.yaml"

IMG_FILES = {
    "cam_high": "base_0_rgb.jpg",
    "cam_left_wrist": "left_wrist_0_rgb.jpg",
    "cam_right_wrist": "right_wrist_0_rgb.jpg",
}
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
# Camera keys that need fisheye undistortion
FISHEYE_KEYS = ("cam_left_wrist", "cam_right_wrist")
VALID_MODES = ("forward", "incremental", "backward")

SUBTASK_LIST = [
    "pick one bowl and put it into the basket",
    "pick one bowl and put it into the basket",
    "pick the blue plate and put it into the basket",
    "pick the pink plate and put it into the basket",
    "pick the green plate and put it into the basket",
]


# ---------------------------------------------------------------------------
# Fisheye undistortion (reuses fisheye_process/convert.py)
# ---------------------------------------------------------------------------

def init_fisheye_remap(config_path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build the fisheye→pinhole remap table once.

    Returns (map_x, map_y, out_width, out_height).
    """
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

    # Force depth off for real-time remap (rotation-only mode)
    cfg.setdefault("depth", {})
    cfg["depth"]["enabled"] = False

    map_x, map_y = build_remap_table(cfg)
    out_w = cfg["pinhole"]["image_width"]
    out_h = cfg["pinhole"]["image_height"]

    proc = cfg.get("processing", {})
    interp = get_interp_flag(proc.get("interpolation", "LINEAR"))
    border_mode = get_border_flag(proc.get("border_mode", "CONSTANT"))
    border_value = proc.get("border_value", 0)

    print(f"[FISHEYE] Remap table built: {out_w}x{out_h}, "
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
# Non-blocking stdin
# ---------------------------------------------------------------------------

def _kbhit(timeout_s: float = 0.0) -> bool:
    dr, _, _ = select.select([sys.stdin], [], [], timeout_s)
    return bool(dr)


def _read_stdin_line(timeout_s: float = 0.0) -> str:
    if not _kbhit(timeout_s):
        return ""
    return sys.stdin.readline().strip()


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
    print(f"[FISHEYE-MULTI] Wrote summary JSON: {out_path}")


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

        # Apply fisheye undistortion to wrist cameras
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
                "id": f"fisheye-multi-{mode}-step_{step:06d}-{before_id}-af_{step:06d}",
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
        description="Multi-subtask online GRM inference with fisheye undistortion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        "--subtasks",
        nargs="+",
        default=None,
        help="Override the built-in subtask list.",
    )
    parser.add_argument(
        "--no-backward",
        action="store_true",
        help="Exclude backward mode from inference and fused progress calculation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_online(args: argparse.Namespace) -> None:
    img_dir = Path(args.img_dir).resolve()
    subtask_list = args.subtasks if args.subtasks else SUBTASK_LIST
    active_modes = [m for m in VALID_MODES if not (args.no_backward and m == "backward")]
    if args.no_backward:
        print(f"[FISHEYE-MULTI] Backward mode disabled. Active modes: {active_modes}")

    # Verify source images
    for key, fname in IMG_FILES.items():
        p = img_dir / fname
        if not p.exists():
            print(f"[FISHEYE-MULTI] Missing: {p}")
            print(f"[FISHEYE-MULTI] Please ensure {fname} exists in {img_dir}")
            raise SystemExit(1)

    # --- Build fisheye remap table (once) ---
    print(f"[FISHEYE-MULTI] Loading fisheye config: {args.fisheye_config}")
    map_x, map_y, out_w, out_h, f_interp, f_border_mode, f_border_value = \
        init_fisheye_remap(args.fisheye_config)

    # --- Load GRM model (once) ---
    print(f"[FISHEYE-MULTI] Loading GRM model: {args.model_path}")
    from examples.inference import GRMInference

    model = GRMInference(args.model_path)
    print("[FISHEYE-MULTI] Model loaded.")

    # --- Output dirs ---
    ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run_root = Path(args.out_root).expanduser().resolve() / f"{ts}_fisheye_session"
    ensure_dir(run_root)
    ensure_dir(run_root / ".cache")
    ref_end_path = copy_goal_image(args.goal_image, run_root / ".cache")

    jsonl_path = run_root / "online_pred.jsonl"
    latest_path = run_root / "latest_progress.json"
    summary_path = run_root / "pred_vllm_fisheye_multi.json"

    metadata = {
        "data_source": "local_img_multi_fisheye",
        "img_dir": str(img_dir),
        "model_path": args.model_path,
        "goal_image": str(Path(args.goal_image).expanduser().resolve()),
        "fisheye_config": str(Path(args.fisheye_config).resolve()),
        "pinhole_output_size": [out_w, out_h],
        "undistorted_cameras": list(FISHEYE_KEYS),
        "subtask_list": subtask_list,
        "active_modes": active_modes,
        "interval": args.interval,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(run_root / "metadata.json", metadata)

    # --- Set stdin to non-blocking raw mode ---
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    subtask_idx = 0
    tracker = ProgressTracker()
    step = 0
    previous = None
    ref_start = None

    def _cam_dirs_for_subtask(idx: int) -> Dict[str, Path]:
        task = subtask_list[idx]
        cache_root = run_root / ".cache" / f"subtask_{idx:03d}_{sanitize_task(task)}"
        dirs = {}
        for key in CAMERA_KEYS:
            dirs[key] = cache_root / key
            ensure_dir(dirs[key])
        return dirs

    def init_subtask(idx: int):
        nonlocal step, previous, ref_start
        task = subtask_list[idx]
        step = 0
        cam_dirs = _cam_dirs_for_subtask(idx)

        print(f"\n{'=' * 60}")
        print(f"[FISHEYE-MULTI] === Subtask {idx + 1}/{len(subtask_list)}: \"{task}\" ===")
        print(f"[FISHEYE-MULTI] Press 0+Enter = next, N+Enter = switch to #N, Ctrl+C = exit.")
        print(f"{'=' * 60}")

        saved = snapshot_current_with_undistort(
            img_dir, cam_dirs, step,
            map_x, map_y, f_interp, f_border_mode, f_border_value,
        )
        ref_start = saved
        previous = saved
        tracker.reset()

        record = {
            "subtask_idx": idx,
            "subtask": task,
            "step": 0,
            "wall_time": time.time(),
            "latency_s": 0.0,
            "progress": 0.0,
            "progress_percent": 0.0,
            "modes": {
                mode: {"pred": "<score>0%</score>", "score": 0.0, "hop": 0.0, "progress": 0.0}
                for mode in VALID_MODES
            },
            "frames": saved,
            "event": "subtask_start",
        }
        append_jsonl(jsonl_path, record)
        write_json(latest_path, record)
        print(f"[FISHEYE-MULTI] Subtask {idx + 1} step 0 — reference start captured.")

    try:
        init_subtask(subtask_idx)
        step = 1

        while True:
            # --- Check for subtask switch ---
            line = _read_stdin_line(timeout_s=0.0)
            if line:
                try:
                    num = int(line)
                except ValueError:
                    continue

                if num == 0:
                    if subtask_idx + 1 < len(subtask_list):
                        subtask_idx += 1
                        init_subtask(subtask_idx)
                        step = 1
                        continue
                    else:
                        print("[FISHEYE-MULTI] Already at the last subtask. Press Ctrl+C to exit.")
                        continue
                elif 1 <= num <= len(subtask_list):
                    subtask_idx = num - 1
                    init_subtask(subtask_idx)
                    step = 1
                    continue
                else:
                    print(f"[FISHEYE-MULTI] Invalid #{num}. Valid: 0 (next) or 1-{len(subtask_list)}.")
                    continue

            # --- Inference ---
            cam_dirs = _cam_dirs_for_subtask(subtask_idx)
            current = snapshot_current_with_undistort(
                img_dir, cam_dirs, step,
                map_x, map_y, f_interp, f_border_mode, f_border_value,
            )

            infer_start = time.time()
            task = subtask_list[subtask_idx]

            samples = build_online_samples(
                task=task,
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

            latency = time.time() - infer_start
            record = {
                "subtask_idx": subtask_idx,
                "subtask": task,
                "step": step,
                "wall_time": time.time(),
                "latency_s": latency,
                "progress": fused,
                "progress_percent": fused * 100.0,
                "modes": mode_results,
                "frames": current,
            }
            append_jsonl(jsonl_path, record)
            write_json(latest_path, record)

            parts = [
                "[FISHEYE-MULTI] [{idx}/{total}] \"{task}\" step={step:06d} "
                "fused={fused:6.2f}%".format(
                    idx=subtask_idx + 1,
                    total=len(subtask_list),
                    task=task,
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
            print(" ".join(parts))

            previous = current
            step += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[FISHEYE-MULTI] Interrupted by user.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        finalize_jsonl(jsonl_path, summary_path)
        print(f"[FISHEYE-MULTI] Output directory: {run_root}")


def main() -> None:
    args = parse_args()
    run_online(args)


if __name__ == "__main__":
    main()
