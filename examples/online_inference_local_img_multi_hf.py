#!/usr/bin/env python3
"""Multi-subtask online GRM inference from local image directory (HuggingFace Transformers).

Same multi-subtask logic as ``online_inference_local_img_multi.py`` but uses
HuggingFace Transformers for model loading and inference instead of vLLM.
This avoids the extra VRAM overhead from vLLM's torch compilation, making it
possible to run 8B models on 24GB GPUs.

The model is loaded **once**; inference starts from the first subtask and
advances when the user presses ``0`` + Enter (next) or ``N`` + Enter
(switch to subtask #N, 1-based).  Switching subtasks resets the reference
start (ref_start) and the progress tracker.

Expected files under ``--img-dir`` (default ``/tmp/img/``)::

    /tmp/img/
        extra_view_images_0.jpg   (cam_high)
        main_images.jpg            (cam_left_wrist)
        extra_view_images_1.jpg   (cam_right_wrist)

Usage:
    python online_inference_local_img_multi_hf.py

    python online_inference_local_img_multi_hf.py \\
        --interval 0.5 \\
        --model-path /path/to/8B/model
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
    "Robo-Dopamine-GRM-2.0-8B-Preview"
)
DEFAULT_GOAL_IMAGE = str(REPO_ROOT / "examples" / "blank_goal.png")
DEFAULT_OUT_ROOT = "./results/online_local_img_multi_hf"
DEFAULT_IMG_DIR = "/tmp/img"
DEFAULT_INTERVAL = 1.0

IMG_FILES = {
    "cam_high": "base_0_rgb.jpg",
    "cam_left_wrist": "left_wrist_0_rgb.jpg",
    "cam_right_wrist": "right_wrist_0_rgb.jpg",
}
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
VALID_MODES = ("forward", "incremental", "backward")

# Subtask list — edit this to match your experiment
SUBTASK_LIST = [
    "pick the green bowl and put it into the basket",
    "pick the blue bowl and put it into the basket",
    "pick the blue plate and put it into the basket",
    "pick the pink plate and put it into the basket",
    "pick the green plate and put it into the basket",
]


# ---------------------------------------------------------------------------
# Non-blocking stdin check (Unix only)
# ---------------------------------------------------------------------------

def _kbhit(timeout_s: float = 0.0) -> bool:
    """Return True if a keypress is available on stdin without blocking."""
    dr, _, _ = select.select([sys.stdin], [], [], timeout_s)
    return bool(dr)


def _read_stdin_line(timeout_s: float = 0.0) -> str:
    """Read a non-blocking line from stdin; return empty string if none."""
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
    print(f"[MULTI] Wrote summary JSON: {out_path}")


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
    run_root = Path(out_root).expanduser().resolve() / f"{ts}_multi_{sanitize_task(task)}"
    cache_root = run_root / ".cache"
    dirs = {"run_root": run_root, "cache_root": cache_root}
    for key in CAMERA_KEYS:
        dirs[key] = cache_root / key
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def snapshot_current(img_dir: Path, dirs: Dict[str, Path], step: int) -> Dict[str, str]:
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
                "id": f"multi-{mode}-step_{step:06d}-{before_id}-af_{step:06d}",
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
        description="Multi-subtask online GRM inference from local image directory.",
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
        "--subtasks",
        nargs="+",
        default=None,
        help="Override the built-in subtask list. "
             "If omitted, SUBTASK_LIST from the script is used.",
    )
    parser.add_argument(
        "--no-backward",
        default=True,
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
        print(f"[MULTI] Backward mode disabled. Active modes: {active_modes}")

    # Verify source images
    for key, fname in IMG_FILES.items():
        p = img_dir / fname
        if not p.exists():
            print(f"[MULTI] Missing: {p}")
            print(f"[MULTI] Please ensure {fname} exists in {img_dir}")
            raise SystemExit(1)

    # --- Load GRM model via HuggingFace Transformers (no vLLM) ---
    print(f"[MULTI] Loading GRM model: {args.model_path}")
    import torch
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
    )
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 76800
        processor.image_processor.min_pixels = 12544
    hf_model.eval()
    print("[MULTI] Model loaded.")

    SYSTEM_PROMPT = (
        "You are a rigorous, impartial vision evaluator for robot task progress. "
        "Your job is to judge whether the AFTER image set moves closer to the task "
        "objective than the BEFORE image set, using the provided reference examples "
        "only as anchors.\n\n"
        "<Task>\n`{task}`\n\n"
        "REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's "
        "actual START/END):\n"
        "- REFERENCE START — Robot Front Image (task just starting): <image>\n"
        "- REFERENCE END — Robot Front Image (task fully completed): <image>\n"
        "</Task>\n\n"
        "BEFORE Robot Front Image: <image>\n"
        "BEFORE Robot Left Wrist Image: <image>\n"
        "BEFORE Robot Right Wrist Image: <image>\n\n"
        "AFTER Robot Front Image: <image>\n"
        "AFTER Robot Left Wrist Image: <image>\n"
        "AFTER Robot Right Wrist Image: <image>\n\n"
        "Goal\n"
        "Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves "
        "closer to accomplishing the task than BEFORE, using the REFERENCE START/END "
        "images as conceptual anchors.\n\n"
        "Progress Estimation (no formulas)\n"
        "1) Calibrate using the references:\n"
        "   - REFERENCE START = \"just beginning\"; REFERENCE END = \"fully completed.\"\n"
        "   - Visually estimate how far BEFORE and AFTER are along this START->END "
        "continuum.\n"
        "2) Direction:\n"
        "   - AFTER better than BEFORE -> positive score.\n"
        "   - AFTER worse than BEFORE -> negative score.\n"
        "   - Essentially the same -> 0.\n"
        "3) Normalize to an integer percentage in [-100%, +100%]:\n"
        "   - For improvements, scale the improvement relative to what remained from "
        "BEFORE to END.\n"
        "   - For regressions, scale the deterioration relative to how far BEFORE had "
        "progressed from START.\n"
        "   - Clip to [-100%, +100%] and round to the nearest integer percent.\n\n"
        "Output Format (STRICT)\n"
        "Return ONLY one line containing the score wrapped in <score> tags, as an "
        "integer percentage with a percent sign:\n"
        "<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>\n"
    )

    def inference_batch(batch_data):
        results = []
        for item in batch_data:
            images = [Image.open(p).convert("RGB") for p in item["image"]]
            prompt_text = SYSTEM_PROMPT.format(task=item["task"])
            parts = prompt_text.split("<image>")
            content = []
            for i, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if i < len(parts) - 1:
                    content.append({"type": "image"})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                output_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                )

            # Decode only the generated part (skip input tokens)
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            pred = processor.tokenizer.decode(generated, skip_special_tokens=True)
            res_item = item.copy()
            res_item["pred"] = pred
            results.append(res_item)
        return results

    # --- Output dirs (shared across subtasks) ---
    ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run_root = Path(args.out_root).expanduser().resolve() / f"{ts}_multi_session"
    ensure_dir(run_root)
    ensure_dir(run_root / ".cache")
    ref_end_path = copy_goal_image(args.goal_image, run_root / ".cache")

    jsonl_path = run_root / "online_pred.jsonl"
    latest_path = run_root / "latest_progress.json"
    summary_path = run_root / "pred_vllm_multi.json"

    metadata = {
        "data_source": "local_img_multi",
        "img_dir": str(img_dir),
        "model_path": args.model_path,
        "goal_image": str(Path(args.goal_image).expanduser().resolve()),
        "subtask_list": subtask_list,
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

    def init_subtask(idx: int):
        nonlocal step, previous, ref_start
        task = subtask_list[idx]
        step = 0

        # Each subtask gets its own cache subdirectory
        cache_root = run_root / ".cache" / f"subtask_{idx:03d}_{sanitize_task(task)}"
        cam_dirs = {}
        for key in CAMERA_KEYS:
            cam_dirs[key] = cache_root / key
            ensure_dir(cam_dirs[key])

        print(f"\n{'=' * 60}")
        print(f"[MULTI] === Subtask {idx + 1}/{len(subtask_list)}: \"{task}\" ===")
        print(f"[MULTI] Press 0+Enter = next subtask, N+Enter = switch to subtask #N (1-based), Ctrl+C = exit.")
        print(f"{'=' * 60}")

        # Snapshot current frames as new ref_start
        saved = {}
        for key in CAMERA_KEYS:
            src = img_dir / IMG_FILES[key]
            out_path = cam_dirs[key] / f"frame_{step:06d}.png"
            shutil.copyfile(str(src), str(out_path))
            saved[key] = str(out_path)

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
        print(f"[MULTI] Subtask {idx + 1} step 0 — reference start captured.")

    try:
        # Initialize first subtask
        init_subtask(subtask_idx)
        step = 1

        while True:
            # --- Check for subtask switch input ---
            line = _read_stdin_line(timeout_s=0.0)
            if line:
                try:
                    num = int(line)
                except ValueError:
                    continue

                if num == 0:
                    # Switch to next subtask
                    if subtask_idx + 1 < len(subtask_list):
                        subtask_idx += 1
                        init_subtask(subtask_idx)
                        step = 1
                        continue
                    else:
                        print("[MULTI] Already at the last subtask. Press Ctrl+C to exit.")
                        continue
                elif 1 <= num <= len(subtask_list):
                    # Switch to specific subtask (1-based)
                    subtask_idx = num - 1
                    init_subtask(subtask_idx)
                    step = 1
                    continue
                else:
                    print(f"[MULTI] Invalid subtask #{num}. Valid range: 0 (next) or 1-{len(subtask_list)}.")
                    continue

            # --- Inference ---
            current = snapshot_current(img_dir, {
                k: (run_root / ".cache" / f"subtask_{subtask_idx:03d}_{sanitize_task(subtask_list[subtask_idx])}" / k)
                for k in CAMERA_KEYS
            }, step=step)

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

            outputs = inference_batch(samples)
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
                "[MULTI] [{idx}/{total}] \"{task}\" step={step:06d} "
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
        print("\n[MULTI] Interrupted by user.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        finalize_jsonl(jsonl_path, summary_path)
        print(f"[MULTI] Output directory: {run_root}")


def main() -> None:
    args = parse_args()
    run_online(args)


if __name__ == "__main__":
    main()
