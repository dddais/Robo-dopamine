#!/usr/bin/env python3
"""
Online ROS1 inference for Robo-Dopamine GRM.

This script keeps the existing offline inference code untouched and reuses
GRMInference.inference_batch() for live, synchronized camera frames.
"""

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = "/home/dais/workspace/Robo-Dopamine/train/checkpoints/my_carrot_finetune_big"
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
VALID_MODES = ("forward", "incremental", "backward")


@dataclass
class SyncedFrames:
    stamp: float
    frames: Dict[str, object]


@dataclass
class RosHandles:
    rospy: object
    subscribers: List[object]
    synchronizer: object


class LatestFrameBuffer:
    """Thread-safe handoff from ROS callbacks to the inference loop."""

    def __init__(self):
        self._cond = threading.Condition()
        self._latest: Optional[SyncedFrames] = None
        self._seq = 0

    def update(self, frames: SyncedFrames) -> None:
        with self._cond:
            self._latest = frames
            self._seq += 1
            self._cond.notify_all()

    def wait_for_next(self, last_seq: int, timeout: Optional[float]) -> tuple:
        with self._cond:
            ok = self._cond.wait_for(lambda: self._seq > last_seq, timeout=timeout)
            if not ok or self._latest is None:
                raise TimeoutError("Timed out waiting for synchronized ROS image frames.")
            return self._seq, self._latest


class ProgressTracker:
    """Applies the same per-mode progress formulas as examples/inference.py."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online Robo-Dopamine GRM progress inference from ROS1 image topics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="GRM checkpoint or HF model path.")
    parser.add_argument("--front-topic", required=True, help="ROS1 sensor_msgs/Image topic for the front camera.")
    parser.add_argument("--left-topic", required=True, help="ROS1 sensor_msgs/Image topic for the left wrist camera.")
    parser.add_argument("--right-topic", required=True, help="ROS1 sensor_msgs/Image topic for the right wrist camera.")
    parser.add_argument("--task", required=True, help="Task instruction used in the GRM prompt.")
    parser.add_argument("--goal-image", required=True, help="Real goal/reference image path for ref_end and backward mode.")
    parser.add_argument("--out-root", default="./results/online", help="Root directory for online inference outputs.")
    parser.add_argument("--sample-period", type=float, default=1.0, help="Minimum seconds between inference samples.")
    parser.add_argument("--sync-slop", type=float, default=0.05, help="Approximate time sync tolerance in seconds.")
    parser.add_argument("--sync-queue-size", type=int, default=10, help="Approximate time sync queue size.")
    parser.add_argument("--wait-timeout", type=float, default=30.0, help="Seconds to wait for the first/new synchronized frame.")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum online inference steps; 0 means run forever.")
    parser.add_argument("--ros-node-name", default="robo_dopamine_online_inference", help="ROS node name.")
    parser.add_argument("--image-encoding", default="bgr8", help="cv_bridge output encoding for raw Image topics.")
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sanitize_task(task: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", task).strip("_")
    return safe[:80] or "task"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    run_root = Path(out_root).expanduser().resolve() / f"{ts}_online_{sanitize_task(task)}"
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


def save_frame_triplet(frames: Dict[str, object], dirs: Dict[str, Path], step: int) -> Dict[str, str]:
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
    run_root: Path,
    step: int,
    ref_start: Dict[str, str],
    ref_end_path: Path,
    previous: Dict[str, str],
    current: Dict[str, str],
) -> List[Dict]:
    del run_root

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
                "id": f"online-{mode}-step_{step:06d}-{before_id}-af_{step:06d}",
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
    print(f"[ONLINE] Wrote summary JSON: {out_path}")


def setup_ros_subscribers(args: argparse.Namespace, buffer: LatestFrameBuffer) -> RosHandles:
    try:
        import message_filters
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
    except ImportError as exc:
        raise ImportError(
            "ROS1 dependencies are required. Source ROS first, for example: "
            "source /opt/ros/noetic/setup.bash"
        ) from exc

    rospy.init_node(args.ros_node_name, anonymous=True)
    bridge = CvBridge()

    def callback(front_msg, left_msg, right_msg):
        try:
            front = bridge.imgmsg_to_cv2(front_msg, desired_encoding=args.image_encoding)
            left = bridge.imgmsg_to_cv2(left_msg, desired_encoding=args.image_encoding)
            right = bridge.imgmsg_to_cv2(right_msg, desired_encoding=args.image_encoding)
            stamp = max(
                front_msg.header.stamp.to_sec(),
                left_msg.header.stamp.to_sec(),
                right_msg.header.stamp.to_sec(),
            )
            buffer.update(
                SyncedFrames(
                    stamp=stamp,
                    frames={
                        "cam_high": front,
                        "cam_left_wrist": left,
                        "cam_right_wrist": right,
                    },
                )
            )
        except Exception as exc:
            rospy.logwarn("Failed to convert synchronized image frames: %s", exc)

    subscribers = [
        message_filters.Subscriber(args.front_topic, Image),
        message_filters.Subscriber(args.left_topic, Image),
        message_filters.Subscriber(args.right_topic, Image),
    ]
    sync = message_filters.ApproximateTimeSynchronizer(
        subscribers,
        queue_size=args.sync_queue_size,
        slop=args.sync_slop,
        allow_headerless=False,
    )
    sync.registerCallback(callback)
    print("[ONLINE] ROS subscribers ready:")
    print(f"  front: {args.front_topic}")
    print(f"  left : {args.left_topic}")
    print(f"  right: {args.right_topic}")
    return RosHandles(rospy=rospy, subscribers=subscribers, synchronizer=sync)


def make_mode_result(mode: str, pred_text: str, tracker: ProgressTracker) -> Dict[str, object]:
    score = parse_score(pred_text)
    stats = tracker.update(mode, score)
    return {
        "pred": pred_text,
        "score": stats["score"],
        "hop": stats["hop"],
        "progress": stats["progress"],
    }


def run_online(args: argparse.Namespace) -> None:
    if args.sample_period <= 0:
        raise ValueError("--sample-period must be > 0")
    if args.sync_slop <= 0:
        raise ValueError("--sync-slop must be > 0")
    if args.sync_queue_size <= 0:
        raise ValueError("--sync-queue-size must be > 0")

    from examples.inference import GRMInference

    dirs = create_run_dirs(args.out_root, args.task)
    ref_end_path = copy_goal_image(args.goal_image, dirs["cache_root"])
    jsonl_path = dirs["run_root"] / "online_pred.jsonl"
    latest_path = dirs["run_root"] / "latest_progress.json"
    summary_path = dirs["run_root"] / "pred_vllm_online.json"

    metadata = {
        "model_path": args.model_path,
        "task": args.task,
        "goal_image": str(Path(args.goal_image).expanduser().resolve()),
        "ref_end": str(ref_end_path),
        "topics": {
            "front": args.front_topic,
            "left": args.left_topic,
            "right": args.right_topic,
        },
        "sample_period": args.sample_period,
        "sync_slop": args.sync_slop,
        "sync_queue_size": args.sync_queue_size,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(dirs["run_root"] / "metadata.json", metadata)

    buffer = LatestFrameBuffer()
    ros_handles = setup_ros_subscribers(args, buffer)
    rospy = ros_handles.rospy

    print(f"[ONLINE] Loading GRM model: {args.model_path}")
    model = GRMInference(args.model_path)
    tracker = ProgressTracker()

    last_seq = 0
    print("[ONLINE] Waiting for first synchronized frame triplet...")
    last_seq, first_synced = buffer.wait_for_next(last_seq, timeout=args.wait_timeout)
    ref_start = save_frame_triplet(first_synced.frames, dirs, step=0)
    previous = ref_start

    initial = {
        "step": 0,
        "wall_time": time.time(),
        "ros_stamp": first_synced.stamp,
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
    print(f"[ONLINE] Step 000000 progress=0.00% saved as reference start.")

    step = 1
    try:
        while not rospy.is_shutdown():
            if args.max_steps > 0 and step > args.max_steps:
                break

            time.sleep(args.sample_period)
            infer_start = time.time()
            last_seq, synced = buffer.wait_for_next(last_seq, timeout=args.wait_timeout)
            current = save_frame_triplet(synced.frames, dirs, step=step)
            samples = build_online_samples(
                task=args.task,
                run_root=dirs["run_root"],
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
                mode_results[mode] = make_mode_result(mode, item.get("pred", ""), tracker)

            fused = clamp(
                sum(float(mode_results[mode]["progress"]) for mode in VALID_MODES) / len(VALID_MODES),
                0.0,
                1.0,
            )
            record = {
                "step": step,
                "wall_time": time.time(),
                "ros_stamp": synced.stamp,
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
                "[ONLINE] Step {step:06d} fused={fused:6.2f}% "
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
        print("\n[ONLINE] Interrupted by user.")
    finally:
        finalize_jsonl(jsonl_path, summary_path)
        print(f"[ONLINE] Output directory: {dirs['run_root']}")


def main() -> None:
    args = parse_args()
    run_online(args)


if __name__ == "__main__":
    main()
