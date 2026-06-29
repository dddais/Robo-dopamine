#!/usr/bin/env python3
"""Run Robo-Dopamine GRM on the non-Markovian radio sub23 case."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.inference import GRMInference  # noqa: E402


DEFAULT_TASK = (
    "pick up the red radio, press the switch with the left hand until the "
    "indicator light turns green, then put the radio down"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview",
    )
    parser.add_argument("--data-dir", default="./aligned_data/xzx_episode_1_sub23")
    parser.add_argument("--out-root", default="./results/xzx_episode_1_sub23_memory_grm")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--goal-image", default="./examples/blank_goal.png")
    parser.add_argument("--frame-interval", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-visualize", action="store_true")
    return parser.parse_args()


def load_mode_stats(pred_path: Path) -> dict:
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    scores = []
    progress = []
    hops = []
    for item in data:
        pred = item.get("pred", "")
        match = re.search(r"([+-]?\d+(?:\.\d+)?)%", pred)
        if match:
            scores.append(float(match.group(1)))
        progress.append(float(item.get("progress", 0.0)) * 100.0)
        hops.append(float(item.get("hop", 0.0)) * 100.0)

    return {
        "pred_path": str(pred_path),
        "samples": len(data),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": mean(scores) if scores else None,
        "progress_final": progress[-1] if progress else None,
        "progress_mean": mean(progress) if progress else None,
        "progress_max": max(progress) if progress else None,
        "progress_min": min(progress) if progress else None,
        "hop_min": min(hops) if hops else None,
        "hop_max": max(hops) if hops else None,
        "progress_series": progress,
    }


def write_progress_plot(summary: dict, out_root: Path, frame_interval: int) -> None:
    modes = [m for m in ["forward", "incremental", "backward"] if m in summary["modes"]]
    if not modes:
        return

    min_len = min(len(summary["modes"][m]["progress_series"]) for m in modes)
    if min_len <= 0:
        return

    all_progress = {
        mode: summary["modes"][mode]["progress_series"][:min_len]
        for mode in modes
    }
    all_progress["avg"] = [
        mean(all_progress[mode][i] for mode in modes)
        for i in range(min_len)
    ]

    fps = 30.0
    time_sec = [(i + 1) * frame_interval / fps for i in range(min_len)]
    colors = {
        "forward": "#2196F3",
        "incremental": "#FF9800",
        "backward": "#4CAF50",
        "avg": "#E91E63",
    }
    labels = {
        "forward": "Forward",
        "incremental": "Incremental",
        "backward": "Backward",
        "avg": "Average (Fused)",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for key in ["forward", "incremental", "backward", "avg"]:
        if key not in all_progress:
            continue
        ax.plot(
            time_sec,
            all_progress[key],
            "--" if key == "avg" else "-",
            color=colors[key],
            linewidth=3 if key == "avg" else 2,
            label=f"{labels[key]} final={all_progress[key][-1]:.1f}%",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Task Progress (%)")
    ax.set_title("GRM Progress on Non-Markovian Radio Sub23")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(y=100, color="gray", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_root / "progress_curve.png", dpi=150)
    plt.close(fig)

    summary["fused"] = {
        "samples": min_len,
        "progress_final": all_progress["avg"][-1],
        "progress_mean": mean(all_progress["avg"]),
        "progress_max": max(all_progress["avg"]),
        "progress_min": min(all_progress["avg"]),
    }


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model = GRMInference(args.model_path)
    summary = {
        "episode_id": "xzx_radio_sub23",
        "task": args.task,
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "goal_image": args.goal_image,
        "frame_interval": args.frame_interval,
        "batch_size": args.batch_size,
        "modes": {},
        "outputs": {},
    }

    for mode in ["forward", "incremental", "backward"]:
        output_dir = model.run_pipeline(
            cam_high_path=os.path.join(args.data_dir, "cam_high.mp4"),
            cam_left_path=os.path.join(args.data_dir, "cam_left_wrist.mp4"),
            cam_right_path=os.path.join(args.data_dir, "cam_right_wrist.mp4"),
            out_root=str(out_root),
            task=args.task,
            frame_interval=args.frame_interval,
            batch_size=args.batch_size,
            goal_image=args.goal_image,
            eval_mode=mode,
            visualize=not args.no_visualize,
        )
        pred_path = Path(output_dir) / "pred_vllm.json"
        summary["outputs"][mode] = str(Path(output_dir))
        summary["modes"][mode] = load_mode_stats(pred_path)
        print(
            f"{mode}: final={summary['modes'][mode]['progress_final']:.2f}% "
            f"mean={summary['modes'][mode]['progress_mean']:.2f}% "
            f"samples={summary['modes'][mode]['samples']}"
        )

    write_progress_plot(summary, out_root, args.frame_interval)
    (out_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary written to {out_root / 'run_summary.json'}")
    print(f"Progress curve written to {out_root / 'progress_curve.png'}")


if __name__ == "__main__":
    main()

