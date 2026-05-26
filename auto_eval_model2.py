"""
续跑脚本：只跑第二个模型 GRM-2.0-8B-Preview
独立进程运行，避免 VLLM 在同一进程中重复加载模型导致死锁。
"""
import os
import sys
import json
import re
import itertools
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from examples.inference import GRMInference

# ============================================================
# 配置（只跑第二个模型）
# ============================================================

OUT_JSON = "_intermediate_results_model_GRM8B_sub.json"
MODEL_PATH = "./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview"
# MODEL_PATH = "/home/dais/workspace/Robo-Dopamine/train/checkpoints/multi_task_finetune"

DATA_DIRS = [
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_1",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_4",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_7",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_11",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1",

    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_2_cube",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_5_cube",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_9_cube",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_13_cube",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_4_cube",
    
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_3_bottle",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_6_bottle",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_8_bottle",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_12_bottle",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_3_bottle",

]

GOAL_IMAGES = [
    "./examples/blank_goal.png",
    # "./examples/exp_suc_1.png",
]

INTERVALS = [20, 10]

TASK_INSTRUCTIONS = [
    # "pick the carrot and put it on yellow plate ",
    # "pick the bottle and put it on yellow plate ",
    # "pick the white cube and put it on yellow plate ",
    "just pick the carrot",
    "just pick the bottle",
    "just pick the cube",
]

OUTPUT_ROOT = "/home/dais/workspace/Robo-Dopamine/results/auto_pick_carrot_fail"
EVAL_MODES = ["forward", "incremental", "backward"]


# ============================================================
# 辅助函数
# ============================================================

def get_model_tag(model_path: str) -> str:
    if "my_carrot_finetune_big" in model_path:
        return "finetune_carrot"
    elif "GRM-2.0-8B-Preview" in model_path:
        return "GRM-2.0-8B"
    elif "multi_task_finetune" in model_path:
        return "multi_task"
    return os.path.basename(model_path)


def get_data_tag(data_dir: str) -> str:
    return os.path.basename(data_dir)


def get_goal_tag(goal_image: str) -> str:
    name = os.path.basename(goal_image)
    if name == "blank_goal.png":
        return "blank"
    elif name == "exp_suc_1.png":
        return "suc_1"
    return os.path.splitext(name)[0]


def get_task_tag(task: str) -> str:
    if "carrot" in task:
        return "carrot"
    elif "bilue" in task or "bottle" in task:
        return "bottle"
    elif "white" in task or "cube" in task:
        return "cube"
    return "_".join(task.strip().split()[:3])


def run_single_combo(model, combo_idx, total_combos, model_tag, data_dir, data_tag,
                     goal_image, goal_tag, interval, task, task_tag):
    combo_name = f"[{combo_idx}/{total_combos}] model={model_tag} data={data_tag} goal={goal_tag} inter={interval} task={task_tag}"

    out_dir = os.path.join(
        OUTPUT_ROOT, model_tag, data_tag, goal_tag, f"inter{interval}", task_tag
    )
    os.makedirs(out_dir, exist_ok=True)

    results_paths = {}
    timing_info = {}
    for mode in EVAL_MODES:
        print(f"\n{'='*60}")
        print(f"{combo_name}  mode={mode}")
        print(f"{'='*60}")

        try:
            t0 = time.time()
            output_dir = model.run_pipeline(
                cam_high_path  = os.path.join(data_dir, "cam_high.mp4"),
                cam_left_path  = os.path.join(data_dir, "cam_left_wrist.mp4"),
                cam_right_path = os.path.join(data_dir, "cam_right_wrist.mp4"),
                out_root       = out_dir,
                task           = task,
                frame_interval = interval,
                batch_size     = 1,
                goal_image     = goal_image,
                eval_mode      = mode,
                visualize      = True,
            )
            elapsed = time.time() - t0
            results_paths[mode] = output_dir

            pred_file = os.path.join(output_dir, "pred_vllm.json")
            num_frames = 0
            if os.path.exists(pred_file):
                with open(pred_file) as f:
                    pred_data = json.load(f)
                    num_frames = len(pred_data)

            per_frame = elapsed / num_frames if num_frames > 0 else 0.0
            timing_info[mode] = {
                "total_s": round(elapsed, 2),
                "num_frames": num_frames,
                "per_frame_s": round(per_frame, 3),
            }
            print(f"[{mode}] Time: {elapsed:.2f}s total, {num_frames} frames, {per_frame:.3f}s/frame")

        except Exception as e:
            print(f"[{mode}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    return results_paths, timing_info


def analyze_and_plot(results_paths, out_dir, task, model_tag, interval):
    all_progress = {}

    for mode, path in results_paths.items():
        pred_file = os.path.join(path, "pred_vllm.json")
        if not os.path.exists(pred_file):
            continue
        with open(pred_file) as f:
            data = json.load(f)

        progress = [d['progress'] * 100 for d in data]
        all_progress[mode] = progress

    if not all_progress:
        return None

    modes = list(all_progress.keys())
    n = len(all_progress[modes[0]])
    avg_progress = []
    for i in range(n):
        avg_progress.append(sum(all_progress[m][i] for m in modes) / len(modes))
    all_progress["avg"] = avg_progress

    fps = 30
    frames = [i * interval for i in range(n)]
    time_sec = [f / fps for f in frames]

    colors = {"forward": "#2196F3", "incremental": "#FF9800", "backward": "#4CAF50", "avg": "#E91E63"}
    labels = {"forward": "Forward", "incremental": "Incremental", "backward": "Backward", "avg": "Average (Fused)"}
    styles = {"forward": "-", "incremental": "-", "backward": "-", "avg": "--"}
    widths = {"forward": 2.0, "incremental": 2.0, "backward": 2.0, "avg": 3.0}

    fig, ax = plt.subplots(figsize=(12, 6))
    for key in ["forward", "incremental", "backward", "avg"]:
        if key in all_progress:
            ax.plot(
                time_sec, all_progress[key],
                styles[key], color=colors[key], linewidth=widths[key],
                label=f"{labels[key]}  (final={all_progress[key][-1]:.1f}%)",
                marker="o" if key == "avg" else None, markersize=3,
            )

    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Task Progress (%)", fontsize=13)
    ax.set_title(
        f"GRM Progress Curve — \"{task.strip()}\"\n"
        f"Model: {model_tag}  |  Interval: {interval}",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="best")
    ax.set_xlim(0, time_sec[-1])
    ax.set_ylim(min(-5, ax.get_ylim()[0]), max(105, ax.get_ylim()[1]))
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(y=100, color="gray", linewidth=0.5, linestyle=":")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(out_dir, "progress_curve.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Progress curve saved: {save_path}")

    summary = {}
    for key in ["forward", "incremental", "backward", "avg"]:
        if key in all_progress:
            summary[key] = all_progress[key][-1]
    return summary


# ============================================================
# 主函数
# ============================================================

def main():
    model_tag = get_model_tag(MODEL_PATH)
    os.makedirs(os.path.join(OUTPUT_ROOT, model_tag), exist_ok=True)

    # 生成所有组合
    combos = list(itertools.product(DATA_DIRS, GOAL_IMAGES, INTERVALS, TASK_INSTRUCTIONS))
    total = len(combos)
    print(f"Model: {model_tag}")
    print(f"Total combinations: {total}")

    # 检查已有的中间结果，跳过已完成的组合
    result_file = os.path.join(OUTPUT_ROOT, OUT_JSON)
    existing = []
    if os.path.exists(result_file):
        with open(result_file) as f:
            existing = json.load(f)
        done_keys = set(
            (r["data_tag"], r["goal_tag"], r["interval"], r["task_tag"])
            for r in existing
        )
        print(f"Found {len(done_keys)} existing results, resuming...")
    else:
        done_keys = set()

    # 加载模型
    print(f"\n{'#'*70}")
    print(f"# Loading model: {model_tag}")
    print(f"# Path: {MODEL_PATH}")
    print(f"{'#'*70}")

    model = GRMInference(MODEL_PATH)
    print("Model loaded successfully!")

    all_results = list(existing)

    for seq, (data_dir, goal_img, interval, task) in enumerate(combos, 1):
        data_tag = get_data_tag(data_dir)
        goal_tag = get_goal_tag(goal_img)
        task_tag = get_task_tag(task)
        combo_key = (data_tag, goal_tag, interval, task_tag)

        # 跳过已完成的
        if combo_key in done_keys:
            print(f"\n[SKIP] [{seq}/{total}] {data_tag}/{goal_tag}/inter{interval}/{task_tag} (already done)")
            continue

        print(f"\n{'='*70}")
        print(f"  [{seq}/{total}] data={data_tag}  goal={goal_tag}  interval={interval}  task={task_tag}")
        print(f"{'='*70}")

        results_paths, timing_info = run_single_combo(
            model, seq, total, model_tag,
            data_dir, data_tag, goal_img, goal_tag,
            interval, task, task_tag,
        )

        out_dir = os.path.join(
            OUTPUT_ROOT, model_tag, data_tag, goal_tag, f"inter{interval}", task_tag
        )
        summary = analyze_and_plot(results_paths, out_dir, task, model_tag, interval)

        all_per_frame = [t["per_frame_s"] for t in timing_info.values() if t["per_frame_s"] > 0]
        avg_per_frame_s = round(np.mean(all_per_frame), 3) if all_per_frame else None

        result_entry = {
            "model_path": MODEL_PATH,
            "model_tag": model_tag,
            "data_dir": data_dir,
            "data_tag": data_tag,
            "goal_image": goal_img,
            "goal_tag": goal_tag,
            "interval": interval,
            "task": task.strip(),
            "task_tag": task_tag,
            "avg_progress": None,
            "timing": timing_info,
            "avg_per_frame_s": avg_per_frame_s,
        }
        if summary:
            result_entry["avg_progress"] = summary.get("avg")
            result_entry["forward_progress"] = summary.get("forward")
            result_entry["incremental_progress"] = summary.get("incremental")
            result_entry["backward_progress"] = summary.get("backward")
        all_results.append(result_entry)

        # 每完成一个就保存
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Model {model_tag} done! Total results: {len(all_results)}")
    print(f"Saved to: {result_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
