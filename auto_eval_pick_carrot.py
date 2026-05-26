"""
自动化批量评估脚本 —— pick carrot 任务
遍历 2模型 × 5数据 × 2目标图 × 2间隔 × 3任务指令 = 120 种组合，
每种组合跑 forward/incremental/backward 三种模式，
最终汇总 Average progress 并生成对比表格。
"""
import os
import sys
import json
import re
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from examples.inference import GRMInference

# ============================================================
# 配置区域
# ============================================================

# 两个模型
MODEL_PATHS = [
    "/home/dais/workspace/Robo-Dopamine/train/checkpoints/my_carrot_finetune_big",
    "./pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview",
]

# 5 个数据目录
DATA_DIRS = [
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_1",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_4",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_7",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3fail_11",
    "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1",
]

# 2 种目标图
GOAL_IMAGES = [
    "./examples/blank_goal.png",
    "./examples/exp_suc_1_sub1.png",
]

# 2 种采样间隔
INTERVALS = [20, 10]

# 3 种任务指令
TASK_INSTRUCTIONS = [
    "pick the carrot and put it on yellow plate ",
    "pick the bilue bottle and put it on yellow plate ",
    "pick the white cube and put it on yellow plate ",
]

# 输出根目录
OUTPUT_ROOT = "/home/dais/workspace/Robo-Dopamine/results/auto_pick_carrot_fail"

# 三种 eval 模式
EVAL_MODES = ["forward", "incremental", "backward"]


# ============================================================
# 辅助函数
# ============================================================

def get_model_tag(model_path: str) -> str:
    """从模型路径提取简短标识"""
    if "my_carrot_finetune_big" in model_path:
        return "finetune_big"
    elif "GRM-2.0-8B-Preview" in model_path:
        return "GRM-2.0-8B"
    return os.path.basename(model_path)


def get_data_tag(data_dir: str) -> str:
    """从数据目录路径提取简短标识"""
    return os.path.basename(data_dir)


def get_goal_tag(goal_image: str) -> str:
    """从目标图路径提取简短标识"""
    name = os.path.basename(goal_image)
    if name == "blank_goal.png":
        return "blank"
    elif name == "exp_suc_1_sub1.png":
        return "suc_sub1"
    return os.path.splitext(name)[0]


def get_task_tag(task: str) -> str:
    """从任务指令提取简短标识"""
    # 取前几个关键单词
    words = task.strip().split()
    if "carrot" in task:
        return "carrot"
    elif "bilue" in task or "bottle" in task:
        return "bottle"
    elif "white" in task or "cube" in task:
        return "cube"
    return "_".join(words[:3])


def run_single_combo(model, combo_idx, total_combos, model_tag, data_dir, data_tag,
                     goal_image, goal_tag, interval, task, task_tag):
    """对单个组合运行三种模式推理，返回结果摘要（含耗时）"""
    combo_name = f"[{combo_idx}/{total_combos}] model={model_tag} data={data_tag} goal={goal_tag} inter={interval} task={task_tag}"

    # 输出子目录命名: model/data/goal/interval/task
    out_dir = os.path.join(
        OUTPUT_ROOT, model_tag, data_tag, goal_tag, f"inter{interval}", task_tag
    )
    os.makedirs(out_dir, exist_ok=True)

    results_paths = {}
    timing_info = {}  # {mode: {"total_s": float, "num_frames": int, "per_frame_s": float}}
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

            # 统计推理帧数
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
    """分析结果，绘制进度曲线，返回各模式的 final progress"""
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

    # 计算 average progress
    modes = list(all_progress.keys())
    n = len(all_progress[modes[0]])
    avg_progress = []
    for i in range(n):
        avg_progress.append(sum(all_progress[m][i] for m in modes) / len(modes))
    all_progress["avg"] = avg_progress

    # 绘图
    fps = 30
    frames = [i * interval for i in range(n)]
    time_sec = [f / fps for f in frames]

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
    styles = {"forward": "-", "incremental": "-", "backward": "-", "avg": "--"}
    widths = {"forward": 2.0, "incremental": 2.0, "backward": 2.0, "avg": 3.0}

    fig, ax = plt.subplots(figsize=(12, 6))
    for key in ["forward", "incremental", "backward", "avg"]:
        if key in all_progress:
            ax.plot(
                time_sec, all_progress[key],
                styles[key],
                color=colors[key],
                linewidth=widths[key],
                label=f"{labels[key]}  (final={all_progress[key][-1]:.1f}%)",
                marker="o" if key == "avg" else None,
                markersize=3,
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

    # 返回各模式的最终值
    summary = {}
    for key in ["forward", "incremental", "backward", "avg"]:
        if key in all_progress:
            summary[key] = all_progress[key][-1]
    return summary


def generate_summary_tables(all_results):
    """生成汇总表格，按不同维度展示 Average progress"""
    output_dir = os.path.join(OUTPUT_ROOT, "summary")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------
    # 表1: 按模型 × (数据+目标图+间隔+任务) 展示 avg progress
    # -------------------------------------------------------
    # 构建所有唯一的组合键
    combo_keys = sorted(set(
        (r["data_tag"], r["goal_tag"], r["interval"], r["task_tag"])
        for r in all_results
    ))
    model_tags = sorted(set(r["model_tag"] for r in all_results))

    # 表1: 全量表格
    fig1, ax1 = plt.subplots(figsize=(max(16, len(combo_keys) * 1.2), max(6, len(model_tags) * 1.5 + 2)))
    ax1.axis("off")
    ax1.set_title("Average Progress Summary (Full Table)", fontsize=14, fontweight="bold", pad=20)

    col_labels = ["Model"] + [f"{ck[0]}\ngoal={ck[1]}\ninter={ck[2]}\ntask={ck[3]}" for ck in combo_keys]
    cell_text = []
    cell_colors = []

    for mt in model_tags:
        row = [mt]
        row_colors = ["#f0f0f0"]
        for ck in combo_keys:
            val = next(
                (r["avg_progress"] for r in all_results
                 if r["model_tag"] == mt and r["data_tag"] == ck[0]
                 and r["goal_tag"] == ck[1] and r["interval"] == ck[2]
                 and r["task_tag"] == ck[3]),
                None
            )
            if val is not None:
                row.append(f"{val:.1f}%")
                # 热力图颜色
                if val >= 80:
                    row_colors.append("#4CAF50")
                elif val >= 50:
                    row_colors.append("#FFC107")
                elif val >= 20:
                    row_colors.append("#FF9800")
                else:
                    row_colors.append("#F44336")
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        cell_text.append(row)
        cell_colors.append(row_colors)

    table1 = ax1.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#d0d0d0"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(7)
    table1.scale(1.0, 2.0)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "summary_full_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Full summary table saved: {os.path.join(output_dir, 'summary_full_table.png')}")

    # -------------------------------------------------------
    # 表2: 按模型 × 数据 展示 avg（聚合其他维度取均值）
    # -------------------------------------------------------
    data_tags = sorted(set(r["data_tag"] for r in all_results))

    fig2, ax2 = plt.subplots(figsize=(max(10, len(data_tags) * 2.5), max(4, len(model_tags) * 1.5 + 2)))
    ax2.axis("off")
    ax2.set_title("Average Progress by Model × Data (mean over goal/interval/task)", fontsize=13, fontweight="bold", pad=20)

    col_labels2 = ["Model"] + data_tags + ["Overall"]
    cell_text2 = []
    cell_colors2 = []

    for mt in model_tags:
        row = [mt]
        row_colors = ["#f0f0f0"]
        model_vals = []
        for dt in data_tags:
            vals = [r["avg_progress"] for r in all_results
                    if r["model_tag"] == mt and r["data_tag"] == dt]
            if vals:
                mean_v = np.mean(vals)
                row.append(f"{mean_v:.1f}%")
                model_vals.append(mean_v)
                if mean_v >= 80:
                    row_colors.append("#4CAF50")
                elif mean_v >= 50:
                    row_colors.append("#FFC107")
                elif mean_v >= 20:
                    row_colors.append("#FF9800")
                else:
                    row_colors.append("#F44336")
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        if model_vals:
            overall = np.mean(model_vals)
            row.append(f"{overall:.1f}%")
            if overall >= 80:
                row_colors.append("#4CAF50")
            elif overall >= 50:
                row_colors.append("#FFC107")
            elif overall >= 20:
                row_colors.append("#FF9800")
            else:
                row_colors.append("#F44336")
        else:
            row.append("N/A")
            row_colors.append("#e0e0e0")
        cell_text2.append(row)
        cell_colors2.append(row_colors)

    table2 = ax2.table(
        cellText=cell_text2,
        colLabels=col_labels2,
        cellColours=cell_colors2,
        colColours=["#d0d0d0"] * len(col_labels2),
        loc="center",
        cellLoc="center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.0, 2.0)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "summary_by_model_data.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Model×Data summary saved: {os.path.join(output_dir, 'summary_by_model_data.png')}")

    # -------------------------------------------------------
    # 表3: 按模型 × 任务指令 展示 avg
    # -------------------------------------------------------
    task_tags = sorted(set(r["task_tag"] for r in all_results))

    fig3, ax3 = plt.subplots(figsize=(max(10, len(task_tags) * 3), max(4, len(model_tags) * 1.5 + 2)))
    ax3.axis("off")
    ax3.set_title("Average Progress by Model × Task (mean over data/goal/interval)", fontsize=13, fontweight="bold", pad=20)

    col_labels3 = ["Model"] + task_tags + ["Overall"]
    cell_text3 = []
    cell_colors3 = []

    for mt in model_tags:
        row = [mt]
        row_colors = ["#f0f0f0"]
        model_vals = []
        for tt in task_tags:
            vals = [r["avg_progress"] for r in all_results
                    if r["model_tag"] == mt and r["task_tag"] == tt]
            if vals:
                mean_v = np.mean(vals)
                row.append(f"{mean_v:.1f}%")
                model_vals.append(mean_v)
                if mean_v >= 80:
                    row_colors.append("#4CAF50")
                elif mean_v >= 50:
                    row_colors.append("#FFC107")
                elif mean_v >= 20:
                    row_colors.append("#FF9800")
                else:
                    row_colors.append("#F44336")
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        if model_vals:
            overall = np.mean(model_vals)
            row.append(f"{overall:.1f}%")
            if overall >= 80:
                row_colors.append("#4CAF50")
            elif overall >= 50:
                row_colors.append("#FFC107")
            elif overall >= 20:
                row_colors.append("#FF9800")
            else:
                row_colors.append("#F44336")
        else:
            row.append("N/A")
            row_colors.append("#e0e0e0")
        cell_text3.append(row)
        cell_colors3.append(row_colors)

    table3 = ax3.table(
        cellText=cell_text3,
        colLabels=col_labels3,
        cellColours=cell_colors3,
        colColours=["#d0d0d0"] * len(col_labels3),
        loc="center",
        cellLoc="center",
    )
    table3.auto_set_font_size(False)
    table3.set_fontsize(9)
    table3.scale(1.0, 2.0)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "summary_by_model_task.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Model×Task summary saved: {os.path.join(output_dir, 'summary_by_model_task.png')}")

    # -------------------------------------------------------
    # 表4: 按模型 × 目标图 展示 avg
    # -------------------------------------------------------
    goal_tags = sorted(set(r["goal_tag"] for r in all_results))

    fig4, ax4 = plt.subplots(figsize=(max(8, len(goal_tags) * 3), max(4, len(model_tags) * 1.5 + 2)))
    ax4.axis("off")
    ax4.set_title("Average Progress by Model × Goal (mean over data/interval/task)", fontsize=13, fontweight="bold", pad=20)

    col_labels4 = ["Model"] + goal_tags + ["Overall"]
    cell_text4 = []
    cell_colors4 = []

    for mt in model_tags:
        row = [mt]
        row_colors = ["#f0f0f0"]
        model_vals = []
        for gt in goal_tags:
            vals = [r["avg_progress"] for r in all_results
                    if r["model_tag"] == mt and r["goal_tag"] == gt]
            if vals:
                mean_v = np.mean(vals)
                row.append(f"{mean_v:.1f}%")
                model_vals.append(mean_v)
                if mean_v >= 80:
                    row_colors.append("#4CAF50")
                elif mean_v >= 50:
                    row_colors.append("#FFC107")
                elif mean_v >= 20:
                    row_colors.append("#FF9800")
                else:
                    row_colors.append("#F44336")
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        if model_vals:
            overall = np.mean(model_vals)
            row.append(f"{overall:.1f}%")
            if overall >= 80:
                row_colors.append("#4CAF50")
            elif overall >= 50:
                row_colors.append("#FFC107")
            elif overall >= 20:
                row_colors.append("#FF9800")
            else:
                row_colors.append("#F44336")
        else:
            row.append("N/A")
            row_colors.append("#e0e0e0")
        cell_text4.append(row)
        cell_colors4.append(row_colors)

    table4 = ax4.table(
        cellText=cell_text4,
        colLabels=col_labels4,
        cellColours=cell_colors4,
        colColours=["#d0d0d0"] * len(col_labels4),
        loc="center",
        cellLoc="center",
    )
    table4.auto_set_font_size(False)
    table4.set_fontsize(9)
    table4.scale(1.0, 2.0)
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "summary_by_model_goal.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Model×Goal summary saved: {os.path.join(output_dir, 'summary_by_model_goal.png')}")

    # -------------------------------------------------------
    # 表5: 按模型 × 间隔 展示 avg
    # -------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(max(8, len(INTERVALS) * 3), max(4, len(model_tags) * 1.5 + 2)))
    ax5.axis("off")
    ax5.set_title("Average Progress by Model × Interval (mean over data/goal/task)", fontsize=13, fontweight="bold", pad=20)

    col_labels5 = ["Model"] + [f"inter={i}" for i in INTERVALS] + ["Overall"]
    cell_text5 = []
    cell_colors5 = []

    for mt in model_tags:
        row = [mt]
        row_colors = ["#f0f0f0"]
        model_vals = []
        for iv in INTERVALS:
            vals = [r["avg_progress"] for r in all_results
                    if r["model_tag"] == mt and r["interval"] == iv]
            if vals:
                mean_v = np.mean(vals)
                row.append(f"{mean_v:.1f}%")
                model_vals.append(mean_v)
                if mean_v >= 80:
                    row_colors.append("#4CAF50")
                elif mean_v >= 50:
                    row_colors.append("#FFC107")
                elif mean_v >= 20:
                    row_colors.append("#FF9800")
                else:
                    row_colors.append("#F44336")
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        if model_vals:
            overall = np.mean(model_vals)
            row.append(f"{overall:.1f}%")
            if overall >= 80:
                row_colors.append("#4CAF50")
            elif overall >= 50:
                row_colors.append("#FFC107")
            elif overall >= 20:
                row_colors.append("#FF9800")
            else:
                row_colors.append("#F44336")
        else:
            row.append("N/A")
            row_colors.append("#e0e0e0")
        cell_text5.append(row)
        cell_colors5.append(row_colors)

    table5 = ax5.table(
        cellText=cell_text5,
        colLabels=col_labels5,
        cellColours=cell_colors5,
        colColours=["#d0d0d0"] * len(col_labels5),
        loc="center",
        cellLoc="center",
    )
    table5.auto_set_font_size(False)
    table5.set_fontsize(9)
    table5.scale(1.0, 2.0)
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, "summary_by_model_interval.png"), dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"Model×Interval summary saved: {os.path.join(output_dir, 'summary_by_model_interval.png')}")

    # -------------------------------------------------------
    # 表6: 按模型 × 数据 展示平均每帧耗时 (秒)
    # -------------------------------------------------------
    fig6, ax6 = plt.subplots(figsize=(max(10, len(data_tags) * 2.5), max(4, len(model_tags) * 1.5 + 2)))
    ax6.axis("off")
    ax6.set_title("Avg Per-Frame Inference Time (s) by Model × Data", fontsize=13, fontweight="bold", pad=20)

    col_labels6 = ["Model"] + data_tags + ["Overall"]
    cell_text6 = []
    cell_colors6 = []

    for mt in model_tags:
        row = [mt]
        row_colors = ["#f0f0f0"]
        model_vals = []
        for dt in data_tags:
            vals = [r["avg_per_frame_s"] for r in all_results
                    if r["model_tag"] == mt and r["data_tag"] == dt
                    and r.get("avg_per_frame_s") is not None]
            if vals:
                mean_v = np.mean(vals)
                row.append(f"{mean_v:.3f}s")
                model_vals.append(mean_v)
                # 颜色：越快越绿，越慢越红
                if mean_v < 2.0:
                    row_colors.append("#4CAF50")
                elif mean_v < 5.0:
                    row_colors.append("#FFC107")
                elif mean_v < 10.0:
                    row_colors.append("#FF9800")
                else:
                    row_colors.append("#F44336")
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        if model_vals:
            overall = np.mean(model_vals)
            row.append(f"{overall:.3f}s")
            if overall < 2.0:
                row_colors.append("#4CAF50")
            elif overall < 5.0:
                row_colors.append("#FFC107")
            elif overall < 10.0:
                row_colors.append("#FF9800")
            else:
                row_colors.append("#F44336")
        else:
            row.append("N/A")
            row_colors.append("#e0e0e0")
        cell_text6.append(row)
        cell_colors6.append(row_colors)

    table6 = ax6.table(
        cellText=cell_text6,
        colLabels=col_labels6,
        cellColours=cell_colors6,
        colColours=["#d0d0d0"] * len(col_labels6),
        loc="center",
        cellLoc="center",
    )
    table6.auto_set_font_size(False)
    table6.set_fontsize(9)
    table6.scale(1.0, 2.0)
    fig6.tight_layout()
    fig6.savefig(os.path.join(output_dir, "summary_per_frame_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print(f"Per-frame time summary saved: {os.path.join(output_dir, 'summary_per_frame_time.png')}")

    # -------------------------------------------------------
    # 保存 JSON 格式的完整汇总数据
    # -------------------------------------------------------
    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full results JSON saved: {json_path}")

    # -------------------------------------------------------
    # 保存 CSV 格式（方便用 Excel 查看）
    # -------------------------------------------------------
    csv_path = os.path.join(output_dir, "all_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = "model,data,goal,interval,task,forward_progress,incremental_progress,backward_progress,avg_progress,avg_per_frame_s\n"
        f.write(header)
        for r in all_results:
            f.write(f"{r['model_tag']},{r['data_tag']},{r['goal_tag']},{r['interval']},{r['task_tag']},"
                    f"{r.get('forward_progress', 'N/A')},{r.get('incremental_progress', 'N/A')},"
                    f"{r.get('backward_progress', 'N/A')},{r.get('avg_progress', 'N/A')},"
                    f"{r.get('avg_per_frame_s', 'N/A')}\n")
    print(f"CSV results saved: {csv_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 生成所有组合
    combos = list(itertools.product(
        MODEL_PATHS, DATA_DIRS, GOAL_IMAGES, INTERVALS, TASK_INSTRUCTIONS
    ))
    total = len(combos)
    print(f"Total combinations: {total}")
    print(f"Output root: {OUTPUT_ROOT}")

    all_results = []

    # 按模型分组，避免重复加载
    for model_path in MODEL_PATHS:
        model_tag = get_model_tag(model_path)
        print(f"\n{'#'*70}")
        print(f"# Loading model: {model_tag}")
        print(f"# Path: {model_path}")
        print(f"{'#'*70}")

        model = GRMInference(model_path)
        print("Model loaded successfully!")

        # 筛选该模型的所有组合
        model_combos = [(i, c) for i, c in enumerate(combos) if c[0] == model_path]

        for seq, (combo_idx, (mp, data_dir, goal_img, interval, task)) in enumerate(model_combos, 1):
            data_tag = get_data_tag(data_dir)
            goal_tag = get_goal_tag(goal_img)
            task_tag = get_task_tag(task)

            print(f"\n{'='*70}")
            print(f"  [{seq}/{len(model_combos)}] for model {model_tag}")
            print(f"  data={data_tag}  goal={goal_tag}  interval={interval}  task={task_tag}")
            print(f"{'='*70}")

            results_paths, timing_info = run_single_combo(
                model, seq, len(model_combos), model_tag,
                data_dir, data_tag, goal_img, goal_tag,
                interval, task, task_tag,
            )

            # 分析并绘图
            out_dir = os.path.join(
                OUTPUT_ROOT, model_tag, data_tag, goal_tag, f"inter{interval}", task_tag
            )
            summary = analyze_and_plot(results_paths, out_dir, task, model_tag, interval)

            # 计算该组合的平均每帧耗时（按数据维度聚合三种模式）
            all_per_frame = [t["per_frame_s"] for t in timing_info.values() if t["per_frame_s"] > 0]
            avg_per_frame_s = round(np.mean(all_per_frame), 3) if all_per_frame else None

            # 记录结果
            result_entry = {
                "model_path": mp,
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

            # 每完成一个组合就保存中间结果
            mid_json = os.path.join(OUTPUT_ROOT, "_intermediate_results.json")
            with open(mid_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 释放模型显存
        del model
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\nModel {model_tag} released from GPU.")

    # 生成汇总表格
    print(f"\n{'#'*70}")
    print("# Generating summary tables...")
    print(f"{'#'*70}")
    generate_summary_tables(all_results)

    # 打印文字版汇总
    print(f"\n{'='*70}")
    print("ALL RESULTS TEXT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'Data':<16} {'Goal':<10} {'Inter':<7} {'Task':<10} {'Fwd%':<8} {'Inc%':<8} {'Bwd%':<8} {'Avg%':<8} {'s/frame':<9}")
    print("-" * 110)
    for r in all_results:
        fwd = f"{r.get('forward_progress', 0):.1f}" if r.get('forward_progress') is not None else "N/A"
        inc = f"{r.get('incremental_progress', 0):.1f}" if r.get('incremental_progress') is not None else "N/A"
        bwd = f"{r.get('backward_progress', 0):.1f}" if r.get('backward_progress') is not None else "N/A"
        avg = f"{r['avg_progress']:.1f}" if r.get('avg_progress') is not None else "N/A"
        pf = f"{r['avg_per_frame_s']:.3f}" if r.get('avg_per_frame_s') is not None else "N/A"
        print(f"{r['model_tag']:<18} {r['data_tag']:<16} {r['goal_tag']:<10} {r['interval']:<7} {r['task_tag']:<10} {fwd:<8} {inc:<8} {bwd:<8} {avg:<8} {pf:<9}")

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"Results at: {OUTPUT_ROOT}")
    print(f"Summary tables at: {os.path.join(OUTPUT_ROOT, 'summary')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
