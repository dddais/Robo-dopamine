"""
合并两个模型的评估结果，生成完整汇总表格。
读取 _intermediate_results.json (模型1) 和 _intermediate_results_model2.json (模型2)，
合并后生成所有汇总表格、CSV、文字报告。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_ROOT = "/home/dais/workspace/Robo-Dopamine/results/auto_pick_carrot_fail"

INTERVALS = [20, 10]


def load_results():
    """加载两个模型的结果"""
    all_results = []

    # file1 = os.path.join(OUTPUT_ROOT, "_intermediate_results.json")
    file2 = os.path.join(OUTPUT_ROOT, "_intermediate_results_model_GRM8B.json")

    # if os.path.exists(file1):
    #     with open(file1) as f:
    #         all_results.extend(json.load(f))
    #     print(f"Loaded {file1}: {len(all_results)} entries")
    # else:
    #     print(f"WARNING: {file1} not found")

    if os.path.exists(file2):
        count_before = len(all_results)
        with open(file2) as f:
            all_results.extend(json.load(f))
        print(f"Loaded {file2}: {len(all_results) - count_before} entries")
    else:
        print(f"WARNING: {file2} not found")

    print(f"Total results: {len(all_results)}")
    return all_results


def _make_table(fig_ax, title, row_labels, col_labels, cell_text, cell_colors,
                col_header_color="#d0d0d0", row_label_color="#f0f0f0",
                fontsize=8, save_path=None, max_cols=None):
    """通用表格绘制工具，支持列过多时自动换行分排"""
    use_wrapped = max_cols is not None and len(col_labels) > max_cols

    if use_wrapped:
        if fig_ax is not None:
            plt.close(fig_ax[0])
        _make_wrapped_table(title, row_labels, col_labels, cell_text, cell_colors,
                            col_header_color, row_label_color, fontsize, save_path, max_cols)
        return

    fig, ax = fig_ax

    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    full_col_labels = ["Model \\ Data"] + col_labels if row_labels else col_labels
    full_col_colors = [col_header_color] * len(full_col_labels)

    all_cell_text = []
    all_cell_colors = []
    for i, rl in enumerate(row_labels):
        all_cell_text.append([rl] + cell_text[i])
        all_cell_colors.append([row_label_color] + cell_colors[i])

    tbl = ax.table(
        cellText=all_cell_text, colLabels=full_col_labels,
        cellColours=all_cell_colors, colColours=full_col_colors,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 2.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_wrapped_table(title, row_labels, col_labels, cell_text, cell_colors,
                        col_header_color, row_label_color, fontsize, save_path, max_cols):
    """将过多列拆成多排，每排最多 max_cols 列"""
    n_cols = len(col_labels)
    n_chunks = (n_cols + max_cols - 1) // max_cols
    n_models = len(row_labels)

    row_h = 0.5
    header_h = 0.6
    title_h = 0.6
    gap_h = 0.3
    total_h = title_h + n_chunks * (header_h + n_models * row_h + gap_h)
    col_w = 1.6
    total_w = max(max_cols + 1, 10) * col_w

    fig, axes = plt.subplots(n_chunks, 1,
                             figsize=(total_w, total_h),
                             gridspec_kw={"hspace": 0.15})
    if n_chunks == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    for ci, ax in enumerate(axes):
        start = ci * max_cols
        end = min(start + max_cols, n_cols)
        chunk_cols = col_labels[start:end]
        chunk_text = [row[start:end] for row in cell_text]
        chunk_colors = [row[start:end] for row in cell_colors]

        ax.axis("off")

        full_col_labels = ["Model \\ Data"] + chunk_cols if row_labels else chunk_cols
        full_col_colors = [col_header_color] * len(full_col_labels)

        all_cell_text = []
        all_cell_colors = []
        for i, rl in enumerate(row_labels):
            all_cell_text.append([rl] + chunk_text[i])
            all_cell_colors.append([row_label_color] + chunk_colors[i])

        tbl = ax.table(
            cellText=all_cell_text, colLabels=full_col_labels,
            cellColours=all_cell_colors, colColours=full_col_colors,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.scale(1.0, 2.0)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_summary_tables(all_results):
    """生成所有汇总表格（所有表格都包含 data 维度）"""
    output_dir = os.path.join(OUTPUT_ROOT, "summary_GRM8B_new")
    os.makedirs(output_dir, exist_ok=True)

    model_tags = sorted(set(r["model_tag"] for r in all_results))
    data_tags = sorted(set(r["data_tag"] for r in all_results))
    goal_tags = sorted(set(r["goal_tag"] for r in all_results))
    task_tags = sorted(set(r["task_tag"] for r in all_results))

    combo_keys = sorted(set(
        (r["data_tag"], r["goal_tag"], r["interval"], r["task_tag"])
        for r in all_results
    ))

    def progress_color(val):
        if val >= 80: return "#4CAF50"
        elif val >= 50: return "#FFC107"
        elif val >= 20: return "#FF9800"
        else: return "#F44336"

    def time_color(val):
        if val < 2.0: return "#4CAF50"
        elif val < 5.0: return "#FFC107"
        elif val < 10.0: return "#FF9800"
        else: return "#F44336"

    def _lookup_avg(results, model, data, goal=None, interval=None, task=None):
        """查找满足条件的 avg_progress 均值"""
        vals = [r["avg_progress"] for r in results
                if r["model_tag"] == model and r["data_tag"] == data
                and r.get("avg_progress") is not None
                and (goal is None or r["goal_tag"] == goal)
                and (interval is None or r["interval"] == interval)
                and (task is None or r["task_tag"] == task)]
        return np.mean(vals) if vals else None

    def _lookup_time(results, model, data, goal=None, interval=None, task=None):
        """查找满足条件的 avg_per_frame_s 均值"""
        vals = [r["avg_per_frame_s"] for r in results
                if r["model_tag"] == model and r["data_tag"] == data
                and r.get("avg_per_frame_s") is not None
                and (goal is None or r["goal_tag"] == goal)
                and (interval is None or r["interval"] == interval)
                and (task is None or r["task_tag"] == task)]
        return np.mean(vals) if vals else None

    # -------------------------------------------------------
    # 表1: 全量表格 (Model × Data_Goal_Inter_Task)
    # -------------------------------------------------------
    col_labels = [f"{ck[0]}\ngoal={ck[1]}\ninter={ck[2]}\ntask={ck[3]}" for ck in combo_keys]
    cell_text = []
    cell_colors = []

    for mt in model_tags:
        row, row_colors = [], []
        for ck in combo_keys:
            val = next(
                (r["avg_progress"] for r in all_results
                 if r["model_tag"] == mt and r["data_tag"] == ck[0]
                 and r["goal_tag"] == ck[1] and r["interval"] == ck[2]
                 and r["task_tag"] == ck[3] and r.get("avg_progress") is not None),
                None
            )
            if val is not None:
                row.append(f"{val:.1f}%")
                row_colors.append(progress_color(val))
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        cell_text.append(row)
        cell_colors.append(row_colors)

    _make_table(None, "Average Progress Summary (Full Table)",
                model_tags, col_labels, cell_text, cell_colors, fontsize=6,
                save_path=os.path.join(output_dir, "summary_full_table.png"),
                max_cols=15)
    print("Saved: summary_full_table.png")

    # -------------------------------------------------------
    # 表2: Model × Data (聚合 goal/interval/task)
    # -------------------------------------------------------
    cell_text2, cell_colors2 = [], []
    for mt in model_tags:
        row, row_colors = [], []
        for dt in data_tags:
            val = _lookup_avg(all_results, mt, dt)
            if val is not None:
                row.append(f"{val:.1f}%")
                row_colors.append(progress_color(val))
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        cell_text2.append(row)
        cell_colors2.append(row_colors)

    fig2, ax2 = plt.subplots(figsize=(max(10, len(data_tags) * 2.5), max(4, len(model_tags) * 1.5 + 2)))
    _make_table((fig2, ax2), "Avg Progress by Model × Data\n(mean over goal/interval/task)",
                model_tags, data_tags, cell_text2, cell_colors2,
                save_path=os.path.join(output_dir, "summary_by_model_data.png"))
    print("Saved: summary_by_model_data.png")

    # -------------------------------------------------------
    # 表3: Model × Data × Task
    # -------------------------------------------------------
    cell_text3, cell_colors3 = [], []
    for mt in model_tags:
        row, row_colors = [], []
        for dt in data_tags:
            for tt in task_tags:
                val = _lookup_avg(all_results, mt, dt, task=tt)
                if val is not None:
                    row.append(f"{val:.1f}%")
                    row_colors.append(progress_color(val))
                else:
                    row.append("N/A")
                    row_colors.append("#e0e0e0")
        cell_text3.append(row)
        cell_colors3.append(row_colors)

    col_labels3 = [f"{dt}\n{tt}" for dt in data_tags for tt in task_tags]
    _make_table(None, "Avg Progress by Model × Data × Task\n(mean over goal/interval)",
                model_tags, col_labels3, cell_text3, cell_colors3, fontsize=7,
                save_path=os.path.join(output_dir, "summary_by_model_data_task.png"),
                max_cols=15)
    print("Saved: summary_by_model_data_task.png")

    # -------------------------------------------------------
    # 表4: Model × Data × Goal
    # -------------------------------------------------------
    cell_text4, cell_colors4 = [], []
    for mt in model_tags:
        row, row_colors = [], []
        for dt in data_tags:
            for gt in goal_tags:
                val = _lookup_avg(all_results, mt, dt, goal=gt)
                if val is not None:
                    row.append(f"{val:.1f}%")
                    row_colors.append(progress_color(val))
                else:
                    row.append("N/A")
                    row_colors.append("#e0e0e0")
        cell_text4.append(row)
        cell_colors4.append(row_colors)

    col_labels4 = [f"{dt}\ngoal={gt}" for dt in data_tags for gt in goal_tags]
    fig4, ax4 = plt.subplots(figsize=(max(12, len(col_labels4) * 1.5), max(4, len(model_tags) * 1.5 + 2)))
    _make_table((fig4, ax4), "Avg Progress by Model × Data × Goal\n(mean over interval/task)",
                model_tags, col_labels4, cell_text4, cell_colors4, fontsize=8,
                save_path=os.path.join(output_dir, "summary_by_model_data_goal.png"))
    print("Saved: summary_by_model_data_goal.png")

    # -------------------------------------------------------
    # 表5: Model × Data × Interval
    # -------------------------------------------------------
    cell_text5, cell_colors5 = [], []
    for mt in model_tags:
        row, row_colors = [], []
        for dt in data_tags:
            for iv in INTERVALS:
                val = _lookup_avg(all_results, mt, dt, interval=iv)
                if val is not None:
                    row.append(f"{val:.1f}%")
                    row_colors.append(progress_color(val))
                else:
                    row.append("N/A")
                    row_colors.append("#e0e0e0")
        cell_text5.append(row)
        cell_colors5.append(row_colors)

    col_labels5 = [f"{dt}\ninter={iv}" for dt in data_tags for iv in INTERVALS]
    _make_table(None, "Avg Progress by Model × Data × Interval\n(mean over goal/task)",
                model_tags, col_labels5, cell_text5, cell_colors5, fontsize=8,
                save_path=os.path.join(output_dir, "summary_by_model_data_interval.png"),
                max_cols=15)
    print("Saved: summary_by_model_data_interval.png")

    # -------------------------------------------------------
    # 表6: 每帧耗时 Model × Data
    # -------------------------------------------------------
    cell_text6, cell_colors6 = [], []
    for mt in model_tags:
        row, row_colors = [], []
        for dt in data_tags:
            val = _lookup_time(all_results, mt, dt)
            if val is not None:
                row.append(f"{val:.3f}s")
                row_colors.append(time_color(val))
            else:
                row.append("N/A")
                row_colors.append("#e0e0e0")
        cell_text6.append(row)
        cell_colors6.append(row_colors)

    fig6, ax6 = plt.subplots(figsize=(max(10, len(data_tags) * 2.5), max(4, len(model_tags) * 1.5 + 2)))
    _make_table((fig6, ax6), "Avg Per-Frame Inference Time (s) by Model × Data\n(mean over goal/interval/task)",
                model_tags, data_tags, cell_text6, cell_colors6,
                save_path=os.path.join(output_dir, "summary_per_frame_time.png"))
    print("Saved: summary_per_frame_time.png")

    # -------------------------------------------------------
    # 表7: 每帧耗时 Model × Data × Interval
    # -------------------------------------------------------
    cell_text7, cell_colors7 = [], []
    for mt in model_tags:
        row, row_colors = [], []
        for dt in data_tags:
            for iv in INTERVALS:
                val = _lookup_time(all_results, mt, dt, interval=iv)
                if val is not None:
                    row.append(f"{val:.3f}s")
                    row_colors.append(time_color(val))
                else:
                    row.append("N/A")
                    row_colors.append("#e0e0e0")
        cell_text7.append(row)
        cell_colors7.append(row_colors)

    col_labels7 = [f"{dt}\ninter={iv}" for dt in data_tags for iv in INTERVALS]
    _make_table(None, "Avg Per-Frame Inference Time (s) by Model × Data × Interval\n(mean over goal/task)",
                model_tags, col_labels7, cell_text7, cell_colors7, fontsize=8,
                save_path=os.path.join(output_dir, "summary_time_by_model_data_interval.png"),
                max_cols=15)
    print("Saved: summary_time_by_model_data_interval.png")

    # -------------------------------------------------------
    # JSON
    # -------------------------------------------------------
    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: all_results.json")

    # -------------------------------------------------------
    # CSV
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
    print(f"Saved: all_results.csv")

    # -------------------------------------------------------
    # 文字版汇总
    # -------------------------------------------------------
    print(f"\n{'='*110}")
    print("ALL RESULTS TEXT SUMMARY")
    print(f"{'='*110}")
    print(f"{'Model':<18} {'Data':<16} {'Goal':<10} {'Inter':<7} {'Task':<10} {'Fwd%':<8} {'Inc%':<8} {'Bwd%':<8} {'Avg%':<8} {'s/frame':<9}")
    print("-" * 110)
    for r in all_results:
        fwd = f"{r.get('forward_progress', 0):.1f}" if r.get('forward_progress') is not None else "N/A"
        inc = f"{r.get('incremental_progress', 0):.1f}" if r.get('incremental_progress') is not None else "N/A"
        bwd = f"{r.get('backward_progress', 0):.1f}" if r.get('backward_progress') is not None else "N/A"
        avg = f"{r['avg_progress']:.1f}" if r.get('avg_progress') is not None else "N/A"
        pf = f"{r['avg_per_frame_s']:.3f}" if r.get('avg_per_frame_s') is not None else "N/A"
        print(f"{r['model_tag']:<18} {r['data_tag']:<16} {r['goal_tag']:<10} {r['interval']:<7} {r['task_tag']:<10} {fwd:<8} {inc:<8} {bwd:<8} {avg:<8} {pf:<9}")

    print(f"\n{'='*110}")
    print("DONE!")
    print(f"Summary output at: {output_dir}")
    print(f"{'='*110}")


def main():
    all_results = load_results()
    if not all_results:
        print("ERROR: No results found!")
        return
    generate_summary_tables(all_results)


if __name__ == "__main__":
    main()
