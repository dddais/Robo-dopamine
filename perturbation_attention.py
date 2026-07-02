"""
扰动法（Perturbation-based Attribution）分析脚本
=================================================

目标：
    分析 GRM-2.0-8B-Preview 模型在输出 task progress score 时，
    主要关注图像的哪些区域。

原理：
    对 AFTER High 图像（当前帧）划分 grid_size × grid_size 网格，
    逐个 mask 掉每个网格，观察 score 的变化。
    变化越大 = 该区域对模型输出越重要。

输出：
    1. 每一帧的 attribution heatmap (PNG)
    2. 一个汇总视频 attention_video.mp4，展示随任务进行，
       影响得分的图像区域如何变化。

用法：
    python perturbation_attention.py

依赖：
    与 examples/inference.py 一致（vLLM + transformers + moviepy）。
"""

import os
import sys
import json
import re
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 视频输出：优先 moviepy（与 examples/inference.py 一致），缺失时退回 cv2
_MOVIEPY_AVAILABLE = False
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    _MOVIEPY_AVAILABLE = True
except ImportError:
    pass

# 复用项目现有推理模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from examples.inference import GRMInference


# ============================================================
# 配置区（按需修改）
# ============================================================

MODEL_PATH = './pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview'

# 数据目录（包含 cam_high.mp4 / cam_left_wrist.mp4 / cam_right_wrist.mp4）
DATA_DIR = "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot"

TASK_INSTRUCTION = "pick the carrot and put it on the plate"
# GOAL_IMAGE = "./examples/xzx_ep1_sub2.png"
GOAL_IMAGE = "./examples/blank_goal.png"

FRAME_INTERVAL = 30          # 采样间隔（每隔多少帧取一帧分析）
EVAL_MODE = "forward"        # forward / incremental / backward

# 扰动法参数
GRID_SIZE = 8                # 网格划分粒度（8×8=64 个区域）
TARGET_VIEW_INDEX = 5        # 扰动 8 张图中的哪一张
                               # 索引说明：
                               #   0=Ref Start, 1=Ref End
                               #   2=BEFORE High, 3=BEFORE Left, 4=BEFORE Right
                               #   5=AFTER High,  6=AFTER Left,  7=AFTER Right
MASK_VALUE = 0               # mask 方式的像素值 (0=纯黑；可选 128=灰色均值)

# 输出根目录
OUTPUT_ROOT = "./results/perturbation_attribution"

# 视频参数
VIDEO_FPS = 2.0              # 汇总视频帧率（每秒显示多少个时间步）


# ============================================================
# 1. 帧提取（复用 inference.py 的逻辑）
# ============================================================

def get_frame_count(path: Path) -> Tuple[str, int]:
    """检测输入是视频还是图像目录，返回 (类型, 帧数)。"""
    if path.is_dir():
        files = sorted([p for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() == ".png"])
        if not files:
            raise RuntimeError(f"No PNG frames in directory: {path}")
        return "dir", len(files)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n <= 0:
            raise RuntimeError(f"Invalid frame count: {path}")
        return "video", n


def make_sample_indices(num_frames: int, interval: int) -> List[int]:
    """生成采样帧索引：0, interval, 2*interval, ..., last。"""
    if num_frames < 1:
        return []
    indices = list(range(0, num_frames, interval))
    last_idx = num_frames - 1
    if not indices or indices[-1] != last_idx:
        indices.append(last_idx)
    return indices


def save_frames(src_path: Path, out_dir: Path, indices: List[int], src_type: str) -> None:
    """从视频或目录提取指定帧到 out_dir，命名 frame_{idx:06d}.png。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    if src_type == "video":
        cap = cv2.VideoCapture(str(src_path))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[WARN] Failed to read frame {idx} from {src_path}")
                continue
            out_path = out_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cap.release()
    else:  # dir
        files = sorted([p for p in src_path.iterdir()
                        if p.is_file() and p.suffix.lower() == ".png"])
        n = len(files)
        for idx in indices:
            if not (0 <= idx < n):
                continue
            shutil.copyfile(files[idx], out_dir / f"frame_{idx:06d}.png")


# ============================================================
# 2. 构建 samples（复用 inference.py 的逻辑）
# ============================================================

def build_samples(
    cache_root: Path,
    task: str,
    indices: List[int],
    ref_end_path: str,
    mode: str = "forward",
) -> List[Dict]:
    """构建推理样本列表，每个样本含 8 张图路径。"""
    items = []
    if len(indices) < 2:
        return items

    for k in range(len(indices) - 1):
        af = indices[k + 1]

        if mode == "incremental":
            bf = indices[k]
            bf_id_str = f"bf_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high"        / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist"  / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "forward":
            bf = indices[0]
            bf_id_str = f"start_{bf:06d}"
            bf_images = [
                str(cache_root / "cam_high"        / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist"  / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "backward":
            bf_id_str = "goal"
            bf_images = [ref_end_path, ref_end_path, ref_end_path]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        af_images = [
            str(cache_root / "cam_high"        / f"frame_{af:06d}.png"),
            str(cache_root / "cam_left_wrist"  / f"frame_{af:06d}.png"),
            str(cache_root / "cam_right_wrist" / f"frame_{af:06d}.png"),
        ]

        items.append({
            "id": f"step-{k:04d}-{bf_id_str}-af_{af:06d}",
            "task": task,
            "image": [
                str(cache_root / "cam_high" / f"frame_{0:06d}.png"),   # 0. Ref Start
                ref_end_path,                                             # 1. Ref End
                bf_images[0],                                             # 2. BEFORE High
                bf_images[1],                                             # 3. BEFORE Left
                bf_images[2],                                             # 4. BEFORE Right
                af_images[0],                                             # 5. AFTER High  ← 扰动目标
                af_images[1],                                             # 6. AFTER Left
                af_images[2],                                             # 7. AFTER Right
            ],
        })
    return items


# ============================================================
# 3. Score 解析
# ============================================================

def parse_score(pred_text: str) -> float:
    """从模型输出 '<score>+30%</score>' 中解析出浮点分数（-100~100）。"""
    try:
        # 优先匹配 <score>...</score>
        m = re.search(r'<score>\s*([+-]?\d+\.?\d*)\s*%?\s*</score>', pred_text)
        if m:
            return float(m.group(1))
        # 兜底：匹配任意百分比
        m = re.search(r'([+-]?\d+\.?\d*)%', pred_text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 0.0


# ============================================================
# 4. 图像 mask 工具
# ============================================================

def mask_grid_on_image(image_path: str, grid_row: int, grid_col: int,
                       grid_size: int = 8, mask_value: int = 0) -> str:
    """
    把图像的 (grid_row, grid_col) 网格区域涂成 mask_value，
    保存到临时文件，返回新路径。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    y1, y2 = grid_row * cell_h, (grid_row + 1) * cell_h
    x1, x2 = grid_col * cell_w, (grid_col + 1) * cell_w
    img[y1:y2, x1:x2] = mask_value

    # 保存到临时文件（同目录加后缀，避免改原文件）
    base, ext = os.path.splitext(image_path)
    new_path = f"{base}_masked_r{grid_row}c{grid_col}{ext}"
    cv2.imwrite(new_path, img)
    return new_path


def cleanup_masked_files(cache_root: Path, grid_size: int) -> None:
    """删除扰动产生的临时 mask 文件。"""
    for sub in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        d = cache_root / sub
        if not d.exists():
            continue
        for p in d.glob("*_masked_r*c*.png"):
            try:
                p.unlink()
            except Exception:
                pass


# ============================================================
# 5. 扰动归因核心
# ============================================================

def compute_attribution_map(
    model: GRMInference,
    item: Dict,
    grid_size: int = 8,
    view_idx: int = 5,
    mask_value: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    对单个样本计算扰动归因图。

    返回:
        attribution_map: [grid_size, grid_size] 每个网格的 |Δscore|
        orig_score: 原始 score（未扰动）
    """
    # 1. 原始 score
    orig_out = model.inference_batch([item])[0]
    orig_score = parse_score(orig_out["pred"])

    # 2. 遍历每个网格
    attribution_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    target_image_path = item["image"][view_idx]

    for r in range(grid_size):
        for c in range(grid_size):
            # 2.1 构造 masked item
            masked_path = mask_grid_on_image(
                target_image_path, r, c, grid_size, mask_value
            )
            masked_item = deepcopy(item)
            masked_item["image"] = list(item["image"])  # 浅拷贝列表
            masked_item["image"][view_idx] = masked_path

            # 2.2 推理
            try:
                masked_out = model.inference_batch([masked_item])[0]
                masked_score = parse_score(masked_out["pred"])
            except Exception as e:
                print(f"[WARN] inference failed at r={r},c={c}: {e}")
                masked_score = orig_score  # 失败时视为无影响

            # 2.3 记录影响（绝对值变化）
            attribution_map[r, c] = abs(orig_score - masked_score)

            # 2.4 删除临时文件
            try:
                os.remove(masked_path)
            except Exception:
                pass

    return attribution_map, orig_score


# ============================================================
# 6. 可视化：单帧 heatmap
# ============================================================

def render_attribution_frame(
    image_path: str,
    attribution_map: np.ndarray,
    orig_score: float,
    step_idx: int,
    frame_idx: int,
    task: str,
    grid_size: int,
    out_path: str,
    panel_w: int = 384,
    panel_h: int = 288,
) -> np.ndarray:
    """
    生成单帧可视化（原图 + heatmap 叠加 + 文字信息），返回 RGB 数组。
    """
    # 1. 读图
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (panel_w, panel_h))

    # 2. 归一化 attribution
    attr = attribution_map.copy()
    max_val = attr.max()
    if max_val > 1e-6:
        attr_norm = attr / max_val
    else:
        attr_norm = attr

    # 3. 上采样到原图大小
    attr_resized = cv2.resize(attr_norm, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)

    # 4. colormap
    heatmap_uint8 = (attr_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 5. 叠加
    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

    # 6. 拼图：[原图 | 热力图 | 叠加]
    canvas = np.hstack([img, heatmap_color, overlay])  # 宽 = 3*panel_w

    # 7. 顶部信息条
    info_h = 50
    info_bar = np.ones((info_h, canvas.shape[1], 3), dtype=np.uint8) * 30
    info_txt = (f"Step {step_idx}  Frame {frame_idx}  |  "
                f"Score: {orig_score:+.1f}%  |  Max impact: {max_val:.1f}  |  "
                f"Grid: {grid_size}x{grid_size}")
    cv2.putText(info_bar, info_txt, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 8. 底部任务说明
    task_h = 30
    task_bar = np.ones((task_h, canvas.shape[1], 3), dtype=np.uint8) * 30
    task_txt = f"Task: {task}"
    cv2.putText(task_bar, task_txt, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 9. 列标题
    title_h = 25
    title_bar = np.ones((title_h, canvas.shape[1], 3), dtype=np.uint8) * 60
    for i, txt in enumerate(["Original", "Attribution", "Overlay"]):
        cv2.putText(title_bar, txt,
                    (i * panel_w + panel_w // 2 - 40, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    final = np.vstack([info_bar, title_bar, canvas, task_bar])

    # 保存单帧 PNG（可选）
    if out_path:
        fig = plt.figure(figsize=(12, 4))
        plt.imshow(final)
        plt.axis('off')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    return final


# ============================================================
# 7. 主流程
# ============================================================

def main():
    print(f"\n{'='*70}")
    print("Perturbation-based Attribution Analysis")
    print(f"{'='*70}")
    print(f"Model:        {MODEL_PATH}")
    print(f"Data:         {DATA_DIR}")
    print(f"Task:         {TASK_INSTRUCTION}")
    print(f"Eval mode:    {EVAL_MODE}")
    print(f"Grid size:    {GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE*GRID_SIZE} regions per frame")
    print(f"Target view:  image[{TARGET_VIEW_INDEX}] (AFTER High)")
    print(f"{'='*70}\n")

    # --- 准备输出目录 ---
    ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run_root = Path(OUTPUT_ROOT) / f"{ts}_{EVAL_MODE}_mode"
    cache_root = run_root / ".cache"
    frames_dir = run_root / "frames"
    heatmaps_dir = run_root / "heatmaps"
    run_root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    heatmaps_dir.mkdir(exist_ok=True)

    # --- 1. 提取帧 ---
    print("[1/5] Extracting frames ...")
    cam_dirs = {
        "cam_high":        cache_root / "cam_high",
        "cam_left_wrist":  cache_root / "cam_left_wrist",
        "cam_right_wrist": cache_root / "cam_right_wrist",
    }
    for d in cam_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    paths = [
        Path(os.path.join(DATA_DIR, "cam_high.mp4")),
        Path(os.path.join(DATA_DIR, "cam_left_wrist.mp4")),
        Path(os.path.join(DATA_DIR, "cam_right_wrist.mp4")),
    ]
    types_counts = [get_frame_count(p) for p in paths]
    counts = [tc[1] for tc in types_counts]
    if len(set(counts)) != 1:
        raise ValueError(f"Frame count mismatch: {counts}")
    total_frames = counts[0]

    indices = make_sample_indices(total_frames, FRAME_INTERVAL)
    print(f"  Total frames: {total_frames}, Sampled: {len(indices)} (interval={FRAME_INTERVAL})")

    for p, key, (stype, _) in zip(paths, cam_dirs.keys(), types_counts):
        save_frames(p, cam_dirs[key], indices, stype)

    # --- 2. Goal image ---
    if GOAL_IMAGE and os.path.exists(GOAL_IMAGE):
        ref_end_path = str(cache_root / "ref_end.png")
        shutil.copy(GOAL_IMAGE, ref_end_path)
    else:
        ref_end_path = str(cam_dirs["cam_high"] / f"frame_{total_frames-1:06d}.png")
    print(f"  Goal image: {ref_end_path}\n")

    # --- 3. 构建 samples ---
    print("[2/5] Building samples ...")
    samples = build_samples(cache_root, TASK_INSTRUCTION, indices, ref_end_path, mode=EVAL_MODE)
    print(f"  Samples: {len(samples)}\n")

    # --- 4. 加载模型 ---
    print("[3/5] Loading model ...")
    model = GRMInference(MODEL_PATH)
    print("  Model loaded.\n")

    # --- 5. 逐帧扰动归因 ---
    print("[4/5] Running perturbation attribution ...")
    video_frames = []  # 存 RGB 数组
    all_records = []

    total_steps = len(samples)
    per_frame_calls = GRID_SIZE * GRID_SIZE + 1  # 1 原始 + N 扰动

    for step_idx, item in enumerate(samples):
        t0 = time.time()
        frame_idx = indices[step_idx + 1]
        target_image_path = item["image"][TARGET_VIEW_INDEX]

        print(f"\n  [Step {step_idx+1}/{total_steps}] frame_idx={frame_idx}")
        print(f"    Running {per_frame_calls} inferences (1 orig + {GRID_SIZE*GRID_SIZE} perturbed) ...")

        attribution_map, orig_score = compute_attribution_map(
            model=model,
            item=item,
            grid_size=GRID_SIZE,
            view_idx=TARGET_VIEW_INDEX,
            mask_value=MASK_VALUE,
        )

        elapsed = time.time() - t0
        max_impact = float(attribution_map.max())
        print(f"    Score: {orig_score:+.1f}%  |  Max impact: {max_impact:.1f}  |  Time: {elapsed:.1f}s")

        # 保存 heatmap
        heatmap_png = str(heatmaps_dir / f"heatmap_step{step_idx:04d}_frame{frame_idx:06d}.png")
        frame_rgb = render_attribution_frame(
            image_path=target_image_path,
            attribution_map=attribution_map,
            orig_score=orig_score,
            step_idx=step_idx,
            frame_idx=frame_idx,
            task=TASK_INSTRUCTION,
            grid_size=GRID_SIZE,
            out_path=heatmap_png,
        )
        video_frames.append(frame_rgb)

        # 记录数值
        all_records.append({
            "step": step_idx,
            "frame_idx": frame_idx,
            "orig_score": orig_score,
            "max_impact": max_impact,
            "mean_impact": float(attribution_map.mean()),
            "attribution_map": attribution_map.tolist(),
        })

    print(f"\n  Attribution done. {len(video_frames)} frames rendered.\n")

    # 清理临时 mask 文件
    cleanup_masked_files(cache_root, GRID_SIZE)

    # --- 6. 保存 JSON + 视频 ---
    print("[5/5] Saving results ...")

    # JSON
    json_path = run_root / "attribution.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model_path": MODEL_PATH,
                "data_dir": DATA_DIR,
                "task": TASK_INSTRUCTION,
                "goal_image": GOAL_IMAGE,
                "frame_interval": FRAME_INTERVAL,
                "eval_mode": EVAL_MODE,
                "grid_size": GRID_SIZE,
                "target_view_index": TARGET_VIEW_INDEX,
                "mask_value": MASK_VALUE,
            },
            "records": all_records,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Attribution JSON: {json_path}")

    # 视频汇总
    video_path = run_root / "attention_video.mp4"
    if video_frames:
        if _MOVIEPY_AVAILABLE:
            clip = ImageSequenceClip(video_frames, fps=VIDEO_FPS)
            clip.write_videofile(str(video_path), logger=None)
        else:
            # 退回 cv2.VideoWriter（要求 RGB->BGR）
            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (w, h))
            for frm in video_frames:
                writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
            writer.release()
        print(f"  Attention video:  {video_path}")

    # 进度曲线（额外彩蛋）
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = [r["step"] for r in all_records]
        scores = [r["orig_score"] for r in all_records]
        impacts = [r["max_impact"] for r in all_records]

        ax2 = ax.twinx()
        ax.plot(steps, scores, "o-", color="#2196F3", label="Score (%)")
        ax2.plot(steps, impacts, "s-", color="#FF5722", label="Max |Δscore|")

        ax.set_xlabel("Step")
        ax.set_ylabel("Original Score (%)", color="#2196F3")
        ax2.set_ylabel("Max Attribution Impact", color="#FF5722")
        ax.set_title(f"Score vs Attribution Impact Over Time\nTask: {TASK_INSTRUCTION}")
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.95))
        fig.tight_layout()
        curve_path = run_root / "score_impact_curve.png"
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        print(f"  Score/impact curve: {curve_path}")
    except Exception as e:
        print(f"  [WARN] failed to plot curve: {e}")

    print(f"\n{'='*70}")
    print(f"ALL DONE. Results saved under: {run_root}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
