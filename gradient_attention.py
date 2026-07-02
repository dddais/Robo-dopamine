"""
梯度归因（Gradient-based Attribution）分析脚本
==============================================

目标：
    用 Integrated Gradients 方法，严谨地分析 GRM-2.0-8B（Qwen3VL 架构）
    在输出 task progress score 时，主要关注输入图像的哪些 patch。

原理：
    把 score 视为标量函数 f(image)，对图像 patch embedding 求梯度。
    ∂score / ∂patch_embedding 大 = 这个 patch 对 score 影响大。
    用 Integrated Gradients（沿路径积分）得到更稳健的归因。

与扰动法的对比：
    - 扰动法：黑盒，mask 每个区域看 score 变化（N 次推理，最直观）
    - 梯度法：白盒，反向传播求导（1 次前向 + 1 次反向，最严谨的因果归因）

注意：
    本脚本不能与 vLLM 同时用（vLLM 是纯推理引擎，不支持 backward）。
    改用 HuggingFace transformers + PyTorch autograd。

用法：
    python gradient_attention.py

输出：
    1. 每帧的 attribution heatmap（PNG）
    2. 汇总视频 gradient_attention_video.mp4
    3. attribution.json（数值结果）
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

# 视频输出：优先 moviepy，缺失时退回 cv2
_MOVIEPY_AVAILABLE = False
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    _MOVIEPY_AVAILABLE = True
except ImportError:
    pass

import torch

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ============================================================
# 配置区（按需修改）
# ============================================================

MODEL_PATH = './pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview'

# 数据目录（包含 cam_high.mp4 / cam_left_wrist.mp4 / cam_right_wrist.mp4）
DATA_DIR = "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot"

TASK_INSTRUCTION = "pick the carrot and put it on the plate"
# GOAL_IMAGE = "./examples/xzx_ep1_sub2.png"
GOAL_IMAGE = "./examples/blank_goal.png"

FRAME_INTERVAL = 30
EVAL_MODE = "forward"        # forward / incremental / backward

# 梯度归因参数
TARGET_VIEW_INDEX = 5        # 归因哪张图（5 = AFTER High）
INTEGRATION_STEPS = 16       # IG 积分步数（越大越准，但越慢）
SCORE_NORMALIZATION = "absolute"   # 'absolute'（|grad|）或 'signed'（带符号）

# 归因目标（改进点 1）
#   'expected_score': 对"期望 score 数值"求梯度（可微的连续值，最对应任务语义）
#   'logit':          对 score 数字 token 的 logit 求梯度（原行为）
ATTRIBUTION_TARGET = "expected_score"

# SmoothGrad 降噪（改进点 2）
#   在每个 IG 步内部，对输入加多次高斯噪声，多次求梯度后平均
#   SMOOTHGRAD_N=1 等价于不加噪（原行为）
SMOOTHGRAD_N = 8             # SmoothGrad 采样次数
SMOOTHGRAD_NOISE_STD = 0.05  # 噪声标准差（相对输入分布的尺度）

# Baseline 选择（改进点 3）
#   'zero':  全零基线（原行为，对应黑色图）
#   'mean':  训练集均值（这里用 processor 归一化后的均值 0）
#   'blur':  高斯模糊版原图（更接近自然图像分布）
BASELINE_MODE = "zero"

# Qwen3VL 图像处理参数（与 examples/inference.py 一致）
MIN_PIXELS = 12544
MAX_PIXELS = 76800

# 输出根目录
OUTPUT_ROOT = "./results/gradient_attribution_new_2"
VIDEO_FPS = 2.0

# 系统提示（与 examples/inference.py 完全一致，保证 score 行为一致）
SYSTEM_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Your job is to judge whether the AFTER image set moves closer to the task objective than the BEFORE image set, using the provided reference examples only as anchors.

<Task>
`{task}`

REFERENCE EXAMPLES (for visual anchoring only; not necessarily this run's actual START/END):
- REFERENCE START — Robot Front Image (task just starting): <image>
- REFERENCE END — Robot Front Image (task fully completed): <image>
</Task>

BEFORE Robot Front Image: <image>
BEFORE Robot Left Wrist Image: <image>
BEFORE Robot Right Wrist Image: <image>

AFTER Robot Front Image: <image>
AFTER Robot Left Wrist Image: <image>
AFTER Robot Right Wrist Image: <image>

Goal
Compare the BEFORE and AFTER three-view sets and judge whether AFTER moves closer to accomplishing the task than BEFORE, using the REFERENCE START/END images as conceptual anchors.

Progress Estimation (no formulas)
1) Calibrate using the references:
   - REFERENCE START = "just beginning"; REFERENCE END = "fully completed."
   - Visually estimate how far BEFORE and AFTER are along this START->END continuum.
2) Direction:
   - AFTER better than BEFORE -> positive score.
   - AFTER worse than BEFORE -> negative score.
   - Essentially the same -> 0.
3) Normalize to an integer percentage in [-100%, +100%]:
   - For improvements, scale the improvement relative to what remained from BEFORE to END.
   - For regressions, scale the deterioration relative to how far BEFORE had progressed from START.
   - Clip to [-100%, +100%] and round to the nearest integer percent.

Evaluation Criteria (apply across all three views)
1) Task Alignment: Evidence directly tied to `{task}`.
2) Completeness & Accuracy: Correct pose, contact, placement, orientation, grasp quality, absence of collisions, stability, etc.
3) View-Specific Evidence & Consistency:
   - Use the **Front** view for global layout, object pose, approach path, end-state geometry, and scene-level constraints.
   - Use the **Left/Right Wrist** views to inspect **fine-grained gripper state** (finger closure, contact location/area, slippage, wedge/misalignment, object deformation, cable/wire/cloth entanglement, unintended contact, occluded collisions).
   - When views disagree, prioritize the view that provides **decisive cues** for the criterion at hand. In particular, wrist views often **override** for grasp/contact validity and safety.
   - If any single view shows a failure that invalidates success (e.g., mis-grasp, collision, unsafe/unstable pose), let that override when judging progress.
4) Ignore Irrelevant Factors: Lighting, color shifts, background clutter, or UI/watermarks that don't affect task success.
5) Ambiguity: If evidence is genuinely inconclusive or conflicting without decisive cues, treat progress as unchanged -> 0%.

Output Format (STRICT)
Return ONLY one line containing the score wrapped in <score> tags, as an integer percentage with a percent sign:
<score>+NN%</score>  or  <score>-NN%</score>  or  <score>0%</score>
"""


# ============================================================
# 1. 帧提取（与 perturbation_attention.py 一致）
# ============================================================

def get_frame_count(path: Path) -> Tuple[str, int]:
    if path.is_dir():
        files = sorted([p for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() == ".png"])
        if not files:
            raise RuntimeError(f"No PNG frames in directory: {path}")
        return "dir", len(files)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Invalid frame count: {path}")
    return "video", n


def make_sample_indices(num_frames: int, interval: int) -> List[int]:
    if num_frames < 1:
        return []
    indices = list(range(0, num_frames, interval))
    last_idx = num_frames - 1
    if not indices or indices[-1] != last_idx:
        indices.append(last_idx)
    return indices


def save_frames(src_path: Path, out_dir: Path, indices: List[int], src_type: str) -> None:
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
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"),
                        frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cap.release()
    else:
        files = sorted([p for p in src_path.iterdir()
                        if p.is_file() and p.suffix.lower() == ".png"])
        n = len(files)
        for idx in indices:
            if 0 <= idx < n:
                shutil.copyfile(files[idx], out_dir / f"frame_{idx:06d}.png")


def build_samples(cache_root: Path, task: str, indices: List[int],
                  ref_end_path: str, mode: str = "forward") -> List[Dict]:
    items = []
    if len(indices) < 2:
        return items
    for k in range(len(indices) - 1):
        af = indices[k + 1]
        if mode == "incremental":
            bf = indices[k]
            bf_images = [
                str(cache_root / "cam_high"        / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist"  / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "forward":
            bf = indices[0]
            bf_images = [
                str(cache_root / "cam_high"        / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_left_wrist"  / f"frame_{bf:06d}.png"),
                str(cache_root / "cam_right_wrist" / f"frame_{bf:06d}.png"),
            ]
        elif mode == "backward":
            bf_images = [ref_end_path, ref_end_path, ref_end_path]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        af_images = [
            str(cache_root / "cam_high"        / f"frame_{af:06d}.png"),
            str(cache_root / "cam_left_wrist"  / f"frame_{af:06d}.png"),
            str(cache_root / "cam_right_wrist" / f"frame_{af:06d}.png"),
        ]
        items.append({
            "id": f"step-{k:04d}",
            "task": task,
            "image": [
                str(cache_root / "cam_high" / f"frame_{0:06d}.png"),
                ref_end_path,
                bf_images[0], bf_images[1], bf_images[2],
                af_images[0], af_images[1], af_images[2],
            ],
        })
    return items


def parse_score(pred_text: str) -> float:
    try:
        m = re.search(r'<score>\s*([+-]?\d+\.?\d*)\s*%?\s*</score>', pred_text)
        if m:
            return float(m.group(1))
        m = re.search(r'([+-]?\d+\.?\d*)%', pred_text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 0.0


# ============================================================
# 2. 构建对话消息（与 inference.py 一致）
# ============================================================

def build_messages(task: str) -> List[Dict]:
    """构建 8 图对话消息，placeholder 与 SYSTEM_PROMPT 里的 <image> 一一对应。"""
    prompt_parts = SYSTEM_PROMPT.format(task=task).split("<image>")
    content = [{"type": "text", "text": prompt_parts[0]}]
    for i in range(8):
        content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_parts[i + 1]})
    return [{"role": "user", "content": content}]


# ============================================================
# 3. 模型封装：Qwen3VL + 梯度支持
# ============================================================

class GradientAttributor:
    """
    封装 Qwen3VL，支持：
    1. 前向推理拿 score
    2. 对目标图像 patch embedding 求梯度归因
    """

    def __init__(self, model_path: str, min_pixels: int = 12544, max_pixels: int = 76800):
        print(f"Loading model from {model_path} ...")
        # 关键：attn_implementation='eager' 才能做反向传播（sdpa 也可以，flash 不行）
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",   # sdpa 支持反向，比 eager 快
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        # 缓存 tokenizer
        self.tok = self.processor.tokenizer

        # 找一些关键 token id
        self.score_open_ids = self.tok.encode("<score>", add_special_tokens=False)
        self.score_close_ids = self.tok.encode("</score>", add_special_tokens=False)
        print(f"  <score> tokens: {self.score_open_ids} -> {[self.tok.decode([i]) for i in self.score_open_ids]}")
        print(f"  Model loaded.\n")

    @torch.no_grad()
    def generate_score(self, image_paths: List[str], task: str,
                       max_new_tokens: int = 64) -> Tuple[str, float]:
        """普通推理，拿 score 文本。"""
        messages = build_messages(task)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[prompt_text], images=images,
            return_tensors="pt", padding=True,
        ).to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        return text, parse_score(text)

    def _build_inputs_with_grad_image(
        self,
        image_paths: List[str],
        task: str,
        target_view_idx: int,
        baseline_value: float = 0.0,
    ) -> Tuple[Dict, Tuple[int, int]]:
        """
        构造前向输入，把目标图像的 pixel_values 设为 requires_grad 的叶子节点。
        返回:
            inputs: 含 input_ids / attention_mask / pixel_values / image_grid_thw
            (start, end): 目标图像在合并后的 pixel_values 中的 patch 起止行
        """
        messages = build_messages(task)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[prompt_text], images=images,
            return_tensors="pt", padding=True,
        )

        # 计算每张图的 patch 数（grid_t * grid_h * grid_w / merge^2）
        # image_grid_thw 形状: [num_images, 3]，每行 (t, h, w)
        # 实际 patch 数 = t * (h/2) * (w/2)  (spatial_merge_size=2)
        grid_thw = inputs["image_grid_thw"]   # [N, 3]
        merge = self.processor.image_processor.merge_size \
            if hasattr(self.processor.image_processor, "merge_size") else 2

        # 每张图的 patch 数
        patch_counts = []
        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].tolist()
            n = t * (h // merge) * (w // merge)
            patch_counts.append(n)

        target_start = sum(patch_counts[:target_view_idx])
        target_end = target_start + patch_counts[target_view_idx]

        # 把 pixel_values 整体搬到模型设备，并要求目标图段梯度
        pixel_values = inputs["pixel_values"].to(self.model.device, dtype=self.model.dtype)
        if baseline_value != 0.0:
            pixel_values[target_start:target_end] = baseline_value

        pixel_values.requires_grad_(False)
        # 关键：只让目标图段可求导
        target_segment = pixel_values[target_start:target_end].clone().detach()
        target_segment = target_segment + torch.zeros_like(target_segment)  # 脱离原计算图
        target_segment.requires_grad_(True)

        # 重组 pixel_values：把目标段替换为可求导版本
        # 用 torch.cat 保证整段是一个连续 tensor，并且能反传到 target_segment
        parts = [
            pixel_values[:target_start].detach(),
            target_segment,
            pixel_values[target_end:].detach(),
        ]
        pixel_values_grad = torch.cat(parts, dim=0)

        inputs_ready = {
            "input_ids": inputs["input_ids"].to(self.model.device),
            "attention_mask": inputs["attention_mask"].to(self.model.device),
            "pixel_values": pixel_values_grad,
            "image_grid_thw": inputs["image_grid_thw"].to(self.model.device),
        }
        # position_ids 如果 processor 给了也带上
        if "position_ids" in inputs:
            inputs_ready["position_ids"] = inputs["position_ids"].to(self.model.device)

        return inputs_ready, (target_start, target_end)

    def _find_score_token_position(self, input_ids: torch.Tensor) -> int:
        """
        在 assistant 生成的第一段 token 中找 score 数字 token 的位置。
        策略：找 prompt 后第一个 token 是 '<' (score_open_ids[0]) 的位置，
        然后向后查找到数字 token。这里我们直接在 prompt 长度后取若干 token。

        但为了做梯度归因，我们不走 generate（generate 不可导），
        而是先用 generate 得到 score 文本，再 teacher-force 做一次前向，
        对最后那个数字 token 的 logit 求梯度。
        """
        # 实际策略见 compute_attribution
        return -1

    @torch.no_grad()
    def _get_target_score_tokens(self, image_paths: List[str], task: str) -> List[int]:
        """
        第一步：用 generate 拿到 score 文本，返回完整的 token 序列（含 prompt + 生成）。
        """
        text, score = self.generate_score(image_paths, task)
        return text, score

    def _compute_digit_token_ids(self) -> List[int]:
        """预计算数字 token '0'-'9' 的 id（可能包含 '.','+' 等也一并返回）。"""
        ids = []
        for d in "0123456789":
            toks = self.tok.encode(d, add_special_tokens=False)
            ids.extend(toks)
        # 去重保序
        seen = set()
        out = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def _compute_expected_score_scalar(
        self,
        logits: torch.Tensor,
        response_ids: List[int],
        target_token_positions: List[int],
        prompt_len: int,
    ) -> torch.Tensor:
        """
        改进点 1：把"离散 score token"转成"连续期望 score 值"。

        对每个数字 token 位置，用 logit 加权计算该位置的期望数字。
        整段拼接解析后得到期望的 score 数值（可微的标量）。

        例：score = "<score>+3.4%</score>"
            数字 token 序列 = ['3', '.', '4']
            位置 0 ('3'): 期望数字 = sum_d(d * softmax(logit)[d])
            位置 1 ('.'): 小数点，不贡献值
            位置 2 ('4'): 期望数字
            合成：expected_score = 3.x + 0.4 = 期望十位 + 期望十分位
        """
        digit_token_ids = self._compute_digit_token_ids()
        # 建一个 (digit_value -> token_id) 的查询：注意一个数字可能有多个 token
        # Qwen tokenizer 里 '0'-'9' 都是单 token，所以这里简化处理
        digit_id_to_value = {}
        for d_str in "0123456789":
            tid = self.tok.encode(d_str, add_special_tokens=False)
            if len(tid) == 1:
                digit_id_to_value[tid[0]] = int(d_str)

        if not digit_id_to_value:
            # 兜底：退回 logit 求和
            target_logits = []
            for pos_in_response in target_token_positions:
                abs_pos = prompt_len + pos_in_response
                token_id = response_ids[pos_in_response]
                target_logits.append(logits[0, abs_pos - 1, token_id])
            return torch.stack(target_logits).sum()

        # 把"位置"按是否小数点分段
        # 简化：第一个数字 token 作为整数位，后面的作为小数位
        # 更稳健：直接按字符串构造，但我们要可微
        # 这里采用：每个数字 token 位置贡献 d × 10^(weight)
        # weight 按位置在数字串里的索引决定（第一位=整数位=10^0，第二位=小数第一位=10^-1...）

        # 先识别哪些是真正的数字 token（在 digit_id_to_value 里）
        numeric_positions = []
        for p in target_token_positions:
            tid = response_ids[p]
            if tid in digit_id_to_value:
                numeric_positions.append(p)

        if not numeric_positions:
            target_logits = []
            for pos_in_response in target_token_positions:
                abs_pos = prompt_len + pos_in_response
                token_id = response_ids[pos_in_response]
                target_logits.append(logits[0, abs_pos - 1, token_id])
            return torch.stack(target_logits).sum()

        # 构造期望 score：第 k 个数字位贡献 expected_digit_k × 10^(-k)
        # （假设第一位是整数位，后面是小数位；这与 GRM 输出 "+3.4%" 等格式一致）
        expected = torch.zeros((), device=logits.device, dtype=logits.dtype)
        for k, pos_in_response in enumerate(numeric_positions):
            abs_pos = prompt_len + pos_in_response
            pos_logits = logits[0, abs_pos - 1]   # [V]
            # 只看数字 token 的 logit，做局部 softmax
            digit_ids = list(digit_id_to_value.keys())
            digit_logits = pos_logits[digit_ids]    # [num_digits]
            digit_probs = torch.softmax(digit_logits, dim=-1)
            # 期望数字
            values = torch.tensor(
                [digit_id_to_value[i] for i in digit_ids],
                device=logits.device, dtype=logits.dtype,
            )
            expected_digit = (digit_probs * values).sum()
            # 第一位（k=0）是整数位，后面是小数位
            weight = 0 if k == 0 else -(k)
            expected = expected + expected_digit * (10.0 ** weight)

        return expected

    def compute_attribution(
        self,
        image_paths: List[str],
        task: str,
        target_view_idx: int = 5,
        integration_steps: int = 16,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Integrated Gradients + SmoothGrad 归因。

        改进点 1（ATTRIBUTION_TARGET='expected_score'）:
            归因目标从"score 数字 token logit"改为"期望 score 数值"，
            这是一个可微的连续标量，更对应任务语义。

        改进点 2（SMOOTHGRAD_N>1）:
            在每个 IG 插值点，对输入加 N 次高斯噪声，求 N 次梯度后平均，
            显著降低梯度噪声。

        改进点 3（BASELINE_MODE）:
            baseline 可选 zero / mean / blur，更贴近自然图像分布。

        返回:
            attribution: [num_target_patches] 的 |grad| L2 范数
            score: 原始 score
            (target_start, target_end): 目标图 patch 范围
        """
        # ---- Step 1: 生成 score 文本，得到 target token id 序列 ----
        score_text, score = self._get_target_score_tokens(image_paths, task)

        full_response_text = score_text
        response_ids = self.tok.encode(full_response_text, add_special_tokens=False)
        response_tensor = torch.tensor([response_ids], dtype=torch.long, device=self.model.device)

        # 找 score 数字 token 在 response 中的位置
        def find_token_span_by_decode(token_ids_list, sub_str):
            text_so_far = ""
            char_offset_start = -1
            for i in range(len(token_ids_list) + 1):
                if char_offset_start < 0:
                    if i == len(token_ids_list):
                        break
                    new_text = self.tok.decode(token_ids_list[:i+1])
                    if sub_str in new_text:
                        char_offset_start = new_text.find(sub_str)
                        start_token = i
                        char_offset_end = char_offset_start + len(sub_str)
                        for j in range(i + 1, len(token_ids_list) + 1):
                            full = self.tok.decode(token_ids_list[:j])
                            if len(full) >= char_offset_end:
                                return start_token, j
                        return start_token, len(token_ids_list)
            return -1, -1

        open_start, open_end = find_token_span_by_decode(response_ids, "<score>")
        close_start, close_end = find_token_span_by_decode(response_ids, "</score>")

        if open_start < 0 or close_start < 0 or close_start <= open_end:
            print(f"[WARN] cannot locate <score>...</score> in response: {score_text!r}")
            target_token_positions = list(range(len(response_ids)))
        else:
            inner_start = open_end
            inner_end = close_start
            target_token_positions = list(range(inner_start, inner_end))
            target_token_positions = [
                p for p in target_token_positions
                if all(c.isdigit() or c == '.' for c in self.tok.decode([response_ids[p]]).strip())
            ]
            if not target_token_positions:
                target_token_positions = list(range(inner_start, inner_end))

        print(f"    Score text: {score_text!r}")
        print(f"    Target token positions (in response): {target_token_positions}")
        print(f"    Target tokens: {[self.tok.decode([response_ids[p]]) for p in target_token_positions]}")
        print(f"    Attribution target: {ATTRIBUTION_TARGET}")
        print(f"    SmoothGrad: N={SMOOTHGRAD_N}, noise_std={SMOOTHGRAD_NOISE_STD}")
        print(f"    Baseline: {BASELINE_MODE}")

        # ---- Step 2: 准备 baseline 和 input ----
        inputs_orig, (t_start, t_end) = self._build_inputs_with_grad_image(
            image_paths, task, target_view_idx, baseline_value=0.0,
        )
        target_input_segment = inputs_orig["pixel_values"][t_start:t_end].detach()

        # 改进点 3：选择 baseline
        if BASELINE_MODE == "zero":
            target_input_baseline = torch.zeros_like(target_input_segment)
        elif BASELINE_MODE == "mean":
            # 用目标图的均值作为基线（对应灰度图）
            target_input_baseline = torch.ones_like(target_input_segment) * target_input_segment.mean()
        elif BASELINE_MODE == "blur":
            # 模糊基线：对 patch 维度做平均池化近似
            # pixel_values 形状 [num_patches, C]，无法做空间模糊，
            # 退化为均值基线
            target_input_baseline = torch.ones_like(target_input_segment) * target_input_segment.mean()
        else:
            target_input_baseline = torch.zeros_like(target_input_segment)

        accum_grads = torch.zeros_like(target_input_segment, dtype=torch.float32)

        prompt_len = inputs_orig["input_ids"].shape[1]
        full_input_ids = torch.cat(
            [inputs_orig["input_ids"], response_tensor], dim=1
        ).to(self.model.device)
        full_attention_mask = torch.ones_like(full_input_ids)

        alphas = [i / integration_steps for i in range(1, integration_steps + 1)]

        # 噪声尺度：相对输入分布的标准差
        input_std = float(target_input_segment.std().item()) + 1e-6
        noise_scale = SMOOTHGRAD_NOISE_STD * input_std

        # ---- Step 3: IG + SmoothGrad 主循环 ----
        for step_idx, alpha in enumerate(alphas):
            interpolated_segment_base = target_input_baseline + alpha * (
                target_input_segment - target_input_baseline
            )

            # 改进点 2：SmoothGrad —— 在这个插值点附近采样 N 次
            step_grads = torch.zeros_like(target_input_segment, dtype=torch.float32)
            for sg_idx in range(SMOOTHGRAD_N):
                if SMOOTHGRAD_N > 1 and noise_scale > 0:
                    noise = torch.randn_like(interpolated_segment_base) * noise_scale
                    interpolated_segment = interpolated_segment_base + noise
                else:
                    interpolated_segment = interpolated_segment_base

                interpolated_segment = interpolated_segment.to(
                    self.model.device, dtype=self.model.dtype
                )
                interpolated_segment.requires_grad_(True)

                parts = [
                    inputs_orig["pixel_values"][:t_start].detach(),
                    interpolated_segment,
                    inputs_orig["pixel_values"][t_end:].detach(),
                ]
                pixel_values_interp = torch.cat(parts, dim=0)

                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=pixel_values_interp,
                    image_grid_thw=inputs_orig["image_grid_thw"],
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits = outputs.logits

                # 改进点 1：选择归因目标
                if ATTRIBUTION_TARGET == "expected_score":
                    target_scalar = self._compute_expected_score_scalar(
                        logits=logits,
                        response_ids=response_ids,
                        target_token_positions=target_token_positions,
                        prompt_len=prompt_len,
                    )
                else:  # 'logit'
                    target_logits = []
                    for pos_in_response in target_token_positions:
                        abs_pos = prompt_len + pos_in_response
                        token_id = response_ids[pos_in_response]
                        target_logits.append(logits[0, abs_pos - 1, token_id])
                    target_scalar = torch.stack(target_logits).sum()

                self.model.zero_grad()
                target_scalar.backward()

                grad = interpolated_segment.grad.detach().float()
                step_grads += grad / SMOOTHGRAD_N

                del outputs, logits, target_scalar, grad
                torch.cuda.empty_cache()

            accum_grads += step_grads / integration_steps

            if (step_idx + 1) % max(1, integration_steps // 4) == 0:
                print(f"    IG step {step_idx+1}/{integration_steps} done")

        # ---- Step 3: 计算 IG 归因 ----
        # IG = (input - baseline) * mean_grad
        ig_attribution = (target_input_segment - target_input_baseline).float() * accum_grads
        # 每个 patch 的归因强度 = L2 范数（把 hidden_dim 聚合）
        # ig_attribution: [num_patches, hidden_dim]
        patch_norms = ig_attribution.norm(dim=-1).cpu().numpy()   # [num_patches]

        if SCORE_NORMALIZATION == "absolute":
            patch_attribution = np.abs(patch_norms)
        else:
            patch_attribution = patch_norms

        return patch_attribution, score, (t_start, t_end)


# ============================================================
# 4. 把 patch 归因重排成 2D heatmap
# ============================================================

def patches_to_2d(
    patch_attribution: np.ndarray,
    grid_thw_target: List[int],
    merge_size: int = 2,
) -> np.ndarray:
    """
    把 [num_patches] 的一维归因重排成 2D 网格。
    grid_thw_target: 目标图的 (t, h, w)
    """
    t, h, w = grid_thw_target
    hm = h // merge_size
    wm = w // merge_size
    # 简化：只取 t=1 的空间分布
    expected = t * hm * wm
    if patch_attribution.shape[0] != expected:
        # 截断或填充
        if patch_attribution.shape[0] > expected:
            patch_attribution = patch_attribution[:expected]
        else:
            pad = expected - patch_attribution.shape[0]
            patch_attribution = np.pad(patch_attribution, (0, pad))
    # 取第一帧（t 维度平均）
    try:
        reshaped = patch_attribution.reshape(t, hm, wm).mean(axis=0)   # [hm, wm]
    except Exception:
        reshaped = patch_attribution[:hm*wm].reshape(hm, wm)
    return reshaped


# ============================================================
# 5. 可视化
# ============================================================

def render_attribution_frame(
    image_path: str,
    patch_attribution: np.ndarray,
    grid_thw_target: List[int],
    orig_score: float,
    step_idx: int,
    frame_idx: int,
    task: str,
    out_path: str,
    panel_w: int = 384,
    panel_h: int = 288,
) -> np.ndarray:
    # 1. 原图
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (panel_w, panel_h))

    # 2. patch → 2D
    heatmap_2d = patches_to_2d(patch_attribution, grid_thw_target)
    max_val = heatmap_2d.max()
    if max_val > 1e-8:
        heatmap_norm = heatmap_2d / max_val
    else:
        heatmap_norm = heatmap_2d

    # 3. 上采样到原图大小
    heatmap_resized = cv2.resize(heatmap_norm, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
    canvas = np.hstack([img, heatmap_color, overlay])

    # 4. 信息条
    info_h = 50
    info_bar = np.ones((info_h, canvas.shape[1], 3), dtype=np.uint8) * 30
    info_txt = (f"Step {step_idx}  Frame {frame_idx}  |  "
                f"Score: {orig_score:+.1f}%  |  Max |IG|: {max_val:.4f}  |  "
                f"Grid: {heatmap_2d.shape}")
    cv2.putText(info_bar, info_txt, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    task_h = 30
    task_bar = np.ones((task_h, canvas.shape[1], 3), dtype=np.uint8) * 30
    cv2.putText(task_bar, f"Task: {task}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    title_h = 25
    title_bar = np.ones((title_h, canvas.shape[1], 3), dtype=np.uint8) * 60
    for i, txt in enumerate(["Original", "IG Attribution", "Overlay"]):
        cv2.putText(title_bar, txt,
                    (i * panel_w + panel_w // 2 - 50, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    final = np.vstack([info_bar, title_bar, canvas, task_bar])

    if out_path:
        fig = plt.figure(figsize=(12, 4))
        plt.imshow(final)
        plt.axis('off')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    return final


# ============================================================
# 6. 主流程
# ============================================================

def main():
    print(f"\n{'='*70}")
    print("Gradient-based Attribution (Integrated Gradients)")
    print(f"{'='*70}")
    print(f"Model:           {MODEL_PATH}")
    print(f"Data:            {DATA_DIR}")
    print(f"Task:            {TASK_INSTRUCTION}")
    print(f"Eval mode:       {EVAL_MODE}")
    print(f"Target view:     image[{TARGET_VIEW_INDEX}] (AFTER High)")
    print(f"IG steps:        {INTEGRATION_STEPS}")
    print(f"{'='*70}\n")

    # --- 输出目录 ---
    ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run_root = Path(OUTPUT_ROOT) / f"{ts}_{EVAL_MODE}_mode"
    cache_root = run_root / ".cache"
    heatmaps_dir = run_root / "heatmaps"
    run_root.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(exist_ok=True)

    # --- 1. 帧提取 ---
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

    if GOAL_IMAGE and os.path.exists(GOAL_IMAGE):
        ref_end_path = str(cache_root / "ref_end.png")
        shutil.copy(GOAL_IMAGE, ref_end_path)
    else:
        ref_end_path = str(cam_dirs["cam_high"] / f"frame_{total_frames-1:06d}.png")

    # --- 2. 构建 samples ---
    print("\n[2/5] Building samples ...")
    samples = build_samples(cache_root, TASK_INSTRUCTION, indices, ref_end_path, mode=EVAL_MODE)
    print(f"  Samples: {len(samples)}\n")

    # --- 3. 加载模型 ---
    print("[3/5] Loading model (transformers + autograd) ...")
    attributor = GradientAttributor(MODEL_PATH, MIN_PIXELS, MAX_PIXELS)

    # --- 4. 逐帧梯度归因 ---
    print("\n[4/5] Running gradient attribution (Integrated Gradients) ...")
    video_frames = []
    all_records = []
    total_steps = len(samples)

    for step_idx, item in enumerate(samples):
        t0 = time.time()
        frame_idx = indices[step_idx + 1]
        target_image_path = item["image"][TARGET_VIEW_INDEX]

        print(f"\n  [Step {step_idx+1}/{total_steps}] frame_idx={frame_idx}")
        print(f"    Running IG with {INTEGRATION_STEPS} steps (1 forward+backward each) ...")

        try:
            patch_attribution, orig_score, (t_start, t_end) = attributor.compute_attribution(
                image_paths=item["image"],
                task=item["task"],
                target_view_idx=TARGET_VIEW_INDEX,
                integration_steps=INTEGRATION_STEPS,
            )
        except Exception as e:
            print(f"    [ERROR] attribution failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        max_attr = float(patch_attribution.max())
        print(f"    Score: {orig_score:+.1f}%  |  Max |IG|: {max_attr:.4f}  |  Time: {elapsed:.1f}s")

        # 拿目标图的 grid_thw
        images_pil = [Image.open(p).convert("RGB") for p in item["image"]]
        messages = build_messages(item["task"])
        prompt_text = attributor.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_inputs = attributor.processor(
            text=[prompt_text], images=images_pil, return_tensors="pt", padding=True
        )
        grid_thw_target = proc_inputs["image_grid_thw"][TARGET_VIEW_INDEX].tolist()

        # 渲染
        heatmap_png = str(heatmaps_dir / f"heatmap_step{step_idx:04d}_frame{frame_idx:06d}.png")
        frame_rgb = render_attribution_frame(
            image_path=target_image_path,
            patch_attribution=patch_attribution,
            grid_thw_target=grid_thw_target,
            orig_score=orig_score,
            step_idx=step_idx,
            frame_idx=frame_idx,
            task=TASK_INSTRUCTION,
            out_path=heatmap_png,
        )
        video_frames.append(frame_rgb)

        all_records.append({
            "step": step_idx,
            "frame_idx": frame_idx,
            "orig_score": orig_score,
            "max_ig": max_attr,
            "mean_ig": float(patch_attribution.mean()),
            "patch_attribution": patch_attribution.tolist(),
            "grid_thw_target": grid_thw_target,
        })

    print(f"\n  Attribution done. {len(video_frames)} frames rendered.\n")

    # --- 5. 保存 ---
    print("[5/5] Saving results ...")

    json_path = run_root / "attribution.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "method": "Integrated Gradients + SmoothGrad",
                "model_path": MODEL_PATH,
                "data_dir": DATA_DIR,
                "task": TASK_INSTRUCTION,
                "goal_image": GOAL_IMAGE,
                "frame_interval": FRAME_INTERVAL,
                "eval_mode": EVAL_MODE,
                "target_view_index": TARGET_VIEW_INDEX,
                "integration_steps": INTEGRATION_STEPS,
                "score_normalization": SCORE_NORMALIZATION,
                "attribution_target": ATTRIBUTION_TARGET,
                "smoothgrad_n": SMOOTHGRAD_N,
                "smoothgrad_noise_std": SMOOTHGRAD_NOISE_STD,
                "baseline_mode": BASELINE_MODE,
            },
            "records": all_records,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Attribution JSON: {json_path}")

    video_path = run_root / "gradient_attention_video.mp4"
    if video_frames:
        if _MOVIEPY_AVAILABLE:
            clip = ImageSequenceClip(video_frames, fps=VIDEO_FPS)
            clip.write_videofile(str(video_path), logger=None)
        else:
            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (w, h))
            for frm in video_frames:
                writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
            writer.release()
        print(f"  Attention video:  {video_path}")

    # 曲线
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = [r["step"] for r in all_records]
        scores = [r["orig_score"] for r in all_records]
        impacts = [r["max_ig"] for r in all_records]
        ax2 = ax.twinx()
        ax.plot(steps, scores, "o-", color="#2196F3", label="Score (%)")
        ax2.plot(steps, impacts, "s-", color="#FF5722", label="Max |IG|")
        ax.set_xlabel("Step")
        ax.set_ylabel("Original Score (%)", color="#2196F3")
        ax2.set_ylabel("Max IG Attribution", color="#FF5722")
        ax.set_title(f"Score vs IG Attribution Over Time\nTask: {TASK_INSTRUCTION}")
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.95))
        fig.tight_layout()
        curve_path = run_root / "score_ig_curve.png"
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        print(f"  Score/IG curve:   {curve_path}")
    except Exception as e:
        print(f"  [WARN] failed to plot curve: {e}")

    print(f"\n{'='*70}")
    print(f"ALL DONE. Results saved under: {run_root}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
