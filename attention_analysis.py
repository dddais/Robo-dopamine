"""
Attention-based attribution for GRM (Qwen3VL).

思路来源：
  - Kang et al., "Your Large Vision-Language Model Only Needs A Few Attention
    Heads For Visual Grounding" (arXiv:2503.06287)
  - Gandikota & Bau, "Gaze Heads: How VLMs Look at What They Describe"
    (arXiv:2606.14703)

核心做法：
  对每次 score 评估，提取 LLM 内部"最后一个 prompt token"对所有 image
  token 的 post-softmax attention 权重。这些权重本身就是一张空间热力图，
  反映模型在准备生成 score 时关注图像的哪些区域。

  与 perturbation / gradient 方法相比，attention 方法：
    - 不需要反向传播，单次前向即可
    - 没有 OOD 问题（输入始终是真实图像）
    - 没有梯度饱和问题
  注意：attention 权重是"模型在看哪里"的相关性信号，不等于"如果遮住这块
  score 会变多少"的因果重要性。

输出：
  - 每个 sample 的热力图 PNG（原图 / attention 热力图 / 叠加）
  - 全 head × 全层的 attention 统计（attention_sum / 空间熵）的 npz
  - 汇总视频，追踪 attention 随任务推进的变化
"""

import os
import sys
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# moviepy 可选（缺失时退回 cv2.VideoWriter）
_MOVIEPY_AVAILABLE = False
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    _MOVIEPY_AVAILABLE = True
except ImportError:
    pass

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# 复用 gradient_attention.py 的公共逻辑（帧提取 / 消息构建 / score 解析 / 可视化）
from gradient_attention import (
    SYSTEM_PROMPT,
    get_frame_count,
    make_sample_indices,
    save_frames,
    build_samples,
    parse_score,
    build_messages,
    patches_to_2d,
)


# ============================================================
# 配置区
# ============================================================

MODEL_PATH = './pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview'

DATA_DIR = "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot"

TASK_INSTRUCTION = "pick the white cube and put it on the plate"
GOAL_IMAGE = "./examples/blank_goal.png"

FRAME_INTERVAL = 20
EVAL_MODE = "forward"

# Attention 分析参数
TARGET_VIEW_INDEX = 5        # 归因哪张图（5 = AFTER High）
# 输出
OUTPUT_ROOT = "./results/attention_analysis_cube_score_sum"
VIDEO_FPS = 2.0
# 前向模式：
#   'generate'    : 正常自回归生成 score（逐 token，attention 取自生成过程）
#                   优点：attention 来自模型真实生成路径，最贴近实际推理
#                   缺点：慢（每生成一个 token 都要保存 attention）
#   'teacher_forcing' : 先 generate 拿 score 文本，再 teacher-force 一次前向
#                   优点：快（只一次前向）
#                   缺点：attention 来自"已知答案后重走一遍"，与真实生成略有差异
FORWARD_MODE = "generate"

# Query 选择：
#   'last_prompt' : 最后一个 prompt token（默认，对齐 2503.06287）
#   'score_digit' : <score> 标签里的数字 token（生成时正在看的）
#   注意：FORWARD_MODE='generate' 时，query 自动取生成过程中最后一个数字 token，
#         此参数只影响 teacher_forcing 模式
QUERY_MODE = "score_digit"

# Head 聚合策略：
#   'mean'   : 所有 head 平均（sanity check，论文里说通常无信息）
#   'topk'   : 按 head 评分选 top-k head 再平均
# 评分标准（HEAD_CRITERION）：
#   'attn_sum'      : attention sum 高（head 在看图） —— 论文标准 1
#   'attn_sum_low_entropy' : attention sum 高 且 空间熵低 —— 论文完整标准 1+2
#   'low_entropy'   : 仅空间熵低（attention 聚焦）
# 注：GRM 是全局 score 任务，不一定要求熵低；先用 attn_sum 试，效果不好再切换
HEAD_AGG = "topk"
HEAD_TOPK = 5
HEAD_CRITERION = "attn_sum"

# 论文细节：排除前 N 层（论文排除前 2 层，因为早期层行为不同）
# 这能解决你观察到的"靠前 layer head 的 attention 集中在左上角"问题
EXCLUDE_FIRST_LAYERS = 2

# 论文标准 1：attention sum 阈值 τ 的选取方式
#   'max_curvature' : 最大曲率法（论文默认，自动）
#   'percentile'    : 按分位数（LOW_ENTROPY_PERCENTILE 之上的算"高 sum"）
ATTN_SUM_THRESHOLD = "max_curvature"

# 论文标准 2：spatial entropy 的计算方式
#   'connected' : 论文原文——二值化 + 连通分量 entropy（推荐）
#   'shannon'   : 简单 Shannon entropy（之前的实现，对"聚焦"更敏感但不严格对应论文）
SPATIAL_ENTROPY_MODE = "connected"

# 是否为固定 top-K head 输出单独视频（对齐论文 2503.06287 的 selection frequency 思想）
# True  : 跑完所有 sample 后，按 selection frequency 选固定 HEAD_TOPK 个 head，每个输出完整视频
# False : 不输出 per-head 视频
OUTPUT_PER_HEAD_VIDEOS = True

# 固定 head 的选取方式：
#   'frequency' : 跨 sample 统计每个 head 被选中的频率，取 top-K（论文方法，推荐）
#                 每个 sample 独立选 top-K，统计谁出现得多
#   'mean_sum'  : 用所有 sample 平均的 attn_sum 排序，取 top-K
HEAD_SELECTION_MODE = "frequency"

# Head 筛选时"熵低"的阈值（仅 HEAD_CRITERION 含 low_entropy 且 ATTN_SUM_THRESHOLD='percentile' 时生效）
LOW_ENTROPY_PERCENTILE = 30

# 视频输出内容：
#   'selected' : 用 HEAD_AGG 筛出的 head 平均（top-k 或 mean）
#   'mean'     : 强制用全 head 平均（独立于 HEAD_AGG）
VIDEO_CONTENT = "selected"

# 是否保存全 head 的 attention 统计（npz，用于后续离线 head 分析）
SAVE_ALL_HEAD_STATS = False

# head_stats_overview 的统计方式：
#   'mean'  : 所有 sample 的 attn_sum / 熵 取平均（推荐，更稳定）
#   'first' : 只用第一个 sample（快但不稳定）
HEAD_STATS_AGG = "mean"


MIN_PIXELS = 12544
MAX_PIXELS = 76800


# ============================================================
# 模型封装：Qwen3VL + attention 抽取
# ============================================================

class AttentionAttributor:
    """
    封装 Qwen3VL，支持：
    1. 前向推理拿 score（teacher forcing，输入 prompt + score 文本）
    2. 抽取 LLM 内部指定 query token → 所有 image token 的 attention 权重
    """

    def __init__(self, model_path: str, min_pixels: int = 12544, max_pixels: int = 76800):
        print(f"Loading model from {model_path} ...")
        # 关键：eager 才能拿到 attention 权重（sdpa / flash 不返回）
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        self.tok = self.processor.tokenizer

        # Qwen3-VL 的 image placeholder token id（<|image_pad|>）
        # 用 convert_tokens_to_ids 拿，避免硬编码
        self.image_pad_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.vision_start_id = self.tok.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.tok.convert_tokens_to_ids("<|vision_end|>")
        print(f"  <|image_pad|> id = {self.image_pad_id}")
        print(f"  <|vision_start|> id = {self.vision_start_id}")
        print(f"  <|vision_end|> id = {self.vision_end_id}")
        print(f"  Model loaded.\n")

    @torch.no_grad()
    def generate_score(self, image_paths: List[str], task: str,
                       max_new_tokens: int = 64) -> Tuple[str, float]:
        """普通推理拿 score 文本。"""
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
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        return text, parse_score(text)

    def _find_image_token_spans(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        在 input_ids 里找出每张图对应的 image token 范围。

        Qwen3-VL 的格式：<|vision_start|> <|image_pad|>×N <|vision_end|>
        每张图是一段连续的 <|image_pad|>，返回 [(start, end_exclusive), ...]。

        end - start = (t * h * w) / merge^2，即 LLM 实际看到的 token 数。
        """
        ids = input_ids[0].tolist() if input_ids.dim() == 2 else input_ids.tolist()
        spans = []
        i = 0
        n = len(ids)
        while i < n:
            if ids[i] == self.image_pad_id:
                j = i
                while j < n and ids[j] == self.image_pad_id:
                    j += 1
                spans.append((i, j))
                i = j
            else:
                i += 1
        return spans

    def _find_query_positions(self, input_ids: torch.Tensor, response_ids: List[int],
                              query_mode: str) -> List[int]:
        """
        找 query token 在完整序列里的绝对位置（用于从 attention 矩阵取行）。
        """
        n = input_ids.shape[1]
        if query_mode == "last_prompt":
            # 最后一个 prompt token（即第一个 response token 之前那个）
            return [n - 1]
        elif query_mode == "score_digit":
            # 找 response 里 <score>...</score> 之间的数字 token
            open_start, open_end = _find_token_span_by_decode(
                self.tok, response_ids, "<score>"
            )
            close_start, close_end = _find_token_span_by_decode(
                self.tok, response_ids, "</score>"
            )
            if open_start < 0 or close_start < 0:
                return [n - 1]
            prompt_len = n - len(response_ids)
            inner = list(range(open_end, close_start))
            # 只保留数字和小数点
            digit_pos = [
                prompt_len + p for p in inner
                if all(c.isdigit() or c == '.' for c in self.tok.decode([response_ids[p]]).strip())
            ]
            return digit_pos if digit_pos else [n - 1]
        else:
            return [n - 1]

    def build_forward_inputs(
        self,
        image_paths: List[str],
        task: str,
        score_text: str,
    ) -> Tuple[Dict, List[Tuple[int, int]], int]:
        """
        构造 teacher-forcing 前向输入（prompt + score 文本）。
        返回:
            inputs: model() 输入
            image_spans: 每张图在 LLM 序列里的 token 范围
            prompt_len: prompt 长度（不含 response）
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

        response_ids = self.tok.encode(score_text, add_special_tokens=False)
        response_tensor = torch.tensor([response_ids], dtype=torch.long)

        full_input_ids = torch.cat(
            [inputs["input_ids"], response_tensor], dim=1
        ).to(self.model.device)
        full_attention_mask = torch.ones_like(full_input_ids)

        prompt_len = inputs["input_ids"].shape[1]

        inputs_ready = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values": inputs["pixel_values"].to(self.model.device, dtype=self.model.dtype),
            "image_grid_thw": inputs["image_grid_thw"].to(self.model.device),
        }
        if "position_ids" in inputs:
            inputs_ready["position_ids"] = inputs["position_ids"].to(self.model.device)

        image_spans = self._find_image_token_spans(full_input_ids)
        return inputs_ready, image_spans, prompt_len

    @torch.no_grad()
    def compute_attention(
        self,
        image_paths: List[str],
        task: str,
        target_view_idx: int = 5,
        query_mode: str = "last_prompt",
        forward_mode: str = "generate",
    ) -> Tuple[np.ndarray, float, Tuple[int, int], Dict]:
        """
        抽取 query token → 目标图 image token 的 attention 权重。

        forward_mode:
            'generate'        : 正常自回归生成，attention 来自生成过程
            'teacher_forcing' : teacher-force 一次前向，attention 来自重放

        返回:
            attention_map: [num_target_tokens]（已按 HEAD_AGG 聚合）
            score: 原始 score
            (target_start, target_end): 目标图 token 范围
            stats: 全 head 统计
        """
        if forward_mode == "generate":
            return self._compute_attention_generate(
                image_paths, task, target_view_idx
            )
        else:
            return self._compute_attention_teacher_forcing(
                image_paths, task, target_view_idx, query_mode
            )

    @torch.no_grad()
    def _compute_attention_generate(
        self,
        image_paths: List[str],
        task: str,
        target_view_idx: int = 5,
    ) -> Tuple[np.ndarray, float, Tuple[int, int], Dict]:
        """
        正常自回归生成模式。

        用 model.generate(output_attentions=True, return_dict_in_generate=True)。
        attention 来自模型真实的逐步生成过程。

        取 query = 最后一个生成的数字 token（score 最后一位数字），
        它的 attention 反映"模型在确定 score 数值时看了哪里"。
        """
        messages = build_messages(task)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[prompt_text], images=images,
            return_tensors="pt", padding=True,
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[1]

        # 定位 image spans（基于 prompt 的 input_ids，生成后位置不变）
        image_spans = self._find_image_token_spans(inputs["input_ids"])
        if target_view_idx >= len(image_spans):
            raise ValueError(
                f"target_view_idx={target_view_idx} 超出图像数 {len(image_spans)}"
            )
        target_start, target_end = image_spans[target_view_idx]
        target_token_count = target_end - target_start

        # 获取目标图的 grid_thw（用于 spatial entropy 的 2D reshape）
        grid_thw_target = inputs["image_grid_thw"][target_view_idx].tolist()

        # 生成 + 收集 attention
        print(f"    [generate mode] generating with output_attentions ...")
        out = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False, temperature=None, top_p=None,
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
        # out.attentions: tuple[len = num_generated_steps]
        #   第 0 步（prefill）: 每层 [1, H, prompt_len, prompt_len]
        #   第 k 步（k>=1, decode）: 每层 [1, H, 1, prompt_len+k]
        gen_ids = out.sequences[0, prompt_len:]
        score_text = self.tok.decode(gen_ids, skip_special_tokens=True)
        score = parse_score(score_text)
        print(f"    Score text: {score_text!r}  score={score}")

        # 找最后一个数字 token 的生成步
        # gen_ids[k] 是第 k+1 个生成 token
        digit_step = None
        for k in range(len(gen_ids) - 1, -1, -1):
            tok_str = self.tok.decode([int(gen_ids[k])]).strip()
            if tok_str and all(c.isdigit() or c == '.' for c in tok_str):
                digit_step = k
                break
        if digit_step is None:
            digit_step = len(gen_ids) - 1
        # 生成步索引：out.attentions[0] 是 prefill，out.attentions[1] 是第 1 个 token
        # 所以第 digit_step 个生成 token 对应 out.attentions[digit_step + 1]
        attn_idx = digit_step + 1
        print(f"    Last digit token at gen step {digit_step}, attn idx {attn_idx}")
        print(f"    Last digit token: {self.tok.decode([int(gen_ids[digit_step])])!r}")

        attentions = out.attentions[attn_idx]   # tuple[num_layers], 每层 [1, H, 1, T]
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        print(f"    Got attentions: {num_layers} layers × {num_heads} heads")

        # 计算 grid 用于 spatial entropy 的 reshape
        grid_h, grid_w = _grid_hw_from_thw(grid_thw_target, self.processor)

        # 抽取 query(=生成 token) → target_img 的 attention
        q_to_target = np.zeros(
            (num_layers, num_heads, target_token_count), dtype=np.float32
        )
        attn_sum = np.zeros((num_layers, num_heads), dtype=np.float32)
        spatial_entropy = np.zeros((num_layers, num_heads), dtype=np.float32)

        for l in range(num_layers):
            # [1, H, 1, T] -> [H, T]
            attn_lh = attentions[l][0, :, 0, :].float().cpu().numpy()
            for h in range(num_heads):
                row = attn_lh[h]                              # [T]
                seg = row[target_start:target_end]            # [target_token_count]
                q_to_target[l, h] = seg
                total = seg.sum()
                attn_sum[l, h] = total
                if SPATIAL_ENTROPY_MODE == "connected" and grid_h > 0 and grid_w > 0:
                    seg_2d = _safe_reshape_2d(seg, grid_h, grid_w)
                    spatial_entropy[l, h] = _spatial_entropy_connected(seg_2d) if total > 1e-12 else float(seg.size)
                else:  # shannon
                    if total > 1e-12:
                        p = seg / total
                        p_nz = p[p > 1e-12]
                        spatial_entropy[l, h] = -float((p_nz * np.log(p_nz)).sum())
                    else:
                        spatial_entropy[l, h] = float(target_token_count) * np.log(target_token_count)

        del out, attentions
        torch.cuda.empty_cache()

        agg_map, selected_heads, agg_info = _aggregate_heads(
            q_to_target, attn_sum, spatial_entropy, target_token_count,
            grid_thw_target=grid_thw_target,
        )

        stats = {
            "attn_sum": attn_sum,
            "spatial_entropy": spatial_entropy,
            "q_to_target_full": q_to_target if (SAVE_ALL_HEAD_STATS or OUTPUT_PER_HEAD_VIDEOS) else None,
            "selected_heads": selected_heads,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "query_positions": [prompt_len + digit_step],
            "target_start": int(target_start),
            "target_end": int(target_end),
            "forward_mode": "generate",
            "agg_info": agg_info,
            "grid_thw_target": grid_thw_target,
        }
        return agg_map, score, (int(target_start), int(target_end)), stats

    @torch.no_grad()
    def _compute_attention_teacher_forcing(
        self,
        image_paths: List[str],
        task: str,
        target_view_idx: int = 5,
        query_mode: str = "last_prompt",
    ) -> Tuple[np.ndarray, float, Tuple[int, int], Dict]:
        """
        Teacher-forcing 模式：先 generate 拿 score 文本，再 teacher-force 一次前向。
        """
        # Step 1: 拿 score 文本
        score_text, score = self.generate_score(image_paths, task)
        response_ids = self.tok.encode(score_text, add_special_tokens=False)
        print(f"    Score text: {score_text!r}  score={score}")

        # Step 2: teacher forcing 前向
        inputs, image_spans, prompt_len = self.build_forward_inputs(
            image_paths, task, score_text
        )
        if target_view_idx >= len(image_spans):
            raise ValueError(
                f"target_view_idx={target_view_idx} 超出图像数 {len(image_spans)}"
            )
        target_start, target_end = image_spans[target_view_idx]
        target_token_count = target_end - target_start
        print(f"    Image spans: {len(image_spans)} images")
        print(f"    Target image[{target_view_idx}] token range: "
              f"[{target_start}, {target_end})  ({target_token_count} tokens)")

        # 找 query token 位置
        query_positions = self._find_query_positions(
            inputs["input_ids"], response_ids, query_mode
        )
        print(f"    Query mode: {query_mode}, positions: {query_positions}")

        # 前向
        outputs = self.model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        attentions = outputs.attentions
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        print(f"    Got attentions: {num_layers} layers × {num_heads} heads")

        # 获取目标图的 grid_thw（用于 spatial entropy 的 2D reshape）
        grid_thw_target = inputs["image_grid_thw"][target_view_idx].tolist()
        grid_h, grid_w = _grid_hw_from_thw(grid_thw_target, self.processor)

        # 抽取 [layer, head] 的 query→target_img attention
        q_to_target = np.zeros(
            (num_layers, num_heads, target_token_count), dtype=np.float32
        )
        attn_sum = np.zeros((num_layers, num_heads), dtype=np.float32)
        spatial_entropy = np.zeros((num_layers, num_heads), dtype=np.float32)

        for l in range(num_layers):
            attn_lh = attentions[l][0].float().cpu().numpy()
            for h in range(num_heads):
                row = attn_lh[h, query_positions, :].mean(axis=0)
                seg = row[target_start:target_end]
                q_to_target[l, h] = seg
                total = seg.sum()
                attn_sum[l, h] = total
                if SPATIAL_ENTROPY_MODE == "connected" and grid_h > 0 and grid_w > 0:
                    seg_2d = _safe_reshape_2d(seg, grid_h, grid_w)
                    spatial_entropy[l, h] = _spatial_entropy_connected(seg_2d) if total > 1e-12 else float(seg.size)
                else:  # shannon
                    if total > 1e-12:
                        p = seg / total
                        p_nz = p[p > 1e-12]
                        spatial_entropy[l, h] = -float((p_nz * np.log(p_nz)).sum())
                    else:
                        spatial_entropy[l, h] = float(target_token_count) * np.log(target_token_count)

        del outputs, attentions
        torch.cuda.empty_cache()

        agg_map, selected_heads, agg_info = _aggregate_heads(
            q_to_target, attn_sum, spatial_entropy, target_token_count,
            grid_thw_target=grid_thw_target,
        )

        stats = {
            "attn_sum": attn_sum,
            "spatial_entropy": spatial_entropy,
            "q_to_target_full": q_to_target if (SAVE_ALL_HEAD_STATS or OUTPUT_PER_HEAD_VIDEOS) else None,
            "selected_heads": selected_heads,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "query_positions": query_positions,
            "target_start": int(target_start),
            "target_end": int(target_end),
            "forward_mode": "teacher_forcing",
            "agg_info": agg_info,
            "grid_thw_target": grid_thw_target,
        }
        return agg_map, score, (int(target_start), int(target_end)), stats


# ============================================================
# 辅助：decode-based 子串定位（复用 gradient_attention 的逻辑）
# ============================================================

def _find_token_span_by_decode(tok, token_ids_list: List[int], sub_str: str) -> Tuple[int, int]:
    """
    在 token_ids_list 里找 sub_str 第一次出现的 token 起止位置。
    用累积 decode 逐 token 比对。返回 (start, end_exclusive)，找不到返回 (-1, -1)。
    """
    char_offset_start = -1
    for i in range(len(token_ids_list) + 1):
        if char_offset_start < 0:
            if i == len(token_ids_list):
                break
            new_text = tok.decode(token_ids_list[:i + 1])
            if sub_str in new_text:
                char_offset_start = new_text.find(sub_str)
                start_token = i
                char_offset_end = char_offset_start + len(sub_str)
                for j in range(i + 1, len(token_ids_list) + 1):
                    full = tok.decode(token_ids_list[:j])
                    if len(full) >= char_offset_end:
                        return start_token, j
                return start_token, len(token_ids_list)
    return -1, -1


def _grid_hw_from_thw(grid_thw: List[int], processor) -> Tuple[int, int]:
    """
    从 image_grid_thw (t, h, w) 计算 LLM 看到的 2D 网格 (hm, wm)。
    hm = h // merge_size, wm = w // merge_size
    """
    merge = getattr(processor.image_processor, "merge_size", 2) if hasattr(processor, "image_processor") else 2
    t, h, w = grid_thw
    return h // merge, w // merge


def _safe_reshape_2d(seg_1d: np.ndarray, h: int, w: int) -> np.ndarray:
    """把 1D attention 安全 reshape 成 2D 网格（处理长度不匹配）。"""
    expected = h * w
    if seg_1d.shape[0] == expected:
        return seg_1d.reshape(h, w)
    if seg_1d.shape[0] > expected:
        return seg_1d[:expected].reshape(h, w)
    # 不足则填充
    padded = np.zeros(expected, dtype=seg_1d.dtype)
    padded[:seg_1d.shape[0]] = seg_1d
    return padded.reshape(h, w)


def _spatial_entropy_connected(seg_2d: np.ndarray) -> float:
    """
    论文 2503.06287 的 spatial entropy 实现。

    步骤：
    1. 把 attention map 二值化（>均值=1，否则=0）
    2. 找 8-连通分量
    3. 对连通分量大小分布算 Shannon entropy

    低 entropy = attention 聚集在少数大块（聚焦）
    高 entropy = attention 碎成很多小块（分散）
    """
    seg = seg_2d.astype(np.float32)
    mean_val = seg.mean()
    binary = (seg > mean_val).astype(np.uint8)
    if binary.sum() == 0:
        return float(seg.size)  # 全 0，最大熵

    # 找连通分量（8-连通）
    num_labels, labels, sizes, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # sizes[0] 是背景，忽略
    component_sizes = sizes[1:]
    if len(component_sizes) == 0:
        return float(seg.size)

    total = component_sizes.sum()
    probs = component_sizes / total
    entropy = -float((probs * np.log(probs + 1e-12)).sum())
    return entropy


def _max_curvature_threshold(sorted_values: np.ndarray) -> float:
    """
    论文 2503.06287 的阈值 τ 选取：最大曲率法（knee detection）。

    输入：升序排列的 attention sum 值
    输出：阈值 τ，曲率最大的点

    原理：把排序后的值画成曲线，找"拐点"——
    曲线从平缓变陡峭（或从陡峭变平缓）的位置。
    """
    n = len(sorted_values)
    if n < 3:
        return float(sorted_values[-1])

    x = np.arange(n, dtype=np.float64)
    y = sorted_values.astype(np.float64)

    # 归一化到 [0,1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # 对每个内部点，算它到首尾连线的距离（knee detection 的标准做法）
    # 首点 (x_norm[0], y_norm[0])，尾点 (x_norm[-1], y_norm[-1])
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    distances = np.zeros(n)
    for i in range(n):
        p = np.array([x_norm[i], y_norm[i]])
        # 点到直线的距离
        line_vec = p2 - p1
        point_vec = p - p1
        line_len = np.linalg.norm(line_vec) + 1e-12
        distance = np.abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]) / line_len
        distances[i] = distance

    # 最大距离的点就是 knee
    knee_idx = int(np.argmax(distances))
    # 阈值取 knee 点附近的值（论文用 knee 点的 sum 值作为 τ）
    return float(sorted_values[knee_idx])


def _most_common_heads(all_records: List[Dict], top_n: int = 8) -> List[Tuple[int, int]]:
    """统计所有 sample 里被选中的 head 出现频率，返回 top_n 最频繁的。"""
    from collections import Counter
    counter = Counter()
    for r in all_records:
        for lh in r.get("selected_heads", []):
            counter[tuple(lh)] += 1
    return [lh for lh, _ in counter.most_common(top_n)]


def _aggregate_heads(
    q_to_target: np.ndarray,
    attn_sum: np.ndarray,
    spatial_entropy: np.ndarray,
    target_token_count: int,
    grid_thw_target: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict]:
    """
    按 HEAD_AGG / HEAD_CRITERION 聚合全 head 的 attention。

    改进点（对齐论文 2503.06287）：
    - 排除前 EXCLUDE_FIRST_LAYERS 层（论文排除前 2 层）
    - ATTN_SUM_THRESHOLD='max_curvature' 时用最大曲率法选 τ
    - HEAD_CRITERION 含 low_entropy 时按 SPATIAL_ENTROPY_MODE 选 entropy 计算

    q_to_target: [L, H, T]
    attn_sum:    [L, H]
    spatial_entropy: [L, H]  （注意：这里传入的应该是 connected entropy）

    返回:
        agg_map: [T] 聚合后的 attention
        selected_heads: 选中的 (layer, head) 列表（mean 模式为空）
        info: 诊断信息（阈值、候选池大小等）
    """
    num_layers, num_heads = attn_sum.shape
    info = {"excluded_layers": list(range(EXCLUDE_FIRST_LAYERS))}

    # 构造 layer mask：True 表示该层参与筛选
    layer_mask = np.array([l >= EXCLUDE_FIRST_LAYERS for l in range(num_layers)])

    if HEAD_AGG == "mean":
        agg_map = q_to_target.mean(axis=(0, 1))
        print(f"    Aggregation: mean over all {num_layers * num_heads} heads")
        return agg_map, [], info

    # topk 模式：按 HEAD_CRITERION 评分排序
    # 先把有效层（排除前 N 层）的 head 收集起来
    valid_indices = []   # (flat_idx, layer, head)
    for l in range(num_layers):
        if not layer_mask[l]:
            continue
        for h in range(num_heads):
            valid_indices.append((l * num_heads + h, l, h))

    print(f"    Candidate pool: {len(valid_indices)} heads "
          f"(excluded first {EXCLUDE_FIRST_LAYERS} layers)")

    if HEAD_CRITERION == "attn_sum":
        # 标准 1：attention sum 高
        if ATTN_SUM_THRESHOLD == "max_curvature":
            # 论文方法：先对所有有效 head 的 sum 排序，找最大曲率点作为阈值 τ
            valid_sums = np.array([attn_sum[l, h] for (_, l, h) in valid_indices])
            sorted_sums = np.sort(valid_sums)
            tau = _max_curvature_threshold(sorted_sums)
            info["tau"] = tau
            print(f"    Max-curvature threshold τ = {tau:.4f}")
            # 选 sum >= τ 的 head，如果不够 HEAD_TOPK 个，就取 top HEAD_TOPK
            candidates = [(l, h) for (_, l, h) in valid_indices if attn_sum[l, h] >= tau]
            if len(candidates) < HEAD_TOPK:
                # 退回到 top-K by sum
                flat_scores = np.array([attn_sum[l, h] for (_, l, h) in valid_indices])
                top_order = np.argsort(flat_scores)[::-1][:HEAD_TOPK]
                candidates = [valid_indices[i][1:] for i in top_order]
                print(f"    τ too strict, fell back to top-{HEAD_TOPK} by attn_sum")
            else:
                # 在候选里按 sum 降序取 top-K
                candidates.sort(key=lambda lh: attn_sum[lh[0], lh[1]], reverse=True)
                candidates = candidates[:HEAD_TOPK]
            selected_heads = candidates
            print(f"    Aggregation: top-{HEAD_TOPK} by attn_sum (τ={tau:.4f})")
        else:
            # 分位数法
            flat_scores = np.array([attn_sum[l, h] for (_, l, h) in valid_indices])
            topk_local = np.argsort(flat_scores)[::-1][:HEAD_TOPK]
            selected_heads = [valid_indices[i][1:] for i in topk_local]
            print(f"    Aggregation: top-{HEAD_TOPK} by attn_sum")

    elif HEAD_CRITERION == "low_entropy":
        # 标准 2：空间熵低（attention 聚焦）
        flat_ent = np.array([spatial_entropy[l, h] for (_, l, h) in valid_indices])
        topk_local = np.argsort(flat_ent)[:HEAD_TOPK]   # 升序：熵小在前
        selected_heads = [valid_indices[i][1:] for i in topk_local]
        print(f"    Aggregation: top-{HEAD_TOPK} by low spatial entropy ({SPATIAL_ENTROPY_MODE})")

    elif HEAD_CRITERION == "attn_sum_low_entropy":
        # 标准 1+2：attention sum 高 且 熵低（论文完整标准）
        # 先用 attn_sum 筛候选（max_curvature 或 top 比例）
        flat_sum = np.array([attn_sum[l, h] for (_, l, h) in valid_indices])
        flat_ent = np.array([spatial_entropy[l, h] for (_, l, h) in valid_indices])

        if ATTN_SUM_THRESHOLD == "max_curvature":
            tau = _max_curvature_threshold(np.sort(flat_sum))
            info["tau"] = tau
            sum_mask = flat_sum >= tau
            print(f"    Max-curvature threshold τ = {tau:.4f}, "
                  f"{int(sum_mask.sum())} heads pass")
        else:
            cutoff = np.percentile(flat_sum, 100 - LOW_ENTROPY_PERCENTILE)
            tau = float(cutoff)
            info["tau"] = tau
            sum_mask = flat_sum >= cutoff

        # 在候选里按熵升序取 top-K
        candidate_local = np.where(sum_mask)[0]
        if len(candidate_local) == 0:
            # 退回到 top-K by sum
            candidate_local = np.argsort(flat_sum)[::-1][:HEAD_TOPK * 4]
        ent_order = candidate_local[np.argsort(flat_ent[candidate_local])]
        topk_local = ent_order[:HEAD_TOPK]
        selected_heads = [valid_indices[i][1:] for i in topk_local]
        print(f"    Aggregation: top-{HEAD_TOPK} by attn_sum(≥τ) + low entropy")
    else:
        flat_scores = np.array([attn_sum[l, h] for (_, l, h) in valid_indices])
        topk_local = np.argsort(flat_scores)[::-1][:HEAD_TOPK]
        selected_heads = [valid_indices[i][1:] for i in topk_local]
        print(f"    Aggregation: top-{HEAD_TOPK} by attn_sum (default)")

    sel_maps = np.stack([q_to_target[l, h] for (l, h) in selected_heads])
    agg_map = sel_maps.mean(axis=0)
    print(f"    Selected heads (layer, head): {selected_heads}")
    return agg_map, selected_heads, info


# ============================================================
# 可视化
# ============================================================

def render_attention_frame(
    image_path: str,
    attention_map: np.ndarray,
    grid_thw_target: List[int],
    orig_score: float,
    step_idx: int,
    frame_idx: int,
    task: str,
    out_path: str,
    panel_w: int = 384,
    panel_h: int = 288,
    head_label: str = "",
) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (panel_w, panel_h))

    heatmap_2d = patches_to_2d(attention_map, grid_thw_target)
    max_val = float(heatmap_2d.max())
    if max_val > 1e-8:
        heatmap_norm = heatmap_2d / max_val
    else:
        heatmap_norm = heatmap_2d

    heatmap_resized = cv2.resize(heatmap_norm, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
    canvas = np.hstack([img, heatmap_color, overlay])

    info_h = 50
    info_bar = np.ones((info_h, canvas.shape[1], 3), dtype=np.uint8) * 30
    label_prefix = f"[{head_label}]  " if head_label else ""
    info_txt = (f"{label_prefix}Step {step_idx}  Frame {frame_idx}  |  "
                f"Score: {orig_score:+.1f}%  |  Max attn: {max_val:.4f}  |  "
                f"Grid: {heatmap_2d.shape}")
    cv2.putText(info_bar, info_txt, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    task_h = 30
    task_bar = np.ones((task_h, canvas.shape[1], 3), dtype=np.uint8) * 30
    cv2.putText(task_bar, f"Task: {task}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    title_h = 25
    title_bar = np.ones((title_h, canvas.shape[1], 3), dtype=np.uint8) * 60
    for i, txt in enumerate(["Original", "Attention", "Overlay"]):
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


def plot_head_stats(stats: Dict, out_path: str):
    """画 attention_sum 和空间熵的 layer×head 热力图（用于 head 分析）。"""
    attn_sum = stats["attn_sum"]           # [L, H]
    entropy = stats["spatial_entropy"]     # [L, H]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(attn_sum, aspect="auto", cmap="viridis")
    axes[0].set_title("Attention Sum (query → target image)")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(entropy, aspect="auto", cmap="viridis_r")
    axes[1].set_title("Spatial Entropy (low = focused)")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    plt.colorbar(im1, ax=axes[1])

    # 标记选中的 head
    for (l, h) in stats.get("selected_heads", []):
        axes[0].plot(h, l, "rx", markersize=10, mew=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_head_scatter(
    attn_sum: np.ndarray,
    spatial_entropy: np.ndarray,
    selected_heads: List[Tuple[int, int]],
    tau: Optional[float],
    out_path: str,
):
    """
    论文 2503.06287 风格的 scatter plot：
    横坐标 = spatial entropy（低=聚焦）
    纵坐标 = attention sum（高=看图）

    理想 head 在右上角（论文的"高 sum + 低 entropy"），
    习惯上把 entropy 放横轴、sum 放纵轴，理想区在左上。

    selected_heads 用红色高亮，并标注 "L{layer}H{head}"。
    tau 用水平虚线标出（attention sum 阈值）。
    """
    L, H = attn_sum.shape
    # 展开成点列表
    sums = attn_sum.flatten()
    ents = spatial_entropy.flatten()
    labels = [(l, h) for l in range(L) for h in range(H)]
    selected_set = set(selected_heads)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 按是否被选中分两组画
    is_sel = np.array([labels[i] in selected_set for i in range(len(labels))])

    # 排除前 EXCLUDE_FIRST_LAYERS 层的点用浅色
    layer_arr = np.array([l for (l, h) in labels])
    is_excluded = layer_arr < EXCLUDE_FIRST_LAYERS

    # 普通 head（未选中、未排除）
    mask_normal = (~is_sel) & (~is_excluded)
    ax.scatter(ents[mask_normal], sums[mask_normal],
               c="lightgray", s=20, alpha=0.6, label=f"others (excl. first {EXCLUDE_FIRST_LAYERS} layers)")

    # 被排除层的 head
    mask_excluded = (~is_sel) & is_excluded
    ax.scatter(ents[mask_excluded], sums[mask_excluded],
               c="#DDDDDD", s=15, alpha=0.4, marker="x", label=f"excluded first {EXCLUDE_FIRST_LAYERS} layers")

    # 选中的 head
    mask_sel = is_sel
    ax.scatter(ents[mask_sel], sums[mask_sel],
               c="red", s=120, alpha=0.9, edgecolors="darkred", linewidths=1.5,
               label=f"selected top-{len(selected_heads)} heads", zorder=5)

    # 给选中的 head 标注名字
    for i in range(len(labels)):
        if mask_sel[i]:
            l, h = labels[i]
            ax.annotate(f"L{l}H{h}", (ents[i], sums[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color="darkred", fontweight="bold")

    # τ 阈值线
    if tau is not None:
        ax.axhline(y=tau, color="blue", linestyle="--", alpha=0.6, label=f"τ={tau:.3f} (max curvature)")

    ax.set_xlabel(f"Spatial Entropy ({SPATIAL_ENTROPY_MODE})  ←  lower = more focused")
    ax.set_ylabel("Attention Sum  ←  higher = attends to image")
    ax.set_title("Head Selection: Attention Sum vs Spatial Entropy\n"
                 "(ideal heads: top-left = high sum + low entropy)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"\n{'=' * 70}")
    print("Attention-based Attribution (Qwen3VL)")
    print(f"{'=' * 70}")
    print(f"Model:           {MODEL_PATH}")
    print(f"Data:            {DATA_DIR}")
    print(f"Task:            {TASK_INSTRUCTION}")
    print(f"Eval mode:       {EVAL_MODE}")
    print(f"Target view:     image[{TARGET_VIEW_INDEX}] (AFTER High)")
    print(f"Forward mode:    {FORWARD_MODE}")
    print(f"Query mode:      {QUERY_MODE} (仅 teacher_forcing 模式生效)")
    print(f"Head aggregation:{HEAD_AGG} (topk={HEAD_TOPK}, criterion={HEAD_CRITERION})")
    print(f"Exclude layers:  first {EXCLUDE_FIRST_LAYERS} layers")
    print(f"Attn sum thresh: {ATTN_SUM_THRESHOLD}")
    print(f"Spatial entropy: {SPATIAL_ENTROPY_MODE}")
    print(f"Video content:   {VIDEO_CONTENT}")
    print(f"Per-head videos: {OUTPUT_PER_HEAD_VIDEOS} (mode={HEAD_SELECTION_MODE})")
    print(f"Save npz:        {SAVE_ALL_HEAD_STATS}")
    print(f"Head stats agg:  {HEAD_STATS_AGG}")
    print(f"{'=' * 70}\n")

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
        ref_end_path = str(cam_dirs["cam_high"] / f"frame_{total_frames - 1:06d}.png")

    # --- 2. 构建 samples ---
    print("\n[2/5] Building samples ...")
    samples = build_samples(cache_root, TASK_INSTRUCTION, indices, ref_end_path, mode=EVAL_MODE)
    print(f"  Samples: {len(samples)}\n")

    # --- 3. 加载模型 ---
    print("[3/5] Loading model (eager attention) ...")
    attributor = AttentionAttributor(MODEL_PATH, MIN_PIXELS, MAX_PIXELS)

    # --- 4. 逐帧 attention 抽取 ---
    print("\n[4/5] Running attention extraction ...")
    video_frames = []                # 聚合后的视频帧
    all_records = []
    accumulated_attn_sum = []        # 累积所有 sample 的 attn_sum，用于最后的 overview
    accumulated_entropy = []
    accumulated_tau = []             # 累积阈值（用于报告平均 τ）
    per_sample_data = []             # 每个 sample 的 (image_path, score, step, frame_idx, q_to_target_full, grid_thw)
    total_steps = len(samples)

    for step_idx, item in enumerate(samples):
        t0 = time.time()
        frame_idx = indices[step_idx + 1]
        target_image_path = item["image"][TARGET_VIEW_INDEX]
        print(f"\n  [Step {step_idx + 1}/{total_steps}] frame_idx={frame_idx}")

        try:
            attention_map, orig_score, (t_start, t_end), stats = attributor.compute_attention(
                image_paths=item["image"],
                task=item["task"],
                target_view_idx=TARGET_VIEW_INDEX,
                query_mode=QUERY_MODE,
                forward_mode=FORWARD_MODE,
            )
        except Exception as e:
            print(f"    [ERROR] attention extraction failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        max_attn = float(attention_map.max())
        print(f"    Score: {orig_score:+.1f}%  |  Max attn: {max_attn:.4f}  |  Time: {elapsed:.1f}s")

        # 累积全 head 统计（用于最后的 head_stats_overview）
        accumulated_attn_sum.append(stats["attn_sum"])
        accumulated_entropy.append(stats["spatial_entropy"])
        if stats.get("agg_info", {}).get("tau") is not None:
            accumulated_tau.append(stats["agg_info"]["tau"])

        # 拿目标图 grid_thw
        grid_thw_target = stats.get("grid_thw_target")
        if grid_thw_target is None:
            images_pil = [Image.open(p).convert("RGB") for p in item["image"]]
            messages = build_messages(item["task"])
            prompt_text = attributor.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            proc_inputs = attributor.processor(
                text=[prompt_text], images=images_pil, return_tensors="pt", padding=True
            )
            grid_thw_target = proc_inputs["image_grid_thw"][TARGET_VIEW_INDEX].tolist()

        # 渲染聚合帧（主视频）
        heatmap_png = str(heatmaps_dir / f"heatmap_step{step_idx:04d}_frame{frame_idx:06d}.png")
        frame_rgb = render_attention_frame(
            image_path=target_image_path,
            attention_map=attention_map,
            grid_thw_target=grid_thw_target,
            orig_score=orig_score,
            step_idx=step_idx,
            frame_idx=frame_idx,
            task=TASK_INSTRUCTION,
            out_path=heatmap_png,
            head_label=f"top-{HEAD_TOPK} {HEAD_CRITERION}",
        )
        video_frames.append(frame_rgb)

        # 保存每个 sample 的完整 head 数据（用于跑完后统一渲染 per-head 视频）
        # 注意：q_to_target_full 只在 OUTPUT_PER_HEAD_VIDEOS 或 SAVE_ALL_HEAD_STATS 开启时保留
        if OUTPUT_PER_HEAD_VIDEOS and stats.get("q_to_target_full") is not None:
            per_sample_data.append({
                "image_path": target_image_path,
                "score": orig_score,
                "step_idx": step_idx,
                "frame_idx": frame_idx,
                "q_to_target_full": stats["q_to_target_full"],   # [L, H, T]
                "grid_thw_target": grid_thw_target,
                "selected_heads": stats.get("selected_heads", []),
            })

        record = {
            "step": step_idx,
            "frame_idx": frame_idx,
            "orig_score": orig_score,
            "max_attn": max_attn,
            "mean_attn": float(attention_map.mean()),
            "attention_map": attention_map.tolist(),
            "grid_thw_target": grid_thw_target,
            "selected_heads": stats.get("selected_heads", []),
            "target_token_range": [t_start, t_end],
            "forward_mode": stats.get("forward_mode", FORWARD_MODE),
        }
        all_records.append(record)

        # 保存这一步的全 head 统计（npz，可选）
        if SAVE_ALL_HEAD_STATS:
            npz_path = run_root / f"head_stats_step{step_idx:04d}.npz"
            save_dict = {
                "attn_sum": stats["attn_sum"],
                "spatial_entropy": stats["spatial_entropy"],
            }
            if stats.get("q_to_target_full") is not None:
                save_dict["q_to_target_full"] = stats["q_to_target_full"]
            np.savez_compressed(str(npz_path), **save_dict)

    print(f"\n  Done. {len(video_frames)} frames rendered.\n")

    # --- 5. 保存 ---
    print("[5/5] Saving results ...")

    json_path = run_root / "attention_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "method": "Attention (query token → target image)",
                "model_path": MODEL_PATH,
                "data_dir": DATA_DIR,
                "task": TASK_INSTRUCTION,
                "goal_image": GOAL_IMAGE,
                "frame_interval": FRAME_INTERVAL,
                "eval_mode": EVAL_MODE,
                "target_view_index": TARGET_VIEW_INDEX,
                "forward_mode": FORWARD_MODE,
                "query_mode": QUERY_MODE,
                "head_agg": HEAD_AGG,
                "head_topk": HEAD_TOPK,
                "head_criterion": HEAD_CRITERION,
                "exclude_first_layers": EXCLUDE_FIRST_LAYERS,
                "attn_sum_threshold": ATTN_SUM_THRESHOLD,
                "spatial_entropy_mode": SPATIAL_ENTROPY_MODE,
                "video_content": VIDEO_CONTENT,
                "per_head_videos": OUTPUT_PER_HEAD_VIDEOS,
                "head_selection_mode": HEAD_SELECTION_MODE,
            },
            "records": all_records,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Results JSON: {json_path}")

    # 头统计汇总：layer×head 热力图 + scatter plot
    if accumulated_attn_sum:
        try:
            if HEAD_STATS_AGG == "first":
                sum_mat = accumulated_attn_sum[0]
                ent_mat = accumulated_entropy[0]
            else:
                sum_mat = np.mean(np.stack(accumulated_attn_sum), axis=0)
                ent_mat = np.mean(np.stack(accumulated_entropy), axis=0)
            common_heads = _most_common_heads(all_records)

            # 热力图
            plot_head_stats(
                {
                    "attn_sum": sum_mat,
                    "spatial_entropy": ent_mat,
                    "selected_heads": common_heads,
                },
                str(run_root / "head_stats_overview.png"),
            )
            print(f"  Head stats overview ({HEAD_STATS_AGG} over {len(accumulated_attn_sum)} samples): "
                  f"{run_root / 'head_stats_overview.png'}")

            # scatter plot（论文 Fig 6 风格）
            mean_tau = float(np.mean(accumulated_tau)) if accumulated_tau else None
            plot_head_scatter(
                attn_sum=sum_mat,
                spatial_entropy=ent_mat,
                selected_heads=common_heads,
                tau=mean_tau,
                out_path=str(run_root / "head_scatter.png"),
            )
            print(f"  Head scatter (sum vs entropy): {run_root / 'head_scatter.png'}")
            if mean_tau is not None:
                print(f"    Mean τ (max curvature): {mean_tau:.4f}")
        except Exception as e:
            print(f"  [WARN] head stats plot failed: {e}")
            import traceback
            traceback.print_exc()

    # 视频汇总
    # VIDEO_CONTENT='selected' : 用 HEAD_AGG/HEAD_CRITERION 筛出的 head（即每帧热力图用的同一份）
    # VIDEO_CONTENT='mean'     : 用全 head 平均（独立于 HEAD_AGG，作为对照）
    video_path = run_root / f"attention_video_{VIDEO_CONTENT}.mp4"
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
        content_desc = {
            "selected": f"top-{HEAD_TOPK} heads by {HEAD_CRITERION}" if HEAD_AGG == "topk" else "mean of all heads",
            "mean": "mean of all heads (forced)",
        }.get(VIDEO_CONTENT, VIDEO_CONTENT)
        print(f"  Attention video ({VIDEO_CONTENT}: {content_desc}): {video_path}")

    # 每个固定 top-K head 单独的视频（对齐论文 selection frequency 思想）
    # 不再每 sample 独立渲染（那样会产生 25+ 个视频），而是：
    # 1. 跑完所有 sample 后，按 selection frequency 选固定 HEAD_TOPK 个 head
    # 2. 用这些固定 head 对每个 sample 渲染一帧，组成完整视频
    if OUTPUT_PER_HEAD_VIDEOS and per_sample_data:
        per_head_dir = run_root / "per_head_videos"
        per_head_dir.mkdir(exist_ok=True)

        # 选固定 head
        if HEAD_SELECTION_MODE == "frequency":
            from collections import Counter
            counter = Counter()
            for sd in per_sample_data:
                for lh in sd["selected_heads"]:
                    counter[tuple(lh)] += 1
            fixed_heads = [lh for lh, _ in counter.most_common(HEAD_TOPK)]
            print(f"  Fixed heads by selection frequency (top-{HEAD_TOPK}):")
            for lh in fixed_heads:
                print(f"    L{lh[0]:02d}H{lh[1]:02d}: selected in {counter[lh]}/{len(per_sample_data)} samples")
        else:  # 'mean_sum'
            mean_sum = np.mean(np.stack(accumulated_attn_sum), axis=0)
            flat = mean_sum.flatten()
            topk = np.argsort(flat)[::-1][:HEAD_TOPK]
            num_heads = mean_sum.shape[1]
            fixed_heads = [(int(i // num_heads), int(i % num_heads)) for i in topk]
            print(f"  Fixed heads by mean attn_sum (top-{HEAD_TOPK}): {fixed_heads}")

        # 用固定 head 渲染每个 sample 的帧
        for (l, h) in fixed_heads:
            frames = []
            for sd in per_sample_data:
                q_full = sd["q_to_target_full"]   # [L, H, T]
                head_map = q_full[l, h]
                head_frame = render_attention_frame(
                    image_path=sd["image_path"],
                    attention_map=head_map,
                    grid_thw_target=sd["grid_thw_target"],
                    orig_score=sd["score"],
                    step_idx=sd["step_idx"],
                    frame_idx=sd["frame_idx"],
                    task=TASK_INSTRUCTION,
                    out_path=None,
                    head_label=f"L{l:02d}H{h:02d}",
                )
                frames.append(head_frame)

            head_video_path = per_head_dir / f"L{l:02d}H{h:02d}.mp4"
            if _MOVIEPY_AVAILABLE:
                clip = ImageSequenceClip(frames, fps=VIDEO_FPS)
                clip.write_videofile(str(head_video_path), logger=None)
            else:
                h_, w_ = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(head_video_path), fourcc, VIDEO_FPS, (w_, h_))
                for frm in frames:
                    writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
                writer.release()
            print(f"    → {head_video_path}")

    # score / max_attn 曲线
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = [r["step"] for r in all_records]
        scores = [r["orig_score"] for r in all_records]
        maxes = [r["max_attn"] for r in all_records]
        ax2 = ax.twinx()
        ax.plot(steps, scores, "o-", color="#2196F3", label="Score (%)")
        ax2.plot(steps, maxes, "s-", color="#FF5722", label="Max attention")
        ax.set_xlabel("Step")
        ax.set_ylabel("Original Score (%)", color="#2196F3")
        ax2.set_ylabel("Max attention", color="#FF5722")
        ax.set_title(f"Score vs Attention Over Time\nTask: {TASK_INSTRUCTION}")
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.95))
        fig.tight_layout()
        curve_path = run_root / "score_attention_curve.png"
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        print(f"  Curve:           {curve_path}")
    except Exception as e:
        print(f"  [WARN] curve plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"ALL DONE. Results saved under: {run_root}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
