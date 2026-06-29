"""
Generate attention and gradient attribution videos for Robo-Dopamine GRM.

This script is intentionally separate from examples.inference.GRMInference:
vLLM is excellent for fast score generation, while attribution needs
Transformers internals such as attention tensors and gradients.
"""

import gc
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from examples.inference import (  # noqa: E402
    SYSTEM_PROMPT,
    build_samples_json,
    ensure_dir,
    get_frame_count,
    make_sample_indices_by_interval,
    save_frames,
)
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402


# ============================
# Config: keep this close to test_my_data_suc.py
# ============================
# MODEL_PATH = "/home/dais/workspace/Robo-Dopamine/train/checkpoints/my_carrot_finetune_big"
MODEL_PATH = './pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview'
DATA_DIR = "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot"
OUTPUT_ROOT = "./results/eval_visual/exp_suc1_inter20_cube_ref_attribution_new"

TASK_INSTRUCTION = "pick the cube and put it on yellow plate "
# GOAL_IMAGE = "./examples/blank_goal.png"
GOAL_IMAGE = "./examples/exp_suc_4_cube.png"
INTERVAL = 20

EVAL_MODES = ["forward", "incremental", "backward"]

# If you already have vLLM outputs, set paths here, for example:
# PRED_JSON_BY_MODE = {"forward": ".../pred_vllm.json"}
# Missing modes will be generated with the HF model.
PRED_JSON_BY_MODE: Dict[str, str] = {}

# Use --max-samples for quick validation; None means all samples.
MAX_SAMPLES: Optional[int] = None

MIN_PIXELS = 12544
MAX_PIXELS = 76800
ATTENTION_LAST_N_LAYERS = 4
ATTENTION_METHOD = "rollout"  # "rollout" or "raw"
HEAD_FUSION = "max"  # "mean", "max", or "min"
DISCARD_RATIO = 0.9
VIDEO_FPS = 4.0
OVERLAY_ALPHA = 0.45


AFTER_IMAGE_INDICES = [5, 6, 7]
VIEW_LABELS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
INPUT_IMAGE_LABELS = [
    "reference_start",
    "reference_end_or_goal",
    "before_cam_high",
    "before_cam_left_wrist",
    "before_cam_right_wrist",
    "after_cam_high",
    "after_cam_left_wrist",
    "after_cam_right_wrist",
]


def sanitize_task(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", task)


def parse_score_fraction(pred: str) -> float:
    match = re.search(r"<score>\s*([+-]?\d+(?:\.\d+)?)%\s*</score>", pred)
    if not match:
        match = re.search(r"([+-]?\d+(?:\.\d+)?)%", pred)
    if not match:
        return 0.0
    return max(-100.0, min(100.0, float(match.group(1)))) / 100.0


def attach_progress(results: List[dict], eval_mode: str) -> List[dict]:
    prev_prog = 0.0
    for item in results:
        raw_score = parse_score_fraction(item.get("pred", ""))
        if eval_mode == "incremental":
            if prev_prog == 0.0:
                curr_progress = raw_score
            elif raw_score >= 0:
                curr_progress = prev_prog + (1 - prev_prog) * raw_score
            else:
                curr_progress = prev_prog + prev_prog * raw_score
            hop = raw_score
        elif eval_mode == "forward":
            curr_progress = raw_score
            hop = curr_progress - prev_prog
        elif eval_mode == "backward":
            curr_progress = max(0.0, min(1.0, 1.0 + raw_score))
            hop = curr_progress - prev_prog
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

        item["hop"] = hop
        item["progress"] = curr_progress
        prev_prog = curr_progress
    return results


def prepare_samples(eval_mode: str, run_root: Path) -> List[dict]:
    cache_root = run_root / ".cache"
    cam_dirs = {
        "cam_high": cache_root / "cam_high",
        "cam_left_wrist": cache_root / "cam_left_wrist",
        "cam_right_wrist": cache_root / "cam_right_wrist",
    }
    for cam_dir in cam_dirs.values():
        ensure_dir(cam_dir)

    paths = [
        Path(DATA_DIR) / "cam_high.mp4",
        Path(DATA_DIR) / "cam_left_wrist.mp4",
        Path(DATA_DIR) / "cam_right_wrist.mp4",
    ]
    types_counts = [get_frame_count(path) for path in paths]
    counts = [count for _, count in types_counts]
    if len(set(counts)) != 1:
        raise ValueError(f"Frame count mismatch among cameras: {counts}")

    indices = make_sample_indices_by_interval(counts[0], INTERVAL)
    print(f"[{eval_mode}] frames={counts[0]}, interval={INTERVAL}, samples={len(indices) - 1}")

    for path, key, (src_type, _) in zip(paths, cam_dirs.keys(), types_counts):
        save_frames(path, cam_dirs[key], indices, src_type)

    ref_end_path = cache_root / "ref_end.png"
    if GOAL_IMAGE and os.path.exists(GOAL_IMAGE):
        shutil.copy(GOAL_IMAGE, ref_end_path)
    else:
        shutil.copy(cam_dirs["cam_high"] / f"frame_{indices[-1]:06d}.png", ref_end_path)

    samples = build_samples_json(
        run_root=run_root,
        task=TASK_INSTRUCTION,
        indices=indices,
        ref_end_path=str(ref_end_path),
        mode=eval_mode,
    )
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]

    with open(run_root / "sample.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    return samples


def build_messages(item: dict, include_images: bool = True) -> List[dict]:
    prompt_parts = SYSTEM_PROMPT.format(task=item["task"]).split("<image>")
    content = []
    for idx in range(8):
        content.append({"type": "text", "text": prompt_parts[idx]})
        image_entry = {"type": "image"}
        if include_images:
            image_entry["image"] = Image.open(item["image"][idx]).convert("RGB")
        content.append(image_entry)
    content.append({"type": "text", "text": prompt_parts[8]})
    return [{"role": "user", "content": content}]


def common_prefix_len(a: Iterable[int], b: Iterable[int]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


class QwenAttributor:
    def __init__(self, model_path: str):
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = MAX_PIXELS
            self.processor.image_processor.min_pixels = MIN_PIXELS

        print(f"Loading HF model from {model_path} ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.config.use_cache = False
        self.device = next(self.model.parameters()).device
        self.spatial_merge_size = int(self.config.vision_config.spatial_merge_size)

    def make_inputs(self, item: dict, suffix: str = "") -> Tuple[dict, int]:
        messages = build_messages(item, include_images=True)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + suffix

        images = [entry["image"] for entry in messages[0]["content"] if entry["type"] == "image"]
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            text=[full_text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        prefix_len = common_prefix_len(
            prompt_inputs["input_ids"][0].tolist(),
            full_inputs["input_ids"][0].tolist(),
        )
        if prefix_len == 0:
            prefix_len = prompt_inputs["input_ids"].shape[1]

        for key, value in full_inputs.items():
            if torch.is_tensor(value):
                full_inputs[key] = value.to(self.device)
        return full_inputs, prefix_len

    @torch.inference_mode()
    def generate_score(self, item: dict) -> str:
        inputs, prompt_len = self.make_inputs(item, suffix="")
        generated = self.model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            use_cache=False,
        )
        new_tokens = generated[0, prompt_len:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if "<score>" not in text:
            match = re.search(r"([+-]?\d+(?:\.\d+)?)%", text)
            if match:
                text = f"<score>{match.group(1)}%</score>"
        return text

    def image_token_slices(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor) -> List[slice]:
        positions = (input_ids[0] == self.config.image_token_id).nonzero(as_tuple=False).flatten()
        split_sizes = (
            image_grid_thw.prod(dim=-1) // (self.spatial_merge_size**2)
        ).tolist()

        slices = []
        offset = 0
        for size in split_sizes:
            start = int(positions[offset].item())
            end = int(positions[offset + size - 1].item()) + 1
            slices.append(slice(start, end))
            offset += size
        return slices

    def token_scores_to_maps(
        self,
        token_scores: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> List[np.ndarray]:
        maps = []
        offset = 0
        for grid in image_grid_thw.cpu().tolist():
            t, h, w = [int(x) for x in grid]
            mh = max(1, h // self.spatial_merge_size)
            mw = max(1, w // self.spatial_merge_size)
            size = t * mh * mw
            chunk = token_scores[offset : offset + size].float().reshape(t, mh, mw)
            heatmap = chunk.mean(dim=0).detach().cpu().numpy()
            maps.append(normalize_heatmap(heatmap))
            offset += size
        return maps

    def collect_attentions(self, inputs: dict) -> List[torch.Tensor]:
        captured = []
        handles = []

        attn_modules = [
            module
            for module in self.model.modules()
            if module.__class__.__name__ == "Qwen3VLTextAttention"
        ]

        def hook(_module, _args, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                captured.append(output[1].detach().float().cpu())

        for module in attn_modules[-ATTENTION_LAST_N_LAYERS:]:
            handles.append(module.register_forward_hook(hook))

        with torch.inference_mode():
            self.model(**inputs, output_attentions=True, use_cache=False, logits_to_keep=0)

        for handle in handles:
            handle.remove()

        return captured

    def fuse_attention_heads(self, attn: torch.Tensor) -> torch.Tensor:
        # attn: [heads, query_len, key_len]
        if HEAD_FUSION == "mean":
            return attn.mean(dim=0)
        if HEAD_FUSION == "max":
            return attn.max(dim=0).values
        if HEAD_FUSION == "min":
            return attn.min(dim=0).values
        raise ValueError(f"Unsupported HEAD_FUSION: {HEAD_FUSION}")

    def discard_low_attention(self, attn: torch.Tensor) -> torch.Tensor:
        if DISCARD_RATIO <= 0:
            return attn
        if not (0 <= DISCARD_RATIO < 1):
            raise ValueError(f"DISCARD_RATIO must be in [0, 1), got {DISCARD_RATIO}")

        flat = attn.reshape(-1)
        drop_count = int(flat.numel() * DISCARD_RATIO)
        if drop_count <= 0:
            return attn

        _, indices = torch.topk(flat, drop_count, largest=False)
        flat = flat.clone()
        flat[indices] = 0
        return flat.reshape_as(attn)

    def raw_attention_maps(self, inputs: dict, answer_start: int) -> List[np.ndarray]:
        captured = self.collect_attentions(inputs)
        image_slices = self.image_token_slices(inputs["input_ids"], inputs["image_grid_thw"])
        image_positions = torch.cat(
            [torch.arange(s.start, s.stop, dtype=torch.long) for s in image_slices]
        )
        answer_end = inputs["input_ids"].shape[1]
        query_positions = torch.arange(max(answer_start - 1, 0), max(answer_end - 1, answer_start))

        if not captured or len(query_positions) == 0:
            token_scores = torch.zeros(len(image_positions), dtype=torch.float32)
        else:
            layer_scores = []
            for attn in captured:
                # attn: [batch, heads, query_len, key_len]
                picked = attn[0][:, query_positions, :][:, :, image_positions]
                layer_scores.append(picked.mean(dim=(0, 1)))
            token_scores = torch.stack(layer_scores).mean(dim=0)

        return self.token_scores_to_maps(token_scores, inputs["image_grid_thw"])

    def rollout_attention_maps(self, inputs: dict, answer_start: int) -> List[np.ndarray]:
        captured = self.collect_attentions(inputs)
        image_slices = self.image_token_slices(inputs["input_ids"], inputs["image_grid_thw"])
        image_positions = torch.cat(
            [torch.arange(s.start, s.stop, dtype=torch.long) for s in image_slices]
        )
        answer_end = inputs["input_ids"].shape[1]
        query_positions = torch.arange(max(answer_start - 1, 0), max(answer_end - 1, answer_start))

        if not captured or len(query_positions) == 0:
            token_scores = torch.zeros(len(image_positions), dtype=torch.float32)
            return self.token_scores_to_maps(token_scores, inputs["image_grid_thw"])

        seq_len = inputs["input_ids"].shape[1]
        device = torch.device("cpu")
        rollout = torch.eye(seq_len, dtype=torch.float32, device=device)

        with torch.no_grad():
            for attn in captured:
                fused = self.fuse_attention_heads(attn[0].float().cpu())
                fused = self.discard_low_attention(fused)
                fused = fused + torch.eye(seq_len, dtype=fused.dtype, device=device)
                fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                rollout = torch.matmul(fused, rollout)

        token_scores = rollout[query_positions][:, image_positions].mean(dim=0)
        return self.token_scores_to_maps(token_scores, inputs["image_grid_thw"])

    def query_positions(self, inputs: dict, answer_start: int) -> torch.Tensor:
        answer_end = inputs["input_ids"].shape[1]
        return torch.arange(max(answer_start - 1, 0), max(answer_end - 1, answer_start))

    def raw_attention_scores_from_captured(
        self,
        captured: List[torch.Tensor],
        inputs: dict,
        answer_start: int,
    ) -> torch.Tensor:
        query_positions = self.query_positions(inputs, answer_start)
        seq_len = inputs["input_ids"].shape[1]

        if not captured or len(query_positions) == 0:
            return torch.zeros(seq_len, dtype=torch.float32)

        layer_scores = []
        for attn in captured:
            picked = attn[0][:, query_positions, :]
            layer_scores.append(picked.mean(dim=(0, 1)))
        return torch.stack(layer_scores).mean(dim=0)

    def rollout_attention_scores_from_captured(
        self,
        captured: List[torch.Tensor],
        inputs: dict,
        answer_start: int,
    ) -> torch.Tensor:
        query_positions = self.query_positions(inputs, answer_start)
        seq_len = inputs["input_ids"].shape[1]

        if not captured or len(query_positions) == 0:
            return torch.zeros(seq_len, dtype=torch.float32)

        device = torch.device("cpu")
        rollout = torch.eye(seq_len, dtype=torch.float32, device=device)
        with torch.no_grad():
            for attn in captured:
                fused = self.fuse_attention_heads(attn[0].float().cpu())
                fused = self.discard_low_attention(fused)
                fused = fused + torch.eye(seq_len, dtype=fused.dtype, device=device)
                fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                rollout = torch.matmul(fused, rollout)
        return rollout[query_positions][:, :].mean(dim=0)

    def select_attention_scores(
        self,
        raw_scores: torch.Tensor,
        rollout_scores: torch.Tensor,
    ) -> torch.Tensor:
        if ATTENTION_METHOD == "raw":
            return raw_scores
        if ATTENTION_METHOD == "rollout":
            return rollout_scores
        raise ValueError(f"Unsupported ATTENTION_METHOD: {ATTENTION_METHOD}")

    def token_group_distribution(
        self,
        token_scores: torch.Tensor,
        inputs: dict,
    ) -> Dict:
        token_scores = token_scores.detach().float().cpu()
        image_slices = self.image_token_slices(inputs["input_ids"], inputs["image_grid_thw"])

        total_mass = float(token_scores.sum().item())
        denom = total_mass if abs(total_mass) > 1e-12 else 1.0

        per_image = []
        image_total = 0.0
        for idx, image_slice in enumerate(image_slices):
            mass = float(token_scores[image_slice].sum().item())
            image_total += mass
            per_image.append(
                {
                    "index": idx,
                    "label": INPUT_IMAGE_LABELS[idx] if idx < len(INPUT_IMAGE_LABELS) else f"image_{idx}",
                    "token_count": image_slice.stop - image_slice.start,
                    "mass": mass,
                    "pct_total": mass / denom,
                }
            )

        image_denom = image_total if abs(image_total) > 1e-12 else 1.0
        for entry in per_image:
            entry["pct_image_total"] = entry["mass"] / image_denom

        def group_mass(indices: List[int]) -> float:
            return float(sum(per_image[i]["mass"] for i in indices if i < len(per_image)))

        reference_mass = group_mass([0, 1])
        before_mass = group_mass([2, 3, 4])
        after_mass = group_mass([5, 6, 7])
        non_image_mass = total_mass - image_total

        return {
            "total_mass": total_mass,
            "non_image": {
                "mass": non_image_mass,
                "pct_total": non_image_mass / denom,
                "note": "Includes text tokens, vision boundary tokens, special tokens, and previous answer tokens.",
            },
            "image_total": {
                "mass": image_total,
                "pct_total": image_total / denom,
            },
            "reference_total": {
                "mass": reference_mass,
                "pct_total": reference_mass / denom,
                "pct_image_total": reference_mass / image_denom,
            },
            "before_total": {
                "mass": before_mass,
                "pct_total": before_mass / denom,
                "pct_image_total": before_mass / image_denom,
            },
            "after_total": {
                "mass": after_mass,
                "pct_total": after_mass / denom,
                "pct_image_total": after_mass / image_denom,
            },
            "per_image": per_image,
        }

    def attention_diagnostics(
        self,
        selected_scores: torch.Tensor,
        raw_scores: torch.Tensor,
        rollout_scores: torch.Tensor,
        inputs: dict,
        answer_start: int,
    ) -> Dict:
        selected_distribution = self.token_group_distribution(selected_scores, inputs)
        raw_distribution = self.token_group_distribution(raw_scores, inputs)
        rollout_distribution = self.token_group_distribution(rollout_scores, inputs)
        answer_end = inputs["input_ids"].shape[1]

        return {
            "attention_method": ATTENTION_METHOD,
            "head_fusion": HEAD_FUSION if ATTENTION_METHOD == "rollout" else None,
            "discard_ratio": DISCARD_RATIO if ATTENTION_METHOD == "rollout" else None,
            "attention_last_n_layers": ATTENTION_LAST_N_LAYERS,
            "seq_len": int(inputs["input_ids"].shape[1]),
            "answer_start": int(answer_start),
            "answer_token_count": int(max(answer_end - answer_start, 0)),
            **selected_distribution,
            "selected_distribution_note": (
                "Top-level distribution matches the attention method used for the video heatmap."
            ),
            "raw_distribution": raw_distribution,
            "rollout_distribution": rollout_distribution,
        }

    def attention_maps(self, inputs: dict, answer_start: int) -> Tuple[List[np.ndarray], Dict]:
        captured = self.collect_attentions(inputs)
        raw_scores = self.raw_attention_scores_from_captured(captured, inputs, answer_start)
        rollout_scores = self.rollout_attention_scores_from_captured(captured, inputs, answer_start)
        token_scores = self.select_attention_scores(raw_scores, rollout_scores)
        image_slices = self.image_token_slices(inputs["input_ids"], inputs["image_grid_thw"])
        image_positions = torch.cat(
            [torch.arange(s.start, s.stop, dtype=torch.long) for s in image_slices]
        )
        maps = self.token_scores_to_maps(token_scores[image_positions], inputs["image_grid_thw"])
        diagnostics = self.attention_diagnostics(
            token_scores,
            raw_scores,
            rollout_scores,
            inputs,
            answer_start,
        )
        return maps, diagnostics

    def gradient_maps(self, inputs: dict, answer_start: int) -> List[np.ndarray]:
        self.model.zero_grad(set_to_none=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        image_grid_thw = inputs["image_grid_thw"]
        core = self.model.model

        inputs_embeds = core.get_input_embeddings()(input_ids)
        image_embeds, deepstack_image_embeds = core.get_image_features(
            inputs["pixel_values"], image_grid_thw
        )
        image_features = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_features.requires_grad_(True)
        image_features.retain_grad()

        image_mask, _ = core.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        visual_pos_masks = image_mask[..., 0]

        position_ids, rope_deltas = core.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask=attention_mask,
        )
        core.rope_deltas = rope_deltas

        outputs = core.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_image_embeds,
            use_cache=False,
        )

        answer_end = input_ids.shape[1]
        pred_positions = torch.arange(
            max(answer_start - 1, 0),
            max(answer_end - 1, answer_start),
            device=outputs.last_hidden_state.device,
        )
        target_positions = pred_positions + 1
        target_ids = input_ids[:, target_positions]

        selected_hidden = outputs.last_hidden_state[:, pred_positions, :]
        logits = self.model.lm_head(selected_hidden).float()
        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        )
        objective = token_log_probs.sum()
        objective.backward()

        grad = image_features.grad
        if grad is None:
            token_scores = torch.zeros(image_features.shape[0], dtype=torch.float32)
        else:
            token_scores = (grad.float() * image_features.detach().float()).sum(dim=-1)
            token_scores = torch.relu(token_scores)
            if float(token_scores.max().detach().cpu()) <= 0:
                token_scores = (grad.float() * image_features.detach().float()).sum(dim=-1).abs()

        maps = self.token_scores_to_maps(token_scores.detach().cpu(), image_grid_thw)
        self.model.zero_grad(set_to_none=True)
        return maps


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    heatmap -= float(heatmap.min())
    denom = float(heatmap.max())
    if denom > 1e-8:
        heatmap /= denom
    return heatmap


def overlay_heatmap(image_path: str, heatmap: np.ndarray, label: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 1.0 - OVERLAY_ALPHA, colored, OVERLAY_ALPHA, 0)
    cv2.putText(
        blended,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return blended


def make_video_frame(
    item: dict,
    attention_maps: List[np.ndarray],
    gradient_maps: List[np.ndarray],
) -> np.ndarray:
    top = []
    bottom = []
    for view_label, image_idx in zip(VIEW_LABELS, AFTER_IMAGE_INDICES):
        top.append(
            overlay_heatmap(
                item["image"][image_idx],
                attention_maps[image_idx],
                f"{ATTENTION_METHOD} attention | {view_label}",
            )
        )
        bottom.append(
            overlay_heatmap(
                item["image"][image_idx],
                gradient_maps[image_idx],
                f"gradient | {view_label}",
            )
        )

    top_row = np.hstack([cv2.resize(frame, (384, 288)) for frame in top])
    bottom_row = np.hstack([cv2.resize(frame, (384, 288)) for frame in bottom])
    frame = np.vstack([top_row, bottom_row])

    info = (
        f"score={item.get('pred', '').strip()}  "
        f"hop={float(item.get('hop', 0.0)):.2f}  "
        f"progress={float(item.get('progress', 0.0)):.2f}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        frame,
        info[:150],
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def write_video(frames: List[np.ndarray], out_path: Path) -> None:
    if not frames:
        print(f"[WARN] no frames to write: {out_path}")
        return
    ensure_dir(out_path.parent)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"Saved: {out_path}")


def load_external_predictions(mode: str) -> Optional[List[dict]]:
    pred_path = PRED_JSON_BY_MODE.get(mode)
    if not pred_path:
        return None
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if MAX_SAMPLES is not None:
        data = data[:MAX_SAMPLES]
    return data


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def print_attention_diagnostic(eval_mode: str, sample_idx: int, diagnostic: Dict) -> None:
    after = diagnostic["after_total"]["pct_total"]
    before = diagnostic["before_total"]["pct_total"]
    reference = diagnostic["reference_total"]["pct_total"]
    image = diagnostic["image_total"]["pct_total"]
    non_image = diagnostic["non_image"]["pct_total"]
    per_after = {
        entry["label"]: entry["pct_total"]
        for entry in diagnostic["per_image"]
        if entry["label"].startswith("after_")
    }
    after_parts = ", ".join(f"{key}={format_pct(value)}" for key, value in per_after.items())
    print(
        f"[{eval_mode} sample {sample_idx:04d}] attention mass: "
        f"image={format_pct(image)}, non_image={format_pct(non_image)}, "
        f"reference={format_pct(reference)}, before={format_pct(before)}, "
        f"after={format_pct(after)}"
        + (f" ({after_parts})" if after_parts else "")
    )


def run_mode(attributor: QwenAttributor, eval_mode: str, out_root: Path) -> None:
    run_root = out_root / f"{eval_mode}_mode_{sanitize_task(TASK_INSTRUCTION)}"
    ensure_dir(run_root)
    samples = prepare_samples(eval_mode, run_root)

    external = load_external_predictions(eval_mode)
    if external is not None:
        for sample, pred_item in zip(samples, external):
            sample["pred"] = pred_item.get("pred", "")
    else:
        print(f"[{eval_mode}] no external pred_vllm.json configured; generating scores with HF.")
        for sample in tqdm(samples, desc=f"{eval_mode} score"):
            sample["pred"] = attributor.generate_score(sample)

    samples = attach_progress(samples, eval_mode)
    with open(run_root / "pred_attribution.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    frames = []
    diagnostics = []
    heatmap_root = run_root / "heatmaps"
    ensure_dir(heatmap_root)

    for idx, item in enumerate(tqdm(samples, desc=f"{eval_mode} attribution")):
        pred = item.get("pred", "").strip()
        if not pred:
            pred = "<score>0%</score>"
        inputs, answer_start = attributor.make_inputs(item, suffix=pred)
        attention, attention_diagnostic = attributor.attention_maps(inputs, answer_start)
        gradient = attributor.gradient_maps(inputs, answer_start)
        attention_diagnostic["sample_index"] = idx
        attention_diagnostic["id"] = item.get("id")
        attention_diagnostic["pred"] = pred
        attention_diagnostic["hop"] = float(item.get("hop", 0.0))
        attention_diagnostic["progress"] = float(item.get("progress", 0.0))
        diagnostics.append(attention_diagnostic)
        print_attention_diagnostic(eval_mode, idx, attention_diagnostic)

        np.savez_compressed(
            heatmap_root / f"sample_{idx:04d}.npz",
            attention=np.array(attention, dtype=object),
            gradient=np.array(gradient, dtype=object),
            attention_method=ATTENTION_METHOD,
            head_fusion=HEAD_FUSION,
            discard_ratio=DISCARD_RATIO,
            attention_diagnostic=json.dumps(attention_diagnostic, ensure_ascii=False),
        )
        frames.append(make_video_frame(item, attention, gradient))

        del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_video(frames, run_root / "attribution_vis.mp4")
    with open(run_root / "attention_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    print(f"Saved: {run_root / 'attention_diagnostics.json'}")


def main() -> None:
    global MODEL_PATH
    global DATA_DIR
    global OUTPUT_ROOT
    global TASK_INSTRUCTION
    global GOAL_IMAGE
    global INTERVAL
    global EVAL_MODES
    global MAX_SAMPLES
    global MIN_PIXELS
    global MAX_PIXELS
    global PRED_JSON_BY_MODE
    global ATTENTION_METHOD
    global HEAD_FUSION
    global DISCARD_RATIO

    parser = argparse.ArgumentParser(
        description="Generate attention and gradient attribution videos for Robo-Dopamine GRM."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-root", default=OUTPUT_ROOT)
    parser.add_argument("--task", default=TASK_INSTRUCTION)
    parser.add_argument("--goal-image", default=GOAL_IMAGE)
    parser.add_argument("--interval", type=int, default=INTERVAL)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=EVAL_MODES,
        choices=["forward", "incremental", "backward"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples per mode for debugging. Default: process all samples.",
    )
    parser.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument(
        "--attention-method",
        default=ATTENTION_METHOD,
        choices=["rollout", "raw"],
        help="How to compute the top-row attention heatmap.",
    )
    parser.add_argument(
        "--head-fusion",
        default=HEAD_FUSION,
        choices=["mean", "max", "min"],
        help="How to fuse attention heads for rollout.",
    )
    parser.add_argument(
        "--discard-ratio",
        type=float,
        default=DISCARD_RATIO,
        help="Fraction of lowest attention entries to discard in rollout.",
    )
    parser.add_argument("--pred-json-forward", default=None)
    parser.add_argument("--pred-json-incremental", default=None)
    parser.add_argument("--pred-json-backward", default=None)
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATA_DIR = args.data_dir
    OUTPUT_ROOT = args.out_root
    TASK_INSTRUCTION = args.task
    GOAL_IMAGE = args.goal_image
    INTERVAL = args.interval
    EVAL_MODES = args.modes
    MAX_SAMPLES = args.max_samples
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    ATTENTION_METHOD = args.attention_method
    HEAD_FUSION = args.head_fusion
    DISCARD_RATIO = args.discard_ratio
    PRED_JSON_BY_MODE = {
        mode: path
        for mode, path in {
            "forward": args.pred_json_forward,
            "incremental": args.pred_json_incremental,
            "backward": args.pred_json_backward,
        }.items()
        if path
    }

    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    out_root = Path(OUTPUT_ROOT) / timestamp
    ensure_dir(out_root)

    attributor = QwenAttributor(MODEL_PATH)
    for mode in EVAL_MODES:
        run_mode(attributor, mode, out_root)

    print(f"Done. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
