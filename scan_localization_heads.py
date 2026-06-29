"""
Scan per-(layer, head) attention to find localization heads in the GRM,
following Kang et al., "Your Large Vision-Language Model Only Needs A Few
Attention Heads For Visual Grounding" (CVPR 2025).

This script is intentionally separate from visualize_my_data_attribution.py:
that script aggregates heads (rollout / raw mean over heads) and therefore
collapses the per-head signal that Kang et al. show is the whole point.
Here we keep the head dimension and evaluate the two Kang criteria per head:

  Criterion 1 (attention sum S_img):  how much of the query token's attention
                                      lands on image tokens at all.
  Criterion 2 (spatial entropy H):    how focused (low entropy) the per-image
                                      attention map is, once binarized.

Outputs (mirroring Kang Fig. 3 and Fig. 6):
  - head_scan_Simg.png        sorted S_img per (layer, head)
  - head_scan_entropy.png     sorted spatial entropy per (layer, head)
  - head_scan_scatter.png     S_img vs. entropy scatter (upper-right = good)
  - head_scan.json            full per-head table
  - top_heads/<L{layer}_H{head}>_<image_label>.png
                              attention heatmap overlaid on the input image
                              for the top-K heads, so you can visually check
                              whether the head actually focuses on the target
                              object (the core question of Kang Fig. 1).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from examples.inference import (  # noqa: E402
    build_samples_json,
    ensure_dir,
    get_frame_count,
    make_sample_indices_by_interval,
    save_frames,
)
from visualize_my_data_attribution import (  # noqa: E402
    GOAL_IMAGE,
    INPUT_IMAGE_LABELS,
    INTERVAL,
    MAX_PIXELS,
    MIN_PIXELS,
    SYSTEM_PROMPT,
    TASK_INSTRUCTION,
    DATA_DIR,
    MODEL_PATH,
    OUTPUT_ROOT,
    QwenAttributor,
    prepare_samples,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Spatial entropy (Kang Criterion 2, Eq. 3 in the paper)
# ---------------------------------------------------------------------------

def spatial_entropy_2d(heatmap_2d: np.ndarray) -> float:
    """
    Binarize the (mh, mw) attention map by its mean, find 8-connected
    components, and return the Shannon entropy over component-size
    probabilities. Lower entropy = more focused.
    """
    flat = heatmap_2d.flatten()
    if flat.size == 0 or flat.max() <= 0:
        return 1.0  # treat empty / all-zero as maximally dispersed

    threshold = flat.mean()
    binary = (flat > threshold).astype(np.uint8).reshape(heatmap_2d.shape)
    if binary.sum() == 0:
        return 1.0

    num_labels, _labels, sizes, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    # background label 0 is also a component; the paper includes foreground
    # clusters. We follow Kang's spirit: focus on foreground clusters.
    if num_labels <= 1:
        return 1.0
    cluster_sizes = sizes[1:].astype(np.float64)  # drop background
    total = cluster_sizes.sum()
    if total <= 0:
        return 1.0
    probs = cluster_sizes / total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return entropy


# ---------------------------------------------------------------------------
# Head scan
# ---------------------------------------------------------------------------

class LocalizationHeadScanner(QwenAttributor):
    """Adds per-(layer, head) attention probing on top of QwenAttributor."""

    def collect_attentions_all_layers(self, inputs: dict) -> List[torch.Tensor]:
        """Hook every Qwen3VLTextAttention layer, not just the last N."""
        captured: List[torch.Tensor] = []
        handles = []

        attn_modules = [
            m for m in self.model.modules()
            if m.__class__.__name__ == "Qwen3VLTextAttention"
        ]

        def hook(_module, _args, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                captured.append(output[1].detach().float().cpu())

        for module in attn_modules:
            handles.append(module.register_forward_hook(hook))

        with torch.inference_mode():
            self.model(**inputs, output_attentions=True, use_cache=False, logits_to_keep=0)

        for handle in handles:
            handle.remove()

        # captured is in registration order; sort by module order just in case.
        # Module iteration in PyTorch is DFS pre-order, which matches layer
        # order for a flat decoder stack, so this is already correct.
        return captured

    @staticmethod
    def _query_position(answer_start: int) -> int:
        # Kang uses the last input text token as the representative query.
        # In the GRM teacher-forced setup, answer_start is the first generated
        # score token; the query we want is the token immediately before it.
        return max(answer_start - 1, 0)

    def per_head_image_attention(
        self,
        inputs: dict,
        answer_start: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return two lists indexed by image (length = number of input images).
        Each element is a (num_layers, num_heads) array:
          s_img[layer, head]   = sum of attention mass on that image's tokens
          entropy[layer, head] = spatial entropy on that image's attention map
        """
        captured = self.collect_attentions_all_layers(inputs)
        image_slices = self.image_token_slices(inputs["input_ids"], inputs["image_grid_thw"])
        q_pos = self._query_position(answer_start)

        num_images = len(image_slices)
        if not captured or num_images == 0:
            return [], []

        num_layers = len(captured)
        num_heads = captured[0].shape[1]

        s_img = [np.zeros((num_layers, num_heads), dtype=np.float64) for _ in range(num_images)]
        entropy = [np.zeros((num_layers, num_heads), dtype=np.float64) for _ in range(num_images)]

        spatial_merge = self.spatial_merge_size

        for layer_idx, attn in enumerate(captured):
            # attn: [batch=1, heads, query_len, key_len]
            row = attn[0][:, q_pos, :]  # [heads, key_len]
            for img_idx, img_slice in enumerate(image_slices):
                start, stop = img_slice.start, img_slice.stop
                block = row[:, start:stop]  # [heads, num_image_tokens]
                s_img[img_idx][layer_idx] = block.sum(dim=-1).numpy()

                # reshape each head's attention to spatial grid for entropy
                grid = inputs["image_grid_thw"][img_idx].cpu().tolist()
                t, h, w = [int(x) for x in grid]
                mh = max(1, h // spatial_merge)
                mw = max(1, w // spatial_merge)
                expected = t * mh * mw
                if block.shape[1] != expected:
                    # mismatch (e.g. temporal > 1); skip entropy for this image
                    continue
                for head_idx in range(num_heads):
                    heatmap_2d = block[head_idx].reshape(mh, mw).numpy()
                    entropy[img_idx][layer_idx, head_idx] = spatial_entropy_2d(heatmap_2d)

        return s_img, entropy

    def top_head_attention_map(
        self,
        inputs: dict,
        answer_start: int,
        layer_idx: int,
        head_idx: int,
        image_idx: int,
    ) -> np.ndarray:
        """Return the (mh, mw) attention map for one (layer, head, image)."""
        captured = self.collect_attentions_all_layers(inputs)
        image_slices = self.image_token_slices(inputs["input_ids"], inputs["image_grid_thw"])
        q_pos = self._query_position(answer_start)

        attn = captured[layer_idx][0, head_idx, q_pos, :]  # [key_len]
        start, stop = image_slices[image_idx].start, image_slices[image_idx].stop
        block = attn[start:stop]

        grid = inputs["image_grid_thw"][image_idx].cpu().tolist()
        t, h, w = [int(x) for x in grid]
        mh = max(1, h // self.spatial_merge_size)
        mw = max(1, w // self.spatial_merge_size)
        return block.reshape(mh, mw).float().numpy()


# ---------------------------------------------------------------------------
# Plotting (Kang Fig. 3 / Fig. 6 style)
# ---------------------------------------------------------------------------

def plot_sorted_curve(values_2d: np.ndarray, ylabel: str, title: str, out_path: Path):
    """values_2d: (num_layers, num_heads). Sort all (layer, head) cells ascending."""
    flat = values_2d.flatten()
    order = np.argsort(flat)
    sorted_vals = flat[order]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sorted_vals, "-", color="#888", linewidth=0.8)
    ax.scatter(np.arange(len(sorted_vals)), sorted_vals, s=6, c="#1f77b4")
    ax.set_xlabel("(layer, head) cells sorted ascending")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_scatter(s_img_2d: np.ndarray, entropy_2d: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sx = s_img_2d.flatten()
    sy = entropy_2d.flatten()
    sc = ax.scatter(sx, sy, s=8, c=np.arange(len(sx)), cmap="viridis")
    ax.set_xlabel("Attention sum S_img (higher = looks at image more)")
    ax.set_ylabel("Spatial entropy H (lower = more focused)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="cell index (arbitrary)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def overlay_attention_on_image(image_path: str, heatmap_2d: np.ndarray, out_path: Path, alpha: float = 0.45):
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    heat_resized = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_norm = heat_resized - heat_resized.min()
    if heat_norm.max() > 0:
        heat_norm = heat_norm / heat_norm.max()
    heat_uint8 = (heat_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heat_color, alpha, 0)
    cv2.imwrite(str(out_path), overlay)


# ---------------------------------------------------------------------------
# Main scan pipeline
# ---------------------------------------------------------------------------

def run_scan(
    model_path: str,
    data_dir: str,
    out_root: str,
    task: str,
    goal_image: str,
    interval: int,
    eval_mode: str,
    max_samples: int,
    top_k: int,
    focus_image_labels: Optional[List[str]],
):
    out_root = Path(out_root)
    ensure_dir(out_root)

    scanner = LocalizationHeadScanner(model_path)

    # Override globals used by prepare_samples
    import visualize_my_data_attribution as vmd
    vmd.DATA_DIR = data_dir
    vmd.TASK_INSTRUCTION = task
    vmd.GOAL_IMAGE = goal_image
    vmd.INTERVAL = interval

    samples = prepare_samples(eval_mode, out_root)
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    print(f"[scan] evaluating {len(samples)} samples in {eval_mode} mode")

    # Aggregate per-head stats across samples
    s_img_accum = None  # list per image, each (layers, heads)
    entropy_accum = None
    count = 0

    # Remember the first sample's image paths and grid for visualization
    first_inputs = None
    first_answer_start = None
    first_item = samples[0] if samples else None

    for i, item in enumerate(samples):
        suffix = "<score>0%</score>"  # teacher force a canonical score token
        inputs, answer_start = scanner.make_inputs(item, suffix=suffix)
        s_img_list, entropy_list = scanner.per_head_image_attention(inputs, answer_start)
        if not s_img_list:
            print(f"[scan] sample {i}: no attention captured, skip")
            continue

        if s_img_accum is None:
            s_img_accum = [np.zeros_like(m) for m in s_img_list]
            entropy_accum = [np.zeros_like(m) for m in entropy_list]

        for img_idx in range(len(s_img_list)):
            s_img_accum[img_idx] += s_img_list[img_idx]
            entropy_accum[img_idx] += entropy_list[img_idx]
        count += 1

        if i == 0:
            first_inputs = inputs
            first_answer_start = answer_start

    if count == 0:
        print("[scan] no valid samples, abort.")
        return

    s_img_mean = [m / count for m in s_img_accum]
    entropy_mean = [m / count for m in entropy_accum]

    # Save per-image tables + figures
    summary = {
        "model_path": model_path,
        "data_dir": data_dir,
        "task": task,
        "eval_mode": eval_mode,
        "num_samples": count,
        "num_images": len(s_img_mean),
        "focus_image_labels": focus_image_labels,
        "per_image": [],
    }

    num_layers, num_heads = s_img_mean[0].shape

    # Decide which images to report on: default to the 3 "after" views, which
    # is what the GRM is most directly judging in forward/incremental modes.
    if focus_image_labels is None:
        focus_image_labels = [
            "after_cam_high",
            "after_cam_left_wrist",
            "after_cam_right_wrist",
        ]
    focus_indices = [
        i for i, lbl in enumerate(INPUT_IMAGE_LABELS) if lbl in focus_image_labels
    ]
    if not focus_indices:
        focus_indices = list(range(len(s_img_mean)))

    # Aggregate over focus images for a single ranking
    s_img_focus = np.zeros_like(s_img_mean[0])
    entropy_focus = np.zeros_like(entropy_mean[0])
    for idx in focus_indices:
        s_img_focus += s_img_mean[idx]
        entropy_focus += entropy_mean[idx]
    s_img_focus /= len(focus_indices)
    entropy_focus /= len(focus_indices)

    # Kang Criterion 1 threshold: maximum curvature on the sorted S_img curve.
    # We don't need the exact tau for ranking; we just report the curve and
    # the top-K heads by S_img, then among those pick lowest entropy.
    flat_s = s_img_focus.flatten()
    flat_h = entropy_focus.flatten()

    # Rank: high S_img AND low entropy. Use a simple product score
    # (normalized S_img) * (1 - normalized entropy) for a single ranking.
    s_norm = (flat_s - flat_s.min()) / (flat_s.max() - flat_s.min() + 1e-12)
    h_norm = (flat_h - flat_h.min()) / (flat_h.max() - flat_h.min() + 1e-12)
    score = s_norm * (1.0 - h_norm)

    ranking = np.argsort(-score)  # descending
    top_cells = []
    for cell in ranking[:top_k * 3]:  # show a few extra in case of ties / dups
        layer = int(cell // num_heads)
        head = int(cell % num_heads)
        top_cells.append({
            "layer": layer,
            "head": head,
            "label": f"L{layer}_H{head}",
            "s_img": float(s_img_focus[layer, head]),
            "entropy": float(entropy_focus[layer, head]),
            "score": float(score[cell]),
        })

    summary["top_heads"] = top_cells[:top_k]
    summary["all_top_seen"] = top_cells

    for img_idx in range(len(s_img_mean)):
        summary["per_image"].append({
            "label": INPUT_IMAGE_LABELS[img_idx] if img_idx < len(INPUT_IMAGE_LABELS) else f"image_{img_idx}",
            "s_img_mean_table": s_img_mean[img_idx].tolist(),
            "entropy_mean_table": entropy_mean[img_idx].tolist(),
        })

    with open(out_root / "head_scan.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Kang Fig. 3 style: sorted S_img curve over focus images
    plot_sorted_curve(
        s_img_focus,
        ylabel="S_img (attention sum on image)",
        title=f"S_img per (layer, head), mean over focus images\n{task} | {eval_mode} | n={count}",
        out_path=out_root / "head_scan_Simg.png",
    )
    plot_sorted_curve(
        entropy_focus,
        ylabel="Spatial entropy H (lower = more focused)",
        title=f"Spatial entropy per (layer, head), mean over focus images\n{task} | {eval_mode} | n={count}",
        out_path=out_root / "head_scan_entropy.png",
    )
    plot_scatter(
        s_img_focus,
        entropy_focus,
        title=f"S_img vs. entropy (upper-right corner = localization head candidate)\n{task} | {eval_mode}",
        out_path=out_root / "head_scan_scatter.png",
    )

    # Overlay the top-K heads' attention on the first sample's images.
    if first_item is not None and top_cells:
        top_dir = out_root / "top_heads"
        ensure_dir(top_dir)
        print(f"[scan] rendering top-{top_k} head overlays into {top_dir}")
        for cell in top_cells[:top_k]:
            layer = cell["layer"]
            head = cell["head"]
            for img_idx in focus_indices:
                try:
                    hmap = scanner.top_head_attention_map(
                        first_inputs, first_answer_start, layer, head, img_idx
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[scan] render fail L{layer}_H{head} img{img_idx}: {exc}")
                    continue
                img_label = INPUT_IMAGE_LABELS[img_idx] if img_idx < len(INPUT_IMAGE_LABELS) else f"image_{img_idx}"
                out_png = top_dir / f"L{layer}_H{head}_{img_label}.png"
                overlay_attention_on_image(first_item["image"][img_idx], hmap, out_png)

    print(f"[scan] done. results in {out_root}")
    print(f"[scan] top heads:")
    for c in summary["top_heads"]:
        print(f"  {c['label']}: S_img={c['s_img']:.4f}, entropy={c['entropy']:.4f}, score={c['score']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-root", default=OUTPUT_ROOT)
    parser.add_argument("--task", default=TASK_INSTRUCTION)
    parser.add_argument("--goal-image", default=GOAL_IMAGE)
    parser.add_argument("--interval", type=int, default=INTERVAL)
    parser.add_argument("--eval-mode", default="forward",
                        choices=["forward", "incremental", "backward"])
    parser.add_argument("--max-samples", type=int, default=5,
                        help="How many samples to average the head stats over.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many top heads to render overlays for.")
    parser.add_argument(
        "--focus-image-labels", nargs="*", default=None,
        help="Which input images to compute S_img/entropy on. "
             "Default: the 3 after views. Options: " + ", ".join(INPUT_IMAGE_LABELS),
    )
    args = parser.parse_args()

    run_scan(
        model_path=args.model_path,
        data_dir=args.data_dir,
        out_root=args.out_root,
        task=args.task,
        goal_image=args.goal_image,
        interval=args.interval,
        eval_mode=args.eval_mode,
        max_samples=args.max_samples,
        top_k=args.top_k,
        focus_image_labels=args.focus_image_labels,
    )


if __name__ == "__main__":
    main()
