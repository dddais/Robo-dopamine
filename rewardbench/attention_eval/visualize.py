"""Static endpoint visualizations for attention steering.

Unlike :mod:`video`, this module intentionally uses the audited endpoint bbox
already frozen in ``eligible.jsonl``.  It never loads a grounding backend or
opens an episode video: baseline, spatial-control, and target-steered GRM
forwards are run per selected evaluation episode solely to recover generation
attention heatmaps and final progress.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from ..io import append_jsonl, write_json
from ..protocol import progress
from .dataset import load_partition
from .masking import Head, matched_wrong_position_set
from .runtime import AttentionRuntime


def _fixed_episodes(rows: list[dict], count: int, seed: int) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['video_sha256']}:{row['example_id']}".encode()
        ).hexdigest(),
    )[:count]


def _line(image: np.ndarray, text: str, y: int, *, scale: float = 0.40) -> int:
    cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1)
    return y + 23


def _overlay(
    path: str,
    bbox: list[float],
    heatmap: list[float] | None,
    grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
    title: str,
    final_progress: float | None,
    selected_mass: float | None,
) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read endpoint image: {path}")
    height, width = image.shape[:2]
    values = np.asarray(heatmap or [], dtype=np.float32)
    grid_h = max(1, int(grid_thw[1]) // spatial_merge_size)
    grid_w = max(1, int(grid_thw[2]) // spatial_merge_size)
    if values.size != grid_h * grid_w:
        raise ValueError(f"Heatmap/grid mismatch: {values.size} vs {grid_h}x{grid_w}")
    grid = values.reshape(grid_h, grid_w)
    # Match ``visualize_stage3_head_attention.py``: normalize the displayed
    # grid, resize smoothly, apply JET at alpha=0.45, and draw the target in
    # lime.  The numeric selected-attention mass remains in the caption so
    # per-panel display normalization is not mistaken for a mass comparison.
    shifted = grid - np.nanmin(grid)
    denominator = float(np.nanmax(shifted))
    normalized = (
        np.zeros_like(shifted, dtype=np.uint8)
        if denominator <= 0
        else np.uint8(np.clip(shifted / denominator, 0, 1) * 255)
    )
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored = cv2.resize(colored, (width, height), interpolation=cv2.INTER_CUBIC)
    panel = cv2.addWeighted(image, 0.55, colored, 0.45, 0)
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.rectangle(panel, (0, 0), (width - 1, 30), (0, 0, 0), -1)
    caption = (
        f"{title} | progress={final_progress:.3f} | selected-attn={selected_mass:.4f}"
        if final_progress is not None and selected_mass is not None
        else title
    )
    cv2.putText(panel, caption, (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
    return panel


def _render(
    destination: Path,
    sample: dict,
    spans: list,
    runtime: AttentionRuntime,
    baseline: dict,
    control: dict | None,
    experiment: dict,
) -> None:
    # The GRM forward still uses its original eight-image input.  This static
    # artifact intentionally renders only the three after_cam_high heatmaps.
    span_by_label = {span.label: span for span in spans}
    after_span = span_by_label["after_cam_high"]
    bbox = sample["last"]["bbox"]
    target_span = after_span
    overlays = [
        _overlay(
            sample["last"]["provenance"]["image_path"], bbox, baseline.get("image_heatmap"),
            target_span.grid_thw, runtime.spatial_merge_size, "baseline",
            progress(baseline.get("signed_score")),
            baseline["hook_diagnostics"].get("bbox_attention_mass"),
        ),
    ]
    if control is not None:
        overlays.append(
            _overlay(
                sample["last"]["provenance"]["image_path"], bbox, control.get("image_heatmap"),
                target_span.grid_thw, runtime.spatial_merge_size,
                "spatial control (wrong-region bias)",
                progress(control.get("signed_score")),
                control["hook_diagnostics"].get("bbox_attention_mass"),
            )
        )
    else:
        unavailable = np.zeros((360, 320, 3), dtype=np.uint8)
        _line(unavailable, "spatial control unavailable", 32, scale=0.38)
        _line(unavailable, "no equal-size non-overlap", 55, scale=0.35)
        overlays.append(unavailable)
    overlays.append(
        _overlay(
            sample["last"]["provenance"]["image_path"], bbox, experiment.get("image_heatmap"),
            target_span.grid_thw, runtime.spatial_merge_size, "experiment (target-bbox bias)",
            progress(experiment.get("signed_score")),
            experiment["hook_diagnostics"].get("bbox_attention_mass"),
        )
    )
    overlays = [cv2.resize(item, (320, 240), interpolation=cv2.INTER_AREA) for item in overlays]
    header = np.zeros((64, 960, 3), dtype=np.uint8)
    y = _line(header, f"{sample['example_id']}  |  target: {sample['target_phrase']}", 21, scale=0.36)
    _line(header, f"task: {sample['task'][:140]}", y, scale=0.34)
    composite = np.vstack([header, np.hstack(overlays)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), composite):
        raise RuntimeError(f"Cannot write visualization: {destination}")


def run_visualize(
    run_dir: str | Path,
    *,
    episode_count: int = 12,
    seed: int | None = None,
    dry_run: bool = False,
) -> Path:
    """Render a fixed evaluation subset with no SAM3/DINO invocation."""
    if episode_count <= 0:
        raise ValueError("episode_count must be positive")
    run_dir = Path(run_dir).resolve()
    steering_manifest = json.loads((run_dir / "steering_manifest.json").read_text(encoding="utf-8"))
    attention = steering_manifest["config"]["attention_eval"]
    effective_seed = int(attention.get("seed", 20260724) if seed is None else seed)
    samples, split = load_partition(run_dir, "evaluation")
    selected = _fixed_episodes(samples, episode_count, effective_seed)
    ranking = json.loads(Path(attention["ranking_path"]).read_text(encoding="utf-8"))
    heads = [Head(int(row["layer"]), int(row["head"])) for row in ranking["ranking"][: int(attention.get("top_k", 8))]]
    tag = f"n{episode_count}_seed{effective_seed}"
    output_root = run_dir / "endpoint_visualizations" / tag
    records = run_dir / f"endpoint_visualizations_{tag}.jsonl"
    runtime = None if dry_run else AttentionRuntime(attention)
    completed = []
    for sample in selected:
        try:
            if dry_run:
                record = {"example_id": sample["example_id"], "status": "dry_run"}
            else:
                assert runtime is not None
                inputs, spans = runtime.prepare(sample)
                del inputs
                positions, image_positions, _ = runtime.target_positions(
                    sample, spans, attention.get("intervention_location", "after_cam_high")
                )
                baseline = runtime.generate(
                    sample, heads=heads, selected_positions=positions,
                    image_positions=image_positions, bias=0,
                )
                target_span = next(span for span in spans if span.label == "after_cam_high")
                wrong_positions = matched_wrong_position_set(
                    target_span, positions, spatial_merge_size=runtime.spatial_merge_size
                )
                control = (
                    runtime.generate(
                        sample, heads=heads, selected_positions=wrong_positions,
                        image_positions=image_positions, bias=float(attention.get("swap_bias", 6)),
                    )
                    if wrong_positions is not None
                    else None
                )
                experiment = runtime.generate(
                    sample, heads=heads, selected_positions=positions, image_positions=image_positions,
                    bias=float(attention.get("swap_bias", 6)),
                )
                image_path = output_root / sample["video_sha256"] / "endpoint_attention.png"
                _render(image_path, sample, spans, runtime, baseline, control, experiment)
                record = {
                    "example_id": sample["example_id"], "video_sha256": sample["video_sha256"],
                    "status": "ok", "image": str(image_path), "bbox": sample["last"]["bbox"],
                    "baseline_score": baseline["signed_score"],
                    "baseline_progress": progress(baseline["signed_score"]),
                    "control_score": control["signed_score"] if control else None,
                    "control_progress": progress(control["signed_score"]) if control else None,
                    "experiment_score": experiment["signed_score"],
                    "experiment_progress": progress(experiment["signed_score"]),
                    "baseline_bbox_mass": baseline["hook_diagnostics"].get("bbox_attention_mass"),
                    "control_bbox_mass": (
                        control["hook_diagnostics"].get("bbox_attention_mass") if control else None
                    ),
                    "experiment_bbox_mass": experiment["hook_diagnostics"].get("bbox_attention_mass"),
                    "token_grid": list(next(span for span in spans if span.label == "after_cam_high").grid_thw),
                }
            append_jsonl(records, record)
            completed.append(record)
        except Exception as exc:
            append_jsonl(records, {"example_id": sample["example_id"], "status": "invalid", "error": str(exc)})
    manifest = run_dir / f"endpoint_visualizations_{tag}.json"
    write_json(
        manifest,
        {
            "selection": "fixed_hash_order_before_effects",
            "requested_episode_count": episode_count,
            "seed": effective_seed,
            "evaluation_split_fingerprint": split["fingerprint"],
            "episodes": [sample["example_id"] for sample in selected],
            "records": str(records),
            "output_root": str(output_root),
            "grounding": "reused audited endpoint bbox from eligible.jsonl; no grounder invoked",
            "conditions": ["baseline", "candidate_wrong", "candidate_target"],
            "dry_run": dry_run,
        },
    )
    return manifest
