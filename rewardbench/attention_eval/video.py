from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..grounding.dino import GroundingDINOGrounder
from ..grounding.parser import build_queries
from ..grounding.sam3 import SAM3Grounder
from ..io import append_jsonl, read_jsonl, write_json
from ..schemas import TargetSpec
from .dataset import load_partition
from .masking import Head
from .runtime import AttentionRuntime


def _fixed_episodes(rows: list[dict], count: int, seed: int) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['video_sha256']}:{row['example_id']}".encode()
        ).hexdigest(),
    )[:count]


def _sample_video(video_path: str, output_dir: Path) -> list[tuple[int, float, str]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps))
    indices = list(range(0, frame_count, step))
    if frame_count and (not indices or indices[-1] != frame_count - 1):
        indices.append(frame_count - 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        path = output_dir / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(path), frame)
        rows.append((index, index / fps, str(path.resolve())))
    cap.release()
    return rows


def run_video(
    run_dir: str | Path,
    *,
    dry_run: bool = False,
    episode_count: int | None = None,
    seed: int | None = None,
) -> Path:
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "steering_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    attention = config["attention_eval"]
    effective_count = int(
        attention.get("video_episode_count", 12) if episode_count is None else episode_count
    )
    effective_seed = int(attention.get("seed", 20260724) if seed is None else seed)
    if effective_count <= 0:
        raise ValueError("episode_count must be positive")
    samples, split = load_partition(run_dir, "evaluation")
    selected = _fixed_episodes(
        samples,
        effective_count,
        effective_seed,
    )
    ranking_data = json.loads(
        Path(attention.get("ranking_path", run_dir / "consensus_ranking.json")).read_text(
            encoding="utf-8"
        )
    )
    heads = [
        Head(int(row["layer"]), int(row["head"]))
        for row in ranking_data["ranking"][: int(attention.get("top_k", 8))]
    ]
    backend_name = Path(attention["grounding_run"]).name
    grounder = (
        GroundingDINOGrounder(config["grounding_dino"])
        if backend_name == "grounding_dino"
        else SAM3Grounder(config["sam3"])
    )
    runtime = None if dry_run else AttentionRuntime(attention)
    targets = {
        row["example_id"]: row
        for row in read_jsonl(Path(attention["grounding_run"]).parent / "targets.jsonl")
    }
    # Overrides are intentionally isolated from the default video artifact so an
    # exploratory sample cannot overwrite the run-configured selection.
    selection_tag = f"n{effective_count}_seed{effective_seed}"
    using_run_default = episode_count is None and seed is None
    video_root = run_dir / "video" if using_run_default else run_dir / "video_samples" / selection_tag
    frame_records = (
        run_dir / "video_frames.jsonl"
        if using_run_default
        else run_dir / f"video_frames_{selection_tag}.jsonl"
    )
    videos = []
    for sample in selected:
        target_row = dict(targets[sample["example_id"]])
        target_row.pop("schema_version", None)
        target_row["attributes"] = tuple(target_row.get("attributes", []))
        target_row["targets"] = tuple(target_row.get("targets", []))
        target = TargetSpec(**target_row)
        episode_dir = video_root / sample["video_sha256"]
        frames = _sample_video(sample["video_path"], episode_dir / "frames")
        episode_rows = []
        previous_box = None
        sam3_tracks = {}
        if backend_name == "sam3" and not dry_run and frames:
            anchor_rows = [
                frames[index]
                for index in sorted(
                    {round(value * (len(frames) - 1) / 4) for value in range(5)}
                )
            ]
            anchors = []
            for frame_index, _, image_path in anchor_rows:
                candidates = grounder.candidates(image_path, build_queries(target))
                selected_anchor = grounder.select(
                    image_path, candidates, len(build_queries(target))
                )
                if selected_anchor is not None:
                    anchors.append((selected_anchor["score"], frame_index))
            if anchors:
                best_anchor = max(anchors)[1]
                sam3_tracks = _normalize_sam3_tracks(
                    grounder.track(
                        sample["video_path"], target.target_phrase, [best_anchor]
                    )
                )
        for frame_index, timestamp, image_path in frames:
            if dry_run:
                selected_box = {
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "score": 0.0,
                    "query": target.target_phrase,
                }
            else:
                if backend_name == "sam3":
                    selected_box = sam3_tracks.get(frame_index)
                else:
                    candidates = grounder.candidates(image_path, build_queries(target))
                    selected_box = _associate_dino(
                        image_path,
                        candidates,
                        previous_box,
                        grounder,
                        len(build_queries(target)),
                    )
            if selected_box is None:
                append_jsonl(
                    frame_records,
                    {
                        "example_id": sample["example_id"],
                        "video_sha256": sample["video_sha256"],
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "status": "no_detection",
                    },
                )
                continue
            previous_box = selected_box
            dynamic = dict(sample)
            dynamic["last"] = {
                "bbox": selected_box["bbox"],
                "provenance": {"image_path": image_path},
            }
            if dry_run:
                baseline = steered = {
                    "signed_score": 0.0,
                    "image_heatmap": None,
                    "hook_diagnostics": {"dry_run": True},
                }
                bbox_positions = []
                image_positions = []
                spans = []
            else:
                assert runtime is not None
                inputs, spans_all = runtime.prepare(dynamic)
                del inputs
                bbox_positions, image_positions, spans = runtime.target_positions(
                    dynamic, spans_all, "after_cam_high"
                )
                baseline = runtime.generate(
                    dynamic,
                    heads=heads,
                    selected_positions=bbox_positions,
                    image_positions=image_positions,
                    bias=0,
                )
                steered = runtime.generate(
                    dynamic,
                    heads=heads,
                    selected_positions=bbox_positions,
                    image_positions=image_positions,
                    bias=float(attention.get("swap_bias", 6)),
                )
            row = {
                "example_id": sample["example_id"],
                "video_sha256": sample["video_sha256"],
                "frame_index": frame_index,
                "timestamp": timestamp,
                "image_path": image_path,
                "bbox": selected_box["bbox"],
                "grounding_score": selected_box["score"],
                "bbox_positions": bbox_positions,
                "token_grid": list(spans[0].grid_thw) if spans else None,
                "baseline_score": baseline["signed_score"],
                "steered_score": steered["signed_score"],
                "baseline_heatmap": baseline.get("image_heatmap"),
                "steered_heatmap": steered.get("image_heatmap"),
                "status": "dry_run" if dry_run else "ok",
            }
            append_jsonl(frame_records, row)
            episode_rows.append(row)
        if not dry_run and episode_rows:
            video_path = _render_episode(episode_rows, episode_dir / "attention.mp4")
            videos.append(str(video_path))
    output = (
        run_dir / "attention_video_manifest.json"
        if using_run_default
        else run_dir / f"attention_video_manifest_{selection_tag}.json"
    )
    write_json(
        output,
        {
            "selection": "fixed_hash_order_before_effects",
            "requested_episode_count": effective_count,
            "seed": effective_seed,
            "evaluation_split_fingerprint": split["fingerprint"],
            "episodes": [sample["example_id"] for sample in selected],
            "videos": videos,
            "frames_jsonl": str(frame_records),
            "fps_sampling": 1,
            "common_color_scale_per_episode": True,
        },
    )
    return output


def _iou(first, second) -> float:
    a, b = first["bbox"], second["bbox"]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - intersection
    )
    return intersection / union if union else 0.0


def _associate_dino(image_path, candidates, previous, grounder, query_count):
    if previous is None:
        return grounder.select(image_path, candidates, query_count)
    with Image.open(image_path) as image:
        width, height = image.size
    legal = []
    from ..schemas import validate_bbox

    for candidate in candidates:
        try:
            candidate["bbox"] = list(validate_bbox(candidate["bbox"], width, height))
        except ValueError:
            continue
        same_query = float(candidate.get("query") == previous.get("query"))
        association = (
            0.65 * float(candidate["score"])
            + 0.25 * _iou(candidate, previous)
            + 0.10 * same_query
        )
        legal.append((association, candidate))
    return max(legal, key=lambda value: value[0])[1] if legal else None


def _normalize_sam3_tracks(outputs) -> dict[int, dict]:
    """Normalize common official predictor output layouts to frame→best instance."""
    if isinstance(outputs, dict):
        outputs = outputs.get("frames") or outputs.get("outputs") or [outputs]
    tracks = {}
    for row in outputs or []:
        if not isinstance(row, dict):
            continue
        frame_index = row.get("frame_index")
        if frame_index is None:
            continue
        boxes = row.get("boxes")
        if boxes is None:
            boxes = row.get("bbox")
        scores = row.get("scores")
        if scores is None:
            scores = row.get("score")
        if boxes is None:
            continue
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().tolist()
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().tolist()
        if boxes and isinstance(boxes[0], (int, float)):
            boxes = [boxes]
        if isinstance(scores, (int, float)):
            scores = [scores]
        if scores is None or len(scores) == 0:
            scores = [1.0] * len(boxes)
        best = max(range(len(boxes)), key=lambda index: float(scores[index]))
        tracks[int(frame_index)] = {
            "bbox": [float(value) for value in boxes[best]],
            "score": float(scores[best]),
            "query": row.get("query", "sam3_video_track"),
        }
    return tracks


def _render_episode(rows: list[dict], destination: Path) -> Path:
    all_values = [
        value
        for row in rows
        for key in ("baseline_heatmap", "steered_heatmap")
        for value in (row.get(key) or [])
    ]
    maximum = max(all_values, default=1.0) or 1.0
    panels = []
    baseline_curve, steered_curve = [], []
    for row in rows:
        frame = cv2.imread(row["image_path"])
        if frame is None:
            continue
        x1, y1, x2, y2 = map(int, row["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        height, width = frame.shape[:2]
        heatmaps = []
        for key in ("baseline_heatmap", "steered_heatmap"):
            values = np.asarray(row.get(key) or [0.0], dtype=np.float32)
            token_grid = row.get("token_grid")
            if token_grid and len(token_grid) == 3:
                grid_h = max(1, int(token_grid[1]) // 2)
                grid_w = max(1, int(token_grid[2]) // 2)
            else:
                grid_w = max(1, int(round(np.sqrt(values.size))))
                while values.size % grid_w:
                    grid_w -= 1
                grid_h = values.size // grid_w
            if grid_h * grid_w != values.size:
                raise ValueError(
                    f"Heatmap/grid mismatch: {values.size} vs {grid_h}x{grid_w}"
                )
            grid = values.reshape(grid_h, grid_w)
            normalized = np.clip(grid / maximum * 255, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            heatmaps.append(cv2.resize(colored, (width, height), interpolation=cv2.INTER_NEAREST))
        baseline_curve.append(float(row["baseline_score"]))
        steered_curve.append(float(row["steered_score"]))
        plot = np.full((height, width, 3), 255, dtype=np.uint8)
        for curve, color in ((baseline_curve, (80, 80, 80)), (steered_curve, (0, 0, 255))):
            if len(curve) > 1:
                points = [
                    (
                        int(index * (width - 1) / max(1, len(rows) - 1)),
                        int((1 - (value + 1) / 2) * (height - 1)),
                    )
                    for index, value in enumerate(curve)
                ]
                cv2.polylines(plot, [np.asarray(points)], False, color, 2)
        panel = np.hstack([frame, heatmaps[0], heatmaps[1], plot])
        panels.append(panel)
    if not panels:
        raise RuntimeError("No renderable video frames")
    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = panels[0].shape[:2]
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height)
    )
    for panel in panels:
        writer.write(panel)
    writer.release()
    return destination
