#!/usr/bin/env python3
"""GroundingDINO-only 3 data x 3 instruction video diagnostics.

This script intentionally does not load GRM.  It tests the same target-only
GroundingDINO path used by the attention experiments:

  task -> target phrase aliases -> candidate QA -> trajectory smoothing

For each visual data source and each override instruction it renders one video
over the full after_cam_high sample sequence with selected boxes and candidate
boxes overlaid.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from grounding import (
    GroundingBox,
    TaskGrounding,
    grounding_box_to_record,
    target_phrase_queries,
    task_to_target_phrase,
)


TASKS = {
    "carrot": "pick the carrot and put it on yellow plate",
    "cube": "pick the white cube and put it on yellow plate",
    "bottle": "pick the bottle and put it on yellow plate",
}

DEFAULT_DATA_ROOTS = {
    "carrot": "/home/dais/workspace/Robo-Dopamine/results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_1_carrot",
    "bottle": "/home/dais/workspace/Robo-Dopamine/results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_3_bottle",
    "cube": "/home/dais/workspace/Robo-Dopamine/results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_4_cube",
}

IMAGE_LABELS = [
    "reference_start",
    "reference_end",
    "before_cam_high",
    "before_cam_left_wrist",
    "before_cam_right_wrist",
    "after_cam_high",
    "after_cam_left_wrist",
    "after_cam_right_wrist",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def frame_id_from_sample(sample: dict, fallback: int) -> str:
    sample_id = str(sample.get("id", ""))
    match = re.search(r"af_(\d+)", sample_id)
    if match:
        return match.group(1)
    return f"{fallback:04d}"


def sample_path_for_data(root: Path, data_name: str, mode: str = "incremental") -> Path:
    task_text = TASKS[data_name].replace(" ", "_")
    search_roots = [root / "blank" / "inter20" / data_name, root / data_name, root]
    matches: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        matches.extend(search_root.glob(f"**/*{mode}_mode_{task_text}*/sample.json"))
    deduped = {str(p.resolve(strict=False)): p.resolve(strict=False) for p in matches}
    matches = sorted(deduped.values())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {mode} sample for data={data_name} under {root}, "
            f"got {len(matches)}: {matches}"
        )
    return matches[0]


def target_image_path(sample: dict, target_label: str) -> Optional[str]:
    idx = IMAGE_LABELS.index(target_label)
    images = sample.get("image", [])
    if idx >= len(images):
        return None
    path = Path(images[idx])
    return str(path) if path.exists() else None


def draw_text_block(cv2, image: np.ndarray, lines: Sequence[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    line_h = 18
    max_width = 0
    kept = []
    for line in lines:
        text = line if len(line) <= 110 else line[:107] + "..."
        kept.append(text)
        (tw, _th), _base = cv2.getTextSize(text, font, scale, thickness)
        max_width = max(max_width, tw)
    height = line_h * len(kept) + 8
    cv2.rectangle(image, (0, 0), (min(image.shape[1], max_width + 10), height), (0, 0, 0), -1)
    for i, text in enumerate(kept):
        cv2.putText(
            image,
            text,
            (5, 16 + i * line_h),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_box(
    cv2,
    image: np.ndarray,
    box: GroundingBox | dict,
    color: tuple[int, int, int],
    thickness: int,
    label: str,
) -> None:
    raw = box.bbox if isinstance(box, GroundingBox) else box.get("bbox")
    if raw is None:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in raw]
    h, w = image.shape[:2]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y0 = max(0, y1 - th - base - 4)
    cv2.rectangle(image, (x1, y0), (min(w - 1, x1 + tw + 6), y0 + th + base + 4), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 3, y0 + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def record_to_box(record: Optional[dict]) -> Optional[GroundingBox]:
    if record is None:
        return None
    return GroundingBox(
        label=str(record.get("label", "")),
        score=float(record.get("score", 0.0)),
        bbox=[float(x) for x in record.get("bbox", [])],
        query=str(record.get("query", "")),
        reject_reason=str(record.get("reject_reason", "")),
        quality=float(record.get("quality", 0.0)),
        source=str(record.get("source", "")),
    )


def render_frame(
    cv2,
    image_path: str,
    sample: dict,
    frame_idx: int,
    data_name: str,
    instruction_name: str,
    task: str,
    target_label: str,
    chosen: Optional[GroundingBox],
    candidates: Sequence[dict],
) -> np.ndarray:
    with Image.open(image_path).convert("RGB") as im:
        image = np.asarray(im).copy()
    # Draw non-selected candidates first, thin orange.
    for ci, cand in enumerate(candidates[:8]):
        cbox = record_to_box(cand)
        if cbox is None:
            continue
        label = f"cand{ci}:{cbox.query}:{cbox.quality:.2f}"
        draw_box(cv2, image, cbox, (255, 165, 0), 1, label)

    if chosen is not None:
        color = (0, 255, 0) if chosen.source == "detector" else (0, 255, 255)
        label = f"SELECT {chosen.query} q={chosen.quality:.2f} s={chosen.score:.2f} {chosen.source}"
        draw_box(cv2, image, chosen, color, 3, label)
    else:
        cv2.putText(
            image,
            "NO SELECTED BOX",
            (20, image.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    lines = [
        f"data={data_name} instr={instruction_name} view={target_label}",
        f"frame={frame_id_from_sample(sample, frame_idx)} idx={frame_idx}/{sample.get('id', '')}",
        f"target={task_to_target_phrase(task)} queries={target_phrase_queries(task)}",
        f"chosen={'none' if chosen is None else chosen.query + '/' + chosen.source}",
    ]
    draw_text_block(cv2, image, lines)
    return image


def write_video(cv2, out_path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError(f"No frames for {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def summarize_records(records: Sequence[dict]) -> dict:
    chosen = [r for r in records if r.get("chosen") is not None]
    by_source: dict[str, int] = {}
    by_query: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for r in chosen:
        c = r["chosen"]
        by_source[c.get("source", "")] = by_source.get(c.get("source", ""), 0) + 1
        by_query[c.get("query", "")] = by_query.get(c.get("query", ""), 0) + 1
        by_label[c.get("label", "")] = by_label.get(c.get("label", ""), 0) + 1
    return {
        "n_samples": len(records),
        "n_chosen": len(chosen),
        "n_missing": len(records) - len(chosen),
        "coverage": len(chosen) / max(1, len(records)),
        "source_counts": by_source,
        "query_counts": by_query,
        "label_counts": by_label,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Render GroundingDINO-only 3x3 bbox diagnostic videos")
    ap.add_argument("--output-root", default="results/attention/groundingdino_3data_3instruction_20260709")
    ap.add_argument("--grounding-model", default="../model/grounding-dino-base")
    ap.add_argument("--grounding-box-threshold", type=float, default=0.12)
    ap.add_argument("--text-threshold", type=float, default=0.20)
    ap.add_argument("--target-label", default="after_cam_high", choices=IMAGE_LABELS)
    ap.add_argument("--mode", default="incremental", choices=["forward", "incremental", "backward"])
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--save-png-frames", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--carrot-root", default=DEFAULT_DATA_ROOTS["carrot"])
    ap.add_argument("--bottle-root", default=DEFAULT_DATA_ROOTS["bottle"])
    ap.add_argument("--cube-root", default=DEFAULT_DATA_ROOTS["cube"])
    args = ap.parse_args()

    import cv2
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[grounding-3x3] loading GroundingDINO on {device}: {args.grounding_model}", flush=True)
    grounding = TaskGrounding(
        model_path=args.grounding_model,
        device=device,
        box_threshold=args.grounding_box_threshold,
        text_threshold=args.text_threshold,
    )

    roots = {
        "carrot": Path(args.carrot_root),
        "bottle": Path(args.bottle_root),
        "cube": Path(args.cube_root),
    }
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    manifest: dict = {
        "args": vars(args),
        "data_roots": {k: str(v) for k, v in roots.items()},
        "videos": [],
    }

    for data_name, root in roots.items():
        sample_json = sample_path_for_data(root, data_name, args.mode)
        samples_all = json.loads(sample_json.read_text())
        samples = list(samples_all)
        if args.max_samples is not None and args.max_samples > 0 and args.max_samples < len(samples):
            step = max(1, len(samples) // max(1, args.max_samples))
            samples = samples[::step][: args.max_samples]
        image_paths = [target_image_path(s, args.target_label) for s in samples]
        valid_pairs = [(i, s, p) for i, (s, p) in enumerate(zip(samples, image_paths)) if p is not None]

        for instr_name, task in TASKS.items():
            combo_name = f"data-{data_name}_instr-{instr_name}_{args.mode}_{args.target_label}"
            out_dir = out_root / combo_name
            out_dir.mkdir(parents=True, exist_ok=True)
            seq_json = out_dir / "bbox_sequence.json"
            print(
                f"[grounding-3x3] {combo_name}: samples={len(valid_pairs)} "
                f"task={task!r}",
                flush=True,
            )
            selected = grounding.ground_best_sequence(
                [p for _i, _s, p in valid_pairs],
                [task for _i, _s, _p in valid_pairs],
                target_only=True,
                write_json=seq_json,
            )
            seq_records = json.loads(seq_json.read_text())
            frames: list[np.ndarray] = []
            combo_records: list[dict] = []
            for local_i, ((original_i, sample, image_path), chosen_box, seq_rec) in enumerate(
                zip(valid_pairs, selected, seq_records)
            ):
                chosen_rec = grounding_box_to_record(chosen_box, image_path)
                candidates = seq_rec.get("candidates", [])
                frame = render_frame(
                    cv2,
                    image_path,
                    sample,
                    original_i,
                    data_name,
                    instr_name,
                    task,
                    args.target_label,
                    chosen_box,
                    candidates,
                )
                frames.append(frame)
                if local_i < max(0, int(args.save_png_frames)):
                    png_path = out_dir / "frames" / f"{local_i:03d}_{frame_id_from_sample(sample, original_i)}.png"
                    png_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame).save(png_path)

                row = {
                    "data": data_name,
                    "instruction": instr_name,
                    "sample_json": str(sample_json),
                    "sample_index": original_i,
                    "sample_id": sample.get("id"),
                    "frame_id": frame_id_from_sample(sample, original_i),
                    "image": image_path,
                    "task": task,
                    "target_phrase": task_to_target_phrase(task),
                    "queries": target_phrase_queries(task),
                    "chosen": chosen_rec,
                    "n_candidates": len(candidates),
                    "video_dir": str(out_dir),
                }
                combo_records.append(row)
                all_rows.append(row)

            video_path = out_dir / f"{combo_name}.mp4"
            write_video(cv2, video_path, frames, args.fps)
            summary = summarize_records(combo_records)
            combo_manifest = {
                "combo": combo_name,
                "data": data_name,
                "instruction": instr_name,
                "task": task,
                "sample_json": str(sample_json),
                "target_label": args.target_label,
                "video": str(video_path),
                "bbox_sequence": str(seq_json),
                "summary": summary,
                "records": combo_records,
            }
            (out_dir / "manifest.json").write_text(json.dumps(combo_manifest, indent=2))
            manifest["videos"].append({
                "combo": combo_name,
                "video": str(video_path),
                "manifest": str(out_dir / "manifest.json"),
                "bbox_sequence": str(seq_json),
                "summary": summary,
            })
            print(
                f"[grounding-3x3] wrote {video_path} "
                f"coverage={summary['n_chosen']}/{summary['n_samples']}",
                flush=True,
            )

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    flat_json = out_root / "per_frame_records.json"
    flat_json.write_text(json.dumps(all_rows, indent=2))

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "data",
            "instruction",
            "sample_index",
            "frame_id",
            "chosen_source",
            "chosen_query",
            "chosen_label",
            "chosen_score",
            "chosen_quality",
            "box_area_frac",
            "box_cx_frac",
            "box_cy_frac",
            "n_candidates",
            "image",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            chosen = row.get("chosen") or {}
            writer.writerow({
                "data": row["data"],
                "instruction": row["instruction"],
                "sample_index": row["sample_index"],
                "frame_id": row["frame_id"],
                "chosen_source": chosen.get("source", ""),
                "chosen_query": chosen.get("query", ""),
                "chosen_label": chosen.get("label", ""),
                "chosen_score": chosen.get("score", ""),
                "chosen_quality": chosen.get("quality", ""),
                "box_area_frac": chosen.get("box_area_frac", ""),
                "box_cx_frac": chosen.get("box_cx_frac", ""),
                "box_cy_frac": chosen.get("box_cy_frac", ""),
                "n_candidates": row["n_candidates"],
                "image": row["image"],
            })
    print(f"[grounding-3x3] wrote {out_root / 'manifest.json'}", flush=True)
    print(f"[grounding-3x3] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
