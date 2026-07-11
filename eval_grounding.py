"""Standalone GroundingDINO diagnostic for GRM task images.

Goal: see how well GroundingDINO grounds the task-relevant nouns on real GRM
frames, before wiring it into the steering pipeline. For each sample it runs
detection on a configurable subset of the 8 GRM views (default: after_cam_high,
after_cam_left_wrist, after_cam_right_wrist — the views the steering experiment
targets), draws the boxes on the frames, and writes:

  results/grounding_eval/<task_slug>__<view>/frame_NN.png   visualizations
  results/grounding_eval/summary.json                       per-frame detection records
  results/grounding_eval/summary.csv                        flat table for quick scan

Usage:
    python eval_grounding.py \
        --sample-json ./results/white_cube_inter20/.../sample.json \
        --sample-json ./results/pick_3_fail/.../sample.json \
        --views after_cam_high after_cam_left_wrist after_cam_right_wrist \
        --max-samples 3 \
        --box-threshold 0.25

The script does NOT load GRM — it only needs GroundingDINO + PIL + opencv, so it
is fast (a few seconds per image) and decouples detector QA from the LVLM stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from grounding import TaskGrounding, task_to_phrases, task_to_target_phrase

# Canonical GRM view order used by scan_localization_heads_best.IMAGE_LABELS.
# Kept local to avoid importing the heavy scan module just for a constant.
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


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "task"


def draw_boxes(image_path: str, boxes, out_path: Path) -> None:
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    # Distinct colors per label so multi-object detections are readable.
    palette = [
        (0, 255, 0),    # green
        (0, 165, 255),  # orange
        (255, 0, 0),    # blue
        (0, 0, 255),    # red
        (255, 255, 0),  # cyan
    ]
    color_by_label: Dict[str, tuple] = {}

    for b in boxes:
        color = color_by_label.setdefault(b.label, palette[len(color_by_label) % len(palette)])
        x1, y1, x2, y2 = [int(round(v)) for v in b.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_txt = f"{b.label} {b.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(image, (x1, y_top), (x1 + tw + 6, y_top + th + baseline + 4), color, -1)
        cv2.putText(image, label_txt, (x1 + 3, y_top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Header with the derived phrase query so it is obvious what was asked.
    cv2.imwrite(str(out_path), image)


def main():
    ap = argparse.ArgumentParser(description="Standalone GroundingDINO diagnostic on GRM frames")
    ap.add_argument("--sample-json", action="append", required=True, help="Can be repeated to mix tasks")
    ap.add_argument("--grounding-model", default="../model/grounding-dino-base")
    ap.add_argument("--box-threshold", type=float, default=0.25)
    ap.add_argument("--text-threshold", type=float, default=0.20)
    ap.add_argument("--views", nargs="+", default=["after_cam_high", "after_cam_left_wrist", "after_cam_right_wrist"],
                    choices=IMAGE_LABELS, help="Which GRM views to evaluate")
    ap.add_argument("--max-samples", type=int, default=None, help="Cap number of samples per sample-json")
    ap.add_argument("--phrase-query", default=None, help="Override the auto-derived phrase (e.g. 'white cube . plate')")
    ap.add_argument("--all-task-phrases", action="store_true",
                    help="Use all derived task noun phrases. Default uses only the primary target object, "
                         "matching steering/ranking.")
    ap.add_argument("--best-only", action="store_true",
                    help="Only draw/store the single highest-score box per image (matches what steer_grm_heads uses). "
                         "Default draws all detections so detector quality (overlapping boxes, false positives) is visible.")
    ap.add_argument("--output", default="./results/grounding_eval")
    args = ap.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[grounding-eval] loading GroundingDINO from {args.grounding_model}")
    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    grounding = TaskGrounding(
        model_path=args.grounding_model,
        device=device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    all_records: List[dict] = []
    sample_idx_global = 0

    for sj_path in args.sample_json:
        samples = json.loads(Path(sj_path).read_text())
        if args.max_samples is not None:
            # Spread samples across the episode instead of taking the first N,
            # which are all near-identical early frames.
            step = max(1, len(samples) // max(1, args.max_samples))
            samples = samples[::step][: args.max_samples]

        task_slug = ""
        for sample in samples:
            task = sample["task"]
            task_slug = slugify(task)
            if args.phrase_query is not None:
                phrase = args.phrase_query
            elif args.all_task_phrases:
                phrase = task_to_phrases(task)
            else:
                phrase = task_to_target_phrase(task)
            print(f"[grounding-eval] sample {sample_idx_global} task={task!r} phrase={phrase!r}")

            for view in args.views:
                idx = IMAGE_LABELS.index(view)
                if idx >= len(sample["image"]):
                    continue
                image_path = sample["image"][idx]
                if not Path(image_path).exists():
                    # sample paths are relative to the Robo-Dopamine repo root
                    alt = Path(sj_path).parent / image_path
                    if alt.exists():
                        image_path = str(alt)
                    else:
                        print(f"    [skip] {view}: missing {image_path}")
                        continue

                boxes = grounding.ground(image_path, task, phrase_query=phrase)
                # --best-only: keep just the top-scoring box, matching what
                # steer_grm_heads.ground_best() actually consumes. The full
                # list is kept by default so detector quality is visible.
                kept = (boxes[:1] if args.best_only else boxes)
                view_dir = out_root / f"{task_slug}__{view}"
                view_dir.mkdir(parents=True, exist_ok=True)
                frame_id = sample.get("id", f"{sample_idx_global:03d}").split("-")[-1]
                out_img = view_dir / f"{frame_id}.png"
                if kept:
                    draw_boxes(image_path, kept, out_img)
                else:
                    # Still copy the frame so the directory structure is complete
                    # and empty detections are visually obvious.
                    Image.open(image_path).convert("RGB").save(out_img)

                for b in kept:
                    all_records.append({
                        "sample_idx": sample_idx_global,
                        "sample_id": sample.get("id", ""),
                        "task": task,
                        "phrase_query": phrase,
                        "view": view,
                        "frame_id": frame_id,
                        "label": b.label,
                        "score": round(b.score, 4),
                        "bbox_x1": round(b.bbox[0], 1),
                        "bbox_y1": round(b.bbox[1], 1),
                        "bbox_x2": round(b.bbox[2], 1),
                        "bbox_y2": round(b.bbox[3], 1),
                        "image": image_path,
                        "viz": str(out_img),
                    })
                if args.best_only and len(boxes) > 1:
                    print(f"    {view}/{frame_id}: {len(boxes)} detected, kept best -> {out_img.name}")
                else:
                    print(f"    {view}/{frame_id}: {len(kept)} boxes -> {out_img.name}")
            sample_idx_global += 1

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(all_records, indent=2))
    print(f"[grounding-eval] wrote {summary_path} ({len(all_records)} detections)")

    csv_path = out_root / "summary.csv"
    if all_records:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
        print(f"[grounding-eval] wrote {csv_path}")

    # Quick aggregate: detection rate per (task, view, label).
    if all_records:
        agg: Dict[tuple, Dict[str, int]] = {}
        view_set = set()
        for r in all_records:
            key = (r["task"], r["view"], r["label"])
            agg.setdefault(key, {"n": 0})
            agg[key]["n"] += 1
            view_set.add(r["view"])
        # Count how many frames per view were attempted (denominator).
        attempted: Dict[tuple, int] = {}
        for r in all_records:
            attempted[(r["task"], r["view"])] = attempted.get((r["task"], r["view"]), 0)
        # approximate denominator from sample_idx_global * views — cheap and good enough
        print("\n[grounding-eval] detection counts (task, view, label -> hits):")
        for key in sorted(agg.keys()):
            task, view, label = key
            print(f"  {task!r} | {view} | {label!r}: {agg[key]['n']}")


if __name__ == "__main__":
    main()
