"""Compare old vs new GroundingDINO grounding on GRM task images.

Runs both `grounding.TaskGrounding` (old: no trailing dot, QA filtering,
color fallback, threshold lowering) and `grounding_new.TaskGrounding`
(new: trailing dot, plain top-score pick) on the same frames, writes
side-by-side visualizations and a single summary.csv with a `version`
column for direct comparison.

Usage:
    python eval_grounding_new.py \
        --sample-json ./results/white_cube_inter20/.../sample.json \
        --views after_cam_high after_cam_left_wrist \
        --max-samples 5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

import grounding as g_old
import grounding_new as g_new


# DATA_JSON=["./results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_1_carrot/blank/inter20/carrot/26-07-08-11-42-09_incremental_mode_pick_the_carrot_and_put_it_on_yellow_plate_/sample.json"]
# DATA_JSON=["./results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_3_bottle/blank/inter20/bottle/26-07-07-15-11-17_incremental_mode_pick_the_bottle_and_put_it_on_yellow_plate_/sample.json"]
DATA_JSON=["./results/auto_pick_3_obj/GRM-2.0-8B/pick3suc_4_cube/blank/inter20/cube/26-07-07-14-56-34_incremental_mode_pick_the_white_cube_and_put_it_on_yellow_plate_/sample.json"]
MAX_SAMPLES=10
OUT="./results/grounding_compare/cube_cube"
TASK="pick the white cube and put it on yellow plate"


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


def draw_box(image_path: str, box, out_path: Path) -> None:
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    if box is None:
        cv2.imwrite(str(out_path), image)
        return
    x1, y1, x2, y2 = [int(round(v)) for v in box.bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label_txt = f"{box.label} {box.score:.2f}"
    (tw, th), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = max(0, y1 - th - baseline - 4)
    cv2.rectangle(image, (x1, y_top), (x1 + tw + 6, y_top + th + baseline + 4), (0, 255, 0), -1)
    cv2.putText(image, label_txt, (x1 + 3, y_top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), image)


def record(version: str, sample_idx: int, sample: dict, task: str,
           view: str, image_path: str, box, out_img: Path) -> dict:
    return {
        "version": version,
        "sample_idx": sample_idx,
        "sample_id": sample.get("id", ""),
        "task": task,
        "view": view,
        "frame_id": sample.get("id", f"{sample_idx:03d}").split("-")[-1],
        "label": box.label if box else "",
        "score": round(box.score, 4) if box else "",
        "bbox_x1": round(box.bbox[0], 1) if box else "",
        "bbox_y1": round(box.bbox[1], 1) if box else "",
        "bbox_x2": round(box.bbox[2], 1) if box else "",
        "bbox_y2": round(box.bbox[3], 1) if box else "",
        "image": image_path,
        "viz": str(out_img),
    }


def resolve_image_path(sample: dict, view: str, sj_path: str) -> Optional[str]:
    idx = IMAGE_LABELS.index(view)
    if idx >= len(sample["image"]):
        return None
    image_path = sample["image"][idx]
    if Path(image_path).exists():
        return image_path
    alt = Path(sj_path).parent / image_path
    if alt.exists():
        return str(alt)
    return None


def write_comparison_videos(records: List[dict], out_root: Path, fps: float = 2.0) -> None:
    """Write one side-by-side (old|new) mp4 per view, ordered by sample_idx."""
    import cv2

    by_view: Dict[str, List[dict]] = {}
    for r in records:
        by_view.setdefault(r["view"], []).append(r)
    for view in by_view:
        by_view[view].sort(key=lambda r: r["sample_idx"])

    for view, recs in by_view.items():
        old_first = cv2.imread(str(recs[0]["viz"]))
        if old_first is None:
            print(f"[video] {view}: cannot read first frame, skip")
            continue
        h, w = old_first.shape[:2]
        canvas = (w * 2, h)
        video_path = out_root / f"compare_{view}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, canvas)
        if not writer.isOpened():
            print(f"[video] {view}: VideoWriter open failed, skip")
            continue

        # Pair records by sample_idx (old then new share the same idx).
        pairs: Dict[int, Dict[str, dict]] = {}
        for r in recs:
            pairs.setdefault(r["sample_idx"], {})[r["version"]] = r

        for sample_idx in sorted(pairs.keys()):
            pair = pairs[sample_idx]
            old_r, new_r = pair.get("old"), pair.get("new")
            if old_r is None or new_r is None:
                continue
            img_old = cv2.imread(str(old_r["viz"]))
            img_new = cv2.imread(str(new_r["viz"]))
            if img_old is None or img_new is None:
                continue
            if img_old.shape != img_new.shape:
                img_new = cv2.resize(img_new, (img_old.shape[1], img_old.shape[0]))
            frame = cv2.hconcat([img_old, img_new])
            label = f"{view}  frame={old_r['frame_id']}  idx={sample_idx}"
            cv2.putText(frame, "OLD", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "NEW", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        print(f"[video] wrote {video_path} ({len(pairs)} frames)")


def main():
    ap = argparse.ArgumentParser(description="Compare old vs new GroundingDINO grounding")
    ap.add_argument("--sample-json", action="append", default=list(DATA_JSON),
                    help="Can be repeated to mix tasks")
    ap.add_argument("--grounding-model", default="../model/grounding-dino-base")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.20)
    ap.add_argument("--views", nargs="+",
                    default=["after_cam_high", "after_cam_left_wrist", "after_cam_right_wrist"],
                    choices=IMAGE_LABELS)
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    ap.add_argument("--task", default=TASK,
                    help="Override the task string from sample.json (e.g. 'pick the carrot and put it on yellow plate')")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="Frame rate for the side-by-side comparison videos")
    ap.add_argument("--no-video", action="store_true", help="Skip writing comparison mp4s")
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()

    out_root = Path(args.output)
    old_dir = out_root / "old"
    new_dir = out_root / "new"
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[compare] loading GroundingDINO from {args.grounding_model} on {device}")
    common = dict(model_path=args.grounding_model, device=device,
                  box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    grounding_old = g_old.TaskGrounding(**common)
    grounding_new = g_new.TaskGrounding(**common)

    all_records: List[dict] = []
    sample_idx = 0

    for sj_path in args.sample_json:
        samples = json.loads(Path(sj_path).read_text())
        if args.max_samples is not None:
            step = max(1, len(samples) // max(1, args.max_samples))
            samples = samples[::step][: args.max_samples]

        for sample in samples:
            task = args.task if args.task is not None else sample["task"]
            task_slug = slugify(task)
            print(f"[compare] sample {sample_idx} task={task!r}")

            for view in args.views:
                image_path = resolve_image_path(sample, view, sj_path)
                if image_path is None:
                    print(f"    [skip] {view}: missing image")
                    continue

                frame_id = sample.get("id", f"{sample_idx:03d}").split("-")[-1]

                box_old = grounding_old.ground_best(image_path, task)
                out_old = old_dir / f"{task_slug}__{view}__{frame_id}.png"
                draw_box(image_path, box_old, out_old)

                box_new = grounding_new.ground_best(image_path, task)
                out_new = new_dir / f"{task_slug}__{view}__{frame_id}.png"
                draw_box(image_path, box_new, out_new)

                all_records.append(record("old", sample_idx, sample, task, view, image_path, box_old, out_old))
                all_records.append(record("new", sample_idx, sample, task, view, image_path, box_new, out_new))

                status_old = f"{box_old.label}/{box_old.score:.2f}" if box_old else "None"
                status_new = f"{box_new.label}/{box_new.score:.2f}" if box_new else "None"
                print(f"    {view}/{frame_id}: old={status_old}  new={status_new}")

            sample_idx += 1

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(all_records, indent=2))
    print(f"[compare] wrote {summary_path} ({len(all_records)} records)")

    csv_path = out_root / "summary.csv"
    if all_records:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
        print(f"[compare] wrote {csv_path}")

    if not args.no_video and all_records:
        write_comparison_videos(all_records, out_root, fps=args.fps)


if __name__ == "__main__":
    main()
