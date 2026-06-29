#!/usr/bin/env python3
"""Inventory local turn-on-radio-like candidates without adding benchmark labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_MD = ROOT / "research_outputs/radio_intake_candidates.md"
DEFAULT_OUT_JSON = ROOT / "research_outputs/radio_intake_candidates.json"
EXPECTED_VIDEOS = {
    "front": "cam_high.mp4",
    "left_wrist": "cam_left_wrist.mp4",
    "right_wrist": "cam_right_wrist.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-root", type=Path, default=ROOT / "aligned_data")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def video_info(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def is_radio_like_dir(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith("xzx") or "radio" in name


def matching_result_dirs(results_root: Path, episode_id: str) -> list[Path]:
    result_dirs = []
    for path in results_root.glob(f"{episode_id}*"):
        if not path.is_dir():
            continue
        suffix = path.name[len(episode_id) :]
        if suffix == "" or (suffix.startswith("_") and not suffix.startswith("_sub")):
            result_dirs.append(path)
    return sorted(result_dirs)


def cached_prediction_summary(results_root: Path, episode_id: str) -> dict:
    result_dirs = matching_result_dirs(results_root, episode_id)
    pred_files = []
    run_summaries = []
    progress_curves = []
    modes = set()
    for result_dir in result_dirs:
        for pred in result_dir.rglob("pred_vllm.json"):
            pred_files.append(pred)
            lower = str(pred.parent).lower()
            for mode in ("forward", "incremental", "backward"):
                if f"{mode}_mode" in lower:
                    modes.add(mode)
        run_summaries.extend(result_dir.glob("run_summary.json"))
        progress_curves.extend(result_dir.glob("progress_curve.png"))
    return {
        "result_dirs": [rel(path) for path in sorted(result_dirs)],
        "num_pred_files": len(pred_files),
        "modes_found": sorted(modes),
        "has_all_three_modes": {"forward", "incremental", "backward"}.issubset(modes),
        "run_summaries": [rel(path) for path in sorted(run_summaries)],
        "progress_curves": [rel(path) for path in sorted(progress_curves)],
    }


def inventory(aligned_root: Path, results_root: Path) -> list[dict]:
    candidates = []
    for data_dir in sorted(path for path in aligned_root.iterdir() if path.is_dir()):
        if not is_radio_like_dir(data_dir):
            continue
        videos = {view: data_dir / filename for view, filename in EXPECTED_VIDEOS.items()}
        missing = [view for view, path in videos.items() if not path.exists()]
        front_info = video_info(videos["front"]) if not missing else None
        pred = cached_prediction_summary(results_root, data_dir.name)
        status = "pending_human_verification"
        if data_dir.name == "xzx_episode_1_sub23":
            status = "already_live_verified_as_xzx_radio_sub23"
        candidates.append(
            {
                "data_dir": rel(data_dir),
                "candidate_episode_id": data_dir.name,
                "annotation_status": status,
                "benchmark_label_status": "not_a_new_benchmark_label",
                "missing_views": missing,
                "video_metadata": front_info,
                "cached_predictions": pred,
                "recommended_next_step": (
                    "Already represented by xzx_radio_sub23; do not duplicate."
                    if status.startswith("already_live")
                    else "Scaffold outside live benchmark, extract keyframes, and human-verify button_press/indicator_green before any benchmark inclusion."
                ),
            }
        )
    return candidates


def write_report(payload: dict, out_md: Path) -> None:
    lines = [
        "# Radio Intake Candidates / Radio 待接入候选",
        "",
        "This inventory lists local radio-like videos that could be inspected next. It does not add benchmark labels and does not change `benchmark_v0/episodes.json`.",
        "",
        "当前 live verified non-Markovian benchmark 仍然只有 `xzx_radio_sub23`。",
        "",
        "## Summary",
        "",
        f"- Candidates found: `{payload['num_candidates']}`",
        f"- Pending candidates: `{payload['num_pending']}`",
        f"- Already live verified entries: `{payload['num_already_live']}`",
        f"- Adds benchmark labels: `{payload['adds_benchmark_labels']}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Status | Frames | FPS | Cached preds | Modes | Next step |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for item in payload["candidates"]:
        meta = item.get("video_metadata") or {}
        pred = item["cached_predictions"]
        lines.append(
            f"| `{item['candidate_episode_id']}` | `{item['annotation_status']}` | "
            f"{meta.get('frames', '')} | {float(meta.get('fps', 0.0)):.2f} | "
            f"{pred['num_pred_files']} | `{', '.join(pred['modes_found'])}` | "
            f"{item['recommended_next_step']} |"
        )
    lines += [
        "",
        "## Required Gate Before Benchmark Inclusion",
        "",
        "- Create scaffold files under `research_outputs/scaffolded_radio_episodes/`.",
        "- Extract event-window keyframes.",
        "- Human-verify `button_press` and `indicator_green`, including frame ids and evidence views.",
        "- Fill success label and cached GRM paths.",
        "- Only then copy finalized files into `benchmark_v0/` and run `research/validate_benchmark_v0.py`.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "conda run -n robo-dopamine python research/inventory_radio_intake_candidates.py",
        "```",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidates = inventory(args.aligned_root, args.results_root)
    payload = {
        "adds_benchmark_labels": False,
        "live_verified_non_markovian_episode": "xzx_radio_sub23",
        "num_candidates": len(candidates),
        "num_pending": sum(1 for item in candidates if item["annotation_status"] == "pending_human_verification"),
        "num_already_live": sum(1 for item in candidates if item["annotation_status"].startswith("already_live")),
        "candidates": candidates,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload, args.out_md)
    print(f"candidates={payload['num_candidates']}")
    print(f"pending={payload['num_pending']}")
    print(f"adds_benchmark_labels={str(payload['adds_benchmark_labels']).lower()}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_json}")


if __name__ == "__main__":
    main()
