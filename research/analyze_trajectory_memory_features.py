#!/usr/bin/env python3
"""Analyze cached GRM trajectories with simple memory-like score features."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research_outputs"
BASE = ROOT / "results/auto_pick_carrot_fail/GRM-2.0-8B"


def infer_scene_object(data_tag: str) -> str:
    if "bottle" in data_tag:
        return "bottle"
    if "cube" in data_tag:
        return "cube"
    return "carrot"


def infer_video_success(data_tag: str) -> bool:
    return "suc" in data_tag


def mode_from_name(name: str) -> str | None:
    for mode in ("forward", "incremental", "backward"):
        if f"_{mode}_mode_" in name:
            return mode
    return None


def auc_score(labels: list[bool], scores: list[float]) -> float | None:
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def best_threshold(labels: list[bool], scores: list[float]) -> dict:
    candidates = sorted(set(scores))
    thresholds = [candidates[0] - 1e-6]
    thresholds += [(a + b) / 2 for a, b in zip(candidates, candidates[1:])]
    thresholds += [candidates[-1] + 1e-6]
    best = None
    for th in thresholds:
        tp = fp = tn = fn = 0
        for y, s in zip(labels, scores):
            pred = s >= th
            if y and pred:
                tp += 1
            elif y:
                fn += 1
            elif pred:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        acc = (tp + tn) / len(labels)
        rec = {
            "threshold": th,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if best is None or (rec["f1"], rec["accuracy"]) > (best["f1"], best["accuracy"]):
            best = rec
    assert best is not None
    return best


def summarize_trajectory(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    progresses = [float(x.get("progress", 0.0)) * 100.0 for x in data]
    hops = [float(x.get("hop", 0.0)) * 100.0 for x in data]
    if not progresses:
        return {}

    max_so_far = progresses[0]
    max_drawdown = 0.0
    for p in progresses:
        max_so_far = max(max_so_far, p)
        max_drawdown = max(max_drawdown, max_so_far - p)

    neg_hops = [h for h in hops if h < 0]
    pos_hops = [h for h in hops if h > 0]
    return {
        "final": progresses[-1],
        "max": max(progresses),
        "min": min(progresses),
        "mean": mean(progresses),
        "max_drawdown": max_drawdown,
        "neg_hop_sum": -sum(neg_hops),
        "pos_hop_sum": sum(pos_hops),
        "min_hop": min(hops) if hops else 0.0,
        "max_hop": max(hops) if hops else 0.0,
        "neg_hop_count": len(neg_hops),
    }


def load_cases() -> list[dict]:
    grouped: dict[tuple, dict] = defaultdict(dict)
    for pred_path in BASE.glob("*/*/inter*/*/*/pred_vllm.json"):
        rel = pred_path.relative_to(BASE)
        data_tag = rel.parts[0]
        goal_tag = rel.parts[1]
        interval = int(re.sub(r"\D", "", rel.parts[2]))
        task_tag = rel.parts[3]
        run_name = rel.parts[4]
        mode = mode_from_name(run_name)
        if mode is None:
            continue

        key = (data_tag, goal_tag, interval, task_tag)
        grouped[key][mode] = summarize_trajectory(pred_path)

    rows = []
    for (data_tag, goal_tag, interval, task_tag), modes in sorted(grouped.items()):
        if not all(m in modes for m in ("forward", "incremental", "backward")):
            continue
        scene_object = infer_scene_object(data_tag)
        video_success = infer_video_success(data_tag)
        task_matches_scene = task_tag == scene_object
        row = {
            "data_tag": data_tag,
            "goal_tag": goal_tag,
            "interval": interval,
            "task_tag": task_tag,
            "scene_object": scene_object,
            "video_success": video_success,
            "task_matches_scene": task_matches_scene,
            "goal_success": bool(video_success and task_matches_scene),
        }
        for mode, stats in modes.items():
            for k, v in stats.items():
                row[f"{mode}_{k}"] = v

        row["baseline_final_avg"] = mean(
            [row["forward_final"], row["incremental_final"], row["backward_final"]]
        )
        row["trajectory_mean_avg"] = mean(
            [row["forward_mean"], row["incremental_mean"], row["backward_mean"]]
        )
        row["max_drawdown_avg"] = mean(
            [row["forward_max_drawdown"], row["incremental_max_drawdown"], row["backward_max_drawdown"]]
        )
        row["neg_hop_sum_avg"] = mean(
            [row["forward_neg_hop_sum"], row["incremental_neg_hop_sum"], row["backward_neg_hop_sum"]]
        )
        rows.append(row)
    return rows


def memory_score(row: dict, drawdown_weight: float, neg_weight: float) -> float:
    return (
        float(row["baseline_final_avg"])
        - drawdown_weight * float(row["max_drawdown_avg"])
        - neg_weight * float(row["neg_hop_sum_avg"])
    )


def evaluate(rows: list[dict]) -> tuple[list[dict], dict]:
    labels = [bool(r["goal_success"]) for r in rows]
    metrics = []

    candidates = {
        "baseline_final_avg": [float(r["baseline_final_avg"]) for r in rows],
        "trajectory_mean_avg": [float(r["trajectory_mean_avg"]) for r in rows],
        "minus_drawdown_0.5": [memory_score(r, 0.5, 0.0) for r in rows],
        "minus_neg_hop_0.25": [memory_score(r, 0.0, 0.25) for r in rows],
    }

    best_grid = None
    for dw in [i / 10 for i in range(0, 21)]:
        for nw in [i / 20 for i in range(0, 21)]:
            scores = [memory_score(r, dw, nw) for r in rows]
            bt = best_threshold(labels, scores)
            record = {
                "feature": "grid_memory_score",
                "drawdown_weight": dw,
                "neg_weight": nw,
                "auc": auc_score(labels, scores),
                **bt,
            }
            if best_grid is None or (record["f1"], record["accuracy"], record["auc"] or -1) > (
                best_grid["f1"],
                best_grid["accuracy"],
                best_grid["auc"] or -1,
            ):
                best_grid = record

    for name, scores in candidates.items():
        metrics.append(
            {
                "feature": name,
                "drawdown_weight": "",
                "neg_weight": "",
                "auc": auc_score(labels, scores),
                **best_threshold(labels, scores),
            }
        )
    assert best_grid is not None
    metrics.append(best_grid)
    return metrics, best_grid


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_auc(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def write_report(rows: list[dict], metrics: list[dict], best_grid: dict) -> None:
    positives = sum(1 for r in rows if r["goal_success"])
    lines = [
        "# Trajectory Memory Feature Analysis",
        "",
        "This report tests whether simple memory-like features computed only from cached GRM score trajectories can improve success/failure judgment.",
        "",
        "Important: these features do not inspect visual event content; they only use scalar progress and hop curves. A failure here motivates explicit event memory rather than scalar smoothing.",
        "",
        "## Data",
        "",
        f"- Complete cases: {len(rows)}",
        f"- Goal-success positives: {positives}",
        f"- Negatives: {len(rows) - positives}",
        "",
        "## Metrics",
        "",
        "| feature | AUROC | best F1 | accuracy | threshold | FP | FN | weights |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in metrics:
        weights = ""
        if m["feature"] == "grid_memory_score":
            weights = f"drawdown={m['drawdown_weight']}, neg={m['neg_weight']}"
        lines.append(
            f"| {m['feature']} | {fmt_auc(m['auc'])} | {m['f1']:.3f} | {m['accuracy']:.3f} | {m['threshold']:.2f} | {m['fp']} | {m['fn']} | {weights} |"
        )

    false_pos = []
    scores = [memory_score(r, best_grid["drawdown_weight"], best_grid["neg_weight"]) for r in rows]
    for row, score in zip(rows, scores):
        if not row["goal_success"] and score >= best_grid["threshold"]:
            false_pos.append((score, row))

    lines += [
        "",
        "## Remaining False Positives Under Best Scalar-Memory Score",
        "",
        "| score | data | scene | task | interval | final_avg | drawdown_avg | neg_hop_avg |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for score, row in sorted(false_pos, key=lambda x: x[0], reverse=True)[:12]:
        lines.append(
            f"| {score:.2f} | {row['data_tag']} | {row['scene_object']} | {row['task_tag']} | {row['interval']} | {row['baseline_final_avg']:.2f} | {row['max_drawdown_avg']:.2f} | {row['neg_hop_sum_avg']:.2f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Scalar trajectory memory helps only if failures leave visible traces in the GRM score curve, such as large regressions or unstable progress. Some cached failure episodes still receive very high final progress and remain difficult to reject with score-only memory. The proposed research should therefore add explicit visual/event memory, e.g. remembering order, forbidden contacts, dropped objects, and transient violations.",
    ]
    (OUT_DIR / "trajectory_memory_features.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_cases()
    metrics, best_grid = evaluate(rows)
    write_csv(OUT_DIR / "trajectory_memory_cases.csv", rows)
    write_csv(OUT_DIR / "trajectory_memory_metrics.csv", metrics)
    write_report(rows, metrics, best_grid)
    print(f"cases={len(rows)}")
    print(f"outputs={OUT_DIR}")
    for m in metrics:
        print(
            f"{m['feature']}: AUROC={fmt_auc(m['auc'])}, F1={m['f1']:.3f}, "
            f"acc={m['accuracy']:.3f}, FP={m['fp']}, FN={m['fn']}"
        )


if __name__ == "__main__":
    main()

