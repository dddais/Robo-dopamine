#!/usr/bin/env python3
"""Aggregate cached Robo-Dopamine result summaries for research notes."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research_outputs"
SUMMARY_FILES = [
    ROOT / "results/auto_pick_carrot_fail/summary_GRM8B_new/all_results.json",
    ROOT / "results/auto_pick_carrot_fail/summary3/all_results.json",
]


def infer_scene_object(data_tag: str) -> str:
    if "bottle" in data_tag:
        return "bottle"
    if "cube" in data_tag:
        return "cube"
    return "carrot"


def infer_video_success(data_tag: str) -> bool:
    return "suc" in data_tag


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in SUMMARY_FILES:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        source = path.parent.name
        for item in data:
            row = dict(item)
            row["source_summary"] = source
            row["scene_object"] = infer_scene_object(row["data_tag"])
            row["video_success"] = infer_video_success(row["data_tag"])
            row["task_matches_scene"] = row["task_tag"] == row["scene_object"]
            row["goal_success"] = bool(row["video_success"] and row["task_matches_scene"])
            rows.append(row)
    return rows


def auc_score(labels: list[bool], scores: list[float]) -> float | None:
    positives = [(s, i) for i, (y, s) in enumerate(zip(labels, scores)) if y]
    negatives = [(s, i) for i, (y, s) in enumerate(zip(labels, scores)) if not y]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def best_threshold(labels: list[bool], scores: list[float]) -> dict:
    candidates = sorted(set(scores))
    thresholds = [min(candidates) - 1e-6]
    thresholds += [(a + b) / 2 for a, b in zip(candidates, candidates[1:])]
    thresholds += [max(candidates) + 1e-6]

    best = None
    for th in thresholds:
        tp = fp = tn = fn = 0
        for y, s in zip(labels, scores):
            pred = s >= th
            if y and pred:
                tp += 1
            elif y and not pred:
                fn += 1
            elif not y and pred:
                fp += 1
            else:
                tn += 1
        acc = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        record = {
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
        if best is None or (record["f1"], record["accuracy"]) > (best["f1"], best["accuracy"]):
            best = record
    assert best is not None
    return best


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def grouped_descriptions(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["source_summary"],
            row["model_tag"],
            row["interval"],
            row["goal_success"],
            row["video_success"],
            row["task_matches_scene"],
        )
        groups[key].append(row)

    out = []
    for key, items in sorted(groups.items(), key=lambda x: tuple(map(str, x[0]))):
        source, model, interval, goal_success, video_success, task_matches = key
        stats = describe([float(r["avg_progress"]) for r in items])
        out.append(
            {
                "source_summary": source,
                "model": model,
                "interval": interval,
                "goal_success": goal_success,
                "video_success": video_success,
                "task_matches_scene": task_matches,
                **{f"avg_progress_{k}": v for k, v in stats.items()},
            }
        )
    return out


def score_metrics(rows: list[dict]) -> list[dict]:
    metrics = []
    score_names = ["avg_progress", "forward_progress", "incremental_progress", "backward_progress"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source_summary"], row["model_tag"], row["interval"])].append(row)

    for (source, model, interval), items in sorted(groups.items(), key=lambda x: tuple(map(str, x[0]))):
        labels = [bool(r["goal_success"]) for r in items]
        for score_name in score_names:
            scores = [float(r[score_name]) for r in items]
            auc = auc_score(labels, scores)
            best = best_threshold(labels, scores)
            metrics.append(
                {
                    "source_summary": source,
                    "model": model,
                    "interval": interval,
                    "score": score_name,
                    "n": len(items),
                    "positives": sum(labels),
                    "auc": auc,
                    **best,
                }
            )
    return metrics


def top_cases(rows: list[dict]) -> dict:
    negatives = [r for r in rows if not r["goal_success"]]
    positives = [r for r in rows if r["goal_success"]]
    return {
        "highest_scoring_negatives": sorted(
            negatives, key=lambda r: float(r["avg_progress"]), reverse=True
        )[:12],
        "lowest_scoring_positives": sorted(
            positives, key=lambda r: float(r["avg_progress"])
        )[:12],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}%"


def write_markdown(rows: list[dict], metrics: list[dict], groups: list[dict], cases: dict) -> None:
    lines = [
        "# Cached GRM Aggregation",
        "",
        "This report aggregates cached Robo-Dopamine summary files. `goal_success` is true only when the video is a success episode and the task prompt matches the object in the video.",
        "",
        "## Data",
        "",
        f"- Records: {len(rows)}",
        f"- Goal-success positives: {sum(1 for r in rows if r['goal_success'])}",
        f"- Negatives: {sum(1 for r in rows if not r['goal_success'])}",
        "",
        "## Best Threshold Metrics",
        "",
        "| source | model | interval | score | positives | AUROC | best F1 | accuracy | threshold | FP | FN |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            "| {source_summary} | {model} | {interval} | {score} | {positives} | {auc} | {f1:.3f} | {accuracy:.3f} | {threshold:.2f} | {fp} | {fn} |".format(
                **{**m, "auc": fmt_pct(m["auc"])}
            )
        )

    lines += [
        "",
        "## Score Distributions",
        "",
        "| source | model | interval | goal_success | video_success | task_match | n | mean | median | min | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        lines.append(
            "| {source_summary} | {model} | {interval} | {goal_success} | {video_success} | {task_matches_scene} | {avg_progress_n} | {avg_progress_mean:.2f} | {avg_progress_median:.2f} | {avg_progress_min:.2f} | {avg_progress_max:.2f} |".format(
                **g
            )
        )

    lines += [
        "",
        "## Highest-Scoring Negatives",
        "",
        "| source | model | data | scene | task | interval | avg | forward | inc | backward |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in cases["highest_scoring_negatives"]:
        lines.append(
            "| {source_summary} | {model_tag} | {data_tag} | {scene_object} | {task_tag} | {interval} | {avg_progress:.2f} | {forward_progress:.2f} | {incremental_progress:.2f} | {backward_progress:.2f} |".format(
                **r
            )
        )

    lines += [
        "",
        "## Lowest-Scoring Positives",
        "",
        "| source | model | data | scene | task | interval | avg | forward | inc | backward |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in cases["lowest_scoring_positives"]:
        lines.append(
            "| {source_summary} | {model_tag} | {data_tag} | {scene_object} | {task_tag} | {interval} | {avg_progress:.2f} | {forward_progress:.2f} | {incremental_progress:.2f} | {backward_progress:.2f} |".format(
                **r
            )
        )

    (OUT_DIR / "cached_grm_aggregation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    rows = [r for r in rows if r.get("model_tag") in {"GRM-2.0-8B", "multi_task"}]
    groups = grouped_descriptions(rows)
    metrics = score_metrics(rows)
    cases = top_cases(rows)

    write_csv(OUT_DIR / "cached_grm_rows.csv", rows)
    write_csv(OUT_DIR / "cached_grm_group_stats.csv", groups)
    write_csv(OUT_DIR / "cached_grm_metrics.csv", metrics)
    (OUT_DIR / "cached_grm_top_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(rows, metrics, groups, cases)

    print(f"records={len(rows)}")
    print(f"outputs={OUT_DIR}")
    for metric in metrics:
        if metric["score"] == "avg_progress":
            print(
                f"{metric['source_summary']} {metric['model']} inter{metric['interval']}: "
                f"AUROC={fmt_pct(metric['auc'])}, F1={metric['f1']:.3f}, "
                f"acc={metric['accuracy']:.3f}, threshold={metric['threshold']:.2f}"
            )


if __name__ == "__main__":
    main()

