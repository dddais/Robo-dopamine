#!/usr/bin/env python3
"""Build intervention dose-response curves from saved RoboRewardBench results.

RoboRewardBench evaluates one endpoint pair per video, so it has no within-video
progress trajectory comparable to the success-episode experiments.  The honest
curve for this experiment is therefore score/paired-shift versus attention-logit
bias.  This module is post-processing only and never reloads the GRM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import assert_unique, file_identity, read_jsonl, strict_dump
from .metrics import example_bootstrap_mean_ci
from .run_experiment import result_key


CONDITION_ORDER = {
    "candidate_target": 0,
    "candidate_wrong": 1,
    "low_rank_target": 2,
    "all_target": 3,
}
CONDITION_LABELS = {
    "candidate_target": "candidate heads × target bbox",
    "candidate_wrong": "candidate heads × wrong region",
    "low_rank_target": "low-ranked heads × target bbox",
    "all_target": "all heads × target bbox",
}
CONDITION_COLORS = {
    "candidate_target": "#D62728",
    "candidate_wrong": "#7F7F7F",
    "low_rank_target": "#FF7F0E",
    "all_target": "#2CA02C",
}
CONDITION_STYLES = {
    "candidate_target": "-",
    "candidate_wrong": "--",
    "low_rank_target": "-.",
    "all_target": ":",
}


def _valid_score(row: Mapping[str, Any]) -> float | None:
    if row.get("status") != "ok" or row.get("score") is None:
        return None
    value = float(row["score"])
    return value if math.isfinite(value) else None


def _series_key(row: Mapping[str, Any]) -> tuple[str, int | None, str, str]:
    return (
        str(row.get("condition")),
        None if row.get("top_k") is None else int(row["top_k"]),
        str(row.get("intervention")),
        str(row.get("target_role")),
    )


def _series_label(key: tuple[str, int | None, str, str]) -> str:
    condition, top_k, _intervention, _target_role = key
    suffix = "all" if top_k is None else f"top-{top_k}"
    return f"{CONDITION_LABELS.get(condition, condition)} ({suffix})"


def _series_marker(key: tuple[str, int | None, str, str]) -> str:
    _condition, top_k, _intervention, _target_role = key
    if top_k == 8:
        return "o"
    if top_k == 64:
        return "s"
    return "D"


def build_curve_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate valid scores and paired shifts for every bias/configuration."""

    if not rows:
        raise ValueError("No experiment records were provided")
    assert_unique((result_key(row) for row in rows), what="experiment result keys")
    families = {str(row.get("run_family_signature")) for row in rows}
    if len(families) != 1:
        raise ValueError("Input results do not belong to one experiment family")

    baseline_rows = [row for row in rows if str(row.get("condition")) == "baseline"]
    baseline_by_id: dict[str, float] = {}
    for row in baseline_rows:
        score = _valid_score(row)
        if score is not None:
            baseline_by_id[str(row["example_id"])] = score
    if not baseline_by_id:
        raise ValueError("No valid baseline scores were found")

    grouped: dict[tuple[tuple[str, int | None, str, str], float], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        condition = str(row.get("condition"))
        if condition == "baseline":
            continue
        grouped[(_series_key(row), float(row.get("swap_bias", 0.0)))].append(row)

    output_rows: list[dict[str, Any]] = []
    sort_key = lambda item: (
        CONDITION_ORDER.get(item[0][0][0], 99),
        -1 if item[0][0][1] is None else item[0][0][1],
        item[0][1],
        item[0][0][2],
        item[0][0][3],
    )
    for index, ((series_key, bias), config_rows) in enumerate(
        sorted(grouped.items(), key=sort_key)
    ):
        condition, top_k, intervention, target_role = series_key
        scores: list[float] = []
        shifts: list[float] = []
        for row in config_rows:
            score = _valid_score(row)
            if score is None:
                continue
            scores.append(score)
            baseline = baseline_by_id.get(str(row["example_id"]))
            if baseline is not None:
                shifts.append(score - baseline)
        score_ci = example_bootstrap_mean_ci(
            scores,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        shift_ci = example_bootstrap_mean_ci(
            shifts,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 10000 + index,
        )
        output_rows.append(
            {
                "condition": condition,
                "top_k": top_k,
                "swap_bias": bias,
                "intervention": intervention,
                "target_role": target_role,
                "series_label": _series_label(series_key),
                "num_records": len(config_rows),
                "num_valid_scores": len(scores),
                "mean_score": sum(scores) / len(scores) if scores else None,
                "score_example_bootstrap_95ci": score_ci,
                "num_paired": len(shifts),
                "mean_paired_shift": sum(shifts) / len(shifts) if shifts else None,
                "paired_shift_example_bootstrap_95ci": shift_ci,
            }
        )

    baseline_values = list(baseline_by_id.values())
    baseline_ci = example_bootstrap_mean_ci(
        baseline_values,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 20000,
    )
    summary = {
        "run_family_signature": next(iter(families)),
        "num_examples": len(baseline_by_id),
        "baseline": {
            "num_valid": len(baseline_values),
            "mean_score": sum(baseline_values) / len(baseline_values),
            "score_example_bootstrap_95ci": baseline_ci,
        },
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "resampling": "observed examples with replacement",
        },
    }
    return output_rows, summary


def write_curve_csv(rows: Sequence[Mapping[str, Any]], destination: str | Path) -> None:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition",
        "top_k",
        "swap_bias",
        "intervention",
        "target_role",
        "series_label",
        "num_records",
        "num_valid_scores",
        "mean_score",
        "score_ci_low",
        "score_ci_high",
        "num_paired",
        "mean_paired_shift",
        "paired_shift_ci_low",
        "paired_shift_ci_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            score_ci = row.get("score_example_bootstrap_95ci")
            shift_ci = row.get("paired_shift_example_bootstrap_95ci")
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "score_ci_low": score_ci[0] if score_ci is not None else None,
                    "score_ci_high": score_ci[1] if score_ci is not None else None,
                    "paired_shift_ci_low": shift_ci[0] if shift_ci is not None else None,
                    "paired_shift_ci_high": shift_ci[1] if shift_ci is not None else None,
                }
            )


def _group_series(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[tuple[str, int | None, str, str], list[Mapping[str, Any]]]]:
    grouped: dict[tuple[str, int | None, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            str(row["condition"]),
            None if row.get("top_k") is None else int(row["top_k"]),
            str(row["intervention"]),
            str(row["target_role"]),
        )
        grouped[key].append(row)
    return sorted(
        (
            (key, sorted(values, key=lambda row: float(row["swap_bias"])))
            for key, values in grouped.items()
        ),
        key=lambda item: (
            CONDITION_ORDER.get(item[0][0], 99),
            -1 if item[0][1] is None else item[0][1],
            item[0][2],
            item[0][3],
        ),
    )


def plot_curves(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    baseline = float(summary["baseline"]["mean_score"])
    ax.axhline(
        baseline,
        color="#1F77B4",
        linewidth=2.0,
        alpha=0.8,
        label=f"baseline mean ({baseline:+.3f})",
    )
    for key, values in _group_series(rows):
        condition = key[0]
        valid = [row for row in values if row.get("mean_score") is not None]
        if not valid:
            continue
        ax.plot(
            [float(row["swap_bias"]) for row in valid],
            [float(row["mean_score"]) for row in valid],
            marker=_series_marker(key),
            color=CONDITION_COLORS.get(condition),
            linestyle=CONDITION_STYLES.get(condition, "-"),
            linewidth=2.0,
            label=_series_label(key),
        )
    ax.set_xlabel("Attention-logit bias")
    ax.set_ylabel("Mean raw GRM score")
    ax.set_title("RoboRewardBench endpoint score dose-response")
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    score_path = root / "score_curve.png"
    fig.savefig(score_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    for key, values in _group_series(rows):
        condition = key[0]
        valid = [row for row in values if row.get("mean_paired_shift") is not None]
        if not valid:
            continue
        x = [float(row["swap_bias"]) for row in valid]
        y = [float(row["mean_paired_shift"]) for row in valid]
        lower = []
        upper = []
        has_ci = True
        for row, mean in zip(valid, y):
            ci = row.get("paired_shift_example_bootstrap_95ci")
            if ci is None:
                has_ci = False
                break
            lower.append(mean - float(ci[0]))
            upper.append(float(ci[1]) - mean)
        ax.errorbar(
            x,
            y,
            yerr=[lower, upper] if has_ci else None,
            marker=_series_marker(key),
            capsize=3,
            color=CONDITION_COLORS.get(condition),
            linestyle=CONDITION_STYLES.get(condition, "-"),
            linewidth=2.0,
            label=_series_label(key),
        )
    ax.set_xlabel("Attention-logit bias")
    ax.set_ylabel("Mean paired score shift vs baseline")
    ax.set_title("RoboRewardBench intervention dose-response")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    shift_path = root / "paired_shift_curve.png"
    fig.savefig(shift_path, dpi=160)
    plt.close(fig)
    return score_path, shift_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative")
    result_paths = [Path(value).expanduser().resolve() for value in args.results]
    rows = [row for path in result_paths for row in read_jsonl(path)]
    curve_rows, summary = build_curve_rows(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "curve.csv"
    json_path = output_dir / "curve.json"
    write_curve_csv(curve_rows, csv_path)
    score_path, shift_path = plot_curves(curve_rows, summary, output_dir)
    payload = {
        "schema_version": 1,
        "curve_semantics": (
            "Intervention dose-response across attention-logit bias. This is not a "
            "temporal task-progress trajectory: RoboRewardBench supplies one scored "
            "endpoint pair per video."
        ),
        "inputs": [file_identity(path) for path in result_paths],
        **summary,
        "rows": curve_rows,
        "artifacts": {
            "csv": str(csv_path),
            "score_curve_png": str(score_path),
            "paired_shift_curve_png": str(shift_path),
        },
    }
    strict_dump(payload, json_path)
    print(json.dumps(payload["artifacts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
