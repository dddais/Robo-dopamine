#!/usr/bin/env python3
"""Score fixed direct-steering conditions on a full or excluded-ID cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mydata_bench.io import read_jsonl, sha256_file, write_json
from mydata_bench.score_contrastive_discrete import _paired_summary, _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", required=True, help="Comma-separated fixed conditions")
    parser.add_argument("--exclude-ids", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")
    conditions = [value for value in args.conditions.split(",") if value]
    if not conditions:
        raise ValueError("At least one condition is required")

    latest: dict[tuple[str, str], dict] = {}
    status_counts: dict[str, int] = {}
    source_rows = 0
    for row in read_jsonl(args.steering):
        source_rows += 1
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "ok":
            latest[(str(row["example_id"]), str(row["condition"]))] = row
    ids = sorted(example_id for example_id, condition in latest if condition == "baseline")
    excluded = (
        set(json.loads(args.exclude_ids.read_text(encoding="utf-8")))
        if args.exclude_ids
        else set()
    )
    ids = [example_id for example_id in ids if example_id not in excluded]
    if not ids:
        raise ValueError("No analysis IDs remain after exclusions")
    missing = {
        condition: [example_id for example_id in ids if (example_id, condition) not in latest]
        for condition in conditions
    }
    missing = {condition: values for condition, values in missing.items() if values}
    if missing:
        raise ValueError(f"Incomplete fixed conditions: { {key: len(value) for key, value in missing.items()} }")

    labels = {example_id: 5 if example_id.startswith("suc/") else 1 for example_id in ids}
    baseline = {
        example_id: int(latest[(example_id, "baseline")]["native_prediction"])
        for example_id in ids
    }
    metadata = {
        example_id: {
            "video_sha256": str(latest[(example_id, "baseline")].get("video_sha256") or example_id),
            "subset": latest[(example_id, "baseline")].get("subset"),
        }
        for example_id in ids
    }
    results = {}
    for condition in conditions:
        predictions = {
            example_id: int(latest[(example_id, condition)]["native_prediction"])
            for example_id in ids
        }
        results[condition] = _summary(predictions, labels)
        results[condition]["paired_vs_baseline"] = _paired_summary(
            predictions,
            baseline,
            labels,
            metadata,
            args.bootstrap_samples,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "method": "fixed_direct_discrete_attention_steering",
            "labels_model_facing": False,
            "steering_path": str(args.steering.resolve()),
            "steering_sha256": sha256_file(args.steering),
            "source_row_count": source_rows,
            "status_counts": dict(sorted(status_counts.items())),
            "conditions": conditions,
            "excluded_ids_file": str(args.exclude_ids.resolve()) if args.exclude_ids else None,
            "excluded_configured_count": len(excluded),
            "analysis_id_count": len(ids),
            "bootstrap_samples": args.bootstrap_samples,
            "baseline": _summary(baseline, labels),
            "by_condition": results,
        },
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
