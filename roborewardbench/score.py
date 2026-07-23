#!/usr/bin/env python3
"""Score saved Robo-Dopamine predictions with RoboRewardBench metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from roborewardbench.data import load_metadata_reference
from roborewardbench.metrics import calibrate_progress, compute_metrics, load_calibration


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return records


def score_records(
    records: Iterable[dict[str, Any]],
    *,
    calibration_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    input_rows = list(records)
    latest_by_id: dict[str, dict[str, Any]] = {}
    rows_without_id = []
    for row in input_rows:
        if row.get("id") is None:
            rows_without_id.append(row)
        else:
            latest_by_id[str(row["id"])] = row
    rows = rows_without_id + list(latest_by_id.values())
    metadata_reference = load_metadata_reference(metadata_path) if metadata_path is not None else None
    expected_records = metadata_reference["records"] if metadata_reference is not None else None
    result = {
        "input_record_count": len(input_rows),
        "deduplicated_record_count": len(rows),
        "duplicate_record_count": len(input_rows) - len(rows),
        "metadata": (
            {
                "path": metadata_reference["path"],
                "sha256": metadata_reference["sha256"],
                "num_records": metadata_reference["num_records"],
            }
            if metadata_reference is not None
            else None
        ),
        "raw": compute_metrics(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            expected_records=expected_records,
        )
    }
    if calibration_path is not None:
        calibration = load_calibration(calibration_path)
        calibrated_rows = []
        for row in rows:
            updated = dict(row)
            if row.get("status", "ok") == "ok":
                updated["calibrated_progress"] = calibrate_progress(row["progress"], calibration)
            calibrated_rows.append(updated)
        result["validation_calibrated"] = compute_metrics(
            calibrated_rows,
            progress_field="calibrated_progress",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            expected_records=expected_records,
        )
        result["calibration"] = {
            "path": str(calibration_path),
            "fit_split": calibration["fit_split"],
            "method": calibration["method"],
            "num_examples": calibration["num_examples"],
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Prediction JSONL from run_benchmark.py")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Exact split metadata.jsonl; required for official_comparable=true",
    )
    parser.add_argument("--output", default=None, help="Output JSON; defaults beside predictions")
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional validation-fitted isotonic calibration JSON",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output) if args.output else Path(args.predictions).with_name("metrics.json")
    result = score_records(
        read_jsonl(args.predictions),
        calibration_path=args.calibration,
        metadata_path=args.metadata,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
