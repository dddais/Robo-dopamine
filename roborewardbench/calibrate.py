#!/usr/bin/env python3
"""Fit a monotonic Robo-Dopamine calibration using validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roborewardbench.metrics import fit_monotonic_calibration, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Validation prediction JSONL")
    parser.add_argument("--output", required=True, help="Calibration JSON to create")
    args = parser.parse_args()

    records = []
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

    latest_by_id = {
        str(record["id"]): record for record in records if record.get("id") is not None
    }
    records_without_id = [record for record in records if record.get("id") is None]
    calibration = fit_monotonic_calibration(records_without_id + list(latest_by_id.values()))
    save_calibration(calibration, args.output)
    print(json.dumps(calibration, indent=2, ensure_ascii=True))
    print(f"Saved validation calibration to {args.output}")


if __name__ == "__main__":
    main()
