#!/usr/bin/env python3
"""Merge disjoint append-only steering cohorts into a derived result artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir / "steering.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")
    rows = []
    seen = set()
    fingerprints = set()
    for path in args.input:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("example_id")), str(row.get("condition")))
                if key in seen:
                    raise ValueError(f"duplicate merged key {key} from {path}:{line_number}")
                seen.add(key)
                fingerprints.add(str(row.get("ranking_fingerprint")))
                rows.append(row)
    if len(fingerprints) != 1:
        raise ValueError(f"inputs use different ranking fingerprints: {sorted(fingerprints)}")
    rows.sort(key=lambda row: (str(row.get("example_id")), str(row.get("condition"))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "type": "derived_disjoint_cohort_union",
        "inputs": [str(path.resolve()) for path in args.input],
        "record_count": len(rows),
        "unique_example_condition_count": len(seen),
        "ranking_fingerprint": next(iter(fingerprints)),
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{output.resolve()} records={len(rows)}")


if __name__ == "__main__":
    main()
