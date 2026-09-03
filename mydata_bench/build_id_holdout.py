#!/usr/bin/env python3
"""Build an ordered, append-only ID holdout as full cohort minus development IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {args.output}")
    full = json.loads(args.full.read_text(encoding="utf-8"))
    excluded = json.loads(args.exclude.read_text(encoding="utf-8"))
    if not isinstance(full, list) or not isinstance(excluded, list):
        raise ValueError("full and exclude inputs must be JSON arrays")
    if len(full) != len(set(full)) or len(excluded) != len(set(excluded)):
        raise ValueError("input ID arrays must not contain duplicates")
    missing = sorted(set(excluded) - set(full))
    if missing:
        raise ValueError(f"excluded IDs absent from full cohort: {missing[:3]}")
    holdout = [example_id for example_id in full if example_id not in set(excluded)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(holdout, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output.resolve()} n={len(holdout)} excluded={len(excluded)}")


if __name__ == "__main__":
    main()
