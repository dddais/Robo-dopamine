#!/usr/bin/env python3
"""Create an append-only, task-stratified paired pilot cohort."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible-ids", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successes-per-task", type=int, default=3)
    args = parser.parse_args()

    eligible = set(json.loads(args.eligible_ids.read_text(encoding="utf-8")))
    metadata = [
        json.loads(line)
        for line in args.metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["id"]): row for row in metadata if row["id"] in eligible}
    failures: dict[str, list[str]] = defaultdict(list)
    successes: dict[str, list[str]] = defaultdict(list)
    for example_id, row in by_id.items():
        if row["split"] == "fail":
            failures[str(row["source_suc_id"])].append(example_id)
        else:
            successes[str(row["task_id"])].append(example_id)

    selected: list[str] = []
    for task_id in sorted(successes):
        paired = [value for value in sorted(successes[task_id]) if failures[value]]
        for success_id in paired[: args.successes_per_task]:
            selected.extend((success_id, sorted(failures[success_id])[0]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sorted(selected), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selected": len(selected),
                "successes": len(selected) // 2,
                "failures": len(selected) // 2,
                "tasks": len({by_id[value]["task_id"] for value in selected}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
