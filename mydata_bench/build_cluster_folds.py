#!/usr/bin/env python3
"""Split an ID cohort into deterministic, source-video-disjoint folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_fold(seed: str, task_id: str, source_suc_id: str, folds: int) -> int:
    payload = f"{seed}\0{task_id}\0{source_suc_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def write_new(path: Path, values: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--dev-fold", type=int, default=0)
    parser.add_argument("--seed", default="stage2-v1")
    parser.add_argument(
        "--stratified-by-task",
        action="store_true",
        help="Sort source clusters by seeded hash within each task, then assign folds round-robin.",
    )
    args = parser.parse_args()

    if args.folds < 2 or not 0 <= args.dev_fold < args.folds:
        raise ValueError("Require folds >= 2 and 0 <= dev-fold < folds")
    ids = json.loads(args.ids.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("IDs must be a unique JSON array")
    metadata_rows = load_jsonl(args.metadata)
    metadata = {str(row["id"]): row for row in metadata_rows}
    missing = [example_id for example_id in ids if example_id not in metadata]
    if missing:
        raise ValueError(f"IDs missing from metadata: {missing[:3]}")

    cluster_metadata = {
        (str(metadata[example_id]["task_id"]), str(metadata[example_id]["source_suc_id"]))
        for example_id in ids
    }
    cluster_fold: dict[str, int] = {}
    if args.stratified_by_task:
        tasks = sorted({task_id for task_id, _ in cluster_metadata})
        for task_id in tasks:
            sources = sorted(
                (source_suc_id for task, source_suc_id in cluster_metadata if task == task_id),
                key=lambda source_suc_id: (
                    stable_fold(args.seed, task_id, source_suc_id, 2**31 - 1),
                    source_suc_id,
                ),
            )
            for index, source_suc_id in enumerate(sources):
                cluster_fold[source_suc_id] = index % args.folds

    dev: list[str] = []
    test: list[str] = []
    for example_id in ids:
        row = metadata[example_id]
        source_suc_id = str(row["source_suc_id"])
        task_id = str(row["task_id"])
        fold = cluster_fold.get(source_suc_id)
        if fold is None:
            fold = stable_fold(args.seed, task_id, source_suc_id, args.folds)
        prior = cluster_fold.setdefault(source_suc_id, fold)
        if prior != fold:
            raise AssertionError(f"Cluster assigned inconsistently: {source_suc_id}")
        (dev if fold == args.dev_fold else test).append(example_id)

    if not dev or not test or set(dev) & set(test) or set(dev) | set(test) != set(ids):
        raise AssertionError("Invalid partition")
    write_new(args.dev_output, dev)
    write_new(args.test_output, test)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "folds": args.folds,
                "dev_fold": args.dev_fold,
                "stratified_by_task": args.stratified_by_task,
                "dev_n": len(dev),
                "test_n": len(test),
                "cluster_n": len(cluster_fold),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
