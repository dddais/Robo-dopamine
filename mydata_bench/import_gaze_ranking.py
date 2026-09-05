#!/usr/bin/env python3
"""Adapt an independent gaze-head ranking to the local ranking contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mydata_bench.io import object_fingerprint, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("Gaze ranking must be a non-empty list")
    pairs = [(int(row["layer"]), int(row["head"])) for row in ranking]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Gaze ranking contains duplicate layer/head pairs")
    if int(summary.get("n_valid_samples", 0)) != 500:
        raise ValueError("Expected the independently discovered 500-comic ranking")
    normalized = [
        {
            "layer": layer,
            "head": head,
            "score": float(row["score"]),
            "source_rank": index + 1,
        }
        for index, (row, (layer, head)) in enumerate(zip(ranking, pairs, strict=True))
    ]
    fingerprint = object_fingerprint(
        {
            "method": "independent_gaze_heads_500_comics",
            "ranking_sha256": sha256_file(args.ranking),
            "summary_sha256": sha256_file(args.summary),
        }
    )
    write_json(
        args.output,
        {
            "fingerprint": fingerprint,
            "ranking": normalized,
            "method": "independent_gaze_heads_500_comics",
            "labels_model_facing": False,
            "n_valid_samples": 500,
            "source_ranking": str(args.ranking.resolve()),
            "source_summary": str(args.summary.resolve()),
            "source_config": summary.get("config"),
        },
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
