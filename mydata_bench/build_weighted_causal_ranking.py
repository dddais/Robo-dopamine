#!/usr/bin/env python3
"""Create an append-only causal ranking artifact with fixed head weights.

The utility never reads evaluation examples or labels.  It only records a
ranking order and weights already selected by a completed development-cohort
analysis, making the resulting inference configuration deterministic and
auditable on a disjoint holdout cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_ranks(value: str) -> list[int]:
    ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(rank < 1 for rank in ranks) or len(ranks) != len(set(ranks)):
        raise ValueError("promote ranks must be unique positive 1-based integers")
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promote-ranks", default="")
    parser.add_argument("--strong-count", type=int, required=True)
    parser.add_argument("--strong-weight", type=float, default=1.0)
    parser.add_argument("--mid-count", type=int)
    parser.add_argument("--mid-weight", type=float)
    parser.add_argument("--tail-weight", type=float, required=True)
    parser.add_argument("--method-detail", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {args.output}")
    mid_count = args.strong_count if args.mid_count is None else args.mid_count
    mid_weight = args.tail_weight if args.mid_weight is None else args.mid_weight
    if (
        args.strong_count < 1
        or mid_count < args.strong_count
        or args.strong_weight <= 0
        or mid_weight <= 0
        or args.tail_weight <= 0
    ):
        raise ValueError("counts and weights must be positive")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    ranking = source.get("ranking")
    if not isinstance(ranking, list) or len(ranking) < 64:
        raise ValueError("source ranking must contain at least 64 heads")
    promote = parse_ranks(args.promote_ranks)
    if any(rank > len(ranking) for rank in promote):
        raise ValueError("promote rank is outside source ranking")
    indices = [rank - 1 for rank in promote]
    indices.extend(index for index in range(len(ranking)) if index not in set(indices))
    reordered = []
    seen = set()
    for new_rank, source_index in enumerate(indices, start=1):
        row = dict(ranking[source_index])
        pair = (int(row["layer"]), int(row["head"]))
        if pair in seen:
            raise ValueError(f"duplicate head in source ranking: {pair}")
        seen.add(pair)
        row["source_rank_before_causal_rerank"] = source_index + 1
        row["rank"] = new_rank
        if new_rank <= args.strong_count:
            row["steering_weight"] = float(args.strong_weight)
        elif new_rank <= mid_count:
            row["steering_weight"] = float(mid_weight)
        else:
            row["steering_weight"] = float(args.tail_weight)
        reordered.append(row)

    artifact = dict(source)
    artifact.update(
        {
            "method": "dev_causal_sparse_weighting_v1",
            "ranking_method_detail": args.method_detail,
            "causal_weighting": {
                "strong_count": args.strong_count,
                "strong_weight": args.strong_weight,
                "mid_count": mid_count,
                "mid_weight": mid_weight,
                "tail_weight": args.tail_weight,
                "promoted_source_ranks": promote,
                "selection_cohort": "pilot_60_pairs",
                "inference_label_access": False,
            },
            "ranking": reordered,
        }
    )
    artifact.pop("fingerprint", None)
    canonical = json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    artifact["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
