#!/usr/bin/env python3
"""Freeze causal-safe heads at unit dose and padding heads at an inert dose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import object_fingerprint, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--padding-multiplier", type=float, default=0.1)
    args = parser.parse_args()
    source_path = Path(args.input).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = source.get("ranking")
    if not isinstance(rows, list) or len(rows) < 64:
        raise ValueError("Expected a complete contrastive causal ranking")
    ranking = []
    for row in rows:
        item = dict(row)
        if bool(item.get("safe_padding", False)):
            item["steering_multiplier"] = float(args.padding_multiplier)
        elif item.get("success_target_correct_margin_delta") is not None and int(
            item.get("safety_tier", 9)
        ) <= 1:
            item["steering_multiplier"] = 1.0
        else:
            item["steering_multiplier"] = float(
                item.get("steering_multiplier", 1.0)
            )
        ranking.append(item)
    artifact = {
        **{key: value for key, value in source.items() if key not in {"ranking", "fingerprint"}},
        "schema_version": "paired-contrastive-causal-frozen-padding-ranking-v1",
        "ranking_source": "paired_contrastive_causal_safe_heads_with_inert_padding",
        "source_contrastive_fingerprint": source["fingerprint"],
        "padding_multiplier": float(args.padding_multiplier),
        "ranking": ranking,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "contrastive_causal_frozen_padding_ranking.json"
    write_json(path, artifact)
    print(path)


if __name__ == "__main__":
    main()
