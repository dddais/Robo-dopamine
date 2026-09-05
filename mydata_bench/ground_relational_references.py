#!/usr/bin/env python3
"""Ground instruction destinations on terminal frames without reading labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mydata_bench.grounding.sam3 import SAM3Grounder
from mydata_bench.io import append_jsonl, read_jsonl, sha256_file
from mydata_bench.schemas import SCHEMA_VERSION


PLACEMENT_REFERENCE = re.compile(
    r"\b(?:place|put|insert)\b.*?"
    r"\b(?:into|inside|in|onto|on|to)\b\s+"
    r"(?:the|a|an)?\s*(?P<reference>[^.,;]+)",
    re.IGNORECASE,
)


def extract_completion_reference(task: str) -> str | None:
    """Extract the placement destination, not a spatial disambiguator."""
    matches = list(PLACEMENT_REFERENCE.finditer(task))
    if not matches:
        return None
    value = re.sub(r"\s+", " ", matches[-1].group("reference")).strip()
    return value.lower() or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = {
        str(row.get("example_id")): row
        for row in read_jsonl(args.output)
    } if args.output.exists() else {}
    grounder = SAM3Grounder(
        {
            "model_path": args.model_path,
            "checkpoint_path": args.checkpoint_path,
            "device": "cuda",
            "threshold": args.threshold,
            "mask_threshold": args.mask_threshold,
            "top_n": args.top_n,
        }
    )
    cache: dict[tuple[str, tuple[str, ...]], dict | None] = {}
    for sample in read_jsonl(args.cohort_inputs):
        example_id = str(sample["example_id"])
        if example_id in previous:
            continue
        reference = extract_completion_reference(str(sample["task"]))
        queries = [reference] if reference else []
        base = {
            "schema_version": SCHEMA_VERSION,
            "example_id": example_id,
            "video_sha256": str(sample["video_sha256"]),
            "last_image_path": str(Path(sample["last_image_path"]).resolve()),
            "reference_object": reference,
            "queries": queries,
            "labels_model_facing": False,
        }
        if not queries:
            append_jsonl(args.output, {**base, "status": "not_applicable"})
            continue
        key = (sha256_file(sample["last_image_path"]), tuple(queries))
        if key not in cache:
            candidates = grounder.candidates(sample["last_image_path"], queries)
            cache[key] = grounder.select(
                sample["last_image_path"], candidates, len(queries)
            )
        selected = cache[key]
        if selected is None:
            append_jsonl(args.output, {**base, "status": "no_detection"})
            continue
        append_jsonl(
            args.output,
            {
                **base,
                "status": "ok",
                "bbox": [float(value) for value in selected["bbox"]],
                "score": float(selected["score"]),
                "query": selected.get("query"),
            },
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
