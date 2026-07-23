#!/usr/bin/env python3
"""Download selected RoboReward splits while preserving their directory layout."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="teetone/RoboReward")
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["test"])
    args = parser.parse_args()

    patterns = ["README.md"]
    for split in args.splits:
        patterns.extend([f"{split}/**", f"{split}/metadata.jsonl"])
    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.output,
        allow_patterns=patterns,
    )
    print(f"Downloaded {args.splits} to {path}")


if __name__ == "__main__":
    main()
