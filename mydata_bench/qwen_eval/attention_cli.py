"""CLI for base-Qwen attention ranking and steering experiments."""

from __future__ import annotations

import argparse

from ..config import load_config
from .attention_experiment import (
    build_ranking_manifest,
    build_cohort_manifest,
    rank,
    score,
    steer,
    validate_ranking_inputs,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python mydata_bench/run_qwen_attention.py")
    commands = root.add_subparsers(dest="command", required=True)
    for name in (
        "prepare-ranking",
        "validate-ranking",
        "prepare-cohort",
        "rank",
        "steer",
        "score",
    ):
        item = commands.add_parser(name)
        item.add_argument("--config", required=True)
        if name in {"rank", "steer"}:
            item.add_argument("--retry-failed", action="store_true")
        if name == "steer":
            item.add_argument("--shard-id", type=int)
            item.add_argument("--num-shards", type=int)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare-ranking":
        print(build_ranking_manifest(config))
    elif args.command == "validate-ranking":
        print(validate_ranking_inputs(config))
    elif args.command == "prepare-cohort":
        print(build_cohort_manifest(config))
    elif args.command == "rank":
        print(rank(config, retry_failed=args.retry_failed))
    elif args.command == "steer":
        attention = config["attention_steer"]
        if (args.shard_id is None) != (args.num_shards is None):
            raise ValueError("--shard-id and --num-shards must be provided together")
        if args.shard_id is not None:
            if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
                raise ValueError("Require 0 <= shard-id < num-shards")
            attention["shard_id"] = args.shard_id
            attention["num_shards"] = args.num_shards
        print(steer(config, retry_failed=args.retry_failed))
    else:
        print(score(config))
