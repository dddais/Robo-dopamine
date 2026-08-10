from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from .dataset import prepare
from .experiment import metrics, rank, steer
from .visualize import run_visualize
from .video import run_video


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python mydata_bench/run_attention_eval.py")
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)
    rank_parser = commands.add_parser("rank")
    rank_parser.add_argument("--source", required=True, choices=("consensus", "in_domain"))
    rank_parser.add_argument("--config", required=True)
    rank_parser.add_argument("--dry-run", action="store_true")
    rank_parser.add_argument("--retry-failed", action="store_true")
    steer_parser = commands.add_parser("steer")
    steer_parser.add_argument("--config", required=True)
    steer_parser.add_argument("--dry-run", action="store_true")
    steer_parser.add_argument("--retry-failed", action="store_true")
    steer_parser.add_argument("--shard-id", type=int)
    steer_parser.add_argument("--num-shards", type=int)
    metrics_parser = commands.add_parser("metrics")
    metrics_parser.add_argument("--run-dir", required=True)
    metrics_parser.add_argument("--config")
    video_parser = commands.add_parser("video")
    video_parser.add_argument("--run-dir", required=True)
    video_parser.add_argument(
        "--count",
        type=int,
        help="Number of reproducibly hash-sampled evaluation episodes (defaults to the run config).",
    )
    video_parser.add_argument(
        "--seed",
        type=int,
        help="Sampling seed (defaults to the frozen run config seed).",
    )
    video_parser.add_argument("--dry-run", action="store_true")
    visualize_parser = commands.add_parser(
        "visualize",
        help="Render endpoint GRM inputs and attention heatmaps; does not run video grounding.",
    )
    visualize_parser.add_argument("--run-dir", required=True)
    visualize_parser.add_argument("--count", type=int, default=12)
    visualize_parser.add_argument("--seed", type=int)
    visualize_parser.add_argument("--dry-run", action="store_true")
    all_parser = commands.add_parser("all")
    all_parser.add_argument("--config", required=True)
    all_parser.add_argument("--dry-run", action="store_true")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare(load_config(args.config)))
    elif args.command == "rank":
        print(
            rank(
                load_config(args.config),
                args.source,
                dry_run=args.dry_run,
                retry_failed=args.retry_failed,
            )
        )
    elif args.command == "steer":
        config = load_config(args.config)
        if (args.shard_id is None) != (args.num_shards is None):
            raise ValueError("--shard-id and --num-shards must be provided together")
        if args.shard_id is not None:
            if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
                raise ValueError("Require 0 <= shard-id < num-shards")
            config["attention_eval"]["shard_id"] = args.shard_id
            config["attention_eval"]["num_shards"] = args.num_shards
        print(
            steer(
                config,
                dry_run=args.dry_run,
                retry_failed=args.retry_failed,
            )
        )
    elif args.command == "metrics":
        config = load_config(args.config) if args.config else None
        print(metrics(args.run_dir, config))
    elif args.command == "video":
        print(
            run_video(
                args.run_dir,
                dry_run=args.dry_run,
                episode_count=args.count,
                seed=args.seed,
            )
        )
    elif args.command == "visualize":
        print(
            run_visualize(
                args.run_dir,
                episode_count=args.count,
                seed=args.seed,
                dry_run=args.dry_run,
            )
        )
    else:
        config = load_config(args.config)
        run_dir = Path(config["attention_eval"]["output_dir"]).resolve()
        print(prepare(config))
        print(rank(config, "consensus", dry_run=args.dry_run))
        print(steer(config, dry_run=args.dry_run))
        print(metrics(run_dir, config))
        print(run_video(run_dir, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
