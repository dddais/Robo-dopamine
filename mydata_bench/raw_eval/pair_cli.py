from __future__ import annotations

import argparse
import sys

from ..config import load_config
from .pairs import prepare, run, score


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python rewardbench/run_paired_raw_eval.py")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        item = commands.add_parser(name)
        item.add_argument("--config", required=True)
        if name == "run":
            item.add_argument("--dry-run", action="store_true")
            item.add_argument("--retry-failed", action="store_true")
    item = commands.add_parser("score")
    item.add_argument("--run-dir", required=True)
    item.add_argument("--bootstrap-samples", type=int, default=10_000)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare(load_config(args.config)))
    elif args.command == "run":
        print(run(load_config(args.config), dry_run=args.dry_run, retry_failed=args.retry_failed))
    else:
        result = score(args.run_dir, bootstrap_samples=args.bootstrap_samples)
        print(f"valid_pairs={result['paired_valid']} invalid={result['invalid_count']}")


if __name__ == "__main__":
    main(sys.argv[1:])
