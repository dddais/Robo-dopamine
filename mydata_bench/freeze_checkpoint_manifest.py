"""CLI for freezing or verifying strict local-checkpoint content manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydata_bench.my_dataset.checkpoint_manifest import (
    freeze_checkpoint_content_manifest,
    verify_checkpoint_content_manifest,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Freeze or verify a complete-content checkpoint manifest"
    )
    commands = root.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--model-path", required=True)
    freeze.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--model-path", required=True)
    verify.add_argument("--manifest", required=True)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "freeze":
        print(freeze_checkpoint_content_manifest(args.model_path, args.output))
        return
    result = verify_checkpoint_content_manifest(args.model_path, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
