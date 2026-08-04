from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ..io import read_jsonl, write_json
from .audit import adjudicate, wilson_interval
from .pipeline import run_grounding, run_parser


def _compare(dino_run: Path, sam3_run: Path) -> dict:
    def accepted(path: Path):
        grouped = defaultdict(dict)
        for row in read_jsonl(path / "grounding.jsonl"):
            if row.get("status") == "ok":
                grouped[row["example_id"]][row["frame"]] = row
        return grouped

    dino = accepted(dino_run)
    sam3 = accepted(sam3_run)
    dino_both = {key for key, value in dino.items() if {"first", "last"} <= value.keys()}
    sam_both = {key for key, value in sam3.items() if {"first", "last"} <= value.keys()}
    common = dino_both & sam_both

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return intersection / union if union else 0.0

    ious = [
        iou(dino[key][frame]["bbox"], sam3[key][frame]["bbox"])
        for key in sorted(common)
        for frame in ("first", "last")
    ]
    total = len(set(dino) | set(sam3))
    result = {
        "dino_dual_endpoint": len(dino_both),
        "sam3_dual_endpoint": len(sam_both),
        "intersection": len(common),
        "union": total,
        "dino_coverage": len(dino_both) / total if total else None,
        "sam3_coverage": len(sam_both) / total if total else None,
        "dino_wilson_ci95": wilson_interval(len(dino_both), total),
        "sam3_wilson_ci95": wilson_interval(len(sam_both), total),
        "common_endpoint_mean_iou": sum(ious) / len(ious) if ious else None,
    }
    import json

    for name, path in (("dino", dino_run), ("sam3", sam3_run)):
        audit_path = path / "audit_summary.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            result[f"{name}_audited_correct_rate"] = audit.get("correct_rate")
            result[f"{name}_audited_correct_wilson_ci95"] = audit.get("wilson_ci95")
    write_json(dino_run.parent / "backend_comparison.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python rewardbench/run_grounding.py")
    commands = root.add_subparsers(dest="command", required=True)
    parse = commands.add_parser("parse")
    parse.add_argument("--config", required=True)
    parse.add_argument("--dry-run", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--backend", required=True, choices=("grounding_dino", "sam3"))
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    audit = commands.add_parser("audit")
    audit.add_argument("--run-dir", required=True)
    audit.add_argument(
        "--reviewers",
        nargs="+",
        choices=("reviewer1", "reviewer2"),
        default=["reviewer1"],
        help="review files to use; the default is the single-human-review protocol",
    )
    compare = commands.add_parser("compare")
    compare.add_argument("--dino-run", required=True)
    compare.add_argument("--sam3-run", required=True)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command in {"parse", "run"}:
        from ..config import load_config

        config = load_config(args.config)
        if args.command == "parse":
            print(run_parser(config, dry_run=args.dry_run))
        else:
            print(
                run_grounding(
                    config,
                    args.backend,
                    dry_run=args.dry_run,
                    retry_failed=args.retry_failed,
                )
            )
    elif args.command == "audit":
        run_dir = Path(args.run_dir).resolve()
        print(
            adjudicate(
                run_dir,
                list(read_jsonl(run_dir / "grounding.jsonl")),
                tuple(args.reviewers),
            )
        )
    else:
        print(_compare(Path(args.dino_run).resolve(), Path(args.sam3_run).resolve()))


if __name__ == "__main__":
    main()
