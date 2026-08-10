from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config, section
from ..data import inventory as build_inventory
from ..data import load_episodes
from ..io import latest_by_id, read_jsonl, write_json
from ..metrics import clustered_stratified_bootstrap, compute_metrics
from .protocols import (
    DISCRETE_PROTOCOLS,
    ROBOREWARDBENCH_NATIVE,
    validate_protocol,
)
from .runner import requested_example_ids, run


def _score(run_dir: Path, bootstrap_samples: int) -> dict:
    paths = sorted(run_dir.glob("records.shard-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No shard records found under {run_dir}")
    latest = {}
    for path in paths:
        latest.update(latest_by_id(read_jsonl(path)))
    rows = list(latest.values())
    metrics = compute_metrics(rows)
    manifests = sorted(run_dir.glob("manifest*.json"))
    expected_ids = None
    protocol = None
    if manifests:
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        evaluation = manifest.get("config", {}).get("qwen_eval", {})
        protocol = validate_protocol(str(evaluation.get("protocol", ROBOREWARDBENCH_NATIVE)))
        if evaluation.get("dataset_root"):
            expected_ids = {
                row.example_id
                for row in load_episodes(
                    evaluation["dataset_root"], evaluation.get("split", "test"), compute_hash=False
                )
            }
            requested = requested_example_ids(evaluation)
            if requested:
                expected_ids &= requested
    actual_ids = set(latest)
    invalid = [row for row in rows if row.get("status") != "ok"]
    completion = {
        "expected_count": len(expected_ids) if expected_ids is not None else None,
        "record_count": len(actual_ids),
        "missing_example_ids": sorted(expected_ids - actual_ids) if expected_ids is not None else None,
        "unexpected_example_ids": sorted(actual_ids - expected_ids) if expected_ids is not None else None,
        "invalid_count": len(invalid),
    }
    completion["formal_scoring_ready"] = bool(
        expected_ids is not None and actual_ids == expected_ids and not invalid
    )
    metrics.update(
        {
            "model_family": "Qwen3-VL-8B-Instruct",
            "protocol": protocol,
            "discrete_output": protocol in DISCRETE_PROTOCOLS,
            "official_native_discrete_output": protocol == ROBOREWARDBENCH_NATIVE,
            "adapter_metric": protocol != ROBOREWARDBENCH_NATIVE,
            "completion": completion,
        }
    )
    if bootstrap_samples:
        metrics["macro_subset_mae_bootstrap"] = clustered_stratified_bootstrap(
            rows,
            lambda draw: float(compute_metrics(draw)["macro_subset_mae"]),
            samples=bootstrap_samples,
        )
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "invalid.json", invalid)
    write_json(run_dir / "completion.json", completion)
    micro = metrics["micro"]
    (run_dir / "metrics.md").write_text(
        "# Qwen3-VL Evaluation Metrics\n\n"
        f"- Protocol: `{protocol}`\n"
        f"- Formal scoring ready: `{completion['formal_scoring_ready']}`\n"
        f"- Valid / invalid: {metrics['num_valid']} / {metrics['num_invalid']}\n"
        f"- Macro subset MAE: {metrics['macro_subset_mae']}\n"
        f"- Micro MAE: {micro.get('mae')}\n"
        f"- Exact / within-one accuracy: {micro.get('exact_accuracy')} / {micro.get('within_one_accuracy')}\n",
        encoding="utf-8",
    )
    return metrics


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python mydata_bench/run_qwen_eval.py")
    commands = root.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--config", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--retry-failed", action="store_true")
    run_parser.add_argument("--shard-id", type=int)
    run_parser.add_argument("--num-shards", type=int)
    score = commands.add_parser("score")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--bootstrap-samples", type=int, default=10_000)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command in {"inventory", "run"}:
        config = load_config(args.config)
        evaluation = section(config, "qwen_eval")
        if args.command == "inventory":
            output = Path(evaluation["output_dir"]).resolve() / "inventory.json"
            write_json(
                output,
                build_inventory(
                    list(load_episodes(evaluation["dataset_root"], evaluation.get("split", "test")))
                ),
            )
            print(output)
        else:
            if (args.shard_id is None) != (args.num_shards is None):
                raise ValueError("--shard-id and --num-shards must be provided together")
            if args.shard_id is not None:
                if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
                    raise ValueError("Require 0 <= shard-id < num-shards")
                evaluation["shard_id"] = args.shard_id
                evaluation["num_shards"] = args.num_shards
            print(run(config, dry_run=args.dry_run, retry_failed=args.retry_failed))
    else:
        metrics = _score(Path(args.run_dir).resolve(), args.bootstrap_samples)
        print(f"valid={metrics['num_valid']} invalid={metrics['num_invalid']}")
