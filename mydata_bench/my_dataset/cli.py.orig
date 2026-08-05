"""Command line interface for the isolated local-dataset baseline pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import load_config
from ..io import sha256_file
from .data import audit_prepared, load_model_inputs, prepare_dataset
from .metrics import score_run
from .runner import run_baseline


def _prepared_paths(prepared_dir: str | Path) -> tuple[Path, Path]:
    root = Path(prepared_dir).resolve()
    return root / "model_inputs" / "inputs.jsonl", root / "scoring" / "labels.jsonl"


def _override_run_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    section = config.get("my_dataset_eval")
    if not isinstance(section, dict):
        raise ValueError("Config must contain a my_dataset_eval mapping")
    for argument, key in (
        (args.shard_id, "shard_id"),
        (args.num_shards, "num_shards"),
        (args.limit_groups, "limit_groups"),
        (args.output_dir, "output_dir"),
    ):
        if argument is not None:
            section[key] = argument


def _inventory(inputs_path: str | Path) -> dict[str, Any]:
    rows = load_model_inputs(inputs_path)
    group_sizes = Counter(str(row["group_id"]) for row in rows)
    return {
        "inputs_path": str(Path(inputs_path).resolve()),
        "inputs_sha256": sha256_file(inputs_path),
        "num_examples": len(rows),
        "num_groups": len(group_sizes),
        "group_size_distribution": dict(
            sorted(Counter(group_sizes.values()).items())
        ),
        "task_counts": dict(sorted(Counter(str(row["task_id"]) for row in rows).items())),
        "task_family_counts": dict(
            sorted(Counter(str(row["task_family"]) for row in rows).items())
        ),
        "evaluation_splits": dict(
            sorted(Counter(str(row["evaluation_split"]) for row in rows).items())
        ),
    }


def _positive_count_or_auto(value: str) -> int | None:
    if value.strip().casefold() == "auto":
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected count must be a positive integer or 'auto'"
        ) from exc
    if result < 1:
        raise argparse.ArgumentTypeError("expected count must be positive")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python mydata_bench/run_my_dataset.py")
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="freeze label-free inputs and scoring labels")
    prepare.add_argument("--config", required=True)

    audit = commands.add_parser("audit", help="verify a prepared dataset")
    audit.add_argument("--prepared-dir", required=True)

    inventory = commands.add_parser("inventory", help="summarize label-free model inputs")
    inventory.add_argument("--inputs", required=True)

    run = commands.add_parser("run", help="run one label-blind model baseline")
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--shard-id", type=int)
    run.add_argument("--num-shards", type=int)
    run.add_argument("--limit-groups", type=int)
    run.add_argument("--output-dir")

    score = commands.add_parser("score", help="join labels and compute group metrics")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--inputs", required=True)
    score.add_argument("--labels", required=True)
    score.add_argument("--bootstrap-samples", type=int, default=10_000)
    score.add_argument("--seed", type=int, default=20260803)

    split = commands.add_parser("split", help="freeze group-level discovery/validation/test partitions")
    split.add_argument("--config", required=True)

    roles = commands.add_parser("roles", help="parse label-free instruction semantic roles")
    roles.add_argument("--config", required=True)

    freeze = commands.add_parser("freeze", help="write the machine-readable white-box preregistration")
    freeze.add_argument("--config", required=True)

    ground = commands.add_parser("ground-prepare", help="build processor-aligned grounding requests")
    ground.add_argument("--config", required=True)

    ground_propose = commands.add_parser(
        "ground-propose", help="run SAM3 candidates without auto-accepting instances"
    )
    ground_propose.add_argument("--config", required=True)
    ground_propose.add_argument("--retry-failed", action="store_true")

    track_prepare = commands.add_parser("ground-track-prepare", help="freeze tracked grounding requests")
    track_prepare.add_argument("--config", required=True)
    track_run = commands.add_parser("ground-track-run", help="run frozen SAM3 tracks")
    track_run.add_argument("--config", required=True)
    track_run.add_argument("--retry-failed", action="store_true")
    track_manual = commands.add_parser("ground-track-manual", help="run reviewer-supplied manual anchors")
    track_manual.add_argument("--config", required=True)
    track_manual.add_argument("--anchors", required=True)
    track_manual.add_argument("--output", required=True)
    track_manual.add_argument("--retry-failed", action="store_true")

    ground_audit = commands.add_parser("ground-audit", help="audit completed human grounding reviews")
    ground_audit.add_argument("--requests", required=True)
    ground_audit.add_argument("--reviews", required=True)
    ground_audit.add_argument("--output-dir", required=True)
    source = ground_audit.add_mutually_exclusive_group(required=True)
    source.add_argument("--proposals")
    source.add_argument("--tracking-artifact")
    ground_audit.add_argument("--manual-tracking-artifact")

    attention = commands.add_parser("attention-prepare", help="build model-specific causal manifests")
    attention.add_argument("--config", required=True)

    assume_grounding = commands.add_parser(
        "assume-grounding",
        help="build explicitly unreviewed exploratory grounding decisions",
    )
    assume_grounding.add_argument("--config", required=True)

    ranking_cohort = commands.add_parser(
        "ranking-cohort",
        help="freeze nested external N=5/10/20 ranking inputs",
    )
    ranking_cohort.add_argument("--config", required=True)

    matrix = commands.add_parser(
        "matrix",
        help="run the frozen exploratory ranking-size by steering-size matrix",
    )
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--retry-failed", action="store_true")

    score_matrix = commands.add_parser(
        "score-matrix",
        help="join labels and score a complete exploratory matrix",
    )
    score_matrix.add_argument("--records", required=True)
    score_matrix.add_argument("--labels", required=True)
    score_matrix.add_argument("--selection-manifest", required=True)
    score_matrix.add_argument("--output-dir", required=True)
    score_matrix.add_argument(
        "--expected-count",
        type=_positive_count_or_auto,
        default=755,
        help="positive integer, or 'auto' for a reviewed subset",
    )
    score_matrix.add_argument(
        "--evaluation-manifest",
        help=(
            "JSONL whose IDs define a same-population rescore; required for "
            "human-reviewed records"
        ),
    )
    score_matrix.add_argument(
        "--reference-records",
        help=(
            "prior unreviewed matrix records; required for human-reviewed "
            "records and compared at exact output-field parity"
        ),
    )

    rank = commands.add_parser("rank", help="rank heads on the frozen discovery partition")
    rank.add_argument("--config", required=True)
    rank.add_argument("--retry-failed", action="store_true")

    equivalence = commands.add_parser(
        "equivalence", help="run the no-hook/bias-zero attention-runtime gate"
    )
    equivalence.add_argument("--config", required=True)

    steer = commands.add_parser("steer", help="run paired controls on validation or frozen test")
    steer.add_argument("--config", required=True)
    steer.add_argument("--retry-failed", action="store_true")

    score_steering_parser = commands.add_parser(
        "score-steering", help="join labels after intervention and compute paired metrics"
    )
    score_steering_parser.add_argument("--records", required=True)
    score_steering_parser.add_argument("--labels", required=True)
    score_steering_parser.add_argument("--output-dir", required=True)
    score_steering_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    score_steering_parser.add_argument("--seed", type=int, default=20260803)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        destination = prepare_dataset(load_config(args.config))
        print(destination)
        return
    if args.command == "audit":
        inputs, labels = _prepared_paths(args.prepared_dir)
        result = audit_prepared(inputs, labels)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.command == "inventory":
        print(json.dumps(_inventory(args.inputs), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "run":
        config = load_config(args.config)
        _override_run_config(config, args)
        print(run_baseline(config, dry_run=args.dry_run, retry_failed=args.retry_failed))
        return
    if args.command == "score":
        metrics = score_run(
            args.run_dir,
            inputs_path=args.inputs,
            labels_path=args.labels,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        completion = metrics["completion"]
        print(
            f"ready={completion['formal_scoring_ready']} "
            f"examples={metrics['num_examples']} groups={metrics['num_groups']} "
            f"pairs={metrics['num_pairs']}"
        )
        return
    if args.command == "split":
        from .splits import build_split

        print(build_split(load_config(args.config)))
        return
    if args.command == "roles":
        from .roles import build_roles

        print(build_roles(load_config(args.config)))
        return
    if args.command == "freeze":
        from .protocol_freeze import freeze_protocol

        print(freeze_protocol(load_config(args.config)))
        return
    if args.command == "ground-prepare":
        from .grounding_manifest import build_grounding_manifest

        print(build_grounding_manifest(load_config(args.config)))
        return
    if args.command == "ground-propose":
        from .grounding_manifest import propose_grounding

        print(
            propose_grounding(
                load_config(args.config), retry_failed=args.retry_failed
            )
        )
        return
    if args.command == "ground-track-prepare":
        from .tracked_grounding import build_tracked_grounding_requests

        print(build_tracked_grounding_requests(load_config(args.config)))
        return
    if args.command == "ground-track-run":
        from .tracked_grounding import run_tracked_grounding

        print(
            run_tracked_grounding(
                load_config(args.config), retry_failed=args.retry_failed
            )
        )
        return
    if args.command == "ground-track-manual":
        from .tracked_grounding import run_manual_retracks

        print(
            run_manual_retracks(
                load_config(args.config),
                args.anchors,
                args.output,
                retry_failed=args.retry_failed,
            )
        )
        return
    if args.command == "ground-audit":
        from .grounding_manifest import audit_grounding_review

        result = audit_grounding_review(
            args.requests,
            args.reviews,
            args.output_dir,
            proposals_path=args.proposals,
            tracking_artifact_path=args.tracking_artifact,
            manual_tracking_artifact_path=args.manual_tracking_artifact,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.command == "attention-prepare":
        from .attention_manifest import build_attention_manifests

        print(build_attention_manifests(load_config(args.config)))
        return
    if args.command == "assume-grounding":
        from .assumed_grounding import build_assumed_grounding

        print(build_assumed_grounding(load_config(args.config)))
        return
    if args.command == "ranking-cohort":
        from .ranking_cohort import freeze_ranking_cohort

        print(freeze_ranking_cohort(load_config(args.config)))
        return
    if args.command == "matrix":
        from .exploratory_matrix import run_exploratory_matrix

        print(
            run_exploratory_matrix(
                load_config(args.config), retry_failed=args.retry_failed
            )
        )
        return
    if args.command == "score-matrix":
        from .exploratory_metrics import score_exploratory_matrix

        result = score_exploratory_matrix(
            args.records,
            args.labels,
            args.selection_manifest,
            args.output_dir,
            expected_count=args.expected_count,
            evaluation_manifest_path=args.evaluation_manifest,
            reference_records_path=args.reference_records,
        )
        print(
            f"complete={result['completion']['complete']} "
            f"examples={result['completion']['example_count']} "
            f"conditions={len(result['conditions'])}"
        )
        return
    if args.command == "rank":
        from .causal_runner import rank_heads

        print(rank_heads(load_config(args.config), retry_failed=args.retry_failed))
        return
    if args.command == "equivalence":
        from .causal_runner import check_zero_bias_equivalence

        print(check_zero_bias_equivalence(load_config(args.config)))
        return
    if args.command == "steer":
        from .causal_runner import steer

        print(steer(load_config(args.config), retry_failed=args.retry_failed))
        return
    if args.command == "score-steering":
        from .causal_metrics import score_steering

        result = score_steering(
            args.records,
            args.labels,
            args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print(
            f"examples={result['complete_examples']} "
            f"correction={result['fail_correction_rate']:.6f} "
            f"harm={result['suc_harm_rate']:.6f} "
            f"net={result['balanced_net_correction']:.6f}"
        )
        return
    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
