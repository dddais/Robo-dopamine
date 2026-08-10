from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config, section
from ..data import inventory as build_inventory
from ..data import load_episodes
from ..io import latest_by_id, read_jsonl, write_json
from ..metrics import clustered_stratified_bootstrap, compute_metrics
from .paper_protocol import published_metric_report
from .runner import _requested_example_ids, run


def _counterfactual_reward1_metrics(
    rows: list[dict], *, counterfactual_ids: set[str]
) -> dict:
    """Score the frozen exact counterfactual cohort after model inference.

    The ID set is derived only during the metrics stage.  It is never passed to
    the model, prompt construction, video sampler, or decoding path.
    """
    selected = [dict(row) for row in rows if row.get("example_id") in counterfactual_ids]
    computed = compute_metrics(selected)
    actual_ids = {str(row["example_id"]) for row in selected}
    invalid = [row for row in selected if row.get("status") != "ok"]
    return {
        "definition": (
            "Exact frozen RoboRewardBench_counterfactual_reward1 ID cohort; "
            "not all reward=1 examples in the full benchmark."
        ),
        "expected_count": len(counterfactual_ids),
        "record_count": len(actual_ids),
        "missing_example_ids": sorted(counterfactual_ids - actual_ids),
        "invalid_count": len(invalid),
        "formal_scoring_ready": actual_ids == counterfactual_ids and not invalid,
        "micro": computed["micro"],
        "macro_subset_mae": computed["macro_subset_mae"],
        "prediction_counts": computed["prediction_counts"],
        "confusion_matrix": computed["confusion_matrix"],
        "predicted_one_rate": computed["reward1"]["predicted_one_rate"],
        "overestimated_rate": computed["reward1"]["overestimated_rate"],
        "label_migration": computed["reward1"]["label_migration"],
        "by_subset": computed["by"]["subset"],
    }


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
    evaluation = {}
    if manifests:
        import json

        evaluation = json.loads(manifests[0].read_text(encoding="utf-8")).get(
            "config", {}
        ).get("roboreward_eval", {})
        if evaluation.get("dataset_root"):
            expected_ids = {
                row.example_id
                for row in load_episodes(
                    evaluation["dataset_root"], evaluation.get("split", "test"), compute_hash=False
                )
            }
            requested_ids = _requested_example_ids(evaluation)
            if requested_ids:
                expected_ids &= requested_ids
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
    counterfactual_root = evaluation.get(
        "counterfactual_dataset_root_for_metrics_only"
    )
    counterfactual_metrics = None
    if counterfactual_root:
        counterfactual_ids = {
            row.example_id
            for row in load_episodes(
                counterfactual_root,
                evaluation.get("split", "test"),
                compute_hash=False,
            )
        }
        if expected_ids is not None and not counterfactual_ids <= expected_ids:
            raise ValueError(
                "counterfactual_dataset_root_for_metrics_only is not a subset "
                "of the evaluated dataset"
            )
        counterfactual_metrics = _counterfactual_reward1_metrics(
            rows, counterfactual_ids=counterfactual_ids
        )
    metrics.update(
        {
            "adapter_metric": evaluation.get("input_representation", "video")
            != "video",
            "official_native_discrete_output": True,
            "checkpoint_native_input": (
                evaluation.get("input_representation", "video") == "video"
                and evaluation.get("video_sampling_mode")
                == "checkpoint_native_video"
            ),
            "input_representation": evaluation.get(
                "input_representation", "video"
            ),
            "model_family": "RoboReward",
            "prediction_contract": "ANSWER: <1-5>",
            "completion": completion,
        }
    )
    # This is intentionally a second, explicitly named report.  ``micro`` is
    # useful for cohort diagnostics, whereas the paper ranks models with this
    # unweighted mean over its fixed 23 subset groups.
    paper_report = published_metric_report(rows)
    metrics["paper_protocol"] = paper_report
    if counterfactual_metrics is not None:
        metrics["counterfactual_reward1"] = counterfactual_metrics
    if bootstrap_samples:
        metrics["macro_subset_mae_bootstrap"] = clustered_stratified_bootstrap(
            rows,
            lambda draw: float(compute_metrics(draw)["macro_subset_mae"]),
            samples=bootstrap_samples,
        )
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "paper_protocol_metrics.json", paper_report)
    write_json(run_dir / "invalid.json", invalid)
    write_json(run_dir / "completion.json", completion)
    micro = metrics["micro"]
    (run_dir / "metrics.md").write_text(
        "# Native RoboReward Evaluation Metrics\n\n"
        f"- Formal scoring ready: `{completion['formal_scoring_ready']}`\n"
        f"- Valid / invalid: {metrics['num_valid']} / {metrics['num_invalid']}\n"
        f"- Macro subset MAE: {metrics['macro_subset_mae']}\n"
        f"- Micro MAE: {micro.get('mae')}\n"
        f"- Exact / within-one accuracy: {micro.get('exact_accuracy')} / {micro.get('within_one_accuracy')}\n",
        encoding="utf-8",
    )
    with (run_dir / "metrics.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## RoboRewardBench 论文 Overall 口径\n\n"
            "- 定义：23 个固定子集的 MAE 等权平均（不是 micro MAE）。\n"
            f"- 论文指标可比：`{paper_report['paper_metric_comparable']}`\n"
            f"- Group-wise MAE：{paper_report['groupwise_mae']}\n"
            f"- RoboArena / OXE group-wise MAE：{paper_report['roboarena_mae']} / {paper_report['oxe_groupwise_mae']}\n"
            f"- 相对论文 RoboReward-8B 0.665 的差值：{paper_report['difference_from_paper_reported_roboreward_8b']}\n"
        )
        if paper_report["validation_errors"]:
            handle.write("- 不可比原因：" + "; ".join(paper_report["validation_errors"]) + "\n")
    if counterfactual_metrics is not None:
        with (run_dir / "metrics.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Exact Counterfactual reward=1 Cohort\n\n"
                f"- Formal scoring ready: `{counterfactual_metrics['formal_scoring_ready']}`\n"
                f"- Valid / invalid: {counterfactual_metrics['micro'].get('n')} / "
                f"{counterfactual_metrics['invalid_count']}\n"
                f"- Predicted reward=1 rate: {counterfactual_metrics['predicted_one_rate']}\n"
                f"- Overestimation rate: {counterfactual_metrics['overestimated_rate']}\n"
            )
    return metrics


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python mydata_bench/run_roboreward_eval.py")
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
        evaluation = section(config, "roboreward_eval")
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


if __name__ == "__main__":
    main()
