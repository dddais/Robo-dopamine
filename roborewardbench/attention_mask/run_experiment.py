#!/usr/bin/env python3
"""Run paired baseline and attention-mask interventions on held-out examples."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset import (
    examples_fingerprint,
    load_attention_examples,
    load_split_partition,
    sha256_file,
)
from .io import (
    file_identity,
    initialize_manifest,
    model_identity,
    object_fingerprint,
    read_jsonl,
    strict_dump,
    strict_jsonl_append,
)
from .masking import (
    Head,
    INTERVENTION_MODES,
    intervention_positions,
    matched_wrong_position_set,
    registered_mask_hooks,
    target_position_set,
    validate_heads,
)
from .modeling import (
    ensure_blank_goal,
    generate_score,
    load_grm,
    model_dimensions,
    prepare_inputs,
)


CONDITIONS = (
    "candidate_target",
    "candidate_wrong",
    "low_rank_target",
    "all_target",
)
RESULT_SCHEMA_VERSION = 1


def _csv_ints(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return list(dict.fromkeys(result))


def _csv_floats(value: str) -> list[float]:
    result = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not result or any(not math.isfinite(item) or item < 0 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated finite non-negative numbers")
    return list(dict.fromkeys(result))


def _csv_conditions(value: str) -> list[str]:
    result = [part.strip() for part in value.split(",") if part.strip()]
    invalid = set(result) - set(CONDITIONS)
    if not result or invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown conditions {sorted(invalid)}; expected values from {CONDITIONS}"
        )
    return list(dict.fromkeys(result))


def _ranking_rows(ranking: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    ranking_name = str(ranking.get("default_ranking", "mean"))
    rows = (ranking.get("rankings") or {}).get(ranking_name)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Head ranking has no rankings.{ranking_name}")
    skip = int(ranking.get("skip_early_layers", 0))
    filtered = [row for row in rows if int(row["layer"]) >= skip]
    if not filtered:
        raise ValueError("Head ranking is empty after skip_early_layers")
    return filtered


def choose_head_groups(
    ranking: Mapping[str, Any],
    top_k: int,
    *,
    num_layers: int,
    num_heads: int,
) -> tuple[list[Head], list[Head]]:
    rows = _ranking_rows(ranking)
    if len(rows) < int(top_k):
        raise ValueError(f"Ranking only has {len(rows)} eligible heads; requested top_k={top_k}")
    candidate = [
        Head(layer=int(row["layer"]), head=int(row["head"]))
        for row in rows[: int(top_k)]
    ]
    candidate_set = {(head.layer, head.head) for head in candidate}
    low_rank: list[Head] = []
    for row in reversed(rows):
        head = Head(layer=int(row["layer"]), head=int(row["head"]))
        if (head.layer, head.head) in candidate_set:
            continue
        low_rank.append(head)
        if len(low_rank) == int(top_k):
            break
    if len(low_rank) != int(top_k):
        raise ValueError(f"Could not select {top_k} disjoint low-ranked control heads")
    return (
        validate_heads(candidate, num_layers=num_layers, num_heads=num_heads),
        validate_heads(low_rank, num_layers=num_layers, num_heads=num_heads),
    )


def validate_ranking_linkage(
    ranking: Mapping[str, Any],
    *,
    evaluation_ids: Sequence[str],
    split_sha256: str,
    target_role: str,
    external_fixed_ranking: bool,
    allow_incomplete_ranking: bool,
) -> None:
    """Validate either an in-split discovery ranking or an external fixed set."""

    if external_fixed_ranking:
        # External transfer rankings predate the held-out manifest schema. Their
        # immutable file identity is recorded in the run manifest, while model
        # dimensions and every selected head are validated after model loading.
        _ranking_rows(ranking)
        return
    if not ranking.get("complete_discovery_partition") and not allow_incomplete_ranking:
        raise ValueError(
            "Head ranking did not cover the complete discovery partition. "
            "Use it only for a smoke test with --allow-incomplete-ranking."
        )
    if str(ranking.get("target_role")) != target_role:
        raise ValueError(
            f"Ranking target_role={ranking.get('target_role')!r} does not match "
            f"experiment target_role={target_role!r}"
        )
    discovery_ids = {str(value) for value in ranking.get("discovery_ids", [])}
    if not discovery_ids:
        raise ValueError("Ranking has no discovery_ids; leakage cannot be checked")
    overlap = discovery_ids & {str(value) for value in evaluation_ids}
    if overlap:
        raise ValueError(
            "Head discovery and evaluation examples overlap: "
            + ", ".join(sorted(overlap)[:10])
        )
    ranking_split = ranking.get("split_manifest") or {}
    if ranking_split.get("sha256") != split_sha256:
        raise ValueError("Head ranking was produced from a different split manifest")


def validate_ranking_model(
    ranking: Mapping[str, Any],
    *,
    current_model_identity: Mapping[str, Any],
    num_layers: int,
    num_heads: int,
    external_fixed_ranking: bool,
) -> None:
    if external_fixed_ranking:
        ranked_layers = ranking.get("num_layers")
        ranked_heads = ranking.get("num_heads")
        if ranked_layers is not None and int(ranked_layers) != int(num_layers):
            raise ValueError(
                f"External ranking has num_layers={ranked_layers}, model has {num_layers}"
            )
        if ranked_heads is not None and int(ranked_heads) != int(num_heads):
            raise ValueError(
                f"External ranking has num_heads={ranked_heads}, model has {num_heads}"
            )
        return
    ranking_model = ranking.get("model") or {}
    if (
        (ranking_model.get("config") or {}).get("sha256")
        != (current_model_identity.get("config") or {}).get("sha256")
        or ranking_model.get("inventory_fingerprint")
        != current_model_identity.get("inventory_fingerprint")
    ):
        raise ValueError("Head ranking and steering run use different model files")


def all_model_heads(num_layers: int, num_heads: int) -> list[Head]:
    return [
        Head(layer=layer, head=head)
        for layer in range(num_layers)
        for head in range(num_heads)
    ]


def result_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("example_id")),
        str(row.get("condition")),
        row.get("top_k"),
        float(row.get("swap_bias", 0.0)),
        str(row.get("intervention")),
        str(row.get("target_role")),
    )


def _run_one(
    torch,
    model,
    processor,
    inputs,
    *,
    heads: Sequence[Head] | None,
    position_set,
    num_query_heads: int,
    intervention: str,
    swap_bias: float,
    decode_only: bool,
    max_new_tokens: int,
) -> tuple[float | None, str, str | None]:
    if heads is None:
        return generate_score(
            torch,
            model,
            processor,
            inputs,
            max_new_tokens=max_new_tokens,
        )
    suppress, boost = intervention_positions(intervention, position_set)
    with registered_mask_hooks(
        model,
        heads=heads,
        suppress_positions=suppress,
        boost_positions=boost,
        num_query_heads=num_query_heads,
        swap_bias=swap_bias,
        decode_only=decode_only,
    ):
        return generate_score(
            torch,
            model,
            processor,
            inputs,
            max_new_tokens=max_new_tokens,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--head-ranking", required=True)
    parser.add_argument(
        "--external-fixed-ranking",
        action="store_true",
        help=(
            "Treat --head-ranking as a frozen ranking produced outside this split. "
            "Skips discovery/split linkage checks but still validates dimensions and heads."
        ),
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-mode", default="manual_correct_ready")
    parser.add_argument("--partition", default="evaluation", choices=["evaluation"])
    parser.add_argument(
        "--target-role",
        default="both",
        choices=["before", "after", "both", "after_high"],
    )
    parser.add_argument("--top-ks", type=_csv_ints, default=_csv_ints("8,64"))
    parser.add_argument("--biases", type=_csv_floats, default=_csv_floats("0,2,4,6"))
    parser.add_argument(
        "--conditions",
        type=_csv_conditions,
        default=_csv_conditions(",".join(CONDITIONS)),
    )
    parser.add_argument(
        "--intervention",
        default="boost_suppress",
        choices=INTERVENTION_MODES,
    )
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--max-pixels", type=int, default=76800)
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument(
        "--allow-incomplete-ranking",
        action="store_true",
        help="Allow a smoke ranking made from only part of discovery; never use for final claims.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    ranking_path = Path(args.head_ranking).expanduser().resolve()
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))

    evaluation_ids, split_data = load_split_partition(args.split_manifest, args.partition)
    expected_mode = str(split_data[args.partition].get("selection_mode"))
    if args.selection_mode != expected_mode:
        raise ValueError(
            f"selection_mode={args.selection_mode} disagrees with split manifest "
            f"{args.partition}.selection_mode={expected_mode}"
        )
    current_split_sha = sha256_file(args.split_manifest)
    validate_ranking_linkage(
        ranking,
        evaluation_ids=evaluation_ids,
        split_sha256=current_split_sha,
        target_role=args.target_role,
        external_fixed_ranking=args.external_fixed_ranking,
        allow_incomplete_ranking=args.allow_incomplete_ranking,
    )

    examples_all = load_attention_examples(
        args.grounding_dir,
        selection_mode=args.selection_mode,
        example_ids=evaluation_ids,
    )
    expected_fingerprint = split_data["evaluation"].get("dataset_fingerprint")
    if examples_fingerprint(examples_all) != expected_fingerprint:
        raise ValueError("Evaluation grounding records changed after the split was frozen")
    examples = [
        example
        for index, example in enumerate(examples_all)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max-examples must be positive")
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("This shard has no evaluation examples")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    run_family_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "grounding_dir": str(Path(args.grounding_dir).expanduser().resolve()),
        "split_manifest": file_identity(args.split_manifest),
        "head_ranking": file_identity(ranking_path),
        "head_set_mode": (
            "external_fixed" if args.external_fixed_ranking else "in_split_discovery"
        ),
        "model": model_identity(args.model_path),
        "code": {
            "run_experiment": file_identity(Path(__file__)),
            "dataset": file_identity(Path(__file__).with_name("dataset.py")),
            "masking": file_identity(Path(__file__).with_name("masking.py")),
            "modeling": file_identity(Path(__file__).with_name("modeling.py")),
        },
        "selection_mode": args.selection_mode,
        "partition": args.partition,
        "evaluation_dataset_fingerprint": examples_fingerprint(examples_all),
        "target_role": args.target_role,
        "top_ks": args.top_ks,
        "biases": args.biases,
        "conditions": args.conditions,
        "intervention": args.intervention,
        "decode_only": args.decode_only,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "num_shards": args.num_shards,
    }
    run_family_signature = object_fingerprint(run_family_payload)
    manifest = initialize_manifest(
        manifest_path,
        {
            **run_family_payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_family_signature": run_family_signature,
            "shard_index": args.shard_index,
            "selected_ids": [example.example_id for example in examples],
            "complete_shard": args.max_examples is None,
            "runtime_label_fields_provided_to_model": [],
            "results_path": str(results_path),
        },
    )
    run_signature = str(manifest["run_signature"])

    completed: set[tuple[Any, ...]] = set()
    for row in read_jsonl(results_path):
        if row.get("run_signature") != run_signature:
            raise ValueError(f"{results_path} contains records from another run signature")
        key = result_key(row)
        if key in completed:
            raise ValueError(f"{results_path} contains duplicate result key {key}")
        completed.add(key)

    print(f"[steer] loading {args.model_path}", flush=True)
    torch, model, processor, dtype = load_grm(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        output_attentions=False,
    )
    num_layers, num_heads, spatial_merge = model_dimensions(model)
    current_model_identity = model_identity(args.model_path)
    validate_ranking_model(
        ranking,
        current_model_identity=current_model_identity,
        num_layers=num_layers,
        num_heads=num_heads,
        external_fixed_ranking=args.external_fixed_ranking,
    )
    head_groups = {
        top_k: choose_head_groups(
            ranking,
            top_k,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        for top_k in args.top_ks
    }
    every_head = all_model_heads(num_layers, num_heads)
    blank_goal = ensure_blank_goal(output_dir / "blank_goal.png")

    total_examples = len(examples)
    for example_index, example in enumerate(examples, 1):
        print(
            f"[steer] shard {args.shard_index}/{args.num_shards} "
            f"{example_index}/{total_examples} {example.example_id}",
            flush=True,
        )
        item = example.model_item(blank_goal)
        if "reward" in item:
            raise AssertionError("Reward leaked into the model item")
        inputs, spans = prepare_inputs(torch, model, processor, item, dtype)
        target_positions = target_position_set(
            spans,
            before_bbox=example.before_bbox,
            after_bbox=example.after_bbox,
            before_image_size=example.before_image_size,
            after_image_size=example.after_image_size,
            spatial_merge_size=spatial_merge,
            target_role=args.target_role,
        )
        wrong_positions = None
        wrong_control_error = None
        try:
            wrong_positions = matched_wrong_position_set(
                spans,
                target_positions,
                spatial_merge_size=spatial_merge,
                seed=args.seed + int(example.grounding_fingerprint[:8], 16),
            )
        except ValueError as exc:
            # A same-size non-overlapping region is mathematically impossible
            # when the target occupies more than half of a visual grid.  Do not
            # silently shrink the control: omit candidate_wrong for that sample
            # and let metrics report the reduced paired coverage.
            wrong_control_error = str(exc)
            print(f"  [wrong-control unavailable] {wrong_control_error}", flush=True)

        jobs: list[tuple[str, int | None, float, Sequence[Head] | None, Any]] = [
            ("baseline", None, 0.0, None, target_positions)
        ]
        for bias in args.biases:
            for top_k in args.top_ks:
                candidates, low_rank = head_groups[top_k]
                if "candidate_target" in args.conditions:
                    jobs.append(
                        ("candidate_target", top_k, bias, candidates, target_positions)
                    )
                if "candidate_wrong" in args.conditions and wrong_positions is not None:
                    jobs.append(
                        ("candidate_wrong", top_k, bias, candidates, wrong_positions)
                    )
                if "low_rank_target" in args.conditions:
                    jobs.append(
                        ("low_rank_target", top_k, bias, low_rank, target_positions)
                    )
            if "all_target" in args.conditions:
                jobs.append(("all_target", None, bias, every_head, target_positions))

        for condition, top_k, bias, heads, position_set in jobs:
            key = (
                example.example_id,
                condition,
                top_k,
                float(bias),
                args.intervention,
                args.target_role,
            )
            if key in completed:
                continue
            started = time.monotonic()
            score: float | None = None
            raw_output = ""
            error: str | None = None
            status = "ok"
            try:
                score, raw_output, parse_error = _run_one(
                    torch,
                    model,
                    processor,
                    inputs,
                    heads=heads,
                    position_set=position_set,
                    num_query_heads=num_heads,
                    intervention=args.intervention,
                    swap_bias=float(bias),
                    decode_only=args.decode_only,
                    max_new_tokens=args.max_new_tokens,
                )
                if parse_error is not None:
                    status = "invalid_parse"
                    error = parse_error
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            record = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "run_signature": run_signature,
                "run_family_signature": run_family_signature,
                "example_id": example.example_id,
                "subset": example.subset,
                "grounding_fingerprint": example.grounding_fingerprint,
                "target_phrase": example.target_phrase,
                "selection_mode": args.selection_mode,
                "condition": condition,
                "top_k": top_k,
                "swap_bias": float(bias),
                "intervention": args.intervention,
                "target_role": args.target_role,
                "decode_only": args.decode_only,
                "score": score,
                "raw_output": raw_output,
                "status": status,
                "error": error,
                "selected_head_count": 0 if heads is None else len(heads),
                "target_token_count": len(position_set.target),
                "other_image_token_count": len(position_set.other_image),
                "wrong_control_available": wrong_positions is not None,
                "wrong_control_error": wrong_control_error,
                "elapsed_seconds": time.monotonic() - started,
            }
            # Labels are intentionally absent. Metrics joins them after all
            # generation has completed.
            if "reward" in record:
                raise AssertionError("Reward leaked into an experiment record")
            strict_jsonl_append(results_path, record)
            completed.add(key)
            print(
                f"  {condition} top_k={top_k} bias={bias:g}: "
                f"status={status} score={score}",
                flush=True,
            )

    strict_dump(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_signature": run_signature,
            "run_family_signature": run_family_signature,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "complete_shard": args.max_examples is None,
            "selected_ids": [example.example_id for example in examples],
            "result_record_count": len(completed),
        },
        output_dir / "completion.json",
    )
    print(f"[steer] complete: {results_path}", flush=True)


if __name__ == "__main__":
    main()
