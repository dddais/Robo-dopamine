"""Deterministic, label-free model-input cohorts constructed offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .data import load_episodes
from .io import (
    object_fingerprint,
    provenance,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def freeze_reward_cohort(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    reward: int,
    split: str = "test",
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Freeze every episode with one source reward without exposing it downstream.

    The reward filter is allowed only in this construction step.  The emitted
    ``episodes.jsonl`` and ``example_ids.json`` deliberately omit reward and
    gpt5_mini_check; parser, grounder, ranking, and GRM use only those files.
    """
    root = Path(dataset_root).resolve()
    destination = Path(output_dir).resolve()
    records = [episode for episode in load_episodes(root, split) if episode.reward == reward]
    records.sort(key=lambda episode: episode.example_id)
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} episodes with reward={reward}, found {len(records)}"
        )
    model_safe_rows = [episode.model_payload() for episode in records]
    example_ids = [row["example_id"] for row in model_safe_rows]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("Duplicate example_id in frozen cohort")
    destination.mkdir(parents=True, exist_ok=True)
    episodes_path = destination / "episodes.jsonl"
    ids_path = destination / "example_ids.json"
    write_jsonl(episodes_path, model_safe_rows)
    write_json(ids_path, example_ids)
    summary = {
        "cohort": f"all_reward_{reward}_{split}",
        "selection_rule": "include every source episode whose metadata reward equals the requested value; lexical example_id order",
        "dataset_root": str(root),
        "split": split,
        "source_reward_used_only_for_offline_selection": reward,
        "selected_count": len(records),
        "selected_subsets": dict(sorted(Counter(row.subset for row in records).items())),
        "source_metadata_sha256": sha256_file(
            root / "metadata.jsonl"
            if (root / "metadata.jsonl").is_file()
            else root / split / "metadata.jsonl"
        ),
        "model_safe_episode_manifest": str(episodes_path),
        "model_safe_episode_manifest_fingerprint": object_fingerprint(model_safe_rows),
        "example_ids_file": str(ids_path),
        "note": (
            "The two emitted model-input artifacts contain no reward or gpt5_mini_check. "
            "Use the source reward only after inference in metrics."
        ),
    }
    write_json(destination / "cohort_summary.json", summary)
    write_json(
        destination / "cohort_manifest.json",
        {**provenance(sys.argv, {}, Path(__file__).resolve().parents[1]), "summary": summary},
    )
    return summary


def freeze_audited_cohort(
    dataset_root: str | Path,
    audit_final: str | Path,
    output_dir: str | Path,
    *,
    split: str = "all",
) -> dict[str, Any]:
    """Freeze every human-audited, formally eligible grounding example."""
    root = Path(dataset_root).resolve()
    audit_path = Path(audit_final).resolve()
    destination = Path(output_dir).resolve()
    eligible = {
        str(row["example_id"])
        for row in read_jsonl(audit_path)
        if row.get("formal_eligible") is True
    }
    episodes = {
        episode.example_id: episode
        for episode in load_episodes(root, split)
    }
    missing = eligible - set(episodes)
    if missing:
        raise ValueError(f"Audit contains unknown dataset IDs: {sorted(missing)[:5]}")
    rows = [episodes[example_id].model_payload() for example_id in sorted(eligible)]
    destination.mkdir(parents=True, exist_ok=True)
    episodes_path = destination / "episodes.jsonl"
    ids_path = destination / "example_ids.json"
    write_jsonl(episodes_path, rows)
    write_json(ids_path, [row["example_id"] for row in rows])
    summary = {
        "cohort": "human_audited_formal_grounding",
        "selection_rule": "audit_final.formal_eligible is true",
        "dataset_root": str(root),
        "split": split,
        "selected_count": len(rows),
        "audit_final": str(audit_path),
        "audit_final_sha256": sha256_file(audit_path),
        "model_safe_episode_manifest": str(episodes_path),
        "model_safe_episode_manifest_fingerprint": object_fingerprint(rows),
        "example_ids_file": str(ids_path),
        "labels_model_facing": False,
    }
    write_json(destination / "cohort_summary.json", summary)
    write_json(
        destination / "cohort_manifest.json",
        {**provenance(sys.argv, {}, Path(__file__).resolve().parents[1]), "summary": summary},
    )
    return summary


def freeze_auto_grounded_cohort(
    dataset_root: str | Path,
    grounding_run: str | Path,
    output_dir: str | Path,
    *,
    split: str = "all",
) -> dict[str, Any]:
    """Freeze examples whose automatic SAM3 tracking has both endpoints."""
    root = Path(dataset_root).resolve()
    run = Path(grounding_run).resolve()
    records_path = run / "grounding.jsonl"
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(records_path):
        example_id = row.get("example_id")
        frame = row.get("frame")
        if isinstance(example_id, str) and frame in {"first", "last"}:
            latest[(example_id, frame)] = row
    endpoint_ok: dict[str, set[str]] = {}
    for (example_id, frame), row in latest.items():
        if row.get("status") == "ok":
            endpoint_ok.setdefault(example_id, set()).add(frame)
    eligible = {
        example_id
        for example_id, frames in endpoint_ok.items()
        if frames == {"first", "last"}
    }
    episodes = {
        episode.example_id: episode for episode in load_episodes(root, split)
    }
    unknown = eligible - set(episodes)
    if unknown:
        raise ValueError(f"Grounding contains unknown dataset IDs: {sorted(unknown)[:5]}")
    rows = [episodes[example_id].model_payload() for example_id in sorted(eligible)]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    episodes_path = destination / "episodes.jsonl"
    ids_path = destination / "example_ids.json"
    write_jsonl(episodes_path, rows)
    write_json(ids_path, [row["example_id"] for row in rows])
    summary = {
        "cohort": "automatic_sam3_tracking_dual_endpoint",
        "selection_rule": "latest automatic grounding status is ok at first and last endpoints",
        "dataset_root": str(root),
        "split": split,
        "selected_count": len(rows),
        "grounding_run": str(run),
        "grounding_records_sha256": sha256_file(records_path),
        "model_safe_episode_manifest": str(episodes_path),
        "model_safe_episode_manifest_fingerprint": object_fingerprint(rows),
        "example_ids_file": str(ids_path),
        "labels_model_facing": False,
        "human_audit_completed": False,
        "automatic_tracking_assumed_correct_by_experiment_plan": True,
    }
    write_json(destination / "cohort_summary.json", summary)
    write_json(destination / "cohort_manifest.json", {**provenance(sys.argv, {}, Path(__file__).resolve().parents[1]), "summary": summary})
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python mydata_bench/prepare_reward_cohort.py",
        description="Freeze a full source-reward cohort before grounding or inference.",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--reward", type=int)
    selector.add_argument("--audit-final")
    selector.add_argument("--grounding-run")
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args(argv)
    if args.grounding_run:
        result = freeze_auto_grounded_cohort(
            args.dataset_root,
            args.grounding_run,
            args.output_dir,
            split=args.split,
        )
    elif args.audit_final:
        result = freeze_audited_cohort(
            args.dataset_root,
            args.audit_final,
            args.output_dir,
            split=args.split,
        )
    else:
        result = freeze_reward_cohort(
            args.dataset_root,
            args.output_dir,
            reward=args.reward,
            split=args.split,
            expected_count=args.expected_count,
        )
    print(result["model_safe_episode_manifest"])


if __name__ == "__main__":
    main()
