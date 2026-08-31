#!/usr/bin/env python3
"""Freeze the cluster-disjoint confirmatory cohort for auto exploration."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import object_fingerprint, read_jsonl, stable_shard, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-inputs", required=True)
    parser.add_argument("--ranking-inputs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cohort = list(read_jsonl(Path(args.cohort_inputs).resolve()))
    ranking = list(read_jsonl(Path(args.ranking_inputs).resolve()))
    ranking_videos = {str(row["video_sha256"]) for row in ranking}
    screening_videos = {
        str(row["video_sha256"])
        for row in cohort
        if stable_shard(str(row["video_sha256"]), 16) == 0
    }
    if ranking_videos & screening_videos:
        raise RuntimeError("Ranking and screening video clusters overlap")
    excluded = ranking_videos | screening_videos
    heldout = [row for row in cohort if str(row["video_sha256"]) not in excluded]
    ids = [str(row["example_id"]) for row in heldout]
    videos = {str(row["video_sha256"]) for row in heldout}
    success_count = sum(example_id.startswith("suc/") for example_id in ids)
    fail_count = sum(example_id.startswith("fail/") for example_id in ids)
    expected = (697, 223, 474, 254, 34, 11)
    observed = (
        len(heldout),
        success_count,
        fail_count,
        len(videos),
        len(ranking_videos),
        len(screening_videos),
    )
    if observed != expected:
        raise RuntimeError(f"Frozen partition mismatch: expected {expected}, observed {observed}")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Held-out example IDs are not unique")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "heldout_example_ids.json", ids)
    write_jsonl(output / "heldout_cohort_inputs.jsonl", heldout)
    manifest = {
        "schema_version": "auto-explore-video-cluster-partition-v1",
        "selection_rule": (
            "exclude all 34 ranking/development video_sha256 clusters and all "
            "stable_shard(video_sha256,16)==0 screening clusters"
        ),
        "selection_uses_labels": False,
        "ranking_video_count": len(ranking_videos),
        "screening_video_count": len(screening_videos),
        "heldout_video_count": len(videos),
        "heldout_record_count": len(heldout),
        "heldout_success_count_post_selection": success_count,
        "heldout_fail_count_post_selection": fail_count,
        "ranking_video_sha256": sorted(ranking_videos),
        "screening_video_sha256": sorted(screening_videos),
        "heldout_video_sha256": sorted(videos),
        "heldout_example_ids_fingerprint": object_fingerprint(ids),
        "heldout_rows_fingerprint": object_fingerprint(heldout),
    }
    manifest["fingerprint"] = object_fingerprint(manifest)
    write_json(output / "partition_manifest.json", manifest)
    print(output / "partition_manifest.json")


if __name__ == "__main__":
    main()
