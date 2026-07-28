"""Freeze the SAM3-audited exact reward=1/reward=5 pair candidate cohort.

The source pair manifest deliberately contains reward labels because it is a
data-construction artifact.  This module never forwards them to a model: the
attention loader converts the selected rows to instruction-only
``EpisodeRecord`` instances before parsing, grounding, or steering.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .io import object_fingerprint, provenance, read_jsonl, sha256_file, write_json, write_jsonl


def freeze_exact_pairs(
    pair_manifest: str | Path,
    eligible_manifest: str | Path,
    output_dir: str | Path,
    *,
    expected_pairs: int | None = 40,
) -> dict[str, Any]:
    """Intersect a frozen raw pair list with audited counterfactual samples.

    ``eligible_manifest`` must be the output of attention ``prepare`` from a
    grounding run with an ``audit_final.jsonl``.  Matching is by the original
    counterfactual ``example_id`` and is cross-checked against video SHA-256;
    this is stricter than matching by video hash alone.
    """
    pair_path = Path(pair_manifest).resolve()
    eligible_path = Path(eligible_manifest).resolve()
    destination = Path(output_dir).resolve()
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)
    if not eligible_path.is_file():
        raise FileNotFoundError(eligible_path)

    pairs = list(read_jsonl(pair_path))
    eligible = list(read_jsonl(eligible_path))
    eligible_by_id = {str(row["example_id"]): row for row in eligible}
    if len(eligible_by_id) != len(eligible):
        raise ValueError("Duplicate example_id in eligible manifest")

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    paired_counterfactual_ids: set[str] = set()
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        video_sha256 = str(pair["video_sha256"])
        if pair_id != video_sha256:
            raise ValueError(f"pair_id must equal video_sha256: {pair_id}")
        counterfactual_id = str(pair["counterfactual_example_id"])
        paired_counterfactual_ids.add(counterfactual_id)
        audited = eligible_by_id.get(counterfactual_id)
        if audited is None:
            excluded.append(
                {
                    "pair_id": pair_id,
                    "video_sha256": video_sha256,
                    "counterfactual_example_id": counterfactual_id,
                    "original_example_id": pair.get("original_example_id"),
                    "reason": "counterfactual_not_in_audited_formal_eligible_manifest",
                }
            )
            continue
        if str(audited.get("video_sha256")) != video_sha256:
            raise ValueError(
                "Counterfactual example ID matches eligible manifest but video hash differs: "
                f"{counterfactual_id}"
            )
        selected.append(pair)

    selected.sort(key=lambda row: str(row["pair_id"]))
    excluded.sort(key=lambda row: str(row["pair_id"]))
    if expected_pairs is not None and len(selected) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} exact-pair candidates, found {len(selected)}. "
            "Refuse to silently redefine the frozen cohort."
        )

    unpaired_eligible = [
        {
            "example_id": str(row["example_id"]),
            "video_sha256": str(row["video_sha256"]),
            "subset": row.get("subset"),
            "target_type": row.get("target_type"),
            "reason": "no_same_video_reward5_original_pair_in_raw_manifest",
        }
        for row in eligible
        if str(row["example_id"]) not in paired_counterfactual_ids
    ]
    unpaired_eligible.sort(key=lambda row: row["example_id"])

    destination.mkdir(parents=True, exist_ok=True)
    selected_path = destination / "paired_reward1_reward5_exact40.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(destination / "excluded_raw_pairs.jsonl", excluded)
    write_jsonl(destination / "audited_reward1_without_raw_pair.jsonl", unpaired_eligible)
    summary = {
        "cohort": "exact_same_video_reward1_reward5_candidates",
        "selection_rule": (
            "raw same-video reward=1/reward=5 pair whose counterfactual source "
            "example_id is in the independently SAM3-audited formal eligible manifest"
        ),
        "raw_pair_count": len(pairs),
        "audited_counterfactual_eligible_count": len(eligible),
        "selected_candidate_pair_count": len(selected),
        "excluded_raw_pair_count": len(excluded),
        "audited_counterfactual_without_raw_pair_count": len(unpaired_eligible),
        "selected_subsets": dict(sorted(Counter(str(row["subset"]) for row in selected).items())),
        "source_artifacts": {
            "raw_pair_manifest": str(pair_path),
            "raw_pair_manifest_sha256": sha256_file(pair_path),
            "audited_eligible_manifest": str(eligible_path),
            "audited_eligible_manifest_sha256": sha256_file(eligible_path),
        },
        "selected_pair_manifest": str(selected_path),
        "selected_pair_fingerprint": object_fingerprint(selected),
        "exclusion_reasons": dict(sorted(Counter(row["reason"] for row in excluded).items())),
        "note": (
            "This is a candidate cohort.  Each reward=5 instruction requires its own "
            "SAM3 grounding and dual endpoint audit; only pairs with both instruction "
            "sides formally eligible are analyzable."
        ),
    }
    write_json(destination / "pair_selection_summary.json", summary)
    write_json(
        destination / "pair_selection_manifest.json",
        {
            **provenance(sys.argv, {}, Path(__file__).resolve().parents[1]),
            "summary": summary,
        },
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python rewardbench/prepare_exact_pairs.py",
        description="Freeze exact same-video pairs from an audited reward=1 cohort.",
    )
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--eligible-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-pairs", type=int, default=40)
    args = parser.parse_args(argv)
    print(
        freeze_exact_pairs(
            args.pair_manifest,
            args.eligible_manifest,
            args.output_dir,
            expected_pairs=args.expected_pairs,
        )["selected_pair_manifest"]
    )


if __name__ == "__main__":
    main()
