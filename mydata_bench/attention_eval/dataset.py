from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..data import load_configured_episodes
from ..io import object_fingerprint, provenance, read_jsonl, write_json, write_jsonl
from ..video import extract_uniform_image_sequence


IMAGE_SEQUENCE_PROTOCOL = "roborewardbench_image_sequence"


def _latest_ok_endpoints(rows: Any) -> dict[str, dict[str, dict]]:
    """Return only endpoints whose latest append-only record is successful."""
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        example_id, frame = row.get("example_id"), row.get("frame")
        if isinstance(example_id, str) and frame in {"first", "last"}:
            latest[(example_id, frame)] = row
    endpoints: dict[str, dict[str, dict]] = defaultdict(dict)
    for (example_id, frame), row in latest.items():
        if row.get("status") == "ok":
            endpoints[example_id][frame] = row
    return endpoints


def prepare_grounded_ranking_samples(
    attention: dict[str, Any], output_path: str | Path
) -> Path:
    """Materialize model-safe inputs from the independent ranking_data split."""
    grounding_run = Path(attention["ranking_grounding_run"]).resolve()
    episodes = list(
        load_configured_episodes(
            {
                "dataset_root": attention["dataset_root"],
                "split": attention.get("ranking_split", "suc"),
                "metadata_file": attention["ranking_metadata_file"],
            }
        )[0]
    )
    expected_source_count = attention.get("ranking_expected_source_count")
    if expected_source_count is not None and len(episodes) != int(expected_source_count):
        raise ValueError(
            "Independent ranking metadata count mismatch: "
            f"expected {int(expected_source_count)}, found {len(episodes)}"
        )
    targets = {
        row["example_id"]: row
        for row in read_jsonl(grounding_run.parent / "targets.jsonl")
    }
    grounding_rows = list(read_jsonl(grounding_run / "grounding.jsonl"))
    if bool(attention.get("ranking_require_complete_grounding", False)):
        latest_seen = {
            (row.get("example_id"), row.get("frame"))
            for row in grounding_rows
            if isinstance(row.get("example_id"), str)
            and row.get("frame") in {"first", "last"}
        }
        missing = [
            (episode.example_id, frame)
            for episode in episodes
            for frame in ("first", "last")
            if (episode.example_id, frame) not in latest_seen
        ]
        if missing:
            missing_ids = sorted({example_id for example_id, _ in missing})
            raise ValueError(
                "Independent ranking grounding is incomplete: "
                f"{len(missing_ids)}/{len(episodes)} source examples are missing "
                f"one or more endpoint records; first missing IDs: {missing_ids[:5]}"
            )
    endpoints = _latest_ok_endpoints(grounding_rows)
    rows = []
    for episode in episodes:
        endpoint = endpoints.get(episode.example_id, {})
        target = targets.get(episode.example_id)
        if {"first", "last"} - set(endpoint) or target is None:
            continue
        first, last = endpoint["first"], endpoint["last"]
        first_path = first.get("provenance", {}).get("image_path")
        last_path = last.get("provenance", {}).get("image_path")
        first_bbox = first.get("bbox")
        bbox = last.get("bbox")
        tracking_path = last.get("provenance", {}).get("tracking_path")
        if not isinstance(first_path, str) or not isinstance(last_path, str):
            continue
        if not isinstance(first_bbox, list) or len(first_bbox) != 4:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not isinstance(tracking_path, str) or not Path(tracking_path).is_file():
            continue
        row = {
            "example_id": episode.example_id,
            "video_sha256": episode.video_sha256,
            "subset": episode.subset,
            "task": episode.task,
            "video_path": episode.video_path,
            "target_type": target.get("entity_type", "object"),
            "target_phrase": target.get("target_phrase"),
            "first": first,
            "last": last,
            "first_image_path": first_path,
            "last_image_path": last_path,
            "first_bbox": [float(value) for value in first_bbox],
            "last_bbox": [float(value) for value in bbox],
            "tracking_path": str(Path(tracking_path).resolve()),
            "ranking_source": "mydata_ranking_data",
            "grounding_fingerprint": last.get("grounding_fingerprint"),
        }
        views = last.get("provenance", {}).get("view_endpoint_paths", {})
        if (
            str(attention.get("protocol")) == "robo_dopamine_forward"
            and {"front", "left_wrist", "right_wrist"} <= set(views)
        ):
            blank = str(Path(attention["blank_goal"]).resolve())
            row["image_paths"] = [
                views["front"]["first"],
                blank,
                views["front"]["first"],
                views["left_wrist"]["first"],
                views["right_wrist"]["first"],
                views["front"]["last"],
                views["left_wrist"]["last"],
                views["right_wrist"]["last"],
            ]
        elif str(attention.get("protocol")) == IMAGE_SEQUENCE_PROTOCOL:
            image_paths, sampling_record = extract_uniform_image_sequence(
                episode.video_path,
                Path(output_path).resolve().parent
                / "image_sequences"
                / "ranking"
                / episode.video_sha256,
                count=int(attention.get("num_images", 8)),
            )
            row["image_paths"] = image_paths
            row["image_source_indices"] = sampling_record[
                "selected_source_indices"
            ]
            row["image_sampling_record"] = sampling_record
            row["sample_fingerprint"] = object_fingerprint(row)
        rows.append(row)
    expected_usable_count = attention.get("ranking_expected_usable_count")
    if expected_usable_count is not None and len(rows) != int(expected_usable_count):
        raise ValueError(
            "Independent ranking usable sample count mismatch after grounding: "
            f"expected {int(expected_usable_count)}, found {len(rows)}"
        )
    if not rows:
        raise ValueError("Independent ranking grounding produced no usable samples")
    path = Path(output_path).resolve()
    write_jsonl(path, rows)
    write_json(
        path.with_suffix(".manifest.json"),
        {
            "source": "ranking_data.jsonl_with_automatic_sam3_tracking",
            "ranking_metadata_file": str(
                Path(attention["ranking_metadata_file"]).resolve()
            ),
            "ranking_grounding_run": str(grounding_run),
            "source_sample_count": len(episodes),
            "sample_count": len(rows),
            "complete_grounding_required": bool(
                attention.get("ranking_require_complete_grounding", False)
            ),
            "labels_model_facing": False,
            "input_representation": (
                "uniform_independent_images_v1"
                if str(attention.get("protocol")) == IMAGE_SEQUENCE_PROTOCOL
                else None
            ),
            "fingerprint": object_fingerprint(rows),
        },
    )
    return path


def _formal_ids(
    grounding_run: Path, eligibility_mode: str = "audited"
) -> set[str]:
    """Return endpoint-eligible IDs under an explicitly declared protocol.

    ``audited`` is the default and is the only mode that can support formal
    causal claims. ``auto_valid_grounding`` requires an automatic ``ok``
    detection at both endpoints, but never represents those boxes as reviewed.
    """
    if eligibility_mode == "audited":
        audit = grounding_run / "audit_final.jsonl"
        if not audit.exists():
            raise FileNotFoundError(
                "Audited attention preparation requires audit_final.jsonl; "
                "use eligibility_mode=auto_valid_grounding only for an "
                "explicitly unaudited exploratory screen."
            )
        return {
            row["example_id"]
            for row in read_jsonl(audit)
            if bool(row.get("formal_eligible"))
        }
    if eligibility_mode == "auto_valid_grounding":
        records = grounding_run / "grounding.jsonl"
        if not records.exists():
            raise FileNotFoundError(records)
        latest: dict[tuple[str, str], dict] = {}
        for row in read_jsonl(records):
            example_id, frame = row.get("example_id"), row.get("frame")
            if isinstance(example_id, str) and frame in {"first", "last"}:
                latest[(example_id, frame)] = row
        endpoints: dict[str, set[str]] = defaultdict(set)
        for (example_id, frame), row in latest.items():
            if row.get("status") == "ok":
                endpoints[example_id].add(frame)
        return {
            example_id
            for example_id, frames in endpoints.items()
            if frames == {"first", "last"}
        }
    raise ValueError(
        "eligibility_mode must be 'audited' or 'auto_valid_grounding', "
        f"got {eligibility_mode!r}"
    )


def grouped_stratified_split(
    rows: list[dict], *, discovery_fraction: float = 1 / 3, seed: int = 20260724
) -> dict:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hash[str(row["video_sha256"])].append(row)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for digest, group in by_hash.items():
        representative = sorted(group, key=lambda row: row["example_id"])[0]
        strata[(representative["subset"], representative["target_type"])].append(digest)
    rng = random.Random(seed)
    discovery_hashes: set[str] = set()
    for hashes in strata.values():
        hashes = sorted(hashes)
        rng.shuffle(hashes)
        count = max(1, round(len(hashes) * discovery_fraction)) if len(hashes) > 1 else 0
        discovery_hashes.update(hashes[:count])
    discovery = sorted(
        row["example_id"] for row in rows if row["video_sha256"] in discovery_hashes
    )
    evaluation = sorted(
        row["example_id"] for row in rows if row["video_sha256"] not in discovery_hashes
    )
    if set(discovery) & set(evaluation):
        raise AssertionError("Split ID leakage")
    discovery_videos = {
        row["video_sha256"] for row in rows if row["example_id"] in set(discovery)
    }
    evaluation_videos = {
        row["video_sha256"] for row in rows if row["example_id"] in set(evaluation)
    }
    if discovery_videos & evaluation_videos:
        raise AssertionError("Video-content leakage across split")
    result = {
        "seed": seed,
        "method": "video_sha256_grouped_subset_target_type_stratified",
        "discovery_fraction": discovery_fraction,
        "discovery": discovery,
        "evaluation": evaluation,
        "discovery_video_sha256": sorted(discovery_videos),
        "evaluation_video_sha256": sorted(evaluation_videos),
    }
    result["fingerprint"] = object_fingerprint(result)
    return result


def prepare(config: dict[str, Any]) -> Path:
    import sys

    attention = config["attention_eval"]
    output_dir = Path(attention["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grounding_run = Path(attention["grounding_run"]).resolve()
    eligibility_mode = str(attention.get("eligibility_mode", "audited"))
    formal = _formal_ids(grounding_run, eligibility_mode)
    requested_ids: set[str] | None = None
    ids_path = attention.get("example_ids_file")
    if ids_path:
        payload = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
            raise ValueError("attention_eval.example_ids_file must be a JSON string list")
        if len(set(payload)) != len(payload):
            raise ValueError("attention_eval.example_ids_file contains duplicate IDs")
        requested_ids = set(payload)
        unavailable = requested_ids - formal
        if unavailable:
            raise ValueError(
                "Frozen cohort contains IDs without eligible endpoint grounding: "
                f"{sorted(unavailable)[:5]}"
            )
        formal = requested_ids
    targets = {
        row["example_id"]: row
        for row in read_jsonl(grounding_run.parent / "targets.jsonl")
    }
    configured_episodes, configured_pairs = load_configured_episodes(attention)
    episodes = {row.example_id: row for row in configured_episodes}
    grounding_rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in read_jsonl(grounding_run / "grounding.jsonl"):
        if row.get("status") == "ok":
            grounding_rows[row["example_id"]][row["frame"]] = row
    eligible = []
    for example_id in sorted(formal):
        endpoint = grounding_rows.get(example_id, {})
        target = targets.get(example_id)
        episode = episodes.get(example_id)
        if (
            target is None
            or episode is None
            or {"first", "last"} - endpoint.keys()
            or target.get("entity_type") not in {"object", "object_part"}
            or target.get("multi_target")
            or target.get("ambiguous")
        ):
            continue
        eligible.append(
            {
                "example_id": example_id,
                "video_sha256": episode.video_sha256,
                "subset": episode.subset,
                "task": episode.task,
                "video_path": episode.video_path,
                "target_type": target["entity_type"],
                "target_phrase": target["target_phrase"],
                "first": endpoint["first"],
                "last": endpoint["last"],
            }
        )
    if requested_ids is not None:
        eligible_ids = {row["example_id"] for row in eligible}
        missing_requested = requested_ids - eligible_ids
        if missing_requested:
            raise ValueError(
                "GRM preparation filtered IDs from the frozen cohort: "
                f"{sorted(missing_requested)[:5]}"
            )
    split = grouped_stratified_split(
        eligible,
        discovery_fraction=float(attention.get("discovery_fraction", 1 / 3)),
        seed=int(attention.get("seed", 20260724)),
    )
    eligible_by_id = {row["example_id"]: row for row in eligible}
    evaluation_subsets = {
        eligible_by_id[example_id]["subset"] for example_id in split["evaluation"]
    }
    formal_gate = {
        "evaluation_count": len(split["evaluation"]),
        "evaluation_subset_count": len(evaluation_subsets),
        "minimum_count": 90,
        "minimum_subsets": 15,
        "eligibility_mode": eligibility_mode,
    }
    if eligibility_mode == "auto_valid_grounding":
        formal_gate.update(
            {
                "status": "exploratory_unaudited_auto_grounding",
                "reason": (
                    "Uses automatic grounding boxes without human endpoint audit; "
                    "not eligible for formal causal claims."
                ),
            }
        )
    else:
        formal_gate["status"] = (
            "confirmatory"
            if len(split["evaluation"]) >= 90 and len(evaluation_subsets) >= 15
            else "exploratory"
        )
    split["formal_gate"] = formal_gate
    split["fingerprint"] = object_fingerprint(
        {key: value for key, value in split.items() if key != "fingerprint"}
    )
    write_json(output_dir / "split.json", split)
    # Model-facing manifest intentionally omits rewards and automated label checks.
    write_jsonl(output_dir / "eligible.jsonl", eligible)
    _paired_manifest(config, output_dir, configured_pairs)
    write_json(
        output_dir / "prepare_manifest.json",
        {
            **provenance(sys.argv, config, Path(__file__).resolve().parents[2]),
            "split_fingerprint": split["fingerprint"],
            "eligible_count": len(eligible),
            "eligibility_mode": eligibility_mode,
            "human_audit_completed": eligibility_mode == "audited",
            "human_audit_required_for_formal_causal_claim": True,
        },
    )
    return output_dir / "split.json"


def _paired_manifest(
    config: dict[str, Any], output_dir: Path, configured_pairs: list[dict] | None = None
) -> None:
    attention = config["attention_eval"]
    if configured_pairs is not None:
        # Pair IDs and tasks are frozen upstream.  Rewards are intentionally
        # absent: this manifest is consumed by model-facing attention code.
        write_jsonl(output_dir / "paired_reward1_reward5.jsonl", configured_pairs)
        write_json(
            output_dir / "paired_manifest_summary.json",
            {
                "num_pairs": len(configured_pairs),
                "num_unique_video_sha256": len(
                    {row["video_sha256"] for row in configured_pairs}
                ),
                "source": "frozen_pair_manifest",
            },
        )
        return
    counterfactual_root = attention.get("counterfactual_dataset_root")
    if not counterfactual_root:
        return
    from ..data import load_episodes

    full = list(load_episodes(attention["dataset_root"], "test"))
    counter = list(load_episodes(counterfactual_root, "test"))
    reward5: dict[str, list] = defaultdict(list)
    counter_by_hash: dict[str, list] = defaultdict(list)
    for row in full:
        if row.reward == 5:
            reward5[row.video_sha256].append(row)
    for row in counter:
        counter_by_hash[row.video_sha256].append(row)
    pairs = []
    for digest in sorted(set(counter_by_hash) & set(reward5)):
        row = sorted(counter_by_hash[digest], key=lambda item: item.example_id)[0]
        original = sorted(reward5[digest], key=lambda item: item.example_id)[0]
        pairs.append(
            {
                "pair_id": digest,
                "video_sha256": digest,
                "video_path": row.video_path,
                "counterfactual_example_id": row.example_id,
                "counterfactual_task": row.task,
                "counterfactual_reward": row.reward,
                "original_example_id": original.example_id,
                "original_task": original.task,
                "original_reward": original.reward,
            }
        )
    write_jsonl(output_dir / "paired_reward1_reward5.jsonl", pairs)
    write_json(
        output_dir / "paired_manifest_summary.json",
        {"num_pairs": len(pairs), "num_unique_video_sha256": len({row["video_sha256"] for row in pairs})},
    )


def load_partition(output_dir: str | Path, partition: str) -> tuple[list[dict], dict]:
    output_dir = Path(output_dir)
    split = json.loads((output_dir / "split.json").read_text(encoding="utf-8"))
    if partition == "all_eligible":
        # Deliberately explicit: this is an exploratory descriptive follow-up
        # over every audited object/object-part record, not the held-out
        # evaluation partition used for confirmatory inference.
        rows = list(read_jsonl(output_dir / "eligible.jsonl"))
        return rows, split
    if partition not in {"discovery", "evaluation"}:
        raise ValueError("partition must be discovery, evaluation, or all_eligible")
    allowed = set(split[partition])
    rows = [row for row in read_jsonl(output_dir / "eligible.jsonl") if row["example_id"] in allowed]
    return rows, split
