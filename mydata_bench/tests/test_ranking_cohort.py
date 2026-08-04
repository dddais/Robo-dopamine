from __future__ import annotations

import json
from pathlib import Path

import pytest

from mydata_bench.io import (
    object_fingerprint,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from mydata_bench.my_dataset.data import _anonymous_id
from mydata_bench.my_dataset.ranking_cohort import (
    FROZEN_LJX_LFZ_S20,
    _label_free_candidates,
    _select_nested,
    _validate_group_closed_evaluation_population,
    freeze_ranking_cohort,
)


def _fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    dataset_name = "fixture"
    source_path = tmp_path / "ranking_data.jsonl"
    inputs_path = tmp_path / "inputs.jsonl"
    split_path = tmp_path / "split.json"
    attention_dir = tmp_path / "attention_inputs"
    output_dir = tmp_path / "ranking_cohort"

    source_rows = []
    prepared = []
    partition_examples = {"discovery": [], "validation": [], "test": []}
    targets = ("cup", "pen", "carrot", "blue block", "yellow cup")
    task_ids = ("task1_1", "task1_3", "task2_1", "task2_2", "task2_3")
    for index in range(20):
        task_id = task_ids[index % len(task_ids)]
        source_id = f"source-record-{index:02d}"
        target = targets[index % len(targets)]
        instruction = f"Pick up the {target} and place it in the plate."
        source_rows.append(
            {
                "id": source_id,
                "source_suc_id": source_id,
                "split": "suc",
                "instruction_video_match": True,
                "task_id": task_id,
                "instruction": instruction,
                "target_obj": target,
                "correct_target_obj": target,
                # This intentionally bogus path must never be opened.
                "video_paths": ["suc/does-not-exist.mp4"],
            }
        )
        example_id = _anonymous_id(f"{dataset_name}-e", source_id, instruction)
        group_id = _anonymous_id(f"{dataset_name}-g", source_id)
        videos = {}
        view_sha = {}
        for view in ("front", "left_wrist", "right_wrist"):
            path = tmp_path / "media" / f"{index}-{view}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{index}-{view}".encode())
            videos[view] = str(path)
            view_sha[view] = f"sha-{index}-{view}"
        row = {
            "schema_version": "fixture",
            "dataset_name": dataset_name,
            "example_id": example_id,
            "group_id": group_id,
            "group_media_sha256": f"media-{index}",
            "task_id": task_id,
            "task_family": "object_identity",
            "instruction": instruction,
            "video_paths": videos,
            "view_sha256": view_sha,
        }
        prepared.append(row)
        partition = ("discovery", "validation", "test")[index % 3]
        partition_examples[partition].append(example_id)

    write_jsonl(source_path, source_rows)
    write_jsonl(inputs_path, prepared)
    write_json(
        split_path,
        {
            "examples": partition_examples,
            "fingerprint": "fixture-split",
        },
    )
    for model in ("roboreward", "qwen", "grm"):
        write_jsonl(
            attention_dir / model / "all.jsonl",
            [
                {
                    "schema_version": "fixture-attention",
                    "example_id": row["example_id"],
                    "group_id": row["group_id"],
                    "group_media_sha256": row["group_media_sha256"],
                    "task_id": row["task_id"],
                    "task_family": row["task_family"],
                    "model_family": model,
                    "task": row["instruction"],
                    "grounding_status": "assumed_valid",
                }
                for row in reversed(prepared)
            ],
        )

    config = {
        "my_dataset_ranking_cohort": {
            "ranking_data_path": str(source_path),
            "inputs_path": str(inputs_path),
            "split_path": str(split_path),
            "attention_inputs_dir": str(attention_dir),
            "output_dir": str(output_dir),
            "expected_source_count": 20,
            "expected_input_count": 20,
            "expected_attention_count": 20,
        }
    }
    return config, source_path, attention_dir


def _bind_reviewed_attention(
    config: dict, attention_dir: Path
) -> dict[str, str]:
    provenance = {
        "schema_version": "my_dataset.review_provenance.v1",
        "requests_sha256": "1" * 64,
        "proposals_sha256": "2" * 64,
        "reviews_sha256": "3" * 64,
        "review_audit_sha256": "4" * 64,
        "review_audit_fingerprint": "5" * 64,
    }
    artifacts = {}
    for model in ("roboreward", "qwen", "grm"):
        path = attention_dir / model / "all.jsonl"
        rows = list(read_jsonl(path))
        for row in rows:
            row.update(
                {
                    "grounding_mode": "human_reviewed",
                    "grounding_resolution": "human_audited",
                    "grounding_status": "audited_eligible",
                    "human_reviewed": True,
                    "claim_status": "reviewed_exploratory",
                    "review_provenance": provenance,
                }
            )
        write_jsonl(path, rows)
        artifacts[f"{model}/all"] = {
            "path": str(path.resolve()),
            "count": len(rows),
            "sha256": sha256_file(path),
            "fingerprint": object_fingerprint(rows),
        }
    manifest = {
        "schema_version": "my_dataset.attention_input.v1",
        "require_review_audit": True,
        "accepted_review_statuses": ["eligible"],
        "grounding_requests_sha256": provenance["requests_sha256"],
        "expected_requests_sha256": provenance["requests_sha256"],
        "grounding_proposals_sha256": provenance["proposals_sha256"],
        "expected_proposals_sha256": provenance["proposals_sha256"],
        "grounding_reviews_sha256": provenance["reviews_sha256"],
        "review_audit_sha256": provenance["review_audit_sha256"],
        "review_audit_fingerprint": provenance["review_audit_fingerprint"],
        "review_audit_schema_version": "my_dataset.grounding_review_audit.v2",
        "review_provenance": provenance,
        "artifacts": artifacts,
    }
    manifest["fingerprint"] = object_fingerprint(manifest)
    write_json(attention_dir / "manifest.json", manifest)
    config["my_dataset_ranking_cohort"]["required_grounding_mode"] = (
        "human_reviewed"
    )
    return provenance


def _bind_tracking_reviewed_attention(
    config: dict, attention_dir: Path
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "schema_version": "my_dataset.tracking_review_provenance.v2",
        "requests_sha256": "1" * 64,
        "tracking_artifact_sha256": "2" * 64,
        "tracking_manifest_sha256": "3" * 64,
        "manual_tracking_artifact_sha256": None,
        "reviews_sha256": "4" * 64,
        "review_audit_sha256": "5" * 64,
        "review_audit_fingerprint": "6" * 64,
    }
    artifacts = {}
    for model in ("roboreward", "qwen", "grm"):
        path = attention_dir / model / "all.jsonl"
        rows = list(read_jsonl(path))
        for row in rows:
            row.update(
                {
                    "grounding_mode": "human_reviewed",
                    "grounding_resolution": "human_audited",
                    "grounding_status": "audited_eligible",
                    "human_reviewed": True,
                    "claim_status": "reviewed_exploratory",
                    "target_grounding_scope": "terminal_only",
                    "control_region_policy": "none",
                    "tracking_review_provenance": provenance,
                }
            )
        write_jsonl(path, rows)
        artifacts[f"{model}/all"] = {
            "path": str(path.resolve()),
            "count": len(rows),
            "sha256": sha256_file(path),
            "fingerprint": object_fingerprint(rows),
        }
    manifest = {
        "schema_version": "my_dataset.attention_input.v1",
        "review_source_kind": "tracked_grounding_v2",
        "require_review_audit": True,
        "accepted_review_statuses": ["eligible"],
        "grounding_requests_sha256": provenance["requests_sha256"],
        "expected_requests_sha256": provenance["requests_sha256"],
        "tracking_artifact_sha256": provenance["tracking_artifact_sha256"],
        "expected_tracking_artifact_sha256": provenance[
            "tracking_artifact_sha256"
        ],
        "tracking_manifest_sha256": provenance["tracking_manifest_sha256"],
        "expected_tracking_manifest_sha256": provenance[
            "tracking_manifest_sha256"
        ],
        "manual_tracking_artifact_sha256": None,
        "expected_manual_tracking_artifact_sha256": None,
        "grounding_reviews_sha256": provenance["reviews_sha256"],
        "review_audit_sha256": provenance["review_audit_sha256"],
        "review_audit_fingerprint": provenance["review_audit_fingerprint"],
        "review_audit_schema_version": (
            "my_dataset.tracked_grounding_review_audit.v2"
        ),
        "tracking_review_provenance": provenance,
        "target_grounding_scope": "terminal_only",
        "control_region_policy": "none",
        "artifacts": artifacts,
    }
    manifest["fingerprint"] = object_fingerprint(manifest)
    write_json(attention_dir / "manifest.json", manifest)
    config["my_dataset_ranking_cohort"]["required_grounding_mode"] = (
        "human_reviewed"
    )
    return provenance


def test_freeze_ranking_cohort_is_nested_identical_and_label_free(tmp_path: Path) -> None:
    config, _source_path, _attention_dir = _fixture(tmp_path)
    manifest_path = freeze_ranking_cohort(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    s5 = manifest["cohorts"]["5"]["source_record_ids"]
    s10 = manifest["cohorts"]["10"]["source_record_ids"]
    s20 = manifest["cohorts"]["20"]["source_record_ids"]
    assert s5 == s10[:5]
    assert s10 == s20[:10]
    assert manifest["selection"]["labels_used_for_selection"] is False
    assert manifest["strict_audit"]["source_suc_media_paths_opened"] is False
    assert manifest["evaluation_cohorts"]["common_unseen_s20"]["example_count"] == 0

    ordered_ids = []
    for model in ("roboreward", "qwen", "grm"):
        rows = list(read_jsonl(manifest_path.parent / model / "ordered_max20.jsonl"))
        assert len(rows) == 20
        assert [row["ranking_order"] for row in rows] == list(range(1, 21))
        assert [row["cohort_role"] for row in rows] == ["external_ranking"] * 20
        serialized = json.dumps(rows)
        for forbidden in (
            "instruction_video_match",
            "target_obj",
            "correct_target_obj",
            "protocol_reward",
            "reward",
        ):
            assert f'"{forbidden}"' not in serialized
        assert "suc/does-not-exist.mp4" not in serialized
        ordered_ids.append([row["example_id"] for row in rows])
    assert ordered_ids[0] == ordered_ids[1] == ordered_ids[2]


def test_freeze_ranking_cohort_rejects_source_integrity_failure(tmp_path: Path) -> None:
    config, source_path, _attention_dir = _fixture(tmp_path)
    rows = list(read_jsonl(source_path))
    rows[0]["correct_target_obj"] = "not the target"
    write_jsonl(source_path, rows)
    with pytest.raises(ValueError, match="target_obj and correct_target_obj differ"):
        freeze_ranking_cohort(config)

def test_ljx_lfz_s20_order_is_frozen() -> None:
    source_path = Path(__file__).resolve().parents[3] / "data/ljx_lfz_task/new/ranking_data.jsonl"
    if not source_path.is_file():
        pytest.skip("LJX/LFZ source data are not available in this checkout")
    rows = list(read_jsonl(source_path))
    candidates = _label_free_candidates(rows, dataset_name="ljx_lfz_cf_v1")
    actual = tuple(row["source_record_id"] for row in _select_nested(candidates))
    assert actual == FROZEN_LJX_LFZ_S20


def test_freeze_ranking_cohort_rejects_cross_model_id_mismatch(tmp_path: Path) -> None:
    config, _source_path, attention_dir = _fixture(tmp_path)
    rows = list(read_jsonl(attention_dir / "qwen" / "all.jsonl"))
    rows[-1]["example_id"] = "foreign-example"
    write_jsonl(attention_dir / "qwen" / "all.jsonl", rows)
    with pytest.raises(ValueError, match="attention/prepared ID mismatch"):
        freeze_ranking_cohort(config)


def test_reviewed_ranking_binds_attention_manifest_and_jsonl(
    tmp_path: Path,
) -> None:
    config, _source_path, attention_dir = _fixture(tmp_path)
    provenance = _bind_reviewed_attention(config, attention_dir)

    manifest_path = freeze_ranking_cohort(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attention_manifest_path = attention_dir / "manifest.json"
    assert manifest["review_provenance"] == provenance
    assert manifest["attention_manifest"] == {
        "path": str(attention_manifest_path.resolve()),
        "sha256": sha256_file(attention_manifest_path),
        "fingerprint": json.loads(
            attention_manifest_path.read_text(encoding="utf-8")
        )["fingerprint"],
    }
    for model in ("roboreward", "qwen", "grm"):
        rows = list(read_jsonl(manifest_path.parent / model / "ordered_max20.jsonl"))
        assert all(row["review_provenance"] == provenance for row in rows)


def test_tracking_reviewed_ranking_propagates_terminal_contract(
    tmp_path: Path,
) -> None:
    config, _source_path, attention_dir = _fixture(tmp_path)
    provenance = _bind_tracking_reviewed_attention(config, attention_dir)

    manifest_path = freeze_ranking_cohort(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["review_source_kind"] == "tracked_grounding_v2"
    assert manifest["tracking_review_provenance"] == provenance
    assert manifest["target_grounding_scope"] == "terminal_only"
    assert manifest["control_region_policy"] == "none"
    assert "review_provenance" not in manifest
    for model in ("roboreward", "qwen", "grm"):
        rows = list(
            read_jsonl(manifest_path.parent / model / "ordered_max20.jsonl")
        )
        assert all(
            row["tracking_review_provenance"] == provenance
            and row["target_grounding_scope"] == "terminal_only"
            and row["control_region_policy"] == "none"
            and "wrong_region_bbox" not in row
            for row in rows
        )


def test_reviewed_ranking_rejects_tampered_same_id_attention_jsonl(
    tmp_path: Path,
) -> None:
    config, _source_path, attention_dir = _fixture(tmp_path)
    _bind_reviewed_attention(config, attention_dir)
    path = attention_dir / "qwen" / "all.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["stale_same_id_payload"] = True
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="SHA-256 differs from parent manifest"):
        freeze_ranking_cohort(config)


def test_reviewed_ranking_uses_all_but_evaluation_uses_complete_groups(
    tmp_path: Path,
) -> None:
    config, _source_path, attention_dir = _fixture(tmp_path)
    cfg = config["my_dataset_ranking_cohort"]
    cfg.update(
        {
            "expected_attention_count": "auto",
            "attention_coverage": "subset",
            "ranking_attention_filename": "all.jsonl",
            "evaluation_attention_filename": "complete_groups.jsonl",
        }
    )
    removed_id = None
    for model in ("roboreward", "qwen", "grm"):
        rows = list(read_jsonl(attention_dir / model / "all.jsonl"))
        removed_id = removed_id or rows[-1]["example_id"]
        complete = [
            row for row in rows if row["example_id"] != removed_id
        ]
        write_jsonl(attention_dir / model / "complete_groups.jsonl", complete)

    manifest_path = freeze_ranking_cohort(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    population = manifest["evaluation_population"]
    assert population["example_count"] == 19
    assert removed_id not in population["example_ids"]
    assert population["ranking_attention_filename"] == "all.jsonl"
    assert population["evaluation_attention_filename"] == "complete_groups.jsonl"
    assert all(
        value["count"] == 20
        for value in manifest["model_outputs"].values()
    )

    for model in ("roboreward", "qwen", "grm"):
        rows = [
            row
            for row in read_jsonl(attention_dir / model / "all.jsonl")
            if row["example_id"] != removed_id
        ]
        write_jsonl(attention_dir / model / "all.jsonl", rows)
    with pytest.raises(ValueError, match="excludes frozen S20 ranking sources"):
        freeze_ranking_cohort(config)


def test_evaluation_population_must_be_nonempty_and_group_closed() -> None:
    prepared = [
        {"example_id": "g1-suc", "group_id": "g1"},
        {"example_id": "g1-fail", "group_id": "g1"},
        {"example_id": "g2-suc", "group_id": "g2"},
    ]
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_group_closed_evaluation_population(prepared, set())
    with pytest.raises(ValueError, match="not group-closed"):
        _validate_group_closed_evaluation_population(
            prepared,
            {"g1-suc", "g2-suc"},
        )
    _validate_group_closed_evaluation_population(
        prepared,
        {"g1-suc", "g1-fail"},
    )
    _validate_group_closed_evaluation_population(
        prepared,
        {"g2-suc"},
    )
