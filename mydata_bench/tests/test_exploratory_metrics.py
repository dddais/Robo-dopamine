from __future__ import annotations

import json
from pathlib import Path

import pytest

from mydata_bench.io import read_jsonl, write_json, write_jsonl
from mydata_bench.my_dataset.exploratory_metrics import (
    GRID_CONDITIONS,
    _prediction,
    score_exploratory_matrix,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    records_path = tmp_path / "matrix.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    selection_path = tmp_path / "selection_manifest.json"
    groups = {
        "group-a": ("task1_1", "object_identity", (5, 3)),
        "group-b": ("task1_1", "object_identity", (4, 1)),
        "group-c": ("task2_1", "attribute_color", (5, 2)),
        "group-d": ("task2_1", "attribute_color", (5, 2)),
    }
    labels = []
    records = [
        {
            "example_id": "group-a-suc",
            "group_id": "group-a",
            "task_id": "task1_1",
            "task_family": "object_identity",
            "condition": "baseline",
            "condition_kind": "baseline",
            "native_prediction": 1,
            "progress": 0.0,
            "run_fingerprint": "old-failed-attempt",
            "claim_status": "exploratory",
            "human_reviewed": False,
            "status": "invalid",
        }
    ]
    for group_id, (task_id, task_family, baseline_predictions) in groups.items():
        for suffix, reward, baseline_prediction in zip(
            ("suc", "fail"), (5, 1), baseline_predictions, strict=True
        ):
            example_id = f"{group_id}-{suffix}"
            is_proxy = example_id == "group-d-fail"
            labels.append(
                {
                    "example_id": example_id,
                    "group_id": group_id,
                    "task_id": task_id,
                    "task_family": task_family,
                    "instruction_video_match": reward == 5,
                    "protocol_reward": reward,
                }
            )
            base = {
                "example_id": example_id,
                "group_id": group_id,
                "task_id": task_id,
                "task_family": task_family,
                "model_family": "fixture",
                "variant_id": "fixture-unreviewed",
                "run_fingerprint": "fixture-run",
                "claim_status": "exploratory",
                "grounding_resolution": "proxy" if is_proxy else "strict",
                "grounding_status": (
                    "auto_proxy_unreviewed"
                    if is_proxy
                    else "auto_assumed_unreviewed"
                ),
                "grounding_selection": {
                    "fallback_used": is_proxy,
                    "selection_method": (
                        "fixture_proxy" if is_proxy else "fixture_strict"
                    ),
                },
                "human_reviewed": False,
                "status": "ok",
            }
            records.append(
                {
                    **base,
                    "condition": "baseline",
                    "condition_kind": "baseline",
                    "native_prediction": baseline_prediction,
                    "progress": (baseline_prediction - 1) / 4,
                }
            )
            for condition in GRID_CONDITIONS:
                ranking_n = int(condition.split("rank_n", 1)[1][:3])
                top_k = int(condition.split("top_k", 1)[1])
                candidate_prediction = baseline_prediction
                if ranking_n == 5 and top_k == 8:
                    candidate_prediction = {
                        "group-a": {"suc": 4, "fail": 1},
                        "group-b": {"suc": 5, "fail": 1},
                        "group-c": {"suc": 5, "fail": 1},
                        "group-d": {"suc": 5, "fail": 1},
                    }[group_id][suffix]
                row = {
                    **base,
                    "condition": condition,
                    "condition_kind": "candidate_target",
                    "ranking_n": ranking_n,
                    "top_k": top_k,
                    "heads": [
                        {"layer": 8 + index // 8, "head": index % 8}
                        for index in range(top_k)
                    ],
                    "bias": 6.0,
                    "scope": "all",
                    "hook_assertion": {"passed": True},
                    "ranking_fingerprint": f"ranking-{ranking_n}",
                }
                # Exercise both prediction paths.  The common-unseen failure is
                # deliberately a signed-score-only record.
                if (
                    ranking_n == 5
                    and top_k == 8
                    and group_id == "group-d"
                    and suffix == "fail"
                ):
                    row["signed_score"] = 0.0
                else:
                    row["native_prediction"] = candidate_prediction
                    row["progress"] = (candidate_prediction - 1) / 4
                records.append(row)
    write_jsonl(records_path, records)
    write_jsonl(labels_path, labels)
    write_json(
        selection_path,
        {
            "fingerprint": "selection-fixture",
            "cohorts": {
                "5": {"size": 5, "group_ids": ["group-a"]},
                "10": {"size": 10, "group_ids": ["group-a", "group-b"]},
                "20": {
                    "size": 20,
                    "group_ids": ["group-a", "group-b", "group-c"],
                },
            },
            "evaluation_cohorts": {
                "all": {"example_count": 8},
                "common_unseen_s20": {"example_count": 2},
                "by_ranking_size": {
                    "5": {
                        "n_specific_unseen": {"example_count": 6},
                        "ranking_source_only": {"example_count": 2},
                    },
                    "10": {
                        "n_specific_unseen": {"example_count": 4},
                        "ranking_source_only": {"example_count": 4},
                    },
                    "20": {
                        "n_specific_unseen": {"example_count": 2},
                        "ranking_source_only": {"example_count": 6},
                    },
                },
            },
        },
    )
    return records_path, labels_path, selection_path


def test_exploratory_matrix_scopes_and_paired_deltas(tmp_path: Path) -> None:
    records_path, labels_path, selection_path = _fixture(tmp_path)
    output_dir = tmp_path / "scoring"
    result = score_exploratory_matrix(
        records_path,
        labels_path,
        selection_path,
        output_dir,
        expected_count=8,
    )

    assert result["completion"]["complete"] is True
    assert result["completion"]["input_record_count"] == 81
    assert result["completion"]["latest_record_count"] == 80
    assert result["scope_expected_counts"] == {
        "all_including_rank_sources": 8,
        "common_unseen_s20": 2,
        "n_specific_unseen": {5: 6, 10: 4, 20: 2},
        "ranking_source_groups_only": {5: 2, 10: 4, 20: 6},
    }
    assert result["grounding_resolution"] == "mixed"
    assert result["grounding_status"] == "mixed"
    assert result["grounding_composition"] == {
        "strict_count": 7,
        "proxy_count": 1,
        "total": 8,
        "proxy_ratio": 0.125,
    }
    assert result["ranking_fingerprints"] == {
        5: "ranking-5",
        10: "ranking-10",
        20: "ranking-20",
    }
    baseline = result["shared_baseline"]
    assert baseline["shared_across_all_grid_conditions"] is True
    assert baseline["all_including_rank_sources"]["n"] == 8
    assert baseline["all_including_rank_sources"]["overall"][
        "exact_accuracy"
    ] == pytest.approx(0.5)
    assert baseline["all_including_rank_sources"]["suc_reward5"][
        "exact_accuracy"
    ] == pytest.approx(0.75)
    assert baseline["all_including_rank_sources"]["fail_reward1"][
        "exact_accuracy"
    ] == pytest.approx(0.25)

    condition = result["conditions"][
        "candidate_target__rank_n005__top_k008"
    ]
    assert condition["grounding_resolution"] == "mixed"
    assert condition["grounding_status"] == "mixed"
    strict = condition["grounding_strata"]["strict_grounding"]
    proxy = condition["grounding_strata"]["proxy_grounding"]
    assert strict["n"] == 7
    assert proxy["n"] == 1
    assert "group_ranking" not in strict
    assert "group_ranking" not in proxy
    all_rows = condition["scopes"]["all_including_rank_sources"]
    assert all_rows["overall"]["exact_accuracy"] == pytest.approx(0.875)
    assert all_rows["versus_baseline"]["exact_delta"] == pytest.approx(0.375)
    assert all_rows["versus_baseline"]["fail_correction_rate"] == pytest.approx(0.75)
    assert all_rows["versus_baseline"]["suc_harm_rate"] == pytest.approx(0.25)
    assert all_rows["versus_baseline"]["mean_progress_shift"] == pytest.approx(-0.125)

    common = condition["scopes"]["common_unseen_s20"]
    assert common["n"] == 2
    assert common["overall"]["exact_accuracy"] == 1.0
    assert common["versus_baseline"]["exact_delta"] == 0.5
    assert common["versus_baseline"]["fail_correction_rate"] == 1.0
    assert common["versus_baseline"]["suc_harm_rate"] == 0.0
    assert common["group_ranking"]["group_macro_pairwise_accuracy"] == 1.0
    assert common["group_ranking"]["strict_top1_accuracy"] == 1.0

    n_specific = condition["scopes"]["n_specific_unseen"]
    ranking_only = condition["scopes"]["ranking_source_groups_only"]
    assert n_specific["n"] == 6
    assert n_specific["versus_baseline"]["exact_delta"] == 0.5
    assert ranking_only["n"] == 2
    assert ranking_only["versus_baseline"]["exact_delta"] == 0.0
    assert ranking_only["versus_baseline"]["fail_correction_rate"] == 1.0
    assert ranking_only["versus_baseline"]["suc_harm_rate"] == 1.0

    assert len(list(read_jsonl(output_dir / "condition_metrics.jsonl"))) == 9
    joined = list(read_jsonl(output_dir / "joined_conditions.jsonl"))
    assert len(joined) == 72
    proxy_rows = [
        row for row in joined if row["grounding_resolution"] == "proxy"
    ]
    assert len(proxy_rows) == 9
    assert {row["example_id"] for row in proxy_rows} == {"group-d-fail"}
    assert {row["grounding_status"] for row in proxy_rows} == {
        "auto_proxy_unreviewed"
    }
    assert all(
        row["grounding_selection"]["fallback_used"] is True
        for row in proxy_rows
    )
    assert json.loads((output_dir / "metrics.json").read_text())["human_reviewed"] is False
    report = (output_dir / "exp_record.md").read_text(encoding="utf-8")
    for required in (
        "assumed_valid",
        "human_reviewed=false",
        "exploratory",
        "ranking/eval overlap",
        "in-sample contaminated",
        "cross-N main comparison",
        "wrong-target",
        "low-rank",
        "layer-matched-random",
        "共享 baseline 汇总",
        "Overall exact",
        "Suc reward5 exact",
        "Fail reward1 exact",
        "proxy grounding",
        "12.50%",
        "fallback grounding",
        "不作为 all-data primary result",
        "auto_proxy_unreviewed",
    ):
        assert required in report
    assert (tmp_path / "exp_record.md").read_text(encoding="utf-8") == report


def test_exploratory_matrix_rejects_incomplete_latest_matrix(tmp_path: Path) -> None:
    records_path, labels_path, selection_path = _fixture(tmp_path)
    records = list(read_jsonl(records_path))
    missing_condition = "candidate_target__rank_n020__top_k064"
    records = [
        row
        for row in records
        if not (
            row["example_id"] == "group-d-fail"
            and row["condition"] == missing_condition
        )
    ]
    write_jsonl(records_path, records)
    with pytest.raises(ValueError, match="Incomplete exploratory matrix"):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "incomplete-scoring",
            expected_count=8,
        )


def test_exploratory_matrix_rejects_grounding_metadata_that_varies_across_conditions(
    tmp_path: Path,
) -> None:
    records_path, labels_path, selection_path = _fixture(tmp_path)
    records = list(read_jsonl(records_path))
    for row in records:
        if (
            row["example_id"] == "group-d-fail"
            and row["condition"]
            == "candidate_target__rank_n020__top_k064"
        ):
            row["grounding_resolution"] = "strict"
            row["grounding_status"] = "auto_assumed_unreviewed"
            row["grounding_selection"] = {
                "fallback_used": False,
                "selection_method": "fixture_strict",
            }
    write_jsonl(records_path, records)
    with pytest.raises(
        ValueError, match="grounding metadata vary across conditions"
    ):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "inconsistent-grounding-scoring",
            expected_count=8,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ({"ranking_n": 10}, "disagree with condition"),
        ({"top_k": 32}, "disagree with condition"),
        ({"heads": [{"layer": 8, "head": 0}] * 7}, "exactly top_k=8"),
        (
            {"heads": [{"layer": 8, "head": 0}] * 8},
            "heads coordinates must be unique",
        ),
        (
            {
                "heads": [{"layer": 7, "head": 0}]
                + [{"layer": 8, "head": index} for index in range(1, 8)]
            },
            "layer must be >=8",
        ),
        (
            {
                "heads": [{"layer": 9, "head": 0}]
                + [{"layer": 8, "head": index} for index in range(1, 8)]
            },
            "heads list varies across examples",
        ),
        (
            {"variant_id": "fixture-drift"},
            "variant_id/reference_variant_id/model_family bindings",
        ),
        ({"bias": 5.5}, "bias must equal"),
        ({"scope": "terminal"}, "scope must equal"),
        ({"hook_assertion": {"passed": False}}, "hook_assertion.passed"),
        ({"ranking_fingerprint": "ranking-corrupt"}, "inconsistent ranking_fingerprint"),
    ),
)
def test_exploratory_matrix_rejects_corrupt_frozen_candidate_contract(
    tmp_path: Path, mutation: dict[str, object], error: str
) -> None:
    records_path, labels_path, selection_path = _fixture(tmp_path)
    records = list(read_jsonl(records_path))
    target_condition = "candidate_target__rank_n005__top_k008"
    for row in records:
        if row["example_id"] == "group-d-fail" and row["condition"] == target_condition:
            row.update(mutation)
            break
    else:  # pragma: no cover - fixture invariant
        raise AssertionError("candidate fixture row not found")
    write_jsonl(records_path, records)
    with pytest.raises(ValueError, match=error):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "corrupt-contract-scoring",
            expected_count=8,
        )


def test_exploratory_matrix_rejects_non_prefix_head_cells(
    tmp_path: Path,
) -> None:
    records_path, labels_path, selection_path = _fixture(tmp_path)
    records = list(read_jsonl(records_path))
    condition = "candidate_target__rank_n005__top_k008"
    replacement = [{"layer": 9, "head": 0}] + [
        {"layer": 8, "head": index} for index in range(1, 8)
    ]
    for row in records:
        if row["condition"] == condition:
            row["heads"] = replacement
    write_jsonl(records_path, records)
    with pytest.raises(ValueError, match="ordered K=64 prefix"):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "non-prefix-heads",
            expected_count=8,
        )


def test_grm_signed_score_uses_canonical_reward_thresholds() -> None:
    assert _prediction({"signed_score": 0.1249}, identity="below") == 1
    assert _prediction({"signed_score": 0.125}, identity="threshold-2") == 2
    assert _prediction({"signed_score": 0.625}, identity="threshold-4") == 4
    assert _prediction({"signed_score": 0.875}, identity="threshold-5") == 5


def test_reviewed_subset_joins_full_labels_and_checks_reference_baseline(
    tmp_path: Path,
) -> None:
    records_path, labels_path, selection_path = _fixture(tmp_path)
    reference_path = tmp_path / "unreviewed-reference.jsonl"
    original = list(read_jsonl(records_path))
    write_jsonl(reference_path, original)
    reviewed = []
    for row in original:
        value = dict(row)
        value.update(
            {
                "grounding_mode": "human_reviewed",
                "grounding_resolution": "human_audited",
                "grounding_status": "audited_eligible",
                "grounding_selection": None,
                "human_reviewed": True,
                "claim_status": "reviewed_exploratory",
                "variant_id": "fixture-reviewed",
                "reference_variant_id": "fixture-unreviewed",
            }
        )
        reviewed.append(value)
    write_jsonl(records_path, reviewed)

    evaluation_path = tmp_path / "complete_groups.jsonl"
    write_jsonl(
        evaluation_path,
        [
            {"example_id": f"{group_id}-{suffix}"}
            for group_id in ("group-c", "group-d")
            for suffix in ("suc", "fail")
        ],
    )
    with pytest.raises(ValueError, match="requires --evaluation-manifest"):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "missing-evaluation-gate",
            expected_count=None,
            reference_records_path=reference_path,
        )
    with pytest.raises(ValueError, match="requires --reference-records"):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "missing-reference-gate",
            expected_count=None,
            evaluation_manifest_path=evaluation_path,
        )
    output_dir = tmp_path / "reviewed-scoring"
    result = score_exploratory_matrix(
        records_path,
        labels_path,
        selection_path,
        output_dir,
        expected_count=None,
        evaluation_manifest_path=evaluation_path,
        reference_records_path=reference_path,
    )
    assert result["grounding_mode"] == "human_reviewed"
    assert result["human_reviewed"] is True
    assert result["claim_status"] == "reviewed_exploratory"
    assert result["completion"]["example_count"] == 4
    assert result["evaluation_filter"]["example_count"] == 4
    assert result["baseline_reference_equivalence"] == {
        "passed": True,
        "example_count": 4,
        "variant_binding": {
            "current_variant_id": "fixture-reviewed",
            "reference_variant_id": "fixture-unreviewed",
        },
        "reference_records_path": str(reference_path.resolve()),
        "reference_records_sha256": result[
            "baseline_reference_equivalence"
        ]["reference_records_sha256"],
        "reference_run_fingerprints": ["fixture-run"],
        "compared_fields": [
            "model_family",
            "group_id",
            "task_id",
            "task_family",
            "raw_output",
            "native_prediction",
            "signed_score",
            "progress",
            "derived_prediction",
            "derived_progress",
        ],
    }
    assert result["grounding_composition"] == {
        "human_audited_count": 4,
        "total": 4,
        "human_audited_ratio": 1.0,
    }
    assert result["scope_expected_counts"] == {
        "all_including_rank_sources": 4,
        "common_unseen_s20": 2,
        "n_specific_unseen": {5: 4, 10: 4, 20: 2},
        "ranking_source_groups_only": {5: 0, 10: 0, 20: 2},
    }
    report = (output_dir / "exp_record.md").read_text(encoding="utf-8")
    assert "human-reviewed exploratory robustness rerun" in report
    assert "不能标为 confirmatory/formal" in report


    wrong_variant_path = tmp_path / "wrong-variant-reference.jsonl"
    wrong_variant_rows = list(read_jsonl(reference_path))
    for row in wrong_variant_rows:
        row["variant_id"] = "fixture-wrong-unreviewed"
    write_jsonl(wrong_variant_path, wrong_variant_rows)
    with pytest.raises(ValueError, match="wrong_reference_variant"):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "wrong-reference-variant",
            expected_count=None,
            evaluation_manifest_path=evaluation_path,
            reference_records_path=wrong_variant_path,
        )

    def reference_with_fingerprint(
        fingerprint: str, filename: str
    ) -> Path:
        path = tmp_path / filename
        rows = list(read_jsonl(reference_path))
        for row in rows:
            if (
                row.get("example_id") == "group-c-suc"
                and row.get("condition") == "baseline"
                and row.get("status") == "ok"
            ):
                row["run_fingerprint"] = fingerprint
                break
        else:
            raise AssertionError("fixture baseline not found")
        write_jsonl(path, rows)
        return path

    missing_fingerprint_path = reference_with_fingerprint(
        "", "missing-fingerprint-reference.jsonl"
    )
    with pytest.raises(ValueError, match="missing_reference_run_fingerprint"):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "missing-reference-fingerprint",
            expected_count=None,
            evaluation_manifest_path=evaluation_path,
            reference_records_path=missing_fingerprint_path,
        )

    mixed_fingerprint_path = reference_with_fingerprint(
        "fixture-other-run", "mixed-fingerprint-reference.jsonl"
    )
    with pytest.raises(ValueError, match="non_unique_reference_run_fingerprint"):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "mixed-reference-fingerprint",
            expected_count=None,
            evaluation_manifest_path=evaluation_path,
            reference_records_path=mixed_fingerprint_path,
        )

    reference_rows = list(read_jsonl(reference_path))
    for row in reference_rows:
        if (
            row.get("example_id") == "group-c-suc"
            and row.get("condition") == "baseline"
            and row.get("status") == "ok"
        ):
            row["raw_output"] = "deliberate-reference-drift"
            break
    else:
        raise AssertionError("fixture baseline not found")
    write_jsonl(reference_path, reference_rows)
    with pytest.raises(ValueError, match="output_field_mismatches"):
        score_exploratory_matrix(
            records_path,
            labels_path,
            selection_path,
            tmp_path / "baseline-output-drift",
            expected_count=None,
            evaluation_manifest_path=evaluation_path,
            reference_records_path=reference_path,
        )
