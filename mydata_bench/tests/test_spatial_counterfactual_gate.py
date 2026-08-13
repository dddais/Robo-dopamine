from __future__ import annotations

import json

import pytest

from mydata_bench.analyze_spatial_counterfactual_gate import (
    load_predictions,
    spatial_counterfactual_gate,
)


@pytest.mark.parametrize(
    ("baseline", "target", "wrong", "expected", "reason"),
    [
        (3, 1, 3, 1, "accept_spatially_specific_endpoint"),
        (3, 5, 4, 5, "accept_spatially_specific_endpoint"),
        (3, 1, 1, 3, "reject_wrong_region_agreement"),
        (4, 5, 5, 4, "reject_wrong_region_agreement"),
        (5, 4, 2, 5, "reject_non_endpoint"),
    ],
)
def test_spatial_counterfactual_gate(
    baseline: int, target: int, wrong: int, expected: int, reason: str
) -> None:
    assert spatial_counterfactual_gate(baseline, target, wrong) == (expected, reason)


@pytest.mark.parametrize("value", [0, 6, 1.5, "1"])
def test_spatial_counterfactual_gate_rejects_invalid_predictions(value) -> None:
    with pytest.raises(ValueError):
        spatial_counterfactual_gate(value, 1, 2)


def test_load_predictions_uses_frozen_baseline_cohort(tmp_path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    conditions = {
        "baseline": 3,
        "candidate_target_k8": 1,
        "candidate_wrong_k8": 3,
        "low_rank_target_k8": 3,
    }
    with (experiment / "steering.jsonl").open("w", encoding="utf-8") as handle:
        for condition, prediction in conditions.items():
            handle.write(
                json.dumps(
                    {
                        "example_id": "fail/task/1",
                        "condition": condition,
                        "status": "ok",
                        "native_prediction": prediction,
                        "video_sha256": "video-a",
                    }
                )
                + "\n"
            )
    metadata = {
        "fail/task/1": {"task_id": "task", "source_suc_id": "suc/task/1"},
        # Dataset rows outside the immutable baseline cohort must be ignored.
        "suc/task/1": {"task_id": "task", "source_suc_id": "suc/task/1"},
    }
    rows = load_predictions(experiment, metadata, top_k=8)
    assert len(rows) == 1
    assert rows[0]["predictions"]["sc_gate"] == 1


def test_load_predictions_rejects_incomplete_frozen_cohort(tmp_path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    with (experiment / "steering.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "example_id": "fail/task/1",
                    "condition": "baseline",
                    "status": "ok",
                    "native_prediction": 3,
                    "video_sha256": "video-a",
                }
            )
            + "\n"
        )
    metadata = {
        "fail/task/1": {"task_id": "task", "source_suc_id": "suc/task/1"}
    }
    with pytest.raises(ValueError, match="complete strict"):
        load_predictions(experiment, metadata, top_k=8)
