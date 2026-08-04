"""RoboRewardBench paper-metric protocol.

The RoboReward paper (arXiv:2601.00675v2, Table ``all``) defines its
``Overall (MAE)`` as the *group-wise* mean of the MAE values from all 23
RoboRewardBench subsets.  It is therefore not the same statistic as the
per-episode (micro) MAE.

The authors have not released a standalone benchmark evaluator.  This module
implements the published scoring definition exactly and makes its assumptions
machine-checkable; it deliberately does not claim that it is the authors'
unreleased evaluator.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


PAPER_VERSION = "RoboRewardBench paper v2 (arXiv:2601.00675v2)"
PAPER_PROTOCOL_ID = "roborewardbench_paper_v2"
PAPER_METRIC_NAME = "group-wise MAE over all 23 RoboRewardBench subsets"
PAPER_REPORTED_ROBOREWARD_8B_MAE = 0.665
HF_MODEL_REVISION = "3a185b4fce2b1253643105be1f234ae618b9732f"

# Order follows Table all in the paper.  ``paper_roboreward_8b_mae`` values
# are retained only as a transparent reference column, not as gold labels.
PAPER_SUBSETS: tuple[dict[str, Any], ...] = (
    {"id": "robo_arena", "name": "RoboArena", "paper_roboreward_8b_mae": 0.768},
    {"id": "austin_sirius_dataset_converted_externally_to_rlds", "name": "Austin Sirius", "paper_roboreward_8b_mae": 0.701},
    {"id": "berkeley_autolab_ur5", "name": "Berkeley Autolab UR5", "paper_roboreward_8b_mae": 0.340},
    {"id": "berkeley_fanuc_manipulation", "name": "Berkeley Fanuc Manipulation", "paper_roboreward_8b_mae": 0.538},
    {"id": "berkeley_mvp_converted_externally_to_rlds", "name": "Berkeley MVP", "paper_roboreward_8b_mae": 0.577},
    {"id": "berkeley_rpt_converted_externally_to_rlds", "name": "Berkeley RPT", "paper_roboreward_8b_mae": 0.674},
    {"id": "bridge", "name": "Berkeley Bridge", "paper_roboreward_8b_mae": 0.770},
    {"id": "cmu_play_fusion", "name": "CMU Play Fusion", "paper_roboreward_8b_mae": 0.587},
    {"id": "dlr_edan_shared_control_converted_externally_to_rlds", "name": "DLR Wheelchair Shared Control", "paper_roboreward_8b_mae": 0.474},
    {"id": "droid", "name": "DROID", "paper_roboreward_8b_mae": 0.670},
    {"id": "fractal20220817_data", "name": "RT-1 Robot Action", "paper_roboreward_8b_mae": 0.830},
    {"id": "iamlab_cmu_pickup_insert_converted_externally_to_rlds", "name": "CMU Franka Pick-Insert", "paper_roboreward_8b_mae": 0.807},
    {"id": "jaco_play", "name": "USC Jaco Play", "paper_roboreward_8b_mae": 0.720},
    {"id": "kaist_nonprehensile_converted_externally_to_rlds", "name": "KAIST Nonprehensile Objects", "paper_roboreward_8b_mae": 0.396},
    {"id": "roboturk", "name": "Roboturk", "paper_roboreward_8b_mae": 0.485},
    {"id": "stanford_hydra_dataset_converted_externally_to_rlds", "name": "Stanford HYDRA", "paper_roboreward_8b_mae": 0.890},
    {"id": "taco_play", "name": "Freiburg Franka Play", "paper_roboreward_8b_mae": 0.473},
    {"id": "tokyo_u_lsmo_converted_externally_to_rlds", "name": "LSMO", "paper_roboreward_8b_mae": 0.493},
    {"id": "ucsd_kitchen_dataset_converted_externally_to_rlds", "name": "UCSD Kitchen", "paper_roboreward_8b_mae": 0.947},
    {"id": "ucsd_pick_and_place_dataset_converted_externally_to_rlds", "name": "UCSD Pick Place", "paper_roboreward_8b_mae": 0.100},
    {"id": "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds", "name": "Tokyo PR2 Tabletop Manipulation", "paper_roboreward_8b_mae": 0.452},
    {"id": "utokyo_xarm_bimanual_converted_externally_to_rlds", "name": "UTokyo xArm Bimanual", "paper_roboreward_8b_mae": 1.394},
    {"id": "viola", "name": "Austin VIOLA", "paper_roboreward_8b_mae": 1.200},
)
PAPER_SUBSET_IDS = frozenset(item["id"] for item in PAPER_SUBSETS)


def _score_error(row: dict[str, Any]) -> str | None:
    example_id = str(row.get("example_id", "<missing example_id>"))
    if row.get("status") != "ok":
        return f"{example_id}: status is not ok"
    prediction = row.get("native_prediction")
    if isinstance(prediction, bool) or not isinstance(prediction, int) or prediction not in range(1, 6):
        return f"{example_id}: native_prediction must be an integer in 1..5"
    reward = row.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, int) or reward not in range(1, 6):
        return f"{example_id}: reward must be an integer in 1..5"
    return None


def published_metric_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate the published 23-group metric and audit eligibility.

    Invalid/partial records never silently contribute to the result.  A report
    is still produced for diagnostics, but ``paper_metric_comparable`` remains
    false unless the input is exactly a complete, valid 23-subset benchmark.
    """
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for row in rows:
        error = _score_error(row)
        if error:
            errors.append(error)
        by_subset[str(row.get("subset", "<missing subset>"))].append(row)

    observed_ids = set(by_subset)
    missing = sorted(PAPER_SUBSET_IDS - observed_ids)
    unexpected = sorted(observed_ids - PAPER_SUBSET_IDS)
    if missing:
        errors.append("missing paper subsets: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected subsets: " + ", ".join(unexpected))
    empty = sorted(key for key in PAPER_SUBSET_IDS if not by_subset.get(key))
    if empty:
        errors.append("empty paper subsets: " + ", ".join(empty))

    per_subset: list[dict[str, Any]] = []
    for descriptor in PAPER_SUBSETS:
        subset_rows = by_subset.get(descriptor["id"], [])
        # Do not publish a partial score for a malformed subset as though it
        # were valid.  Its MAE is intentionally null until every row validates.
        subset_errors = [error for row in subset_rows if (error := _score_error(row))]
        value = (
            mean(abs(row["native_prediction"] - row["reward"]) for row in subset_rows)
            if subset_rows and not subset_errors
            else None
        )
        per_subset.append(
            {
                **descriptor,
                "n": len(subset_rows),
                "mae": value,
                "difference_from_paper_roboreward_8b": (
                    value - descriptor["paper_roboreward_8b_mae"] if value is not None else None
                ),
            }
        )

    comparable = not errors and len(per_subset) == len(PAPER_SUBSETS)
    groupwise_mae = mean(item["mae"] for item in per_subset) if comparable else None
    roboarena = next(item for item in per_subset if item["id"] == "robo_arena")
    oxe = [item["mae"] for item in per_subset if item["id"] != "robo_arena"]
    return {
        "protocol": PAPER_PROTOCOL_ID,
        "metric_definition": PAPER_METRIC_NAME,
        "source": {
            "paper": PAPER_VERSION,
            "paper_table": "Table all / Table 1",
            "paper_reported_roboreward_8b_groupwise_mae": PAPER_REPORTED_ROBOREWARD_8B_MAE,
            "checkpoint_hf_revision": HF_MODEL_REVISION,
            "official_standalone_evaluator_released": False,
        },
        "paper_metric_comparable": comparable,
        "validation_errors": errors,
        "expected_subset_count": len(PAPER_SUBSETS),
        "observed_subset_count": len(observed_ids),
        "missing_subset_ids": missing,
        "unexpected_subset_ids": unexpected,
        "groupwise_mae": groupwise_mae,
        "difference_from_paper_reported_roboreward_8b": (
            groupwise_mae - PAPER_REPORTED_ROBOREWARD_8B_MAE
            if groupwise_mae is not None
            else None
        ),
        "roboarena_mae": roboarena["mae"] if comparable else None,
        "oxe_groupwise_mae": mean(oxe) if comparable else None,
        "per_subset": per_subset,
    }
