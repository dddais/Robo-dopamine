from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from mydata_bench.attention_eval.experiment import (
    _required_steering_conditions,
    _steering_sample_complete,
    metrics as attention_metrics,
)
from mydata_bench.attention_eval.dataset import _latest_ok_endpoints

from mydata_bench.attention_eval.masking import ImageSpan, resolve_negative_positions
from mydata_bench.qwen_eval.attention import (
    duplicate_temporal_frames,
    reduce_temporal_bboxes,
)
from mydata_bench.data import load_episodes
from mydata_bench.grounding.base import select_relational_candidate
from mydata_bench.grounding.parser import heuristic_parse
from mydata_bench.grounding.sam3 import SAM3Grounder
from mydata_bench.raw_eval.pairs import _pair_rows
from mydata_bench.review_grounding_web import ReviewStore
from mydata_bench.write_exp_records import grm_distribution, grm_distribution_label


DATASET = Path("/home/dais/workspace/data/ljx_lfz_task/new")


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        (0.0, 1),
        (0.199999, 1),
        (0.2, 2),
        (0.4, 3),
        (0.6, 4),
        (0.8, 5),
        (1.0, 5),
    ],
)
def test_grm_distribution_uses_equal_width_progress_bins(
    progress: float, expected: int
) -> None:
    assert grm_distribution_label(progress) == expected


def test_grm_distribution_reports_all_five_labels() -> None:
    rows = [
        {"progress": progress, "split": split, "task_id": "task"}
        for progress, split in zip(
            (0.0, 0.2, 0.4, 0.6, 0.8),
            ("suc", "fail", "suc", "fail", "suc"),
            strict=True,
        )
    ]
    table = grm_distribution(rows)
    assert "label 1 (0–20%)" in table
    assert "label 5 (80–100%)" in table
    assert "uncertain" not in table
    assert "| task:task | 5 | 1 (20.00%) | 1 (20.00%) | 1 (20.00%) | 1 (20.00%) | 1 (20.00%) |" in table


def test_ranking_grounding_uses_latest_endpoint_status() -> None:
    rows = [
        {"example_id": "a", "frame": "first", "status": "ok"},
        {"example_id": "a", "frame": "last", "status": "ok"},
        {"example_id": "a", "frame": "last", "status": "no_detection"},
        {"example_id": "b", "frame": "first", "status": "invalid"},
        {"example_id": "b", "frame": "first", "status": "ok"},
        {"example_id": "b", "frame": "last", "status": "ok"},
    ]
    endpoints = _latest_ok_endpoints(rows)
    assert set(endpoints["a"]) == {"first"}
    assert set(endpoints["b"]) == {"first", "last"}


@pytest.mark.skipif(not DATASET.is_dir(), reason="ljx_lfz_task dataset is unavailable")
def test_real_dataset_inventory_without_video_hashing() -> None:
    rows = list(load_episodes(DATASET, "all", compute_hash=False))
    assert len(rows) == 755
    assert Counter(row.split for row in rows) == {"suc": 169, "fail": 586}
    assert Counter(row.reward for row in rows) == {5: 169, 1: 586}
    assert all(
        set(row.views) == {"front", "left_wrist", "right_wrist"}
        for row in rows
    )
    assert all(
        row.instruction_video_match is (row.split == "suc")
        for row in rows
    )


def test_one_success_can_pair_with_multiple_counterfactuals(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    paths = [
        "suc/task/1/faceImg.mp4",
        "suc/task/1/leftImg.mp4",
        "suc/task/1/rightImg.mp4",
    ]
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    rows = [
        {
            "id": "suc/task/1",
            "split": "suc",
            "task_id": "task",
            "instruction": "Pick up the cup and place it in the plate.",
            "instruction_video_match": True,
            "source_suc_id": "suc/task/1",
            "video_paths": paths,
        },
        {
            "id": "fail/task/1/a",
            "split": "fail",
            "task_id": "task",
            "instruction": "Pick up the pen and place it in the plate.",
            "instruction_video_match": False,
            "source_suc_id": "suc/task/1",
            "video_paths": paths,
        },
        {
            "id": "fail/task/1/b",
            "split": "fail",
            "task_id": "task",
            "instruction": "Pick up the carrot and place it in the plate.",
            "instruction_video_match": False,
            "source_suc_id": "suc/task/1",
            "video_paths": paths,
        },
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    pairs = _pair_rows(
        {
            "paired_raw_eval": {
                "dataset_root": str(root),
                "output_dir": str(tmp_path / "output"),
                "model_path": str(tmp_path / "unused-model"),
                "split": "all",
            }
        }
    )
    assert len(pairs) == 2
    assert {row["source_suc_id"] for row in pairs} == {"suc/task/1"}
    assert len({row["pair_id"] for row in pairs}) == 2
    assert len({row["video_sha256"] for row in pairs}) == 1


@pytest.mark.parametrize(
    ("phrase", "relation"),
    [
        ("cup to the left of the teapot", "left_of"),
        ("cup to the right of the yellow block", "right_of"),
        ("cup closest to the carrot", "closest_to"),
        ("cup farthest from the purple cup", "farthest_from"),
    ],
)
def test_relational_phrase_is_not_split_by_destination_parser(
    phrase: str, relation: str
) -> None:
    target = heuristic_parse(f"Pick up the {phrase} and place it in the plate.")
    assert target.target_phrase == phrase
    assert target.head_noun == "cup"
    assert target.reference_object
    assert target.relation == relation


@pytest.mark.parametrize(
    ("relation", "expected_x1"),
    [
        ("left_of", 10),
        ("right_of", 75),
        ("closest_to", 75),
        ("farthest_from", 10),
    ],
)
def test_relational_geometry(relation: str, expected_x1: float, tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    Image.new("RGB", (120, 80), "white").save(image)
    targets = [
        {"bbox": [10, 20, 20, 30], "score": 0.8, "query": "cup"},
        {"bbox": [75, 20, 85, 30], "score": 0.7, "query": "cup"},
    ]
    references = [
        {"bbox": [60, 20, 70, 30], "score": 0.9, "query": "yellow block"}
    ]
    selected, reference, reason = select_relational_candidate(
        str(image), targets, references, relation
    )
    assert selected is not None and reference is not None
    assert selected["bbox"][0] == expected_x1
    assert reason.endswith(relation)


def test_negative_scope_contracts() -> None:
    spans = [
        ImageSpan("before", "a.png", 10, 14, (1, 2, 2)),
        ImageSpan("after", "b.png", 20, 24, (1, 2, 2)),
    ]
    selected = [20, 21]
    assert resolve_negative_positions(spans, selected, "all_visual")[0] == [
        10,
        11,
        12,
        13,
        22,
        23,
    ]
    assert resolve_negative_positions(spans, selected, "target_span")[0] == [22, 23]
    assert resolve_negative_positions(spans, selected, "other_spans")[0] == [
        10,
        11,
        12,
        13,
    ]
    assert resolve_negative_positions(spans, selected, "none")[0] == []
    with pytest.raises(ValueError):
        resolve_negative_positions(spans, selected, "unknown")


def test_duplicate_temporal_frames_creates_identical_pairs() -> None:
    frames = np.arange(3 * 2).reshape(3, 2)
    duplicated = duplicate_temporal_frames(frames)
    assert duplicated.tolist() == [
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
        [4, 5],
        [4, 5],
    ]


def test_temporal_bbox_reducers() -> None:
    boxes = [[10, 20, 30, 40], [15, 10, 35, 36]]
    assert reduce_temporal_bboxes(boxes, "last") == [15, 10, 35, 36]
    assert reduce_temporal_bboxes(boxes, "union") == [10, 10, 35, 40]
    assert reduce_temporal_bboxes(boxes, "intersection") == [15, 20, 30, 36]
    with pytest.raises(ValueError, match="empty"):
        reduce_temporal_bboxes([[0, 0, 1, 1], [2, 2, 3, 3]], "intersection")


def test_tracking_media_must_stay_inside_run_dir(tmp_path: Path) -> None:
    run = tmp_path / "sam3"
    run.mkdir()
    inside = run / "tracks" / "preview.mp4"
    inside.parent.mkdir()
    inside.write_bytes(b"video")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"image")
    template = {
        "example_id": "sample",
        "tracking_preview_path": str(inside),
        "tracking_contact_sheet_path": str(outside),
        "endpoints": {},
    }
    (run / "audit_template.jsonl").write_text(
        json.dumps(template) + "\n", encoding="utf-8"
    )
    store = ReviewStore(run, "reviewer1")
    assert store.media_path("sample", "preview") == inside.resolve()
    with pytest.raises(ValueError, match="escapes"):
        store.media_path("sample", "contact")


def test_sam3_box_prompt_tracking_and_session_close(monkeypatch: pytest.MonkeyPatch) -> None:
    class Capture:
        def isOpened(self):
            return True

        def get(self, key):
            return 100 if key in {3, 4} else 0

        def release(self):
            return None

    class Predictor:
        def __init__(self):
            self.requests = []

        def handle_request(self, request):
            self.requests.append(request)
            if request["type"] == "start_session":
                return {"session_id": "session"}
            if request["type"] == "add_prompt":
                return {
                    "outputs": {
                        "out_obj_ids": np.array([7]),
                        "out_probs": np.array([0.9]),
                        "out_boxes_xywh": np.array([[0.1, 0.2, 0.2, 0.2]]),
                        "out_binary_masks": np.ones((1, 4, 4), dtype=bool),
                    }
                }
            return {"is_success": True}

        def handle_stream_request(self, request):
            self.requests.append(request)
            yield {
                "frame_index": 1,
                "outputs": {
                    "out_obj_ids": np.array([7]),
                    "out_probs": np.array([0.8]),
                    "out_boxes_xywh": np.array([[0.2, 0.2, 0.2, 0.2]]),
                    "out_binary_masks": np.ones((1, 4, 4), dtype=bool),
                },
            }

    predictor = Predictor()
    grounder = SAM3Grounder({"model_path": "/unused", "device": "cpu"})
    grounder._video_predictor = predictor
    monkeypatch.setattr(
        "mydata_bench.grounding.sam3.cv2.VideoCapture", lambda _path: Capture()
    )
    rows = grounder.track("/tmp/input.mp4", [10, 20, 30, 40])
    add = next(row for row in predictor.requests if row["type"] == "add_prompt")
    assert add["bounding_boxes"] == [[0.1, 0.2, 0.2, 0.2]]
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert predictor.requests[-1]["type"] == "close_session"


def test_grm_steering_resume_requires_every_planned_control() -> None:
    attention = {
        "top_k": 8,
        "top_k_values": [8, 32, 64],
        "include_all_heads_control": False,
        "run_sensitivity": False,
        "query_scope_sensitivity": [],
        "run_duplicate_location_sensitivity": False,
    }
    required = _required_steering_conditions(attention)
    assert set(required) == {
        "baseline",
        "candidate_target",
        "candidate_wrong",
        "low_rank_target",
        "candidate_target_k8",
        "candidate_wrong_k8",
        "low_rank_target_k8",
        "candidate_target_k32",
        "candidate_wrong_k32",
        "low_rank_target_k32",
        "candidate_target_k64",
        "candidate_wrong_k64",
        "low_rank_target_k64",
    }
    rows = {
        condition: {"condition": condition, "status": "ok"}
        for condition in required
    }
    rows["candidate_wrong_k32"]["status"] = "missing_control"
    assert _steering_sample_complete(rows, required, dry_run=False)
    rows.pop("low_rank_target_k64")
    assert not _steering_sample_complete(rows, required, dry_run=False)


def test_grm_metrics_report_each_top_k_estimand(tmp_path: Path) -> None:
    run = tmp_path / "attention"
    run.mkdir()
    scores = {
        "baseline": 0.0,
        "candidate_target": 0.8,
        "candidate_wrong": 0.2,
        "low_rank_target": 0.4,
    }
    for top_k, target, wrong, low in (
        (8, 0.8, 0.2, 0.4),
        (32, 0.6, 0.1, 0.3),
        (64, 0.4, 0.0, 0.2),
    ):
        scores[f"candidate_target_k{top_k}"] = target
        scores[f"candidate_wrong_k{top_k}"] = wrong
        scores[f"low_rank_target_k{top_k}"] = low
    records = [
        {
            "example_id": "sample",
            "video_sha256": "video",
            "condition": condition,
            "signed_score": score,
            "status": "ok",
        }
        for condition, score in scores.items()
    ]
    (run / "steering.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (run / "eligible.jsonl").write_text(
        json.dumps({"example_id": "sample", "subset": "task"}) + "\n",
        encoding="utf-8",
    )
    result = attention_metrics(
        run,
        {
            "attention_eval": {
                "bootstrap_samples": 20,
                "eligibility_mode": "auto_valid_grounding",
                "steering_partition": "all_eligible",
                "top_k": 8,
                "top_k_values": [8, 32, 64],
                "include_all_heads_control": False,
                "run_sensitivity": False,
                "query_scope_sensitivity": [],
                "run_duplicate_location_sensitivity": False,
            }
        },
    )
    assert set(result["top_k_estimands"]) == {"8", "32", "64"}
    assert result["top_k_estimands"]["8"]["n_formal_contrasts"] == 1
    assert result["top_k_estimands"]["8"]["estimands"]["target_shift"][
        "mean"
    ] == pytest.approx(0.8)
    assert result["top_k_estimands"]["8"]["estimands"][
        "spatial_specificity"
    ]["mean"] == pytest.approx(0.6)
    assert result["top_k_estimands"]["8"]["estimands"]["head_specificity"][
        "mean"
    ] == pytest.approx(0.4)
