from __future__ import annotations

import json
import shutil
from pathlib import Path

from mydata_bench.io import read_jsonl, write_jsonl
from mydata_bench.my_dataset.data import (
    FORBIDDEN_MODEL_FIELDS,
    audit_prepared,
    load_labels,
    load_model_inputs,
    prepare_dataset,
)
from mydata_bench.my_dataset.metrics import score_run
from mydata_bench.my_dataset.runner import run_baseline
from mydata_bench.roboreward_eval.runner import _native_video_message


VIEWS = ("faceImg.mp4", "leftImg.mp4", "rightImg.mp4")


def test_native_roboreward_message_honors_content_order() -> None:
    helm_message = _native_video_message(
        "move block", "/tmp/episode.mp4", content_order="text_then_video"
    )
    model_card_message = _native_video_message(
        "move block", "/tmp/episode.mp4", content_order="video_then_text"
    )

    assert [item["type"] for item in helm_message[0]["content"]] == [
        "text",
        "video",
    ]
    assert [item["type"] for item in model_card_message[0]["content"]] == [
        "video",
        "text",
    ]
    assert model_card_message[0]["content"][0]["video"] == "/tmp/episode.mp4"
    assert "Task: move block" in model_card_message[0]["content"][1]["text"]


def _make_group(
    root: Path, *, group_number: int, task_id: str, correct: str, wrong: str
) -> list[dict]:
    raw_dir = root.parent / "raw" / f"group-{group_number}"
    raw_dir.mkdir(parents=True)
    success_dir = root / "suc" / f"task-{group_number}" / "1"
    failure_dir = root / "fail" / f"task-{group_number}" / "1"
    success_dir.mkdir(parents=True)
    failure_dir.mkdir(parents=True)
    for index, name in enumerate(VIEWS):
        source = raw_dir / name
        source.write_bytes(f"group={group_number};view={index}".encode())
        shutil.copyfile(source, success_dir / name)
        shutil.copyfile(source, failure_dir / name)
    source_id = f"suc/task-{group_number}/1"
    common = {
        "task_id": task_id,
        "trajectory_index": 1,
        "source_suc_id": source_id,
        "source_raw_video_dir": str(raw_dir),
    }
    return [
        {
            **common,
            "id": source_id,
            "split": "suc",
            "instruction": correct,
            "target_obj": correct,
            "correct_target_obj": correct,
            "instruction_video_match": True,
            "video_paths": [
                str((success_dir / name).relative_to(root)) for name in VIEWS
            ],
        },
        {
            **common,
            "id": f"fail/task-{group_number}/1",
            "split": "fail",
            "instruction": wrong,
            "target_obj": wrong,
            "correct_target_obj": correct,
            "instruction_video_match": False,
            "video_paths": [
                str((failure_dir / name).relative_to(root)) for name in VIEWS
            ],
        },
    ]


def _prepare(tmp_path: Path) -> Path:
    source_root = tmp_path / "new"
    rows = _make_group(
        source_root,
        group_number=1,
        task_id="task1_1",
        correct="put the cup in the plate",
        wrong="put the pen in the plate",
    )
    rows += _make_group(
        source_root,
        group_number=2,
        task_id="task4_1",
        correct="put the left cup in the plate",
        wrong="put the right cup in the plate",
    )
    write_jsonl(source_root / "metadata.jsonl", rows)
    prepared = tmp_path / "prepared"
    prepare_dataset(
        {
            "my_dataset": {
                "dataset_name": "unit_cf",
                "source_root": str(source_root),
                "prepared_dir": str(prepared),
                "expected_groups": 2,
                "expected_examples": 4,
                "verify_counterfactual_bytes": True,
                "audit_video_metadata": False,
            }
        }
    )
    return prepared


def test_prepare_separates_model_inputs_and_labels(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    inputs_path = prepared / "model_inputs" / "inputs.jsonl"
    labels_path = prepared / "scoring" / "labels.jsonl"
    inputs = load_model_inputs(inputs_path)
    labels = load_labels(labels_path)
    assert len(inputs) == len(labels) == 4
    assert len({row["group_id"] for row in inputs}) == 2
    assert all(not (FORBIDDEN_MODEL_FIELDS & row.keys()) for row in inputs)
    assert all("/raw/" in row["video_paths"]["front"] for row in inputs)
    assert audit_prepared(inputs_path, labels_path)["passed"] is True


def test_native_dry_run_does_not_open_labels(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output_dir = tmp_path / "run"
    records_path = run_baseline(
        {
            "my_dataset_eval": {
                "model_family": "qwen",
                "inputs_path": str(prepared / "model_inputs" / "inputs.jsonl"),
                "output_dir": str(output_dir),
                "model_path": str(model_dir),
                "input_protocol": "checkpoint_native_front_video_v1",
            }
        },
        dry_run=True,
    )
    records = list(read_jsonl(records_path))
    assert len(records) == 4
    assert {row["status"] for row in records} == {"dry_run"}
    assert {row["protocol"] for row in records} == {
        "checkpoint_native_front_video_v1"
    }
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["labels_opened_by_inference"] is False
    assert "labels" not in json.dumps(manifest["config"])


def test_group_metrics_do_not_overweight_counterfactual_count(tmp_path: Path) -> None:
    prepared = _prepare(tmp_path)
    inputs_path = prepared / "model_inputs" / "inputs.jsonl"
    labels_path = prepared / "scoring" / "labels.jsonl"
    inputs = load_model_inputs(inputs_path)
    labels = {row["example_id"]: row for row in load_labels(labels_path)}
    groups = sorted({row["group_id"] for row in inputs})
    records = []
    for row in inputs:
        matched = labels[row["example_id"]]["instruction_video_match"]
        # First group separates correctly.  Second group ties at reward 5.
        if row["group_id"] == groups[0]:
            prediction = 5 if matched else 1
        else:
            prediction = 5
        records.append(
            {
                "example_id": row["example_id"],
                "group_id": row["group_id"],
                "task_id": row["task_id"],
                "task_family": row["task_family"],
                "model_family": "qwen",
                "protocol": "checkpoint_native_front_video_v1",
                "native_prediction": prediction,
                "progress": (prediction - 1) / 4,
                "raw_output": f"ANSWER: {prediction}",
                "status": "ok",
            }
        )
    run_dir = tmp_path / "scored_run"
    write_jsonl(run_dir / "records.shard-00.jsonl", records)
    result = score_run(
        run_dir,
        inputs_path=inputs_path,
        labels_path=labels_path,
        bootstrap_samples=100,
        seed=7,
    )
    assert result["completion"]["formal_scoring_ready"] is True
    assert result["num_examples"] == 4
    assert result["num_groups"] == 2
    assert result["num_pairs"] == 2
    assert set(result["example_by_task_id"]) == {"task1_1", "task4_1"}
    assert set(result["example_by_task_family"]) == {
        "left_right_relation",
        "object_identity",
    }
    assert result["group_macro"]["pairwise_accuracy"] == 0.5
    assert result["group_macro"]["strict_top1_correct"] == 0.5
    assert result["group_macro"]["group_exact_accuracy"] == 0.75
