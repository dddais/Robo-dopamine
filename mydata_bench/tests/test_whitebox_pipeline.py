from __future__ import annotations

import json
from pathlib import Path

import pytest

from mydata_bench.io import read_jsonl, write_json, write_jsonl
from mydata_bench.my_dataset.attention_manifest import build_attention_manifests
from mydata_bench.my_dataset.causal_metrics import score_steering
from mydata_bench.my_dataset.media import grm_multiview_image_paths
from mydata_bench.my_dataset.roles import parse_instruction
from mydata_bench.my_dataset.splits import grouped_three_way_split
from mydata_bench.qwen_eval.protocols import (
    ROBOREWARDBENCH_NATIVE,
    native_video_message,
    protocol_descriptor,
)


def test_qwen_native_descriptor_and_message_freeze_media_order(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    message = native_video_message(
        "move cup", video, content_order="video_then_text"
    )
    descriptor = protocol_descriptor(
        ROBOREWARDBENCH_NATIVE, content_order="video_then_text"
    )
    assert [value["type"] for value in message[0]["content"]] == ["video", "text"]
    assert descriptor["content_order"] == "video_then_text"
    assert descriptor["media_order"] == ["video", "text"]
    assert descriptor["input"] == "original_mp4_video_then_text"


def test_instruction_roles_cover_all_relation_templates() -> None:
    ordinal = parse_instruction(
        "Pick up the third cup from the left and place it in the plate.",
        "ordinal_position",
    )
    relation = parse_instruction(
        "Pick up the cup to the right of the teapot and place it in the plate.",
        "left_right_relation",
    )
    distance = parse_instruction(
        "Pick up the pen farthest from the black tape and place it in the plate.",
        "distance_relation",
    )
    assert ordinal["target_instance"] == "third cup from the left"
    assert relation["reference_object"] == "teapot"
    assert relation["relation"] == "right"
    assert distance["target_instance"] == "pen farthest from black tape"
    assert all(value["requires_instance_review"] for value in (ordinal, relation, distance))


def test_three_way_split_has_no_group_or_media_leakage() -> None:
    rows = []
    for task_id in ("task1_1", "task3_3"):
        for group_index in range(4):
            for variant in range(3):
                rows.append(
                    {
                        "example_id": f"{task_id}-g{group_index}-e{variant}",
                        "group_id": f"{task_id}-g{group_index}",
                        "task_id": task_id,
                        "group_media_sha256": f"{task_id}-media-{group_index}",
                    }
                )
    split = grouped_three_way_split(rows, seed=3)
    assert split["group_counts"] == {"discovery": 2, "validation": 2, "test": 4}
    for first, second in (("discovery", "validation"), ("discovery", "test"), ("validation", "test")):
        assert not set(split["groups"][first]) & set(split["groups"][second])
        assert not set(split["media_sha256"][first]) & set(split["media_sha256"][second])
    assignment = {
        group_id: partition
        for partition, values in split["groups"].items()
        for group_id in values
    }
    assert all(
        len({assignment[row["group_id"]] for row in rows if row["group_id"] == group_id}) == 1
        for group_id in assignment
    )


def test_grm_layout_uses_real_left_and_right_views(tmp_path: Path) -> None:
    paths = {}
    for view in ("front", "left_wrist", "right_wrist"):
        first = tmp_path / f"{view}_first.png"
        last = tmp_path / f"{view}_last.png"
        first.write_bytes(view.encode())
        last.write_bytes((view + "last").encode())
        paths[view] = {"first_path": str(first), "last_path": str(last)}
    blank = tmp_path / "blank.png"
    blank.write_bytes(b"blank")
    images = grm_multiview_image_paths(paths, blank)
    assert images[3].endswith("left_wrist_first.png")
    assert images[4].endswith("right_wrist_first.png")
    assert images[6].endswith("left_wrist_last.png")
    assert images[7].endswith("right_wrist_last.png")
    assert len(set(images[2:5])) == 3
    assert len(set(images[5:8])) == 3


def _attention_fixture(tmp_path: Path) -> tuple[dict, Path]:
    videos = {}
    for view in ("front", "left_wrist", "right_wrist"):
        path = tmp_path / f"{view}.mp4"
        path.write_bytes(view.encode())
        videos[view] = str(path)
    images = []
    for index in range(8):
        path = tmp_path / f"slot{index}.png"
        path.write_bytes(str(index).encode())
        images.append(str(path))
    inputs_path = tmp_path / "inputs.jsonl"
    write_jsonl(
        inputs_path,
        [
            {
                "example_id": "e1",
                "group_id": "g1",
                "group_media_sha256": "group-sha",
                "task_id": "task1_1",
                "task_family": "object_identity",
                "instruction": "Pick up the cup and place it in the plate.",
                "video_paths": videos,
                "view_sha256": {key: key + "-sha" for key in videos},
            }
        ],
    )
    requests = tmp_path / "requests.jsonl"
    write_jsonl(
        requests,
        [
            {
                "example_id": "e1",
                "model_frames": {
                    "roboreward": {
                        "sampled_frame_indices": [0, 3],
                        "source_frame_index": 3,
                        "image_path": images[5],
                    },
                    "qwen": {
                        "sampled_frame_indices": [0, 1, 2, 3],
                        "source_frame_index": 3,
                        "image_path": images[5],
                    },
                    "grm": {"image_paths": images},
                },
            }
        ],
    )
    reviews = tmp_path / "reviews.jsonl"
    write_jsonl(
        reviews,
        [
            {
                "example_id": "e1",
                "status": "eligible",
                "models": {
                    "roboreward": {
                        "target": {"image_path": images[5], "bbox": [1, 2, 20, 30]},
                        "wrong_region": {"image_path": images[5], "bbox": [30, 2, 49, 30]},
                    },
                    "qwen": {
                        "target": {"image_path": images[5], "bbox": [1, 2, 20, 30]},
                        "wrong_region": {"image_path": images[5], "bbox": [30, 2, 49, 30]},
                    },
                    "grm": {
                        "target": {"image_path": images[5], "bbox": [1, 2, 20, 30]},
                        "wrong_region": {"image_path": images[5], "bbox": [30, 2, 49, 30]},
                    },
                },
            }
        ],
    )
    split = tmp_path / "split.json"
    write_json(split, {"examples": {"discovery": ["e1"], "validation": [], "test": []}})
    config = {
        "my_dataset_attention": {
            "inputs_path": str(inputs_path),
            "grounding_requests_path": str(requests),
            "grounding_reviews_path": str(reviews),
            "split_path": str(split),
            "output_dir": str(tmp_path / "attention"),
        }
    }
    return config, reviews


def test_attention_manifests_are_model_specific_and_label_free(tmp_path: Path) -> None:
    config, _reviews = _attention_fixture(tmp_path)
    manifest_path = build_attention_manifests(config)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["labels_opened"] is False
    grm = list(read_jsonl(tmp_path / "attention" / "grm" / "discovery.jsonl"))[0]
    qwen = list(read_jsonl(tmp_path / "attention" / "qwen" / "discovery.jsonl"))[0]
    assert grm["image_paths"][3].endswith("slot3.png")
    assert grm["image_paths"][4].endswith("slot4.png")
    assert qwen["processor_frame_indices"] == [0, 1, 2, 3]
    serialized = json.dumps([grm, qwen])
    assert "protocol_reward" not in serialized
    assert "instruction_video_match" not in serialized


def test_legacy_attention_checks_request_metadata_only_when_present(
    tmp_path: Path,
) -> None:
    config, _reviews = _attention_fixture(tmp_path)
    requests_path = Path(
        config["my_dataset_attention"]["grounding_requests_path"]
    )
    request = list(read_jsonl(requests_path))[0]
    request["group_id"] = "different-group"
    write_jsonl(requests_path, [request])

    with pytest.raises(ValueError, match="request metadata differs"):
        build_attention_manifests(config)


def test_causal_metrics_keep_correction_and_harm_denominators_separate(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {"example_id": "fail", "protocol_reward": 1, "instruction_video_match": False},
            {"example_id": "suc", "protocol_reward": 5, "instruction_video_match": True},
        ],
    )
    records = []
    for example_id, baseline, target in (("fail", 3, 1), ("suc", 5, 4)):
        for condition, prediction in {
            "baseline": baseline,
            "candidate_target": target,
            "candidate_wrong": baseline,
            "low_rank_target": baseline,
            "layer_matched_random_target": baseline,
        }.items():
            records.append(
                {
                    "example_id": example_id,
                    "group_id": example_id + "-group",
                    "task_id": "task1_1",
                    "task_family": "object_identity",
                    "condition": condition,
                    "native_prediction": prediction,
                    "progress": (prediction - 1) / 4,
                    "status": "ok",
                }
            )
    records_path = tmp_path / "steering.jsonl"
    write_jsonl(records_path, records)
    result = score_steering(
        records_path,
        labels,
        tmp_path / "scoring",
        bootstrap_samples=20,
        seed=5,
    )
    assert result["fail_correction_rate"] == 1.0
    assert result["suc_harm_rate"] == 1.0
    assert result["balanced_net_correction"] == 0.0
