from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mydata_bench.config import load_config
from mydata_bench.qwen_eval.attention_experiment import (
    _steering_biases,
    _steering_condition_kinds,
)
from mydata_bench.qwen_eval.attention import (
    build_discrete_prefix_constraint,
    extract_discrete_label_logprobs,
)
from mydata_bench.qwen_eval.protocols import (
    INTERLEAVED_REWARD_PROMPT,
    ROBOREWARDBENCH_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
    image_sequence_messages,
    interleaved_image_sequence_messages,
    parse_protocol_output,
    protocol_descriptor,
)
from mydata_bench.video import extract_uniform_image_sequence


def _write_video(path: Path, frame_count: int = 10) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()


def test_uniform_image_sequence_is_fixed_and_terminal_aligned(tmp_path: Path) -> None:
    video = tmp_path / "rollout.avi"
    _write_video(video)
    paths, record = extract_uniform_image_sequence(
        video, tmp_path / "images", count=8
    )
    assert len(paths) == 8
    assert record["selected_source_indices"] == [0, 1, 3, 4, 5, 6, 8, 9]
    assert record["terminal_source_index"] == 9
    assert record["terminal_frame_in_last_image"] is True
    cached_paths, cached_record = extract_uniform_image_sequence(
        video, tmp_path / "images", count=8
    )
    assert cached_paths == paths
    assert cached_record == record


def test_uniform_image_sequence_rejects_short_rollout(tmp_path: Path) -> None:
    video = tmp_path / "short.avi"
    _write_video(video, frame_count=7)
    with pytest.raises(ValueError, match="requires 8 decoded frames"):
        extract_uniform_image_sequence(video, tmp_path / "images", count=8)


def test_image_sequence_protocol_keeps_each_frame_as_an_image_item(
    tmp_path: Path,
) -> None:
    images = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        assert cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
        images.append(str(path))
    text_first = image_sequence_messages(
        "move the cup", images, content_order="text_then_images"
    )[0]["content"]
    images_first = image_sequence_messages(
        "move the cup", images, content_order="images_then_text"
    )[0]["content"]
    assert [item["type"] for item in text_first] == [
        "text",
        "image",
        "image",
        "image",
    ]
    assert [item["type"] for item in images_first] == [
        "image",
        "image",
        "image",
        "text",
    ]
    descriptor = protocol_descriptor(
        ROBOREWARDBENCH_IMAGE_SEQUENCE,
        content_order="text_then_images",
    )
    assert descriptor["adapter_protocol"] is True
    assert parse_protocol_output(
        ROBOREWARDBENCH_IMAGE_SEQUENCE, "ANSWER: 4"
    ) == {"native_prediction": 4, "progress": 0.75}


def test_interleaved_protocol_places_text_around_every_image(
    tmp_path: Path,
) -> None:
    images = []
    for index in range(8):
        path = tmp_path / f"{index}.png"
        assert cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
        images.append(str(path))
    messages = interleaved_image_sequence_messages("move the cup", images)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [item["type"] for item in content] == ["text", "image"] * 8 + [
        "text"
    ]
    assert "move the cup" in content[0]["text"]
    assert "OBSERVATION 1" in content[0]["text"]
    assert "OBSERVATION 8" in content[-3]["text"]
    assert "Rubric for end-of-episode progress" in content[-1]["text"]
    assert "ANSWER: <score>" in content[-1]["text"]
    assert INTERLEAVED_REWARD_PROMPT.count("<image>") == 8
    descriptor = protocol_descriptor(
        ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
        content_order="interleaved",
    )
    assert descriptor["content_order"] == "interleaved"
    assert descriptor["output"] == "ANSWER: <1-5>"


def test_crossmodel_config_matrix_is_complete_and_isolated() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "v2_crossmodel"
    configs = sorted(root.glob("*.yaml"))
    assert len(configs) == 16
    for path in configs:
        config = load_config(path)
        section = config.get("roboreward_eval") or config.get("qwen_eval") or config.get(
            "attention_steer"
        )
        assert section is not None
        assert "experiments_v2_corssmodel" in section["output_dir"]
        assert section.get("num_images") == 8
        if "attention_steer" in config or "qwen_eval" in config:
            assert section.get("protocol") in {
                ROBOREWARDBENCH_IMAGE_SEQUENCE,
                ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
            }
        if path.name.startswith(("attention_09_", "attention_10_", "attention_11_", "attention_12_")):
            assert section.get("temporal_intervention_scope") == "all_frames"
            assert section.get("negative_scope") == "all_visual"
            assert section.get("ranking_score_kind") == "raw_mass"
            assert section.get("skip_early_layers") == 8
            assert section.get("steering_query_scope") == "all"
            assert section.get("top_k_values") == [8, 32, 64]
            assert section.get("swap_bias") == 6
        if path.name.startswith(("attention_13_", "attention_14_", "attention_15_", "attention_16_")):
            assert (
                section.get("protocol")
                == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE
            )
            assert section.get("content_order") == "interleaved"
            assert section.get("ranking_score_kind") == "raw_mass"
            assert section.get("skip_early_layers") == 8
            assert section.get("steering_query_scope") == "all"
            assert section.get("top_k_values") == [8, 32, 64]
            assert section.get("swap_bias") == 6
            if path.name.startswith(("attention_13_", "attention_15_")):
                assert section.get("temporal_intervention_scope") == "all_frames"
                assert section.get("negative_scope") == "all_visual"
            else:
                assert section.get("temporal_intervention_scope") == "last_frame"
                assert section.get("negative_scope") == "target_span"


def test_auto_research_k_aware_bias_and_control_contract() -> None:
    attention = {
        "swap_bias": 6,
        "swap_bias_by_top_k": {"8": 2.0, 32: 1.0},
        "control_conditions": [],
    }
    assert _steering_biases(attention, [8, 16, 32]) == {
        8: 2.0,
        16: 6.0,
        32: 1.0,
    }
    assert _steering_condition_kinds(attention) == ("candidate_target",)
    assert _steering_condition_kinds(
        {"control_conditions": ["candidate_target_only"]}
    ) == ("candidate_target", "candidate_target_only")


def test_extract_discrete_label_logprobs_handles_space_prefixed_tokens() -> None:
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [20 + int(text.strip())] if text.startswith(" ") else [10 + int(text)]

        def decode(self, values: list[int], skip_special_tokens: bool = True) -> str:
            assert skip_special_tokens is True
            return "ANSWER: " if values else ""

    scores = [torch.zeros(1, 30), torch.zeros(1, 30), torch.zeros(1, 30)]
    scores[2][0, 21:26] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    result = extract_discrete_label_logprobs(
        Tokenizer(), torch.tensor([7, 9, 23]), scores, 3, torch
    )
    assert result["generated_step"] == 2
    assert result["token_prefix"] == " "
    assert max(result["values"], key=result["values"].get) == "5"


def test_discrete_prefix_constraint_only_allows_documented_answers() -> None:
    class Tokenizer:
        eos_token_id = 99

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [1, 2, 20 + int(text[-1])]

    allowed = build_discrete_prefix_constraint(Tokenizer(), prompt_length=2)
    assert allowed(0, torch.tensor([8, 9])) == [1]
    assert allowed(0, torch.tensor([8, 9, 1])) == [2]
    assert allowed(0, torch.tensor([8, 9, 1, 2])) == [21, 22, 23, 24, 25]
    assert allowed(0, torch.tensor([8, 9, 1, 2, 23])) == [99]
