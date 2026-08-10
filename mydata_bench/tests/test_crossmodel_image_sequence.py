from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mydata_bench.config import load_config
from mydata_bench.qwen_eval.protocols import (
    ROBOREWARDBENCH_IMAGE_SEQUENCE,
    image_sequence_messages,
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


def test_crossmodel_config_matrix_is_complete_and_isolated() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "v2_crossmodel"
    configs = sorted(root.glob("*.yaml"))
    assert len(configs) == 8
    for path in configs:
        config = load_config(path)
        section = config.get("roboreward_eval") or config.get("qwen_eval") or config.get(
            "attention_steer"
        )
        assert section is not None
        assert "experiments_v2_corssmodel" in section["output_dir"]
        assert section.get("num_images") == 8
        if "attention_steer" in config or "qwen_eval" in config:
            assert section.get("protocol") == ROBOREWARDBENCH_IMAGE_SEQUENCE
