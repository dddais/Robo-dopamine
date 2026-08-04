from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rewardbench.qwen_eval.cli import _score
from rewardbench.qwen_eval.attention import (
    build_forward_image_spans,
    forward_image_paths,
)
from rewardbench.protocol import IMAGE_LABELS
from rewardbench.qwen_eval.protocols import (
    ROBO_DOPAMINE_FORWARD,
    ROBOREWARDBENCH_NATIVE,
    dopamine_forward_messages,
    dopamine_forward_payload,
    native_video_payload,
    parse_protocol_output,
    protocol_descriptor,
    validate_protocol,
)
from rewardbench.qwen_eval.runner import requested_example_ids, run
from rewardbench.schemas import EpisodeRecord, FrameRecord


class QwenProtocolTests(unittest.TestCase):
    def test_forward_attention_slots_bind_terminal_cam_high_to_index_five(self) -> None:
        paths = forward_image_paths("first.png", "last.png", "blank.png")
        spans = [(10 * index, 10 * index + 4) for index in range(8)]
        grids = [(1, 4, 4)] * 8
        image_spans = build_forward_image_spans(paths, spans, grids)
        self.assertEqual([span.label for span in image_spans], list(IMAGE_LABELS))
        self.assertEqual(
            [Path(span.path).name for span in image_spans],
            [
                "first.png",
                "blank.png",
                "first.png",
                "first.png",
                "first.png",
                "last.png",
                "last.png",
                "last.png",
            ],
        )
        target = next(span for span in image_spans if span.label == "after_cam_high")
        self.assertEqual(image_spans.index(target), 5)
        self.assertEqual(Path(target.path).name, "last.png")
        self.assertEqual((target.start, target.end, target.grid_thw), (50, 54, (1, 4, 4)))

    def test_protocols_are_strict_and_semantically_distinct(self) -> None:
        native = protocol_descriptor(ROBOREWARDBENCH_NATIVE)
        dopamine = protocol_descriptor(ROBO_DOPAMINE_FORWARD)
        self.assertEqual(native["output"], "ANSWER: <1-5>")
        self.assertEqual(dopamine["output"], "<score>[+-]NN%</score>")
        self.assertEqual(
            parse_protocol_output(ROBOREWARDBENCH_NATIVE, "ANSWER: 4"),
            {"native_prediction": 4, "progress": 0.75},
        )
        self.assertEqual(
            parse_protocol_output(ROBO_DOPAMINE_FORWARD, "<score>-50%</score>"),
            {"signed_score": -0.5, "progress": 0.0},
        )
        with self.assertRaisesRegex(ValueError, "Unknown qwen_eval.protocol"):
            validate_protocol("unknown")

    def test_native_payload_is_label_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "episode.mp4"
            video.write_bytes(b"video")
            episode = EpisodeRecord(
                example_id="toy/episode.mp4",
                video_path=str(video),
                task="move block",
                reward=5,
                subset="toy",
                video_sha256="a" * 64,
                gpt5_mini_check="must never enter the payload",
            )
            payload = native_video_payload(episode)
            self.assertEqual(payload["protocol"], ROBOREWARDBENCH_NATIVE)
            self.assertNotIn("reward", payload)
            self.assertNotIn("gpt5_mini_check", payload)
            self.assertIn("Task: move block", payload["prompt"])

    def test_dopamine_messages_interleave_all_eight_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in range(3):
                path = root / f"image{index}.png"
                Image.new("RGB", (10, 10), "white").save(path)
                paths.append(str(path))
            episode = EpisodeRecord(
                example_id="toy/video.mp4",
                video_path=str(root / "video.mp4"),
                task="move block",
                reward=1,
                subset="toy",
                video_sha256="b" * 64,
            )
            frames = FrameRecord(
                example_id=episode.example_id,
                video_sha256=episode.video_sha256,
                first_index=0,
                last_index=1,
                first_path=paths[0],
                last_path=paths[1],
                width=10,
                height=10,
                first_sha256="c" * 64,
                last_sha256="d" * 64,
                reported_frame_count=2,
            )
            payload = dopamine_forward_payload(episode, frames, paths[2], prompt_mode="official")
            message = dopamine_forward_messages(payload)
            content = message[0]["content"]
            images = [item["image"] for item in content if item["type"] == "image"]
            self.assertEqual(len(images), 8)
            self.assertEqual(images, [str(Path(path).resolve()) for path in payload["image"]])
            self.assertLess(
                next(index for index, item in enumerate(content) if item["type"] == "image"),
                len(content) - 1,
            )


class QwenDryRunTests(unittest.TestCase):
    def _dataset(self, root: Path) -> tuple[Path, Path]:
        data = root / "data" / "test" / "toy"
        data.mkdir(parents=True)
        video = data / "episode.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
        )
        for index in range(3):
            writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
        writer.release()
        metadata = root / "data" / "test" / "metadata.jsonl"
        metadata.write_text(
            json.dumps(
                {
                    "file_name": "toy/episode.avi",
                    "task": "move block",
                    "reward": 1,
                    "gpt5_mini_check": "not model-facing",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        blank = root / "blank.png"
        Image.new("RGB", (32, 24), "white").save(blank)
        return root / "data", blank

    def _make_ok_and_score(self, output: Path) -> dict:
        record_path = output / "records.shard-00.jsonl"
        row = json.loads(record_path.read_text(encoding="utf-8"))
        row["status"] = "ok"
        record_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return _score(output, bootstrap_samples=10)

    def test_native_dry_run_records_direct_discrete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, blank = self._dataset(root)
            output = root / "native"
            config = {
                "qwen_eval": {
                    "dataset_root": str(data),
                    "output_dir": str(output),
                    "model_path": str(root / "missing-qwen"),
                    "protocol": ROBOREWARDBENCH_NATIVE,
                    "blank_goal": str(blank),
                }
            }
            run(config, dry_run=True)
            row = json.loads((output / "records.shard-00.jsonl").read_text())
            self.assertEqual(row["protocol"], ROBOREWARDBENCH_NATIVE)
            self.assertEqual(row["raw_output"], "ANSWER: 1")
            self.assertEqual(row["native_prediction"], 1)
            self.assertIsNone(row["frame_record"])
            metrics = self._make_ok_and_score(output)
            self.assertTrue(metrics["official_native_discrete_output"])
            self.assertFalse(metrics["adapter_metric"])
            self.assertTrue(metrics["completion"]["formal_scoring_ready"])

    def test_dopamine_dry_run_records_eight_image_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, blank = self._dataset(root)
            output = root / "dopamine"
            ids = root / "ids.json"
            ids.write_text(json.dumps(["toy/episode.avi"]), encoding="utf-8")
            config = {
                "qwen_eval": {
                    "dataset_root": str(data),
                    "output_dir": str(output),
                    "model_path": str(root / "missing-qwen"),
                    "protocol": ROBO_DOPAMINE_FORWARD,
                    "prompt_mode": "official",
                    "blank_goal": str(blank),
                    "example_ids_file": str(ids),
                }
            }
            self.assertEqual(requested_example_ids(config["qwen_eval"]), {"toy/episode.avi"})
            run(config, dry_run=True)
            row = json.loads((output / "records.shard-00.jsonl").read_text())
            self.assertEqual(row["protocol"], ROBO_DOPAMINE_FORWARD)
            self.assertEqual(row["raw_output"], "<score>0%</score>")
            self.assertEqual(row["signed_score"], 0.0)
            self.assertEqual(row["frame_record"]["first_index"], 0)
            metrics = self._make_ok_and_score(output)
            self.assertFalse(metrics["official_native_discrete_output"])
            self.assertTrue(metrics["adapter_metric"])
            self.assertTrue(metrics["completion"]["formal_scoring_ready"])


if __name__ == "__main__":
    unittest.main()
