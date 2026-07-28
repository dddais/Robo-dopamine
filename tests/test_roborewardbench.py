from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import jsonschema

from rewardbench.data import inventory, load_episodes, normalize_subset
from rewardbench.metrics import clustered_stratified_bootstrap, compute_metrics
from rewardbench.protocol import (
    IMAGE_LABELS,
    OFFICIAL_SYSTEM_PROMPT,
    SIMPLIFIED_SYSTEM_PROMPT,
    chat_messages,
    native_endpoint_payload,
    parse_score,
    progress,
    progress_to_reward,
    system_prompt,
)
from rewardbench.raw_eval.runner import OFFICIAL_SAMPLING, sampling_kwargs
from rewardbench.schemas import EpisodeRecord, FrameRecord
from rewardbench.video import extract_endpoints, uniform_indices


class DataProtocolTests(unittest.TestCase):
    def test_prompt_modes_and_official_prompt_source_match(self) -> None:
        source_path = Path(__file__).resolve().parents[1] / "examples" / "inference.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        original = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
                for target in node.targets
            )
        )
        self.assertEqual(OFFICIAL_SYSTEM_PROMPT, original)
        self.assertNotEqual(OFFICIAL_SYSTEM_PROMPT, SIMPLIFIED_SYSTEM_PROMPT)
        self.assertEqual(system_prompt("official"), original)
        self.assertEqual(
            sum(item["type"] == "image" for item in chat_messages("move block", "official")[0]["content"]),
            8,
        )
        with self.assertRaisesRegex(ValueError, "prompt_mode"):
            system_prompt("unknown")

    def test_official_sampling_defaults_and_explicit_override(self) -> None:
        self.assertEqual(sampling_kwargs({}), OFFICIAL_SAMPLING)
        self.assertEqual(
            sampling_kwargs({"temperature": 0, "top_p": 1, "top_k": -1, "max_tokens": 16}),
            {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 16},
        )

    def test_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test").mkdir()
            (root / "test" / "metadata.jsonl").write_text(
                json.dumps(
                    {"file_name": "../../escape.mp4", "task": "move block", "reward": 1}
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                list(load_episodes(root, compute_hash=False, require_video=False))

    def test_metadata_inventory_and_model_payload_do_not_leak_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "test" / "toy" / "x.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"x")
            (root / "test" / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "file_name": "toy/x.mp4",
                        "task": "move block",
                        "reward": 1,
                        "gpt5_mini_check": "private label explanation",
                    }
                )
                + "\n"
            )
            episode = next(load_episodes(root))
            self.assertNotIn("reward", episode.model_payload())
            self.assertNotIn("gpt5_mini_check", episode.model_payload())
            self.assertEqual(inventory([episode])["num_records"], 1)
            self.assertEqual(normalize_subset("toy\\x.mp4"), "toy")

    def test_native_eight_image_order_and_blank_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, last, blank = (root / name for name in ("first.png", "last.png", "blank.png"))
            for path in (first, last, blank):
                path.write_bytes(b"x")
            episode = EpisodeRecord("x", "x.mp4", "move block", 1, "toy", "a" * 64)
            frames = FrameRecord(
                "x", "a" * 64, 0, 9, str(first), str(last), 10, 10,
                "b" * 64, "c" * 64, 10,
            )
            payload = native_endpoint_payload(episode, frames, blank)
            self.assertEqual(payload["image_labels"], list(IMAGE_LABELS))
            self.assertEqual(payload["image"][0], str(first))
            self.assertEqual(payload["image"][1], str(blank.resolve()))
            self.assertEqual(payload["image"][2:5], [str(first)] * 3)
            self.assertEqual(payload["image"][5:8], [str(last)] * 3)
            self.assertNotIn("reward", payload)

    def test_episode_static_json_schema(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "rewardbench"
            / "schemas"
            / "episode.schema.json"
        )
        schema = json.loads(schema_path.read_text())
        episode = EpisodeRecord("x", "x.mp4", "move block", 1, "toy", "a" * 64)
        jsonschema.validate(episode.to_dict(), schema)

    def test_video_reads_real_first_and_terminal_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "toy.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
            )
            for value in (10, 50, 200):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()
            result = extract_endpoints("x", "a" * 64, video, root / "frames")
            self.assertEqual(result.first_index, 0)
            self.assertEqual(result.last_index, 2)
            first = cv2.imread(result.first_path).mean()
            last = cv2.imread(result.last_path).mean()
            self.assertLess(first, last)
            with self.assertRaisesRegex(RuntimeError, "Cannot open"):
                extract_endpoints("bad", "b" * 64, root / "missing.mp4", root / "bad")


class ScoreMetricTests(unittest.TestCase):
    def test_strict_parser_and_boundaries(self) -> None:
        self.assertEqual(parse_score("<score>+25%</score>"), 0.25)
        self.assertEqual(parse_score("<score>-100%</score>"), -1.0)
        for invalid in ("prefix <score>0%</score>", "<score>25</score>", "<score>101%</score>"):
            with self.assertRaises(ValueError):
                parse_score(invalid)
        self.assertEqual(progress(-0.1), 0)
        self.assertEqual(progress_to_reward(0.124999), 1)
        self.assertEqual(progress_to_reward(0.125), 2)
        self.assertEqual(progress_to_reward(0.875), 5)

    def test_macro_micro_confusion_and_bootstrap(self) -> None:
        rows = [
            {"status": "ok", "progress": 0, "reward": 1, "subset": "a", "video_sha256": "a"},
            {"status": "ok", "progress": 0, "reward": 5, "subset": "b", "video_sha256": "b"},
            {"status": "ok", "progress": 0, "reward": 5, "subset": "b", "video_sha256": "c"},
            {"status": "ok", "progress": 0, "reward": 5, "subset": "b", "video_sha256": "d"},
        ]
        result = compute_metrics(rows)
        self.assertEqual(result["macro_subset_mae"], 2)
        self.assertEqual(result["micro"]["mae"], 3)
        self.assertEqual(result["confusion_matrix"]["5"]["1"], 3)
        boot = clustered_stratified_bootstrap(
            rows, lambda draw: compute_metrics(draw)["macro_subset_mae"], samples=50
        )
        self.assertEqual(boot["ci95"], [2.0, 2.0])

    def test_uniform_indices_include_true_endpoints(self) -> None:
        self.assertEqual(uniform_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(uniform_indices(2, 8), [0, 1])


if __name__ == "__main__":
    unittest.main()
