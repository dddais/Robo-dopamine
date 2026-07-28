from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rewardbench.data import load_pair_episodes
from rewardbench.io import sha256_file
from rewardbench.io import read_jsonl, write_jsonl
from rewardbench.raw_eval.pairs import prepare, score


class PairedRawEvalTests(unittest.TestCase):
    def test_pair_attention_inputs_are_instruction_specific_and_label_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "episode.mp4"
            video.write_bytes(b"synthetic-video")
            digest = sha256_file(video)
            manifest = root / "pairs.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "pair_id": digest,
                        "video_sha256": digest,
                        "video_path": str(video),
                        "subset": "toy",
                        "counterfactual_example_id": "counter/a.mp4",
                        "counterfactual_task": "Move the red block.",
                        "original_example_id": "original/a.mp4",
                        "original_task": "Move the blue cup.",
                    }
                ],
            )
            episodes, pairs = load_pair_episodes(manifest)
            self.assertEqual(len(episodes), 2)
            self.assertEqual(len(pairs), 1)
            self.assertNotEqual(episodes[0].example_id, episodes[1].example_id)
            self.assertEqual(episodes[0].video_path, episodes[1].video_path)
            self.assertEqual(episodes[0].video_sha256, episodes[1].video_sha256)
            for episode in episodes:
                self.assertNotIn("reward", episode.model_payload())
                self.assertNotIn("gpt5_mini_check", episode.model_payload())

    def test_prepare_freezes_same_video_different_instruction_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counter, original = root / "counter", root / "original"
            for dataset, task, reward in ((counter, "push red can", 1), (original, "knock blue bottle", 5)):
                video = dataset / "test" / "toy" / "episode.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(b"identical video bytes")
                (dataset / "test" / "metadata.jsonl").write_text(
                    json.dumps({"file_name": "toy/episode.mp4", "task": task, "reward": reward}) + "\n"
                )
            output = root / "out"
            config = {
                "paired_raw_eval": {
                    "counterfactual_dataset_root": str(counter),
                    "original_dataset_root": str(original),
                    "output_dir": str(output),
                    "model_path": str(root / "model"),
                    "expected_pairs": 1,
                }
            }
            path = prepare(config)
            rows = list(read_jsonl(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["counterfactual_reward"], 1)
            self.assertEqual(rows[0]["original_reward"], 5)
            self.assertNotEqual(rows[0]["counterfactual_task"], rows[0]["original_task"])

    def test_score_joins_only_complete_pairs_and_uses_original_minus_counterfactual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = {
                "pair_id": "a" * 64,
                "video_sha256": "a" * 64,
                "subset": "toy",
                "video_path": "unused.mp4",
                "counterfactual_example_id": "counter.mp4",
                "counterfactual_task": "counter task",
                "counterfactual_reward": 1,
                "original_example_id": "original.mp4",
                "original_task": "original task",
                "original_reward": 5,
            }
            write_jsonl(root / "pairs.jsonl", [pair])
            write_jsonl(
                root / "paired_records.shard-00.jsonl",
                [
                    {
                        "record_id": f"{pair['pair_id']}:counterfactual",
                        "example_id": "counter.mp4",
                        "status": "ok",
                        "signed_score": 0.1,
                        "progress": 0.1,
                    },
                    {
                        "record_id": f"{pair['pair_id']}:original",
                        "example_id": "original.mp4",
                        "status": "ok",
                        "signed_score": 0.8,
                        "progress": 0.8,
                    },
                ],
            )
            result = score(root, bootstrap_samples=20)
            self.assertTrue(result["formal_scoring_ready"])
            self.assertEqual(result["paired_valid"], 1)
            self.assertAlmostEqual(result["instruction_conditioned_score_difference"]["mean"], 0.7)
            self.assertEqual(result["prediction_direction"]["original_prediction_gt_counterfactual_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
