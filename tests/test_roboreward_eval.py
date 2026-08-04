from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from rewardbench.roboreward_eval.cli import _counterfactual_reward1_metrics, _score
from rewardbench.roboreward_eval.paper_protocol import (
    PAPER_PROTOCOL_ID,
    PAPER_REPORTED_ROBOREWARD_8B_MAE,
    PAPER_SUBSETS,
    published_metric_report,
)
from rewardbench.roboreward_eval.runner import (
    _validate_paper_protocol_configuration,
    _requested_example_ids,
    _sample_video_at_fps,
    _sample_video_uniform_frames,
    _native_video_message,
    _use_checkpoint_native_video,
    native_prompt,
    parse_native_score,
    run,
)


class RoboRewardProtocolTests(unittest.TestCase):
    def test_frozen_id_file_is_label_free_and_rejects_ambiguous_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example_ids.json"
            path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
            self.assertEqual(
                _requested_example_ids({"example_ids_file": str(path)}), {"a", "b"}
            )
            with self.assertRaisesRegex(ValueError, "only one"):
                _requested_example_ids(
                    {"example_ids": ["a"], "example_ids_file": str(path)}
                )

    def test_exact_counterfactual_metrics_does_not_pool_other_reward1_rows(self) -> None:
        rows = [
            {
                "example_id": "counterfactual",
                "subset": "s",
                "reward": 1,
                "progress": 0.0,
                "status": "ok",
            },
            {
                "example_id": "other_reward1",
                "subset": "s",
                "reward": 1,
                "progress": 1.0,
                "status": "ok",
            },
        ]
        summary = _counterfactual_reward1_metrics(
            rows, counterfactual_ids={"counterfactual"}
        )
        self.assertEqual(summary["expected_count"], 1)
        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(summary["prediction_counts"], {
            "1": 1, "2": 0, "3": 0, "4": 0, "5": 0,
        })
        self.assertEqual(summary["predicted_one_rate"], 1.0)

    def test_documented_answer_parser_is_strict(self) -> None:
        self.assertEqual(parse_native_score("ANSWER: 4"), 4)
        self.assertEqual(parse_native_score("Reasoning...\nanswer : 2\n"), 2)
        with self.assertRaisesRegex(ValueError, "ANSWER"):
            parse_native_score("The reward is 1")
        self.assertIn("Task: move the block", native_prompt("move the block"))

    def test_native_prompt_matches_public_helm_rubric(self) -> None:
        expected = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Task: move the block"""
        self.assertEqual(native_prompt("move the block"), expected)

    def test_complete_video_sampling_retains_terminal_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
            )
            for value in range(15):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()
            paths, record = _sample_video_at_fps(video, root / "frames", sample_fps=1.0)
            self.assertEqual(record["selected_source_indices"], [0, 10, 14])
            self.assertEqual(record["terminal_source_index"], 14)
            self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_uniform_sampling_is_fixed_and_retains_terminal_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
            )
            for value in range(15):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()
            paths, record = _sample_video_uniform_frames(
                video, root / "frames", num_frames=8
            )
            self.assertEqual(record["video_input_protocol"], "uniform_fixed_frames_v1")
            self.assertEqual(
                record["selected_source_indices"], [0, 2, 4, 6, 8, 10, 12, 14]
            )
            self.assertEqual(record["terminal_source_index"], 14)
            self.assertEqual(len(paths), 8)

    def test_uniform_sampling_requires_positive_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "num_frames"):
                _sample_video_uniform_frames(
                    Path(directory) / "does-not-matter.mp4",
                    Path(directory) / "frames",
                    num_frames=0,
                )

    def test_checkpoint_native_video_preserves_original_mp4_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
            )
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.release()
            paths, record = _use_checkpoint_native_video(video)
            self.assertEqual(paths, [str(video.resolve())])
            self.assertEqual(record["video_input_protocol"], "checkpoint_native_mp4_v1")
            self.assertFalse(record["custom_frame_extraction"])

    def test_checkpoint_native_video_honors_content_order(self) -> None:
        helm_message = _native_video_message(
            "move block", "/tmp/episode.mp4", content_order="text_then_video"
        )
        model_card_message = _native_video_message(
            "move block", "/tmp/episode.mp4", content_order="video_then_text"
        )

        helm_content = helm_message[0]["content"]
        model_card_content = model_card_message[0]["content"]
        self.assertEqual(
            [item["type"] for item in helm_content], ["text", "video"]
        )
        self.assertEqual(
            [item["type"] for item in model_card_content], ["video", "text"]
        )
        self.assertEqual(model_card_content[0]["video"], "/tmp/episode.mp4")
        self.assertIn("Task: move block", model_card_content[1]["text"])

    def test_dry_run_preserves_native_discrete_metric_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data" / "test" / "toy"
            data.mkdir(parents=True)
            video = data / "episode.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
            )
            for _ in range(3):
                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.release()
            metadata = root / "data" / "test" / "metadata.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "file_name": "toy/episode.avi",
                        "task": "move block",
                        "reward": 1,
                        "gpt5_mini_check": "must not reach the model",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "output"
            ids_path = root / "example_ids.json"
            ids_path.write_text(json.dumps(["toy/episode.avi"]), encoding="utf-8")
            config = {
                "roboreward_eval": {
                    "dataset_root": str(root / "data"),
                    "split": "test",
                    "output_dir": str(output),
                    "model_path": str(root / "missing-model"),
                    "video_sample_fps": 1.0,
                    "example_ids_file": str(ids_path),
                }
            }
            run(config, dry_run=True)
            # Dry records prove plumbing only; replacing their status with ok
            # isolates the score adapter without loading a model.
            path = output / "records.shard-00.jsonl"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["status"] = "ok"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            metrics = _score(output, bootstrap_samples=10)
            self.assertTrue(metrics["official_native_discrete_output"])
            self.assertFalse(metrics["adapter_metric"])
            self.assertEqual(metrics["prediction_counts"], {"1": 1, "2": 0, "3": 0, "4": 0, "5": 0})
            self.assertEqual(metrics["reward1"]["predicted_one_rate"], 1.0)

    def test_paper_protocol_uses_unweighted_subset_mae_not_micro_mae(self) -> None:
        rows = []
        for descriptor in PAPER_SUBSETS:
            rows.append(
                {
                    "example_id": descriptor["id"] + "/ok",
                    "subset": descriptor["id"],
                    "reward": 1,
                    "native_prediction": 1,
                    "status": "ok",
                }
            )
        # A second, completely wrong RoboArena episode changes one group's MAE
        # to 2.0; it must not receive twice the weight in the paper metric.
        rows.append(
            {
                "example_id": "robo_arena/wrong",
                "subset": "robo_arena",
                "reward": 1,
                "native_prediction": 5,
                "status": "ok",
            }
        )
        report = published_metric_report(rows)
        self.assertTrue(report["paper_metric_comparable"])
        self.assertAlmostEqual(report["groupwise_mae"], 2.0 / 23.0)
        self.assertAlmostEqual(report["roboarena_mae"], 2.0)
        self.assertAlmostEqual(report["oxe_groupwise_mae"], 0.0)
        # The same rows would produce a different micro value: 4 error / 24 rows.
        self.assertNotAlmostEqual(report["groupwise_mae"], 4.0 / 24.0)

    def test_paper_protocol_rejects_incomplete_or_non_native_scores(self) -> None:
        row = {
            "example_id": "one",
            "subset": "robo_arena",
            "reward": 1,
            "progress": 0.0,
            "status": "ok",
        }
        report = published_metric_report([row])
        self.assertFalse(report["paper_metric_comparable"])
        self.assertIsNone(report["groupwise_mae"])
        self.assertTrue(any("native_prediction" in item for item in report["validation_errors"]))

    def test_paper_reference_values_round_to_reported_overall(self) -> None:
        reference_mean = sum(item["paper_roboreward_8b_mae"] for item in PAPER_SUBSETS) / len(PAPER_SUBSETS)
        self.assertAlmostEqual(reference_mean, PAPER_REPORTED_ROBOREWARD_8B_MAE, places=3)

    def test_paper_run_configuration_rejects_custom_sampling_or_cohorts(self) -> None:
        valid = {
            "evaluation_protocol": PAPER_PROTOCOL_ID,
            "video_sampling_mode": "checkpoint_native_video",
            "preprocessor_mode": "checkpoint_default",
            "content_order": "text_then_video",
            "do_sample": False,
        }
        _validate_paper_protocol_configuration(valid)
        invalid = {**valid, "video_sampling_mode": "full_1fps"}
        with self.assertRaisesRegex(ValueError, "video_sampling_mode"):
            _validate_paper_protocol_configuration(invalid)
        invalid = {**valid, "example_ids": ["only-one"]}
        with self.assertRaisesRegex(ValueError, "complete benchmark"):
            _validate_paper_protocol_configuration(invalid)


if __name__ == "__main__":
    unittest.main()
