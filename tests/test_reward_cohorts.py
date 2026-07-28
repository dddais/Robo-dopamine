from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rewardbench.cohorts import freeze_reward_cohort
from rewardbench.io import read_jsonl
from rewardbench.attention_eval.experiment import _single_label_cohort_metrics


class RewardCohortTests(unittest.TestCase):
    def test_freeze_reward_cohort_omits_labels_from_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            for name, reward in (("success.mp4", 5), ("failure.mp4", 1)):
                video = root / "test" / "toy" / name
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(name.encode())
            metadata = root / "test" / "metadata.jsonl"
            metadata.write_text(
                "\n".join(
                    json.dumps({"file_name": f"toy/{name}", "task": name, "reward": reward})
                    for name, reward in (("success.mp4", 5), ("failure.mp4", 1))
                )
                + "\n",
                encoding="utf-8",
            )
            out = root.parent / "out"
            summary = freeze_reward_cohort(root, out, reward=5, expected_count=1)
            rows = list(read_jsonl(out / "episodes.jsonl"))
            self.assertEqual([row["example_id"] for row in rows], ["toy/success.mp4"])
            self.assertNotIn("reward", rows[0])
            self.assertNotIn("gpt5_mini_check", rows[0])
            self.assertEqual(summary["selected_count"], 1)

    def test_single_label_metrics_only_use_configured_label_after_inference(self) -> None:
        grouped = {
            "a": {
                "baseline": {"video_sha256": "a", "signed_score": 0.9},
                "candidate_target": {"video_sha256": "a", "signed_score": 0.7},
                "candidate_wrong": {"video_sha256": "a", "signed_score": 0.4},
                "low_rank_target": {"video_sha256": "a", "signed_score": 0.8},
            }
        }
        result = _single_label_cohort_metrics(
            grouped, expected_reward=5, samples=20, non_inferiority_margin=-0.05
        )
        self.assertEqual(result["expected_reward_prediction_rate"]["baseline"], 1.0)
        self.assertEqual(result["expected_reward_prediction_rate"]["candidate_target"], 0.0)
        self.assertEqual(result["baseline_expected_reward_to_other_flip_count"], 1)
        self.assertEqual(result["non_inferiority_decision"], "evidence_of_harm_beyond_margin")


if __name__ == "__main__":
    unittest.main()
