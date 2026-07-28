from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rewardbench.exact_pairs import freeze_exact_pairs
from rewardbench.io import read_jsonl, write_jsonl
from rewardbench.attention_eval.experiment import _paired_non_destructive_metrics


class ExactPairManifestTests(unittest.TestCase):
    def test_selects_only_id_and_hash_matched_audited_counterfactuals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.jsonl"
            eligible = root / "eligible.jsonl"
            first_hash, second_hash = "a" * 64, "b" * 64
            write_jsonl(
                pairs,
                [
                    {
                        "pair_id": first_hash,
                        "video_sha256": first_hash,
                        "subset": "toy",
                        "counterfactual_example_id": "counter/first.mp4",
                        "counterfactual_task": "counter first",
                        "original_example_id": "original/first.mp4",
                        "original_task": "original first",
                    },
                    {
                        "pair_id": second_hash,
                        "video_sha256": second_hash,
                        "subset": "toy",
                        "counterfactual_example_id": "counter/second.mp4",
                        "counterfactual_task": "counter second",
                        "original_example_id": "original/second.mp4",
                        "original_task": "original second",
                    },
                ],
            )
            write_jsonl(
                eligible,
                [
                    {"example_id": "counter/first.mp4", "video_sha256": first_hash, "subset": "toy", "target_type": "object"},
                    {"example_id": "counter/unpaired.mp4", "video_sha256": "c" * 64, "subset": "toy", "target_type": "object"},
                ],
            )
            result = freeze_exact_pairs(pairs, eligible, root / "out", expected_pairs=1)
            selected = list(read_jsonl(root / "out" / "paired_reward1_reward5_exact40.jsonl"))
            excluded = list(read_jsonl(root / "out" / "excluded_raw_pairs.jsonl"))
            self.assertEqual([row["pair_id"] for row in selected], [first_hash])
            self.assertEqual(excluded[0]["pair_id"], second_hash)
            self.assertEqual(result["audited_counterfactual_without_raw_pair_count"], 1)

    def test_refuses_an_unexpected_candidate_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "a" * 64
            write_jsonl(
                root / "pairs.jsonl",
                [{"pair_id": digest, "video_sha256": digest, "counterfactual_example_id": "x"}],
            )
            write_jsonl(root / "eligible.jsonl", [])
            with self.assertRaisesRegex(ValueError, "Expected 1 exact-pair candidates"):
                freeze_exact_pairs(root / "pairs.jsonl", root / "eligible.jsonl", root / "out", expected_pairs=1)

    def test_non_destructive_metrics_reports_delta_distribution_and_flips(self) -> None:
        digest = "a" * 64
        pair = {
            "pair_id": digest,
            "video_sha256": digest,
            "subset": "toy",
            "counterfactual_example_id": "pair/a/counterfactual",
            "original_example_id": "pair/a/original",
        }
        grouped = {
            pair["counterfactual_example_id"]: {
                "baseline": {"signed_score": -0.2},
                "candidate_target": {"signed_score": -0.6},
            },
            pair["original_example_id"]: {
                "baseline": {"signed_score": 0.9},
                "candidate_target": {"signed_score": 0.7},
                "candidate_wrong": {"signed_score": 0.8},
                "low_rank_target": {"signed_score": 0.85},
            },
        }
        result = _paired_non_destructive_metrics(
            [pair], grouped, samples=20, non_inferiority_margin=-0.05
        )
        self.assertEqual(result["complete_pair_count"], 1)
        self.assertEqual(result["label_distribution"]["baseline"]["5"], 1)
        self.assertEqual(result["label_distribution"]["candidate_target"]["4"], 1)
        self.assertEqual(result["reward5_to_less_than5_flip_count"], 1)
        self.assertEqual(result["non_inferiority_decision"], "evidence_of_harm_beyond_margin")


if __name__ == "__main__":
    unittest.main()
