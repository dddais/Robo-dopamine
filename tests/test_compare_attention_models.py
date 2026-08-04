from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rewardbench.compare_attention_models import (
    compare_rankings,
    steering_effect_rows,
)


class CrossModelComparisonTests(unittest.TestCase):
    def test_ranking_comparison_reports_exact_overlap_and_spearman(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rankings = [
                [(0, 0), (0, 1), (1, 0), (1, 1)],
                [(0, 0), (1, 0), (0, 1), (1, 1)],
            ]
            models = {}
            for index, ranking in enumerate(rankings):
                path = root / f"ranking{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "num_layers": 2,
                            "num_heads": 2,
                            "skip_early_layers": 0,
                            "ranking": [
                                {"layer": layer, "head": head}
                                for layer, head in ranking
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                models[f"m{index}"] = {"ranking_path": str(path)}
            result = compare_rankings(models, top_k=2)
            pair = result["pairwise"]["m0__vs__m1"]
            self.assertEqual(pair["intersection_count"], 1)
            self.assertEqual(pair["jaccard"], 1 / 3)
            self.assertAlmostEqual(pair["full_ranking_spearman"], 0.8)

    def test_steering_effect_rows_use_latest_complete_paired_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            values = {
                "baseline": 3,
                "candidate_target": 1,
                "candidate_wrong": 2,
                "low_rank_target": 3,
            }
            for condition, prediction in values.items():
                rows.append(
                    {
                        "example_id": "x",
                        "video_sha256": "v",
                        "subset": "s",
                        "condition": condition,
                        "status": "ok",
                        "native_prediction": prediction,
                    }
                )
            (root / "steering.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            effects = steering_effect_rows(root, expected_reward=1)
            self.assertEqual(len(effects), 1)
            self.assertEqual(effects[0]["target_shift"], -0.5)
            self.assertEqual(effects[0]["spatial_specificity"], -0.25)
            self.assertEqual(effects[0]["head_specificity"], -0.5)
            self.assertEqual(effects[0]["absolute_error_change"], -2)
            self.assertTrue(effects[0]["corrected"])


if __name__ == "__main__":
    unittest.main()
