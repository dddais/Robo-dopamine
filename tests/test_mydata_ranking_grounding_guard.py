from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mydata_bench.attention_eval.dataset import prepare_grounded_ranking_samples


class RankingGroundingGuardTests(unittest.TestCase):
    def test_rejects_unexpected_ranking_source_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention = {
                "dataset_root": str(root),
                "ranking_metadata_file": str(root / "ranking_data.jsonl"),
                "ranking_grounding_run": str(root / "sam3"),
                "ranking_expected_source_count": 2,
            }
            episodes = [SimpleNamespace(example_id="only-one")]
            with patch(
                "mydata_bench.attention_eval.dataset.load_configured_episodes",
                return_value=(episodes, None),
            ):
                with self.assertRaisesRegex(
                    ValueError, "expected 2, found 1"
                ):
                    prepare_grounded_ranking_samples(
                        attention, root / "grounded_ranking_inputs.jsonl"
                    )

    def test_rejects_missing_endpoint_records_before_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention = {
                "dataset_root": str(root),
                "ranking_metadata_file": str(root / "ranking_data.jsonl"),
                "ranking_grounding_run": str(root / "sam3"),
                "ranking_expected_source_count": 1,
                "ranking_require_complete_grounding": True,
            }
            episodes = [SimpleNamespace(example_id="ranking-a")]

            def fake_read_jsonl(path: str | Path) -> list[dict]:
                if Path(path).name == "targets.jsonl":
                    return [{"example_id": "ranking-a"}]
                return [
                    {
                        "example_id": "ranking-a",
                        "frame": "first",
                        "status": "no_detection",
                    }
                ]

            with (
                patch(
                    "mydata_bench.attention_eval.dataset.load_configured_episodes",
                    return_value=(episodes, None),
                ),
                patch(
                    "mydata_bench.attention_eval.dataset.read_jsonl",
                    side_effect=fake_read_jsonl,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "1/1 source examples are missing"
                ):
                    prepare_grounded_ranking_samples(
                        attention, root / "grounded_ranking_inputs.jsonl"
                    )


    def test_rejects_unusable_grounding_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention = {
                "dataset_root": str(root),
                "ranking_metadata_file": str(root / "ranking_data.jsonl"),
                "ranking_grounding_run": str(root / "sam3"),
                "ranking_expected_source_count": 1,
                "ranking_expected_usable_count": 1,
                "ranking_require_complete_grounding": True,
            }
            episodes = [SimpleNamespace(example_id="ranking-a")]

            def fake_read_jsonl(path: str | Path) -> list[dict]:
                if Path(path).name == "targets.jsonl":
                    return [{"example_id": "ranking-a"}]
                return [
                    {
                        "example_id": "ranking-a",
                        "frame": frame,
                        "status": "no_detection",
                    }
                    for frame in ("first", "last")
                ]

            with (
                patch(
                    "mydata_bench.attention_eval.dataset.load_configured_episodes",
                    return_value=(episodes, None),
                ),
                patch(
                    "mydata_bench.attention_eval.dataset.read_jsonl",
                    side_effect=fake_read_jsonl,
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "expected 1, found 0"
                ):
                    prepare_grounded_ranking_samples(
                        attention, root / "grounded_ranking_inputs.jsonl"
                    )


if __name__ == "__main__":
    unittest.main()
