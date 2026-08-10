from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mydata_bench.attention_eval.ranking import aggregate_in_domain, consensus_ranking


class IndependentRankingTests(unittest.TestCase):
    def test_complete_single_source_is_not_labeled_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            rows = [
                {"layer": layer, "head": head, "score": float(4 - layer - head)}
                for layer in range(2)
                for head in range(2)
            ]
            path.write_text(
                json.dumps(
                    {"num_layers": 2, "num_heads": 2, "rankings": {"mean": rows}}
                )
            )
            result = consensus_ranking(
                [path], expected_layers=2, expected_heads=2
            )
            self.assertEqual(
                result["ranking_source"], "independent_single_source_ranking"
            )
            self.assertEqual(result["method"], "single_source_normalized_rank")
            self.assertEqual(len(result["ranking"]), 4)


    def test_in_domain_aggregation_uses_latest_record_per_example(self) -> None:
        def record(example_id: str, value: float, status: str = "ok") -> dict:
            row = {"example_id": example_id, "status": status}
            if status == "ok":
                row.update(
                    {
                        "raw_mass": [[value, value], [value, value]],
                        "excess_mass": [[value, value], [value, value]],
                    }
                )
            return row

        result = aggregate_in_domain(
            [record("a", 1.0), record("b", 3.0), record("a", 5.0)],
            num_layers=2,
            num_heads=2,
            skip_layers=0,
        )
        self.assertEqual(result["n_discovery_samples"], 2)
        self.assertEqual(result["per_sample_example_ids"], ["a", "b"])
        self.assertEqual(result["ranking"][0]["mean_raw_mass"], 4.0)

        latest_failure = aggregate_in_domain(
            [record("a", 1.0), record("b", 3.0), record("a", 0.0, "invalid")],
            num_layers=2,
            num_heads=2,
            skip_layers=0,
        )
        self.assertEqual(latest_failure["n_discovery_samples"], 1)
        self.assertEqual(latest_failure["per_sample_example_ids"], ["b"])
