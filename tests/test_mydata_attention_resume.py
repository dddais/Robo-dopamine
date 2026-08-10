from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mydata_bench.attention_eval.experiment import _record as grm_record
from mydata_bench.qwen_eval.attention_experiment import _completed_conditions


class AttentionResumeTests(unittest.TestCase):
    def test_resume_requires_current_fingerprints_and_latest_ok_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steering.jsonl"
            rows = [
                {"example_id": "b", "condition": "baseline", "status": "ok", "ranking_fingerprint": "rank"},
                {"example_id": "a", "grounding_fingerprint": "g", "condition": "wrong-rank", "status": "ok", "ranking_fingerprint": "old"},
                {"example_id": "a", "grounding_fingerprint": "g", "condition": "baseline", "status": "ok", "ranking_fingerprint": "rank"},
                {"example_id": "a", "grounding_fingerprint": "g", "condition": "target", "status": "ok", "ranking_fingerprint": "rank"},
                {"example_id": "a", "grounding_fingerprint": "g", "condition": "target", "status": "invalid", "ranking_fingerprint": "rank"},
                {"example_id": "a", "grounding_fingerprint": "g2", "condition": "baseline", "status": "ok", "ranking_fingerprint": "rank"},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            done = _completed_conditions(path, ranking_fingerprint="rank")
            self.assertEqual(done[("a", "g")], {"baseline"})
            self.assertEqual(done[("a", "g2")], {"baseline"})
            self.assertFalse(any(key[0] == "b" for key in done))

    def test_grm_record_carries_both_resume_fingerprints(self) -> None:
        sample = {"example_id": "a", "video_sha256": "v", "last": {"grounding_fingerprint": "ground"}}
        result = {
            "hook_diagnostics": {"dry_run": True},
            "raw_output": "<score>0%</score>",
            "signed_score": 0.0,
        }
        row = grm_record(
            sample, "source", "rank", (), 0, "baseline", result, [1], [1]
        )
        self.assertEqual(row["grounding_fingerprint"], "ground")
        self.assertEqual(row["ranking_fingerprint"], "rank")


if __name__ == "__main__":
    unittest.main()
