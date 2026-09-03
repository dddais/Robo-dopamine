from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mydata_bench.attention_eval.experiment import _record as grm_record
from mydata_bench.attention_eval.masking import make_attention_mask_hook
from mydata_bench.qwen_eval.attention_experiment import _completed_conditions


class AttentionResumeTests(unittest.TestCase):
    def test_per_head_bias_weights_scale_selected_heads(self) -> None:
        mask = torch.zeros((1, 1, 2, 5), dtype=torch.float32)
        hook = make_attention_mask_hook(
            [0, 2], [1], [3], 3, 4, head_bias_weights=[1.0, 0.25]
        )
        _, output = hook(None, (), {"attention_mask": mask})
        changed = output["attention_mask"]
        self.assertTrue(torch.all(changed[0, 0, :, 1] == 4))
        self.assertTrue(torch.all(changed[0, 0, :, 3] == -4))
        self.assertTrue(torch.all(changed[0, 2, :, 1] == 1))
        self.assertTrue(torch.all(changed[0, 2, :, 3] == -1))
        self.assertTrue(torch.all(changed[0, 1] == 0))

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
