from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from rewardbench.attention_eval.dataset import _formal_ids, grouped_stratified_split
from rewardbench.attention_eval.experiment import metrics as attention_metrics
from rewardbench.attention_eval.experiment import steer
from rewardbench.attention_eval.masking import (
    Head,
    ImageSpan,
    bbox_to_token_positions,
    make_attention_mask_hook,
    matched_wrong_position_set,
    select_low_ranked_heads,
)
from rewardbench.attention_eval.ranking import aggregate_in_domain, consensus_ranking
from rewardbench.attention_eval.stats import (
    exact_mcnemar_pvalue,
    holm,
    paired_cluster_bootstrap,
    paired_sign_flip_pvalue,
)


class MappingAndControlTests(unittest.TestCase):
    def test_cell_intersection_and_grid_length_validation(self) -> None:
        span = ImageSpan("after_cam_high", "x", 10, 26, (1, 8, 8))
        positions = bbox_to_token_positions(span, [0, 0, 50, 50], (100, 100), 2)
        self.assertEqual(positions, [10, 11, 14, 15])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            bbox_to_token_positions(
                ImageSpan("x", "x", 0, 15, (1, 8, 8)),
                [0, 0, 10, 10],
                (100, 100),
                2,
            )

    def test_wrong_region_matches_and_does_not_overlap(self) -> None:
        span = ImageSpan("after_cam_high", "x", 10, 26, (1, 8, 8))
        target = [10, 11, 14, 15]
        wrong = matched_wrong_position_set(span, target, spatial_merge_size=2)
        self.assertIsNotNone(wrong)
        self.assertEqual(len(wrong), len(target))
        self.assertFalse(set(wrong) & set(target))
        self.assertIsNone(
            matched_wrong_position_set(
                ImageSpan("x", "x", 0, 4, (1, 4, 4)),
                [0, 1, 2, 3],
                spatial_merge_size=2,
            )
        )

    def test_low_rank_control_never_overlaps_candidates(self) -> None:
        ranking = [{"layer": 0, "head": index} for index in range(8)]
        candidate = [Head(0, 0), Head(0, 1)]
        low = select_low_ranked_heads(ranking, 2, candidate)
        self.assertFalse(set(low) & set(candidate))


class HookTests(unittest.TestCase):
    def test_hook_only_modifies_selected_heads_and_new_text_keys_are_zero(self) -> None:
        diagnostics = {}
        hook = make_attention_mask_hook([1], [2], [3], 3, 6, diagnostics)
        mask = torch.zeros((1, 1, 4, 6))
        _, output = hook(None, (), {"attention_mask": mask})
        changed = output["attention_mask"]
        self.assertTrue(torch.equal(changed[:, 0], mask[:, 0]))
        self.assertEqual(changed[0, 1, 0, 2], 6)
        self.assertEqual(changed[0, 1, 0, 3], -6)
        self.assertEqual(changed[0, 1, 0, 5], 0)
        self.assertEqual(diagnostics["prefill_calls"], 1)

        decode_mask = torch.zeros((1, 1, 1, 8))
        _, decoded = hook(None, (), {"attention_mask": decode_mask})
        self.assertTrue(torch.equal(decoded["attention_mask"][..., 6:], torch.zeros(1, 3, 1, 2)))
        self.assertEqual(diagnostics["decode_calls"], 1)

    def test_zero_bias_is_numerically_identical(self) -> None:
        hook = make_attention_mask_hook([1], [2], [3], 3, 0)
        mask = torch.randn((1, 1, 4, 6))
        _, output = hook(None, (), {"attention_mask": mask})
        self.assertTrue(torch.equal(output["attention_mask"], mask.expand(1, 3, 4, 6)))

    def test_positive_bias_increases_selected_head_bbox_mass(self) -> None:
        logits = torch.zeros((1, 3, 1, 6))
        baseline = torch.softmax(logits, dim=-1)[0, 1, 0, 2]
        hook = make_attention_mask_hook([1], [2], [3, 4], 3, 6)
        _, output = hook(None, (), {"attention_mask": torch.zeros((1, 1, 1, 6))})
        steered = torch.softmax(logits + output["attention_mask"], dim=-1)[0, 1, 0, 2]
        self.assertGreater(steered, baseline)

    def test_query_scopes_separate_prefill_last_prompt_and_decode(self) -> None:
        prefill_mask = torch.zeros((1, 1, 4, 6))
        decode_mask = torch.zeros((1, 1, 1, 6))

        prefill_diagnostics = {}
        prefill = make_attention_mask_hook(
            [1],
            [2],
            [3],
            3,
            6,
            prefill_diagnostics,
            query_scope="prefill",
        )
        _, prefill_output = prefill(None, (), {"attention_mask": prefill_mask})
        self.assertTrue(torch.all(prefill_output["attention_mask"][0, 1, :, 2] == 6))
        self.assertIsNone(prefill(None, (), {"attention_mask": decode_mask}))
        self.assertEqual(prefill_diagnostics["prefill_applied_calls"], 1)
        self.assertEqual(prefill_diagnostics["decode_applied_calls"], 0)

        last_diagnostics = {}
        last_prompt = make_attention_mask_hook(
            [1],
            [2],
            [3],
            3,
            6,
            last_diagnostics,
            query_scope="last_prompt",
        )
        _, last_output = last_prompt(None, (), {"attention_mask": prefill_mask})
        changed = last_output["attention_mask"]
        self.assertTrue(torch.all(changed[0, 1, :-1, 2] == 0))
        self.assertEqual(changed[0, 1, -1, 2], 6)
        self.assertEqual(changed[0, 1, -1, 3], -6)
        self.assertIsNone(last_prompt(None, (), {"attention_mask": decode_mask}))
        self.assertEqual(last_diagnostics["applied_query_rows"], 1)

        decode_diagnostics = {}
        decode = make_attention_mask_hook(
            [1],
            [2],
            [3],
            3,
            6,
            decode_diagnostics,
            query_scope="decode",
        )
        self.assertIsNone(decode(None, (), {"attention_mask": prefill_mask}))
        _, decode_output = decode(None, (), {"attention_mask": decode_mask})
        self.assertEqual(decode_output["attention_mask"][0, 1, 0, 2], 6)
        self.assertEqual(decode_diagnostics["prefill_applied_calls"], 0)
        self.assertEqual(decode_diagnostics["decode_applied_calls"], 1)

    def test_query_scope_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown query_scope"):
            make_attention_mask_hook(
                [1], [2], [3], 3, 6, query_scope="score_token"
            )


class SplitAndRankingTests(unittest.TestCase):
    def test_auto_grounding_eligibility_requires_two_latest_ok_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            rows = [
                {"example_id": "good", "frame": "first", "status": "ok"},
                {"example_id": "good", "frame": "last", "status": "ok"},
                {"example_id": "partial", "frame": "first", "status": "ok"},
                {"example_id": "retried", "frame": "first", "status": "ok"},
                {"example_id": "retried", "frame": "last", "status": "ok"},
                {"example_id": "retried", "frame": "last", "status": "invalid"},
            ]
            (run_dir / "grounding.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual(
                _formal_ids(run_dir, "auto_valid_grounding"), {"good"}
            )
            with self.assertRaises(FileNotFoundError):
                _formal_ids(run_dir)
            with self.assertRaisesRegex(ValueError, "eligibility_mode"):
                _formal_ids(run_dir, "not-a-mode")

    def test_video_hash_split_has_no_leakage(self) -> None:
        rows = [
            {
                "example_id": f"x{i}",
                "video_sha256": f"h{i // 2}",
                "subset": "s",
                "target_type": "object",
            }
            for i in range(10)
        ]
        split = grouped_stratified_split(rows)
        self.assertFalse(
            set(split["discovery_video_sha256"]) & set(split["evaluation_video_sha256"])
        )

    def test_consensus_validates_complete_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for source in range(3):
                path = Path(directory) / f"{source}.json"
                rows = [
                    {"layer": layer, "head": head, "score": 1}
                    for layer in range(2)
                    for head in range(2)
                ]
                if source == 1:
                    rows.reverse()
                path.write_text(
                    json.dumps(
                        {"num_layers": 2, "num_heads": 2, "rankings": {"mean": rows}}
                    )
                )
                paths.append(path)
            result = consensus_ranking(paths, expected_layers=2, expected_heads=2)
            self.assertEqual(len(result["ranking"]), 4)
            self.assertIn("fingerprint", result)
            filtered = consensus_ranking(
                paths, expected_layers=2, expected_heads=2, skip_early_layers=1
            )
            self.assertEqual(len(filtered["ranking"]), 2)
            self.assertEqual(filtered["skip_early_layers"], 1)
            self.assertTrue(all(row["layer"] >= 1 for row in filtered["ranking"]))

    def test_consensus_excludes_early_layers_before_borda_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            # Both source rankings put an excluded layer-0 head first.  The
            # eligible head with the best *post-filter* rank must still win.
            for source in range(2):
                path = Path(directory) / f"{source}.json"
                rows = [
                    {"layer": 0, "head": 0},
                    {"layer": 1, "head": 1},
                    {"layer": 1, "head": 0},
                    {"layer": 0, "head": 1},
                ]
                path.write_text(
                    json.dumps(
                        {"num_layers": 2, "num_heads": 2, "rankings": {"mean": rows}}
                    )
                )
                paths.append(path)
            result = consensus_ranking(
                paths, expected_layers=2, expected_heads=2, skip_early_layers=1
            )
            self.assertEqual(
                [(row["layer"], row["head"]) for row in result["ranking"]],
                [(1, 1), (1, 0)],
            )

    def test_in_domain_ranks_excess_and_skips_early_layers(self) -> None:
        raw = [[[0, 0], [0.1, 0.2], [0.3, 0.1]]]
        excess = [[[0, 0], [0.0, 0.1], [0.4, 0.2]]]
        result = aggregate_in_domain(
            [{"example_id": "x", "status": "ok", "raw_mass": raw[0], "excess_mass": excess[0]}],
            num_layers=3,
            num_heads=2,
            skip_layers=1,
        )
        self.assertEqual((result["ranking"][0]["layer"], result["ranking"][0]["head"]), (2, 0))
        self.assertNotIn(0, {row["layer"] for row in result["ranking"]})

    def test_cluster_bootstrap_and_holm(self) -> None:
        rows = [
            {"video_sha256": "a", "effect": 1.0},
            {"video_sha256": "a", "effect": 1.0},
            {"video_sha256": "b", "effect": 1.0},
        ]
        result = paired_cluster_bootstrap(rows, "effect", samples=50)
        self.assertEqual(result["ci95"], [1.0, 1.0])
        self.assertEqual(result["n_records"], 3)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["strata_cluster_counts"], {"__all__": 2})
        adjusted = holm({"a": 0.01, "b": 0.04})
        self.assertEqual(adjusted, {"a": 0.02, "b": 0.04})

    def test_cluster_inference_is_video_level_and_subset_stratified(self) -> None:
        unique = [
            {"video_sha256": "a", "subset": "s1", "effect": 1.0},
            {"video_sha256": "b", "subset": "s1", "effect": -0.5},
            {"video_sha256": "c", "subset": "s2", "effect": 0.25},
        ]
        duplicated = [unique[0], dict(unique[0]), *unique[1:]]
        self.assertEqual(
            paired_sign_flip_pvalue(unique, "effect", samples=200, seed=7),
            paired_sign_flip_pvalue(duplicated, "effect", samples=200, seed=7),
        )
        result = paired_cluster_bootstrap(duplicated, "effect", samples=50, seed=7)
        self.assertEqual(result["n_records"], 4)
        self.assertEqual(result["n_clusters"], 3)
        self.assertEqual(result["n_strata"], 2)
        self.assertEqual(result["strata_cluster_counts"], {"s1": 2, "s2": 1})

    def test_cluster_cannot_cross_subsets(self) -> None:
        rows = [
            {"video_sha256": "a", "subset": "s1", "effect": 1.0},
            {"video_sha256": "a", "subset": "s2", "effect": 1.0},
        ]
        with self.assertRaisesRegex(ValueError, "multiple subsets"):
            paired_cluster_bootstrap(rows, "effect", samples=10)

    def test_exact_mcnemar_uses_only_discordant_correctness_pairs(self) -> None:
        rows = (
            [{"base": False, "candidate": True}] * 5
            + [{"base": True, "candidate": False}]
            + [{"base": True, "candidate": True}] * 20
        )
        self.assertAlmostEqual(
            exact_mcnemar_pvalue(rows, "base", "candidate"),
            0.21875,
        )


class QueryScopeMetricsTests(unittest.TestCase):
    def test_dry_run_can_disable_expensive_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            ranking_path = run_dir / "ranking.json"
            ranking_path.write_text(
                json.dumps(
                    {
                        "ranking_source": "frozen_cross_domain_consensus",
                        "fingerprint": "ranking",
                        "ranking": [
                            {"layer": layer, "head": head}
                            for layer in range(2)
                            for head in range(2)
                        ],
                    }
                )
            )
            (run_dir / "eligible.jsonl").write_text(
                json.dumps({"example_id": "x", "video_sha256": "0" * 64}) + "\n"
            )
            (run_dir / "split.json").write_text(
                json.dumps({"discovery": [], "evaluation": ["x"], "fingerprint": "split"})
            )
            output = steer(
                {
                    "attention_eval": {
                        "output_dir": str(run_dir),
                        "ranking_path": str(ranking_path),
                        "model_path": str(run_dir / "missing-model"),
                        "grounding_run": str(run_dir / "sam3"),
                        "num_layers": 2,
                        "num_heads": 2,
                        "top_k": 1,
                        "swap_bias": 6,
                        "include_all_heads_control": False,
                        "run_sensitivity": False,
                        "run_duplicate_location_sensitivity": False,
                        "run_paired": False,
                        "steering_query_scope": "last_prompt",
                        "query_scope_sensitivity": [],
                    }
                },
                dry_run=True,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [row["condition"] for row in rows],
                [
                    "baseline",
                    "candidate_target",
                    "candidate_wrong",
                    "low_rank_target",
                ],
            )

    def test_dry_run_materializes_all_scope_control_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            ranking_path = run_dir / "ranking.json"
            ranking_path.write_text(
                json.dumps(
                    {
                        "ranking_source": "frozen_cross_domain_consensus",
                        "fingerprint": "ranking",
                        "ranking": [
                            {"layer": layer, "head": head}
                            for layer in range(2)
                            for head in range(2)
                        ],
                    }
                )
            )
            (run_dir / "eligible.jsonl").write_text(
                json.dumps({"example_id": "x", "video_sha256": "0" * 64}) + "\n"
            )
            (run_dir / "split.json").write_text(
                json.dumps(
                    {
                        "discovery": [],
                        "evaluation": ["x"],
                        "fingerprint": "split",
                    }
                )
            )
            config = {
                "attention_eval": {
                    "output_dir": str(run_dir),
                    "ranking_path": str(ranking_path),
                    "model_path": str(run_dir / "missing-model"),
                    "grounding_run": str(run_dir / "sam3"),
                    "num_layers": 2,
                    "num_heads": 2,
                    "top_k": 1,
                    "swap_bias": 6,
                    "top_k_sensitivity": [],
                    "bias_sensitivity": [],
                    "run_duplicate_location_sensitivity": False,
                    "run_paired": False,
                    "steering_query_scope": "all",
                    "query_scope_sensitivity": [
                        "all",
                        "prefill",
                        "last_prompt",
                        "decode",
                    ],
                    "query_scope_sensitivity_conditions": [
                        "candidate_target",
                        "candidate_wrong",
                        "low_rank_target",
                    ],
                }
            }
            output = steer(config, dry_run=True)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            by_condition = {row["condition"]: row for row in records}
            self.assertEqual(len(records), 17)
            for scope in ("all", "prefill", "last_prompt", "decode"):
                for condition in (
                    "candidate_target",
                    "candidate_wrong",
                    "low_rank_target",
                ):
                    name = f"query_scope_{scope}_{condition}"
                    self.assertEqual(by_condition[name]["query_scope"], scope)
            self.assertTrue(
                by_condition["query_scope_all_candidate_target"]["hook_diagnostics"][
                    "exact_primary_condition_reuse"
                ]
            )

    def test_metrics_reports_scope_effects_and_legacy_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            base = {
                "example_id": "x",
                "video_sha256": "video",
                "status": "ok",
            }
            records = [
                {
                    **base,
                    "condition": "baseline",
                    "signed_score": 0.0,
                    "hook_diagnostics": {"bbox_attention_mass": 0.1},
                },
                {
                    **base,
                    "condition": "candidate_target",
                    "signed_score": -0.1,
                    "hook_diagnostics": {"bbox_attention_mass": 0.8},
                },
                {
                    **base,
                    "condition": "candidate_wrong",
                    "signed_score": 0.0,
                    "hook_diagnostics": {},
                },
                {
                    **base,
                    "condition": "low_rank_target",
                    "signed_score": 0.0,
                    "hook_diagnostics": {},
                },
            ]
            scope_scores = {
                "all": -0.1,
                "prefill": -0.07,
                "last_prompt": -0.05,
                "decode": -0.02,
            }
            for scope, score in scope_scores.items():
                records.extend(
                    [
                        {
                            **base,
                            "condition": f"query_scope_{scope}_candidate_target",
                            "signed_score": score,
                            "hook_diagnostics": {"bbox_attention_mass": 0.8},
                        },
                        {
                            **base,
                            "condition": f"query_scope_{scope}_candidate_wrong",
                            "signed_score": 0.0,
                            "hook_diagnostics": {},
                        },
                        {
                            **base,
                            "condition": f"query_scope_{scope}_low_rank_target",
                            "signed_score": 0.0,
                            "hook_diagnostics": {},
                        },
                    ]
                )
            (run_dir / "steering.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )
            (run_dir / "eligible.jsonl").write_text(
                json.dumps({"example_id": "x", "subset": "subset"}) + "\n"
            )
            result = attention_metrics(
                run_dir,
                {
                    "attention_eval": {
                        "bootstrap_samples": 50,
                        "query_scope_sensitivity": list(scope_scores),
                    }
                },
            )
            scopes = result["query_scope_ablation"]["scopes"]
            self.assertEqual(scopes["last_prompt"]["estimands"]["target_shift"]["mean"], -0.05)
            self.assertAlmostEqual(
                scopes["decode"]["estimands"]["candidate_score_minus_all_scope"]["mean"],
                0.08,
            )
            self.assertEqual(scopes["all"]["bbox_mass_increase_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
