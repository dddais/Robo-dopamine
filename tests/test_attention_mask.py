from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:  # Lightweight repository-test environments may omit torch.
    torch = None

from roborewardbench.attention_mask.dataset import (
    build_audit_holdout_split,
    build_evaluation_only_split,
    load_attention_examples,
    load_split_partition,
)
from roborewardbench.attention_mask.curve import build_curve_rows
from roborewardbench.attention_mask.io import initialize_manifest
from roborewardbench.attention_mask.masking import (
    IMAGE_LABELS,
    ROLE_LABELS,
    Head,
    ImageSpan,
    bbox_to_token_positions,
    make_attention_mask_hook,
    matched_wrong_position_set,
    target_position_set,
)
from roborewardbench.attention_mask.metrics import analyze_records
from roborewardbench.attention_mask.rank_heads import (
    aggregate_rankings,
    bbox_ranking_score,
)
from roborewardbench.attention_mask.run_experiment import (
    validate_ranking_linkage,
    validate_ranking_model,
)
from roborewardbench.attention_mask.visualize import aggregate_head_grid
from roborewardbench.dopamine_eval.audit import grounding_result_fingerprint


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def grounding_row(example_id: str, image: Path, *, steering_ready: bool) -> dict:
    endpoint = {
        "image_path": str(image),
        "image_size": [10, 10],
        "selected": {"bbox": [0.0, 0.0, 5.0, 10.0], "score": 0.8, "label": "block"},
    }
    return {
        "example_id": example_id,
        "task": "pick the block",
        "subset": "toy",
        "selected_parse": {"target_phrase": "block"},
        "grounding_queries": ["block"],
        "before": endpoint,
        "after": endpoint,
        "pair_consistency": {"consistent": True},
        "steering_ready": steering_ready,
        "status": "accepted_both",
        "visualization_file": "audit.jpg",
    }


class DatasetTests(unittest.TestCase):
    def test_manual_selection_checks_fingerprint_and_never_exposes_reward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"not decoded by this test")
            row = grounding_row("toy/example.mp4", image, steering_ready=True)
            write_jsonl(root / "grounding_results.jsonl", [row])
            write_jsonl(
                root / "frame_manifest.jsonl",
                [{
                    "example_id": row["example_id"],
                    "before": {"image_path": str(image)},
                    "after": {"image_path": str(image)},
                }],
            )
            write_jsonl(
                root / "manual_audit.jsonl",
                [{
                    "example_id": row["example_id"],
                    "manual_label": "correct",
                    "grounding_fingerprint": grounding_result_fingerprint(row),
                }],
            )
            examples = load_attention_examples(
                root, selection_mode="manual_correct_ready", require_images=False
            )
            self.assertEqual(len(examples), 1)
            item = examples[0].model_item(root / "blank.png")
            self.assertNotIn("reward", item)
            self.assertEqual(item["image"][2:5], [str(image.resolve())] * 3)
            self.assertEqual(item["image"][5:8], [str(image.resolve())] * 3)

            audit = json.loads((root / "manual_audit.jsonl").read_text().splitlines()[0])
            audit["grounding_fingerprint"] = "stale"
            write_jsonl(root / "manual_audit.jsonl", [audit])
            with self.assertRaisesRegex(ValueError, "changed after manual review"):
                load_attention_examples(
                    root, selection_mode="manual_correct_ready", require_images=False
                )

    def test_audit_holdout_is_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"x")
            rows = [
                grounding_row("toy/discovery.mp4", image, steering_ready=False),
                grounding_row("toy/evaluation.mp4", image, steering_ready=True),
            ]
            write_jsonl(root / "grounding_results.jsonl", rows)
            write_jsonl(
                root / "frame_manifest.jsonl",
                [
                    {
                        "example_id": row["example_id"],
                        "before": {"image_path": str(image)},
                        "after": {"image_path": str(image)},
                    }
                    for row in rows
                ],
            )
            write_jsonl(
                root / "manual_audit.jsonl",
                [
                    {
                        "example_id": row["example_id"],
                        "manual_label": "correct",
                        "grounding_fingerprint": grounding_result_fingerprint(row),
                    }
                    for row in rows
                ],
            )
            destination = root / "split.json"
            manifest = build_audit_holdout_split(root, destination)
            self.assertEqual(manifest["discovery"]["count"], 1)
            self.assertEqual(manifest["evaluation"]["count"], 1)
            discovery, _ = load_split_partition(destination, "discovery")
            evaluation, _ = load_split_partition(destination, "evaluation")
            self.assertFalse(set(discovery) & set(evaluation))

    def test_auto_detected_requires_both_endpoint_boxes_and_freezes_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"x")
            detected = grounding_row(
                "toy/detected.mp4", image, steering_ready=False
            )
            missing_after = json.loads(
                json.dumps(
                    grounding_row(
                        "toy/missing-after.mp4", image, steering_ready=False
                    )
                )
            )
            missing_after["after"]["selected"] = None
            rows = [detected, missing_after]
            write_jsonl(root / "grounding_results.jsonl", rows)
            write_jsonl(
                root / "frame_manifest.jsonl",
                [
                    {
                        "example_id": row["example_id"],
                        "before": {"image_path": str(image)},
                        "after": {"image_path": str(image)},
                    }
                    for row in rows
                ],
            )
            write_jsonl(
                root / "manual_audit.jsonl",
                [
                    {
                        "example_id": detected["example_id"],
                        "manual_label": "incorrect",
                        "grounding_fingerprint": "intentionally-stale",
                    }
                ],
            )
            examples = load_attention_examples(
                root, selection_mode="auto_detected", require_images=False
            )
            self.assertEqual(
                [example.example_id for example in examples],
                ["toy/detected.mp4"],
            )
            self.assertIsNone(examples[0].manual_label)
            destination = root / "auto-split.json"
            manifest = build_evaluation_only_split(
                root,
                destination,
                selection_mode="auto_detected",
            )
            self.assertEqual(manifest["strategy"], "evaluation_only")
            self.assertEqual(manifest["discovery"]["count"], 0)
            self.assertEqual(manifest["evaluation"]["count"], 1)
            evaluation, _ = load_split_partition(destination, "evaluation")
            self.assertEqual(evaluation, ["toy/detected.mp4"])


class MaskingTests(unittest.TestCase):
    @staticmethod
    def spans() -> list[ImageSpan]:
        return [
            ImageSpan(label=label, path=f"{label}.png", start=index * 4, end=index * 4 + 4, grid_thw=(1, 4, 4))
            for index, label in enumerate(IMAGE_LABELS)
        ]

    def test_bbox_is_copied_to_three_before_and_three_after_spans(self) -> None:
        positions = target_position_set(
            self.spans(),
            before_bbox=(0, 0, 5, 10),
            after_bbox=(5, 0, 10, 10),
            before_image_size=(10, 10),
            after_image_size=(10, 10),
            spatial_merge_size=2,
            target_role="both",
        )
        self.assertEqual(len(positions.per_span_target), 6)
        self.assertEqual(len(positions.target), 12)
        self.assertEqual(len(positions.other_image), 12)
        for label, span_positions in positions.per_span_target.items():
            self.assertEqual(len(span_positions), 2, label)

    def test_role_positions_partition_only_requested_endpoint_images(self) -> None:
        spans = self.spans()
        span_by_label = {span.label: span for span in spans}
        reference_positions = {
            position
            for label in IMAGE_LABELS[:2]
            for position in range(span_by_label[label].start, span_by_label[label].end)
        }
        for role in ("before", "after", "both", "after_high"):
            positions = target_position_set(
                spans,
                before_bbox=(0, 0, 5, 10),
                after_bbox=(5, 0, 10, 10),
                before_image_size=(10, 10),
                after_image_size=(10, 10),
                spatial_merge_size=2,
                target_role=role,
            )
            expected = {
                position
                for label in ROLE_LABELS[role]
                for position in range(
                    span_by_label[label].start,
                    span_by_label[label].end,
                )
            }
            self.assertEqual(set(positions.per_span_target), set(ROLE_LABELS[role]))
            self.assertFalse(set(positions.target) & set(positions.other_image))
            self.assertEqual(
                set(positions.target) | set(positions.other_image),
                expected,
            )
            self.assertFalse(
                (set(positions.target) | set(positions.other_image))
                & reference_positions
            )

    def test_span_protocol_rejects_duplicate_and_unexpected_labels(self) -> None:
        kwargs = {
            "before_bbox": (0, 0, 5, 10),
            "after_bbox": (5, 0, 10, 10),
            "before_image_size": (10, 10),
            "after_image_size": (10, 10),
            "spatial_merge_size": 2,
            "target_role": "both",
        }
        spans = self.spans()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            target_position_set(spans + [spans[-1]], **kwargs)
        unexpected = list(spans)
        unexpected[-1] = ImageSpan(
            label="unexpected",
            path=unexpected[-1].path,
            start=unexpected[-1].start,
            end=unexpected[-1].end,
            grid_thw=unexpected[-1].grid_thw,
        )
        with self.assertRaisesRegex(ValueError, "eight-image protocol"):
            target_position_set(unexpected, **kwargs)

    def test_small_bbox_uses_cell_intersection(self) -> None:
        span = self.spans()[0]
        positions = bbox_to_token_positions(
            span,
            (4.9, 4.9, 5.1, 5.1),
            (10, 10),
            2,
            method="intersection",
        )
        self.assertEqual(len(positions), 4)

    def test_wrong_region_is_disjoint_and_token_count_matched(self) -> None:
        spans = self.spans()
        target = target_position_set(
            spans,
            before_bbox=(0, 0, 5, 10),
            after_bbox=(0, 0, 5, 10),
            before_image_size=(10, 10),
            after_image_size=(10, 10),
            spatial_merge_size=2,
            target_role="both",
        )
        wrong = matched_wrong_position_set(
            spans, target, spatial_merge_size=2, seed=7
        )
        self.assertEqual(len(target.target), len(wrong.target))
        self.assertFalse(set(target.target) & set(wrong.target))
        for label in target.per_span_target:
            self.assertEqual(
                len(target.per_span_target[label]),
                len(wrong.per_span_target[label]),
            )
            self.assertFalse(
                set(target.per_span_target[label])
                & set(wrong.per_span_target[label])
            )

    @unittest.skipUnless(torch is not None, "torch is required for hook tensor checks")
    def test_hook_changes_only_selected_head_for_prefill_and_decode(self) -> None:
        hook = make_attention_mask_hook(
            head_indices=[1],
            suppress_positions=[2, 3],
            boost_positions=[4],
            num_query_heads=4,
            swap_bias=2.0,
            decode_only=False,
        )
        _, kwargs = hook(
            None,
            (),
            {"attention_mask": torch.zeros((1, 1, 3, 6), dtype=torch.float32)},
        )
        updated = kwargs["attention_mask"]
        self.assertEqual(tuple(updated.shape), (1, 4, 3, 6))
        self.assertTrue(torch.all(updated[:, 0] == 0))
        self.assertTrue(torch.all(updated[:, 2:] == 0))
        self.assertTrue(torch.all(updated[:, 1, :, 2:4] == -2))
        self.assertTrue(torch.all(updated[:, 1, :, 4] == 2))

        decode_only = make_attention_mask_hook(
            head_indices=[1],
            suppress_positions=[],
            boost_positions=[4],
            num_query_heads=4,
            swap_bias=2.0,
            decode_only=True,
        )
        self.assertIsNone(
            decode_only(
                None,
                (),
                {"attention_mask": torch.zeros((1, 1, 3, 6))},
            )
        )
        self.assertIsNotNone(
            decode_only(
                None,
                (),
                {"attention_mask": torch.zeros((1, 1, 1, 6))},
            )
        )

    @unittest.skipUnless(torch is not None, "torch is required for hook tensor checks")
    def test_hook_preserves_offsets_across_short_exact_and_extended_key_lengths(self) -> None:
        hook = make_attention_mask_hook(
            head_indices=[1, 3],
            suppress_positions=[2],
            boost_positions=[4],
            num_query_heads=4,
            swap_bias=1.5,
            decode_only=False,
        )
        for key_length in (3, 5, 8):
            original = torch.full((2, 1, 2, key_length), -7.0)
            _, kwargs = hook(None, (), {"attention_mask": original})
            updated = kwargs["attention_mask"]
            self.assertEqual(tuple(updated.shape), (2, 4, 2, key_length))
            self.assertTrue(torch.all(updated[:, 0] == -7.0))
            self.assertTrue(torch.all(updated[:, 2] == -7.0))
            self.assertTrue(torch.all(updated[:, [1, 3], :, 2] == -8.5))
            if key_length > 4:
                self.assertTrue(torch.all(updated[:, [1, 3], :, 4] == -5.5))
            if key_length > 5:
                self.assertTrue(torch.all(updated[:, [1, 3], :, 5:] == -7.0))

    @unittest.skipUnless(torch is not None, "torch is required for hook tensor checks")
    def test_hook_rejects_nonbroadcastable_head_dimension(self) -> None:
        hook = make_attention_mask_hook(
            head_indices=[1],
            suppress_positions=[2],
            boost_positions=[4],
            num_query_heads=4,
            swap_bias=1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "mask head dimension"):
            hook(
                None,
                (),
                {"attention_mask": torch.zeros((1, 2, 3, 5))},
            )

    @unittest.skipUnless(torch is not None, "torch is required for hook tensor checks")
    def test_aggregate_heatmap_uses_requested_layer_head_and_absolute_span(self) -> None:
        attentions = [
            torch.zeros((1, 2, 6, 12), dtype=torch.float32),
            torch.zeros((1, 2, 6, 12), dtype=torch.float32),
        ]
        span = ImageSpan(
            label="after_cam_high",
            path="frame.png",
            start=4,
            end=8,
            grid_thw=(1, 4, 4),
        )
        attentions[0][0, 1, 5, 4:8] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        attentions[1][0, 0, 5, 4:8] = torch.tensor([0.5, 0.6, 0.7, 0.8])
        grid, metrics = aggregate_head_grid(
            attentions,
            heads=[Head(0, 1), Head(1, 0)],
            query_position=5,
            span=span,
            spatial_merge_size=2,
            target_positions=[4, 6],
        )
        np.testing.assert_allclose(
            grid,
            np.array([[0.3, 0.4], [0.5, 0.6]]),
            rtol=1e-6,
        )
        self.assertAlmostEqual(metrics["span_mass"], 1.8)
        self.assertAlmostEqual(metrics["bbox_mass"], 0.8)
        self.assertEqual(metrics["target_token_count_in_span"], 2)


class RankingAndMetricsTests(unittest.TestCase):
    def test_external_fixed_ranking_skips_discovery_linkage_but_checks_dimensions(self) -> None:
        old_style = {
            "default_ranking": "mean",
            "rankings": {
                "mean": [
                    {"layer": 1, "head": 0, "score": 1.0},
                    {"layer": 0, "head": 1, "score": 0.5},
                ]
            },
            "skip_early_layers": 0,
            "num_layers": 2,
            "num_heads": 2,
        }
        validate_ranking_linkage(
            old_style,
            evaluation_ids=["toy/eval.mp4"],
            split_sha256="split",
            target_role="after",
            external_fixed_ranking=True,
            allow_incomplete_ranking=False,
        )
        with self.assertRaisesRegex(ValueError, "complete discovery"):
            validate_ranking_linkage(
                old_style,
                evaluation_ids=["toy/eval.mp4"],
                split_sha256="split",
                target_role="after",
                external_fixed_ranking=False,
                allow_incomplete_ranking=False,
            )
        validate_ranking_model(
            old_style,
            current_model_identity={},
            num_layers=2,
            num_heads=2,
            external_fixed_ranking=True,
        )
        with self.assertRaisesRegex(ValueError, "num_layers"):
            validate_ranking_model(
                old_style,
                current_model_identity={},
                num_layers=3,
                num_heads=2,
                external_fixed_ranking=True,
            )

    def test_aggregate_ranking_orders_largest_mass_first(self) -> None:
        rankings = aggregate_rankings(
            [
                np.array([[0.1, 0.9], [0.3, 0.2]]),
                np.array([[0.2, 0.8], [0.4, 0.1]]),
            ]
        )
        self.assertEqual(
            (rankings["mean"][0]["layer"], rankings["mean"][0]["head"]),
            (0, 1),
        )

    def test_excess_mass_removes_bbox_area_baseline(self) -> None:
        target = np.array([[0.4, 0.2]])
        image = np.array([[0.8, 0.4]])
        corrected = bbox_ranking_score(
            target,
            image,
            target_token_count=5,
            role_image_token_count=10,
            score_mode="excess_mass",
        )
        np.testing.assert_allclose(corrected, np.zeros_like(corrected))
        full_frame = bbox_ranking_score(
            image,
            image,
            target_token_count=10,
            role_image_token_count=10,
            score_mode="excess_mass",
        )
        np.testing.assert_allclose(full_frame, np.zeros_like(full_frame))

    def test_paired_metrics_use_common_examples(self) -> None:
        rows = []
        scores = {
            "a": {"baseline": 0.1, "candidate_target": 0.3, "candidate_wrong": 0.1, "low_rank_target": 0.15, "all_target": 0.2},
            "b": {"baseline": 0.2, "candidate_target": 0.4, "candidate_wrong": 0.2, "low_rank_target": 0.25, "all_target": 0.3},
        }
        for example_id, conditions in scores.items():
            for condition, score in conditions.items():
                rows.append({
                    "run_family_signature": "family",
                    "run_signature": "shard",
                    "example_id": example_id,
                    "subset": "s1" if example_id == "a" else "s2",
                    "condition": condition,
                    "top_k": 8 if condition not in {"baseline", "all_target"} else None,
                    "swap_bias": 2.0 if condition != "baseline" else 0.0,
                    "intervention": "boost_suppress",
                    "target_role": "both",
                    "score": score,
                    "status": "ok",
                })
        result = analyze_records(
            rows,
            bootstrap_samples=100,
            bootstrap_seed=0,
            shard_completions=[{
                "run_family_signature": "family",
                "run_signature": "shard",
                "shard_index": 0,
                "num_shards": 1,
                "complete_shard": True,
                "selected_ids": ["a", "b"],
                "result_record_count": len(rows),
                "_observed_result_record_count": len(rows),
                "_observed_run_signatures": ["shard"],
                "_observed_example_ids": ["a", "b"],
            }],
        )
        self.assertTrue(result["input_completeness"]["complete"])
        candidate_name = (
            "candidate_target|top_k=8|bias=2|boost_suppress|role=both"
        )
        shift = result["configurations"][candidate_name][
            "paired_score_shift_vs_baseline"
        ]
        self.assertAlmostEqual(shift["micro_mean"], 0.2)
        wrong_contrast = next(
            value
            for key, value in result["paired_control_contrasts"].items()
            if candidate_name in key and key.endswith("wrong_region")
        )
        self.assertEqual(wrong_contrast["paired_ids"], 2)
        self.assertAlmostEqual(
            wrong_contrast["signed_difference_of_shifts"]["micro_mean"], 0.2
        )

    def test_curve_is_bias_dose_response_with_paired_example_shifts(self) -> None:
        rows = []
        for example_id, baseline in (("a", 0.1), ("b", 0.3)):
            rows.append(
                {
                    "run_family_signature": "family",
                    "example_id": example_id,
                    "condition": "baseline",
                    "top_k": None,
                    "swap_bias": 0.0,
                    "intervention": "boost_suppress",
                    "target_role": "both",
                    "score": baseline,
                    "status": "ok",
                }
            )
            for bias, shift in ((0.0, 0.0), (2.0, -0.1)):
                rows.append(
                    {
                        "run_family_signature": "family",
                        "example_id": example_id,
                        "condition": "candidate_target",
                        "top_k": 8,
                        "swap_bias": bias,
                        "intervention": "boost_suppress",
                        "target_role": "both",
                        "score": baseline + shift,
                        "status": "ok",
                    }
                )
        curve, summary = build_curve_rows(
            rows,
            bootstrap_samples=100,
            bootstrap_seed=0,
        )
        self.assertEqual(summary["num_examples"], 2)
        by_bias = {row["swap_bias"]: row for row in curve}
        self.assertAlmostEqual(by_bias[0.0]["mean_paired_shift"], 0.0)
        self.assertAlmostEqual(by_bias[2.0]["mean_paired_shift"], -0.1)
        self.assertEqual(by_bias[2.0]["num_paired"], 2)

    def test_resume_manifest_rejects_signature_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            first = initialize_manifest(path, {"parameter": 1, "created_at": "first"})
            second = initialize_manifest(path, {"parameter": 1, "created_at": "later"})
            self.assertEqual(first["run_signature"], second["run_signature"])
            with self.assertRaisesRegex(ValueError, "does not match"):
                initialize_manifest(path, {"parameter": 2})


if __name__ == "__main__":
    unittest.main()
