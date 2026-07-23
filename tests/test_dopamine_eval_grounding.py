from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from roborewardbench.dopamine_eval.audit import (
    grounding_result_fingerprint,
    merge_manual_annotations,
    summarize_manual_audit,
    wilson_interval,
)
from roborewardbench.dopamine_eval.grounding import (
    box_iou,
    pair_consistency,
    select_temporal_candidate_pair,
)
from roborewardbench.dopamine_eval.instruction_parser import (
    build_grounding_queries,
    compare_parses,
    extract_json_object,
    heuristic_parse,
    normalize_parse_payload,
)
from roborewardbench.dopamine_eval.pipeline import (
    DatasetExample,
    _detector_cache_matches,
    _parse_cache_matches,
)
from roborewardbench.dopamine_eval.vlm_audit import (
    _cache_matches,
    _selected_crop,
    normalize_vlm_annotation,
)


class InstructionParserTests(unittest.TestCase):
    def test_normalize_valid_payload(self) -> None:
        parsed = normalize_parse_payload(
            {
                "target_phrase": "small beige block",
                "target_head": "block",
                "attributes": ["small", "beige", "beige"],
                "target_type": "object",
                "parent_object": None,
                "reference_phrase": "left peg",
                "ambiguous": False,
            }
        )
        self.assertEqual(parsed.target_head, "block")
        self.assertEqual(parsed.attributes, ("small", "beige"))
        self.assertEqual(parsed.reference_phrase, "left peg")

    def test_empty_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_phrase"):
            normalize_parse_payload({"target_phrase": "", "target_head": "table"})

    def test_extract_wrapped_json(self) -> None:
        raw = 'prefix ```json\n{"target_phrase":"pot","target_head":"pot","attributes":[],"target_type":"object","parent_object":null,"reference_phrase":null,"ambiguous":false}\n```'
        payload = extract_json_object(raw)
        self.assertEqual(payload["target_phrase"], "pot")

    def test_heuristic_keeps_destination_out_of_target(self) -> None:
        parsed = heuristic_parse("Place the beige block onto the left peg of the tray.")
        self.assertEqual(parsed.target_phrase.lower(), "beige block")
        self.assertNotIn("peg", parsed.target_phrase.lower())

    def test_part_queries_include_parent(self) -> None:
        queries = build_grounding_queries(
            {
                "target_phrase": "pot's right handle",
                "target_head": "handle",
                "attributes": ["right"],
                "target_type": "object_part",
                "parent_object": "pot",
                "reference_phrase": None,
                "ambiguous": False,
            }
        )
        self.assertEqual(queries[0], "pot's right handle")
        self.assertIn("handle of pot", queries)
        self.assertNotIn("right handle", queries[-1] if len(queries) > 4 else "")

    def test_compare_parse_compatible(self) -> None:
        first = {
            "target_phrase": "rectangular peg board",
            "target_head": "peg board",
            "attributes": ["rectangular"],
            "target_type": "object",
            "parent_object": None,
            "reference_phrase": "table",
            "ambiguous": False,
        }
        second = {
            "target_phrase": "rectangular pegboard",
            "target_head": "pegboard",
            "attributes": ["rectangular"],
            "target_type": "object",
            "parent_object": None,
            "reference_phrase": "table",
            "ambiguous": False,
        }
        diagnostics = compare_parses(first, second)
        self.assertIn(diagnostics["agreement_level"], {"exact", "compatible"})


class GroundingUtilityTests(unittest.TestCase):
    def test_box_iou(self) -> None:
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_pair_consistency(self) -> None:
        before = {"bbox": [10, 10, 30, 30]}
        after = {"bbox": [11, 11, 31, 31]}
        result = pair_consistency(before, after, image_size=[100, 100])
        self.assertTrue(result["available"])
        self.assertTrue(result["consistent"])
        self.assertGreater(result["iou"], 0.8)

    def test_pair_unavailable(self) -> None:
        result = pair_consistency(None, {"bbox": [1, 1, 2, 2]}, image_size=[10, 10])
        self.assertFalse(result["available"])
        self.assertFalse(result["consistent"])

    def test_temporal_pair_can_replace_unstable_top1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            Image.new("RGB", (100, 100), (220, 210, 170)).save(image_path)
            before = [
                {"label": "object", "score": 0.50, "bbox": [60, 60, 90, 90]},
                {"label": "object", "score": 0.40, "bbox": [10, 10, 30, 30]},
            ]
            after = [
                {"label": "object", "score": 0.48, "bbox": [45, 45, 75, 75]},
                {"label": "object", "score": 0.39, "bbox": [11, 10, 31, 30]},
            ]
            selected = select_temporal_candidate_pair(
                before,
                after,
                before_image_path=image_path,
                after_image_path=image_path,
                image_size=[100, 100],
                attributes=[],
            )
            self.assertEqual(selected["before"]["bbox"], [10, 10, 30, 30])
            self.assertEqual(selected["after"]["bbox"], [11, 10, 31, 30])

    def test_single_pair_margin_is_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            Image.new("RGB", (20, 20), (100, 100, 100)).save(image_path)
            candidate = [{"label": "object", "score": 0.5, "bbox": [1, 1, 10, 10]}]
            selected = select_temporal_candidate_pair(
                candidate,
                candidate,
                before_image_path=image_path,
                after_image_path=image_path,
                image_size=[20, 20],
            )
            self.assertIsNone(selected["pair_margin"])
            json.dumps(selected, allow_nan=False)


class ManualAuditTests(unittest.TestCase):
    def test_grounding_fingerprint_changes_with_selected_box(self) -> None:
        first = {
            "example_id": "a",
            "task": "pick block",
            "before": {"selected": {"bbox": [0, 0, 1, 1], "score": 0.5}},
            "after": {"selected": {"bbox": [0, 0, 1, 1], "score": 0.5}},
        }
        second = json.loads(json.dumps(first))
        second["after"]["selected"]["bbox"] = [1, 1, 2, 2]
        self.assertNotEqual(
            grounding_result_fingerprint(first),
            grounding_result_fingerprint(second),
        )

    def test_merge_requires_complete_identity_match(self) -> None:
        sample = [{"example_id": "a", "steering_ready": True}]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            merge_manual_annotations(sample, [])

    def test_summary_keeps_ready_and_nonready_separate(self) -> None:
        sample = [
            {"example_id": "a", "steering_ready": True},
            {"example_id": "b", "steering_ready": True},
            {"example_id": "c", "steering_ready": False},
        ]
        annotations = [
            {"example_id": "a", "manual_label": "correct", "reason": "target"},
            {
                "example_id": "b",
                "manual_label": "incorrect",
                "failure_category": "wrong_object",
                "reason": "distractor",
            },
            {"example_id": "c", "manual_label": "correct", "reason": "target"},
        ]
        rows = merge_manual_annotations(sample, annotations)
        summary = summarize_manual_audit(rows, population_total=10, population_steering_ready=4)
        self.assertEqual(summary["overall"]["correct"], 2)
        self.assertEqual(summary["steering_ready"]["correct"], 1)
        self.assertEqual(summary["steering_ready"]["incorrect"], 1)
        self.assertEqual(summary["not_steering_ready"]["correct"], 1)
        self.assertAlmostEqual(summary["population_steering_ready_rate"], 0.4)

    def test_wilson_interval_contains_observed_fraction(self) -> None:
        interval = wilson_interval(19, 23)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertLess(interval[0], 19 / 23)
        self.assertGreater(interval[1], 19 / 23)


class CacheValidationTests(unittest.TestCase):
    def test_parse_cache_is_bound_to_task_model_and_signature(self) -> None:
        example = DatasetExample(
            index=0,
            example_id="subset/video.mp4",
            video_path=Path("/tmp/video.mp4"),
            task="pick the block",
            subset="subset",
        )
        row = {
            "example_id": example.example_id,
            "task": example.task,
            "model_path": "/models/qwen",
            "model_cache_signature": "abc",
        }
        self.assertTrue(
            _parse_cache_matches(
                row,
                example,
                model_path="/models/qwen",
                model_cache_signature="abc",
            )
        )
        self.assertFalse(
            _parse_cache_matches(
                row,
                example,
                model_path="/models/qwen",
                model_cache_signature="changed",
            )
        )

    def test_detector_cache_is_bound_to_query_frame_and_parameters(self) -> None:
        item = {
            "example_id": "subset/video.mp4",
            "frame_role": "before",
            "queries": ["red block", "block"],
        }
        provenance = {
            "model_path": "/models/dino",
            "model_cache_signature": "abc",
            "image_path": "/tmp/frame.png",
            "image_size_bytes": 123,
            "image_mtime_ns": 456,
            "detection_threshold": 0.15,
            "text_threshold": 0.15,
            "accept_threshold": 0.25,
            "top_k": 10,
        }
        row = {**item, **provenance}
        self.assertTrue(_detector_cache_matches(row, item, provenance))
        changed = {**provenance, "accept_threshold": 0.30}
        self.assertFalse(_detector_cache_matches(row, item, changed))


class VLMAuditUtilityTests(unittest.TestCase):
    def test_normalize_vlm_annotation_enforces_manual_contract(self) -> None:
        annotation = normalize_vlm_annotation(
            '{"manual_label":"incorrect","failure_category":"wrong_object",'
            '"reason":"The green box encloses the peg rather than the beige block.",'
            '"confidence":0.9}',
            review_basis="endpoint_visualization",
        )
        self.assertEqual(annotation["manual_label"], "incorrect")
        self.assertEqual(annotation["failure_category"], "wrong_object")
        self.assertEqual(annotation["review_basis"], "endpoint_visualization")

    def test_normalize_vlm_annotation_rejects_invalid_incorrect_category(self) -> None:
        with self.assertRaisesRegex(ValueError, "failure_category"):
            normalize_vlm_annotation(
                '{"manual_label":"incorrect","failure_category":"made_up",'
                '"reason":"wrong object","confidence":0.5}',
                review_basis="endpoint_visualization",
            )

    def test_selected_crop_keeps_small_box_visible(self) -> None:
        image = Image.new("RGB", (96, 96), "white")
        crop = _selected_crop(image, [40, 40, 45, 45])
        self.assertGreaterEqual(crop.width, 224)
        self.assertGreaterEqual(crop.height, 224)

    def test_vlm_cache_binds_box_model_and_prompt(self) -> None:
        row = {
            "complete": True,
            "grounding_fingerprint": "box-v1",
            "model_signature": "model-v1",
            "prompt_hash": "prompt-v1",
            "annotation": {"manual_label": "correct"},
        }
        self.assertTrue(
            _cache_matches(row, fingerprint="box-v1", signature="model-v1", prompt_hash="prompt-v1")
        )
        self.assertFalse(
            _cache_matches(row, fingerprint="box-v2", signature="model-v1", prompt_hash="prompt-v1")
        )
        self.assertFalse(
            _cache_matches(
                {**row, "error": "out of memory"},
                fingerprint="box-v1",
                signature="model-v1",
                prompt_hash="prompt-v1",
            )
        )


if __name__ == "__main__":
    unittest.main()
