from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rewardbench.grounding.audit import grounding_fingerprint, wilson_interval
from rewardbench.grounding.base import Grounder, mask_to_bbox
from rewardbench.grounding.parser import (
    build_queries,
    extract_json_object,
    heuristic_parse,
    normalize_target,
)
from rewardbench.io import object_fingerprint


class InstructionParserTests(unittest.TestCase):
    def test_slide_rotate_part_multi_target_and_robot_part(self) -> None:
        slide = heuristic_parse("Slide the red block to the left.")
        self.assertEqual(slide.target_phrase, "red block")
        self.assertEqual(slide.reference_object, "left")
        part = heuristic_parse("Rotate the pot's right handle clockwise.")
        self.assertEqual(part.entity_type, "object_part")
        self.assertEqual(part.parent_object, "pot")
        multi = heuristic_parse(
            "Insert the blue gear onto the right peg, followed by the red gear."
        )
        self.assertTrue(multi.multi_target)
        self.assertFalse(multi.formal_scope)
        gripper = heuristic_parse("Move the gripper to the green area.")
        self.assertEqual(gripper.entity_type, "robot_part")
        self.assertFalse(gripper.formal_scope)
        relational = heuristic_parse(
            "Slide the pot so its handle touches the ranch bottle."
        )
        self.assertEqual(relational.target_phrase, "pot")
        phrasal_turn = heuristic_parse("Turn on the left stovetop burner.")
        self.assertEqual(phrasal_turn.target_phrase, "left stovetop burner")
        self.assertEqual(phrasal_turn.head_noun, "burner")

    def test_json_constraints_and_part_queries(self) -> None:
        payload = extract_json_object(
            '```json\n{"target_phrase":"pot handle","head_noun":"handle"}\n```'
        )
        target = normalize_target(
            "x",
            {
                **payload,
                "entity_type": "object_part",
                "parent_object": "pot",
                "attributes": ["right"],
            },
            parser="test",
            parser_fingerprint="x",
        )
        self.assertIn("handle of pot", build_queries(target))
        with self.assertRaises(ValueError):
            normalize_target(
                "x", {"target_phrase": "", "head_noun": ""}, parser="x", parser_fingerprint="x"
            )


class DummyGrounder(Grounder):
    backend = "dummy"

    def candidates(self, image_path, queries):
        return [
            {"bbox": [1, 1, 8, 8], "score": 0.5, "query_priority": 1},
            {"bbox": [2, 2, 9, 9], "score": 0.5, "query_priority": 0},
            {"bbox": [-1, 0, 2, 2], "score": 0.9, "query_priority": 0},
        ]


class GrounderSchemaTests(unittest.TestCase):
    def test_backend_selection_bbox_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (10, 10)).save(image)
            grounder = DummyGrounder({"threshold": 0.1})
            selected = grounder.ground(str(image), ["block", "object"])
            self.assertEqual(selected["bbox"], [2.0, 2.0, 9.0, 9.0])
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[3:8, 2:9] = 1
            self.assertEqual(mask_to_bbox(mask), (2.0, 3.0, 9.0, 8.0))

    def test_fingerprint_invalidates_cache_and_wilson(self) -> None:
        first = DummyGrounder({"threshold": 0.1}).fingerprint
        second = DummyGrounder({"threshold": 0.2}).fingerprint
        self.assertNotEqual(first, second)
        rows = [
            {
                "example_id": "x",
                "frame": "first",
                "bbox": [0, 0, 1, 1],
                "score": 1,
                "backend": "dummy",
                "provenance": {},
            }
        ]
        old = grounding_fingerprint(rows)
        rows[0]["bbox"] = [1, 1, 2, 2]
        self.assertNotEqual(old, grounding_fingerprint(rows))
        low, high = wilson_interval(19, 23)
        self.assertLess(low, 19 / 23)
        self.assertGreater(high, 19 / 23)


if __name__ == "__main__":
    unittest.main()
