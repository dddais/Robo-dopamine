from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rewardbench.review_grounding_web import PAGE, ReviewStore


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _template_row(example_id: str, number: int) -> dict:
    return {
        "data_number": number,
        "visualization_number": number,
        "example_id": example_id,
        "grounding_fingerprint": f"fingerprint-{number}",
        "instruction": f"instruction {number}",
        "endpoints": {
            "first": {"visualization_path": f"first-{number}.jpg"},
            "last": {"visualization_path": f"last-{number}.jpg"},
        },
    }


class ReviewGroundingWebTest(unittest.TestCase):
    def test_previous_navigation_and_correction_are_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            rows = [_template_row("example-1", 1), _template_row("example-2", 2)]
            _write_jsonl(run_dir / "audit_template.jsonl", rows)
            store = ReviewStore(run_dir, "reviewer1")

            initial = store.state()
            self.assertEqual(initial["current"]["example_id"], "example-1")
            self.assertEqual(initial["position"], 1)
            self.assertIsNone(initial["current_label"])
            self.assertIsNone(initial["previous_example_id"])

            store.submit("example-1", "correct")
            current = store.state()
            self.assertEqual(current["current"]["example_id"], "example-2")
            self.assertEqual(current["previous_example_id"], "example-1")
            self.assertIsNone(current["next_example_id"])

            previous = store.state("example-1")
            self.assertEqual(previous["current"]["example_id"], "example-1")
            self.assertEqual(previous["current_label"], "correct")
            self.assertEqual(previous["next_example_id"], "example-2")

            by_position = store.state(position=2)
            self.assertEqual(by_position["current"]["example_id"], "example-2")
            self.assertEqual(by_position["position"], 2)

            store.submit("example-1", "incorrect")
            corrected = store.state("example-1")
            self.assertEqual(corrected["current_label"], "incorrect")
            output_lines = (run_dir / "reviewer1.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(output_lines), 2)

    def test_completed_state_can_return_to_last_example(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            _write_jsonl(
                run_dir / "audit_template.jsonl",
                [_template_row("example-1", 1), _template_row("example-2", 2)],
            )
            store = ReviewStore(run_dir, "reviewer1")
            store.submit("example-1", "correct")
            store.submit("example-2", "uncertain")

            state = store.state()
            self.assertTrue(state["done"])
            self.assertEqual(state["previous_example_id"], "example-2")

    def test_unknown_navigation_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            _write_jsonl(
                run_dir / "audit_template.jsonl",
                [_template_row("example-1", 1)],
            )
            store = ReviewStore(run_dir, "reviewer1")

            with self.assertRaisesRegex(KeyError, "Unknown review example"):
                store.state("missing")
            with self.assertRaisesRegex(ValueError, "position must be in"):
                store.state(position=0)

    def test_page_exposes_previous_and_next_navigation(self) -> None:
        self.assertIn("Previous (←)", PAGE)
        self.assertIn("Next (→)", PAGE)
        self.assertIn("ArrowLeft", PAGE)
        self.assertIn("ArrowRight", PAGE)
        self.assertIn("const nextAfterSubmit=nextExampleId", PAGE)
        self.assertIn("await load(nextAfterSubmit)", PAGE)
        self.assertIn('id="jump"', PAGE)
        self.assertIn("goToPosition()", PAGE)
        self.assertIn("e.key==='Enter'", PAGE)
        self.assertNotIn("Resume unfinished", PAGE)
