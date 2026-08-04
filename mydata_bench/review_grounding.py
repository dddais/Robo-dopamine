"""Interactive local reviewer for GroundingDINO/SAM3 endpoint boxes.

Example:
    python rewardbench/review_grounding.py \
      --run-dir rewardbench/grounding/outputs/pairs_attention_dino_official/grounding_dino \
      --reviewer reviewer1

The program never uploads images or instructions.  It resumes from the review
JSONL if it already exists, and stores one append-only decision per displayed
instruction-conditioned input.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Support the project's preferred direct invocation:
# ``python rewardbench/review_grounding.py ...``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mydata_bench.io import append_jsonl, read_jsonl


KEYS = {
    ord("y"): "correct",
    ord("Y"): "correct",
    ord("n"): "incorrect",
    ord("N"): "incorrect",
    ord("u"): "uncertain",
    ord("U"): "uncertain",
}


def _load_templates(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "audit_template.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python rewardbench/run_grounding.py audit --run-dir {run_dir}` first."
        )
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No reviewable rows in {path}")
    return rows


def _load_targets(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir.parent / "targets.jsonl"
    if not path.is_file():
        return {}
    return {row["example_id"]: row for row in read_jsonl(path)}


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["example_id"]
        for row in read_jsonl(path)
        if row.get("first_label") in {"correct", "incorrect", "uncertain"}
        and row.get("last_label") in {"correct", "incorrect", "uncertain"}
    }


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read visualization: {path}")
    return image


def _fit(image: np.ndarray, height: int, width: int) -> np.ndarray:
    scale = min(height / image.shape[0], width / image.shape[1])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _text(canvas: np.ndarray, value: str, x: int, y: int, *, scale: float = 0.58) -> int:
    for line in textwrap.wrap(value, width=112, break_long_words=False) or [""]:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)
        y += round(31 * scale / 0.58)
    return y


def render(row: dict[str, Any], target: dict[str, Any] | None, position: int, total: int) -> np.ndarray:
    first = _read_image(row["endpoints"]["first"]["visualization_path"])
    last = _read_image(row["endpoints"]["last"]["visualization_path"])
    header_height, panel_height, panel_width = 235, 700, 900
    canvas = np.full((header_height + panel_height, panel_width * 2, 3), 255, dtype=np.uint8)
    target_phrase = (target or {}).get("target_phrase", "(target unavailable)")
    entity_type = (target or {}).get("entity_type", "unknown")
    y = 34
    y = _text(canvas, f"{position}/{total}    data #{row.get('data_number', position)}    visual #{row.get('visualization_number', '?')}", 24, y, scale=0.62)
    y = _text(canvas, f"Task: {row['instruction']}", 24, y + 5)
    _text(canvas, f"Target object: {target_phrase}  [{entity_type}]", 24, y + 5)
    canvas[header_height:, :panel_width] = _fit(first, panel_height, panel_width)
    canvas[header_height:, panel_width:] = _fit(last, panel_height, panel_width)
    cv2.putText(canvas, "FIRST endpoint", (25, header_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "LAST endpoint", (panel_width + 25, header_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Y: yes/correct   N: no/incorrect   U: uncertain   Q or Esc: save and quit",
        (24, canvas.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


def decision_record(row: dict[str, Any], reviewer_id: str, label: str, display_order: int) -> dict[str, Any]:
    """Map a single yes/no keypress to both endpoint labels.

    The compact interaction deliberately asks one validity question: does this
    instruction-specific target box correctly identify the object in the
    endpoint images?  More granular endpoint labels remain available later by
    editing this JSONL before adjudication, if needed.
    """
    return {
        "data_number": row.get("data_number", display_order),
        "visualization_number": row.get("visualization_number"),
        "display_order": display_order,
        "example_id": row["example_id"],
        "grounding_fingerprint": row["grounding_fingerprint"],
        "reviewer_id": reviewer_id,
        "first_label": label,
        "last_label": label,
        "error_categories": [],
        "reason": "",
    }


def review(run_dir: Path, reviewer_id: str) -> Path:
    if reviewer_id not in {"reviewer1", "reviewer2"}:
        raise ValueError("reviewer must be reviewer1 or reviewer2")
    templates = _load_templates(run_dir)
    targets = _load_targets(run_dir)
    output = run_dir / f"{reviewer_id}.jsonl"
    completed = _completed_ids(output)
    pending = [row for row in templates if row["example_id"] not in completed]
    if not pending:
        print(f"All {len(templates)} rows already reviewed: {output}")
        return output
    window = f"Grounding review — {reviewer_id}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1500, 900)
    try:
        for index, row in enumerate(pending, start=1):
            image = render(row, targets.get(row["example_id"]), index, len(pending))
            cv2.imshow(window, image)
            while True:
                key = cv2.waitKeyEx(0)
                if key in KEYS:
                    append_jsonl(output, decision_record(row, reviewer_id, KEYS[key], index))
                    break
                if key in {27, ord("q"), ord("Q")}:
                    print(f"Saved {index - 1} new decisions; resume with the same command: {output}")
                    return output
    finally:
        cv2.destroyAllWindows()
    print(f"Completed {len(pending)} new decisions: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive local grounding reviewer")
    parser.add_argument("--run-dir", required=True, help="Grounding backend output directory")
    parser.add_argument("--reviewer", default="reviewer1", choices=("reviewer1", "reviewer2"))
    args = parser.parse_args()
    review(Path(args.run_dir).resolve(), args.reviewer)


if __name__ == "__main__":
    main()
