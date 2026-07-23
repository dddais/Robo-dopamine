"""Load frozen grounding records for attention-mask experiments.

The grounding pipeline stores detector outputs, endpoint frames, and a human
audit in separate JSONL files.  This module joins those files by ``example_id``
and verifies that every human label still refers to the exact selected boxes
that were reviewed.

Reward is intentionally absent from :class:`AttentionExample`.  Ground-truth
labels may be joined later by ``metrics.py`` but are never available to model
input construction or head discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from roborewardbench.dopamine_eval.audit import grounding_result_fingerprint


SELECTION_MODES = (
    "manual_correct_ready",
    "manual_correct",
    "auto_ready",
    "auto_detected",
)
SPLIT_SCHEMA_VERSION = 1
SPLIT_STRATEGIES = (
    "manual_audit_ready_holdout",
    "evaluation_only",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            rows.append(row)
    return rows


def _index_unique(rows: Iterable[Mapping[str, Any]], source: str | Path) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id:
            raise ValueError(f"{source} contains an empty example_id")
        if example_id in indexed:
            raise ValueError(f"{source} contains duplicate example_id {example_id!r}")
        indexed[example_id] = row
    return indexed


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_bbox(
    value: Any,
    image_size: tuple[int, int],
    *,
    example_id: str,
    role: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{example_id}: {role} selected bbox must contain four values")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box):
        raise ValueError(f"{example_id}: {role} selected bbox contains a non-finite value")
    x1, y1, x2, y2 = box
    width, height = image_size
    tolerance = 1e-3
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{example_id}: {role} selected bbox is empty: {box}")
    if (
        x1 < -tolerance
        or y1 < -tolerance
        or x2 > width + tolerance
        or y2 > height + tolerance
    ):
        raise ValueError(
            f"{example_id}: {role} selected bbox {box} exceeds image size {image_size}"
        )
    return (
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    )


def _validated_image_size(value: Any, *, example_id: str, role: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{example_id}: {role} image_size must be [width, height]")
    width, height = (int(value[0]), int(value[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"{example_id}: {role} has invalid image size {(width, height)}")
    return width, height


def _same_file(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(
        strict=False
    )


@dataclass(frozen=True)
class AttentionExample:
    """A label-free model input plus frozen endpoint target boxes."""

    example_id: str
    task: str
    subset: str
    before_path: Path
    after_path: Path
    before_bbox: tuple[float, float, float, float]
    after_bbox: tuple[float, float, float, float]
    before_image_size: tuple[int, int]
    after_image_size: tuple[int, int]
    grounding_fingerprint: str
    steering_ready: bool
    manual_label: str | None
    target_phrase: str

    def model_item(self, blank_goal_path: str | Path) -> dict[str, Any]:
        """Build the exact eight-image Robo-Dopamine input.

        RoboRewardBench has one camera.  Its before/after frame is repeated in
        the three GRM camera slots, matching ``run_benchmark.make_inference_item``.
        The reward label is not a field of this dataclass and therefore cannot
        leak into the returned object.
        """

        goal = str(Path(blank_goal_path).expanduser().resolve(strict=False))
        before = str(self.before_path)
        after = str(self.after_path)
        return {
            "id": self.example_id,
            "task": self.task,
            "image": [
                before,
                goal,
                before,
                before,
                before,
                after,
                after,
                after,
            ],
        }

    def canonical_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task": self.task,
            "subset": self.subset,
            "before_path": str(self.before_path),
            "after_path": str(self.after_path),
            "before_bbox": list(self.before_bbox),
            "after_bbox": list(self.after_bbox),
            "before_image_size": list(self.before_image_size),
            "after_image_size": list(self.after_image_size),
            "grounding_fingerprint": self.grounding_fingerprint,
            "steering_ready": self.steering_ready,
            "manual_label": self.manual_label,
            "target_phrase": self.target_phrase,
        }


def examples_fingerprint(examples: Sequence[AttentionExample]) -> str:
    payload = [example.canonical_record() for example in sorted(examples, key=lambda row: row.example_id)]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selection_accepts(
    result: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
    mode: str,
) -> bool:
    if mode == "manual_correct_ready":
        return bool(
            audit is not None
            and audit.get("manual_label") == "correct"
            and result.get("steering_ready")
        )
    if mode == "manual_correct":
        return bool(audit is not None and audit.get("manual_label") == "correct")
    if mode == "auto_ready":
        return bool(result.get("steering_ready"))
    if mode == "auto_detected":
        before = result.get("before") or {}
        after = result.get("after") or {}
        return bool(
            isinstance(before.get("selected"), Mapping)
            and isinstance(after.get("selected"), Mapping)
        )
    raise ValueError(f"Unknown selection mode {mode!r}; expected one of {SELECTION_MODES}")


def load_attention_examples(
    grounding_dir: str | Path,
    *,
    selection_mode: str = "manual_correct_ready",
    example_ids: Iterable[str] | None = None,
    require_images: bool = True,
) -> list[AttentionExample]:
    """Load and validate a grounding subset.

    ``manual_correct*`` modes require exact fingerprint agreement between
    ``manual_audit.jsonl`` and the current ``grounding_results.jsonl``.
    ``auto_ready`` remains useful for exploratory work but does not imply that
    the selected object is correct.
    """

    if selection_mode not in SELECTION_MODES:
        raise ValueError(
            f"Unknown selection mode {selection_mode!r}; expected one of {SELECTION_MODES}"
        )
    root = Path(grounding_dir).expanduser().resolve()
    grounding_path = root / "grounding_results.jsonl"
    frame_path = root / "frame_manifest.jsonl"
    audit_path = root / "manual_audit.jsonl"
    if not grounding_path.is_file():
        raise FileNotFoundError(f"Grounding results not found: {grounding_path}")
    if not frame_path.is_file():
        raise FileNotFoundError(f"Frame manifest not found: {frame_path}")
    if selection_mode.startswith("manual_") and not audit_path.is_file():
        raise FileNotFoundError(f"Manual audit not found: {audit_path}")

    results = list(_index_unique(_read_jsonl(grounding_path), grounding_path).values())
    frames = _index_unique(_read_jsonl(frame_path), frame_path)
    audits = (
        _index_unique(_read_jsonl(audit_path), audit_path)
        if audit_path.is_file()
        else {}
    )
    requested = None if example_ids is None else {str(value) for value in example_ids}
    selected: list[AttentionExample] = []

    for result in results:
        example_id = str(result.get("example_id", ""))
        if not example_id:
            raise ValueError(f"{grounding_path} contains an empty example_id")
        if requested is not None and example_id not in requested:
            continue
        audit = audits.get(example_id)
        if not _selection_accepts(result, audit, selection_mode):
            continue

        fingerprint = grounding_result_fingerprint(result)
        if selection_mode.startswith("manual_") and audit is not None:
            frozen = str(audit.get("grounding_fingerprint", ""))
            if not frozen:
                raise ValueError(f"{example_id}: audited record has no grounding_fingerprint")
            if frozen != fingerprint:
                raise ValueError(
                    f"{example_id}: grounding changed after manual review; re-audit before use"
                )

        frame = frames.get(example_id)
        if frame is None:
            raise ValueError(f"{example_id}: missing from frame_manifest.jsonl")
        before = result.get("before") or {}
        after = result.get("after") or {}
        before_selected = before.get("selected")
        after_selected = after.get("selected")
        if not isinstance(before_selected, Mapping) or not isinstance(after_selected, Mapping):
            raise ValueError(f"{example_id}: both endpoint selected boxes are required")

        before_path = Path(str(before.get("image_path", ""))).expanduser().resolve(strict=False)
        after_path = Path(str(after.get("image_path", ""))).expanduser().resolve(strict=False)
        frame_before = (frame.get("before") or {}).get("image_path")
        frame_after = (frame.get("after") or {}).get("image_path")
        if not frame_before or not _same_file(before_path, str(frame_before)):
            raise ValueError(f"{example_id}: before frame disagrees with frame manifest")
        if not frame_after or not _same_file(after_path, str(frame_after)):
            raise ValueError(f"{example_id}: after frame disagrees with frame manifest")
        if require_images:
            if not before_path.is_file():
                raise FileNotFoundError(f"{example_id}: before image not found: {before_path}")
            if not after_path.is_file():
                raise FileNotFoundError(f"{example_id}: after image not found: {after_path}")

        before_size = _validated_image_size(
            before.get("image_size"), example_id=example_id, role="before"
        )
        after_size = _validated_image_size(
            after.get("image_size"), example_id=example_id, role="after"
        )
        selected.append(
            AttentionExample(
                example_id=example_id,
                task=str(result.get("task", "")),
                subset=str(result.get("subset", "unknown")),
                before_path=before_path,
                after_path=after_path,
                before_bbox=_validated_bbox(
                    before_selected.get("bbox"),
                    before_size,
                    example_id=example_id,
                    role="before",
                ),
                after_bbox=_validated_bbox(
                    after_selected.get("bbox"),
                    after_size,
                    example_id=example_id,
                    role="after",
                ),
                before_image_size=before_size,
                after_image_size=after_size,
                grounding_fingerprint=fingerprint,
                steering_ready=bool(result.get("steering_ready")),
                manual_label=(
                    str(audit.get("manual_label"))
                    if selection_mode.startswith("manual_") and audit is not None
                    else None
                ),
                target_phrase=str((result.get("selected_parse") or {}).get("target_phrase", "")),
            )
        )

    if requested is not None:
        found = {example.example_id for example in selected}
        missing = requested - found
        if missing:
            raise ValueError(
                f"{len(missing)} requested ids did not satisfy selection_mode={selection_mode}: "
                + ", ".join(sorted(missing)[:10])
            )
    selected.sort(key=lambda row: (row.subset, row.example_id))
    return selected


def load_split_partition(
    split_manifest: str | Path,
    partition: str,
) -> tuple[list[str], dict[str, Any]]:
    path = Path(split_manifest).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported split schema in {path}")
    if partition not in {"discovery", "evaluation"}:
        raise ValueError("partition must be 'discovery' or 'evaluation'")
    block = data.get(partition)
    if not isinstance(block, dict) or not isinstance(block.get("ids"), list):
        raise ValueError(f"{path} has no valid {partition} block")
    ids = [str(value) for value in block["ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate {partition} ids")
    if int(block.get("count", -1)) != len(ids):
        raise ValueError(
            f"{path} {partition}.count={block.get('count')} disagrees with {len(ids)} ids"
        )
    other_name = "evaluation" if partition == "discovery" else "discovery"
    other_ids = {str(value) for value in (data.get(other_name) or {}).get("ids", [])}
    overlap = set(ids) & other_ids
    if overlap:
        raise ValueError(f"Discovery/evaluation overlap in {path}: {sorted(overlap)[:10]}")
    return ids, data


def build_audit_holdout_split(
    grounding_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Use audited non-ready boxes for discovery and audited ready boxes for evaluation.

    Manual review established that both groups contain the intended object.
    The automatic quality gate then provides a natural disjoint holdout:
    correct/non-ready examples discover heads, while correct/ready examples are
    reserved for the causal evaluation.
    """

    root = Path(grounding_dir).expanduser().resolve()
    all_correct = load_attention_examples(root, selection_mode="manual_correct")
    discovery = [example for example in all_correct if not example.steering_ready]
    evaluation = [example for example in all_correct if example.steering_ready]
    if not discovery:
        raise ValueError("No manual-correct/non-ready examples are available for discovery")
    if not evaluation:
        raise ValueError("No manual-correct/ready examples are available for evaluation")
    discovery_ids = {example.example_id for example in discovery}
    evaluation_ids = {example.example_id for example in evaluation}
    if discovery_ids & evaluation_ids:
        raise AssertionError("Internal error: discovery and evaluation ids overlap")

    grounding_path = root / "grounding_results.jsonl"
    audit_path = root / "manual_audit.jsonl"
    frame_path = root / "frame_manifest.jsonl"
    manifest: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "manual_audit_ready_holdout",
        "interpretation_boundary": (
            "Head discovery uses manually correct but automatic-non-ready records; "
            "evaluation uses manually correct and automatic-ready records. This is "
            "a held-out experiment within a purposefully audited test subset, not "
            "an official full RoboRewardBench test evaluation."
        ),
        "source": {
            "grounding_dir": str(root),
            "grounding_results_sha256": sha256_file(grounding_path),
            "manual_audit_sha256": sha256_file(audit_path),
            "frame_manifest_sha256": sha256_file(frame_path),
        },
        "discovery": {
            "selection_mode": "manual_correct",
            "count": len(discovery),
            "ids": sorted(discovery_ids),
            "dataset_fingerprint": examples_fingerprint(discovery),
        },
        "evaluation": {
            "selection_mode": "manual_correct_ready",
            "count": len(evaluation),
            "ids": sorted(evaluation_ids),
            "dataset_fingerprint": examples_fingerprint(evaluation),
        },
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_evaluation_only_split(
    grounding_dir: str | Path,
    output: str | Path,
    *,
    selection_mode: str = "auto_detected",
) -> dict[str, Any]:
    """Freeze an automatically selected population for external-head evaluation.

    This split deliberately has no discovery partition. It is intended for
    transfer experiments whose head set was frozen outside RoboRewardBench.
    Automatic selection is not evidence that a detected box contains the
    correct instruction target; the manifest records that interpretation
    boundary explicitly.
    """

    if selection_mode not in SELECTION_MODES:
        raise ValueError(
            f"Unknown selection mode {selection_mode!r}; expected one of {SELECTION_MODES}"
        )
    root = Path(grounding_dir).expanduser().resolve()
    evaluation = load_attention_examples(root, selection_mode=selection_mode)
    if not evaluation:
        raise ValueError(
            f"No examples satisfy evaluation selection_mode={selection_mode}"
        )
    evaluation_ids = {example.example_id for example in evaluation}
    source: dict[str, Any] = {
        "grounding_dir": str(root),
        "grounding_results_sha256": sha256_file(root / "grounding_results.jsonl"),
        "frame_manifest_sha256": sha256_file(root / "frame_manifest.jsonl"),
    }
    audit_path = root / "manual_audit.jsonl"
    if selection_mode.startswith("manual_") and audit_path.is_file():
        source["manual_audit_sha256"] = sha256_file(audit_path)
    automatic = selection_mode.startswith("auto_")
    manifest: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "evaluation_only",
        "interpretation_boundary": (
            f"Evaluation uses selection_mode={selection_mode}. "
            + (
                "Target identity is not human-confirmed. "
                if automatic
                else "Manual labels and grounding fingerprints are required. "
            )
            + "The frozen head set must come from an external dataset; this split "
            "cannot support in-split head discovery."
        ),
        "source": source,
        "discovery": {
            "selection_mode": None,
            "count": 0,
            "ids": [],
            "dataset_fingerprint": examples_fingerprint([]),
        },
        "evaluation": {
            "selection_mode": selection_mode,
            "count": len(evaluation),
            "ids": sorted(evaluation_ids),
            "dataset_fingerprint": examples_fingerprint(evaluation),
        },
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the audited, leakage-resistant attention experiment split."
    )
    parser.add_argument("--grounding-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strategy",
        default="manual_audit_ready_holdout",
        choices=SPLIT_STRATEGIES,
    )
    parser.add_argument(
        "--selection-mode",
        default="auto_detected",
        choices=SELECTION_MODES,
        help="Used only by --strategy evaluation_only.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.strategy == "manual_audit_ready_holdout":
        manifest = build_audit_holdout_split(args.grounding_dir, args.output)
    else:
        manifest = build_evaluation_only_split(
            args.grounding_dir,
            args.output,
            selection_mode=args.selection_mode,
        )
    print(
        f"Wrote {args.output}: discovery={manifest['discovery']['count']}, "
        f"evaluation={manifest['evaluation']['count']}"
    )


if __name__ == "__main__":
    main()
