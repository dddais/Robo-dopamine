from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from ..io import object_fingerprint, read_jsonl, write_json, write_jsonl

LABELS = {"correct", "incorrect", "uncertain"}


def latest_endpoint_rows(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        latest[(row["example_id"], row.get("frame"))] = row
    return list(latest.values())


def grounding_fingerprint(rows: list[dict]) -> str:
    selected = [
        {
            "example_id": row["example_id"],
            "frame": row["frame"],
            "bbox": row.get("bbox"),
            "score": row.get("score"),
            "backend": row["backend"],
            "provenance": row.get("provenance"),
        }
        for row in rows
    ]
    return object_fingerprint(selected)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def _visualization_order(run_dir: Path, rows: list[dict]) -> dict[str, int]:
    """Use the reviewer-visible lexical order of ``visualizations/<SHA>``.

    File managers present these directories sorted by name.  Never use raw
    directory iteration order here: it is filesystem-dependent and can differ
    from the order seen by the reviewer.
    """
    path = run_dir / "audit_visualization_order.json"
    available = {str(row["video_sha256"]) for row in rows}
    if path.exists():
        import json

        order = json.loads(path.read_text(encoding="utf-8"))
        if set(order) == available:
            return {digest: int(number) for digest, number in order.items()}
    root = run_dir / "visualizations"
    ordered = sorted(
        entry.name for entry in root.iterdir() if entry.is_dir() and entry.name in available
    )
    ordered.extend(sorted(available - set(ordered)))
    result = {digest: number for number, digest in enumerate(ordered, start=1)}
    write_json(path, result)
    return result


def _audit_visualization_path(run_dir: Path, row: dict) -> Path:
    side = str(row["example_id"]).rsplit("/", 1)[-1]
    return (
        run_dir
        / "visualizations"
        / str(row["video_sha256"])
        / "by_instruction"
        / side
        / f"{row['frame']}.jpg"
    )


def _render_audit_visualization(run_dir: Path, row: dict) -> Path:
    """Render instruction-specific boxes without overwriting paired inputs."""
    output = _audit_visualization_path(run_dir, row)
    image_path = row.get("provenance", {}).get("image_path")
    if not image_path:
        raise ValueError(f"Missing image path for audit visualization: {row['example_id']}")
    image = Image.open(image_path).convert("RGB")
    bbox = row.get("bbox")
    if bbox is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(bbox), outline="red", width=max(2, image.width // 300))
        draw.text((bbox[0], max(0, bbox[1] - 14)), row["example_id"].rsplit("/", 1)[-1], fill="red")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def create_template(run_dir: Path, rows: list[dict]) -> Path:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["example_id"], []).append(row)
    order = _visualization_order(run_dir, rows)
    templates = []
    ordered_samples = sorted(
        grouped.items(),
        key=lambda value: (
            order[str(value[1][0]["video_sha256"])],
            str(value[0]).rsplit("/", 1)[-1],
        ),
    )
    for data_number, (example_id, sample) in enumerate(ordered_samples, start=1):
        templates.append(
            {
                "data_number": data_number,
                "visualization_number": order[str(sample[0]["video_sha256"])],
                "example_id": example_id,
                "grounding_fingerprint": grounding_fingerprint(sample),
                "instruction": sample[0].get("provenance", {}).get("task"),
                "video_path": sample[0].get("provenance", {}).get("video_path"),
                "endpoints": {
                    row["frame"]: {
                        "image_path": row.get("provenance", {}).get("image_path"),
                        "bbox": row.get("bbox"),
                        "mask_path": row.get("mask_path"),
                        "score": row.get("score"),
                        "visualization_path": str(_render_audit_visualization(run_dir, row)),
                    }
                    for row in sample
                },
                "reviewer_id": "",
                "first_label": "",
                "last_label": "",
                "error_categories": [],
                "reason": "",
            }
        )
    path = run_dir / "audit_template.jsonl"
    write_jsonl(path, templates)
    return path


def create_reviewer_template(run_dir: Path, reviewer_id: str) -> dict[str, str | int]:
    """Materialize an ordered, blank reviewer JSONL from the audit template.

    This intentionally refuses to overwrite a review that may already contain
    human decisions.  The compact Markdown index is only a navigation aid; the
    JSONL is the authoritative file consumed by adjudication.
    """
    if reviewer_id not in {"reviewer1", "reviewer2"}:
        raise ValueError("reviewer_id must be reviewer1 or reviewer2")
    source = run_dir / "audit_template.jsonl"
    if not source.is_file():
        raise FileNotFoundError("Run `grounding audit` once to create audit_template.jsonl")
    output = run_dir / f"{reviewer_id}.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing review: {output}")
    rows = list(read_jsonl(source))
    for number, row in enumerate(rows, start=1):
        # Stable order is the audit-template JSONL / visualization navigation
        # order.  Keep it in the review record so reviewers can refer to an
        # unambiguous data number without changing the adjudication key.
        row["data_number"] = number
        row["reviewer_id"] = reviewer_id
        row["first_label"] = ""
        row["last_label"] = ""
        row["error_categories"] = []
        row["reason"] = ""
    write_jsonl(output, rows)
    index = run_dir / f"{reviewer_id}_index.md"
    lines = [
        f"# {reviewer_id} Grounding Audit Index",
        "",
        "填写 JSONL 中相同行的 `first_label`、`last_label`、`error_categories` 和 `reason`。",
        "可用标签：`correct`、`incorrect`、`uncertain`。",
        "",
        "| Data # | Visual # | example_id | instruction | first_label | last_label |",
        "|---:|---:|---|---|---|---|",
    ]
    for number, row in enumerate(rows, start=1):
        task = str(row.get("instruction", "")).replace("|", "\\|")
        lines.append(
            f"| {number} | {row.get('visualization_number', '')} | "
            f"`{row['example_id']}` | {task} |  |  |"
        )
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"review_path": str(output), "index_path": str(index), "count": len(rows)}


def adjudicate(run_dir: Path, rows: list[dict]) -> dict:
    rows = latest_endpoint_rows(rows)
    successful = {
        row["example_id"]
        for row in rows
        if row.get("frame") == "first" and row.get("status") == "ok"
    } & {
        row["example_id"]
        for row in rows
        if row.get("frame") == "last" and row.get("status") == "ok"
    }
    rows = [row for row in rows if row["example_id"] in successful]
    review_paths = [run_dir / "reviewer1.jsonl", run_dir / "reviewer2.jsonl"]
    if not all(path.exists() for path in review_paths):
        template = create_template(run_dir, rows)
        return {"status": "awaiting_reviews", "template": str(template)}
    reviews = [
        {row["example_id"]: row for row in read_jsonl(path)} for path in review_paths
    ]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["example_id"], []).append(row)
    decisions = []
    disagreements = []
    for example_id, sample in sorted(grouped.items()):
        expected = grounding_fingerprint(sample)
        pair = [review[example_id] for review in reviews if example_id in review]
        if len(pair) != 2:
            raise ValueError(f"Missing dual review for {example_id}")
        for review in pair:
            if review.get("grounding_fingerprint") != expected:
                raise ValueError(f"Stale grounding fingerprint for {example_id}")
            if review.get("first_label") not in LABELS or review.get("last_label") not in LABELS:
                raise ValueError(f"Invalid audit label for {example_id}")
        agree = all(
            pair[0][field] == pair[1][field] for field in ("first_label", "last_label")
        )
        if agree:
            final = {
                "first_label": pair[0]["first_label"],
                "last_label": pair[0]["last_label"],
                "source": "reviewer_agreement",
            }
        else:
            disagreements.append(example_id)
            final = {"first_label": "uncertain", "last_label": "uncertain", "source": "needs_adjudication"}
        decisions.append(
            {
                "example_id": example_id,
                "grounding_fingerprint": expected,
                **final,
                "error_categories": sorted(
                    {
                        category
                        for review in pair
                        for category in review.get("error_categories", [])
                    }
                ),
                "reviewer_reasons": [review.get("reason", "") for review in pair],
                "formal_eligible": final["first_label"] == final["last_label"] == "correct",
            }
        )
    adjudication_path = run_dir / "adjudication.jsonl"
    if disagreements and adjudication_path.exists():
        third = {row["example_id"]: row for row in read_jsonl(adjudication_path)}
        for row in decisions:
            if row["example_id"] not in disagreements:
                continue
            value = third.get(row["example_id"])
            if not value or value.get("grounding_fingerprint") != row["grounding_fingerprint"]:
                continue
            if value.get("first_label") not in LABELS or value.get("last_label") not in LABELS:
                continue
            row.update(
                first_label=value["first_label"],
                last_label=value["last_label"],
                source="adjudicator",
                formal_eligible=value["first_label"] == value["last_label"] == "correct",
                error_categories=sorted(
                    set(row.get("error_categories", []))
                    | set(value.get("error_categories", []))
                ),
                adjudicator_reason=value.get("reason", ""),
            )
    write_jsonl(run_dir / "audit_final.jsonl", decisions)
    resolved = [row for row in decisions if row["source"] != "needs_adjudication"]
    correct = sum(row["formal_eligible"] for row in resolved)
    summary = {
        "status": "complete" if len(resolved) == len(decisions) else "needs_adjudication",
        "total": len(decisions),
        "resolved": len(resolved),
        "formal_eligible": correct,
        "correct_rate": correct / len(resolved) if resolved else None,
        "wilson_ci95": wilson_interval(correct, len(resolved)),
        "unresolved": [row["example_id"] for row in decisions if row["source"] == "needs_adjudication"],
        "error_category_counts": dict(
            sorted(
                Counter(
                    category
                    for row in resolved
                    for category in row.get("error_categories", [])
                ).items()
            )
        ),
    }
    write_json(run_dir / "audit_summary.json", summary)
    return summary
