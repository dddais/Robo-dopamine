#!/usr/bin/env python3
"""Audit selected-head cardinality, weights, and hook exposure in steering JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def latest_rows(path: Path) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[(str(row["condition"]), str(row["example_id"]))] = row
    return list(latest.values())


def diagnostic_heads(row: dict) -> list[tuple[int, int, float]]:
    result = []
    for layer, values in row.get("hook_diagnostics", {}).get("per_layer", {}).items():
        for head, weight in values.get("head_bias_weights", {}).items():
            result.append((int(layer), int(head), float(weight)))
    return sorted(result)


def selected_heads(row: dict) -> list[tuple[int, int, float]]:
    return sorted(
        (int(head["layer"]), int(head["head"]), float(head.get("steering_weight", 1.0)))
        for head in row.get("heads", [])
    )


def audit(path: Path) -> dict:
    rows = latest_rows(path)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition"])].append(row)
    conditions = {}
    for condition, values in sorted(groups.items()):
        if condition == "baseline":
            conditions[condition] = {
                "n": len(values),
                "status_ok_all": all(row.get("status") == "ok" for row in values),
                "hook_inactive_all": all(not row.get("hook_diagnostics", {}).get("hook_active", False) for row in values),
            }
            continue
        match = re.search(r"_k(\d+)$", condition)
        expected_k = int(match.group(1)) if match else None
        head_specs = [selected_heads(row) for row in values]
        weight_counts: dict[str, int] = defaultdict(int)
        for _, _, weight in head_specs[0]:
            weight_counts[f"{weight:g}"] += 1
        per_layer_values = [entry for row in values for entry in row.get("hook_diagnostics", {}).get("per_layer", {}).values()]
        conditions[condition] = {
            "n": len(values),
            "expected_k": expected_k,
            "head_count_values": sorted({len(spec) for spec in head_specs}),
            "unique_head_count_values": sorted({len({(layer, head) for layer, head, _ in spec}) for spec in head_specs}),
            "head_spec_consistent_all_rows": all(spec == head_specs[0] for spec in head_specs),
            "nonzero_weight_all": all(weight > 0 for spec in head_specs for _, _, weight in spec),
            "weight_counts": dict(sorted(weight_counts.items())),
            "hook_active_all": all(row.get("hook_diagnostics", {}).get("hook_active", False) for row in values),
            "diagnostic_weights_match_selected_all": all(diagnostic_heads(row) == selected_heads(row) for row in values),
            "layer_applied_calls_min": min(int(item.get("applied_calls", 0)) for item in per_layer_values),
            "prefill_applied_calls_min": min(int(item.get("prefill_applied_calls", 0)) for item in per_layer_values),
            "decode_applied_calls_min": min(int(item.get("decode_applied_calls", 0)) for item in per_layer_values),
            "status_ok_all": all(row.get("status") == "ok" for row in values),
        }
    return {"path": str(path.resolve()), "row_count_latest": len(rows), "conditions": conditions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {args.output}")
    report = {"schema_version": "1.0.0", "audits": [audit(path) for path in args.input]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
