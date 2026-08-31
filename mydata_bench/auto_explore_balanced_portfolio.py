#!/usr/bin/env python3
"""Build a label-free-at-inference portfolio of complementary causal heads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config
from .io import object_fingerprint, write_json


def _specialists(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for field in ("success_ranking", "fail_ranking"):
        for source in data[field]:
            if not bool(source.get("branch_safe", False)):
                continue
            row = dict(source)
            sign = -1.0 if float(row["steering_multiplier"]) < 0 else 1.0
            key = (int(row["layer"]), int(row["head"]), sign)
            if key in seen:
                continue
            seen.add(key)
            row["validated_direction_sign"] = sign
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    section = config.get("balanced_portfolio")
    if not isinstance(section, dict):
        raise ValueError("Expected balanced_portfolio configuration section")

    source_path = Path(section["adaptive_ranking_path"]).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("inference_uses_labels") is not False:
        raise ValueError("Source artifact must explicitly exclude inference labels")
    candidates = _specialists(source)
    success = [
        row
        for row in candidates
        if float(row["success_target_correct_margin_delta"]) > 0
        and float(row["fail_target_correct_margin_delta"]) < 0
    ]
    fail = [
        row
        for row in candidates
        if float(row["success_target_correct_margin_delta"]) < 0
        and float(row["fail_target_correct_margin_delta"]) > 0
    ]

    pairs = []
    for success_row in success:
        for fail_row in fail:
            success_key = (int(success_row["layer"]), int(success_row["head"]))
            fail_key = (int(fail_row["layer"]), int(fail_row["head"]))
            if success_key == fail_key:
                continue
            s1 = float(success_row["success_target_correct_margin_delta"])
            f1 = float(success_row["fail_target_correct_margin_delta"])
            s2 = float(fail_row["success_target_correct_margin_delta"])
            f2 = float(fail_row["fail_target_correct_margin_delta"])
            denominator = f2 - s2
            if denominator <= 0:
                continue
            ratio = (s1 - f1) / denominator
            if ratio <= 0:
                continue
            weight_success = 1.0 / max(1.0, ratio)
            weight_fail = ratio / max(1.0, ratio)
            success_target = weight_success * s1 + weight_fail * s2
            fail_target = weight_success * f1 + weight_fail * f2
            success_spatial = (
                weight_success
                * float(success_row["success_spatial_correct_margin_delta"])
                + weight_fail
                * float(fail_row["success_spatial_correct_margin_delta"])
            )
            fail_spatial = (
                weight_success
                * float(success_row["fail_spatial_correct_margin_delta"])
                + weight_fail
                * float(fail_row["fail_spatial_correct_margin_delta"])
            )
            if min(success_target, fail_target, success_spatial, fail_spatial) <= 0:
                continue
            score = min(success_target, fail_target) + 0.5 * min(
                success_spatial, fail_spatial
            )
            pairs.append(
                {
                    "score": score,
                    "success_target_effect": success_target,
                    "fail_target_effect": fail_target,
                    "success_spatial_effect": success_spatial,
                    "fail_spatial_effect": fail_spatial,
                    "success_weight": weight_success,
                    "fail_weight": weight_fail,
                    "success_row": success_row,
                    "fail_row": fail_row,
                }
            )
    pairs.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["success_row"]["layer"]),
            int(item["success_row"]["head"]),
            int(item["fail_row"]["layer"]),
            int(item["fail_row"]["head"]),
        )
    )
    pair_count = int(section.get("pair_count", 4))
    selected_pairs = []
    used_heads = set()
    for pair in pairs:
        keys = {
            (int(pair["success_row"]["layer"]), int(pair["success_row"]["head"])),
            (int(pair["fail_row"]["layer"]), int(pair["fail_row"]["head"])),
        }
        if used_heads & keys:
            continue
        selected_pairs.append(pair)
        used_heads |= keys
        if len(selected_pairs) == pair_count:
            break
    if len(selected_pairs) != pair_count:
        raise RuntimeError(f"Only found {len(selected_pairs)}/{pair_count} disjoint pairs")

    maximum = max(float(pair["score"]) for pair in selected_pairs)
    selected = []
    pair_summaries = []
    for pair_index, pair in enumerate(selected_pairs, start=1):
        pair_scale = math.sqrt(float(pair["score"]) / maximum)
        pair_summaries.append(
            {
                key: value
                for key, value in pair.items()
                if key not in {"success_row", "fail_row"}
            }
            | {
                "pair_id": pair_index,
                "pair_scale": pair_scale,
                "success_head": [
                    int(pair["success_row"]["layer"]),
                    int(pair["success_row"]["head"]),
                ],
                "fail_head": [
                    int(pair["fail_row"]["layer"]),
                    int(pair["fail_row"]["head"]),
                ],
            }
        )
        for role, weight_field in (
            ("success", "success_weight"),
            ("fail", "fail_weight"),
        ):
            row = dict(pair[f"{role}_row"])
            sign = float(row["validated_direction_sign"])
            row.update(
                {
                    "portfolio_pair_id": pair_index,
                    "portfolio_role": role,
                    "portfolio_pair_score": float(pair["score"]),
                    "portfolio_pair_scale": pair_scale,
                    "portfolio_balance_weight": float(pair[weight_field]),
                    "steering_multiplier": sign
                    * pair_scale
                    * float(pair[weight_field]),
                }
            )
            selected.append(row)

    fallback_path = Path(section["fallback_ranking_path"]).resolve()
    fallback_data = json.loads(fallback_path.read_text(encoding="utf-8"))
    fallback_rows = [
        dict(row)
        for row in fallback_data["ranking"]
        if (int(row["layer"]), int(row["head"])) not in used_heads
    ]
    padding_top_k = int(section.get("padding_top_k", 64))
    need = padding_top_k - len(selected)
    if need < 0 or len(fallback_rows) < need + padding_top_k:
        raise RuntimeError("Fallback ranking cannot support portfolio padding/control")
    start = len(fallback_rows) - padding_top_k - need
    padding_source = fallback_rows[start : start + need]
    padding_keys = {(int(row["layer"]), int(row["head"])) for row in padding_source}
    floor = float(section.get("padding_multiplier", 0.1))
    padding = [
        {
            **row,
            "safe_padding": True,
            "portfolio_padding": True,
            "steering_multiplier": floor,
        }
        for row in padding_source
    ]
    remaining = [
        row
        for row in fallback_rows
        if (int(row["layer"]), int(row["head"])) not in padding_keys
    ]
    ranking = [
        {**row, "rank": index}
        for index, row in enumerate([*selected, *padding, *remaining], start=1)
    ]
    if len(ranking) != len(fallback_data["ranking"]) or len(
        {(int(row["layer"]), int(row["head"])) for row in ranking}
    ) != len(ranking):
        raise RuntimeError("Portfolio ranking is not a complete unique head order")
    artifact = {
        "schema_version": "complementary-balanced-causal-portfolio-v1",
        "ranking_source": "development_only_complementary_success_fail_causal_pairs",
        "ranking_method_detail": (
            "four disjoint success-specialist/fail-specialist pairs; pair weights "
            "analytically equalize development success/fail target-margin effects; "
            "requires positive paired target and spatial effects; pair strength square-root scaled; "
            "inert padding to top64"
        ),
        "labels_model_facing": False,
        "inference_uses_labels": False,
        "source_adaptive_fingerprint": source["fingerprint"],
        "fallback_ranking_fingerprint": fallback_data["fingerprint"],
        "pair_summaries": pair_summaries,
        "selected_head_count": len(selected),
        "padding_head_count": len(padding),
        "ranking": ranking,
    }
    artifact["fingerprint"] = object_fingerprint(artifact)
    output = Path(section["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "balanced_causal_portfolio_ranking.json"
    write_json(path, artifact)
    print(path)


if __name__ == "__main__":
    main()
