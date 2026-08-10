#!/usr/bin/env python3
"""Write a reproducible ranking-head overlap report for mydata_bench v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path


TOP_K_VALUES = (8, 32, 64)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ranking_file(experiment: Path) -> Path:
    candidates = (
        experiment / "consensus_ranking.json",
        experiment / "in_domain_ranking.json",
    )
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one ranking file in {experiment}, found {found}")
    return found[0]


def model_family(name: str) -> str:
    if "roboreward" in name:
        return "RoboReward-8B"
    if "qwen" in name:
        return "Qwen3-VL-8B"
    if "grm" in name:
        return "GRM"
    raise ValueError(f"Unknown model family for {name}")


def load_experiment(experiment: Path) -> dict:
    path = ranking_file(experiment)
    data = read_json(path)
    rows = data.get("ranking")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No ranking rows in {path}")
    ranking = [(int(row["layer"]), int(row["head"])) for row in rows]
    if len(ranking) != len(set(ranking)):
        raise ValueError(f"Duplicate heads in {path}")
    canonical = json.dumps(ranking, separators=(",", ":")).encode()
    discovery_ids = data.get("per_sample_example_ids")
    grounded_inputs = experiment / "grounded_ranking_inputs.jsonl"
    if isinstance(discovery_ids, list):
        discovery_samples = len(discovery_ids)
    elif isinstance(data.get("n_discovery_samples"), int):
        discovery_samples = int(data["n_discovery_samples"])
    elif grounded_inputs.is_file():
        with grounded_inputs.open(encoding="utf-8") as handle:
            discovery_samples = sum(bool(line.strip()) for line in handle)
    else:
        discovery_samples = None
    return {
        "experiment": experiment.name,
        "model_family": model_family(experiment.name),
        "ranking_path": str(path.resolve()),
        "ranking": ranking,
        "eligible_heads": len(ranking),
        "num_layers": int(data["num_layers"]),
        "num_heads": int(data["num_heads"]),
        "skip_early_layers": int(data.get("skip_early_layers", 0)),
        "score_kind": data.get("ranking_score_kind", data.get("ranking_source", "unknown")),
        "discovery_samples": discovery_samples,
        "ranking_sequence_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def overlap(first: dict, second: dict, top_k: int) -> dict:
    first_top = set(first["ranking"][:top_k])
    second_top = set(second["ranking"][:top_k])
    shared = sorted(first_top & second_top)
    union = first_top | second_top
    return {
        "first": first["experiment"],
        "second": second["experiment"],
        "first_model": first["model_family"],
        "second_model": second["model_family"],
        "top_k": top_k,
        "intersection_count": len(shared),
        "intersection_fraction": len(shared) / top_k,
        "jaccard": len(shared) / len(union),
        "shared_heads": [list(head) for head in shared],
        "cross_model": first["model_family"] != second["model_family"],
    }


def markdown_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def head_text(heads: list[tuple[int, int]]) -> str:
    return ", ".join(f"L{layer}H{head}" for layer, head in heads)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def build_report(
    experiments_root: Path, experiment_names: list[str] | None = None
) -> dict:
    names = experiment_names or [
        "attention_06_roboreward_last_frame",
        "attention_07_roboreward_all_frames",
        "attention_08_qwen_last_frame",
        "attention_09_qwen_all_frames",
        "attention_10_grm_forward_after",
        "attention_11_grm_forward_before_after",
        "attention_12_grm_incremental_after",
        "attention_13_grm_incremental_before_after",
        "attention_14_roboreward_last_frame",
        "attention_15_roboreward_all_frames",
    ]
    experiments = [load_experiment(experiments_root / name) for name in names]
    universes = {frozenset(item["ranking"]) for item in experiments}
    dimensions = {
        (
            item["num_layers"],
            item["num_heads"],
            item["skip_early_layers"],
            item["eligible_heads"],
        )
        for item in experiments
    }
    if len(universes) != 1 or len(dimensions) != 1:
        raise ValueError("Rankings do not share the same eligible-head universe")
    pairwise = [
        overlap(first, second, top_k)
        for first, second in combinations(experiments, 2)
        for top_k in TOP_K_VALUES
    ]
    groups: dict[str, list[str]] = defaultdict(list)
    for item in experiments:
        groups[item["ranking_sequence_sha256"]].append(item["experiment"])
    return {
        "experiments_root": str(experiments_root.resolve()),
        "top_k_values": list(TOP_K_VALUES),
        "common_dimensions": list(next(iter(dimensions))),
        "experiments": [
            {
                key: value
                for key, value in item.items()
                if key != "ranking"
            }
            | {
                "top_heads": {
                    str(top_k): [list(head) for head in item["ranking"][:top_k]]
                    for top_k in TOP_K_VALUES
                }
            }
            for item in experiments
        ],
        "identical_full_ranking_groups": [
            values for values in groups.values() if len(values) > 1
        ],
        "pairwise": pairwise,
    }


def write_markdown(path: Path, report: dict) -> None:
    experiments = report["experiments"]
    lines = [
        "# mydata_bench v2 ranking head 重合度",
        "",
        "> 本文件由 `mydata_bench/write_ranking_overlap.py` 从各实验 ranking JSON 自动生成。",
        "",
        "## 口径",
        "",
        "- 所有实验都在同一个 896-head universe 中比较：36 层 × 32 heads，排除前 8 层。",
        "- 当前 ranking 来源有 36 条清单，其中 grounding 后可用 34 条；报告不强求 36/36 可用。",
        "- `交集/k` 表示两个 top-k 集合共享 head 数占 k 的比例；Jaccard 为 `|交集|/|并集|`。",
        "- 跨模型比较与同模型协议变化分开列出，避免把输入顺序、forward/incremental 或干预位置混成模型差异。",
        "",
        "## 各实验 top-8",
        "",
        markdown_table(
            ("实验", "模型", "ranking 文件", "发现样本", "top-8"),
            [
                (
                    item["experiment"],
                    item["model_family"],
                    Path(item["ranking_path"]).name,
                    item["discovery_samples"],
                    head_text([tuple(head) for head in item["top_heads"]["8"]]),
                )
                for item in experiments
            ],
        ),
        "",
        "## 完整 ranking 完全相同的实验组",
        "",
    ]
    groups = report["identical_full_ranking_groups"]
    if groups:
        lines.extend(f"- {', '.join(group)}" for group in groups)
    else:
        lines.append("- 无。")
    for cross_model, title in ((True, "跨模型重合度"), (False, "同模型协议重合度")):
        lines.extend(["", f"## {title}"])
        for top_k in TOP_K_VALUES:
            rows = [
                row
                for row in report["pairwise"]
                if row["top_k"] == top_k and row["cross_model"] == cross_model
            ]
            lines.extend(
                [
                    "",
                    f"### top-{top_k}",
                    "",
                    markdown_table(
                        ("实验 A", "实验 B", "共享", "交集/k", "Jaccard"),
                        [
                            (
                                row["first"],
                                row["second"],
                                row["intersection_count"],
                                percent(row["intersection_fraction"]),
                                percent(row["jaccard"]),
                            )
                            for row in rows
                        ],
                    ),
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=repo_root / "results/mydata_bench/experiments_v2",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "results/mydata_bench/experiments_v2/ranking_overlap.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=repo_root / "results/mydata_bench/experiments_v2/ranking_overlap.md",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help=(
            "Attention experiment directory name to compare; repeat for a "
            "custom matrix. Omit to preserve the original v2 ten-run report."
        ),
    )
    args = parser.parse_args()
    report = build_report(args.experiments_root, args.experiments)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_md, report)
    print(args.output_json.resolve())
    print(args.output_md.resolve())


if __name__ == "__main__":
    main()
