#!/usr/bin/env python3
"""统计 RoboRewardBench 四类数据的样本、标签和 subset 分布。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roborewardbench.data import CATEGORY_ORDER, summarize_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="test/metadata.jsonl 路径")
    parser.add_argument("--output", default=None, help="可选的 JSON 输出路径")
    args = parser.parse_args()

    summary = summarize_metadata(args.metadata)
    print(f"metadata: {summary['metadata_path']}")
    print(f"sha256:   {summary['metadata_sha256']}")
    print(f"总样本数: {summary['num_records']}")
    for category in CATEGORY_ORDER:
        row = summary["categories"][category]
        rewards = ", ".join(
            f"{label}:{count}" for label, count in row["reward_counts"].items() if count
        )
        print(f"\n{row['name_zh']}: {row['count']}（reward {rewards}）")
        for subset, count in row["subset_counts"].items():
            print(f"  {subset}: {count}")

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"\nJSON 已保存到 {output}")


if __name__ == "__main__":
    main()
