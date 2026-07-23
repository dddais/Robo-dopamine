"""RoboRewardBench metadata loading, validation, and source categorization."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping

TEMPORAL_CLIP_PATTERN = re.compile(r"_attempt_\d+_score_([1-4])\.mp4$")

CATEGORY_ROBOARENA = "roboarena_natural"
CATEGORY_OXE_TEMPORAL = "oxe_temporal_clip"
CATEGORY_OXE_COUNTERFACTUAL = "oxe_counterfactual"
CATEGORY_OXE_ORIGINAL = "oxe_original_success"
CATEGORY_ORDER = (
    CATEGORY_ROBOARENA,
    CATEGORY_OXE_TEMPORAL,
    CATEGORY_OXE_COUNTERFACTUAL,
    CATEGORY_OXE_ORIGINAL,
)
CATEGORY_NAMES_ZH = {
    CATEGORY_ROBOARENA: "RoboArena 自然 rollout",
    CATEGORY_OXE_TEMPORAL: "OXE 时间截断",
    CATEGORY_OXE_COUNTERFACTUAL: "OXE 反事实任务",
    CATEGORY_OXE_ORIGINAL: "OXE 原始成功样本",
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_subset(file_name: str) -> str:
    """Use the first path component as the benchmark group name."""

    normalized = file_name.replace("\\", "/").lstrip("/")
    subset = normalized.split("/", 1)[0]
    return "robo_arena" if subset == "roboarena" else subset


def iter_metadata_rows(metadata_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated rows from a RoboReward metadata JSONL file."""

    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            file_name = row.get("file_name") or row.get("video")
            if not isinstance(file_name, str) or not file_name:
                raise ValueError(f"Missing string file_name at {path}:{line_number}")
            if "task" not in row:
                raise ValueError(f"Missing task at {path}:{line_number}")
            reward = int(row["reward"])
            if reward not in range(1, 6) or float(row["reward"]) != reward:
                raise ValueError(f"Invalid reward {row['reward']!r} at {path}:{line_number}")
            yield {**row, "file_name": file_name, "reward": reward}


def load_metadata_reference(metadata_path: str | Path) -> dict[str, Any]:
    """Load the exact sample identity and labels used for comparability checks."""

    path = Path(metadata_path).expanduser().resolve()
    records: dict[str, dict[str, Any]] = {}
    for row in iter_metadata_rows(path):
        example_id = str(row["file_name"])
        if example_id in records:
            raise ValueError(f"Duplicate file_name in metadata: {example_id}")
        records[example_id] = {
            "subset": normalize_subset(example_id),
            "reward": int(row["reward"]),
            "task": str(row["task"]),
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "num_records": len(records),
        "records": records,
    }


def classify_source(file_name: str, reward: int) -> str:
    """Classify a released RoboReward example into one of four paper categories.

    The public release encodes temporal clips as
    ``*_attempt_<n>_score_<1-4>.mp4``. Non-clipped OXE examples with reward 5
    are the original successful demonstrations; non-clipped OXE examples with
    rewards 1--4 are counterfactual task relabels of successful videos.
    """

    label = int(reward)
    if label not in range(1, 6):
        raise ValueError(f"Reward must be in 1..5, got {reward!r}")
    if normalize_subset(file_name) == "robo_arena":
        return CATEGORY_ROBOARENA
    match = TEMPORAL_CLIP_PATTERN.search(file_name.replace("\\", "/"))
    if match is not None:
        encoded_label = int(match.group(1))
        if encoded_label != label:
            raise ValueError(
                f"Temporal-clip filename encodes score {encoded_label}, but metadata has {label}: "
                f"{file_name}"
            )
        return CATEGORY_OXE_TEMPORAL
    if label == 5:
        return CATEGORY_OXE_ORIGINAL
    return CATEGORY_OXE_COUNTERFACTUAL


def summarize_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Compute deterministic category/subset/reward counts for a metadata file."""

    reference = load_metadata_reference(metadata_path)
    category_counts: Counter[str] = Counter()
    reward_counts: dict[str, Counter[int]] = defaultdict(Counter)
    subset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example_id, row in reference["records"].items():
        category = classify_source(example_id, row["reward"])
        category_counts[category] += 1
        reward_counts[category][row["reward"]] += 1
        subset_counts[category][row["subset"]] += 1

    return {
        "metadata_path": reference["path"],
        "metadata_sha256": reference["sha256"],
        "num_records": reference["num_records"],
        "categories": {
            category: {
                "name_zh": CATEGORY_NAMES_ZH[category],
                "count": category_counts[category],
                "reward_counts": {
                    str(label): reward_counts[category][label] for label in range(1, 6)
                },
                "subset_counts": dict(sorted(subset_counts[category].items())),
            }
            for category in CATEGORY_ORDER
        },
    }


def metadata_matches_records(
    records: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    """Compare deduplicated predictions with exact metadata identities and labels."""

    problems: list[str] = []
    actual_ids = set(records)
    expected_ids = set(expected)
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        unexpected = actual_ids - expected_ids
        if missing:
            problems.append(f"missing_ids={len(missing)}")
        if unexpected:
            problems.append(f"unexpected_ids={len(unexpected)}")
    for example_id in sorted(actual_ids & expected_ids):
        actual = records[example_id]
        target = expected[example_id]
        for field in ("subset", "reward", "task"):
            if str(actual.get(field)) != str(target.get(field)):
                problems.append(f"{field}_mismatch:{example_id}")
                break
    return not problems, problems
