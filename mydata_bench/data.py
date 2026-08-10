from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .io import object_fingerprint, read_jsonl, sha256_file
from .schemas import EpisodeRecord


def metadata_path(dataset_root: str | Path, split: str = "test") -> Path:
    root = Path(dataset_root).resolve()
    root_level = root / "metadata.jsonl"
    return root_level if root_level.is_file() else root / split / "metadata.jsonl"


def normalize_subset(file_name: str) -> str:
    normalized = file_name.replace("\\", "/")
    return PurePosixPath(normalized).parts[0]


def classify_source(subset: str) -> str:
    return "RoboArena" if subset.lower() in {"robo_arena", "roboarena"} else "OXE"


def load_episodes(
    dataset_root: str | Path,
    split: str = "test",
    *,
    metadata_file: str | Path | None = None,
    compute_hash: bool = True,
    require_video: bool = True,
) -> Iterator[EpisodeRecord]:
    root = Path(dataset_root).resolve()
    # RoboRewardBench stores <root>/<split>/metadata.jsonl with one MP4 in
    # file_name. ljx_lfz_task/new stores one root-level manifest whose rows
    # carry three view paths and a logical suc/fail split.
    root_metadata = (
        Path(metadata_file).resolve()
        if metadata_file is not None
        else root / "metadata.jsonl"
    )
    is_ljx = (root / "metadata.jsonl").is_file()
    split_root = root if is_ljx else (root / split).resolve()
    metadata_path = root_metadata if is_ljx else split_root / "metadata.jsonl"
    hash_cache: dict[tuple[int, int, int], str] = {}

    def cached_sha256(path: str) -> str:
        stat = Path(path).stat()
        identity = (int(stat.st_dev), int(stat.st_ino), int(stat.st_size))
        if identity not in hash_cache:
            hash_cache[identity] = sha256_file(path)
        return hash_cache[identity]
    for row in read_jsonl(metadata_path):
        if is_ljx:
            row_split = str(row.get("split", ""))
            if split not in {"all", "test", ""} and row_split != split:
                continue
            raw_paths = row.get("video_paths")
            if not isinstance(raw_paths, list) or not raw_paths:
                raise ValueError(f"Missing video_paths for {row.get('id')}")
            views: dict[str, str] = {}
            for raw_path in raw_paths:
                relative = str(raw_path).replace("\\", "/")
                video = (root / relative).resolve()
                try:
                    video.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"Video path escapes dataset root: {relative}") from exc
                if require_video and not video.is_file():
                    raise FileNotFoundError(video)
                name = Path(relative).name
                camera = {
                    "faceImg.mp4": "front",
                    "leftImg.mp4": "left_wrist",
                    "rightImg.mp4": "right_wrist",
                }.get(name, Path(name).stem)
                views[camera] = str(video)
            primary = views.get("front") or next(iter(views.values()))
            digest = (
                object_fingerprint(
                    {
                        camera: cached_sha256(path)
                        for camera, path in sorted(views.items())
                    }
                )
                if compute_hash
                else ""
            )
            matched = bool(row.get("instruction_video_match", row_split == "suc"))
            yield EpisodeRecord(
                example_id=str(row["id"]),
                video_path=primary,
                task=str(row["instruction"]),
                reward=5 if matched else 1,
                subset=str(row.get("task_id") or "unknown_task"),
                video_sha256=digest,
                split=row_split,
                view_paths=tuple(sorted(views.items())),
                source_suc_id=str(row.get("source_suc_id") or row["id"]),
                target_obj=str(row.get("target_obj") or "") or None,
                correct_target_obj=str(row.get("correct_target_obj") or "") or None,
                instruction_video_match=matched,
            )
            continue
        file_name = str(row["file_name"]).replace("\\", "/")
        video = (split_root / file_name).resolve()
        try:
            video.relative_to(split_root)
        except ValueError as exc:
            raise ValueError(f"Video path escapes dataset split: {file_name}") from exc
        if require_video and not video.is_file():
            raise FileNotFoundError(video)
        digest = sha256_file(video) if compute_hash and video.is_file() else ""
        yield EpisodeRecord(
            example_id=file_name,
            video_path=str(video),
            task=str(row["task"]),
            reward=int(row["reward"]),
            subset=normalize_subset(file_name),
            video_sha256=digest,
            split=split,
            gpt5_mini_check=row.get("gpt5_mini_check"),
        )


def pair_input_id(pair_id: str, side: str) -> str:
    """Stable, instruction-specific identifier for one side of a video pair."""
    if side not in {"counterfactual", "original"}:
        raise ValueError(f"Unknown pair side: {side}")
    return f"pair/{pair_id}/{side}"


def load_pair_episodes(pair_manifest: str | Path) -> tuple[list[EpisodeRecord], list[dict[str, Any]]]:
    """Load the two instruction-conditioned inputs for each frozen video pair.

    The manifest is produced by ``run_paired_raw_eval.py prepare``.  Its reward
    fields are used only to validate the frozen pair construction; the returned
    ``EpisodeRecord.model_payload`` deliberately excludes them.
    """
    path = Path(pair_manifest).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    episodes: list[EpisodeRecord] = []
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        pair_id = str(row["pair_id"])
        digest = str(row["video_sha256"])
        video_path = str(Path(row["video_path"]).resolve())
        if not Path(video_path).is_file():
            raise FileNotFoundError(video_path)
        if not pair_id:
            raise ValueError("pair_id must be non-empty")
        if len(digest) != 64:
            raise ValueError(f"video_sha256 must be a SHA-256 digest: {digest}")
        subset = str(row["subset"])
        mapped = {
            "pair_id": pair_id,
            "video_sha256": digest,
            "video_path": video_path,
            "subset": subset,
            "counterfactual_example_id": pair_input_id(pair_id, "counterfactual"),
            "counterfactual_source_example_id": str(row["counterfactual_example_id"]),
            "counterfactual_task": str(row["counterfactual_task"]),
            "original_example_id": pair_input_id(pair_id, "original"),
            "original_source_example_id": str(row["original_example_id"]),
            "original_task": str(row["original_task"]),
            "view_paths": dict(row.get("view_paths", {})),
        }
        for side, reward in (("counterfactual", 1), ("original", 5)):
            example_id = mapped[f"{side}_example_id"]
            if example_id in seen:
                raise ValueError(f"Duplicate pair input ID: {example_id}")
            seen.add(example_id)
            episodes.append(
                EpisodeRecord(
                    example_id=example_id,
                    video_path=video_path,
                    task=mapped[f"{side}_task"],
                    reward=reward,
                    subset=subset,
                    video_sha256=digest,
                    split="paired",
                    view_paths=tuple(sorted(mapped["view_paths"].items())),
                )
            )
        pairs.append(mapped)
    if not episodes:
        raise ValueError(f"Pair manifest is empty: {path}")
    return episodes, pairs


def load_configured_episodes(config: dict[str, Any]) -> tuple[list[EpisodeRecord], list[dict[str, Any]] | None]:
    """Load ordinary dataset episodes or instruction-conditioned pair inputs."""
    pair_manifest = config.get("pair_manifest")
    if pair_manifest:
        return load_pair_episodes(pair_manifest)
    return (
        list(
            load_episodes(
                config["dataset_root"],
                config.get("split", "test"),
                metadata_file=config.get("metadata_file"),
                compute_hash=True,
            )
        ),
        None,
    )


def inventory(episodes: list[EpisodeRecord]) -> dict:
    return {
        "num_records": len(episodes),
        "num_unique_videos": len({row.video_sha256 for row in episodes}),
        "num_unique_instructions": len({row.task for row in episodes}),
        "subsets": dict(sorted(Counter(row.subset for row in episodes).items())),
        "rewards": {
            str(key): value
            for key, value in sorted(Counter(row.reward for row in episodes).items())
        },
        "sources": dict(sorted(Counter(classify_source(row.subset) for row in episodes).items())),
    }
