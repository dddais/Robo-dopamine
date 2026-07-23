#!/usr/bin/env python3
"""Run Robo-Dopamine on RoboRewardBench with resumable terminal scoring."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np

from roborewardbench.data import (
    iter_metadata_rows,
    load_metadata_reference,
    normalize_subset,
    sha256_file,
)
from roborewardbench.score import read_jsonl, score_records

SCORE_PATTERN = re.compile(
    r"<score>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%\s*</score>"
)


@dataclass(frozen=True)
class Example:
    example_id: str
    video_path: Path
    task: str
    reward: int
    subset: str
    split: str


def iter_local_examples(dataset_root: str | Path, split: str) -> Iterator[Example]:
    root = Path(dataset_root).expanduser().resolve()
    split_root = root / split
    metadata_path = split_root / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    seen_ids: set[str] = set()
    for row in iter_metadata_rows(metadata_path):
        file_name = str(row["file_name"])
        if file_name in seen_ids:
            raise ValueError(f"Duplicate file_name in metadata: {file_name}")
        seen_ids.add(file_name)
        video_path = (split_root / file_name).resolve()
        try:
            video_path.relative_to(split_root)
        except ValueError as exc:
            raise ValueError(
                f"Video path escapes the selected split directory: {file_name}"
            ) from exc
        yield Example(
            example_id=file_name,
            video_path=video_path,
            task=str(row["task"]),
            reward=int(row["reward"]),
            subset=normalize_subset(file_name),
            split=split,
        )


def uniform_indices(num_frames: int, max_states: int) -> list[int]:
    if num_frames < 2:
        raise ValueError(f"A rollout needs at least two frames, got {num_frames}")
    if max_states < 2:
        raise ValueError("max_states must be at least 2")
    count = min(num_frames, max_states)
    indices = np.rint(np.linspace(0, num_frames - 1, count)).astype(int).tolist()
    indices[0] = 0
    indices[-1] = num_frames - 1
    return list(dict.fromkeys(indices))


def one_fps_indices(num_frames: int, source_fps: float, max_states: int | None) -> list[int]:
    if num_frames < 2:
        raise ValueError(f"A rollout needs at least two frames, got {num_frames}")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Video reports invalid FPS: {source_fps}")
    step = max(1, int(round(source_fps)))
    indices = list(range(0, num_frames, step))
    if indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    if max_states is not None and len(indices) > max_states:
        return uniform_indices(num_frames, max_states)
    return indices


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    sampling: str,
    max_states: int | None,
) -> tuple[list[Path], list[int], float]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if sampling == "uniform":
        if max_states is None:
            cap.release()
            raise ValueError("uniform sampling requires a positive max_states")
        indices = uniform_indices(num_frames, max_states)
    elif sampling == "1fps":
        indices = one_fps_indices(num_frames, source_fps, max_states)
    else:
        cap.release()
        raise ValueError(f"Unknown frame sampling mode: {sampling}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    decoded_indices: list[int] = []
    try:
        for position, idx in enumerate(indices):
            # Some released MP4 containers over-report CAP_PROP_FRAME_COUNT by
            # one. For the requested terminal state, walk backward to the last
            # decodable frame while never crossing the preceding sampled state.
            lower_bound = decoded_indices[-1] + 1 if decoded_indices else 0
            candidates = (
                range(idx, lower_bound - 1, -1)
                if position == len(indices) - 1
                else (idx,)
            )
            frame = None
            actual_idx = idx
            for candidate in candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
                ok, candidate_frame = cap.read()
                if ok and candidate_frame is not None:
                    frame = candidate_frame
                    actual_idx = candidate
                    break
            if not ok or frame is None:
                raise RuntimeError(f"Failed to decode frame {idx} from {video_path}")
            frame_path = output_dir / f"frame_{actual_idx:06d}.png"
            if not cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]):
                raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
            paths.append(frame_path)
            decoded_indices.append(actual_idx)
    finally:
        cap.release()
    return paths, decoded_indices, source_fps


def write_blank_goal(path: str | Path, size: int = 224) -> Path:
    path = Path(path)
    blank = np.full((size, size, 3), 128, dtype=np.uint8)
    if not cv2.imwrite(str(path), blank):
        raise RuntimeError(f"Failed to write blank goal image: {path}")
    return path


def make_inference_item(
    *,
    task: str,
    start: Path,
    goal: Path,
    before: Path,
    after: Path,
    item_id: str,
) -> dict[str, Any]:
    # RoboRewardBench is single-view, so repeat that view in the three GRM slots.
    return {
        "id": item_id,
        "task": task,
        "image": [
            str(start),
            str(goal),
            str(before),
            str(before),
            str(before),
            str(after),
            str(after),
            str(after),
        ],
    }


def parse_score(text: str) -> float:
    match = SCORE_PATTERN.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"Expected exactly <score>NUMBER%</score>, got {text!r}")
    percentage = float(match.group(1))
    if not np.isfinite(percentage):
        raise ValueError(f"Score must be finite: {percentage}")
    if not -100 <= percentage <= 100:
        raise ValueError(f"Score is outside [-100, 100]: {percentage}")
    return percentage / 100.0


def aggregate_incremental(hops: Sequence[float]) -> float:
    progress = 0.0
    for hop in hops:
        if hop >= 0:
            progress = progress + (1.0 - progress) * hop
        else:
            progress = progress + progress * hop
    return min(max(float(progress), 0.0), 1.0)


def outputs_by_id(items: Sequence[dict[str, Any]], outputs: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Validate model output identity instead of assuming output order."""

    expected_ids = [str(item["id"]) for item in items]
    if len(outputs) != len(items):
        raise RuntimeError(f"Model returned {len(outputs)} outputs for {len(items)} inputs")
    result: dict[str, str] = {}
    for output in outputs:
        if output.get("id") is None:
            raise RuntimeError("Model output is missing its input id")
        item_id = str(output["id"])
        if item_id in result:
            raise RuntimeError(f"Model returned duplicate output id: {item_id}")
        result[item_id] = str(output.get("pred", ""))
    if set(result) != set(expected_ids):
        missing = set(expected_ids) - set(result)
        unexpected = set(result) - set(expected_ids)
        raise RuntimeError(
            f"Model output ids do not match inputs: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return result


def predict_example(
    model: Any,
    example: Example,
    *,
    mode: str,
    frame_sampling: str,
    max_states: int | None,
    batch_size: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="roborewardbench_") as temp_dir:
        temp = Path(temp_dir)
        effective_sampling = "endpoints" if mode == "forward" else frame_sampling
        effective_max_states = 2 if mode == "forward" else max_states
        frames, indices, source_fps = extract_frames(
            example.video_path,
            temp / "frames",
            sampling="uniform" if mode == "forward" else frame_sampling,
            max_states=effective_max_states,
        )
        goal = write_blank_goal(temp / "blank_goal.png")
        if mode == "forward":
            items = [make_inference_item(
                task=example.task,
                start=frames[0],
                goal=goal,
                before=frames[0],
                after=frames[-1],
                item_id=f"{example.example_id}:forward-terminal",
            )]
        elif mode == "incremental":
            items = [
                make_inference_item(
                    task=example.task,
                    start=frames[0],
                    goal=goal,
                    before=frames[idx],
                    after=frames[idx + 1],
                    item_id=f"{example.example_id}:incremental-{idx}",
                )
                for idx in range(len(frames) - 1)
            ]
        else:
            raise ValueError("mode must be 'forward' or 'incremental'")

        outputs = []
        for start_idx in range(0, len(items), batch_size):
            outputs.extend(model.inference_batch(items[start_idx : start_idx + batch_size]))
        output_text = outputs_by_id(items, outputs)
        raw_outputs = [output_text[str(item["id"])] for item in items]
        scores = [parse_score(raw) for raw in raw_outputs]
        progress = min(max(scores[-1], 0.0), 1.0) if mode == "forward" else aggregate_incremental(scores)

    return {
        "progress": progress,
        "raw_scores": scores,
        "raw_outputs": raw_outputs,
        "sampled_frame_indices": indices,
        "source_fps": source_fps,
        "effective_frame_sampling": effective_sampling,
    }


def predict_forward_batch(
    model: Any,
    examples: Sequence[Example],
) -> dict[str, dict[str, Any] | Exception]:
    """Predict several terminal examples in one vLLM generation call.

    Frame extraction failures remain isolated to their own example. A generation
    failure is returned for every prepared example in that batch so the JSONL
    runner can persist and optionally retry them.
    """

    outcomes: dict[str, dict[str, Any] | Exception] = {}
    if not examples:
        return outcomes
    with tempfile.TemporaryDirectory(prefix="roborewardbench_batch_") as temp_dir:
        temp = Path(temp_dir)
        goal = write_blank_goal(temp / "blank_goal.png")
        items: list[dict[str, Any]] = []
        metadata: dict[str, tuple[list[int], float]] = {}
        for position, example in enumerate(examples):
            try:
                frames, indices, source_fps = extract_frames(
                    example.video_path,
                    temp / f"frames_{position:04d}",
                    sampling="uniform",
                    max_states=2,
                )
                item = make_inference_item(
                    task=example.task,
                    start=frames[0],
                    goal=goal,
                    before=frames[0],
                    after=frames[-1],
                    item_id=f"{example.example_id}:forward-terminal",
                )
                items.append(item)
                metadata[str(item["id"])] = (indices, source_fps)
            except Exception as exc:
                outcomes[example.example_id] = exc

        if not items:
            return outcomes
        try:
            output_text = outputs_by_id(items, model.inference_batch(items))
        except Exception as exc:
            for example in examples:
                outcomes.setdefault(example.example_id, exc)
        else:
            for example in examples:
                if example.example_id in outcomes:
                    continue
                item_id = f"{example.example_id}:forward-terminal"
                try:
                    raw = output_text[item_id]
                    score = parse_score(raw)
                    indices, source_fps = metadata[item_id]
                except Exception as exc:
                    outcomes[example.example_id] = exc
                    continue
                outcomes[example.example_id] = {
                    "progress": min(max(score, 0.0), 1.0),
                    "raw_scores": [score],
                    "raw_outputs": [raw],
                    "sampled_frame_indices": indices,
                    "source_fps": source_fps,
                    "effective_frame_sampling": "endpoints",
                }
    return outcomes


def load_model(model_path: str, temperature: float, seed: int) -> Any:
    # Keep vLLM imports out of metric-only commands and unit tests.
    from examples.inference import GRMInference
    from vllm import SamplingParams

    model = GRMInference(model_path)
    model.sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=32,
        seed=seed,
    )
    return model


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fingerprint_files(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    """Hash a stable ordered file set for exact resume provenance."""

    files = []
    combined = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        relative = str(path.relative_to(root))
        file_hash = sha256_file(path)
        size = path.stat().st_size
        files.append({"path": relative, "size": size, "sha256": file_hash})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_hash.encode("ascii"))
        combined.update(b"\0")
    return {"sha256": combined.hexdigest(), "files": files}


def source_fingerprint() -> dict[str, Any]:
    """Fingerprint inference-affecting local source, including untracked files."""

    repository_root = Path(__file__).resolve().parents[1]
    sources = list((repository_root / "roborewardbench").glob("*.py"))
    sources.append(repository_root / "examples" / "inference.py")
    return fingerprint_files(sources, repository_root)


def model_fingerprint(model_path: str) -> dict[str, Any]:
    """Hash all files for a local checkpoint; retain the identifier for Hub models."""

    path = Path(model_path).expanduser()
    if not path.exists():
        return {
            "kind": "hub_identifier",
            "identifier": model_path,
            "resolved_revision": None,
        }
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"Model path is not a directory: {resolved}")
    files = [candidate for candidate in resolved.rglob("*") if candidate.is_file()]
    fingerprint = fingerprint_files(files, resolved)
    return {
        "kind": "local_directory",
        "path": str(resolved),
        **fingerprint,
    }


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("vllm", "torch", "transformers", "opencv-python-headless", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def run_signature(
    args: argparse.Namespace,
    max_states: int | None,
    *,
    metadata_reference: dict[str, Any],
    dataset_files: dict[str, Any],
    source: dict[str, Any],
    model: dict[str, Any],
    dependencies: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "metadata_sha256": metadata_reference["sha256"],
        "metadata_num_records": metadata_reference["num_records"],
        "dataset_files_sha256": dataset_files["sha256"],
        "dataset_files_num_files": len(dataset_files["files"]),
        "split": args.split,
        "model": model,
        "source_sha256": source["sha256"],
        "dependency_versions": dependencies,
        "mode": args.mode,
        "frame_sampling": args.frame_sampling,
        "max_states": max_states,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "seed": args.seed,
        "start": args.start,
        "limit": args.limit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root containing <split>/metadata.jsonl")
    parser.add_argument("--split", choices=("val", "validation", "test"), default="test")
    parser.add_argument("--model", default="tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("forward", "incremental"), default="forward")
    parser.add_argument("--frame-sampling", choices=("uniform", "1fps"), default="1fps")
    parser.add_argument(
        "--max-states",
        type=int,
        default=0,
        help="State cap; 0 keeps all 1 FPS states. Use 8 with --frame-sampling uniform for Robometer parity.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--calibration", default=None, help="Validation-fitted calibration JSON for scoring")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.temperature < 0:
        raise ValueError("temperature cannot be negative")
    if args.start < 0:
        raise ValueError("start cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    max_states = None if args.max_states == 0 else args.max_states
    if max_states is not None and max_states < 2:
        raise ValueError("max-states must be 0 (unlimited) or at least 2")
    split_dir = "val" if args.split == "validation" else args.split
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata_path = dataset_root / split_dir / "metadata.jsonl"
    metadata_reference = load_metadata_reference(metadata_path)
    examples = list(iter_local_examples(dataset_root, split_dir))[args.start :]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(
            f"No examples selected from {metadata_path}; check --start={args.start} "
            f"and --limit={args.limit}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    previous = read_jsonl(predictions_path) if predictions_path.exists() else []
    completed = {
        str(row["id"])
        for row in previous
        if not args.retry_invalid or row.get("status", "ok") == "ok"
    }

    print("Fingerprinting source, selected videos, and model for safe resume ...")
    source = source_fingerprint()
    dataset_files = fingerprint_files(
        [example.video_path for example in examples],
        dataset_root,
    )
    model_artifact = model_fingerprint(args.model)
    dependencies = dependency_versions()
    signature = run_signature(
        args,
        max_states,
        metadata_reference=metadata_reference,
        dataset_files=dataset_files,
        source=source,
        model=model_artifact,
        dependencies=dependencies,
    )
    manifest_path = output_dir / "run_manifest.json"
    if predictions_path.exists() and not manifest_path.exists():
        raise ValueError(
            "Output directory contains predictions.jsonl but no run_manifest.json; "
            "refusing an unsafe resume."
        )
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("run_signature") != signature:
            raise ValueError(
                "Output directory belongs to a different code, data, model, environment, or "
                "argument signature. Choose a new --output-dir or restore the original run."
            )
        manifest.setdefault("resume_events", []).append({"resumed_unix": time.time()})
    else:
        manifest = {
            "schema_version": 3,
            "created_unix": time.time(),
            "git_revision": git_revision(),
            "arguments": vars(args),
            "run_signature": signature,
            "source_fingerprint": source,
            "dataset_files_fingerprint": dataset_files,
            "model_fingerprint": model_artifact,
            "metadata": {
                "path": metadata_reference["path"],
                "sha256": metadata_reference["sha256"],
                "num_records": metadata_reference["num_records"],
            },
            "num_selected_examples": len(examples),
            "resume_events": [],
            "protocol": {
                "single_view": True,
                "goal_image": "neutral_gray_placeholder",
                "terminal_only": args.mode == "forward",
                "effective_frame_sampling": (
                    "endpoints" if args.mode == "forward" else args.frame_sampling
                ),
                "benchmark_metric": "23-subset macro MAE",
                "continuous_adapter": "fixed_nearest_ordinal_bins",
            },
        }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    pending = [example for example in examples if example.example_id not in completed]
    print(f"Selected {len(examples)} examples; {len(pending)} pending after resume check")
    model = load_model(args.model, args.temperature, args.seed) if pending else None

    def base_record(example: Example) -> dict[str, Any]:
        return {
            "id": example.example_id,
            "video_path": str(example.video_path),
            "task": example.task,
            "reward": example.reward,
            "subset": example.subset,
            "split": "validation" if split_dir == "val" else split_dir,
            "mode": args.mode,
            "frame_sampling": args.frame_sampling,
            "max_states": max_states,
        }

    def persist(example: Example, outcome: dict[str, Any] | Exception, elapsed: float, index: int) -> None:
        record = base_record(example)
        if isinstance(outcome, Exception):
            record.update({
                "status": "invalid",
                "error_type": type(outcome).__name__,
                "error": str(outcome),
            })
        else:
            record.update(outcome)
            record["status"] = "ok"
        record["elapsed_seconds"] = elapsed
        append_jsonl(predictions_path, record)
        print(
            f"[{index}/{len(pending)}] {record['status']} "
            f"subset={example.subset} id={example.example_id}"
        )

    if args.mode == "forward":
        completed_in_turn = 0
        for start_idx in range(0, len(pending), args.batch_size):
            batch = pending[start_idx : start_idx + args.batch_size]
            started = time.time()
            outcomes = predict_forward_batch(model, batch)
            amortized_elapsed = (time.time() - started) / len(batch)
            for example in batch:
                completed_in_turn += 1
                persist(
                    example,
                    outcomes[example.example_id],
                    amortized_elapsed,
                    completed_in_turn,
                )
    else:
        for index, example in enumerate(pending, 1):
            started = time.time()
            try:
                outcome: dict[str, Any] | Exception = predict_example(
                    model,
                    example,
                    mode=args.mode,
                    frame_sampling=args.frame_sampling,
                    max_states=max_states,
                    batch_size=args.batch_size,
                )
            except Exception as exc:
                outcome = exc
            persist(example, outcome, time.time() - started, index)

    if not predictions_path.exists():
        print("No predictions were needed or generated")
        return

    all_records = read_jsonl(predictions_path)
    latest_by_id = {str(row["id"]): row for row in all_records}
    selected_ids = {example.example_id for example in examples}
    selected_records = [row for key, row in latest_by_id.items() if key in selected_ids]
    if any(row.get("status", "ok") == "ok" for row in selected_records):
        metrics = score_records(
            selected_records,
            calibration_path=args.calibration,
            metadata_path=metadata_path,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
        )
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        print(json.dumps(metrics, indent=2, ensure_ascii=True))
    else:
        print("No valid predictions; metrics were not generated")


if __name__ == "__main__":
    main()
