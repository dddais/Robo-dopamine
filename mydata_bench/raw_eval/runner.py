from __future__ import annotations

import json
import os
import sys
import traceback
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import section
from ..data import load_episodes, metadata_path
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    latest_by_id,
    provenance,
    read_jsonl,
    sha256_file,
    stable_shard,
    write_json,
)
from ..protocol import (
    accumulate_incremental_progress,
    chat_messages,
    multiview_endpoint_payload,
    native_endpoint_payload,
    official_incremental_indices,
    parse_score,
    progress,
    system_prompt,
    temporal_chat_messages,
)
from ..schemas import SCHEMA_VERSION
from ..video import extract_endpoints, extract_frame_at, extract_uniform


OFFICIAL_SAMPLING = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 1024,
}


def sampling_kwargs(config: dict[str, Any]) -> dict[str, float | int]:
    """Sampling settings, defaulting to examples/inference.py exactly."""
    return {
        "temperature": float(config.get("temperature", OFFICIAL_SAMPLING["temperature"])),
        "top_p": float(config.get("top_p", OFFICIAL_SAMPLING["top_p"])),
        "top_k": int(config.get("top_k", OFFICIAL_SAMPLING["top_k"])),
        "max_tokens": int(config.get("max_tokens", OFFICIAL_SAMPLING["max_tokens"])),
    }


class VLLMGRM:
    def __init__(self, config: dict[str, Any]):
        # vLLM 0.11 V1 forks its engine by default. Importing Transformers or
        # probing CUDA first then makes torch reject CUDA re-initialization in
        # the child. Spawn is deterministic and safe for this CLI entry point.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "Raw inference requires the robo-dopamine environment with transformers and vLLM"
            ) from exc
        self.processor = AutoProcessor.from_pretrained(
            config["model_path"], trust_remote_code=True
        )
        self.prompt_mode = str(config.get("prompt_mode", "official"))
        # Validate the selected prompt before expensive model construction.
        system_prompt(self.prompt_mode)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            image_processor.min_pixels = int(config.get("min_pixels", 12544))
            image_processor.max_pixels = int(config.get("max_pixels", 76800))
        self.model = LLM(
            model=config["model_path"],
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(config.get("max_model_len", 8192)),
            limit_mm_per_prompt={"image": 8},
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
        # Match examples/inference.py unless an explicit experiment config
        # deliberately requests a different decoding regime.
        self.sampling = SamplingParams(**sampling_kwargs(config))

    def infer(self, payload: dict[str, Any]) -> str:
        images = [Image.open(path).convert("RGB") for path in payload["image"]]
        messages = (
            temporal_chat_messages(payload["task"], len(images))
            if payload["protocol"] == "temporal8_single_view_ablation_v1"
            else chat_messages(
                payload["task"], str(payload.get("prompt_mode", self.prompt_mode))
            )
        )
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.model.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": images}}],
            sampling_params=self.sampling,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()


def run(config: dict[str, Any], *, dry_run: bool = False, retry_failed: bool = False) -> Path:
    raw = section(config, "raw_eval")
    output_dir = Path(raw["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_id = int(raw.get("shard_id", 0))
    num_shards = int(raw.get("num_shards", 1))
    records_path = output_dir / f"records.shard-{shard_id:02d}.jsonl"
    previous = latest_by_id(read_jsonl(records_path)) if records_path.exists() else {}
    prompt_mode = str(raw.get("prompt_mode", "official"))
    eval_mode = str(raw.get("eval_mode", "forward"))
    if eval_mode not in {"forward", "incremental"}:
        raise ValueError("raw_eval.eval_mode must be 'forward' or 'incremental'")
    incremental_protocol = str(
        raw.get("incremental_protocol", "official_accumulated_v1")
    )
    if eval_mode == "incremental" and incremental_protocol != "official_accumulated_v1":
        raise ValueError(
            "raw_eval.incremental_protocol must be official_accumulated_v1"
        )
    frame_interval = int(raw.get("frame_interval", 20))
    if frame_interval < 1:
        raise ValueError("raw_eval.frame_interval must be positive")
    prompt_template = system_prompt(prompt_mode)
    manifest = provenance(sys.argv, config, Path(__file__).resolve().parents[2])
    manifest["model_fingerprint"] = artifact_fingerprint(raw["model_path"])
    manifest["metadata_sha256"] = sha256_file(
        metadata_path(raw["dataset_root"], raw.get("split", "test"))
    )
    manifest["shard_id"] = shard_id
    manifest["num_shards"] = num_shards
    manifest["raw_protocol"] = {
        "prompt_mode": prompt_mode,
        "prompt_template_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "sampling": sampling_kwargs(raw),
        "eval_mode": eval_mode,
        "frame_interval": frame_interval,
        "comparison": (
            "start_to_terminal"
            if eval_mode == "forward"
            else "all_adjacent_interval_hops"
        ),
        "incremental_protocol": (
            "official_accumulated_v1" if eval_mode == "incremental" else None
        ),
        "reported_progress": (
            "official_accumulation_then_clip_0_1"
            if eval_mode == "incremental"
            else "clip_signed_score_0_1"
        ),
    }
    manifest["source_fingerprints"] = {
        "mydata_bench/protocol.py": sha256_file(Path(__file__).resolve().parents[1] / "protocol.py"),
        "mydata_bench/raw_eval/runner.py": sha256_file(Path(__file__).resolve()),
    }
    write_json(
        output_dir
        / ("manifest.json" if num_shards == 1 else f"manifest.shard-{shard_id:02d}.json"),
        manifest,
    )
    episodes = list(
        load_episodes(raw["dataset_root"], raw.get("split", "test"), compute_hash=True)
    )
    requested_ids = set(raw.get("example_ids", []))
    if requested_ids:
        episodes = [row for row in episodes if row.example_id in requested_ids]
        missing = requested_ids - {row.example_id for row in episodes}
        if missing:
            raise ValueError(f"Unknown raw_eval example_ids: {sorted(missing)}")
    all_episodes = list(episodes)
    episodes = [
        row for row in episodes if stable_shard(row.video_sha256, num_shards) == shard_id
    ]
    limit = int(raw.get("limit", 0))
    if limit:
        episodes = episodes[:limit]
    engine = None if dry_run else VLLMGRM(raw)
    frames_root = output_dir / "frames"
    blank_goal = raw.get(
        "blank_goal",
        str(Path(__file__).resolve().parents[2] / "examples" / "blank_goal.png"),
    )
    for episode in episodes:
        old = previous.get(episode.example_id)
        if old and (old.get("status") == "ok" or not retry_failed):
            continue
        attempt = int(old.get("attempt", 0)) + 1 if old else 1
        base = {
            "schema_version": SCHEMA_VERSION,
            "example_id": episode.example_id,
            "video_sha256": episode.video_sha256,
            "subset": episode.subset,
            "reward": episode.reward,
            "attempt": attempt,
        }
        try:
            frame_dir = frames_root / episode.video_sha256
            frames_by_view = {
                view: extract_endpoints(
                    episode.example_id,
                    episode.video_sha256,
                    path,
                    frame_dir / view,
                )
                for view, path in episode.views.items()
            }
            before_frame_indices = {
                view: record.first_index for view, record in frames_by_view.items()
            }
            if {"front", "left_wrist", "right_wrist"} <= set(frames_by_view):
                frames = frames_by_view["front"]
                if eval_mode == "incremental":
                    terminal_indices = {
                        view: frames_by_view[view].last_index
                        for view in ("front", "left_wrist", "right_wrist")
                    }
                    if len(set(terminal_indices.values())) != 1:
                        raise ValueError(
                            "incremental mode requires synchronized camera frame counts; "
                            f"terminal indices are {terminal_indices}"
                        )
                    sampled_indices = official_incremental_indices(
                        frames.last_index, frame_interval
                    )
                    if len(sampled_indices) < 2:
                        raise ValueError("incremental mode requires at least two frames")
                    incremental_steps = []
                    accumulated = None
                    payload = None
                    for hop_index, (before_index, after_index) in enumerate(
                        zip(sampled_indices, sampled_indices[1:])
                    ):
                        hop_paths: dict[str, dict[str, str]] = {
                            "before": {}, "after": {}
                        }
                        for view in ("front", "left_wrist", "right_wrist"):
                            record = frames_by_view[view]
                            for endpoint, index in (
                                ("before", before_index), ("after", after_index)
                            ):
                                if index == record.first_index:
                                    path = record.first_path
                                elif index == record.last_index:
                                    path = record.last_path
                                else:
                                    _actual, path = extract_frame_at(
                                        episode.views[view],
                                        frame_dir / view / f"frame_{index:06d}.png",
                                        index,
                                    )
                                hop_paths[endpoint][view] = path
                        payload = multiview_endpoint_payload(
                            episode,
                            frames_by_view,
                            blank_goal,
                            prompt_mode=prompt_mode,
                            eval_mode=eval_mode,
                            before_paths=hop_paths["before"],
                            after_paths=hop_paths["after"],
                        )
                        if {"reward", "gpt5_mini_check"} & payload.keys():
                            raise AssertionError("Label leakage into model payload")
                        output = (
                            "<score>0%</score>"
                            if dry_run
                            else engine.infer(payload)
                        )
                        hop_score = parse_score(output)
                        accumulated = accumulate_incremental_progress(
                            accumulated, hop_score
                        )
                        incremental_steps.append(
                            {
                                "hop_index": hop_index,
                                "before_frame_index": before_index,
                                "after_frame_index": after_index,
                                "raw_output": output,
                                "hop_score": hop_score,
                                "accumulated_progress_unclipped": accumulated,
                            }
                        )
                    assert payload is not None and accumulated is not None
                    before_frame_indices = {
                        view: sampled_indices[-2]
                        for view in ("front", "left_wrist", "right_wrist")
                    }
                    signed = incremental_steps[-1]["hop_score"]
                    reported_progress = progress(accumulated)
                    output = incremental_steps[-1]["raw_output"]
                    status = "dry_run" if dry_run else "ok"
                else:
                    payload = multiview_endpoint_payload(
                        episode,
                        frames_by_view,
                        blank_goal,
                        prompt_mode=prompt_mode,
                        eval_mode=eval_mode,
                    )
            else:
                if eval_mode != "forward":
                    raise ValueError(
                        "incremental mode requires all three camera views"
                    )
                frames = extract_endpoints(
                    episode.example_id,
                    episode.video_sha256,
                    episode.video_path,
                    frame_dir,
                )
                payload = native_endpoint_payload(
                    episode, frames, blank_goal, prompt_mode=prompt_mode
                )
            if eval_mode != "incremental":
                if {"reward", "gpt5_mini_check"} & payload.keys():
                    raise AssertionError("Label leakage into model payload")
                if dry_run:
                    output = "<score>0%</score>"
                    status = "dry_run"
                    signed = 0.0
                else:
                    assert engine is not None
                    output = engine.infer(payload)
                    signed = parse_score(output)
                    status = "ok"
                sampled_indices = None
                incremental_steps = None
                accumulated = None
                reported_progress = progress(signed)
            append_jsonl(
                records_path,
                {
                    **base,
                    "task": episode.task,
                    "frame_record": frames.to_dict(),
                    "frame_records": {
                        view: record.to_dict() for view, record in frames_by_view.items()
                    },
                    "protocol": payload["protocol"],
                    "prompt_mode": payload["prompt_mode"],
                    "eval_mode": eval_mode,
                    "before_frame_indices": before_frame_indices,
                    "incremental_protocol": (
                        "official_accumulated_v1"
                        if eval_mode == "incremental"
                        else None
                    ),
                    "sampled_frame_indices": sampled_indices,
                    "hop_count": (
                        len(incremental_steps) if incremental_steps is not None else None
                    ),
                    "incremental_steps": incremental_steps,
                    "raw_output": output,
                    "signed_score": signed,
                    "accumulated_progress_unclipped": accumulated,
                    "progress": reported_progress,
                    "status": status,
                },
            )
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    **base,
                    "status": "invalid",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    ablation = raw.get("temporal_ablation", {})
    if ablation.get("enabled", False):
        _run_temporal_ablation(
            all_episodes,
            output_dir,
            engine,
            dry_run=dry_run,
            max_per_subset=int(ablation.get("max_per_subset", 10)),
            frame_count=int(ablation.get("frame_count", 8)),
            retry_failed=retry_failed,
            shard_id=shard_id,
            num_shards=num_shards,
        )
    return records_path


def _run_temporal_ablation(
    episodes,
    output_dir: Path,
    engine: VLLMGRM | None,
    *,
    dry_run: bool,
    max_per_subset: int,
    frame_count: int,
    retry_failed: bool,
    shard_id: int,
    num_shards: int,
) -> None:
    by_subset: dict[str, list] = {}
    for episode in episodes:
        by_subset.setdefault(episode.subset, []).append(episode)
    selected = []
    for subset in sorted(by_subset):
        selected.extend(
            sorted(by_subset[subset], key=lambda row: (row.video_sha256, row.example_id))[
                :max_per_subset
            ]
        )
    selected = [
        row for row in selected if stable_shard(row.video_sha256, num_shards) == shard_id
    ]
    path = (
        output_dir / "temporal8_ablation.jsonl"
        if num_shards == 1
        else output_dir / f"temporal8_ablation.shard-{shard_id:02d}.jsonl"
    )
    previous = latest_by_id(read_jsonl(path)) if path.exists() else {}
    for episode in selected:
        old = previous.get(episode.example_id)
        if old and (old.get("status") == "ok" or not retry_failed):
            continue
        base = {
            "schema_version": SCHEMA_VERSION,
            "example_id": episode.example_id,
            "video_sha256": episode.video_sha256,
            "subset": episode.subset,
            "reward": episode.reward,
            "protocol": "temporal8_single_view_ablation_v1",
        }
        try:
            frames = extract_uniform(
                episode.video_path,
                output_dir / "temporal_frames" / episode.video_sha256,
                frame_count,
            )
            if len(frames) < 2:
                raise RuntimeError("Fewer than two temporal frames decoded")
            payload = {
                **episode.model_payload(),
                "protocol": "temporal8_single_view_ablation_v1",
                "image": [path for _, path in frames],
            }
            output = "<score>0%</score>" if dry_run else engine.infer(payload)
            signed = parse_score(output)
            append_jsonl(
                path,
                {
                    **base,
                    "frame_indices": [index for index, _ in frames],
                    "raw_output": output,
                    "signed_score": signed,
                    "progress": progress(signed),
                    "status": "dry_run" if dry_run else "ok",
                },
            )
        except Exception as exc:
            append_jsonl(path, {**base, "status": "invalid", "error": str(exc)})
    if num_shards > 1:
        paths = [
            output_dir / f"temporal8_ablation.shard-{index:02d}.jsonl"
            for index in range(num_shards)
        ]
        if all(item.exists() for item in paths):
            from ..io import deterministic_merge

            deterministic_merge(paths, output_dir / "temporal8_ablation.jsonl")
