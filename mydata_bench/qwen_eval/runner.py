from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from ..config import section
from ..data import load_episodes, metadata_path
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    latest_by_id,
    object_fingerprint,
    provenance,
    read_jsonl,
    sha256_file,
    stable_shard,
    write_json,
)
from ..schemas import SCHEMA_VERSION
from ..video import extract_endpoints, extract_uniform_image_sequence
from ..protocol import multiview_endpoint_payload
from .protocols import (
    IMAGE_SEQUENCE_PROTOCOLS,
    ROBO_DOPAMINE_FORWARD,
    ROBOREWARDBENCH_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_NATIVE,
    dopamine_forward_messages,
    dopamine_forward_payload,
    image_sequence_messages,
    image_sequence_payload,
    interleaved_image_sequence_messages,
    interleaved_image_sequence_payload,
    native_video_payload,
    parse_protocol_output,
    protocol_descriptor,
    validate_protocol,
)


def requested_example_ids(evaluation: dict[str, Any]) -> set[str]:
    """Read exactly one label-free frozen ID allow-list."""
    inline = evaluation.get("example_ids", [])
    path_value = evaluation.get("example_ids_file")
    if inline and path_value:
        raise ValueError("Use only one of qwen_eval.example_ids or example_ids_file")
    if path_value:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        inline = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inline, list) or not all(isinstance(value, str) for value in inline):
        raise ValueError("Qwen example IDs must be a JSON array of strings")
    requested = set(inline)
    if len(requested) != len(inline):
        raise ValueError("Duplicate example IDs in Qwen allow-list")
    return requested


def _model_load_kwargs(config: dict[str, Any], dtype: Any) -> dict[str, Any]:
    """Keep the historical backend default unless explicitly overridden."""
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": config.get("device_map", "auto"),
    }
    if config.get("attn_implementation"):
        kwargs["attn_implementation"] = str(config["attn_implementation"])
    return kwargs


class Qwen3VLBaseline:
    """Direct Transformers runner for both explicitly frozen baseline contracts."""

    def __init__(self, config: dict[str, Any], protocol: str):
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen evaluation requires torch and transformers in the robo-dopamine environment"
            ) from exc
        self.torch = torch
        self.protocol = validate_protocol(protocol)
        processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
        self.processor = AutoProcessor.from_pretrained(config["model_path"], **processor_kwargs)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if "min_pixels" in config:
                image_processor.min_pixels = int(config["min_pixels"])
            if "max_pixels" in config:
                image_processor.max_pixels = int(config["max_pixels"])
        dtype_name = str(config.get("torch_dtype", "bfloat16"))
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch_dtype {dtype_name!r}") from exc
        model_kwargs = _model_load_kwargs(config, dtype)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["model_path"], **model_kwargs
        ).eval()
        self.attention_backend = str(
            getattr(self.model.config, "_attn_implementation", "unknown")
        )
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        self.do_sample = bool(config.get("do_sample", False))
        self.temperature = float(config.get("temperature", 0.7))
        self.top_p = float(config.get("top_p", 0.8))
        self.top_k = int(config.get("top_k", 20))

    def _generation_kwargs(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            value.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )
        return value

    def _decode(self, inputs: Any) -> str:
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **self._generation_kwargs())
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _infer_native_video(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        text_item = {"type": "text", "text": payload["prompt"]}
        video_item = {"type": "video", "video": payload["video_path"]}
        content_order = str(payload.get("content_order", "text_then_video"))
        if content_order == "text_then_video":
            content = [text_item, video_item]
        elif content_order == "video_then_text":
            content = [video_item, text_item]
        else:
            raise ValueError(
                "content_order must be 'text_then_video' or 'video_then_text', "
                f"got {content_order!r}"
            )
        message = [
            {
                "role": "user",
                "content": content,
            }
        ]
        inputs = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_metadata=True,
        )
        metadata = inputs.pop("video_metadata", None)
        if metadata is None:
            raise ValueError("native video processor did not return video metadata")
        converter = getattr(inputs, "convert_to_tensors", None)
        if callable(converter):
            converted = converter(tensor_type="pt")
            if converted is not None:
                inputs = converted
        video_token_id = int(getattr(self.model.config, "video_token_id", 151656))
        grid = inputs.get("video_grid_thw")
        if isinstance(metadata, (list, tuple)):
            if len(metadata) != 1:
                raise ValueError("expected metadata for exactly one native video")
            metadata = metadata[0]
        fields = ("total_num_frames", "fps", "width", "height", "duration", "video_backend", "frames_indices")
        metadata_record = {}
        for field in fields:
            value = getattr(metadata, field, None)
            metadata_record[field] = value.tolist() if hasattr(value, "tolist") else value
        return self._decode(inputs), {
            "processor_native_video": True,
            "video_grid_thw": grid.detach().cpu().tolist() if grid is not None else None,
            "video_token_count": int((inputs["input_ids"] == video_token_id).sum()),
            "video_metadata": metadata_record,
            "content_order": content_order,
            "attention_backend": self.attention_backend,
        }

    def _infer_endpoint_images(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        paths = [str(Path(path).resolve()) for path in payload["image"]]
        message = dopamine_forward_messages(payload)
        inputs = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
        grid = inputs.get("image_grid_thw")
        return self._decode(inputs), {
            "processor_native_video": False,
            "image_count": len(paths),
            "image_grid_thw": grid.detach().cpu().tolist() if grid is not None else None,
            "image_token_count": int((inputs["input_ids"] == image_token_id).sum()),
            "attention_backend": self.attention_backend,
        }

    def _infer_image_sequence(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        paths = [str(Path(path).resolve()) for path in payload["image"]]
        content_order = str(payload["content_order"])
        messages = (
            interleaved_image_sequence_messages(payload["task"], paths)
            if payload["protocol"]
            == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE
            else image_sequence_messages(
                payload["task"], paths, content_order=content_order
            )
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
        grid = inputs.get("image_grid_thw")
        grid_count = int(grid.shape[0]) if grid is not None else 0
        if grid_count != len(paths):
            raise RuntimeError(
                "Independent-image processor alignment failed: "
                f"images={len(paths)}, grids={grid_count}"
            )
        return self._decode(inputs), {
            "processor_native_video": False,
            "input_representation": (
                "uniform_interleaved_images_v1"
                if payload["protocol"]
                == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE
                else "uniform_independent_images_v1"
            ),
            "image_count": len(paths),
            "image_grid_thw": grid.detach().cpu().tolist(),
            "image_token_count": int((inputs["input_ids"] == image_token_id).sum()),
            "content_order": content_order,
            "media_order": payload["media_order"],
            "sampling_record": payload["sampling_record"],
            "attention_backend": self.attention_backend,
        }

    def infer(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if payload["protocol"] == ROBOREWARDBENCH_NATIVE:
            return self._infer_native_video(payload)
        if payload["protocol"] in IMAGE_SEQUENCE_PROTOCOLS:
            return self._infer_image_sequence(payload)
        if payload["protocol"] == ROBO_DOPAMINE_FORWARD:
            return self._infer_endpoint_images(payload)
        raise ValueError(f"Unknown payload protocol {payload['protocol']!r}")


def run(config: dict[str, Any], *, dry_run: bool = False, retry_failed: bool = False) -> Path:
    evaluation = section(config, "qwen_eval")
    output_dir = Path(evaluation["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = validate_protocol(str(evaluation.get("protocol", ROBOREWARDBENCH_NATIVE)))
    prompt_mode = str(evaluation.get("prompt_mode", "official"))
    content_order = str(evaluation.get("content_order", "text_then_video"))
    requested_ids = requested_example_ids(evaluation)
    shard_id = int(evaluation.get("shard_id", 0))
    num_shards = int(evaluation.get("num_shards", 1))
    if not 0 <= shard_id < num_shards:
        raise ValueError("qwen_eval.shard_id must be in [0, num_shards)")
    records_path = output_dir / f"records.shard-{shard_id:02d}.jsonl"
    previous = latest_by_id(read_jsonl(records_path)) if records_path.exists() else {}
    manifest = provenance(sys.argv, config, Path(__file__).resolve().parents[2])
    manifest.update(
        {
            "model_fingerprint": artifact_fingerprint(evaluation["model_path"]),
            "metadata_sha256": sha256_file(
                metadata_path(
                    evaluation["dataset_root"], evaluation.get("split", "test")
                )
            ),
            "shard_id": shard_id,
            "num_shards": num_shards,
            "qwen_protocol": {
                **protocol_descriptor(
                    protocol,
                    prompt_mode=prompt_mode,
                    content_order=content_order,
                ),
                "frozen_id_cohort": {
                    "requested_id_count": len(requested_ids) if requested_ids else None,
                    "requested_ids_fingerprint": (
                        object_fingerprint(sorted(requested_ids)) if requested_ids else None
                    ),
                    "example_ids_file": (
                        str(Path(evaluation["example_ids_file"]).resolve())
                        if evaluation.get("example_ids_file")
                        else None
                    ),
                    "example_ids_file_sha256": (
                        sha256_file(evaluation["example_ids_file"])
                        if evaluation.get("example_ids_file")
                        else None
                    ),
                },
            },
        }
    )
    write_json(
        output_dir / ("manifest.json" if num_shards == 1 else f"manifest.shard-{shard_id:02d}.json"),
        manifest,
    )
    episodes = list(
        load_episodes(evaluation["dataset_root"], evaluation.get("split", "test"), compute_hash=True)
    )
    if requested_ids:
        episodes = [row for row in episodes if row.example_id in requested_ids]
        missing = requested_ids - {row.example_id for row in episodes}
        if missing:
            raise ValueError(f"Unknown qwen_eval example_ids: {sorted(missing)}")
    episodes = [
        row for row in episodes if stable_shard(row.video_sha256, num_shards) == shard_id
    ]
    limit = int(evaluation.get("limit", 0))
    if limit:
        episodes = episodes[:limit]
    engine = None if dry_run else Qwen3VLBaseline(evaluation, protocol)
    blank_goal = evaluation.get(
        "blank_goal", str(Path(__file__).resolve().parents[2] / "examples" / "blank_goal.png")
    )
    for episode in episodes:
        old = previous.get(episode.example_id)
        # A dry run is an intentional plumbing check, not a completed model
        # prediction.  Let the subsequent real invocation replace it without
        # requiring the more consequential --retry-failed flag.
        if old and (
            old.get("status") == "ok"
            or (old.get("status") != "dry_run" and not retry_failed)
        ):
            continue
        base = {
            "schema_version": SCHEMA_VERSION,
            "example_id": episode.example_id,
            "video_sha256": episode.video_sha256,
            "subset": episode.subset,
            "reward": episode.reward,
            "attempt": int(old.get("attempt", 0)) + 1 if old else 1,
        }
        try:
            model_payload = episode.model_payload()
            if {"reward", "gpt5_mini_check"} & model_payload.keys():
                raise AssertionError("Label leakage into Qwen model payload")
            frame_record = None
            frame_records = None
            if protocol == ROBOREWARDBENCH_NATIVE:
                payload = native_video_payload(episode, content_order=content_order)
            elif protocol in IMAGE_SEQUENCE_PROTOCOLS:
                image_paths, sampling_record = extract_uniform_image_sequence(
                    episode.video_path,
                    output_dir / "image_sequences" / episode.video_sha256,
                    count=int(evaluation.get("num_images", 8)),
                )
                payload = (
                    interleaved_image_sequence_payload(
                        episode, image_paths, sampling_record
                    )
                    if protocol
                    == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE
                    else image_sequence_payload(
                        episode,
                        image_paths,
                        sampling_record,
                        content_order=content_order,
                    )
                )
            else:
                frame_dir = output_dir / "frames" / episode.video_sha256
                frames_by_view = {
                    view: extract_endpoints(
                        episode.example_id,
                        episode.video_sha256,
                        path,
                        frame_dir / view,
                    )
                    for view, path in episode.views.items()
                }
                if {"front", "left_wrist", "right_wrist"} <= set(frames_by_view):
                    payload = multiview_endpoint_payload(
                        episode,
                        frames_by_view,
                        blank_goal,
                        prompt_mode=prompt_mode,
                    )
                    frame_record = frames_by_view["front"]
                    frame_records = {
                        view: record.to_dict()
                        for view, record in frames_by_view.items()
                    }
                else:
                    frame_record = extract_endpoints(
                        episode.example_id,
                        episode.video_sha256,
                        episode.video_path,
                        frame_dir,
                    )
                    payload = dopamine_forward_payload(
                        episode, frame_record, blank_goal, prompt_mode=prompt_mode
                    )
            if dry_run:
                raw_output = (
                    "ANSWER: 1"
                    if protocol in {
                        ROBOREWARDBENCH_NATIVE,
                        ROBOREWARDBENCH_IMAGE_SEQUENCE,
                        ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
                    }
                    else "<score>0%</score>"
                )
                parsed = parse_protocol_output(protocol, raw_output)
                diagnostics = {"dry_run": True}
                status = "dry_run"
            else:
                assert engine is not None
                raw_output, diagnostics = engine.infer(payload)
                parsed = parse_protocol_output(protocol, raw_output)
                status = "ok"
            append_jsonl(
                records_path,
                {
                    **base,
                    "task": episode.task,
                    "protocol": protocol,
                    "raw_output": raw_output,
                    "input_diagnostics": diagnostics,
                    "frame_record": frame_record.to_dict() if frame_record else None,
                    "frame_records": frame_records,
                    **parsed,
                    "status": status,
                },
            )
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    **base,
                    "protocol": protocol,
                    "status": "invalid",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    return records_path
