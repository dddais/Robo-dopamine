"""Cross-model attention ranking and steering for Qwen3-VL checkpoints.

The module deliberately supports only the two frozen Qwen input contracts used
by :mod:`rewardbench.qwen_eval`: direct RoboRewardBench MP4 input and the
eight-image Robo-Dopamine forward prompt.  RoboReward-8B uses the same Qwen3-VL
backbone and calls this engine through its own thin entry point, while keeping a
separate model/configuration provenance trail.

Ranking follows the GRM protocol: rank query heads by raw attention mass from
the last prompt token to an audited target bbox, then steer selected heads by an
additive pre-softmax target-vs-other-visual-token bias.  Excess mass and visual
enrichment are retained as robustness rankings.  Labels are never read by this
runtime; score-time metrics join them only after inference.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from ..attention_eval.masking import (
    QUERY_SCOPES,
    Head,
    ImageSpan,
    bbox_to_token_positions,
    make_attention_mask_hook,
    matched_wrong_position_set,
    resolve_negative_positions,
)
from ..attention_eval.runtime import find_contiguous_spans
from ..protocol import IMAGE_LABELS, progress
from ..roboreward_eval.runner import ROBOREWARD_PROMPT, parse_native_score
from .protocols import (
    DISCRETE_PROTOCOLS,
    ROBO_DOPAMINE_FORWARD,
    ROBOREWARDBENCH_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_NATIVE,
    dopamine_forward_messages,
    image_sequence_messages,
    interleaved_image_sequence_messages,
    parse_protocol_output,
    validate_protocol,
)

TEMPORAL_SPAN_MODES = frozenset({"native_pairs", "duplicate_frames"})
TEMPORAL_BBOX_REDUCERS = frozenset({"last", "union", "intersection"})


def _native_video_message(
    task: str, video_input: Any, *, content_order: str = "text_then_video"
) -> list[dict[str, Any]]:
    """Build a native request in the same explicit order as its baseline."""
    text = {"type": "text", "text": ROBOREWARD_PROMPT.format(task=task)}
    resolved_video = (
        str(Path(video_input).resolve())
        if isinstance(video_input, (str, Path))
        else video_input
    )
    video = {"type": "video", "video": resolved_video}
    if content_order == "text_then_video":
        content = [text, video]
    elif content_order == "video_then_text":
        content = [video, text]
    else:
        raise ValueError("Unknown native attention content_order")
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def duplicate_temporal_frames(frames: Any, repeat: int = 2) -> Any:
    """Repeat every decoded source frame consecutively along the time axis."""
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if hasattr(frames, "repeat_interleave"):
        return frames.repeat_interleave(repeat, dim=0)
    if isinstance(frames, np.ndarray):
        return np.repeat(frames, repeat, axis=0)
    if isinstance(frames, (list, tuple)):
        return [frame for frame in frames for _ in range(repeat)]
    raise TypeError(f"Unsupported decoded video type {type(frames)!r}")


def reduce_temporal_bboxes(
    bboxes: Sequence[Sequence[float]], reducer: str
) -> list[float]:
    """Reduce tracked boxes belonging to one temporal tubelet."""
    mode = str(reducer)
    if mode not in TEMPORAL_BBOX_REDUCERS:
        choices = ", ".join(sorted(TEMPORAL_BBOX_REDUCERS))
        raise ValueError(f"Unknown temporal_bbox_reduce {mode!r}; choose one of {choices}")
    values = [list(map(float, bbox)) for bbox in bboxes]
    if not values or any(len(bbox) != 4 for bbox in values):
        raise ValueError("Temporal bbox reduction requires one or more xyxy boxes")
    if mode == "last":
        result = values[-1]
    elif mode == "union":
        result = [
            min(bbox[0] for bbox in values),
            min(bbox[1] for bbox in values),
            max(bbox[2] for bbox in values),
            max(bbox[3] for bbox in values),
        ]
    else:
        result = [
            max(bbox[0] for bbox in values),
            max(bbox[1] for bbox in values),
            min(bbox[2] for bbox in values),
            min(bbox[3] for bbox in values),
        ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Temporal bbox intersection is empty")
    return result


def prepare_native_video_processor_input(
    processor: Any,
    task: str,
    video_path: str | Path,
    *,
    content_order: str,
    temporal_span_mode: str,
) -> tuple[Any, dict[str, Any]]:
    """Apply native processing with an auditable temporal-span contract."""
    mode = str(temporal_span_mode)
    if mode not in TEMPORAL_SPAN_MODES:
        choices = ", ".join(sorted(TEMPORAL_SPAN_MODES))
        raise ValueError(f"Unknown temporal_span_mode {mode!r}; choose one of {choices}")
    resolved = str(Path(video_path).resolve())
    if mode == "native_pairs":
        raw = processor.apply_chat_template(
            _native_video_message(task, resolved, content_order=content_order),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_metadata=True,
        )
        return raw, {"temporal_span_mode": mode}

    video_processor = processor.video_processor
    temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 2))
    if temporal_patch_size != 2:
        raise RuntimeError(
            "duplicate_frames currently requires the checkpoint's temporal_patch_size=2"
        )
    frames, source_metadata = video_processor.fetch_videos(
        resolved,
        sample_indices_fn=video_processor.sample_frames,
    )
    source_indices_value = getattr(source_metadata, "frames_indices", None)
    source_indices = (
        source_indices_value.tolist()
        if hasattr(source_indices_value, "tolist")
        else list(source_indices_value or [])
    )
    if not source_indices or len(source_indices) != len(frames):
        raise RuntimeError("Native source-frame sampling metadata is incomplete")
    duplicated_indices = [
        int(index)
        for index in source_indices
        for _ in range(temporal_patch_size)
    ]
    duplicated_frames = duplicate_temporal_frames(frames, temporal_patch_size)
    metadata = {
        field: getattr(source_metadata, field, None)
        for field in (
            "total_num_frames",
            "fps",
            "width",
            "height",
            "duration",
            "video_backend",
        )
    }
    # Keep original-video indices so duplicated tubelets retain timestamps.
    metadata["frames_indices"] = duplicated_indices
    native_size = getattr(video_processor, "size", None)
    shortest_edge = (
        getattr(native_size, "shortest_edge", None)
        if native_size is not None
        else None
    )
    longest_edge = (
        getattr(native_size, "longest_edge", None)
        if native_size is not None
        else None
    )
    if isinstance(native_size, dict):
        shortest_edge = native_size.get("shortest_edge", shortest_edge)
        longest_edge = native_size.get("longest_edge", longest_edge)
    if shortest_edge is None or longest_edge is None:
        raise RuntimeError("Video processor did not expose its native pixel budget")
    # Qwen3-VL's smart_resize applies a total-video pixel budget. Doubling T
    # would otherwise reduce H/W, confounding temporal and spatial ablations.
    duplicated_size = {
        "shortest_edge": int(shortest_edge) * temporal_patch_size,
        "longest_edge": int(longest_edge) * temporal_patch_size,
    }
    raw = processor.apply_chat_template(
        _native_video_message(task, duplicated_frames, content_order=content_order),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_metadata=True,
        do_sample_frames=False,
        video_metadata=metadata,
        size=duplicated_size,
    )
    return raw, {
        "temporal_span_mode": mode,
        "source_sample_frame_indices": [int(value) for value in source_indices],
        "processor_input_source_frame_indices": duplicated_indices,
        "source_sample_frame_count": len(source_indices),
        "processor_input_frame_count": len(duplicated_indices),
        "temporal_duplication_factor": temporal_patch_size,
        "duplicated_video_pixel_budget": duplicated_size,
    }


def _move_inputs(torch, model, inputs: Any, dtype: Any) -> dict[str, Any]:
    device = model.device
    result = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            result[key] = value
        elif key in {"pixel_values", "pixel_values_videos"}:
            result[key] = value.to(device=device, dtype=dtype)
        else:
            result[key] = value.to(device=device)
    return result


def _spatial_merge_size(config: Any) -> int:
    value = getattr(config, "spatial_merge_size", None)
    if value is None:
        value = getattr(getattr(config, "vision_config", None), "spatial_merge_size", None)
    return int(value if value is not None else 2)


@dataclass(frozen=True)
class PreparedAttentionInput:
    inputs: dict[str, Any]
    spans: list[ImageSpan]
    target_span: ImageSpan
    target_image_path: str
    visual_positions: list[int]
    protocol: str
    video_metadata: dict[str, Any] | None = None


def forward_image_paths(first: str | Path, last: str | Path, blank: str | Path) -> list[str]:
    """Return the eight images in the canonical Robo-Dopamine slot order."""
    first_path = str(Path(first).resolve())
    last_path = str(Path(last).resolve())
    blank_path = str(Path(blank).resolve())
    return [
        first_path,
        blank_path,
        first_path,
        first_path,
        first_path,
        last_path,
        last_path,
        last_path,
    ]


def build_forward_image_spans(
    paths: Sequence[str],
    spans: Sequence[tuple[int, int]],
    grids: Sequence[tuple[int, int, int]],
) -> list[ImageSpan]:
    """Bind processor spans to the frozen eight-image semantic contract."""
    expected = len(IMAGE_LABELS)
    if len(paths) != expected or len(spans) != expected or len(grids) != expected:
        raise ValueError(
            "Forward eight-image alignment requires exactly "
            f"{expected} paths, spans, and grids"
        )
    return [
        ImageSpan(label, str(path), int(start), int(end), tuple(int(v) for v in grid))
        for label, path, (start, end), grid in zip(IMAGE_LABELS, paths, spans, grids)
    ]


class QwenAttentionRuntime:
    """Eager-attention runtime for native and forward Qwen3-VL contracts."""

    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - environment failure
            raise RuntimeError("attention steering requires torch and transformers") from exc
        self.torch = torch
        self.config = dict(config)
        self.protocol = validate_protocol(str(config["protocol"]))
        if self.protocol == ROBOREWARDBENCH_IMAGE_SEQUENCE:
            default_order = "text_then_images"
        elif self.protocol == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE:
            default_order = "interleaved"
        else:
            default_order = "text_then_video"
        self.content_order = str(config.get("content_order", default_order))
        if self.protocol == ROBOREWARDBENCH_IMAGE_SEQUENCE:
            valid_orders = {"text_then_images", "images_then_text"}
        elif self.protocol == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE:
            valid_orders = {"interleaved"}
        else:
            valid_orders = {"text_then_video", "video_then_text"}
        if self.protocol != ROBO_DOPAMINE_FORWARD and self.content_order not in valid_orders:
            raise ValueError(
                f"Unknown attention content_order {self.content_order!r} for "
                f"protocol {self.protocol!r}"
            )
        self.temporal_scope = str(config.get("temporal_intervention_scope", "last_frame"))
        if self.temporal_scope not in {"last_frame", "all_frames"}:
            raise ValueError("temporal_intervention_scope must be last_frame or all_frames")
        self.temporal_span_mode = str(config.get("temporal_span_mode", "native_pairs"))
        if self.temporal_span_mode not in TEMPORAL_SPAN_MODES:
            choices = ", ".join(sorted(TEMPORAL_SPAN_MODES))
            raise ValueError(
                f"Unknown temporal_span_mode {self.temporal_span_mode!r}; "
                f"choose one of {choices}"
            )
        self.temporal_bbox_reduce = str(config.get("temporal_bbox_reduce", "last"))
        if self.temporal_bbox_reduce not in TEMPORAL_BBOX_REDUCERS:
            choices = ", ".join(sorted(TEMPORAL_BBOX_REDUCERS))
            raise ValueError(
                f"Unknown temporal_bbox_reduce {self.temporal_bbox_reduce!r}; "
                f"choose one of {choices}"
            )
        if (
            self.protocol != ROBOREWARDBENCH_NATIVE
            and self.temporal_span_mode != "native_pairs"
        ):
            raise ValueError("temporal_span_mode applies only to native video")
        dtype_name = str(config.get("torch_dtype", config.get("dtype", "bfloat16")))
        try:
            self.dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown attention torch dtype {dtype_name!r}") from exc
        self.processor = AutoProcessor.from_pretrained(
            config["model_path"], trust_remote_code=True
        )
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            if "min_pixels" in config:
                image_processor.min_pixels = int(config["min_pixels"])
            if "max_pixels" in config:
                image_processor.max_pixels = int(config["max_pixels"])
        # Full Qwen evaluation permits its checkpoint default of 768 frames.
        # Attention tensors scale quadratically in sequence length, therefore
        # attention runs explicitly freeze a smaller native-video cap.
        if self.protocol == ROBOREWARDBENCH_NATIVE:
            cap = config.get("attention_video_max_frames")
            if cap is not None:
                cap = int(cap)
                if cap < 2:
                    raise ValueError("attention_video_max_frames must be >= 2")
                self.processor.video_processor.max_frames = cap
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["model_path"],
            torch_dtype=self.dtype,
            device_map=config.get("device_map", "auto"),
            attn_implementation="eager",
        ).eval()
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.num_layers = int(text_config.num_hidden_layers)
        self.num_heads = int(text_config.num_attention_heads)
        self.merge_size = _spatial_merge_size(self.model.config)

    @property
    def layers(self):
        candidates = (
            lambda: self.model.model.language_model.layers,
            lambda: self.model.language_model.layers,
            lambda: self.model.model.layers,
        )
        for getter in candidates:
            try:
                return getter()
            except AttributeError:
                continue
        raise RuntimeError("Cannot locate Qwen decoder layers for attention hooks")

    def _forward_paths(self, sample: dict[str, Any]) -> list[str]:
        configured = sample.get("image_paths")
        if configured is not None:
            if not isinstance(configured, list) or len(configured) != len(IMAGE_LABELS):
                raise ValueError("image_paths must contain the canonical eight images")
            paths = [str(Path(path).resolve()) for path in configured]
        else:
            paths = forward_image_paths(
                sample["first_image_path"],
                sample["last_image_path"],
                self.config["blank_goal"],
            )
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing forward attention images: {missing}")
        return paths

    def _prepare_forward(self, sample: dict[str, Any]) -> PreparedAttentionInput:
        paths = self._forward_paths(sample)
        message = dopamine_forward_messages(
            {
                "task": sample["task"],
                "prompt_mode": str(self.config.get("prompt_mode", "official")),
                "image": paths,
            }
        )
        inputs = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        token_id = int(getattr(self.model.config, "image_token_id", 151655))
        spans = find_contiguous_spans(inputs["input_ids"][0].tolist(), token_id)
        grids = inputs.get("image_grid_thw")
        if grids is None or len(spans) != 8 or int(grids.shape[0]) != 8:
            raise RuntimeError(
                "Forward eight-image token alignment failed: "
                f"spans={len(spans)}, grids={None if grids is None else int(grids.shape[0])}"
            )
        grid_rows = [tuple(int(value) for value in row) for row in grids.tolist()]
        image_spans = build_forward_image_spans(paths, spans, grid_rows)
        target = next(span for span in image_spans if span.label == "after_cam_high")
        expected_target_path = str(Path(sample["last_image_path"]).resolve())
        if target.path != expected_target_path:
            raise RuntimeError(
                "Forward target slot is not bound to the terminal cam_high image"
            )
        visual = [position for span in image_spans for position in range(span.start, span.end)]
        return PreparedAttentionInput(
            inputs=_move_inputs(self.torch, self.model, inputs, self.dtype),
            spans=image_spans,
            target_span=target,
            target_image_path=str(Path(sample["last_image_path"]).resolve()),
            visual_positions=visual,
            protocol=self.protocol,
        )

    def _prepare_image_sequence(
        self, sample: dict[str, Any]
    ) -> PreparedAttentionInput:
        paths = [str(Path(path).resolve()) for path in sample["image_paths"]]
        if not paths:
            raise ValueError("Image-sequence attention input is empty")
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing image-sequence inputs: {missing}")
        messages = (
            interleaved_image_sequence_messages(sample["task"], paths)
            if self.protocol == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE
            else image_sequence_messages(
                sample["task"], paths, content_order=self.content_order
            )
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        token_id = int(getattr(self.model.config, "image_token_id", 151655))
        token_spans = find_contiguous_spans(inputs["input_ids"][0].tolist(), token_id)
        grids = inputs.get("image_grid_thw")
        grid_count = int(grids.shape[0]) if grids is not None else 0
        if len(token_spans) != len(paths) or grid_count != len(paths):
            raise RuntimeError(
                "Independent-image token alignment failed: "
                f"images={len(paths)}, spans={len(token_spans)}, grids={grid_count}"
            )
        image_spans = [
            ImageSpan(
                f"image_t{index}",
                path,
                int(start),
                int(end),
                tuple(int(value) for value in grid),
            )
            for index, (path, (start, end), grid) in enumerate(
                zip(paths, token_spans, grids.tolist())
            )
        ]
        source_indices = [int(value) for value in sample["image_source_indices"]]
        if len(source_indices) != len(image_spans):
            raise RuntimeError("Image spans and sampled source indices do not align")
        sampling = dict(sample.get("image_sampling_record", {}))
        terminal = sampling.get("terminal_source_index")
        if not source_indices or terminal is None or source_indices[-1] != int(terminal):
            raise RuntimeError("Last independent image is not the source terminal frame")
        sampling.update(
            {
                "span_source_frame_indices": [[index] for index in source_indices],
                "target_source_frame_indices": [source_indices[-1]],
                "target_image_span": image_spans[-1].label,
                "independent_image_spans": True,
            }
        )
        visual = [
            position
            for span in image_spans
            for position in range(span.start, span.end)
        ]
        return PreparedAttentionInput(
            inputs=_move_inputs(self.torch, self.model, inputs, self.dtype),
            spans=image_spans,
            target_span=image_spans[-1],
            target_image_path=paths[-1],
            visual_positions=visual,
            protocol=self.protocol,
            video_metadata=sampling,
        )

    def _prepare_native(self, sample: dict[str, Any]) -> PreparedAttentionInput:
        video_path = str(Path(sample["video_path"]).resolve())
        if not Path(video_path).is_file():
            raise FileNotFoundError(video_path)
        raw, temporal_record = prepare_native_video_processor_input(
            self.processor,
            sample["task"],
            video_path,
            content_order=self.content_order,
            temporal_span_mode=self.temporal_span_mode,
        )
        metadata = raw.pop("video_metadata", None)
        converter = getattr(raw, "convert_to_tensors", None)
        if callable(converter):
            converted = converter(tensor_type="pt")
            if converted is not None:
                raw = converted
        token_id = int(getattr(self.model.config, "video_token_id", 151656))
        spans = find_contiguous_spans(raw["input_ids"][0].tolist(), token_id)
        grids = raw.get("video_grid_thw")
        if grids is None or int(grids.shape[0]) != 1:
            raise RuntimeError("Expected exactly one native video grid")
        temporal, height, width = (int(value) for value in grids[0].tolist())
        if len(spans) != temporal:
            raise RuntimeError(
                "Native video token alignment failed: "
                f"token_spans={len(spans)}, grid_temporal={temporal}"
            )
        image_spans = [
            ImageSpan(f"video_t{index}", video_path, start, end, (1, height, width))
            for index, (start, end) in enumerate(spans)
        ]
        visual = [position for span in image_spans for position in range(span.start, span.end)]
        metadata_value = metadata[0] if isinstance(metadata, (list, tuple)) and len(metadata) == 1 else metadata
        record = dict(temporal_record)
        for field in ("total_num_frames", "fps", "frames_indices", "width", "height", "duration", "video_backend"):
            value = getattr(metadata_value, field, None)
            record[field] = value.tolist() if hasattr(value, "tolist") else value
        frame_indices = record.get("frames_indices")
        total_frames = record.get("total_num_frames")
        if not isinstance(frame_indices, list) or not frame_indices:
            raise RuntimeError("Native video processor did not report sampled frame indices")
        if not isinstance(total_frames, int) or total_frames < 1:
            raise RuntimeError("Native video processor did not report total frame count")
        if int(frame_indices[-1]) != total_frames - 1:
            raise RuntimeError(
                "Native video sampling omitted the terminal frame; cannot align endpoint bbox"
            )
        record.setdefault(
            "processor_input_source_frame_indices",
            [int(value) for value in frame_indices],
        )
        record.setdefault("processor_input_frame_count", len(frame_indices))
        record.setdefault(
            "source_sample_frame_indices",
            [int(value) for value in frame_indices],
        )
        record.setdefault("source_sample_frame_count", len(frame_indices))
        record["target_video_span"] = image_spans[-1].label
        if len(frame_indices) % temporal == 0:
            frames_per_span = len(frame_indices) // temporal
            span_sources = [
                frame_indices[index * frames_per_span : (index + 1) * frames_per_span]
                for index in range(temporal)
            ]
            record["span_source_frame_indices"] = span_sources
            record["target_source_frame_indices"] = span_sources[-1]
            record["target_span_alignment"] = "terminal_merged_time_group"
        else:
            # Some processors pad/merge a final temporal group internally.
            # The reported terminal index still proves that the final token
            # span is the only endpoint-aligned span; do not infer a false
            # one-to-one frame grouping from a non-divisible count.
            record["span_source_frame_indices"] = (
                [None] * (temporal - 1) + [[frame_indices[-1]]]
            )
            record["target_source_frame_indices"] = [frame_indices[-1]]
            record["target_span_alignment"] = "terminal_in_final_padded_or_merged_group"
        return PreparedAttentionInput(
            inputs=_move_inputs(self.torch, self.model, raw, self.dtype),
            spans=image_spans,
            target_span=image_spans[-1],
            target_image_path=str(Path(sample["last_image_path"]).resolve()),
            visual_positions=visual,
            protocol=self.protocol,
            video_metadata=record,
        )

    def prepare(self, sample: dict[str, Any]) -> PreparedAttentionInput:
        if self.protocol == ROBO_DOPAMINE_FORWARD:
            return self._prepare_forward(sample)
        if self.protocol in {
            ROBOREWARDBENCH_IMAGE_SEQUENCE,
            ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
        }:
            return self._prepare_image_sequence(sample)
        return self._prepare_native(sample)

    def target_positions(self, sample: dict[str, Any], prepared: PreparedAttentionInput) -> list[int]:
        with Image.open(prepared.target_image_path) as image:
            size = image.size
        if prepared.protocol == ROBO_DOPAMINE_FORWARD:
            selected = [(prepared.target_span, sample["last_bbox"], None)]
        elif prepared.protocol in {
            ROBOREWARDBENCH_IMAGE_SEQUENCE,
            ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
        }:
            metadata = prepared.video_metadata or {}
            source_indices = [
                int(values[0])
                for values in metadata.get("span_source_frame_indices", [])
                if isinstance(values, list) and len(values) == 1
            ]
            if len(source_indices) != len(prepared.spans):
                raise RuntimeError(
                    "Sampled source indices cannot be mapped to image spans"
                )
            chosen = (
                [(prepared.spans[-1], source_indices[-1])]
                if self.temporal_scope == "last_frame"
                else list(zip(prepared.spans, source_indices))
            )
            by_index = {}
            if self.temporal_scope == "all_frames":
                track = json.loads(
                    Path(sample["tracking_path"]).read_text(encoding="utf-8")
                )
                by_index = {
                    int(row["frame_index"]): row
                    for row in track.get("frames", [])
                    if isinstance(row.get("bbox"), list) and len(row["bbox"]) == 4
                }
                if not by_index:
                    raise ValueError("track.json contains no valid bboxes")
            selected = []
            alignment = []
            for span, source_index in chosen:
                if self.temporal_scope == "last_frame":
                    bbox = [float(value) for value in sample["last_bbox"]]
                    track_index = source_index
                else:
                    track_index = min(
                        by_index, key=lambda value: abs(value - source_index)
                    )
                    bbox = [float(value) for value in by_index[track_index]["bbox"]]
                selected.append((span, bbox, source_index))
                alignment.append(
                    {
                        "span": span.label,
                        "source_frame_indices": [source_index],
                        "tracking_frame_indices": [track_index],
                        "applied_bbox": bbox,
                    }
                )
            metadata["intervention_span_tracking_alignment"] = alignment
        else:
            metadata = prepared.video_metadata or {}
            span_sources = metadata.get("span_source_frame_indices")
            if (
                not isinstance(span_sources, list)
                or len(span_sources) != len(prepared.spans)
                or any(not isinstance(indices, list) or not indices for indices in span_sources)
            ):
                raise RuntimeError(
                    "Processor frame indices cannot be mapped to every native video span"
                )
            chosen = (
                [(prepared.spans[-1], span_sources[-1])]
                if self.temporal_scope == "last_frame"
                else list(zip(prepared.spans, span_sources))
            )
            need_tracking = (
                self.temporal_scope == "all_frames"
                or self.temporal_bbox_reduce != "last"
            )
            by_index = {}
            if need_tracking:
                track = json.loads(
                    Path(sample["tracking_path"]).read_text(encoding="utf-8")
                )
                rows = track.get("frames")
                if not isinstance(rows, list) or not rows:
                    raise ValueError("track.json contains no frames")
                by_index = {
                    int(row["frame_index"]): row
                    for row in rows
                    if isinstance(row.get("bbox"), list) and len(row["bbox"]) == 4
                }
                if not by_index:
                    raise ValueError("track.json contains no valid bboxes")
            selected = []
            alignment = []
            for span, source_indices in chosen:
                source_values = [int(value) for value in source_indices]
                if (
                    self.temporal_scope == "last_frame"
                    and self.temporal_bbox_reduce == "last"
                ):
                    bbox = [float(value) for value in sample["last_bbox"]]
                    track_indices = [source_values[-1]]
                else:
                    track_indices = [
                        min(by_index, key=lambda value: abs(value - source_index))
                        for source_index in source_values
                    ]
                    bbox = reduce_temporal_bboxes(
                        [by_index[index]["bbox"] for index in track_indices],
                        self.temporal_bbox_reduce,
                    )
                selected.append((span, bbox, source_values[-1]))
                alignment.append(
                    {
                        "span": span.label,
                        "source_frame_indices": source_values,
                        "tracking_frame_indices": track_indices,
                        "bbox_reduce": self.temporal_bbox_reduce,
                        "applied_bbox": bbox,
                    }
                )
            metadata["intervention_span_tracking_alignment"] = alignment
        positions = []
        for span, bbox, _source_index in selected:
            positions.extend(
                bbox_to_token_positions(span, bbox, size, self.merge_size)
            )
        positions = sorted(set(positions))
        if not positions:
            raise ValueError("Target bbox did not map to any visual tokens")
        if prepared.video_metadata is not None:
            prepared.video_metadata["temporal_intervention_scope"] = self.temporal_scope
            prepared.video_metadata["temporal_bbox_reduce"] = self.temporal_bbox_reduce
            prepared.video_metadata["selected_target_span_labels"] = [
                span.label for span, _bbox, _source_index in selected
            ]
        return positions

    def wrong_control_positions(
        self, prepared: PreparedAttentionInput, target_positions: Sequence[int]
    ) -> tuple[list[int], str]:
        """Return an equal-size, disjoint visual control with explicit provenance.

        Prefer GRM's same-image far-spatial-region control.  When a coarse
        visual grid makes the audited box cover that whole plane, use a
        different equal-grid image/time plane; this preserves cardinality and
        disjointness instead of silently dropping the frozen cohort sample.
        """
        per_span = []
        for span in prepared.spans:
            positions = [
                int(position)
                for position in target_positions
                if span.start <= int(position) < span.end
            ]
            if not positions:
                continue
            wrong = matched_wrong_position_set(
                span, positions, spatial_merge_size=self.merge_size
            )
            if wrong is None:
                per_span = []
                break
            per_span.extend(wrong)
        if len(per_span) == len(target_positions):
            return sorted(per_span), "matched_farthest_region_in_each_selected_span"
        other = sorted(set(prepared.visual_positions) - set(target_positions))
        if len(other) >= len(target_positions):
            return other[: len(target_positions)], "other_visual_tokens_fallback"
        raise RuntimeError("No equal-size non-overlapping visual control region")

    def collect_mass(self, sample: dict[str, Any]) -> dict[str, Any]:
        prepared = self.prepare(sample)
        target = self.target_positions(sample, prepared)
        query = int(prepared.inputs["input_ids"].shape[1] - 1)
        raw = np.zeros((self.num_layers, self.num_heads), dtype=np.float64)
        visual = np.zeros_like(raw)
        observed: set[int] = set()
        handles = []

        def collector(layer: int):
            def hook(_module, _args, output):
                if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
                    raise RuntimeError(
                        f"Qwen eager attention layer {layer} did not expose weights"
                    )
                weights = output[1]
                matrix = weights[0, :, query, :]
                raw[layer] = (
                    matrix[:, target].sum(dim=-1).detach().float().cpu().numpy()
                )
                visual[layer] = (
                    matrix[:, prepared.visual_positions]
                    .sum(dim=-1)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                observed.add(layer)
                # The decoder layer discards attention weights.  Removing them
                # here prevents one quadratic tensor per layer being retained
                # by version-dependent model-output plumbing.
                return output[0], None

            return hook

        try:
            for layer, decoder in enumerate(self.layers):
                handles.append(decoder.self_attn.register_forward_hook(collector(layer)))
            with self.torch.inference_mode():
                self.model(**prepared.inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if observed != set(range(self.num_layers)):
            missing = sorted(set(range(self.num_layers)) - observed)
            raise RuntimeError(f"Missing Qwen attention observations for layers {missing}")
        fraction = len(target) / len(prepared.visual_positions)
        enrichment = np.divide(
            raw,
            visual,
            out=np.zeros_like(raw),
            where=visual > 0,
        ) - fraction
        return {
            "example_id": sample["example_id"],
            "raw_mass": raw.tolist(),
            "image_mass": visual.tolist(),
            "excess_mass": (raw - fraction * visual).tolist(),
            "visual_enrichment": enrichment.tolist(),
            "bbox_token_fraction": fraction,
            "bbox_positions": target,
            "visual_positions": prepared.visual_positions,
            "target_span": prepared.target_span.__dict__,
            "video_metadata": prepared.video_metadata,
            "status": "ok",
        }

    @contextmanager
    def steering_hooks(
        self,
        heads: Sequence[Head],
        selected_positions: Sequence[int],
        visual_positions: Sequence[int],
        bias: float,
        query_scope: str,
        negative_scope: str,
        spans: Sequence[ImageSpan],
        diagnostics: dict[str, Any],
    ):
        if query_scope not in QUERY_SCOPES:
            raise ValueError(f"Unknown query scope {query_scope!r}")
        grouped: dict[int, list[int]] = {}
        for head in heads:
            grouped.setdefault(int(head.layer), []).append(int(head.head))
        handles = []
        other, selected_span_labels = resolve_negative_positions(
            spans, selected_positions, negative_scope
        )
        diagnostics.update(
            {
                "negative_scope": negative_scope,
                "selected_span_labels": selected_span_labels,
                "all_visual_span_labels": [span.label for span in spans],
                "selected_token_count": len(set(selected_positions)),
                "negative_token_count": len(other),
                "selected_negative_disjoint": not bool(
                    set(selected_positions) & set(other)
                ),
            }
        )
        try:
            for layer, layer_heads in sorted(grouped.items()):
                layer_diagnostics: dict[str, Any] = {}
                hook = make_attention_mask_hook(
                    layer_heads,
                    selected_positions,
                    other,
                    self.num_heads,
                    bias,
                    layer_diagnostics,
                    query_scope=query_scope,
                )
                handles.append(
                    self.layers[layer].self_attn.register_forward_pre_hook(
                        hook, with_kwargs=True
                    )
                )
                diagnostics.setdefault("per_layer", {})[str(layer)] = layer_diagnostics
            yield
        finally:
            for handle in handles:
                handle.remove()

    def generate(
        self,
        sample: dict[str, Any],
        *,
        prepared: PreparedAttentionInput | None = None,
        heads: Sequence[Head] = (),
        selected_positions: Sequence[int] = (),
        visual_positions: Sequence[int] = (),
        bias: float = 0.0,
        query_scope: str = "last_prompt",
        negative_scope: str | None = None,
    ) -> dict[str, Any]:
        prepared = prepared or self.prepare(sample)
        if not selected_positions:
            selected_positions = self.target_positions(sample, prepared)
        if not visual_positions:
            visual_positions = prepared.visual_positions
        diagnostics: dict[str, Any] = {
            "protocol": self.protocol,
            "query_scope": query_scope,
            "hook_active": bool(heads and bias != 0),
            "video_metadata": prepared.video_metadata,
        }
        negative = str(
            negative_scope
            if negative_scope is not None
            else self.config.get("negative_scope", "other_spans")
        )
        context = (
            self.steering_hooks(
                heads,
                selected_positions,
                visual_positions,
                bias,
                query_scope,
                negative,
                prepared.spans,
                diagnostics,
            )
            if heads and bias != 0
            else _nullcontext()
        )
        with context:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **prepared.inputs,
                    max_new_tokens=int(self.config.get("max_new_tokens", 32)),
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                    # Steering is applied by pre-forward hooks; retaining every
                    # generation attention matrix would only inflate memory and
                    # is not required for its diagnostics.
                    output_attentions=False,
                    return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
        sequence = generated.sequences[0, prepared.inputs["input_ids"].shape[1] :]
        raw = self.processor.tokenizer.decode(sequence, skip_special_tokens=True).strip()
        parsed = parse_protocol_output(self.protocol, raw)
        result = {
            "raw_output": raw,
            "hook_diagnostics": diagnostics,
            "target_positions": list(selected_positions),
            "visual_positions": list(visual_positions),
            **parsed,
        }
        if self.protocol in DISCRETE_PROTOCOLS:
            # Keep an explicit parsed field for metric code and make failures
            # impossible to silently coerce into a low score.
            result["native_prediction"] = parse_native_score(raw)
        return result


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False
