from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2

from ..config import section
from ..data import load_episodes, metadata_path
from ..io import (
    append_jsonl,
    artifact_fingerprint,
    latest_by_id,
    object_fingerprint,
    provenance,
    sha256_file,
    stable_shard,
    write_json,
)
from ..protocol import progress
from ..schemas import SCHEMA_VERSION
from ..video import extract_uniform_image_sequence
from .paper_protocol import PAPER_PROTOCOL_ID


# Byte-for-byte text from the local RoboReward-4B/-8B model cards, apart from
# substituting the task instruction in the final line.
ROBOREWARD_PROMPT = """Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Task: {task}"""

ANSWER_RE = re.compile(r"\bANSWER\s*:\s*([1-5])\b", flags=re.IGNORECASE)


def _model_load_kwargs(config: dict[str, Any], dtype: Any) -> dict[str, Any]:
    """Keep the historical backend default unless explicitly overridden."""
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": config.get("device_map", "auto"),
    }
    if config.get("attn_implementation"):
        kwargs["attn_implementation"] = str(config["attn_implementation"])
    return kwargs


def _validate_paper_protocol_configuration(evaluation: dict[str, Any]) -> None:
    """Reject a config that labels a custom input path as the paper protocol.

    This checks only conditions documented by the local, public reproduction
    contract.  It does not pretend to verify details of the authors'
    unreleased evaluator.
    """
    protocol = evaluation.get("evaluation_protocol")
    if protocol is None:
        return
    if protocol != PAPER_PROTOCOL_ID:
        raise ValueError(
            "roboreward_eval.evaluation_protocol must be "
            f"{PAPER_PROTOCOL_ID!r}, got {protocol!r}"
        )
    required = {
        "video_sampling_mode": "checkpoint_native_video",
        "preprocessor_mode": "checkpoint_default",
        "content_order": "text_then_video",
        "do_sample": False,
    }
    for field, expected in required.items():
        actual = evaluation.get(field)
        if actual != expected:
            raise ValueError(
                f"{PAPER_PROTOCOL_ID} requires roboreward_eval.{field}="
                f"{expected!r}, got {actual!r}"
            )
    if evaluation.get("example_ids") or evaluation.get("example_ids_file"):
        raise ValueError(
            f"{PAPER_PROTOCOL_ID} requires the complete benchmark; "
            "do not configure an example-ID cohort"
        )


def _requested_example_ids(evaluation: dict[str, Any]) -> set[str]:
    """Load one label-free, frozen ID allow-list for evaluation.

    ``example_ids_file`` mirrors the grounding pipeline's cohort contract.  It
    is intentionally only an episode-ID list: reward labels and audit details
    never enter the RoboReward model payload.
    """
    inline = evaluation.get("example_ids", [])
    path_value = evaluation.get("example_ids_file")
    if inline and path_value:
        raise ValueError("Use only one of roboreward_eval.example_ids or example_ids_file")
    if path_value:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        inline = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inline, list) or not all(isinstance(value, str) for value in inline):
        raise ValueError("RoboReward example IDs must be a JSON array of strings")
    requested = set(inline)
    if len(requested) != len(inline):
        raise ValueError("Duplicate example IDs in RoboReward allow-list")
    return requested


def native_prompt(task: str) -> str:
    return ROBOREWARD_PROMPT.format(task=task)


def parse_native_score(text: str) -> int:
    """Parse only the documented ``ANSWER: <score>`` output contract.

    Unlike the older robometer wrapper, this evaluator never turns an
    unparseable response into reward 1.  That fallback would artificially
    improve the reward=1 counterfactual baseline.
    """
    match = ANSWER_RE.search(text.strip())
    if not match:
        raise ValueError(f"No documented 'ANSWER: <1-5>' score in {text!r}")
    return int(match.group(1))


def _sample_video_at_fps(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    sample_fps: float,
) -> tuple[list[str], dict[str, Any]]:
    """Decode a complete rollout at a fixed temporal sampling rate.

    The resulting JPEG list is passed as a Qwen video, avoiding torchcodec
    dependence and matching the reference RoboReward wrapper's frame-list
    protocol. The true terminal frame is always retained.
    """
    if sample_fps <= 0:
        raise ValueError("video_sample_fps must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Video reports a non-positive FPS: {video_path}")
    stride = max(1, round(source_fps / sample_fps))
    selected: list[tuple[int, Any]] = []
    last: tuple[int, Any] | None = None
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if index % stride == 0:
            selected.append((index, frame))
        last = (index, frame)
        index += 1
    cap.release()
    if last is None:
        raise RuntimeError(f"Cannot decode any frames: {video_path}")
    if not selected or selected[-1][0] != last[0]:
        selected.append(last)
    # Qwen's video input requires at least two frames. Keep an explicit record
    # when a one-frame rollout must be duplicated.
    duplicated_single_frame = len(selected) == 1
    if duplicated_single_frame:
        selected.append(selected[0])
    paths: list[str] = []
    indices: list[int] = []
    for ordinal, (frame_index, frame) in enumerate(selected):
        path = output_dir / f"frame_{ordinal:04d}_source_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise RuntimeError(f"Cannot write sampled frame: {path}")
        paths.append(str(path.resolve()))
        indices.append(frame_index)
    return paths, {
        "video_input_protocol": "complete_rollout_1fps_frame_list_v1",
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "stride_frames": stride,
        "decoded_frame_count": index,
        "selected_source_indices": indices,
        "terminal_source_index": last[0],
        "duplicated_single_frame": duplicated_single_frame,
    }


def _sample_video_uniform_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    num_frames: int,
) -> tuple[list[str], dict[str, Any]]:
    """Select a fixed number of uniformly spaced frames, including the end.

    This is intentionally separate from the legacy 1-FPS protocol.  The
    RoboReward checkpoint's shipped video preprocessor declares ``max_frames:
    8``; selecting the frames ourselves and passing ``do_sample_frames=False``
    makes the actual visual input auditable rather than relying on a version-
    dependent video loader to downsample it.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Video reports a non-positive FPS: {video_path}")
    # Count frames by decoding rather than trusting container metadata.  A
    # second sequential pass avoids storing an entire rollout in memory.
    decoded_frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded_frame_count += 1
    cap.release()
    if decoded_frame_count == 0:
        raise RuntimeError(f"Cannot decode any frames: {video_path}")
    selected_count = min(int(num_frames), decoded_frame_count)
    if selected_count == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            round(index * (decoded_frame_count - 1) / (selected_count - 1))
            for index in range(selected_count)
        ]
    selected_set = set(selected_indices)
    cap = cv2.VideoCapture(str(video_path))
    selected: dict[int, Any] = {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if index in selected_set:
            selected[index] = frame
        index += 1
    cap.release()
    if set(selected_indices) - set(selected):
        raise RuntimeError(f"Could not re-decode selected frames: {video_path}")
    duplicated_single_frame = selected_count == 1
    output_indices = list(selected_indices)
    if duplicated_single_frame:
        output_indices.append(selected_indices[0])
    paths: list[str] = []
    for ordinal, frame_index in enumerate(output_indices):
        path = output_dir / f"frame_{ordinal:04d}_source_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(path), selected[frame_index], [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise RuntimeError(f"Cannot write sampled frame: {path}")
        paths.append(str(path.resolve()))
    return paths, {
        "video_input_protocol": "uniform_fixed_frames_v1",
        "requested_frame_count": int(num_frames),
        "source_fps": source_fps,
        "effective_sample_fps": source_fps * selected_count / decoded_frame_count,
        "decoded_frame_count": decoded_frame_count,
        "selected_source_indices": output_indices,
        "terminal_source_index": decoded_frame_count - 1,
        "duplicated_single_frame": duplicated_single_frame,
        "width": width,
        "height": height,
    }


def _use_checkpoint_native_video(video_path: str | Path) -> tuple[list[str], dict[str, Any]]:
    """Pass the original MP4 to the Qwen video loader without pre-sampling.

    This is distinct from the two frame-list protocols above.  The checkpoint
    processor decodes the MP4 and applies its own ``fps=2``, ``min_frames=4``
    and ``max_frames=8`` policy.  In particular, a short rollout need not
    have eight frames.  Keeping the MP4 intact is therefore necessary for a
    checkpoint-native reproduction attempt and makes it clear that no JPEG
    re-encoding or custom temporal grid was introduced by this evaluator.
    """
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    return [str(path)], {
        "video_input_protocol": "checkpoint_native_mp4_v1",
        "source_video_path": str(path),
        "custom_frame_extraction": False,
    }


def _native_video_message(
    task: str, video_path: str, *, content_order: str = "text_then_video"
) -> list[dict[str, Any]]:
    """Build a native-MP4 message in the explicitly requested media order.

    ``text_then_video`` reproduces the public HELM request, while
    ``video_then_text`` follows the video-inference examples referenced by the
    RoboReward model card.  Do not turn ``video_path`` into a ``file://`` URI
    here: Transformers' processor accepts a local path, while the URI is
    rejected by its native video loader.
    """
    text_item = {"type": "text", "text": native_prompt(task)}
    video_item = {"type": "video", "video": video_path}
    if content_order == "text_then_video":
        content = [text_item, video_item]
    elif content_order == "video_then_text":
        content = [video_item, text_item]
    else:
        raise ValueError(
            "content_order must be 'video_then_text' or 'text_then_video', "
            f"got {content_order!r}"
        )
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def _image_sequence_message(
    task: str,
    image_paths: list[str],
    *,
    content_order: str,
) -> list[dict[str, Any]]:
    """Build the discrete rubric request with independent image items."""
    paths = [str(Path(path).resolve()) for path in image_paths]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image-sequence inputs: {missing}")
    text = {"type": "text", "text": native_prompt(task)}
    images = [{"type": "image", "image": path} for path in paths]
    if content_order == "text_then_images":
        content = [text, *images]
    elif content_order == "images_then_text":
        content = [*images, text]
    else:
        raise ValueError(
            "image-sequence content_order must be 'text_then_images' or "
            f"'images_then_text', got {content_order!r}"
        )
    return [{"role": "user", "content": content}]


def _native_video_metadata(metadata: Any) -> dict[str, Any]:
    """Make the processor's video metadata JSON-safe for the run record."""
    if isinstance(metadata, (list, tuple)):
        if len(metadata) != 1:
            raise ValueError("expected metadata for exactly one native video")
        metadata = metadata[0]
    fields = (
        "total_num_frames",
        "fps",
        "width",
        "height",
        "duration",
        "video_backend",
        "frames_indices",
    )
    result: dict[str, Any] = {}
    for name in fields:
        value = getattr(metadata, name, None)
        if hasattr(value, "tolist"):
            value = value.tolist()
        result[name] = value
    return result


def _sample_video(
    evaluation: dict[str, Any], video_path: str | Path, output_dir: str | Path
) -> tuple[list[str], dict[str, Any]]:
    mode = str(evaluation.get("video_sampling_mode", "full_1fps"))
    if mode == "full_1fps":
        return _sample_video_at_fps(
            video_path,
            output_dir,
            sample_fps=float(evaluation.get("video_sample_fps", 1.0)),
        )
    if mode == "uniform_fixed_frames":
        return _sample_video_uniform_frames(
            video_path,
            output_dir,
            num_frames=int(evaluation.get("num_frames", 8)),
        )
    if mode == "checkpoint_native_video":
        return _use_checkpoint_native_video(video_path)
    raise ValueError(
        "video_sampling_mode must be 'full_1fps', 'uniform_fixed_frames', or "
        "'checkpoint_native_video', "
        f"got {mode!r}"
    )


class NativeRoboReward:
    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "RoboReward evaluation requires transformers, torch, and qwen-vl-utils"
            ) from exc
        self.torch = torch
        self.process_vision_info = process_vision_info
        self.preprocessor_mode = str(
            config.get("preprocessor_mode", "legacy_no_resize")
        )
        if self.preprocessor_mode not in {"legacy_no_resize", "checkpoint_default"}:
            raise ValueError(
                "preprocessor_mode must be 'legacy_no_resize' or "
                f"'checkpoint_default', got {self.preprocessor_mode!r}"
            )
        self.input_representation = str(config.get("input_representation", "video"))
        if self.input_representation not in {"video", "independent_images"}:
            raise ValueError(
                "input_representation must be 'video' or 'independent_images', "
                f"got {self.input_representation!r}"
            )
        default_order = (
            "video_then_text"
            if self.input_representation == "video"
            else "images_then_text"
        )
        self.content_order = str(config.get("content_order", default_order))
        valid_orders = (
            {"video_then_text", "text_then_video"}
            if self.input_representation == "video"
            else {"images_then_text", "text_then_images"}
        )
        if self.content_order not in valid_orders:
            raise ValueError(
                f"Invalid content_order {self.content_order!r} for "
                f"input_representation={self.input_representation!r}; "
                f"choose one of {sorted(valid_orders)}"
            )
        self.video_sampling_mode = str(config.get("video_sampling_mode", "full_1fps"))
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.video_sampling_mode != "checkpoint_native_video":
            # Existing JPEG-list modes have already selected their temporal
            # inputs, so prevent a second sampling pass by the processor.
            processor_kwargs.update(
                {
                    "do_sample_frames": False,
                    "fps": float(config.get("video_sample_fps", 1.0)),
                }
            )
        self.processor = AutoProcessor.from_pretrained(
            config["model_path"], **processor_kwargs
        )
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
        )
        self.model.eval()
        self.attention_backend = str(
            getattr(self.model.config, "_attn_implementation", "unknown")
        )
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        self.do_sample = bool(config.get("do_sample", False))
        self.temperature = float(config.get("temperature", 0.7))
        self.top_p = float(config.get("top_p", 0.8))
        self.top_k = int(config.get("top_k", 20))

    def infer(
        self, task: str, frame_paths: list[str], video_record: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        if self.input_representation == "independent_images":
            message = _image_sequence_message(
                task, frame_paths, content_order=self.content_order
            )
            inputs = self.processor.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
            grid = inputs.get("image_grid_thw")
            grid_count = int(grid.shape[0]) if grid is not None else 0
            if grid_count != len(frame_paths):
                raise RuntimeError(
                    "Independent-image processor alignment failed: "
                    f"images={len(frame_paths)}, grids={grid_count}"
                )
            input_diagnostics = {
                "input_representation": "uniform_independent_images_v1",
                "processor_native_video": False,
                "image_count": len(frame_paths),
                "image_grid_thw": grid.detach().cpu().tolist(),
                "image_token_count": int(
                    (inputs["input_ids"] == image_token_id).sum()
                ),
                "preprocessor_mode": self.preprocessor_mode,
                "content_order": self.content_order,
                "media_order": (
                    ["text", "image_sequence"]
                    if self.content_order == "text_then_images"
                    else ["image_sequence", "text"]
                ),
            }
        else:
            is_native_mp4 = (
                video_record.get("video_input_protocol") == "checkpoint_native_mp4_v1"
            )
        if self.input_representation == "video" and is_native_mp4:
            if len(frame_paths) != 1:
                raise ValueError("checkpoint_native_video requires exactly one MP4 path")
            # Let AutoProcessor decode and sample the original MP4.  This is
            # deliberately not routed through qwen-vl-utils: its decoder and
            # frame-count rounding differ from the processor's native path.
            message = _native_video_message(
                task, frame_paths[0], content_order=self.content_order
            )
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
            if not callable(converter):
                raise TypeError("native video processor cannot convert inputs to tensors")
            converted = converter(tensor_type="pt")
            if converted is not None:
                inputs = converted
            video_token_id = int(getattr(self.model.config, "video_token_id", 151656))
            grid = inputs.get("video_grid_thw")
            input_diagnostics = {
                "video_grid_thw": (
                    grid.detach().cpu().tolist() if grid is not None else None
                ),
                "video_token_count": int((inputs["input_ids"] == video_token_id).sum()),
                "preprocessor_mode": self.preprocessor_mode,
                "processor_native_video": True,
                "video_metadata": _native_video_metadata(metadata),
            }
        elif self.input_representation == "video":
            sample_fps = float(
                video_record.get(
                    "effective_sample_fps", video_record.get("sample_fps", 1.0)
                )
            )
            video_content = [f"file://{path}" for path in frame_paths]
            video_item: dict[str, Any] = {"type": "video", "video": video_content}
            video_item["sample_fps"] = sample_fps
            text_item = {"type": "text", "text": native_prompt(task)}
            content = (
                [video_item, text_item]
                if self.content_order == "video_then_text"
                else [text_item, video_item]
            )
            message = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            text = self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = self.process_vision_info(
                [message],
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if video_inputs is not None:
                videos, metadata = zip(*video_inputs)
                videos, metadata = list(videos), list(metadata)
            else:
                videos, metadata = None, None
            processor_kwargs = {
                "text": [text],
                "images": image_inputs,
                "videos": videos,
                "video_metadata": metadata,
                "padding": True,
                "return_tensors": "pt",
                **video_kwargs,
            }
            # The original exploratory 1-FPS run disabled resize.  The fixed-frame
            # protocol instead lets the checkpoint's *video* preprocessor apply its
            # own shipped resize settings.
            if self.preprocessor_mode == "legacy_no_resize":
                processor_kwargs["do_resize"] = False
            inputs = self.processor(
                **processor_kwargs
            )
            video_token_id = int(getattr(self.model.config, "video_token_id", 151656))
            grid = inputs.get("video_grid_thw")
            input_diagnostics = {
                "video_grid_thw": (
                    grid.detach().cpu().tolist() if grid is not None else None
                ),
                "video_token_count": int((inputs["input_ids"] == video_token_id).sum()),
                "preprocessor_mode": self.preprocessor_mode,
                "processor_native_video": False,
            }
        if self.input_representation == "video":
            input_diagnostics["content_order"] = self.content_order
            input_diagnostics["media_order"] = (
                ["text", "video"]
                if self.content_order == "text_then_video"
                else ["video", "text"]
            )
        input_diagnostics["attention_backend"] = self.attention_backend
        device = self.model.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            generation.update(
                {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}
            )
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated)
        ]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return text, input_diagnostics


def run(config: dict[str, Any], *, dry_run: bool = False, retry_failed: bool = False) -> Path:
    evaluation = section(config, "roboreward_eval")
    _validate_paper_protocol_configuration(evaluation)
    output_dir = Path(evaluation["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_id = int(evaluation.get("shard_id", 0))
    num_shards = int(evaluation.get("num_shards", 1))
    records_path = output_dir / f"records.shard-{shard_id:02d}.jsonl"
    sampling_mode = str(evaluation.get("video_sampling_mode", "full_1fps"))
    input_representation = str(evaluation.get("input_representation", "video"))
    if input_representation == "independent_images" and sampling_mode != "uniform_fixed_frames":
        raise ValueError(
            "independent_images requires video_sampling_mode='uniform_fixed_frames'"
        )
    content_order = str(evaluation.get("content_order", "video_then_text"))
    image_count = int(
        evaluation.get("num_images", evaluation.get("num_frames", 8))
    )
    valid_orders = (
        {"text_then_images", "images_then_text"}
        if input_representation == "independent_images"
        else {"text_then_video", "video_then_text"}
    )
    if content_order not in valid_orders:
        raise ValueError(
            f"Invalid content_order {content_order!r} for "
            f"input_representation={input_representation!r}"
        )
    requested_ids = _requested_example_ids(evaluation)
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
            "native_protocol": {
                "evaluation_protocol": evaluation.get("evaluation_protocol"),
                "prompt_sha256": hashlib.sha256(
                    ROBOREWARD_PROMPT.encode("utf-8")
                ).hexdigest(),
                "prompt_contract": "RoboReward model-card prompt; ANSWER: <1-5>",
                "input_representation": input_representation,
                "model_media_type": (
                    "independent_images"
                    if input_representation == "independent_images"
                    else "video"
                ),
                "media_order": (
                    ["text", "image_sequence"]
                    if str(evaluation.get("content_order")) == "text_then_images"
                    else ["image_sequence", "text"]
                    if input_representation == "independent_images"
                    else ["text", "video"]
                    if str(evaluation.get("content_order", "video_then_text"))
                    == "text_then_video"
                    else ["video", "text"]
                ),
                "video_input": {
                    "sampling_mode": sampling_mode,
                    "protocol": (
                        "uniform_independent_images_v1"
                        if input_representation == "independent_images"
                        else {
                            "full_1fps": "complete_rollout_1fps_frame_list_v1",
                            "uniform_fixed_frames": "uniform_fixed_frames_v1",
                            "checkpoint_native_video": "checkpoint_native_mp4_v1",
                        }[sampling_mode]
                    ),
                    "num_frames": (
                        image_count
                        if sampling_mode == "uniform_fixed_frames"
                        else None
                    ),
                    "num_images": (
                        image_count
                        if input_representation == "independent_images"
                        else None
                    ),
                },
                "preprocessor_mode": str(
                    evaluation.get("preprocessor_mode", "legacy_no_resize")
                ),
                "decoding": {
                    "do_sample": bool(evaluation.get("do_sample", False)),
                    "max_new_tokens": int(evaluation.get("max_new_tokens", 128)),
                },
                "content_order": content_order,
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
            raise ValueError(f"Unknown roboreward_eval example_ids: {sorted(missing)}")
    episodes = [
        row for row in episodes if stable_shard(row.video_sha256, num_shards) == shard_id
    ]
    limit = int(evaluation.get("limit", 0))
    if limit:
        episodes = episodes[:limit]
    engine = None if dry_run else NativeRoboReward(evaluation)
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
            # The model-facing payload is deliberately label-free.
            payload = episode.model_payload()
            if {"reward", "gpt5_mini_check"} & payload.keys():
                raise AssertionError("Label leakage into RoboReward model payload")
            if input_representation == "independent_images":
                frame_paths, video_record = extract_uniform_image_sequence(
                    payload["video_path"],
                    output_dir / "image_sequences" / episode.video_sha256,
                    count=image_count,
                )
            else:
                frame_paths, video_record = _sample_video(
                    evaluation,
                    payload["video_path"],
                    output_dir / "frames" / sampling_mode / episode.video_sha256,
                )
            if dry_run:
                raw_output, prediction, status = "ANSWER: 1", 1, "dry_run"
                input_diagnostics = {
                    "dry_run": True,
                    "preprocessor_mode": str(
                        evaluation.get("preprocessor_mode", "legacy_no_resize")
                    ),
                }
            else:
                assert engine is not None
                raw_output, input_diagnostics = engine.infer(
                    payload["task"], frame_paths, video_record
                )
                prediction = parse_native_score(raw_output)
                status = "ok"
            append_jsonl(
                records_path,
                {
                    **base,
                    "task": payload["task"],
                    "video_record": video_record,
                    "input_diagnostics": input_diagnostics,
                    "raw_output": raw_output,
                    "native_prediction": prediction,
                    "progress": progress((prediction - 1) / 4),
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
    return records_path
