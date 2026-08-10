"""Explicit input/output contracts for the Qwen3-VL baseline and ablations.

The contracts are intentionally separate.  A direct 1--5 RoboRewardBench
answer and a Robo-Dopamine signed progress hop are not interchangeable model
outputs, even though both can be summarized with the benchmark's ordinal
metrics after inference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..protocol import chat_messages, native_endpoint_payload, parse_score, progress
from ..roboreward_eval.runner import ROBOREWARD_PROMPT, parse_native_score
from ..schemas import EpisodeRecord, FrameRecord


ROBOREWARDBENCH_NATIVE = "roborewardbench_native"
ROBOREWARDBENCH_IMAGE_SEQUENCE = "roborewardbench_image_sequence"
ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE = (
    "roborewardbench_interleaved_image_sequence"
)
ROBO_DOPAMINE_FORWARD = "robo_dopamine_forward"
PROTOCOLS = {
    ROBOREWARDBENCH_NATIVE,
    ROBOREWARDBENCH_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
    ROBO_DOPAMINE_FORWARD,
}
DISCRETE_PROTOCOLS = {
    ROBOREWARDBENCH_NATIVE,
    ROBOREWARDBENCH_IMAGE_SEQUENCE,
    ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
}
IMAGE_SEQUENCE_PROTOCOLS = frozenset(
    {
        ROBOREWARDBENCH_IMAGE_SEQUENCE,
        ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
    }
)


INTERLEAVED_REWARD_PROMPT = """You are a rigorous, impartial vision evaluator for robot task progress.
Given the task and an ordered sequence of eight single-view observations from
one robot rollout, assign a discrete end-of-episode progress score
(1, 2, 3, 4, or 5).

<Task>
`{task}`
</Task>

<Robot Trajectory>
The following observations are ordered chronologically and belong to the same
robot rollout. Observation 1 is the initial state, and Observation 8 is the
true terminal state.

OBSERVATION 1 — INITIAL:
<image>

OBSERVATION 2:
<image>

OBSERVATION 3:
<image>

OBSERVATION 4:
<image>

OBSERVATION 5:
<image>

OBSERVATION 6:
<image>

OBSERVATION 7:
<image>

OBSERVATION 8 — TERMINAL:
<image>
</Robot Trajectory>

Rubric for end-of-episode progress:
Judge the task progress from the terminal state in Observation 8.
Use Observations 1–7 only as temporal context for understanding what changed.

1 - No Success: The terminal state shows no goal-relevant change for the command.
2 - Minimal Progress: The terminal state shows a small but insufficient change
    toward the goal.
3 - Partial Completion: The terminal state shows good progress toward the goal
    but violates more than one requirement or one major requirement.
4 - Near Completion: The terminal state is correct in region and intent but
    misses a single minor requirement.
5 - Perfect Completion: The terminal state satisfies all requirements.

Output Format (STRICT)
Return ONLY one line in exactly this format:
ANSWER: <score>

Replace <score> with exactly one integer from 1 to 5.
Do not output any explanation or additional text.
"""


def validate_protocol(value: str) -> str:
    if value not in PROTOCOLS:
        choices = ", ".join(sorted(PROTOCOLS))
        raise ValueError(f"Unknown qwen_eval.protocol {value!r}; choose one of {choices}")
    return value


def protocol_descriptor(
    protocol: str,
    *,
    prompt_mode: str = "official",
    content_order: str = "text_then_video",
) -> dict[str, Any]:
    """Return the model-facing contract that is frozen into every manifest."""
    protocol = validate_protocol(protocol)
    if protocol == ROBOREWARDBENCH_NATIVE:
        if content_order not in {"text_then_video", "video_then_text"}:
            raise ValueError(
                "content_order must be 'text_then_video' or 'video_then_text', "
                f"got {content_order!r}"
            )
        media_order = ["text", "video"] if content_order == "text_then_video" else ["video", "text"]
        return {
            "name": protocol,
            "input": f"original_mp4_{content_order}",
            "output": "ANSWER: <1-5>",
            "progress_mapping": "(answer - 1) / 4",
            "prompt_sha256": hashlib.sha256(ROBOREWARD_PROMPT.encode("utf-8")).hexdigest(),
            "prompt_contract": "RoboRewardBench discrete progress rubric",
            "content_order": content_order,
            "media_order": media_order,
        }
    if protocol == ROBOREWARDBENCH_IMAGE_SEQUENCE:
        if content_order not in {"text_then_images", "images_then_text"}:
            raise ValueError(
                "content_order must be 'text_then_images' or 'images_then_text' "
                f"for {ROBOREWARDBENCH_IMAGE_SEQUENCE}, got {content_order!r}"
            )
        media_order = (
            ["text", "image_sequence"]
            if content_order == "text_then_images"
            else ["image_sequence", "text"]
        )
        return {
            "name": protocol,
            "input": f"uniform_independent_images_{content_order}",
            "output": "ANSWER: <1-5>",
            "progress_mapping": "(answer - 1) / 4",
            "prompt_sha256": hashlib.sha256(ROBOREWARD_PROMPT.encode("utf-8")).hexdigest(),
            "prompt_contract": "RoboRewardBench discrete progress rubric",
            "content_order": content_order,
            "media_order": media_order,
            "adapter_protocol": True,
        }
    if protocol == ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE:
        if content_order != "interleaved":
            raise ValueError(
                f"{ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE} requires "
                f"content_order='interleaved', got {content_order!r}"
            )
        return {
            "name": protocol,
            "input": "uniform_independent_images_grm_interleaved",
            "output": "ANSWER: <1-5>",
            "progress_mapping": "(answer - 1) / 4",
            "prompt_sha256": hashlib.sha256(
                INTERLEAVED_REWARD_PROMPT.encode("utf-8")
            ).hexdigest(),
            "prompt_contract": (
                "shared RoboReward/Qwen chronological interleaved-image rubric"
            ),
            "content_order": content_order,
            "media_order": [
                "task_text",
                "alternating_observation_text_and_image_x8",
                "rubric_and_output_text",
            ],
            "adapter_protocol": True,
        }
    # ``native_endpoint_payload`` validates the requested named prompt.
    from ..protocol import system_prompt

    prompt = system_prompt(prompt_mode)
    return {
        "name": protocol,
        "input": "eight_image_single_view_endpoint_layout",
        "output": "<score>[+-]NN%</score>",
        "progress_mapping": "clip(signed_score, 0, 1)",
        "prompt_mode": prompt_mode,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_contract": "Robo-Dopamine forward endpoint prompt",
        "media_order": ["text_and_image_placeholders"],
    }


def native_video_payload(
    episode: EpisodeRecord, *, content_order: str = "text_then_video"
) -> dict[str, Any]:
    """Build the benchmark-native direct-video request without labels."""
    protocol_descriptor(ROBOREWARDBENCH_NATIVE, content_order=content_order)
    payload = episode.model_payload()
    path = Path(payload["video_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "protocol": ROBOREWARDBENCH_NATIVE,
        "task": payload["task"],
        "video_path": str(path),
        "content_order": content_order,
        "media_order": (
            ["text", "video"] if content_order == "text_then_video"
            else ["video", "text"]
        ),
        "prompt": ROBOREWARD_PROMPT.format(task=payload["task"]),
    }


def image_sequence_messages(
    task: str,
    image_paths: list[str] | tuple[str, ...],
    *,
    content_order: str,
) -> list[dict[str, Any]]:
    """Build the discrete rubric request with independent image items."""
    protocol_descriptor(
        ROBOREWARDBENCH_IMAGE_SEQUENCE, content_order=content_order
    )
    paths = [str(Path(path).resolve()) for path in image_paths]
    if not paths:
        raise ValueError("image sequence must contain at least one image")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image-sequence inputs: {missing}")
    text = {"type": "text", "text": ROBOREWARD_PROMPT.format(task=task)}
    images = [{"type": "image", "image": path} for path in paths]
    content = (
        [text, *images]
        if content_order == "text_then_images"
        else [*images, text]
    )
    return [{"role": "user", "content": content}]


def interleaved_image_sequence_messages(
    task: str,
    image_paths: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Insert eight chronological images at the shared GRM-style placeholders."""
    protocol_descriptor(
        ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
        content_order="interleaved",
    )
    paths = [str(Path(path).resolve()) for path in image_paths]
    if len(paths) != 8:
        raise ValueError(
            "Interleaved image-sequence protocol requires exactly eight images"
        )
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing interleaved image inputs: {missing}")
    parts = INTERLEAVED_REWARD_PROMPT.format(task=task).split("<image>")
    if len(parts) != len(paths) + 1:
        raise RuntimeError(
            "Interleaved prompt/image contract mismatch: "
            f"parts={len(parts)}, images={len(paths)}"
        )
    content: list[dict[str, Any]] = []
    for text, path in zip(parts[:-1], paths):
        content.append({"type": "text", "text": text})
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": parts[-1]})
    return [{"role": "user", "content": content}]


def image_sequence_payload(
    episode: EpisodeRecord,
    image_paths: list[str],
    sampling_record: dict[str, Any],
    *,
    content_order: str,
) -> dict[str, Any]:
    """Build a label-free sampled-image request for baseline inference."""
    protocol_descriptor(
        ROBOREWARDBENCH_IMAGE_SEQUENCE, content_order=content_order
    )
    payload = episode.model_payload()
    paths = [str(Path(path).resolve()) for path in image_paths]
    return {
        "protocol": ROBOREWARDBENCH_IMAGE_SEQUENCE,
        "task": payload["task"],
        "image": paths,
        "sampling_record": dict(sampling_record),
        "content_order": content_order,
        "media_order": (
            ["text", "image_sequence"]
            if content_order == "text_then_images"
            else ["image_sequence", "text"]
        ),
    }


def interleaved_image_sequence_payload(
    episode: EpisodeRecord,
    image_paths: list[str],
    sampling_record: dict[str, Any],
) -> dict[str, Any]:
    """Build the shared GRM-style chronological image request."""
    protocol_descriptor(
        ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
        content_order="interleaved",
    )
    payload = episode.model_payload()
    paths = [str(Path(path).resolve()) for path in image_paths]
    return {
        "protocol": ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE,
        "task": payload["task"],
        "image": paths,
        "sampling_record": dict(sampling_record),
        "content_order": "interleaved",
        "media_order": [
            "task_text",
            "alternating_observation_text_and_image_x8",
            "rubric_and_output_text",
        ],
    }


def dopamine_forward_payload(
    episode: EpisodeRecord,
    frames: FrameRecord,
    blank_goal: str | Path,
    *,
    prompt_mode: str,
) -> dict[str, Any]:
    """Build the official Robo-Dopamine single-view forward input."""
    payload = native_endpoint_payload(
        episode, frames, blank_goal, prompt_mode=prompt_mode
    )
    payload["protocol"] = ROBO_DOPAMINE_FORWARD
    return payload


def dopamine_forward_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Insert each endpoint image at its exact GRM prompt placeholder.

    The official prompt carries semantic labels immediately around its eight
    ``<image>`` placeholders.  Keeping the images in a trailing block changes
    the text/image order and is therefore a different multimodal protocol.
    """
    paths = [str(Path(path).resolve()) for path in payload["image"]]
    template = chat_messages(payload["task"], str(payload["prompt_mode"]))
    if len(template) != 1 or template[0].get("role") != "user":
        raise ValueError("Expected one user message from the Robo-Dopamine prompt")
    images = iter(paths)
    content: list[dict[str, Any]] = []
    image_count = 0
    for item in template[0]["content"]:
        if item.get("type") == "image":
            try:
                path = next(images)
            except StopIteration as exc:
                raise ValueError("Fewer endpoint images than prompt placeholders") from exc
            content.append({"type": "image", "image": path})
            image_count += 1
        else:
            content.append(dict(item))
    try:
        next(images)
    except StopIteration:
        pass
    else:
        raise ValueError("More endpoint images than prompt placeholders")
    if image_count != 8:
        raise ValueError(f"Expected eight Robo-Dopamine image placeholders, got {image_count}")
    return [{"role": "user", "content": content}]


def parse_protocol_output(protocol: str, raw_output: str) -> dict[str, float | int]:
    """Strictly parse a completed model answer into benchmark progress."""
    protocol = validate_protocol(protocol)
    if protocol in DISCRETE_PROTOCOLS:
        prediction = parse_native_score(raw_output)
        return {
            "native_prediction": prediction,
            "progress": (prediction - 1) / 4,
        }
    signed = parse_score(raw_output)
    return {"signed_score": signed, "progress": progress(signed)}
