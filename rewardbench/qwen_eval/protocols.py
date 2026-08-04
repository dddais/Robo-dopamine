"""Immutable input/output contracts for the Qwen3-VL baseline.

The two contracts are intentionally separate.  A direct 1--5 RoboRewardBench
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
ROBO_DOPAMINE_FORWARD = "robo_dopamine_forward"
PROTOCOLS = {ROBOREWARDBENCH_NATIVE, ROBO_DOPAMINE_FORWARD}


def validate_protocol(value: str) -> str:
    if value not in PROTOCOLS:
        choices = ", ".join(sorted(PROTOCOLS))
        raise ValueError(f"Unknown qwen_eval.protocol {value!r}; choose one of {choices}")
    return value


def protocol_descriptor(protocol: str, *, prompt_mode: str = "official") -> dict[str, Any]:
    """Return the model-facing contract that is frozen into every manifest."""
    protocol = validate_protocol(protocol)
    if protocol == ROBOREWARDBENCH_NATIVE:
        return {
            "name": protocol,
            "input": "original_mp4_text_then_video",
            "output": "ANSWER: <1-5>",
            "progress_mapping": "(answer - 1) / 4",
            "prompt_sha256": hashlib.sha256(ROBOREWARD_PROMPT.encode("utf-8")).hexdigest(),
            "prompt_contract": "RoboRewardBench discrete progress rubric",
            "media_order": ["text", "video"],
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


def native_video_payload(episode: EpisodeRecord) -> dict[str, Any]:
    """Build the benchmark-native direct-video request without labels."""
    payload = episode.model_payload()
    path = Path(payload["video_path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "protocol": ROBOREWARDBENCH_NATIVE,
        "task": payload["task"],
        "video_path": str(path),
        "media_order": ["text", "video"],
        "prompt": ROBOREWARD_PROMPT.format(task=payload["task"]),
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
    if protocol == ROBOREWARDBENCH_NATIVE:
        prediction = parse_native_score(raw_output)
        return {
            "native_prediction": prediction,
            "progress": (prediction - 1) / 4,
        }
    signed = parse_score(raw_output)
    return {"signed_score": signed, "progress": progress(signed)}
