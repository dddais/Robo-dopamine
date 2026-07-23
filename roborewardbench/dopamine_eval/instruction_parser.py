"""Local-LLM instruction parsing for target-object grounding.

The parser extracts the entity that the robot is directly commanded to act on.
Destination and reference entities are retained separately so they cannot be
mistaken for the steering target.  Model output is validated and normalized
before it enters GroundingDINO.
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TARGET_TYPES = ("object", "object_part", "robot_part", "region", "ambiguous")

SYSTEM_PROMPT = r"""
You are a strict information extraction component for robot manipulation tasks.

Extract the entity that the robot is DIRECTLY commanded to manipulate, move,
grasp, push, pull, open, close, rotate, or otherwise act on.

Rules:
1. Do not select a destination, supporting surface, receptacle, landmark, or
   reference object unless the instruction directly commands acting on it.
2. If the instruction directly targets a part, target_phrase and target_head
   must name that part, and parent_object must name the containing object.
3. A gripper/end effector/robot arm is a robot_part when directly commanded.
4. Copy target_phrase from the instruction when possible. Keep useful visual
   attributes such as color, size, shape, material, left/right, and ordinal.
5. reference_phrase contains the destination or comparison object, if any.
6. Set ambiguous=true only when the directly manipulated entity cannot be
   uniquely determined from the instruction text.
7. Return exactly one JSON object. Do not emit Markdown or an explanation.

Allowed target_type values:
  object, object_part, robot_part, region, ambiguous

Output schema:
{
  "target_phrase": string,
  "target_head": string,
  "attributes": [string, ...],
  "target_type": "object" | "object_part" | "robot_part" | "region" | "ambiguous",
  "parent_object": string | null,
  "reference_phrase": string | null,
  "ambiguous": boolean
}

Examples:

Instruction: Place the small beige block onto the left peg of the tray.
Output: {"target_phrase":"small beige block","target_head":"block","attributes":["small","beige"],"target_type":"object","parent_object":null,"reference_phrase":"left peg of the tray","ambiguous":false}

Instruction: Move the rectangular peg board to the top-left corner of the table.
Output: {"target_phrase":"rectangular peg board","target_head":"peg board","attributes":["rectangular"],"target_type":"object","parent_object":null,"reference_phrase":"top-left corner of the table","ambiguous":false}

Instruction: Slide the pot so its handle touches the ranch bottle.
Output: {"target_phrase":"pot","target_head":"pot","attributes":[],"target_type":"object","parent_object":null,"reference_phrase":"ranch bottle","ambiguous":false}

Instruction: Pick up the pot's right handle.
Output: {"target_phrase":"pot's right handle","target_head":"handle","attributes":["right"],"target_type":"object_part","parent_object":"pot","reference_phrase":null,"ambiguous":false}

Instruction: Put the gripper over the left edge of the workspace.
Output: {"target_phrase":"gripper","target_head":"gripper","attributes":[],"target_type":"robot_part","parent_object":null,"reference_phrase":"left edge of the workspace","ambiguous":false}

Instruction: Push the dark cloth backward on the tabletop.
Output: {"target_phrase":"dark cloth","target_head":"cloth","attributes":["dark"],"target_type":"object","parent_object":null,"reference_phrase":"tabletop","ambiguous":false}

Instruction: Turn the faucet knob clockwise.
Output: {"target_phrase":"faucet knob","target_head":"knob","attributes":[],"target_type":"object_part","parent_object":"faucet","reference_phrase":null,"ambiguous":false}
""".strip()


@dataclass(frozen=True)
class TargetParse:
    target_phrase: str
    target_head: str
    attributes: tuple[str, ...]
    target_type: str
    parent_object: str | None
    reference_phrase: str | None
    ambiguous: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["attributes"] = list(self.attributes)
        return payload


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:")


def _optional_text(value: Any) -> str | None:
    text = _clean_text(value)
    if not text or text.lower() in {"none", "null", "n/a", "unknown"}:
        return None
    return text


def _derive_head(phrase: str) -> str:
    phrase = phrase.lower().replace("'s", " ")
    before_of = re.split(r"\bof\b", phrase, maxsplit=1)[0]
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", before_of)
    ignored = {
        "the", "a", "an", "small", "large", "big", "tiny", "left", "right",
        "front", "back", "rear", "top", "bottom", "red", "blue", "green",
        "yellow", "orange", "white", "black", "dark", "light", "beige",
        "metallic", "metal", "wooden", "plastic", "rectangular", "round",
    }
    content = [token for token in tokens if token not in ignored]
    return content[-1] if content else (tokens[-1] if tokens else "")


def normalize_parse_payload(payload: Mapping[str, Any]) -> TargetParse:
    """Validate and normalize a model-produced JSON object.

    Raises ``ValueError`` for an unusable target.  Minor schema variations are
    normalized but never silently replace a missing target with a destination.
    """

    phrase = _clean_text(payload.get("target_phrase"))
    if not phrase:
        raise ValueError("target_phrase is empty")

    head = _clean_text(payload.get("target_head")) or _derive_head(phrase)
    if not head:
        raise ValueError("target_head is empty")

    raw_attributes = payload.get("attributes", [])
    if isinstance(raw_attributes, str):
        raw_attributes = [part for part in re.split(r"[,;/]", raw_attributes) if part.strip()]
    if not isinstance(raw_attributes, (list, tuple)):
        raise ValueError("attributes must be a list or string")
    attributes: list[str] = []
    seen_attributes: set[str] = set()
    for item in raw_attributes:
        value = _clean_text(item).lower()
        if value and value not in seen_attributes:
            attributes.append(value)
            seen_attributes.add(value)

    target_type = _clean_text(payload.get("target_type", "object")).lower().replace(" ", "_")
    aliases = {
        "part": "object_part",
        "objectpart": "object_part",
        "robot": "robot_part",
        "robotpart": "robot_part",
        "spatial_region": "region",
        "unknown": "ambiguous",
    }
    target_type = aliases.get(target_type, target_type)
    if target_type not in TARGET_TYPES:
        raise ValueError(f"unsupported target_type: {target_type!r}")

    ambiguous_raw = payload.get("ambiguous", target_type == "ambiguous")
    if isinstance(ambiguous_raw, str):
        ambiguous = ambiguous_raw.strip().lower() in {"true", "1", "yes", "y"}
    else:
        ambiguous = bool(ambiguous_raw)
    if target_type == "ambiguous":
        ambiguous = True

    return TargetParse(
        target_phrase=phrase,
        target_head=head,
        attributes=tuple(attributes),
        target_type=target_type,
        parent_object=_optional_text(payload.get("parent_object")),
        reference_phrase=_optional_text(payload.get("reference_phrase")),
        ambiguous=ambiguous,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from possibly wrapped output."""

    start = text.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                value = json.loads(candidate)
                if not isinstance(value, dict):
                    raise ValueError("JSON value is not an object")
                return value
    raise ValueError("model output contains an unterminated JSON object")


_ACTION_PATTERN = re.compile(
    r"\b(?:pick(?:\s+up)?|grasp|grab|lift|take|move|place|put|push|pull|slide|"
    r"insert|open|close|turn|rotate|fold|stack|pour|wipe|press|bring|set)\b\s+",
    flags=re.IGNORECASE,
)
_BOUNDARY_PATTERN = re.compile(
    r"\b(?:and\s+then|and|then|to|into|onto|on|in|from|toward|towards|over|"
    r"under|beside|near|so\s+that|so\s+its|until|at|with)\b",
    flags=re.IGNORECASE,
)


def heuristic_parse(instruction: str) -> TargetParse:
    """Explicitly marked fallback used only if both LLM outputs are invalid."""

    text = _clean_text(instruction)
    match = _ACTION_PATTERN.search(text)
    remainder = text[match.end() :] if match else text
    boundary = _BOUNDARY_PATTERN.search(remainder)
    phrase = remainder[: boundary.start()] if boundary else remainder
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE).strip()
    if not phrase:
        phrase = text
    lower = phrase.lower()
    parent = None
    target_type = "object"
    if "gripper" in lower or "end effector" in lower or "robot arm" in lower:
        target_type = "robot_part"
    possessive = re.match(r"(.+?)['’]s\s+(.+)$", phrase)
    if possessive:
        parent = possessive.group(1).strip()
        target_type = "object_part"
    head = _derive_head(phrase) or phrase
    reference = remainder[boundary.start() :].strip() if boundary else None
    return TargetParse(
        target_phrase=phrase,
        target_head=head,
        attributes=(),
        target_type=target_type,
        parent_object=parent,
        reference_phrase=reference,
        ambiguous=False,
    )


def _normalized_tokens(text: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("pegboard", "peg board"))
    normalized: list[str] = []
    for token in tokens:
        if len(token) > 3 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized.append(token)
    return tuple(normalized)


def compare_parses(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic agreement diagnostics for two valid parse dicts."""

    a = normalize_parse_payload(first)
    b = normalize_parse_payload(second)
    a_head = _normalized_tokens(a.target_head)
    b_head = _normalized_tokens(b.target_head)
    head_exact = a_head == b_head
    a_phrase = set(_normalized_tokens(a.target_phrase))
    b_phrase = set(_normalized_tokens(b.target_phrase))
    union = a_phrase | b_phrase
    phrase_jaccard = len(a_phrase & b_phrase) / len(union) if union else 0.0
    head_compatible = bool(set(a_head) & set(b_head)) or (
        bool(a_head) and a_head[-1] in b_phrase
    ) or (bool(b_head) and b_head[-1] in a_phrase)
    if head_exact and phrase_jaccard >= 0.8 and a.target_type == b.target_type:
        level = "exact"
    elif head_exact or head_compatible or phrase_jaccard >= 0.5:
        level = "compatible"
    else:
        level = "disagree"
    return {
        "agreement_level": level,
        "head_exact": head_exact,
        "head_compatible": head_compatible,
        "phrase_jaccard": phrase_jaccard,
        "type_agreement": a.target_type == b.target_type,
    }


def build_grounding_queries(parse: Mapping[str, Any] | TargetParse, max_queries: int = 4) -> list[str]:
    """Build ordered GroundingDINO queries without adding destination objects."""

    target = parse if isinstance(parse, TargetParse) else normalize_parse_payload(parse)
    candidates: list[str] = [target.target_phrase]
    head = target.target_head
    attributes = list(target.attributes)
    if target.target_type == "object_part" and target.parent_object:
        candidates.extend(
            [
                f"{head} of {target.parent_object}",
                f"{target.parent_object} {head}",
                head,
            ]
        )
    else:
        if attributes:
            candidates.append(" ".join([*attributes, head]))
        candidates.append(head)
    lower_head = head.lower()
    if target.target_type == "robot_part" or lower_head in {"gripper", "end effector"}:
        candidates.extend(["robot gripper", "robot end effector"])

    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = _clean_text(candidate).lower()
        value = re.sub(r"^(?:the|a|an)\s+", "", value)
        if value and value not in seen:
            output.append(value)
            seen.add(value)
        if len(output) >= max_queries:
            break
    return output


def model_slug(model_path: str | Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Path(model_path).name.lower()).strip("_")


class LocalInstructionParser:
    """Batched deterministic parser backed by a local causal language model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 160,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        dtype_value = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype_value,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self._torch = torch

    def _prompt(self, instruction: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Instruction: {instruction}\nOutput:"},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def parse_batch(self, instructions: Sequence[str]) -> list[dict[str, Any]]:
        if not instructions:
            return []
        prompts = [self._prompt(instruction) for instruction in instructions]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)
        input_width = int(inputs.input_ids.shape[1])
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        results: list[dict[str, Any]] = []
        for row in generated:
            raw = self.tokenizer.decode(row[input_width:], skip_special_tokens=True).strip()
            try:
                payload = extract_json_object(raw)
                parsed = normalize_parse_payload(payload)
                results.append({"valid": True, "parsed": parsed.to_dict(), "raw": raw, "error": None})
            except Exception as exc:
                results.append(
                    {
                        "valid": False,
                        "parsed": None,
                        "raw": raw,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return results

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            del self.tokenizer
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def __enter__(self) -> "LocalInstructionParser":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
