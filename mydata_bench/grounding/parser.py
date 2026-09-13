from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..io import object_fingerprint
from ..schemas import TargetSpec

# Keep phrasal ``turn on/off`` ahead of bare ``turn``.  Otherwise the
# destination splitter treats the leading "on" as a preposition and produces
# an empty target phrase for instructions such as "Turn on the left burner".
VERBS = r"(?:move|slide|push|rotate|turn\s+(?:on|off)|turn|touch|insert|place|put|pick(?:\s+up)?|lift|grasp|open|close|pull|press)"
DESTINATION = re.compile(
    r"\b(?:onto|into|inside|through|towards?|to|on|under|above|beside|next to)\b",
    re.IGNORECASE,
)
SEQUENCE = re.compile(r"\b(?:followed by|then|after that|and then|before)\b", re.I)
PARTS = {
    "handle",
    "lid",
    "cap",
    "door",
    "drawer",
    "button",
    "knob",
    "lever",
    "switch",
    "peg",
    "wheel",
    "edge",
}
ROBOT_PARTS = {"gripper", "robot arm", "arm", "wrist", "finger", "end effector"}
STOPWORDS = {
    "the",
    "a",
    "an",
    "its",
    "this",
    "that",
    "carefully",
    "gently",
    "slightly",
    "clockwise",
    "counterclockwise",
}
COLORS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "beige",
    "brown",
    "gray",
    "grey",
}
ATTRIBUTES = COLORS | {
    "small",
    "large",
    "big",
    "tiny",
    "left",
    "right",
    "top",
    "bottom",
    "round",
    "square",
    "rectangular",
}


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Parser did not return a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Parser output must be a JSON object")
    return value


def _clean_phrase(value: str) -> str:
    value = re.sub(r"[.,;:!?]+$", "", value.strip())
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


SPATIAL_RELATIONS = {
    "left_of", "right_of", "closest_to", "farthest_from",
    "nth_from_left", "nth_from_right", "nth_from_top", "nth_from_bottom",
}
PARSER_VERSION = "semantic_spatial_v2"

def normalize_target(
    example_id: str,
    value: dict[str, Any],
    *,
    parser: str,
    parser_fingerprint: str,
    raw_output: str | None = None,
) -> TargetSpec:
    phrase = _clean_phrase(str(value.get("target_phrase", "")))
    head = _clean_phrase(str(value.get("head_noun") or value.get("target_head") or ""))
    if not phrase or not head:
        raise ValueError("target_phrase and head_noun must be non-empty")
    attributes = tuple(dict.fromkeys(str(x).lower() for x in value.get("attributes", [])))
    targets = tuple(_clean_phrase(str(x)) for x in value.get("targets", []) if str(x).strip())
    entity = str(value.get("entity_type") or value.get("target_type") or "object")
    if entity not in {"object", "object_part", "robot_part", "spatial_region", "unknown"}:
        entity = "unknown"
    description = spatial_description(phrase)
    if description:
        value = {**value, **description}
    relation = value.get("relation")
    relation = str(relation).strip().lower() if relation is not None else None
    if relation not in SPATIAL_RELATIONS:
        relation = None
    ordinal_index = value.get("ordinal_index")
    if relation and relation.startswith("nth_from_"):
        if isinstance(ordinal_index, str) and ordinal_index.isdigit():
            ordinal_index = int(ordinal_index)
        if type(ordinal_index) is not int or ordinal_index < 1:
            raise ValueError("Spatial ordinals require a positive ordinal_index")
    else:
        ordinal_index = None
    return TargetSpec(
        example_id=example_id,
        target_phrase=phrase,
        head_noun=head,
        attributes=attributes,
        entity_type=entity,
        parent_object=value.get("parent_object"),
        reference_object=(value.get("reference_object") or value.get("reference_phrase")) if relation else None,
        relation=relation,
        targets=targets,
        multi_target=bool(value.get("multi_target", len(targets) > 1)),
        ambiguous=bool(value.get("ambiguous", False)),
        parser=parser,
        parser_fingerprint=parser_fingerprint,
        raw_output=raw_output,
        subject_phrase=value.get("subject_phrase"),
        ordinal_index=ordinal_index,
    )


RELATIONAL_TARGET = re.compile(
    r"^(?P<subject>.+?)\s+(?P<relation>to the left of|to the right of|left of|right of|closest to|nearest to|farthest from|furthest from)\s+(?P<reference>.+)$",
    re.IGNORECASE,
)
RELATION_NAMES = {
    "to the left of": "left_of",
    "to the right of": "right_of",
    "closest to": "closest_to",
    "farthest from": "farthest_from",
    "left of": "left_of",
    "right of": "right_of",
    "nearest to": "closest_to",
    "furthest from": "farthest_from",
}
ORDINALS = {word: index for index, word in enumerate(
    ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"), 1
)}
ORDINAL_TARGET = re.compile(
    r"^(?P<rank>" + "|".join(ORDINALS) + r"|\d+(?:st|nd|rd|th))\s+"
    r"(?P<subject>.+?)\s+from\s+(?:the\s+)?(?P<direction>left|right|top|bottom)$", re.I
)


def spatial_description(phrase: str) -> dict[str, Any] | None:
    """Parse object identity separately from a camera-space selection operator."""
    phrase = _clean_phrase(phrase)
    match = RELATIONAL_TARGET.fullmatch(phrase)
    if match:
        return {
            "subject_phrase": _clean_phrase(match.group("subject")),
            "reference_object": _clean_phrase(match.group("reference")),
            "relation": RELATION_NAMES[match.group("relation").lower()],
        }
    match = ORDINAL_TARGET.fullmatch(phrase)
    if match:
        rank = match.group("rank").lower()
        index = ORDINALS[rank] if rank in ORDINALS else int(re.match(r"\d+", rank).group())
        return {
            "subject_phrase": _clean_phrase(match.group("subject")),
            "relation": f"nth_from_{match.group('direction').lower()}",
            "ordinal_index": index,
        }
    match = re.fullmatch(r"(leftmost|rightmost|topmost|bottommost|left|right)\s+(.+)", phrase, re.I)
    if match:
        return {"subject_phrase": _clean_phrase(match[2]),
                "relation": f"nth_from_{match[1].lower().removesuffix('most')}", "ordinal_index": 1}
    match = re.fullmatch(r"(.+?)\s+on\s+the\s+(left|right)", phrase, re.I)
    if match:
        return {"subject_phrase": _clean_phrase(match[1]),
                "relation": f"nth_from_{match[2].lower()}", "ordinal_index": 1}
    return None


def _relational_parse(task: str, example_id: str) -> TargetSpec | None:
    normalized = re.sub(r"\s+", " ", task.strip())
    action = re.search(rf"\b{VERBS}\b\s+(.*)", normalized, re.I)
    if action is None or SEQUENCE.search(action[1]):
        return None
    phrase = re.split(rf"\s+and\s+(?:{VERBS})\b", action[1], maxsplit=1, flags=re.I)[0]
    phrase = _clean_phrase(phrase)
    description = spatial_description(phrase)
    if description is None:
        return None
    # A relation consumes its reference before a later destination clause.
    if description.get("reference_object"):
        description["reference_object"] = _clean_phrase(re.split(
            r"\s+(?:into|onto|in|inside|with)\s+", description["reference_object"], maxsplit=1, flags=re.I
        )[0])
    subject = description["subject_phrase"]
    words = [word.lower() for word in re.findall(r"[A-Za-z0-9_-]+", subject)]
    meaningful = [word for word in words if word not in STOPWORDS]
    if not meaningful:
        return None
    head = " ".join(word for word in meaningful if word not in ATTRIBUTES) or meaningful[-1]
    subject_type = heuristic_parse(f"Touch {subject}.")
    payload = {
        "target_phrase": phrase,
        "head_noun": head,
        "attributes": tuple(word for word in words if word in ATTRIBUTES),
        "entity_type": subject_type.entity_type,
        "parent_object": subject_type.parent_object,
        **description,
        "targets": [phrase],
        "multi_target": False,
        "ambiguous": False,
    }
    return normalize_target(
        example_id,
        payload,
        parser=PARSER_VERSION,
        parser_fingerprint=object_fingerprint(
            {"parser": PARSER_VERSION, "relations": RELATION_NAMES}
        ),
    )


def heuristic_parse(task: str, example_id: str = "") -> TargetSpec:
    relational = _relational_parse(task, example_id)
    if relational is not None:
        return relational
    normalized = re.sub(r"\s+", " ", task.strip())
    match = re.search(rf"\b{VERBS}\b\s+(.*)", normalized, re.I)
    tail = match.group(1) if match else normalized
    clauses = SEQUENCE.split(tail)
    multi = len(clauses) > 1
    first = clauses[0]
    first = re.split(rf"\s+and\s+(?:{VERBS})\b", first, maxsplit=1, flags=re.I)[0]
    destination = DESTINATION.split(first, maxsplit=1)
    phrase = _clean_phrase(destination[0])
    phrase = re.split(
        r"\b(?:so that|so (?:its|the)|until|while)\b",
        phrase,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    phrase = re.sub(r"^(?:your|robot(?:'s)?)\s+", "", phrase, flags=re.I)
    phrase = re.sub(
        r"\s+(?:clockwise|counterclockwise|gently|carefully|slightly)$",
        "",
        phrase,
        flags=re.I,
    )
    # "touch X with the gripper": X is the operated target.
    phrase = re.split(r"\bwith\b", phrase, maxsplit=1, flags=re.I)[0].strip()
    # Preserve a parse record for malformed or non-object instructions rather
    # than aborting an entire batch.  ``unknown``/``ambiguous`` samples are
    # excluded from formal object-head analysis by ``TargetSpec.formal_scope``.
    if not phrase:
        return normalize_target(
            example_id,
            {
                "target_phrase": "unknown target",
                "head_noun": "target",
                "entity_type": "unknown",
                "targets": [],
                "multi_target": multi,
                "ambiguous": True,
            },
            parser="heuristic_v1",
            parser_fingerprint=object_fingerprint({"parser": "heuristic_v1"}),
        )
    possessive = re.match(
        r"(.+?)(?:'s\s+|\s+)(left |right |top |bottom )?("
        + "|".join(PARTS)
        + r")$",
        phrase,
        re.I,
    )
    words = [word.lower() for word in re.findall(r"[A-Za-z0-9_-]+", phrase)]
    lower_phrase = phrase.lower()
    parent = None
    if possessive:
        parent = _clean_phrase(possessive.group(1))
        head = possessive.group(3).lower()
        entity = "object_part"
    else:
        meaningful = [word for word in words if word not in STOPWORDS]
        head = meaningful[-1] if meaningful else "unknown"
        if any(part in lower_phrase for part in ROBOT_PARTS):
            entity = "robot_part"
        elif head in PARTS and len(meaningful) > 1:
            entity = "object_part"
            parent = meaningful[-2]
        elif any(token in lower_phrase for token in ("region", "area", "space", "spot")):
            entity = "spatial_region"
        else:
            entity = "object"
    attrs = tuple(word for word in words if word in ATTRIBUTES)
    targets = [phrase]
    if multi:
        for clause in clauses[1:]:
            candidate = _clean_phrase(DESTINATION.split(clause, maxsplit=1)[0])
            candidate = re.sub(rf"^(?:{VERBS})\s+", "", candidate, flags=re.I)
            if candidate:
                targets.append(candidate)
    reference = _clean_phrase(destination[1]) if len(destination) > 1 else None
    payload = {
        "target_phrase": phrase,
        "head_noun": head,
        "attributes": attrs,
        "entity_type": entity,
        "parent_object": parent,
        "reference_object": reference,
        "targets": targets,
        "multi_target": multi,
        "ambiguous": not bool(phrase) or head == "unknown",
    }
    return normalize_target(
        example_id,
        payload,
        parser="heuristic_v1",
        parser_fingerprint=object_fingerprint({"parser": "heuristic_v1"}),
    )


PARSER_PROMPT = """Extract the entity directly manipulated by the robot instruction.
Return exactly one JSON object with keys:
target_phrase, head_noun, attributes (array), entity_type
(object|object_part|robot_part|spatial_region|unknown), parent_object,
reference_object, relation, subject_phrase, ordinal_index,
targets (ordered array), multi_target, ambiguous.
Do not treat a destination/reference as the manipulated target. Preserve
possessive object parts. Mark sequential multiple targets as multi_target.
Preserve compound object names such as "salmon sushi". For spatial instructions,
subject_phrase contains the complete object description without the relation.
relation is left_of|right_of|closest_to|farthest_from|nth_from_left|nth_from_right|
nth_from_top|nth_from_bottom or null. Ordinals are one-based; leftmost means
nth_from_left with ordinal_index=1. Do not put a destination in reference_object.
Instruction: {task}
"""


class InstructionParser:
    def __init__(self, model_path: str | None = None, *, use_model: bool = True):
        self.model_path = str(Path(model_path).resolve()) if model_path else None
        self.use_model = bool(use_model and model_path)
        self._model = None
        self._tokenizer = None

    @property
    def fingerprint(self) -> str:
        return object_fingerprint(
            {
                "implementation": PARSER_VERSION,
                "model_path": self.model_path,
                "use_model": self.use_model,
                "prompt": PARSER_PROMPT,
                "temperature": 0,
            }
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen parsing requires torch and transformers") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def parse(self, task: str, example_id: str = "") -> TargetSpec:
        if not self.use_model:
            return replace(heuristic_parse(task, example_id), parser_fingerprint=self.fingerprint)
        self._load()
        assert self._tokenizer is not None and self._model is not None
        messages = [{"role": "user", "content": PARSER_PROMPT.format(task=task)}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        raw = self._tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        try:
            parsed = normalize_target(
                example_id,
                extract_json_object(raw),
                parser="Qwen3-4B-Instruct-2507",
                parser_fingerprint=self.fingerprint,
                raw_output=raw,
            )
            heuristic = heuristic_parse(task, example_id)
            # Spatial relation clauses are kept as one semantic unit. This
            # deterministic override prevents subword/token-boundary behavior
            # in the LLM parser from dropping "closest to" or "left/right of".
            if heuristic.relation is not None:
                parsed = replace(
                    parsed,
                    target_phrase=heuristic.target_phrase,
                    head_noun=heuristic.head_noun,
                    attributes=heuristic.attributes,
                    entity_type=heuristic.entity_type,
                    reference_object=heuristic.reference_object,
                    relation=heuristic.relation,
                    targets=heuristic.targets,
                    multi_target=False,
                    ambiguous=False,
                    subject_phrase=heuristic.subject_phrase,
                    ordinal_index=heuristic.ordinal_index,
                )
            # Structural syntax is more reliable than free-form model JSON for
            # safety-critical exclusions. A missed sequential marker would
            # otherwise leak a multi-object task into the formal single-object
            # causal population.
            if SEQUENCE.search(task):
                parsed = replace(
                    parsed,
                    multi_target=True,
                    targets=heuristic.targets,
                )
            if (
                heuristic.entity_type == "robot_part"
                and any(part in parsed.target_phrase.lower() for part in ROBOT_PARTS)
            ):
                parsed = replace(parsed, entity_type="robot_part")
            return parsed
        except (ValueError, json.JSONDecodeError):
            fallback = heuristic_parse(task, example_id)
            value = fallback.to_dict()
            value["parser"] = "heuristic_v1_after_qwen_failure"
            value["parser_fingerprint"] = self.fingerprint
            value["raw_output"] = raw
            value.pop("schema_version", None)
            return TargetSpec(**value)


def build_queries(target: TargetSpec) -> list[str]:
    description = spatial_description(target.target_phrase) or {}
    subject = (
        target.subject_phrase or description.get("subject_phrase") or target.target_phrase
        if target.relation or description else target.target_phrase
    )
    subject = _clean_phrase(subject)
    queries = [subject]
    if target.entity_type == "object_part" and target.parent_object:
        queries.append(f"{target.head_noun} of {target.parent_object}")
    # A bare class/parent query silently discards color, subtype, or part
    # identity. Synonyms belong in an explicit dataset-specific alias map.
    return list(dict.fromkeys(query.strip().lower() for query in queries if query.strip()))


def reference_queries(target: TargetSpec) -> list[str]:
    if not target.reference_object:
        return []
    return [str(target.reference_object).strip().lower()]
