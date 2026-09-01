"""Incremental three-layer diagnostics for Qwen3-VL gaze-head steering.

This module is deliberately separate from the historical ranking/steering
pipeline.  It reuses its frozen cohort, processor alignment, eager runtime and
attention-mask hooks, while adding the measurements required by the concise
debug guide: attention traces, object probes and full candidate-sequence reward
log-probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw

from .attention_eval.masking import Head, ImageSpan
from .attention_eval.runtime import find_contiguous_spans
from .config import load_config, section
from .io import append_jsonl, artifact_fingerprint, object_fingerprint, read_jsonl, sha256_file, write_json
from .qwen_eval.attention import PreparedAttentionInput, QwenAttentionRuntime, _move_inputs
from .qwen_eval.protocols import ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE
from .roboreward_eval.runner import parse_native_score


PROBE_TEMPLATE = """You are checking whether a robot trajectory follows a command.
Command: {task}

The following eight observations are chronological. Observation 1 is initial
and Observation 8 is terminal. Identify the object requested by the command,
the object the robot actually manipulated, and whether they match.

{observations}

Return exactly three lines and no explanation:
REQUESTED_OBJECT: <short noun phrase>
MANIPULATED_OBJECT: <short noun phrase>
MATCH: <YES, NO, or UNCERTAIN>
"""


def matched_random_heads(
    gaze_heads: Sequence[Head], *, num_heads: int, seed: int
) -> list[Head]:
    """Choose deterministic controls with the same per-layer head counts."""
    import random

    if num_heads < 1:
        raise ValueError("num_heads must be positive")
    grouped: dict[int, list[int]] = defaultdict(list)
    for item in gaze_heads:
        grouped[int(item.layer)].append(int(item.head))
    rng = random.Random(int(seed))
    result: list[Head] = []
    for layer, gaze in sorted(grouped.items()):
        excluded = set(gaze)
        candidates = [head for head in range(num_heads) if head not in excluded]
        if len(candidates) < len(gaze):
            raise ValueError(f"Not enough random controls in layer {layer}")
        result.extend(Head(layer, head) for head in rng.sample(candidates, len(gaze)))
    return sorted(result, key=lambda item: (item.layer, item.head))


def find_subsequence(sequence: Sequence[int], query: Sequence[int], *, last: bool = False) -> list[int]:
    if not query:
        return []
    starts = [
        index
        for index in range(len(sequence) - len(query) + 1)
        if list(sequence[index : index + len(query)]) == list(query)
    ]
    if not starts:
        return []
    start = starts[-1] if last else starts[0]
    return list(range(start, start + len(query)))


def _phrase_positions(tokenizer: Any, input_ids: Sequence[int], phrase: str, *, last: bool = False) -> list[int]:
    for value in (phrase, f" {phrase}"):
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        found = find_subsequence(input_ids, token_ids, last=last)
        if found:
            return found
    return []


def query_position_groups(
    runtime: QwenAttentionRuntime,
    sample: dict[str, Any],
    prepared: PreparedAttentionInput,
) -> dict[str, list[int]]:
    """Locate the guide's prompt-query alternatives without hard-coded indices."""
    ids = prepared.inputs["input_ids"][0].detach().cpu().tolist()
    groups: dict[str, list[int]] = {"last_prompt_token": [len(ids) - 1]}
    phrase = str(sample.get("target_phrase") or "").strip()
    head = str(sample.get("head_noun") or "").strip()
    attributes = sample.get("attributes") or []
    if phrase:
        found = _phrase_positions(runtime.processor.tokenizer, ids, phrase)
        if found:
            groups["instruction_target_phrase"] = found
    if head:
        found = _phrase_positions(runtime.processor.tokenizer, ids, head)
        if found:
            groups["instruction_object_category"] = found
    attribute_positions = []
    for attribute in attributes:
        attribute_positions.extend(
            _phrase_positions(runtime.processor.tokenizer, ids, str(attribute))
        )
    if attribute_positions:
        groups["instruction_attributes"] = sorted(set(attribute_positions))
    reward = _phrase_positions(runtime.processor.tokenizer, ids, "ANSWER", last=True)
    if reward:
        groups["reward_anchor"] = reward
    return groups


def _probe_messages(task: str, paths: Sequence[str]) -> list[dict[str, Any]]:
    observations = "\n".join(
        f"OBSERVATION {index}{' — INITIAL' if index == 1 else ' — TERMINAL' if index == 8 else ''}:\n<image>"
        for index in range(1, len(paths) + 1)
    )
    parts = PROBE_TEMPLATE.format(task=task, observations=observations).split("<image>")
    if len(parts) != len(paths) + 1:
        raise RuntimeError("Object-probe image placeholder mismatch")
    content: list[dict[str, Any]] = []
    for text, path in zip(parts[:-1], paths):
        content.append({"type": "text", "text": text})
        content.append({"type": "image", "image": str(Path(path).resolve())})
    content.append({"type": "text", "text": parts[-1]})
    return [{"role": "user", "content": content}]


def prepare_object_probe(
    runtime: QwenAttentionRuntime,
    sample: dict[str, Any],
    reward_prepared: PreparedAttentionInput,
) -> PreparedAttentionInput:
    paths = [str(Path(path).resolve()) for path in sample["image_paths"]]
    inputs = runtime.processor.apply_chat_template(
        _probe_messages(sample["task"], paths),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    token_id = int(getattr(runtime.model.config, "image_token_id", 151655))
    spans = find_contiguous_spans(inputs["input_ids"][0].tolist(), token_id)
    grids = inputs.get("image_grid_thw")
    if grids is None or len(spans) != len(paths) or int(grids.shape[0]) != len(paths):
        raise RuntimeError("Object-probe processor alignment failed")
    image_spans = [
        ImageSpan(
            f"image_t{index}",
            path,
            int(start),
            int(end),
            tuple(int(value) for value in grid),
        )
        for index, (path, (start, end), grid) in enumerate(zip(paths, spans, grids.tolist()))
    ]
    visual = [position for span in image_spans for position in range(span.start, span.end)]
    metadata = json.loads(json.dumps(reward_prepared.video_metadata or {}))
    return PreparedAttentionInput(
        inputs=_move_inputs(runtime.torch, runtime.model, inputs, runtime.dtype),
        spans=image_spans,
        target_span=image_spans[-1],
        target_image_path=paths[-1],
        visual_positions=visual,
        protocol=runtime.protocol,
        video_metadata=metadata,
    )


def _steering_context(
    runtime: QwenAttentionRuntime,
    prepared: PreparedAttentionInput,
    heads: Sequence[Head],
    target: Sequence[int],
    *,
    bias: float,
    query_scope: str,
    negative_scope: str,
    diagnostics: dict[str, Any],
):
    if not heads:
        return nullcontext()
    return runtime.steering_hooks(
        heads,
        target,
        prepared.visual_positions,
        bias,
        query_scope,
        negative_scope,
        prepared.spans,
        diagnostics,
    )


def generate_with_attention_trace(
    runtime: QwenAttentionRuntime,
    sample: dict[str, Any],
    prepared: PreparedAttentionInput,
    heads: Sequence[Head],
    target: Sequence[int],
    *,
    bias: float,
    query_scope: str,
    negative_scope: str,
    max_new_tokens: int,
    query_groups: dict[str, list[int]],
    trace_heads: Sequence[Head] | None = None,
) -> dict[str, Any]:
    """Generate while collecting A(R) at named prefill and decode queries."""
    per_layer_heads: dict[int, list[int]] = defaultdict(list)
    for item in (trace_heads if trace_heads is not None else heads):
        per_layer_heads[int(item.layer)].append(int(item.head))
    traces: list[dict[str, Any]] = []
    call_index: Counter[int] = Counter()
    handles = []

    def collector(layer: int, selected_heads: list[int]):
        def hook(_module, _args, output):
            if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
                raise RuntimeError(f"Layer {layer} did not expose eager attention weights")
            weights = output[1][0]
            query_length = int(weights.shape[-2])
            call = call_index[layer]
            call_index[layer] += 1
            if query_length == 1:
                groups = {"first_generated_token_query" if call == 1 else f"decode_query_{call - 1}": [0]}
                phase = "decode"
            else:
                groups = query_groups
                phase = "prefill"
            for name, positions in groups.items():
                valid = [int(value) for value in positions if 0 <= int(value) < query_length]
                if not valid:
                    continue
                for head in selected_heads:
                    matrix = weights[head, valid, :]
                    traces.append(
                        {
                            "layer": layer,
                            "head": head,
                            "call": call,
                            "phase": phase,
                            "query": name,
                            "query_token_count": len(valid),
                            "target_mass": float(matrix[:, list(target)].sum(dim=-1).mean().detach().float().cpu()),
                            "visual_mass": float(matrix[:, prepared.visual_positions].sum(dim=-1).mean().detach().float().cpu()),
                        }
                    )
            return output[0], None

        return hook

    diagnostics: dict[str, Any] = {
        "hook_active": bool(heads),
        "bias": float(bias),
        "query_scope": query_scope,
        "negative_scope": negative_scope,
    }
    try:
        for layer, selected in sorted(per_layer_heads.items()):
            handles.append(runtime.layers[layer].self_attn.register_forward_hook(collector(layer, selected)))
        context = _steering_context(
            runtime,
            prepared,
            heads if bias != 0 or diagnostics["hook_active"] else (),
            target,
            bias=bias,
            query_scope=query_scope,
            negative_scope=negative_scope,
            diagnostics=diagnostics,
        )
        with context, runtime.torch.inference_mode():
            generated = runtime.model.generate(
                **prepared.inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=False,
                pad_token_id=runtime.processor.tokenizer.pad_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()
    sequence = generated[0, prepared.inputs["input_ids"].shape[1] :]
    raw = runtime.processor.tokenizer.decode(sequence, skip_special_tokens=True).strip()
    try:
        prediction = parse_native_score(raw)
        parse_error = None
    except ValueError as exc:
        prediction = None
        parse_error = str(exc)
    return {
        "raw_output": raw,
        "generated_token_ids": sequence.detach().cpu().tolist(),
        "native_prediction": prediction,
        "parse_error": parse_error,
        "attention_trace": traces,
        "hook_diagnostics": diagnostics,
    }


def generate_probe(
    runtime: QwenAttentionRuntime,
    prepared: PreparedAttentionInput,
    heads: Sequence[Head],
    target: Sequence[int],
    *,
    bias: float,
    query_scope: str = "all",
    negative_scope: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    context = _steering_context(
        runtime,
        prepared,
        heads,
        target,
        bias=bias,
        query_scope=query_scope,
        negative_scope=negative_scope,
        diagnostics=diagnostics,
    )
    with context, runtime.torch.inference_mode():
        generated = runtime.model.generate(
            **prepared.inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            output_attentions=False,
            pad_token_id=runtime.processor.tokenizer.pad_token_id,
        )
    sequence = generated[0, prepared.inputs["input_ids"].shape[1] :]
    raw = runtime.processor.tokenizer.decode(sequence, skip_special_tokens=True).strip()
    parsed = {}
    for key in ("REQUESTED_OBJECT", "MANIPULATED_OBJECT", "MATCH"):
        match = re.search(rf"(?im)^\s*{key}\s*:\s*(.+?)\s*$", raw)
        parsed[key.lower()] = match.group(1).strip() if match else None
    if parsed["match"] is not None:
        parsed["match"] = parsed["match"].upper().strip(" .")
    return {"raw_output": raw, "parsed": parsed, "hook_diagnostics": diagnostics}


def reward_sequence_logprobs(
    runtime: QwenAttentionRuntime,
    prepared: PreparedAttentionInput,
    heads: Sequence[Head],
    target: Sequence[int],
    *,
    bias: float,
    query_scope: str = "all",
    negative_scope: str,
    cached_decode: bool = False,
) -> dict[str, Any]:
    """Teacher-force complete ANSWER sequences and normalize over rewards 1..5."""
    torch = runtime.torch
    prompt_length = int(prepared.inputs["input_ids"].shape[1])
    values = []
    for reward in range(1, 6):
        candidate = f"ANSWER: {reward}"
        candidate_ids = runtime.processor.tokenizer.encode(candidate, add_special_tokens=False)
        ids = torch.tensor(
            [candidate_ids],
            device=runtime.model.device,
            dtype=prepared.inputs["input_ids"].dtype,
        )
        diagnostics: dict[str, Any] = {}
        context = _steering_context(
            runtime,
            prepared,
            heads,
            target,
            bias=bias,
            query_scope=query_scope,
            negative_scope=negative_scope,
            diagnostics=diagnostics,
        )
        if cached_decode:
            cache_position = torch.arange(
                prompt_length,
                device=prepared.inputs["input_ids"].device,
                dtype=torch.long,
            )
            full_ids = prepared.inputs["input_ids"]
            full_attention_mask = prepared.inputs.get("attention_mask")
            media_inputs = {
                key: value
                for key, value in prepared.inputs.items()
                if key not in {"input_ids", "attention_mask"}
            }
            selected_values = []
            with context, torch.inference_mode():
                outputs = runtime.model(
                    **prepared.inputs,
                    use_cache=True,
                    cache_position=cache_position,
                    return_dict=True,
                )
                for token_index, token_id in enumerate(candidate_ids):
                    logprobs = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
                    selected_values.append(float(logprobs[int(token_id)].detach().cpu()))
                    if token_index == len(candidate_ids) - 1:
                        break
                    next_token = ids[:, token_index : token_index + 1]
                    full_ids = torch.cat([full_ids, next_token], dim=1)
                    if full_attention_mask is not None:
                        full_attention_mask = torch.cat(
                            [
                                full_attention_mask,
                                torch.ones(
                                    (full_attention_mask.shape[0], 1),
                                    device=full_attention_mask.device,
                                    dtype=full_attention_mask.dtype,
                                ),
                            ],
                            dim=1,
                        )
                    cache_position = cache_position[-1:] + 1
                    model_inputs = runtime.model.prepare_inputs_for_generation(
                        full_ids,
                        past_key_values=outputs.past_key_values,
                        attention_mask=full_attention_mask,
                        cache_position=cache_position,
                        use_cache=True,
                        **media_inputs,
                    )
                    outputs = runtime.model(**model_inputs, return_dict=True)
            token_values = selected_values
        else:
            combined = dict(prepared.inputs)
            combined["input_ids"] = torch.cat([prepared.inputs["input_ids"], ids], dim=1)
            if "attention_mask" in prepared.inputs:
                extension = torch.ones(
                    (prepared.inputs["attention_mask"].shape[0], len(candidate_ids)),
                    device=prepared.inputs["attention_mask"].device,
                    dtype=prepared.inputs["attention_mask"].dtype,
                )
                combined["attention_mask"] = torch.cat(
                    [prepared.inputs["attention_mask"], extension], dim=1
                )
            with context, torch.inference_mode():
                logits = runtime.model(**combined, use_cache=False).logits[0]
            prediction_logits = logits[
                prompt_length - 1 : prompt_length + len(candidate_ids) - 1
            ]
            token_logprobs = torch.log_softmax(prediction_logits.float(), dim=-1)
            selected = token_logprobs[
                torch.arange(len(candidate_ids), device=token_logprobs.device),
                ids[0].long(),
            ]
            token_values = selected.detach().cpu().tolist()
        values.append(
            {
                "reward": reward,
                "candidate": candidate,
                "token_ids": candidate_ids,
                "token_logprobs": token_values,
                "sequence_logprob": float(sum(token_values)),
            }
        )
        if not cached_decode:
            del logits, prediction_logits, token_logprobs, selected
    sequence = [row["sequence_logprob"] for row in values]
    peak = max(sequence)
    weights = [math.exp(value - peak) for value in sequence]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    for row, probability in zip(values, probabilities):
        row["probability"] = probability

    def lse(indices: Iterable[int]) -> float:
        selected = [sequence[index] for index in indices]
        maximum = max(selected)
        return maximum + math.log(sum(math.exp(value - maximum) for value in selected))

    return {
        "scoring_mode": (
            "cached_autoregressive" if cached_decode else "teacher_forced_single_forward"
        ),
        "candidates": values,
        "expected_reward": sum((index + 1) * probability for index, probability in enumerate(probabilities)),
        "high_low_margin": lse((3, 4)) - lse((0, 1)),
        "argmax_reward": max(range(1, 6), key=lambda reward: sequence[reward - 1]),
    }


def write_token_overlays(
    runtime: QwenAttentionRuntime,
    sample: dict[str, Any],
    prepared: PreparedAttentionInput,
    target: Sequence[int],
    output_dir: Path,
) -> list[str]:
    alignments = {
        row["span"]: row
        for row in (prepared.video_metadata or {}).get("intervention_span_tracking_alignment", [])
    }
    written = []
    safe_id = sample["example_id"].replace("/", "__")
    for span in prepared.spans:
        selected = [position for position in target if span.start <= position < span.end]
        if not selected:
            continue
        with Image.open(span.path).convert("RGB") as image:
            draw = ImageDraw.Draw(image, "RGBA")
            width, height = image.size
            _t, raw_h, raw_w = span.grid_thw
            grid_h = raw_h // runtime.merge_size
            grid_w = raw_w // runtime.merge_size
            for position in selected:
                cell = position - span.start
                row, column = divmod(cell, grid_w)
                x1, x2 = column * width / grid_w, (column + 1) * width / grid_w
                y1, y2 = row * height / grid_h, (row + 1) * height / grid_h
                draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 70), outline=(255, 0, 0, 255), width=2)
            bbox = alignments.get(span.label, {}).get("applied_bbox")
            if bbox:
                draw.rectangle(tuple(bbox), outline=(0, 255, 0, 255), width=3)
            path = output_dir / "overlays" / f"{safe_id}__{span.label}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                image.save(path)
            written.append(str(path.resolve()))
    return written


def _load_heads(path: Path, top_k: int) -> tuple[list[Head], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("ranking")
    if not isinstance(rows, list) or len(rows) < top_k:
        raise ValueError(f"Ranking has fewer than {top_k} heads: {path}")
    return [Head(int(row["layer"]), int(row["head"])) for row in rows[:top_k]], str(data.get("fingerprint") or sha256_file(path))


def _join_targets(samples: list[dict[str, Any]], targets_path: Path) -> None:
    targets = {row["example_id"]: row for row in read_jsonl(targets_path)}
    for sample in samples:
        target = targets.get(sample["example_id"], {})
        for key in ("target_phrase", "head_noun", "attributes", "entity_type"):
            sample[key] = target.get(key)


def _condition_spec(
    gaze: Sequence[Head], random_heads: Sequence[Head], strong_bias: float
) -> list[tuple[str, Sequence[Head], float, str, str]]:
    return [
        ("baseline", (), 0.0, "all", "none"),
        ("zero_bias_hook", gaze, 0.0, "all", "all_visual"),
        ("gaze_prefill", gaze, strong_bias, "prefill", "all_visual"),
        ("gaze_decode", gaze, strong_bias, "decode", "all_visual"),
        ("gaze_both_target_vs_all", gaze, strong_bias, "all", "all_visual"),
        ("gaze_last_prompt", gaze, strong_bias, "last_prompt", "all_visual"),
        ("random_both_target_vs_all", random_heads, strong_bias, "all", "all_visual"),
        ("gaze_both_evidence_preserving", gaze, strong_bias, "all", "none"),
    ]


def select_condition_specs(
    conditions: Sequence[tuple[str, Sequence[Head], float, str, str]],
    requested: Sequence[str] | None,
) -> list[tuple[str, Sequence[Head], float, str, str]]:
    """Select an ordered condition subset while rejecting misspelled names."""
    if requested is None:
        return list(conditions)
    by_name = {item[0]: item for item in conditions}
    unknown = [str(name) for name in requested if str(name) not in by_name]
    if unknown:
        raise ValueError(f"Unknown gaze-debug conditions: {unknown}")
    return [by_name[str(name)] for name in requested]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in records if row.get("status") == "ok"]
    attention: dict[str, dict[str, Any]] = {}
    for condition in sorted({name for row in ok for name in row.get("generation", {})}):
        traces = [
            trace
            for row in ok
            for trace in row["generation"].get(condition, {}).get("attention_trace", [])
        ]
        by_query = {}
        for query in sorted({row["query"] for row in traces}):
            values = [row["target_mass"] for row in traces if row["query"] == query]
            by_query[query] = {"n": len(values), "mean_target_mass": mean(values)}
        attention[condition] = by_query
    reward = {}
    for condition in sorted({name for row in ok for name in row.get("reward_logprobs", {})}):
        rows = [(row, row["reward_logprobs"][condition]) for row in ok if condition in row.get("reward_logprobs", {})]
        suc = [value["expected_reward"] for row, value in rows if row["example_id"].startswith("suc/")]
        fail = [value["expected_reward"] for row, value in rows if row["example_id"].startswith("fail/")]
        reward[condition] = {
            "n": len(rows),
            "mean_expected_reward": mean(value["expected_reward"] for _, value in rows),
            "mean_high_low_margin": mean(value["high_low_margin"] for _, value in rows),
            "succ_minus_fail_expected_reward": mean(suc) - mean(fail) if suc and fail else None,
        }
    probe = {}
    for condition in sorted({name for row in ok for name in row.get("object_probe", {})}):
        rows = [(row, row["object_probe"][condition]) for row in ok if condition in row.get("object_probe", {})]
        valid = []
        for row, value in rows:
            prediction = value.get("parsed", {}).get("match")
            expected = "YES" if row["example_id"].startswith("suc/") else "NO"
            if prediction in {"YES", "NO", "UNCERTAIN"}:
                valid.append(prediction == expected)
        probe[condition] = {"n": len(rows), "parsed_n": len(valid), "match_accuracy": mean(valid) if valid else None}
    zero_generation = []
    zero_logprob = []
    for row in ok:
        baseline = row.get("generation", {}).get("baseline", {})
        zero = row.get("generation", {}).get("zero_bias_hook", {})
        if baseline and zero:
            zero_generation.append(baseline.get("generated_token_ids") == zero.get("generated_token_ids"))
        baseline_lp = row.get("reward_logprobs", {}).get("baseline", {})
        zero_lp = row.get("reward_logprobs", {}).get("zero_bias_hook", {})
        if baseline_lp and zero_lp:
            zero_logprob.extend(
                abs(a["sequence_logprob"] - b["sequence_logprob"])
                for a, b in zip(baseline_lp["candidates"], zero_lp["candidates"])
            )
    return {
        "record_count": len(records),
        "ok_count": len(ok),
        "invalid_count": len(records) - len(ok),
        "zero_bias": {
            "generation_exact_count": sum(zero_generation),
            "generation_n": len(zero_generation),
            "max_abs_sequence_logprob_difference": max(zero_logprob) if zero_logprob else None,
        },
        "attention": attention,
        "object_probe": probe,
        "reward": reward,
    }


def run(config: dict[str, Any]) -> Path:
    debug = section(config, "gaze_debug")
    attention = section(config, "attention_steer")
    if attention.get("protocol") != ROBOREWARDBENCH_INTERLEAVED_IMAGE_SEQUENCE:
        raise ValueError("The initial debug runner requires the shared interleaved protocol")
    output = Path(debug["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "diagnostics.jsonl"
    requested = list(debug["example_ids"])
    samples_by_id = {row["example_id"]: row for row in read_jsonl(debug["cohort_inputs"])}
    missing = set(requested) - set(samples_by_id)
    if missing:
        raise ValueError(f"Debug IDs missing from frozen cohort: {sorted(missing)}")
    samples = [samples_by_id[value] for value in requested]
    _join_targets(samples, Path(debug["targets_path"]).resolve())
    top_k = int(debug.get("top_k", 8))
    gaze, ranking_fingerprint = _load_heads(Path(debug["ranking_path"]).resolve(), top_k)
    runtime = QwenAttentionRuntime(attention)
    random_heads = matched_random_heads(gaze, num_heads=runtime.num_heads, seed=int(debug.get("random_seed", 20260901)))
    conditions = _condition_spec(gaze, random_heads, float(debug.get("strong_bias", 12.0)))
    generation_conditions = select_condition_specs(
        conditions, debug.get("generation_conditions")
    )
    completed = {
        row["example_id"]
        for row in read_jsonl(records_path)
        if row.get("status") == "ok"
    } if records_path.exists() else set()
    manifest = {
        "schema_version": "gaze_debug_v1",
        "config_path": config.get("_config_path"),
        "model_fingerprint": artifact_fingerprint(attention["model_path"]),
        "model_path": str(Path(attention["model_path"]).resolve()),
        "runtime_attention_backend": getattr(runtime.model.config, "_attn_implementation", None),
        "num_layers": runtime.num_layers,
        "num_query_heads": runtime.num_heads,
        "num_kv_heads": int(getattr(getattr(runtime.model.config, "text_config", runtime.model.config), "num_key_value_heads")),
        "ranking_path": str(Path(debug["ranking_path"]).resolve()),
        "ranking_fingerprint": ranking_fingerprint,
        "gaze_heads": [item.__dict__ for item in gaze],
        "random_heads": [item.__dict__ for item in random_heads],
        "random_layer_counts_match": Counter(item.layer for item in gaze) == Counter(item.layer for item in random_heads),
        "example_ids": requested,
        "conditions": [name for name, *_ in generation_conditions],
        "deterministic_decoding": True,
        "no_git_provenance_call": True,
        "fingerprint": object_fingerprint({"config": config, "ranking": ranking_fingerprint, "ids": requested}),
    }
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    for index, sample in enumerate(samples, 1):
        if sample["example_id"] in completed:
            print(f"[{index}/{len(samples)}] skip completed {sample['example_id']}", flush=True)
            continue
        print(f"[{index}/{len(samples)}] prepare {sample['example_id']}", flush=True)
        base = {
            "schema_version": "gaze_debug_v1",
            "example_id": sample["example_id"],
            "video_sha256": sample["video_sha256"],
            "task": sample["task"],
            "target_phrase": sample.get("target_phrase"),
            "head_noun": sample.get("head_noun"),
            "attributes": sample.get("attributes"),
            "ranking_fingerprint": ranking_fingerprint,
        }
        try:
            prepared = runtime.prepare(sample)
            target = runtime.target_positions(sample, prepared)
            queries = query_position_groups(runtime, sample, prepared)
            overlays = write_token_overlays(runtime, sample, prepared, target, output)
            generation = {}
            max_new_tokens = int(debug.get("reward_max_new_tokens", 8))
            for name, heads, bias, scope, negative in generation_conditions:
                print(f"[{index}/{len(samples)}] generation {name}", flush=True)
                generation[name] = generate_with_attention_trace(
                    runtime,
                    sample,
                    prepared,
                    heads,
                    target,
                    bias=bias,
                    query_scope=scope,
                    negative_scope=negative,
                    max_new_tokens=max_new_tokens,
                    query_groups=queries,
                    trace_heads=heads if heads else gaze,
                )
            reward_logprobs = {}
            score_names = set(debug.get("logprob_conditions", [
                "baseline", "zero_bias_hook", "gaze_both_target_vs_all",
                "random_both_target_vs_all", "gaze_both_evidence_preserving",
            ]))
            for name, heads, bias, scope, negative in conditions:
                if name not in score_names:
                    continue
                print(f"[{index}/{len(samples)}] reward-logprob {name}", flush=True)
                reward_logprobs[name] = reward_sequence_logprobs(
                    runtime,
                    prepared,
                    heads,
                    target,
                    bias=bias,
                    query_scope=scope,
                    negative_scope=negative,
                    cached_decode=bool(debug.get("reward_cached_decode", False)),
                )
            probe_prepared = prepare_object_probe(runtime, sample, prepared)
            probe_target = runtime.target_positions(sample, probe_prepared)
            object_probe = {}
            probe_names = set(debug.get("probe_conditions", [
                "baseline", "gaze_both_target_vs_all", "gaze_both_evidence_preserving",
            ]))
            for name, heads, bias, scope, negative in conditions:
                if name not in probe_names:
                    continue
                print(f"[{index}/{len(samples)}] object-probe {name}", flush=True)
                object_probe[name] = generate_probe(
                    runtime,
                    probe_prepared,
                    heads,
                    probe_target,
                    bias=bias,
                    query_scope=scope,
                    negative_scope=negative,
                    max_new_tokens=int(debug.get("probe_max_new_tokens", 48)),
                )
            append_jsonl(
                records_path,
                {
                    **base,
                    "status": "ok",
                    "query_positions": queries,
                    "target_positions": target,
                    "visual_positions": prepared.visual_positions,
                    "spans": [span.__dict__ for span in prepared.spans],
                    "video_metadata": prepared.video_metadata,
                    "overlays": overlays,
                    "generation": generation,
                    "reward_logprobs": reward_logprobs,
                    "object_probe": object_probe,
                },
            )
            print(f"[{index}/{len(samples)}] completed {sample['example_id']}", flush=True)
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
            print(f"[{index}/{len(samples)}] invalid: {type(exc).__name__}: {exc}", flush=True)
    records = list(read_jsonl(records_path))
    summary_path = output / "summary.json"
    if not summary_path.exists():
        write_json(summary_path, summarize(records))
    print(summary_path, flush=True)
    return summary_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mydata_bench.gaze_debug")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
