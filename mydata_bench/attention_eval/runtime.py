from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..protocol import (
    IMAGE_LABELS,
    accumulate_incremental_progress,
    chat_messages,
    official_incremental_indices,
    parse_score,
    progress,
    system_prompt,
)
from ..video import extract_frame_at
from .masking import (
    QUERY_SCOPES,
    Head,
    ImageSpan,
    bbox_to_token_positions,
    make_attention_mask_hook,
    matched_wrong_position_set,
    resolve_negative_positions,
)


def find_contiguous_spans(values: Sequence[int], token_id: int) -> list[tuple[int, int]]:
    spans = []
    index = 0
    while index < len(values):
        if values[index] != token_id:
            index += 1
            continue
        start = index
        while index < len(values) and values[index] == token_id:
            index += 1
        spans.append((start, index))
    return spans


def infer_image_spans(inputs, config, image_paths: Sequence[str]) -> list[ImageSpan]:
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    image_token_id = int(getattr(config, "image_token_id", 151655))
    token_spans = find_contiguous_spans(ids, image_token_id)
    grids = inputs.get("image_grid_thw")
    if grids is None:
        raise RuntimeError("Processor did not return image_grid_thw")
    grid_list = [tuple(int(x) for x in row) for row in grids.detach().cpu().tolist()]
    if len(token_spans) != 8 or len(grid_list) != 8 or len(image_paths) != 8:
        raise RuntimeError(
            f"Eight-image alignment failed: spans={len(token_spans)}, "
            f"grids={len(grid_list)}, images={len(image_paths)}"
        )
    return [
        ImageSpan(IMAGE_LABELS[index], image_paths[index], start, end, grid_list[index])
        for index, (start, end) in enumerate(token_spans)
    ]


def sample_images(
    sample: dict[str, Any],
    blank_goal: str | Path,
    config: dict[str, Any],
    step: dict[str, Any] | None = None,
) -> list[str]:
    if step is not None:
        before_paths = step["before_paths"]
        after_paths = step["after_paths"]
        images = [
            step["reference_start_path"],
            str(Path(blank_goal).resolve()),
            before_paths["front"],
            before_paths["left_wrist"],
            before_paths["right_wrist"],
            after_paths["front"],
            after_paths["left_wrist"],
            after_paths["right_wrist"],
        ]
        sample["_runtime_before_bbox"] = list(step["before_bbox"])
        sample["_runtime_after_bbox"] = list(step["after_bbox"])
        sample["_runtime_before_frame_index"] = int(step["before_frame_index"])
        sample["_runtime_after_frame_index"] = int(step["after_frame_index"])
        missing = [path for path in images if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing attention input images: {missing}")
        return images
    first = sample["first"]["provenance"]["image_path"]
    last = sample["last"]["provenance"]["image_path"]
    provenance = sample["last"].get("provenance", {})
    views = provenance.get("view_endpoint_paths", {})
    view_paths = provenance.get("view_paths", {})
    required = {"front", "left_wrist", "right_wrist"}
    eval_mode = str(config.get("eval_mode", "forward"))
    if eval_mode not in {"forward", "incremental"}:
        raise ValueError("attention_eval.eval_mode must be forward or incremental")
    before_bbox = sample["first"].get("bbox")
    before_index = int(sample["first"].get("frame_index", 0))
    if required <= set(views):
        before_paths = {view: views[view]["first"] for view in required}
        if eval_mode == "incremental":
            track_path = provenance.get("tracking_path")
            track = json.loads(Path(track_path).read_text(encoding="utf-8"))
            track_rows = [
                row for row in track.get("frames", [])
                if isinstance(row.get("bbox"), list) and len(row["bbox"]) == 4
            ]
            if not track_rows:
                raise ValueError("track.json contains no valid bboxes")
            terminal = int(track.get("terminal_frame_index", track_rows[-1]["frame_index"]))
            requested = max(0, terminal - int(config.get("frame_interval", 20)))
            tracked = min(track_rows, key=lambda row: abs(int(row["frame_index"]) - requested))
            before_index = int(tracked["frame_index"])
            before_bbox = [float(value) for value in tracked["bbox"]]
            cache = Path(config["output_dir"]) / "runtime_frames" / sample["video_sha256"]
            before_paths = {}
            for view in sorted(required):
                _index, path = extract_frame_at(
                    view_paths[view],
                    cache / view / f"before_{before_index:06d}.png",
                    before_index,
                )
                before_paths[view] = path
        images = [
            views["front"]["first"],
            str(Path(blank_goal).resolve()),
            before_paths["front"],
            before_paths["left_wrist"],
            before_paths["right_wrist"],
            views["front"]["last"],
            views["left_wrist"]["last"],
            views["right_wrist"]["last"],
        ]
    else:
        if eval_mode != "forward":
            raise ValueError("incremental attention requires all three camera views")
        images = [first, str(Path(blank_goal).resolve()), first, first, first, last, last, last]
    if not isinstance(before_bbox, list) or len(before_bbox) != 4:
        raise ValueError("Missing before-frame bbox")
    sample["_runtime_before_bbox"] = [float(value) for value in before_bbox]
    sample["_runtime_after_bbox"] = [float(value) for value in sample["last"]["bbox"]]
    sample["_runtime_before_frame_index"] = before_index
    missing = [path for path in images if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attention input images: {missing}")
    return images


def incremental_steps(
    sample: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Materialize every adjacent hop in the official GRM protocol."""
    provenance = sample["last"].get("provenance", {})
    endpoints = provenance.get("view_endpoint_paths", {})
    view_paths = provenance.get("view_paths", {})
    required = {"front", "left_wrist", "right_wrist"}
    if not required <= set(endpoints) or not required <= set(view_paths):
        raise ValueError("official incremental attention requires all three camera views")
    track_path = provenance.get("tracking_path")
    if not isinstance(track_path, str) or not Path(track_path).is_file():
        raise ValueError("official incremental attention requires a track.json file")
    track = json.loads(Path(track_path).read_text(encoding="utf-8"))
    track_rows = [
        row for row in track.get("frames", [])
        if isinstance(row.get("bbox"), list) and len(row["bbox"]) == 4
    ]
    if not track_rows:
        raise ValueError("track.json contains no valid bboxes")
    terminal = int(track.get("terminal_frame_index", track_rows[-1]["frame_index"]))
    grounded_terminal = int(sample["last"].get("frame_index", terminal))
    if grounded_terminal != terminal:
        raise ValueError(
            "tracking/grounding terminal frame mismatch: "
            f"track={terminal}, grounding={grounded_terminal}"
        )
    indices = official_incremental_indices(
        terminal, int(config.get("frame_interval", 20))
    )
    if len(indices) < 2:
        raise ValueError("official incremental attention requires at least two frames")
    by_index = {int(row["frame_index"]): row for row in track_rows}

    def tracked_bbox(index: int) -> tuple[int, list[float]]:
        row = by_index.get(index)
        if row is None:
            row = min(track_rows, key=lambda item: abs(int(item["frame_index"]) - index))
        return int(row["frame_index"]), [float(value) for value in row["bbox"]]

    cache = Path(config["output_dir"]) / "runtime_frames" / sample["video_sha256"]

    def image_path(view: str, index: int) -> str:
        if index == 0:
            return str(Path(endpoints[view]["first"]).resolve())
        if index == terminal:
            return str(Path(endpoints[view]["last"]).resolve())
        _actual, path = extract_frame_at(
            view_paths[view], cache / view / f"frame_{index:06d}.png", index
        )
        return path

    steps = []
    for hop_index, (before_index, after_index) in enumerate(zip(indices, indices[1:])):
        before_bbox_index, before_bbox = tracked_bbox(before_index)
        after_bbox_index, after_bbox = tracked_bbox(after_index)
        steps.append(
            {
                "hop_index": hop_index,
                "before_frame_index": before_index,
                "after_frame_index": after_index,
                "before_bbox_frame_index": before_bbox_index,
                "after_bbox_frame_index": after_bbox_index,
                "before_bbox": before_bbox,
                "after_bbox": after_bbox,
                "before_paths": {
                    view: image_path(view, before_index) for view in sorted(required)
                },
                "after_paths": {
                    view: image_path(view, after_index) for view in sorted(required)
                },
                "reference_start_path": str(Path(endpoints["front"]["first"]).resolve()),
            }
        )
    return steps


class AttentionRuntime:
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.config = config
        dtype_name = config.get("dtype", "bfloat16")
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype_name]
        model_path = config["model_path"]
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.min_pixels = int(config.get("min_pixels", 12544))
            self.processor.image_processor.max_pixels = int(config.get("max_pixels", 76800))
        kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "device_map": config.get("device_map", "auto"),
        }
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=self.dtype, **kwargs
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=self.dtype, **kwargs
            )
        self.model.eval()
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.num_layers = int(text_config.num_hidden_layers)
        self.num_heads = int(text_config.num_attention_heads)
        self.spatial_merge_size = int(
            getattr(self.model.config, "spatial_merge_size", config.get("spatial_merge_size", 2))
        )

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
        raise RuntimeError("Cannot locate language-model decoder layers")

    def prepare(self, sample: dict[str, Any], step: dict[str, Any] | None = None):
        images_paths = sample_images(
            sample, self.config["blank_goal"], self.config, step=step
        )
        images = [Image.open(path).convert("RGB") for path in images_paths]
        prompt_mode = str(self.config.get("prompt_mode", "official"))
        messages = chat_messages(sample["task"], prompt_mode=prompt_mode)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        spans = infer_image_spans(inputs, self.model.config, images_paths)
        device = next(self.model.parameters()).device
        moved = {}
        for key, value in inputs.items():
            if not self.torch.is_tensor(value):
                moved[key] = value
            elif key in {"pixel_values", "pixel_values_videos"}:
                moved[key] = value.to(device=device, dtype=self.dtype)
            else:
                moved[key] = value.to(device=device)
        return moved, spans

    @property
    def prompt_mode(self) -> str:
        return str(self.config.get("prompt_mode", "official"))

    @property
    def prompt_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(system_prompt(self.prompt_mode).encode("utf-8")).hexdigest()

    def target_positions(
        self,
        sample: dict[str, Any],
        spans: list[ImageSpan],
        location: str = "after_cam_high",
    ) -> tuple[list[int], list[int], list[ImageSpan]]:
        if location == "after_all_duplicates":
            selected_spans = [span for span in spans if span.label.startswith("after_cam_")]
        elif location in {"before_after_cam_high", "after_before_cam_high"}:
            selected_spans = [
                span for span in spans
                if span.label in {"before_cam_high", "after_cam_high"}
            ]
        else:
            selected_spans = [span for span in spans if span.label == location]
        if not selected_spans:
            raise ValueError(f"No image span for intervention location {location}")
        selected = []
        image_positions = []
        for span in selected_spans:
            if span.label.startswith("before_cam_"):
                bbox = sample.get("_runtime_before_bbox")
            else:
                bbox = sample.get("_runtime_after_bbox", sample["last"]["bbox"])
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Missing bbox for intervention span {span.label}")
            with Image.open(span.path) as image:
                size = image.size
            selected.extend(
                bbox_to_token_positions(span, bbox, size, self.spatial_merge_size)
            )
            image_positions.extend(range(span.start, span.end))
        return sorted(set(selected)), sorted(set(image_positions)), selected_spans

    def collect_mass(self, sample: dict[str, Any]) -> dict[str, Any]:
        if str(self.config.get("eval_mode", "forward")) == "incremental":
            return self._collect_incremental_mass(sample)
        inputs, spans = self.prepare(sample)
        target, image_positions, _ = self.target_positions(sample, spans, self.config.get("intervention_location", "after_cam_high"))
        with self.torch.inference_mode():
            outputs = self.model(**inputs, output_attentions=True, use_cache=False)
        if outputs.attentions is None:
            raise RuntimeError("Eager model did not return attentions")
        raw = np.zeros((self.num_layers, self.num_heads), dtype=np.float64)
        image_mass = np.zeros_like(raw)
        query = int(inputs["input_ids"].shape[1] - 1)
        for layer, attention in enumerate(outputs.attentions):
            matrix = attention[0, :, query, :].detach().float().cpu().numpy()
            raw[layer] = matrix[:, target].sum(axis=-1)
            image_mass[layer] = matrix[:, image_positions].sum(axis=-1)
        fraction = len(target) / len(image_positions)
        excess = raw - fraction * image_mass
        return {
            "example_id": sample["example_id"],
            "video_sha256": sample["video_sha256"],
            "partition": "discovery",
            "query_mode": "last_prompt",
            "query_positions": [query],
            "bbox_positions": target,
            "image_positions": image_positions,
            "bbox_token_fraction": fraction,
            "raw_mass": raw.tolist(),
            "image_mass": image_mass.tolist(),
            "excess_mass": excess.tolist(),
            "status": "ok",
        }

    def incremental_plan(
        self, sample: dict[str, Any], location: str | None = None
    ) -> list[dict[str, Any]]:
        """Resolve per-hop images, tracked bboxes, and aligned token columns."""
        selected_location = str(
            location
            if location is not None
            else self.config.get("intervention_location", "after_cam_high")
        )
        plan = []
        for step in incremental_steps(sample, self.config):
            inputs, spans = self.prepare(sample, step=step)
            del inputs
            target, image_positions, target_spans = self.target_positions(
                sample, spans, selected_location
            )
            wrong_parts = []
            for target_span in target_spans:
                span_positions = [
                    position for position in target
                    if target_span.start <= position < target_span.end
                ]
                matched = matched_wrong_position_set(
                    target_span,
                    span_positions,
                    spatial_merge_size=self.spatial_merge_size,
                )
                if matched is None:
                    wrong_parts = []
                    break
                wrong_parts.extend(matched)
            wrong = (
                sorted(set(wrong_parts))
                if len(wrong_parts) == len(target)
                else None
            )
            plan.append(
                {
                    **step,
                    "target_positions": target,
                    "wrong_positions": wrong,
                    "image_positions": image_positions,
                    "target_span_labels": [span.label for span in target_spans],
                }
            )
        return plan

    def _collect_incremental_mass(self, sample: dict[str, Any]) -> dict[str, Any]:
        plan = self.incremental_plan(sample)
        raw_steps = []
        image_steps = []
        queries = []
        for step in plan:
            inputs, _spans = self.prepare(sample, step=step)
            with self.torch.inference_mode():
                outputs = self.model(**inputs, output_attentions=True, use_cache=False)
            if outputs.attentions is None:
                raise RuntimeError("Eager model did not return attentions")
            target = step["target_positions"]
            image_positions = step["image_positions"]
            raw = np.zeros((self.num_layers, self.num_heads), dtype=np.float64)
            image_mass = np.zeros_like(raw)
            query = int(inputs["input_ids"].shape[1] - 1)
            for layer, attention in enumerate(outputs.attentions):
                matrix = attention[0, :, query, :].detach().float().cpu().numpy()
                raw[layer] = matrix[:, target].sum(axis=-1)
                image_mass[layer] = matrix[:, image_positions].sum(axis=-1)
            raw_steps.append(raw)
            image_steps.append(image_mass)
            queries.append(query)
        raw_mean = np.mean(raw_steps, axis=0)
        image_mean = np.mean(image_steps, axis=0)
        fractions = [
            len(step["target_positions"]) / len(step["image_positions"])
            for step in plan
        ]
        excess_steps = [
            raw - fraction * image_mass
            for raw, image_mass, fraction in zip(raw_steps, image_steps, fractions)
        ]
        return {
            "example_id": sample["example_id"],
            "video_sha256": sample["video_sha256"],
            "partition": "discovery",
            "query_mode": "last_prompt",
            "query_positions": queries,
            "incremental_protocol": "official_accumulated_v1",
            "sampled_frame_indices": [
                plan[0]["before_frame_index"],
                *[step["after_frame_index"] for step in plan],
            ],
            "hop_count": len(plan),
            "bbox_positions": plan[-1]["target_positions"],
            "image_positions": plan[-1]["image_positions"],
            "bbox_token_fraction": float(np.mean(fractions)),
            "raw_mass": raw_mean.tolist(),
            "image_mass": image_mean.tolist(),
            "excess_mass": np.mean(excess_steps, axis=0).tolist(),
            "aggregation": "arithmetic_mean_over_official_incremental_hops",
            "status": "ok",
        }

    @contextmanager
    def steering_hooks(
        self,
        heads: Sequence[Head],
        selected_positions: Sequence[int],
        other_positions: Sequence[int],
        bias: float,
        diagnostics: dict,
        query_scope: str,
    ):
        by_layer: dict[int, list[int]] = {}
        for head in heads:
            by_layer.setdefault(head.layer, []).append(head.head)
        handles = []
        try:
            diagnostics["layers"] = sorted(by_layer)
            for layer, head_indices in by_layer.items():
                layer_diagnostics = {}
                hook = make_attention_mask_hook(
                    head_indices,
                    selected_positions,
                    other_positions,
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
        heads: Sequence[Head] = (),
        selected_positions: Sequence[int] = (),
        image_positions: Sequence[int] = (),
        bias: float = 0.0,
        query_scope: str | None = None,
        negative_scope: str | None = None,
        step: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        inputs, spans = self.prepare(sample, step=step)
        scope = str(
            query_scope
            if query_scope is not None
            else self.config.get("steering_query_scope", "all")
        )
        if scope not in QUERY_SCOPES:
            choices = ", ".join(sorted(QUERY_SCOPES))
            raise ValueError(f"Unknown steering query scope {scope!r}; choose one of {choices}")
        negative = str(
            negative_scope
            if negative_scope is not None
            else self.config.get("negative_scope", "other_spans")
        )
        other, selected_span_labels = resolve_negative_positions(
            spans, selected_positions, negative
        )
        diagnostics: dict[str, Any] = {
            "query_scope": scope,
            "negative_scope": negative,
            "selected_span_labels": selected_span_labels,
            "all_visual_span_labels": [span.label for span in spans],
            "selected_token_count": len(set(selected_positions)),
            "negative_token_count": len(other),
            "selected_negative_disjoint": not bool(
                set(selected_positions) & set(other)
            ),
        }
        context = (
            self.steering_hooks(
                heads,
                selected_positions,
                other,
                bias,
                diagnostics,
                scope,
            )
            if heads and bias != 0
            else _nullcontext()
        )
        with context:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=int(self.config.get("max_new_tokens", 16)),
                    # Attention interventions need an exact, per-sample
                    # baseline.  We therefore use greedy decoding rather than
                    # the raw-evaluation sampling endpoint; this is recorded
                    # in every steering manifest as an attention-runtime
                    # protocol, not conflated with the vLLM raw baseline.
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
        generated = output.sequences[0, inputs["input_ids"].shape[1] :]
        # Remove tokenizer control tokens such as <|im_end|>, then apply the
        # same strict full-string score parser used by raw evaluation.
        text = self.processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()
        signed = parse_score(text)
        bbox_mass = self._generated_bbox_mass(output.attentions, heads, selected_positions)
        image_heatmap = self._generated_image_heatmap(
            output.attentions, heads, image_positions
        )
        diagnostics["bbox_attention_mass"] = bbox_mass
        diagnostics["hook_active"] = bool(heads and bias != 0)
        return {
            "raw_output": text,
            "signed_score": signed,
            "hook_diagnostics": diagnostics,
            "spans": [span.__dict__ for span in spans],
            "image_heatmap": image_heatmap,
        }

    def generate_incremental(
        self,
        sample: dict[str, Any],
        plan: list[dict[str, Any]],
        *,
        heads: Sequence[Head] = (),
        position_kind: str = "target",
        bias: float = 0.0,
        query_scope: str | None = None,
        negative_scope: str | None = None,
    ) -> dict[str, Any]:
        """Run one intervention condition on every hop and accumulate progress."""
        if position_kind not in {"target", "wrong"}:
            raise ValueError("position_kind must be 'target' or 'wrong'")
        if not plan:
            raise ValueError("incremental plan is empty")
        accumulated = None
        hop_rows = []
        masses = []
        final_result = None
        for step in plan:
            positions = step[f"{position_kind}_positions"]
            if positions is None:
                raise ValueError(
                    "equal-size non-overlapping wrong region unavailable for at least one hop"
                )
            result = self.generate(
                sample,
                heads=heads,
                selected_positions=positions,
                image_positions=step["image_positions"],
                bias=bias,
                query_scope=query_scope,
                negative_scope=negative_scope,
                step=step,
            )
            hop_score = float(result["signed_score"])
            accumulated = accumulate_incremental_progress(accumulated, hop_score)
            mass = result.get("hook_diagnostics", {}).get("bbox_attention_mass")
            if mass is not None:
                masses.append(float(mass))
            hop_rows.append(
                {
                    "hop_index": int(step["hop_index"]),
                    "before_frame_index": int(step["before_frame_index"]),
                    "after_frame_index": int(step["after_frame_index"]),
                    "before_bbox_frame_index": int(step["before_bbox_frame_index"]),
                    "after_bbox_frame_index": int(step["after_bbox_frame_index"]),
                    "raw_output": result["raw_output"],
                    "hop_score": hop_score,
                    "accumulated_progress_unclipped": accumulated,
                    "target_token_count": len(step["target_positions"]),
                    "selected_token_count": len(positions),
                }
            )
            final_result = result
        assert accumulated is not None and final_result is not None
        reported = progress(accumulated)
        diagnostics = dict(final_result.get("hook_diagnostics", {}))
        diagnostics.update(
            {
                "bbox_attention_mass": float(np.mean(masses)) if masses else None,
                "bbox_attention_mass_aggregation": "mean_over_incremental_hops",
                "incremental_protocol": "official_accumulated_v1",
                "incremental_steering_scope": "all_hops",
                "hop_count": len(plan),
                "position_kind": position_kind,
            }
        )
        return {
            "raw_output": final_result["raw_output"],
            # Steering estimands compare final episode progress, not the final
            # local hop.  Preserve the latter explicitly below.
            "signed_score": reported,
            "progress": reported,
            "last_hop_score": hop_rows[-1]["hop_score"],
            "accumulated_progress_unclipped": accumulated,
            "incremental_protocol": "official_accumulated_v1",
            "sampled_frame_indices": [
                plan[0]["before_frame_index"],
                *[step["after_frame_index"] for step in plan],
            ],
            "hop_count": len(plan),
            "incremental_steps": hop_rows,
            "hook_diagnostics": diagnostics,
            "spans": final_result.get("spans", []),
            "image_heatmap": final_result.get("image_heatmap"),
        }

    def _generated_bbox_mass(self, attentions, heads, positions) -> float | None:
        if not attentions or not positions or not heads:
            return None
        first_step = attentions[0]
        values = []
        for head in heads:
            matrix = first_step[head.layer][0, head.head]
            query_row = matrix[-1]
            valid = [position for position in positions if position < query_row.shape[-1]]
            if valid:
                values.append(float(query_row[valid].sum().detach().float().cpu()))
        return float(np.mean(values)) if values else None

    def _generated_image_heatmap(self, attentions, heads, positions) -> list[float] | None:
        if not attentions or not positions or not heads:
            return None
        first_step = attentions[0]
        vectors = []
        for head in heads:
            matrix = first_step[head.layer][0, head.head]
            query_row = matrix[-1]
            if max(positions) < query_row.shape[-1]:
                vectors.append(
                    query_row[list(positions)].detach().float().cpu().numpy()
                )
        return np.mean(vectors, axis=0).tolist() if vectors else None


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False
