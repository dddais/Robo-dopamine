from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..protocol import IMAGE_LABELS, chat_messages, parse_score, system_prompt
from .masking import (
    QUERY_SCOPES,
    VISUAL_SCOPES,
    Head,
    ImageSpan,
    bbox_to_token_positions,
    make_attention_mask_hook,
    matched_wrong_position_set,
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


def sample_images(sample: dict[str, Any], blank_goal: str | Path) -> list[str]:
    configured = sample.get("image_paths")
    if configured is not None:
        if not isinstance(configured, list) or len(configured) != len(IMAGE_LABELS):
            raise ValueError("image_paths must contain the canonical eight GRM slots")
        images = [str(Path(path).resolve()) for path in configured]
        missing = [path for path in images if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing attention input images: {missing}")
        return images
    layout = str(sample.get("input_layout", "legacy_single_view_duplicates"))
    if layout != "legacy_single_view_duplicates":
        raise ValueError(
            "Non-legacy GRM samples must provide eight explicit image_paths; "
            f"got input_layout={layout!r}"
        )
    first = sample["first"]["provenance"]["image_path"]
    last = sample["last"]["provenance"]["image_path"]
    images = [first, str(Path(blank_goal).resolve()), first, first, first, last, last, last]
    missing = [path for path in images if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attention input images: {missing}")
    return images


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

    def prepare(self, sample: dict[str, Any]):
        images_paths = sample_images(sample, self.config["blank_goal"])
        images = []
        try:
            for path in images_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            prompt_mode = str(self.config.get("prompt_mode", "official"))
            messages = chat_messages(sample["task"], prompt_mode=prompt_mode)
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        finally:
            for image in images:
                image.close()
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
        visual_scope: str = "target_slot_only",
    ) -> tuple[list[int], list[int], list[ImageSpan]]:
        if visual_scope not in VISUAL_SCOPES:
            choices = ", ".join(sorted(VISUAL_SCOPES))
            raise ValueError(
                f"Unknown visual scope {visual_scope!r}; choose one of {choices}"
            )
        selected_spans = (
            [span for span in spans if span.label.startswith("after_cam_")]
            if location == "after_all_duplicates"
            else [span for span in spans if span.label == location]
        )
        if not selected_spans:
            raise ValueError(f"No image span for intervention location {location}")
        bbox = (
            sample["last_bbox"]
            if "last_bbox" in sample
            else sample["last"]["bbox"]
        )
        image_path = (
            sample["last_image_path"]
            if "last_image_path" in sample
            else sample["last"]["provenance"]["image_path"]
        )
        with Image.open(image_path) as image:
            size = image.size
        selected: list[int] = []
        for span in selected_spans:
            selected.extend(
                bbox_to_token_positions(span, bbox, size, self.spatial_merge_size)
            )
        universe_spans = (
            selected_spans if visual_scope == "target_slot_only" else spans
        )
        image_positions = [
            position
            for span in universe_spans
            for position in range(span.start, span.end)
        ]
        return sorted(set(selected)), image_positions, selected_spans

    def wrong_control_positions(
        self,
        sample: dict[str, Any],
        spans: list[ImageSpan],
        target_positions: Sequence[int],
        location: str = "after_cam_high",
    ) -> tuple[list[int], str]:
        _target, _visual, selected_spans = self.target_positions(
            sample, spans, location, "target_slot_only"
        )
        if len(selected_spans) != 1:
            raise ValueError("Wrong-region control requires one target image span")
        span = selected_spans[0]
        wrong_bbox = sample.get("wrong_region_bbox")
        if wrong_bbox is not None:
            image_path = (
                sample["last_image_path"]
                if "last_image_path" in sample
                else sample["last"]["provenance"]["image_path"]
            )
            with Image.open(image_path) as image:
                size = image.size
            wrong = bbox_to_token_positions(
                span, wrong_bbox, size, self.spatial_merge_size
            )
            source = "audited_same_target_image"
        else:
            wrong = matched_wrong_position_set(
                span,
                target_positions,
                spatial_merge_size=self.spatial_merge_size,
            )
            source = "same_target_span_farthest_translated_footprint"
        if (
            wrong is None
            or not wrong
            or len(wrong) != len(set(target_positions))
            or set(wrong) & set(target_positions)
        ):
            raise ValueError(
                "No equal-size disjoint wrong region exists in the target span"
            )
        return list(wrong), source


    def collect_mass(self, sample: dict[str, Any]) -> dict[str, Any]:
        inputs, spans = self.prepare(sample)
        visual_scope = str(
            self.config.get("ranking_visual_scope", "target_slot_only")
        )
        target, image_positions, _ = self.target_positions(
            sample,
            spans,
            str(self.config.get("intervention_location", "after_cam_high")),
            visual_scope,
        )
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
            "visual_scope": visual_scope,
            "bbox_token_fraction": fraction,
            "raw_mass": raw.tolist(),
            "image_mass": image_mass.tolist(),
            "excess_mass": excess.tolist(),
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
        prepared: tuple[dict[str, Any], list[ImageSpan]] | None = None,
        heads: Sequence[Head] = (),
        selected_positions: Sequence[int] = (),
        image_positions: Sequence[int] = (),
        bias: float = 0.0,
        query_scope: str | None = None,
    ) -> dict[str, Any]:
        inputs, spans = prepared if prepared is not None else self.prepare(sample)
        scope = str(
            query_scope
            if query_scope is not None
            else self.config.get("steering_query_scope", "all")
        )
        if scope not in QUERY_SCOPES:
            choices = ", ".join(sorted(QUERY_SCOPES))
            raise ValueError(f"Unknown steering query scope {scope!r}; choose one of {choices}")
        visual_scope = str(
            self.config.get("intervention_visual_scope", "target_slot_only")
        )
        _target, expected_visual, _selected_spans = self.target_positions(
            sample,
            spans,
            str(self.config.get("intervention_location", "after_cam_high")),
            visual_scope,
        )
        if list(image_positions) != expected_visual:
            raise ValueError(
                "Supplied image positions differ from configured intervention universe"
            )
        if not set(selected_positions) < set(image_positions):
            raise ValueError(
                "Selected target/control positions must be a proper subset of visual universe"
            )
        other = sorted(set(image_positions) - set(selected_positions))
        diagnostics: dict[str, Any] = {
            "query_scope": scope,
            "visual_scope": visual_scope,
            "selected_token_count": len(set(selected_positions)),
            "visual_token_count": len(set(image_positions)),
            "negative_visual_token_count": len(other),
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
        capture_attentions = bool(
            self.config.get("capture_generation_attentions", True)
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
                    output_attentions=capture_attentions,
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
        attentions = getattr(output, "attentions", None)
        bbox_mass = self._generated_bbox_mass(attentions, heads, selected_positions)
        image_heatmap = self._generated_image_heatmap(
            attentions, heads, image_positions
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
