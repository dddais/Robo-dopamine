"""Minimal Transformers runtime shared by head ranking and steering."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .masking import IMAGE_LABELS, ImageSpan


SCORE_PATTERN = re.compile(
    r"<score>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%\s*</score>",
    flags=re.IGNORECASE,
)


def resolve_dtype(torch, value: str):
    normalized = str(value).lower()
    if normalized == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dtype: {value}")
    return aliases[normalized]


def load_grm(
    model_path: str | Path,
    *,
    dtype: str = "auto",
    device_map: str = "none",
    max_pixels: int | None = 76800,
    min_pixels: int | None = 12544,
    output_attentions: bool = False,
):
    """Load the GRM with eager attention, which is required by mask hooks."""

    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    torch_dtype = resolve_dtype(torch, dtype)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        if max_pixels is not None:
            processor.image_processor.max_pixels = int(max_pixels)
        if min_pixels is not None:
            processor.image_processor.min_pixels = int(min_pixels)

    kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "device_map": None if device_map == "none" else device_map,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch_dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch_dtype, **kwargs
        )
    except Exception as first_error:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(config, "model_type", "") != "qwen3_vl":
            raise
        from transformers import Qwen3VLForConditionalGeneration

        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, dtype=torch_dtype, **kwargs
            )
        except TypeError:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch_dtype, **kwargs
            )
        except Exception:
            raise first_error

    model.eval()
    model.config.output_attentions = bool(output_attentions)
    if kwargs["device_map"] is None and torch.cuda.is_available():
        model.to("cuda")
    return torch, model, processor, torch_dtype


def model_dimensions(model) -> tuple[int, int, int]:
    text_config = getattr(model.config, "text_config", model.config)
    vision_config = getattr(model.config, "vision_config", None)
    num_layers = int(getattr(text_config, "num_hidden_layers"))
    num_heads = int(getattr(text_config, "num_attention_heads"))
    spatial_merge = int(getattr(vision_config, "spatial_merge_size", 2))
    return num_layers, num_heads, spatial_merge


def build_grm_prompt(processor, task: str) -> str:
    # The canonical prompt is maintained by the existing attention toolchain.
    # Import lazily so dataset/unit-test utilities do not import matplotlib.
    from scan_localization_heads_best import build_prompt

    return build_prompt(processor, task, analysis_suffix=None, score_suffix=False)


def _contiguous_spans(values: Sequence[int], token_id: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(values):
        if int(values[index]) != int(token_id):
            index += 1
            continue
        start = index
        while index < len(values) and int(values[index]) == int(token_id):
            index += 1
        spans.append((start, index))
    return spans


def infer_image_spans(
    inputs: Mapping[str, Any],
    config,
    image_paths: Sequence[str],
) -> list[ImageSpan]:
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    image_token_id = int(getattr(config, "image_token_id", 151655))
    token_spans = _contiguous_spans(ids, image_token_id)
    grids = inputs.get("image_grid_thw")
    if grids is None:
        raise RuntimeError("Processor did not return image_grid_thw")
    grid_values = [tuple(int(value) for value in row) for row in grids.detach().cpu().tolist()]
    if len(image_paths) != len(IMAGE_LABELS):
        raise ValueError(f"Expected eight image paths, received {len(image_paths)}")
    if len(token_spans) != len(image_paths) or len(grid_values) != len(image_paths):
        raise RuntimeError(
            f"Image-token alignment failure: paths={len(image_paths)}, "
            f"spans={len(token_spans)}, grids={len(grid_values)}"
        )
    return [
        ImageSpan(
            label=IMAGE_LABELS[index],
            path=str(image_paths[index]),
            start=token_span[0],
            end=token_span[1],
            grid_thw=grid_values[index],
        )
        for index, token_span in enumerate(token_spans)
    ]


def move_inputs_to_device(torch, inputs: Mapping[str, Any], device, dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif key in {"pixel_values", "pixel_values_videos"}:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def prepare_inputs(torch, model, processor, item: Mapping[str, Any], dtype):
    from PIL import Image

    image_paths = [str(value) for value in item["image"]]
    images = []
    try:
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
        prompt = build_grm_prompt(processor, str(item["task"]))
        inputs = processor(text=[prompt], images=images, return_tensors="pt")
    finally:
        for image in images:
            image.close()
    spans = infer_image_spans(inputs, model.config, image_paths)
    device = next(model.parameters()).device
    return move_inputs_to_device(torch, inputs, device, dtype), spans


def last_prompt_query_position(inputs: Mapping[str, Any], spans: Sequence[ImageSpan], config) -> int:
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    last_image_end = max(span.end for span in spans)
    special = {
        int(getattr(config, "image_token_id", 151655)),
        int(getattr(config, "video_token_id", 151656)),
        int(getattr(config, "vision_start_token_id", 151652)),
        int(getattr(config, "vision_end_token_id", 151653)),
    }
    candidates = [
        index
        for index in range(last_image_end, len(ids))
        if int(ids[index]) not in special
    ]
    if not candidates:
        raise RuntimeError("No non-special prompt token follows the image spans")
    return candidates[-1]


def generate_score(
    torch,
    model,
    processor,
    inputs: Mapping[str, Any],
    *,
    max_new_tokens: int = 16,
) -> tuple[float | None, str, str | None]:
    """Greedily generate a score and return ``(score, raw_text, parse_error)``."""

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
            output_attentions=False,
            return_dict_in_generate=True,
        )
    prompt_length = int(inputs["input_ids"].shape[1])
    generated = output.sequences[0, prompt_length:]
    raw = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    matches = SCORE_PATTERN.findall(raw)
    if len(matches) != 1 or SCORE_PATTERN.fullmatch(raw) is None:
        return None, raw, "expected exactly one <score>NUMBER%</score> line"
    try:
        percentage = float(matches[0])
    except ValueError:
        return None, raw, "score is not numeric"
    if not -100.0 <= percentage <= 100.0:
        return None, raw, "score is outside [-100, 100]"
    return percentage / 100.0, raw, None


def ensure_blank_goal(path: str | Path, size: int = 224) -> Path:
    from PIL import Image

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        image = Image.new("RGB", (int(size), int(size)), (128, 128, 128))
        try:
            image.save(destination)
        finally:
            image.close()
    return destination

