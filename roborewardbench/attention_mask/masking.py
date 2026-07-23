"""Visual-token mapping and per-head attention-mask interventions."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


IMAGE_LABELS = (
    "reference_start",
    "reference_end",
    "before_cam_high",
    "before_cam_left_wrist",
    "before_cam_right_wrist",
    "after_cam_high",
    "after_cam_left_wrist",
    "after_cam_right_wrist",
)
ROLE_LABELS = {
    "before": IMAGE_LABELS[2:5],
    "after": IMAGE_LABELS[5:8],
    "both": IMAGE_LABELS[2:8],
    # Compatibility scope for the original success-trajectory experiment.
    "after_high": ("after_cam_high",),
}
INTERVENTION_MODES = ("boost_suppress", "suppress_image")


@dataclass(frozen=True)
class ImageSpan:
    label: str
    path: str
    start: int
    end: int
    grid_thw: tuple[int, int, int]

    @property
    def token_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Head:
    layer: int
    head: int


@dataclass(frozen=True)
class PositionSet:
    target: tuple[int, ...]
    other_image: tuple[int, ...]
    per_span_target: Mapping[str, tuple[int, ...]]


def merged_grid_shape(span: ImageSpan, spatial_merge_size: int) -> tuple[int, int, int]:
    merge = max(1, int(spatial_merge_size))
    t, height, width = span.grid_thw
    if height % merge or width % merge:
        raise ValueError(
            f"{span.label}: grid {span.grid_thw} is not divisible by spatial_merge_size={merge}"
        )
    shape = int(t), int(height // merge), int(width // merge)
    if span.token_count != shape[0] * shape[1] * shape[2]:
        raise ValueError(
            f"{span.label}: token span length {span.token_count} disagrees with merged grid {shape}"
        )
    return shape


def bbox_to_token_positions(
    span: ImageSpan,
    bbox_xyxy: Sequence[float],
    image_size: tuple[int, int],
    spatial_merge_size: int,
    *,
    method: str = "intersection",
) -> list[int]:
    """Map a pixel bbox to absolute LM image-token positions.

    ``intersection`` selects every grid cell touched by the box, which avoids
    dropping small robot objects that happen to fall between token centers.
    ``center`` is provided for exact comparisons with the gaze-heads utility.
    """

    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four numbers")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    width, height = (int(image_size[0]), int(image_size[1]))
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 <= x1 or y2 <= y1:
        return []

    t, grid_h, grid_w = merged_grid_shape(span, spatial_merge_size)
    if method == "intersection":
        x_edges = np.linspace(0.0, float(width), grid_w + 1)
        y_edges = np.linspace(0.0, float(height), grid_h + 1)
        x_keep = (x_edges[:-1] < x2) & (x_edges[1:] > x1)
        y_keep = (y_edges[:-1] < y2) & (y_edges[1:] > y1)
        cell_mask = np.outer(y_keep, x_keep)
    elif method == "center":
        x_centers = (np.arange(grid_w) + 0.5) * width / grid_w
        y_centers = (np.arange(grid_h) + 0.5) * height / grid_h
        cx, cy = np.meshgrid(x_centers, y_centers)
        cell_mask = (x1 <= cx) & (cx <= x2) & (y1 <= cy) & (cy <= y2)
    else:
        raise ValueError("method must be 'intersection' or 'center'")

    flat = np.tile(cell_mask.reshape(-1), t)
    return [span.start + index for index, keep in enumerate(flat.tolist()) if keep]


def _labels_for_role(role: str) -> tuple[str, ...]:
    if role not in ROLE_LABELS:
        raise ValueError(f"Unknown target role {role!r}; expected one of {tuple(ROLE_LABELS)}")
    return ROLE_LABELS[role]


def target_position_set(
    spans: Sequence[ImageSpan],
    *,
    before_bbox: Sequence[float],
    after_bbox: Sequence[float],
    before_image_size: tuple[int, int],
    after_image_size: tuple[int, int],
    spatial_merge_size: int,
    target_role: str = "both",
    method: str = "intersection",
) -> PositionSet:
    """Apply endpoint bboxes to the image spans selected by ``target_role``."""

    span_by_label = {span.label: span for span in spans}
    if len(span_by_label) != len(spans):
        raise ValueError("Duplicate image-span labels are not allowed")
    expected = set(IMAGE_LABELS)
    missing = expected - set(span_by_label)
    unexpected = set(span_by_label) - expected
    if missing or unexpected:
        raise ValueError(
            f"Image-span labels disagree with the eight-image protocol: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    labels = _labels_for_role(target_role)
    per_span: dict[str, tuple[int, ...]] = {}
    all_image: list[int] = []
    target: list[int] = []
    for label in labels:
        span = span_by_label[label]
        role = "before" if label.startswith("before_") else "after"
        positions = bbox_to_token_positions(
            span,
            before_bbox if role == "before" else after_bbox,
            before_image_size if role == "before" else after_image_size,
            spatial_merge_size,
            method=method,
        )
        if not positions:
            raise ValueError(f"{label}: bbox maps to zero visual tokens")
        per_span[label] = tuple(positions)
        target.extend(positions)
        all_image.extend(range(span.start, span.end))
    target_set = set(target)
    other = [position for position in all_image if position not in target_set]
    return PositionSet(
        target=tuple(sorted(target_set)),
        other_image=tuple(sorted(other)),
        per_span_target=per_span,
    )


def _rectangular_wrong_local_positions(
    span: ImageSpan,
    target_absolute: Sequence[int],
    spatial_merge_size: int,
    rng: random.Random,
) -> list[int]:
    """Choose a same-shape, non-overlapping rectangle on the merged token grid."""

    t, height, width = merged_grid_shape(span, spatial_merge_size)
    if t != 1:
        raise ValueError("Wrong-region control currently requires single-frame image spans")
    target_local = sorted({int(value) - span.start for value in target_absolute})
    if not target_local:
        raise ValueError(f"{span.label}: target positions are empty")
    target_set = set(target_local)
    rows = [index // width for index in target_local]
    cols = [index % width for index in target_local]
    box_h = max(rows) - min(rows) + 1
    box_w = max(cols) - min(cols) + 1

    candidates: list[list[int]] = []
    for row in range(0, height - box_h + 1):
        for col in range(0, width - box_w + 1):
            local = [
                (row + dr) * width + col + dc
                for dr in range(box_h)
                for dc in range(box_w)
            ]
            if len(local) == len(target_local) and not (set(local) & target_set):
                candidates.append(local)
    if candidates:
        return rng.choice(candidates)

    complement = [index for index in range(height * width) if index not in target_set]
    if len(complement) < len(target_local):
        raise ValueError(
            f"{span.label}: cannot construct a non-overlapping wrong region with "
            f"{len(target_local)} tokens in a {height}x{width} grid"
        )
    target_row = sum(rows) / len(rows)
    target_col = sum(cols) / len(cols)
    complement.sort(
        key=lambda index: (
            -((index // width - target_row) ** 2 + (index % width - target_col) ** 2),
            index,
        )
    )
    return complement[: len(target_local)]


def matched_wrong_position_set(
    spans: Sequence[ImageSpan],
    target: PositionSet,
    *,
    spatial_merge_size: int,
    seed: int,
) -> PositionSet:
    """Construct a spatially contiguous, token-count-matched wrong-region control."""

    span_by_label = {span.label: span for span in spans}
    wrong: list[int] = []
    per_span: dict[str, tuple[int, ...]] = {}
    all_image: list[int] = []
    for offset, (label, true_positions) in enumerate(sorted(target.per_span_target.items())):
        span = span_by_label[label]
        rng = random.Random(int(seed) + 104729 * (offset + 1))
        local = _rectangular_wrong_local_positions(
            span, true_positions, spatial_merge_size, rng
        )
        positions = tuple(sorted(span.start + index for index in local))
        if set(positions) & set(true_positions):
            raise AssertionError(f"{label}: wrong region overlaps target")
        if len(positions) != len(true_positions):
            raise AssertionError(f"{label}: wrong-region token count is not matched")
        per_span[label] = positions
        wrong.extend(positions)
        all_image.extend(range(span.start, span.end))
    wrong_set = set(wrong)
    other = [position for position in all_image if position not in wrong_set]
    return PositionSet(
        target=tuple(sorted(wrong_set)),
        other_image=tuple(sorted(other)),
        per_span_target=per_span,
    )


def intervention_positions(
    mode: str,
    positions: PositionSet,
) -> tuple[list[int], list[int]]:
    """Return ``(suppress, boost)`` positions for the selected recipe."""

    if mode == "boost_suppress":
        return list(positions.other_image), list(positions.target)
    if mode == "suppress_image":
        return list(positions.other_image), []
    raise ValueError(f"Unknown intervention {mode!r}; expected one of {INTERVENTION_MODES}")


def validate_heads(
    heads: Iterable[Head | tuple[int, int]],
    *,
    num_layers: int,
    num_heads: int,
) -> list[Head]:
    validated: list[Head] = []
    seen: set[tuple[int, int]] = set()
    for value in heads:
        head = value if isinstance(value, Head) else Head(int(value[0]), int(value[1]))
        if not 0 <= head.layer < num_layers:
            raise ValueError(f"Layer index out of range: {head.layer}")
        if not 0 <= head.head < num_heads:
            raise ValueError(f"Head index out of range: {head.head}")
        key = (head.layer, head.head)
        if key not in seen:
            seen.add(key)
            validated.append(head)
    if not validated:
        raise ValueError("At least one attention head is required")
    return validated


def group_heads_by_layer(heads: Iterable[Head | tuple[int, int]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for value in heads:
        head = value if isinstance(value, Head) else Head(int(value[0]), int(value[1]))
        grouped.setdefault(head.layer, []).append(head.head)
    return {layer: sorted(set(indices)) for layer, indices in sorted(grouped.items())}


def make_attention_mask_hook(
    *,
    head_indices: Sequence[int],
    suppress_positions: Sequence[int],
    boost_positions: Sequence[int],
    num_query_heads: int,
    swap_bias: float,
    decode_only: bool = False,
):
    """Create a Qwen3-VL eager-attention pre-hook.

    The incoming causal mask is ``[batch, 1, query, key]``.  Adding a
    ``[1, num_query_heads, 1, key]`` bias broadcasts the mask over heads while
    modifying only the explicitly selected head rows.
    """

    import torch

    indices = sorted({int(value) for value in head_indices})
    if any(value < 0 or value >= int(num_query_heads) for value in indices):
        raise ValueError("head_indices contains an out-of-range query head")
    suppress = sorted({int(value) for value in suppress_positions})
    boost = sorted({int(value) for value in boost_positions})
    if set(suppress) & set(boost):
        raise ValueError("suppress_positions and boost_positions overlap")
    if any(value < 0 for value in suppress + boost):
        raise ValueError("Attention-mask positions must be non-negative")
    if not indices or not (suppress or boost) or float(swap_bias) == 0.0:
        def no_op(_module, _args, _kwargs):
            return None

        return no_op

    base_len = max(suppress + boost) + 1
    base = torch.zeros((1, int(num_query_heads), 1, base_len), dtype=torch.float32)
    head_tensor = torch.tensor(indices, dtype=torch.long)
    if suppress:
        position_tensor = torch.tensor(suppress, dtype=torch.long)
        base[0, head_tensor[:, None], 0, position_tensor[None, :]] = -float(swap_bias)
    if boost:
        position_tensor = torch.tensor(boost, dtype=torch.long)
        base[0, head_tensor[:, None], 0, position_tensor[None, :]] = float(swap_bias)
    cache: dict[tuple[Any, Any], Any] = {}

    def hook(_module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            return None
        if mask.ndim != 4:
            raise RuntimeError(
                f"Expected a 4-D eager attention mask, received shape {tuple(mask.shape)}"
            )
        mask_heads = int(mask.shape[1])
        if mask_heads not in {1, int(num_query_heads)}:
            raise RuntimeError(
                "Expected eager mask head dimension 1 or num_query_heads="
                f"{int(num_query_heads)}, received {mask_heads}"
            )
        if decode_only and int(mask.shape[-2]) > 1:
            return None
        key = (mask.device, mask.dtype)
        if key not in cache:
            cache[key] = base.to(device=mask.device, dtype=mask.dtype)
        bias = cache[key]
        key_length = int(mask.shape[-1])
        if key_length < base_len:
            applied = bias[..., :key_length]
        elif key_length > base_len:
            applied = torch.nn.functional.pad(bias, (0, key_length - base_len))
        else:
            applied = bias
        updated = dict(kwargs)
        updated["attention_mask"] = mask + applied
        return args, updated

    return hook


def language_model_layers(model):
    try:
        return model.model.language_model.layers
    except AttributeError as exc:
        raise AttributeError(
            "Expected Qwen3-VL layers at model.model.language_model.layers"
        ) from exc


@contextlib.contextmanager
def registered_mask_hooks(
    model,
    *,
    heads: Sequence[Head | tuple[int, int]],
    suppress_positions: Sequence[int],
    boost_positions: Sequence[int],
    num_query_heads: int,
    swap_bias: float,
    decode_only: bool,
) -> Iterator[None]:
    grouped = group_heads_by_layer(heads)
    layers = language_model_layers(model)
    handles = []
    try:
        for layer, head_indices in grouped.items():
            if not 0 <= layer < len(layers):
                raise ValueError(f"Layer index out of range: {layer}")
            hook = make_attention_mask_hook(
                head_indices=head_indices,
                suppress_positions=suppress_positions,
                boost_positions=boost_positions,
                num_query_heads=num_query_heads,
                swap_bias=swap_bias,
                decode_only=decode_only,
            )
            handles.append(
                layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
            )
        yield
    finally:
        for handle in handles:
            handle.remove()
