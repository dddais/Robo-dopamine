from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from mydata_bench.attention_eval.masking import (
    ImageSpan,
    make_attention_mask_hook,
    matched_wrong_position_set,
)
from mydata_bench.io import sha256_file
from mydata_bench.qwen_eval.attention import (
    PreparedAttentionInput,
    QwenAttentionRuntime,
    build_native_video_spans,
    temporal_source_frame_groups,
)
from mydata_bench.qwen_eval.protocols import ROBOREWARDBENCH_NATIVE


def test_native_video_timestamp_delimited_runs_bind_to_temporal_patches() -> None:
    spans = build_native_video_spans(
        "video.mp4",
        [(100, 108), (112, 120), (124, 132)],
        (3, 4, 8),
        spatial_merge_size=2,
    )
    assert [(span.start, span.end) for span in spans] == [
        (100, 108),
        (112, 120),
        (124, 132),
    ]
    assert [span.label for span in spans] == ["video_t0", "video_t1", "video_t2"]


def test_native_video_rejects_multi_patch_single_run_for_current_qwen() -> None:
    with pytest.raises(ValueError, match="one video-token run per temporal patch"):
        build_native_video_spans(
            "video.mp4",
            [(100, 124)],
            (3, 4, 8),
            spatial_merge_size=2,
        )


def test_native_video_single_temporal_patch_has_one_run() -> None:
    spans = build_native_video_spans(
        "video.mp4",
        [(100, 108)],
        (1, 4, 8),
        spatial_merge_size=2,
    )
    assert [(span.start, span.end) for span in spans] == [(100, 108)]


@pytest.mark.parametrize(
    ("token_spans", "grid", "error"),
    (
        (
            [(10, 14), (20, 24)],
            (3, 4, 4),
            "one video-token run per temporal patch",
        ),
        ([(10, 17), (20, 24)], (2, 4, 4), "temporal token/grid mismatch"),
        ([(10, 14), (20, 23)], (2, 4, 4), "temporal token/grid mismatch"),
        ([(10, 18)], (2, 5, 4), "incompatible with spatial merge"),
    ),
)
def test_native_video_span_contract_fails_closed(
    token_spans: list[tuple[int, int]],
    grid: tuple[int, int, int],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        build_native_video_spans(
            "video.mp4",
            token_spans,
            grid,
            spatial_merge_size=2,
        )


def test_temporal_source_groups_repeat_only_the_final_sampled_frame() -> None:
    assert temporal_source_frame_groups(
        [0, 3, 7], temporal_patch_size=2, temporal_grid_size=2
    ) == [[0, 3], [7, 7]]
    with pytest.raises(ValueError, match="does not match"):
        temporal_source_frame_groups(
            [0, 3, 7], temporal_patch_size=2, temporal_grid_size=3
        )


def _tracked_frame(path: Path, index: int, bbox: list[int]) -> dict:
    return {
        "source_frame_index": index,
        "image_path": str(path),
        "image_sha256": sha256_file(path),
        "bbox_xyxy": bbox,
        "visible": True,
    }


def test_terminal_temporal_patch_uses_union_of_moving_tracked_bboxes(
    tmp_path: Path,
) -> None:
    frames = []
    for index in range(4):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (100, 100), "white").save(path)
        bbox = [0, 0, 40, 40] if index == 2 else [60, 60, 100, 100]
        frames.append(_tracked_frame(path, index, bbox))

    target_span = ImageSpan("video_t1", "video.mp4", 20, 24, (1, 4, 4))
    prepared = PreparedAttentionInput(
        inputs={},
        spans=[
            ImageSpan("video_t0", "video.mp4", 16, 20, (1, 4, 4)),
            target_span,
        ],
        target_span=target_span,
        target_image_path=str(tmp_path / "frame-3.png"),
        visual_positions=list(range(16, 24)),
        protocol=ROBOREWARDBENCH_NATIVE,
        target_source_frame_indices=(2, 3),
        video_metadata={"frames_indices": [0, 1, 2, 3]},
    )
    runtime = object.__new__(QwenAttentionRuntime)
    runtime.protocol = ROBOREWARDBENCH_NATIVE
    runtime.merge_size = 2
    runtime._sha256_cache = {}
    sample = {
        "target_token_grounding_scope": (
            "terminal_temporal_patch_tracked_bbox_union"
        ),
        "tracked_processor_frames": frames,
    }

    assert runtime.target_positions(sample, prepared) == [20, 23]
    assert prepared.video_metadata["target_visible_source_frame_indices"] == [2, 3]
    assert prepared.video_metadata["target_bbox_policy"] == (
        "token_union_over_visible_frames_in_temporal_patch"
    )

    incomplete = dict(sample)
    incomplete["tracked_processor_frames"] = frames[:-1]
    with pytest.raises(ValueError, match="do not exactly cover"):
        runtime.target_positions(incomplete, prepared)


def test_visual_scope_distinguishes_terminal_patch_from_whole_video() -> None:
    target = ImageSpan("video_t1", "video.mp4", 20, 24, (1, 4, 4))
    prepared = PreparedAttentionInput(
        inputs={},
        spans=[ImageSpan("video_t0", "video.mp4", 16, 20, (1, 4, 4)), target],
        target_span=target,
        target_image_path="frame.png",
        visual_positions=list(range(16, 24)),
        protocol=ROBOREWARDBENCH_NATIVE,
    )
    runtime = object.__new__(QwenAttentionRuntime)
    assert runtime.visual_positions_for_scope(prepared, "target_slot_only") == [
        20,
        21,
        22,
        23,
    ]
    assert runtime.visual_positions_for_scope(prepared, "all_visual") == list(
        range(16, 24)
    )


def _call_hook(scope: str, *, query_length: int, key_length: int = 8):
    torch = pytest.importorskip("torch")
    diagnostics: dict = {}
    hook = make_attention_mask_hook(
        [1],
        [2],
        [3, 4],
        4,
        6.0,
        diagnostics,
        query_scope=scope,
    )
    mask = torch.zeros((1, 1, query_length, key_length), dtype=torch.float32)
    result = hook(None, (), {"attention_mask": mask})
    return result, diagnostics


@pytest.mark.parametrize("scope", ("all", "prefill"))
def test_prefill_scopes_apply_selected_head_bias_to_every_query_row(
    scope: str,
) -> None:
    result, diagnostics = _call_hook(scope, query_length=3)
    assert result is not None
    _args, kwargs = result
    biased = kwargs["attention_mask"]
    assert tuple(biased.shape) == (1, 4, 3, 8)
    assert biased[0, 1, :, 2].tolist() == [6.0, 6.0, 6.0]
    assert biased[0, 1, :, 3].tolist() == [-6.0, -6.0, -6.0]
    assert biased[0, 0].count_nonzero().item() == 0
    assert diagnostics["applied_query_rows"] == 3
    assert diagnostics["observed_query_rows"] == 3
    assert diagnostics["prefill_applied_query_rows"] == 3
    assert diagnostics["prefill_query_rows"] == 3


def test_last_prompt_scope_changes_only_the_final_prefill_query() -> None:
    result, diagnostics = _call_hook("last_prompt", query_length=3)
    assert result is not None
    _args, kwargs = result
    biased = kwargs["attention_mask"]
    assert biased[0, 1, :2].count_nonzero().item() == 0
    assert biased[0, 1, 2, 2].item() == 6.0
    assert biased[0, 1, 2, 4].item() == -6.0
    assert diagnostics["applied_query_rows"] == 1
    decode, _ = _call_hook("last_prompt", query_length=1, key_length=10)
    assert decode is None


def test_decode_scope_preserves_zero_bias_for_new_text_keys() -> None:
    prefill, _ = _call_hook("decode", query_length=3)
    assert prefill is None
    decode, diagnostics = _call_hook("decode", query_length=1, key_length=10)
    assert decode is not None
    _args, kwargs = decode
    biased = kwargs["attention_mask"]
    assert biased[0, 1, 0, 2].item() == 6.0
    assert biased[0, 1, 0, 3].item() == -6.0
    assert biased[..., 8:].count_nonzero().item() == 0
    assert diagnostics["decode_applied_calls"] == 1
    assert diagnostics["new_text_keys_zero_bias"] is True


def test_attention_hook_rejects_missing_declared_target_key() -> None:
    torch = pytest.importorskip("torch")
    hook = make_attention_mask_hook(
        [1], [7], [2, 3], 4, 6.0, {}, query_scope="all"
    )
    mask = torch.zeros((1, 1, 3, 7), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="does not contain every declared"):
        hook(None, (), {"attention_mask": mask})


def test_wrong_region_translation_is_equal_size_and_disjoint() -> None:
    span = ImageSpan("video_t1", "video.mp4", 20, 36, (1, 8, 8))
    target = [20, 21, 24]
    wrong = matched_wrong_position_set(span, target, spatial_merge_size=2)
    assert wrong is not None
    assert len(wrong) == len(target)
    assert not set(wrong) & set(target)
    assert set(wrong) < set(range(span.start, span.end))
