from __future__ import annotations

import pytest

from auto_eval_robomimic_abs import (
    fixed_sample_indices,
    original_sample_indices,
    parse_args,
    record_key,
    source_sample_indices,
)


def test_fixed_sample_indices_are_uniform_and_include_endpoints() -> None:
    indices = fixed_sample_indices(length=101, comparisons=8)

    assert indices == [0, 13, 25, 38, 50, 63, 75, 88, 100]
    assert len(indices) == 9
    assert all(left < right for left, right in zip(indices, indices[1:]))


def test_fixed_sample_indices_require_k_plus_one_frames() -> None:
    with pytest.raises(ValueError, match="at least 9 frames"):
        fixed_sample_indices(length=8, comparisons=8)


@pytest.mark.parametrize(
    ("length", "interval", "expected"),
    [
        (1, 20, [0]),
        (41, 20, [0, 20, 40]),
        (42, 20, [0, 20, 40, 41]),
    ],
)
def test_interval_sampling_remains_unchanged(
    length: int, interval: int, expected: list[int]
) -> None:
    assert original_sample_indices(length, interval) == expected


def test_source_sampling_prefers_fixed_k() -> None:
    assert source_sample_indices(101, interval=None, fixed_samples=4) == [
        0,
        25,
        50,
        75,
        100,
    ]


def test_parse_args_defaults_to_interval_20() -> None:
    args = parse_args([])

    assert args.frame_interval == 20
    assert args.fixed_samples is None


def test_parse_args_accepts_fixed_samples() -> None:
    args = parse_args(["--fixed-samples", "8"])

    assert args.frame_interval is None
    assert args.fixed_samples == 8


def test_sampling_options_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--frame-interval", "10", "--fixed-samples", "8"])


def test_record_key_keeps_old_interval_records_compatible() -> None:
    common = {
        "model_path": "model",
        "dataset": "can",
        "episode_index": 1,
        "quality": "better",
        "task": "pick",
        "view_mode": "image-only",
        "mode": "incremental",
        "source_frame_interval": 20,
        "goal_id": "blank",
    }
    old_interval_record = dict(common)
    explicit_interval_record = {
        **common,
        "sampling_strategy": "interval",
        "fixed_samples": None,
    }
    fixed_record = {
        **common,
        "sampling_strategy": "fixed",
        "source_frame_interval": None,
        "fixed_samples": 8,
    }

    assert record_key(old_interval_record) == record_key(explicit_interval_record)
    assert record_key(fixed_record) != record_key(old_interval_record)

