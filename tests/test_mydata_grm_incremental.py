from __future__ import annotations

import json
from pathlib import Path

import pytest

from mydata_bench.attention_eval import runtime as attention_runtime
from mydata_bench.protocol import (
    accumulate_incremental_progress,
    official_incremental_indices,
)


def test_official_incremental_indices_include_short_terminal_hop() -> None:
    indices = official_incremental_indices(402, 20)
    assert indices == [*range(0, 401, 20), 402]
    assert list(zip(indices, indices[1:]))[-1] == (400, 402)
    assert len(indices) - 1 == 21


def test_official_incremental_accumulation_matches_reference_formula() -> None:
    assert accumulate_incremental_progress(None, 0.6) == pytest.approx(0.6)
    assert accumulate_incremental_progress(0.6, 0.25) == pytest.approx(0.7)
    assert accumulate_incremental_progress(0.8, -0.25) == pytest.approx(0.6)
    # The reference implementation deliberately preserves a negative first hop.
    assert accumulate_incremental_progress(None, -0.2) == pytest.approx(-0.2)


def test_attention_incremental_steps_use_every_hop_and_per_frame_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = 42
    endpoints = {}
    view_paths = {}
    for view in ("front", "left_wrist", "right_wrist"):
        first = tmp_path / view / "first.png"
        last = tmp_path / view / "last.png"
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"first")
        last.write_bytes(b"last")
        endpoints[view] = {"first": str(first), "last": str(last)}
        view_paths[view] = str(tmp_path / f"{view}.mp4")

    track_path = tmp_path / "track.json"
    track_path.write_text(
        json.dumps(
            {
                "terminal_frame_index": terminal,
                "frames": [
                    {"frame_index": index, "bbox": [index, 1, index + 2, 4]}
                    for index in range(terminal + 1)
                ],
            }
        ),
        encoding="utf-8",
    )

    extracted = []

    def fake_extract(_video, output_path, frame_index):
        extracted.append(frame_index)
        return frame_index, str(Path(output_path))

    monkeypatch.setattr(attention_runtime, "extract_frame_at", fake_extract)
    sample = {
        "video_sha256": "a" * 64,
        "last": {
            "provenance": {
                "tracking_path": str(track_path),
                "view_endpoint_paths": endpoints,
                "view_paths": view_paths,
            }
        },
    }
    steps = attention_runtime.incremental_steps(
        sample, {"output_dir": str(tmp_path / "output"), "frame_interval": 20}
    )

    assert [
        (step["before_frame_index"], step["after_frame_index"])
        for step in steps
    ] == [(0, 20), (20, 40), (40, 42)]
    assert steps[1]["before_bbox"] == [20.0, 1.0, 22.0, 4.0]
    assert steps[1]["after_bbox"] == [40.0, 1.0, 42.0, 4.0]
    # Each of the two intermediate indices is decoded for all three views;
    # endpoints are reused rather than redundantly decoded.
    assert extracted.count(20) == 6
    assert extracted.count(40) == 6


def test_attention_condition_accumulates_all_hops_independently() -> None:
    class FakeRuntime:
        def generate(self, _sample, *, step, **_kwargs):
            score = (0.6, 0.25)[step["hop_index"]]
            return {
                "raw_output": f"<score>+{int(score * 100)}%</score>",
                "signed_score": score,
                "hook_diagnostics": {
                    "query_scope": "all",
                    "bbox_attention_mass": 0.1 + step["hop_index"],
                },
            }

    plan = [
        {
            "hop_index": index,
            "before_frame_index": index * 20,
            "after_frame_index": (index + 1) * 20,
            "before_bbox_frame_index": index * 20,
            "after_bbox_frame_index": (index + 1) * 20,
            "target_positions": [10 + index],
            "wrong_positions": [20 + index],
            "image_positions": list(range(10, 30)),
        }
        for index in range(2)
    ]
    result = attention_runtime.AttentionRuntime.generate_incremental(
        FakeRuntime(), {}, plan, position_kind="target"
    )
    assert result["progress"] == pytest.approx(0.7)
    assert result["last_hop_score"] == pytest.approx(0.25)
    assert result["hop_count"] == 2
    assert result["sampled_frame_indices"] == [0, 20, 40]
    assert result["hook_diagnostics"]["bbox_attention_mass"] == pytest.approx(0.6)
