from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from mydata_bench.io import (
    object_fingerprint,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from mydata_bench.my_dataset.tracked_grounding import (
    TRACKED_GROUNDING_ARTIFACT_SCHEMA,
    TRACKED_GROUNDING_MANUAL_ANCHOR_SCHEMA,
    TRACKED_GROUNDING_REQUEST_SCHEMA,
    _candidate_provider_provenance,
    _parse_predictor_row,
    _predictor_from_config,
    _proposal_from_candidates,
    _run_visual_propagation,
    _track_candidate,
    build_tracked_grounding_requests,
    derive_manual_anchor_id,
    run_manual_retracks,
    run_tracked_grounding,
)
from mydata_bench.my_dataset.review_audit import audit_tracked_grounding_review
from mydata_bench.review_sam3_grounding_web import TrackedGroundingReviewStore


WIDTH = 32
HEIGHT = 24
FRAME_COUNT = 4


def _write_video(path: Path) -> tuple[list[np.ndarray], float]:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for index in range(FRAME_COUNT):
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        image[:, :] = (index * 20, index * 30, index * 40)
        image[2:8, 2 + index : 8 + index] = (255, 255, 255)
        writer.write(image)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    decoded = []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    while True:
        ok, image = capture.read()
        if not ok:
            break
        decoded.append(image)
    capture.release()
    assert len(decoded) == FRAME_COUNT
    return decoded, fps


def _save(path: Path, image: np.ndarray) -> dict[str, Any]:
    assert cv2.imwrite(str(path), image)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def _fixture_config(tmp_path: Path) -> tuple[dict[str, Any], "FakeProvider", "FakePredictor"]:
    video = tmp_path / "front.avi"
    decoded, fps = _write_video(video)
    video_sha = sha256_file(video)
    blank = _save(tmp_path / "blank.png", np.zeros_like(decoded[0]))
    endpoints: dict[str, dict[str, dict[str, Any]]] = {}
    for view in ("front", "left_wrist", "right_wrist"):
        endpoints[view] = {
            "first": _save(tmp_path / f"{view}-first.png", decoded[0]),
            "last": _save(tmp_path / f"{view}-last.png", decoded[-1]),
        }
    example_id = "example-1"
    group_id = "group-1"
    instruction = "Pick up the cup and place it in the plate."
    group_sha = object_fingerprint("group-media")
    inputs = tmp_path / "inputs.jsonl"
    roles = tmp_path / "roles.jsonl"
    split = tmp_path / "split.json"
    write_jsonl(
        inputs,
        [
            {
                "schema_version": "my_dataset.counterfactual.v1",
                "dataset_name": "fixture",
                "example_id": example_id,
                "group_id": group_id,
                "task_id": "task1_fixture",
                "task_family": "object_identity",
                "instruction": instruction,
                "evaluation_split": "test",
                "video_paths": {
                    "front": str(video),
                    "left_wrist": str(video),
                    "right_wrist": str(video),
                },
                "view_sha256": {
                    "front": video_sha,
                    "left_wrist": video_sha,
                    "right_wrist": video_sha,
                },
                "group_media_sha256": group_sha,
            }
        ],
    )
    write_jsonl(
        roles,
        [
            {
                "schema_version": "my_dataset.semantic_roles.v1",
                "example_id": example_id,
                "group_id": group_id,
                "task_id": "task1_fixture",
                "task_family": "object_identity",
                "instruction": instruction,
                "grounding_strategy": "object_identity",
                "target_phrase": "cup",
                "reference_object": None,
                "relation": None,
                "ordinal": None,
                "direction": None,
                "requires_instance_review": False,
            }
        ],
    )
    write_json(
        split,
        {
            "examples": {
                "discovery": [],
                "validation": [],
                "test": [example_id],
            },
            "example_counts": {"discovery": 0, "validation": 0, "test": 1},
        },
    )
    runs: dict[str, str] = {}
    for model in ("roboreward", "qwen"):
        run = tmp_path / model
        run.mkdir()
        diagnostics: dict[str, Any] = {
            "content_order": "video_then_text",
            "video_grid_thw": [[1, 2, 2]],
            "video_metadata": {
                "frames_indices": [0, 1, 2, 3],
                "total_num_frames": FRAME_COUNT,
                "width": WIDTH,
                "height": HEIGHT,
                "fps": fps,
            },
        }
        if model == "roboreward":
            diagnostics["video_record"] = {"source_video_path": str(video.resolve())}
        write_jsonl(
            run / "records.shard-00.jsonl",
            [
                {
                    "attempt": 1,
                    "example_id": example_id,
                    "group_id": group_id,
                    "group_media_sha256": group_sha,
                    "instruction": instruction,
                    "model_family": model,
                    "status": "ok",
                    "input_diagnostics": diagnostics,
                }
            ],
        )
        runs[model] = str(run)
    grm_run = tmp_path / "grm"
    grm_run.mkdir()
    frame_record = {}
    for view in ("front", "left_wrist", "right_wrist"):
        frame_record[view] = {
            "first_index": 0,
            "first_path": endpoints[view]["first"]["path"],
            "first_sha256": endpoints[view]["first"]["sha256"],
            "last_index": FRAME_COUNT - 1,
            "last_path": endpoints[view]["last"]["path"],
            "last_sha256": endpoints[view]["last"]["sha256"],
            "reported_frame_count": FRAME_COUNT,
            "width": WIDTH,
            "height": HEIGHT,
            "video_sha256": video_sha,
        }
    write_jsonl(
        grm_run / "records.shard-00.jsonl",
        [
            {
                "attempt": 1,
                "example_id": example_id,
                "group_id": group_id,
                "group_media_sha256": group_sha,
                "instruction": instruction,
                "model_family": "grm",
                "status": "ok",
                "frame_record": frame_record,
            }
        ],
    )
    runs["grm"] = str(grm_run)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[2:8, 2:8] = True
    provider = FakeProvider(
        {
            "cup": [
                {"query": "cup", "bbox": [2, 2, 8, 8], "score": 0.9, "_mask": mask},
                {"query": "cup", "bbox": [20, 2, 27, 9], "score": 0.7, "_mask": mask},
            ]
        }
    )
    predictor = FakePredictor()
    output = tmp_path / "tracked"
    config = {
        "my_dataset_tracked_grounding": {
            "inputs_path": str(inputs),
            "roles_path": str(roles),
            "split_path": str(split),
            "output_dir": str(output),
            "blank_goal": blank["path"],
            "baseline_runs": runs,
            "sam3": {
                "_candidate_provider": provider,
                "_predictor": predictor,
            },
        }
    }
    return config, provider, predictor


class FakeProvider:
    def __init__(self, values: dict[str, list[dict[str, Any]]]):
        self.values = values
        self.calls: list[str] = []
        self.fingerprint = "fake-provider"

    def candidates(self, image_path: str, queries: list[str]) -> list[dict[str, Any]]:
        assert Path(image_path).is_file()
        assert len(queries) == 1
        self.calls.append(queries[0])
        return copy.deepcopy(self.values.get(queries[0], []))


class FakePredictor:
    def __init__(
        self,
        *,
        missing_terminal: bool = False,
        empty_at: int | None = None,
        missing_id_at: int | None = None,
        malformed_at: int | None = None,
        raise_stream: bool = False,
    ):
        self.missing_terminal = missing_terminal
        self.empty_at = empty_at
        self.missing_id_at = missing_id_at
        self.malformed_at = malformed_at
        self.raise_stream = raise_stream
        self.requests: list[dict[str, Any]] = []
        self.stream_requests: list[dict[str, Any]] = []
        self.sessions = 0
        self.closed = 0
        self.shutdown_called = False
        self.anchor_bbox: list[float] | None = None

    def _row(self, frame_index: int) -> dict[str, Any]:
        assert self.anchor_bbox is not None
        x, y, box_width, box_height = self.anchor_bbox
        x1 = max(0, min(WIDTH - 1, int(round(x * WIDTH)) + frame_index))
        y1 = max(0, min(HEIGHT - 1, int(round(y * HEIGHT))))
        x2 = min(WIDTH, x1 + max(1, int(round(box_width * WIDTH))))
        y2 = min(HEIGHT, y1 + max(1, int(round(box_height * HEIGHT))))
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[y1:y2, x1:x2] = True
        if self.empty_at == frame_index:
            mask[:] = False
        ids = np.asarray([7], dtype=np.int64)
        if self.missing_id_at == frame_index:
            ids = np.asarray([8], dtype=np.int64)
        row = {
            "frame_index": frame_index,
            "outputs": {
                "out_obj_ids": ids,
                "out_probs": np.asarray([0.95], dtype=np.float32),
                "out_boxes_xywh": np.asarray(
                    [[x1 / WIDTH, y1 / HEIGHT, (x2 - x1) / WIDTH, (y2 - y1) / HEIGHT]],
                    dtype=np.float32,
                ),
                "out_binary_masks": mask[None, :, :],
            },
        }
        if self.malformed_at == frame_index:
            row["outputs"]["out_probs"] = np.asarray([], dtype=np.float32)
        return row

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(request))
        kind = request["type"]
        if kind == "start_session":
            self.sessions += 1
            return {"session_id": f"session-{self.sessions}"}
        if kind == "add_prompt":
            assert request["text"] is None
            assert request["bounding_box_labels"] == [1]
            self.anchor_bbox = list(request["bounding_boxes"][0])
            return self._row(0)
        if kind == "close_session":
            self.closed += 1
            return {"closed": True}
        raise AssertionError(kind)

    def handle_stream_request(self, request: dict[str, Any]):
        self.stream_requests.append(copy.deepcopy(request))
        if self.raise_stream:
            raise RuntimeError("stream failed")
        limit = FRAME_COUNT - 1 if self.missing_terminal else FRAME_COUNT
        return iter(self._row(index) for index in range(limit))

    def shutdown(self) -> None:
        self.shutdown_called = True


def _candidate(
    identity: str, bbox: list[float], score: float = 0.9, role: str = "target"
) -> dict[str, Any]:
    return {
        "candidate_id": identity,
        "query_role": role,
        "query": "cup" if role == "target" else "teapot",
        "bbox_xyxy": bbox,
        "score": score,
        "center_xy": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
        "mask_path": None,
        "mask_sha256": None,
    }


def _proposal_request(roles: dict[str, Any]) -> dict[str, Any]:
    return {"roles": roles}


def test_request_builder_freezes_three_model_frame_bindings(tmp_path: Path) -> None:
    config, _, _ = _fixture_config(tmp_path)
    path = build_tracked_grounding_requests(config)
    request = list(read_jsonl(path))[0]
    assert request["schema_version"] == TRACKED_GROUNDING_REQUEST_SCHEMA
    assert request["partition"] == "test"
    assert request["source"]["split"]["sha256"] == sha256_file(
        config["my_dataset_tracked_grounding"]["split_path"]
    )
    assert request["first_frame"]["source_frame_index"] == 0
    assert [frame["source_frame_index"] for frame in request["key_frames"]] == [0, 1, 2, 3]
    assert request["model_frame_bindings"]["roboreward"]["content_order"] == "video_then_text"
    assert request["model_frame_bindings"]["qwen"]["sampled_frame_indices"] == [0, 1, 2, 3]
    assert len(request["model_frame_bindings"]["grm"]["image_paths"]) == 8
    terminals = {
        value["terminal"]["source_frame_index"]
        for value in request["model_frame_bindings"].values()
    }
    assert terminals == {FRAME_COUNT - 1}
    assert request["source"]["baseline_records"]["grm"]["file_sha256"]


def test_ordinal_default_and_at_most_two_alternatives() -> None:
    targets = [
        _candidate("left", [0, 0, 4, 4], 0.7),
        _candidate("middle", [10, 0, 14, 4], 0.6),
        _candidate("right", [20, 0, 24, 4], 0.9),
        _candidate("extra", [26, 0, 30, 4], 0.5),
    ]
    proposal = _proposal_from_candidates(
        _proposal_request(
            {
                "grounding_strategy": "ordinal_position",
                "ordinal": "second",
                "direction": "left",
                "target_phrase": "cup",
                "reference_object": None,
            }
        ),
        targets,
        [],
    )
    assert proposal["algorithmic_default"]["candidate_id"] == "middle"
    assert len(proposal["options"]) == 3
    assert sum(item["selection"] == "alternative" for item in proposal["options"]) == 2


@pytest.mark.parametrize(
    ("strategy", "relation", "expected"),
    [
        ("left_right_relation", "left", "near-left"),
        ("left_right_relation", "right", "right"),
        ("distance_relation", "closest to", "near-left"),
        ("distance_relation", "farthest from", "far-left"),
    ],
)
def test_reference_geometry_selects_without_proxy(
    strategy: str, relation: str, expected: str
) -> None:
    targets = [
        _candidate("near-left", [8, 8, 12, 12], 0.5),
        _candidate("far-left", [0, 8, 4, 12], 0.99),
        _candidate("right", [26, 8, 30, 12], 0.7),
    ]
    reference = _candidate("reference", [16, 8, 20, 12], role="reference")
    proposal = _proposal_from_candidates(
        _proposal_request(
            {
                "grounding_strategy": strategy,
                "relation": relation,
                "target_phrase": "cup",
                "reference_object": "teapot",
            }
        ),
        targets,
        [reference],
    )
    assert proposal["algorithmic_default"]["candidate_id"] == expected
    assert all(item["query_role"] == "target" for item in proposal["options"])


@pytest.mark.parametrize("references", [[], [
    _candidate("r1", [14, 8, 18, 12], role="reference"),
    _candidate("r2", [20, 8, 24, 12], role="reference"),
]])
def test_reference_missing_or_ambiguous_needs_review(
    references: list[dict[str, Any]],
) -> None:
    targets = [_candidate("target", [2, 2, 8, 8])]
    proposal = _proposal_from_candidates(
        _proposal_request(
            {
                "grounding_strategy": "distance_relation",
                "relation": "closest to",
                "target_phrase": "cup",
                "reference_object": "teapot",
            }
        ),
        targets,
        references,
    )
    assert proposal["status"] == "needs_review"
    assert proposal["algorithmic_default"] is None
    assert len(proposal["options"]) <= 3
    assert all(item["candidate_id"] == "target" for item in proposal["options"])


def test_needs_review_keeps_green_preselection_plus_two_others() -> None:
    targets = [
        _candidate("one", [1, 1, 4, 4], 0.9),
        _candidate("two", [6, 1, 9, 4], 0.8),
        _candidate("three", [11, 1, 14, 4], 0.7),
        _candidate("four", [16, 1, 19, 4], 0.6),
    ]
    proposal = _proposal_from_candidates(
        _proposal_request(
            {
                "grounding_strategy": "distance_relation",
                "relation": "closest to",
                "target_phrase": "cup",
                "reference_object": "missing-reference",
            }
        ),
        targets,
        [],
    )
    assert proposal["algorithmic_default"] is None
    assert [item["candidate_id"] for item in proposal["options"]] == [
        "one",
        "two",
        "three",
    ]
    assert all(item["selection"] == "alternative" for item in proposal["options"])


def test_target_detection_overlapping_unique_reference_is_excluded() -> None:
    overlapping = _candidate("same-instance", [16, 8, 20, 12], 0.99)
    valid = _candidate("actual-target", [8, 8, 12, 12], 0.4)
    reference = _candidate("reference", [16, 8, 20, 12], role="reference")
    proposal = _proposal_from_candidates(
        _proposal_request(
            {
                "grounding_strategy": "distance_relation",
                "relation": "closest to",
                "target_phrase": "cup",
                "reference_object": "purple cup",
            }
        ),
        [overlapping, valid],
        [reference],
    )
    assert proposal["algorithmic_default"]["candidate_id"] == "actual-target"
    assert [item["candidate_id"] for item in proposal["options"]] == ["actual-target"]
    assert proposal["excluded_target_candidates"][0]["candidate_id"] == "same-instance"
    assert (
        proposal["excluded_target_candidates"][0]["exclusion_reason"]
        == "overlaps_unique_reference_instance"
    )


def test_distance_tie_only_rejects_when_selected_extreme_is_tied() -> None:
    reference = _candidate("reference", [16, 8, 20, 12], role="reference")
    targets = [
        _candidate("left-near", [8, 8, 12, 12]),
        _candidate("right-near", [24, 8, 28, 12]),
        _candidate("unique-far", [0, 8, 4, 12]),
    ]
    base_roles = {
        "grounding_strategy": "distance_relation",
        "target_phrase": "cup",
        "reference_object": "teapot",
    }
    closest = _proposal_from_candidates(
        _proposal_request({**base_roles, "relation": "closest to"}),
        targets,
        [reference],
    )
    assert closest["status"] == "needs_review"
    assert closest["algorithmic_default"] is None
    farthest = _proposal_from_candidates(
        _proposal_request({**base_roles, "relation": "farthest from"}),
        targets,
        [reference],
    )
    assert farthest["algorithmic_default"]["candidate_id"] == "unique-far"


def test_full_run_uses_visual_prompt_stream_and_locked_id(tmp_path: Path) -> None:
    config, provider, predictor = _fixture_config(tmp_path)
    tracks_path = run_tracked_grounding(config)
    artifact = list(read_jsonl(tracks_path))[0]
    assert artifact["schema_version"] == TRACKED_GROUNDING_ARTIFACT_SCHEMA
    assert artifact["status"] == "ok"
    assert (
        artifact["proposal"]["provider_provenance"]["backend"]
        == "injected_test_double"
    )
    assert provider.calls == ["cup"]
    add = next(item for item in predictor.requests if item["type"] == "add_prompt")
    assert add["text"] is None
    assert add["bounding_boxes"] == [[2 / WIDTH, 2 / HEIGHT, 6 / WIDTH, 6 / HEIGHT]]
    assert predictor.stream_requests[0]["type"] == "propagate_in_video"
    track = artifact["candidate_tracks"][0]
    assert track["locked_obj_id"] == 7
    assert track["terminal_by_model"]["qwen"]["obj_id"] == 7
    assert track["continuity"]["frame_coverage_complete"] is True
    assert predictor.closed == predictor.sessions
    assert predictor.shutdown_called


def test_cache_reuses_identical_visual_anchor(tmp_path: Path) -> None:
    config, _, predictor = _fixture_config(tmp_path)
    requests_path = build_tracked_grounding_requests(config)
    request = list(read_jsonl(requests_path))[0]
    candidate = _candidate("one", [2, 2, 8, 8])
    provenance = {
        "official_source_path": "fake",
        "model_builder_sha256": "a",
        "video_predictor_sha256": "b",
        "checkpoint_path": "fake",
        "checkpoint_sha256": "c",
        "tracker_fingerprint": "tracker",
    }
    first = _track_candidate(
        predictor, provenance, request, "proposal", candidate, "sam3_candidate",
        Path(config["my_dataset_tracked_grounding"]["output_dir"]), {},
    )
    candidate["candidate_id"] = "two"
    second = _track_candidate(
        predictor, provenance, request, "proposal", candidate, "sam3_candidate",
        Path(config["my_dataset_tracked_grounding"]["output_dir"]), {},
    )
    assert predictor.sessions == 1
    assert first["predictor_provenance"]["cache_hit"] is False
    assert second["predictor_provenance"]["cache_hit"] is True


def test_cache_rejects_wrong_schema_even_with_valid_fingerprint(
    tmp_path: Path,
) -> None:
    config, _, predictor = _fixture_config(tmp_path)
    request = list(read_jsonl(build_tracked_grounding_requests(config)))[0]
    candidate = _candidate("one", [2, 2, 8, 8])
    provenance = {"tracker_fingerprint": "tracker"}
    first = _track_candidate(
        predictor,
        provenance,
        request,
        "proposal",
        candidate,
        "sam3_candidate",
        Path(config["my_dataset_tracked_grounding"]["output_dir"]),
        {},
    )
    cache_path = Path(first["predictor_provenance"]["cache_path"])
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    cached["schema_version"] = "mydata_bench.tracked_grounding.cache.v1"
    fingerprint_view = dict(cached)
    fingerprint_view.pop("fingerprint")
    cached["fingerprint"] = object_fingerprint(fingerprint_view)
    write_json(cache_path, cached)

    second = _track_candidate(
        predictor,
        provenance,
        request,
        "proposal",
        candidate,
        "sam3_candidate",
        Path(config["my_dataset_tracked_grounding"]["output_dir"]),
        {},
    )
    assert predictor.sessions == 2
    assert second["predictor_provenance"]["cache_hit"] is False


@pytest.mark.parametrize(
    "predictor",
    [
        FakePredictor(empty_at=2),
        FakePredictor(missing_id_at=2),
        FakePredictor(malformed_at=2),
        FakePredictor(missing_terminal=True),
    ],
)
def test_tracking_failures_close_session(
    tmp_path: Path, predictor: FakePredictor
) -> None:
    config, _, _ = _fixture_config(tmp_path)
    request = list(read_jsonl(build_tracked_grounding_requests(config)))[0]
    candidate = _candidate("target", [2, 2, 8, 8])
    provenance = {"tracker_fingerprint": "tracker"}
    with pytest.raises((ValueError, RuntimeError)):
        _run_visual_propagation(
            predictor,
            request,
            candidate,
            provenance,
            tmp_path / "cache",
            "cache-key",
            {},
        )
    assert predictor.closed == 1


def test_stream_exception_still_closes_session(tmp_path: Path) -> None:
    config, _, _ = _fixture_config(tmp_path)
    request = list(read_jsonl(build_tracked_grounding_requests(config)))[0]
    predictor = FakePredictor(raise_stream=True)
    with pytest.raises(RuntimeError, match="stream failed"):
        _run_visual_propagation(
            predictor,
            request,
            _candidate("target", [2, 2, 8, 8]),
            {"tracker_fingerprint": "tracker"},
            tmp_path / "cache",
            "cache-key",
            {},
        )
    assert predictor.closed == 1


def test_official_nested_output_shape_is_strict() -> None:
    predictor = FakePredictor()
    predictor.anchor_bbox = [2 / WIDTH, 2 / HEIGHT, 6 / WIDTH, 6 / HEIGHT]
    frame, objects = _parse_predictor_row(predictor._row(0), WIDTH, HEIGHT)
    assert frame == 0
    assert objects[7]["bbox_xyxy"] == [2.0, 2.0, 8.0, 8.0]
    malformed = predictor._row(0)
    malformed["outputs"]["out_boxes_xywh"] = np.asarray([[0.0, 0.0, 2.0, 1.0]])
    with pytest.raises(ValueError, match="normalized"):
        _parse_predictor_row(malformed, WIDTH, HEIGHT)


def test_missing_official_video_source_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="official_source_path"):
        _predictor_from_config({})


def test_injected_predictor_fingerprint_binds_orchestrator_source() -> None:
    _, provenance = _predictor_from_config({"_predictor": FakePredictor()})
    orchestrator = Path(provenance["orchestrator_source_path"])
    assert orchestrator.resolve().name == "tracked_grounding.py"
    assert provenance["orchestrator_source_sha256"] == sha256_file(orchestrator)
    fingerprint_view = dict(provenance)
    fingerprint = fingerprint_view.pop("tracker_fingerprint")
    assert fingerprint == object_fingerprint(fingerprint_view)


def test_proposer_provenance_hashes_actual_weight_bytes(tmp_path: Path) -> None:
    model = tmp_path / "sam3-image-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"sam3"}\n', encoding="utf-8")
    weights = model / "model.safetensors"
    weights.write_bytes(b"fixture-weight-bytes")
    provider = FakeProvider({})
    provenance = _candidate_provider_provenance(
        provider,
        {
            "model_path": str(model),
            "threshold": 0.1,
            "mask_threshold": 0.5,
            "top_n_per_query": 20,
        },
    )
    assert provenance["backend"] == "transformers_sam3_image"
    assert provenance["model_weight_files"] == [
        {
            "path": "model.safetensors",
            "size": weights.stat().st_size,
            "sha256": sha256_file(weights),
        }
    ]
    assert provenance["model_weights_fingerprint"]
    orchestrator = Path(provenance["orchestrator_source_path"])
    assert provenance["orchestrator_source_sha256"] == sha256_file(orchestrator)
    assert provenance["proposer_fingerprint"]


def test_manual_bbox_is_repropagated_to_terminal(tmp_path: Path) -> None:
    config, _, predictor = _fixture_config(tmp_path)
    run_tracked_grounding(config)
    request_path = (
        Path(config["my_dataset_tracked_grounding"]["output_dir"]) / "requests.jsonl"
    )
    request = list(read_jsonl(request_path))[0]
    manual = {
        "schema_version": TRACKED_GROUNDING_MANUAL_ANCHOR_SCHEMA,
        "manual_anchor_id": derive_manual_anchor_id(
            request["example_id"],
            request["first_frame"]["image_sha256"],
            [10.0, 4.0, 16.0, 10.0],
        ),
        "example_id": request["example_id"],
        "request_fingerprint": request["request_fingerprint"],
        "first_frame_index": 0,
        "first_image_path": request["first_frame"]["image_path"],
        "first_image_sha256": request["first_frame"]["image_sha256"],
        "bbox_xyxy": [10.0, 4.0, 16.0, 10.0],
        "reviewer_id": "reviewer",
        "reviewed_at": None,
        "note": None,
    }
    manual["fingerprint"] = object_fingerprint(manual)
    anchors = tmp_path / "manual_anchors.jsonl"
    write_jsonl(anchors, [manual])
    output = tmp_path / "manual_tracks.jsonl"
    run_manual_retracks(config, anchors, output)
    artifact = list(read_jsonl(output))[0]
    track = artifact["candidate_tracks"][0]
    assert artifact["selection_source"] == "manual_bbox"
    assert track["source"] == "manual_bbox"
    assert track["anchor"]["bbox_xyxy"] == manual["bbox_xyxy"]
    assert track["terminal_by_model"]["roboreward"]["bbox_xyxy"] != manual["bbox_xyxy"]
    assert predictor.stream_requests

    tampered = dict(manual)
    tampered["bbox_xyxy"] = [11.0, 4.0, 17.0, 10.0]
    tampered["fingerprint"] = object_fingerprint(
        {key: value for key, value in tampered.items() if key != "fingerprint"}
    )
    bad_anchors = tmp_path / "tampered_manual_anchors.jsonl"
    write_jsonl(bad_anchors, [tampered])
    with pytest.raises(ValueError, match="ID does not match"):
        run_manual_retracks(
            config, bad_anchors, tmp_path / "tampered_manual_tracks.jsonl"
        )


def test_append_only_retry_failed(tmp_path: Path) -> None:
    config, _, predictor = _fixture_config(tmp_path)
    predictor.missing_terminal = True
    tracks = run_tracked_grounding(config)
    assert list(read_jsonl(tracks))[-1]["status"] == "invalid"
    sessions = predictor.sessions
    run_tracked_grounding(config)
    assert predictor.sessions == sessions
    assert len(list(read_jsonl(tracks))) == 1
    predictor.missing_terminal = False
    run_tracked_grounding(config, retry_failed=True)
    rows = list(read_jsonl(tracks))
    assert [row["attempt"] for row in rows] == [1, 2]
    assert rows[-1]["status"] == "ok"


def test_tracking_review_audit_binds_selected_track_and_terminals(
    tmp_path: Path,
) -> None:
    config, _, _ = _fixture_config(tmp_path)
    tracks_path = run_tracked_grounding(config)
    run_dir = Path(config["my_dataset_tracked_grounding"]["output_dir"])
    review_dir = tmp_path / "reviewed"
    store = TrackedGroundingReviewStore(
        run_dir, "reviewer-test", output_dir=review_dir
    )
    current = store.state(position=1)["current"]
    candidate_id = current["default_candidate_id"]
    assert candidate_id
    store.submit(
        {
            "example_id": current["example_id"],
            "status": "eligible",
            "candidate_id": candidate_id,
            "decision_source": "accept_default",
        }
    )

    result = audit_tracked_grounding_review(
        run_dir / "requests.jsonl",
        review_dir / "reviews.jsonl",
        review_dir,
        tracking_artifact_path=tracks_path,
    )
    assert result["passed"] is True
    assert result["eligible_example_count"] == 1
    assert result["skipped_example_count"] == 0
    assert result["tracking_continuity_verified"] is True
    assert result["manual_tracking_artifact_sha256"] is None
    assert result["target_grounding_scope"] == "terminal_only"
    assert result["control_region_policy"] == "none"



def _audit_invalid_reasons(result: dict[str, Any]) -> set[str]:
    reasons: set[str] = set()
    for item in result.get("invalid", []):
        reason = item.get("reason")
        if reason:
            reasons.add(str(reason))
        reasons.update(str(value) for value in item.get("reasons", []))
    return reasons


def test_tracking_review_audit_requires_review_self_fingerprint(
    tmp_path: Path,
) -> None:
    config, _, _ = _fixture_config(tmp_path)
    tracks_path = run_tracked_grounding(config)
    run_dir = Path(config["my_dataset_tracked_grounding"]["output_dir"])
    review_dir = tmp_path / "reviewed-no-fingerprint"
    store = TrackedGroundingReviewStore(
        run_dir, "reviewer-test", output_dir=review_dir
    )
    current = store.state(position=1)["current"]
    store.submit(
        {
            "example_id": current["example_id"],
            "status": "skipped",
            "disposition_code": "reviewer_skip",
        }
    )
    row = list(read_jsonl(store.reviews_path))[0]
    row.pop("fingerprint")
    write_jsonl(store.history_path, [row])
    write_jsonl(store.reviews_path, [row])

    result = audit_tracked_grounding_review(
        run_dir / "requests.jsonl",
        store.reviews_path,
        review_dir,
        tracking_artifact_path=tracks_path,
    )
    assert result["passed"] is False
    assert "review_fingerprint_invalid" in _audit_invalid_reasons(result)


def test_tracking_review_audit_rejects_legacy_skip_code_even_if_refingerprinted(
    tmp_path: Path,
) -> None:
    config, _, _ = _fixture_config(tmp_path)
    tracks_path = run_tracked_grounding(config)
    run_dir = Path(config["my_dataset_tracked_grounding"]["output_dir"])
    review_dir = tmp_path / "reviewed-old-skip"
    store = TrackedGroundingReviewStore(
        run_dir, "reviewer-test", output_dir=review_dir
    )
    current = store.state(position=1)["current"]
    store.submit(
        {
            "example_id": current["example_id"],
            "status": "skipped",
            "disposition_code": "reviewer_skip",
        }
    )
    latest = list(read_jsonl(store.reviews_path))[0]
    legacy = copy.deepcopy(latest)
    legacy["disposition"] = {"code": "no_correct_candidate"}
    legacy.pop("fingerprint")
    legacy["fingerprint"] = object_fingerprint(legacy)
    # Even a non-latest physical history row must satisfy the singleton skip code.
    write_jsonl(store.history_path, [legacy, latest])
    write_jsonl(store.reviews_path, [latest])

    result = audit_tracked_grounding_review(
        run_dir / "requests.jsonl",
        store.reviews_path,
        review_dir,
        tracking_artifact_path=tracks_path,
    )
    assert result["passed"] is False
    assert "skipped_disposition_code_invalid" in _audit_invalid_reasons(result)
