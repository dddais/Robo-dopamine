from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mydata_bench.grounding.sam3 import SAM3Grounder
from mydata_bench.io import (
    object_fingerprint,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from mydata_bench.my_dataset.assumed_grounding import (
    MODELS,
    build_assumed_grounding,
    build_assumed_grounding_reviews,
)
from mydata_bench.my_dataset.attention_manifest import build_attention_manifests


def _request(
    tmp_path: Path, roles: dict[str, Any], example_id: str = "e1"
) -> tuple[dict[str, Any], dict[str, str]]:
    root = tmp_path / example_id
    root.mkdir(parents=True, exist_ok=True)
    model_images: dict[str, str] = {}
    for model in MODELS:
        path = root / f"{model}.png"
        Image.new("RGB", (100, 100), color=(16, 32, 48)).save(path)
        model_images[model] = str(path)
    grm_images = []
    for index in range(8):
        path = root / f"grm-slot-{index}.png"
        Image.new("RGB", (100, 100), color=(index, index, index)).save(path)
        grm_images.append(str(path))
    grm_images[5] = model_images["grm"]
    request = {
        "schema_version": "my_dataset.grounding_request.v1",
        "example_id": example_id,
        "group_id": f"g-{example_id}",
        "partition": "discovery",
        "task_id": "task-fixture",
        "instruction": "Pick up the requested object and place it safely.",
        "roles": roles,
        "model_frames": {
            "roboreward": {
                "input_layout": "native_front_video",
                "view": "front",
                "image_path": model_images["roboreward"],
                "image_sha256": sha256_file(model_images["roboreward"]),
                "sampled_frame_indices": [0, 4],
                "source_frame_index": 4,
                "video_grid_thw": [1, 2, 2],
                "content_order": "video_then_text",
            },
            "qwen": {
                "input_layout": "native_front_video",
                "view": "front",
                "image_path": model_images["qwen"],
                "image_sha256": sha256_file(model_images["qwen"]),
                "sampled_frame_indices": [0, 2, 4],
                "source_frame_index": 4,
                "video_grid_thw": [1, 2, 2],
                "content_order": "video_then_text",
            },
            "grm": {
                "input_layout": "grm_native_three_view_endpoints_v1",
                "image_paths": grm_images,
                "terminal_views": {
                    "front": {
                        "view": "front",
                        "source_frame_index": 4,
                        "image_path": model_images["grm"],
                        "image_sha256": sha256_file(model_images["grm"]),
                    },
                    "left_wrist": {
                        "view": "left_wrist",
                        "source_frame_index": 4,
                        "image_path": grm_images[6],
                        "image_sha256": sha256_file(grm_images[6]),
                    },
                    "right_wrist": {
                        "view": "right_wrist",
                        "source_frame_index": 4,
                        "image_path": grm_images[7],
                        "image_sha256": sha256_file(grm_images[7]),
                    },
                },
                "primary_target_view": "front",
                "primary_target_slot": "after_cam_high",
            },
        },
        "requested_regions": [
            "manipulated_object",
            "wrong_object_or_background",
        ],
    }
    return request, model_images


def _requested_queries(roles: dict[str, Any]) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    for role, key in (
        ("manipulated_object", "target_phrase"),
        ("reference_object", "reference_object"),
        ("destination", "destination"),
    ):
        value = roles.get(key)
        if isinstance(value, str) and value and value not in {
            query["text"] for query in queries
        }:
            queries.append({"role": role, "text": value})
    return queries


def _proposal_rows(
    request: dict[str, Any],
    model_images: dict[str, str],
    candidates: list[dict[str, Any]],
    *,
    candidates_by_model: dict[str, list[dict[str, Any]]] | None = None,
    proposal_images: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        image_path = (
            proposal_images.get(model, model_images[model])
            if proposal_images
            else model_images[model]
        )
        rows.append(
            {
                "schema_version": "my_dataset.grounding_request.v1",
                "example_id": request["example_id"],
                "group_id": request["group_id"],
                "partition": request["partition"],
                "model_family": model,
                "image_path": image_path,
                "image_sha256": sha256_file(image_path),
                "queries": _requested_queries(request["roles"]),
                "requires_instance_review": False,
                "auto_accepted": False,
                "status": "ok",
                "candidates": (
                    candidates_by_model.get(model, candidates)
                    if candidates_by_model
                    else candidates
                ),
            }
        )
    return rows


def _grounding_files(
    tmp_path: Path,
    roles: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    candidates_by_model: dict[str, list[dict[str, Any]]] | None = None,
    proposal_images: dict[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    request, model_images = _request(tmp_path, roles)
    requests_path = tmp_path / "requests.jsonl"
    proposals_path = tmp_path / "proposals.jsonl"
    write_jsonl(requests_path, [request])
    rows = _proposal_rows(
        request,
        model_images,
        candidates,
        candidates_by_model=candidates_by_model,
        proposal_images=proposal_images,
    )
    write_jsonl(proposals_path, rows)
    return requests_path, proposals_path, tmp_path / "assumed"


@pytest.mark.parametrize(
    ("roles", "candidates", "expected_bbox"),
    [
        (
            {
                "grounding_strategy": "object_identity",
                "target_phrase": "cup",
            },
            [
                {
                    "query": "cup",
                    "bbox": ["bad", 0, 10, 10],
                    "score": 0.99,
                },
                {
                    "query": "cup",
                    "bbox": [0, 0, 10, 10],
                    "score": 0.9,
                    "candidate_index": {"malformed": True},
                },
                {"query": "cup", "bbox": [20, 0, 30, 10], "score": 0.8},
            ],
            [0, 0, 10, 10],
        ),
        (
            {
                "grounding_strategy": "attribute_color",
                "target_phrase": "red block",
            },
            [
                {
                    "query": "red block",
                    "bbox": [10, 0, 20, 10],
                    "score": 0.7,
                },
            ],
            [10, 0, 20, 10],
        ),
        (
            {
                "grounding_strategy": "simple",
                "target_phrase": "pen",
            },
            [{"query": "pen", "bbox": [5, 5, 15, 15], "score": 0.6}],
            [5, 5, 15, 15],
        ),
        (
            {
                "grounding_strategy": "ordinal_position",
                "target_phrase": "cup",
                "ordinal": "second",
                "direction": "left",
            },
            [
                {"query": "cup", "bbox": [0, 0, 10, 10], "score": 0.8},
                {
                    "query": "cup",
                    "bbox": [0.2, 0.2, 10.2, 10.2],
                    "score": 0.9,
                },
                {"query": "cup", "bbox": [20, 0, 30, 10], "score": 0.7},
            ],
            [20, 0, 30, 10],
        ),
        (
            {
                "grounding_strategy": "ordinal_position",
                "target_phrase": "cup",
                "ordinal": "first",
                "direction": "right",
            },
            [
                {"query": "cup", "bbox": [0, 0, 10, 10], "score": 0.9},
                {"query": "cup", "bbox": [20, 0, 30, 10], "score": 0.7},
            ],
            [20, 0, 30, 10],
        ),
        (
            {
                "grounding_strategy": "left_right_relation",
                "target_phrase": "cup",
                "reference_object": "teapot",
                "relation": "left",
            },
            [
                {"query": "teapot", "bbox": [48, 48, 52, 52], "score": 0.9},
                {"query": "teapot", "bbox": [-2, 48, 2, 52], "score": 0.5},
                {"query": "cup", "bbox": [43, -2, 47, 2], "score": 0.99},
                {"query": "cup", "bbox": [28, 48, 32, 52], "score": 0.7},
                {"query": "cup", "bbox": [58, 48, 62, 52], "score": 0.8},
            ],
            [28, 48, 32, 52],
        ),
        (
            {
                "grounding_strategy": "distance_relation",
                "target_phrase": "cup",
                "reference_object": "teapot",
                "relation": "closest to",
            },
            [
                {"query": "teapot", "bbox": [48, 48, 52, 52], "score": 0.9},
                {"query": "cup", "bbox": [53, 48, 57, 52], "score": 0.5},
                {"query": "cup", "bbox": [78, 48, 82, 52], "score": 0.99},
            ],
            [53, 48, 57, 52],
        ),
        (
            {
                "grounding_strategy": "distance_relation",
                "target_phrase": "cup",
                "reference_object": "teapot",
                "relation": "farthest from",
            },
            [
                {"query": "teapot", "bbox": [48, 48, 52, 52], "score": 0.9},
                {"query": "cup", "bbox": [53, 48, 57, 52], "score": 0.99},
                {"query": "cup", "bbox": [78, 48, 82, 52], "score": 0.5},
            ],
            [78, 48, 82, 52],
        ),
    ],
)
def test_assumed_grounding_selection_strategies(
    tmp_path: Path,
    roles: dict[str, Any],
    candidates: list[dict[str, Any]],
    expected_bbox: list[float],
) -> None:
    requests, proposals, output = _grounding_files(
        tmp_path, roles, candidates
    )
    reviews_path = build_assumed_grounding_reviews(
        requests, proposals, output, expected_count=1
    )
    review = list(read_jsonl(reviews_path))[0]
    assert review["status"] == "assumed_valid"
    assert review["human_reviewed"] is False
    assert review["claim_status"] == "exploratory"
    assert review["grounding_resolution"] == "strict"
    assert review["grounding_status"] == "auto_assumed_unreviewed"
    assert review["review_id"].startswith("assumed-")
    for model in MODELS:
        target = review["models"][model]["target"]
        assert target["bbox"] == expected_bbox
        assert target["query"] == roles["target_phrase"]
        assert set(target) == {
            "image_path",
            "image_sha256",
            "bbox",
            "bbox_original",
            "bbox_clipped_to_image",
            "image_size",
            "score",
            "query",
            "selection_method",
            "fallback_used",
            "proposal_source_path",
            "proposal_source_sha256",
        }
        assert target["image_sha256"] == sha256_file(target["image_path"])
        assert target["proposal_source_path"] == str(proposals.resolve())
        assert target["proposal_source_sha256"] == sha256_file(proposals)
        assert target["fallback_used"] is False


def test_invalid_model_makes_review_invalid_without_fabricated_bbox(
    tmp_path: Path,
) -> None:
    roles = {"grounding_strategy": "object_identity", "target_phrase": "cup"}
    valid = [{"query": "cup", "bbox": [0, 0, 10, 10], "score": 0.8}]
    requests, proposals, output = _grounding_files(
        tmp_path,
        roles,
        valid,
        candidates_by_model={"qwen": []},
    )
    reviews_path = build_assumed_grounding_reviews(
        requests, proposals, output, expected_count=1
    )
    review = list(read_jsonl(reviews_path))[0]
    assert review["status"] == "invalid"
    assert review["models"]["qwen"]["target"] is None
    assert review["models"]["qwen"]["invalid_reasons"] == [
        "missing_target_candidate"
    ]
    assert review["models"]["roboreward"]["target"] is not None
    assert review["models"]["grm"]["target"] is not None
    assert review["invalid_reasons"] == {
        "qwen": ["missing_target_candidate"]
    }
    audit = json.loads((output / "assumed_review_audit.json").read_text())
    manifest = json.loads(
        (output / "assumed_review_manifest.json").read_text()
    )
    assert audit["all_valid"] is False
    assert audit["invalid_examples"][0]["example_id"] == "e1"
    assert manifest["status_counts"] == {"invalid": 1}
    assert manifest["request_count"] == 1
    assert manifest["proposal_count"] == 3
    assert manifest["review_count"] == 1
    assert manifest["labels_opened"] is False


def test_strict_and_proxy_assumptions_are_never_collapsed(
    tmp_path: Path,
) -> None:
    strict_roles = {
        "grounding_strategy": "object_identity",
        "target_phrase": "cup",
    }
    proxy_roles = {
        "grounding_strategy": "object_identity",
        "target_phrase": "cup",
        "destination": "plate",
    }
    strict_request, strict_images = _request(
        tmp_path, strict_roles, "e-strict"
    )
    proxy_request, proxy_images = _request(
        tmp_path, proxy_roles, "e-proxy"
    )
    # Candidate pooling is keyed by frozen image SHA.  Use different pixels
    # here so the strict cup proposal from the first request cannot
    # legitimately be reused by the proxy request.
    for index, model in enumerate(MODELS):
        Image.new(
            "RGB",
            (100, 100),
            color=(64 + index, 80 + index, 96 + index),
        ).save(proxy_images[model])
        image_sha256 = sha256_file(proxy_images[model])
        if model in {"roboreward", "qwen"}:
            proxy_request["model_frames"][model]["image_sha256"] = image_sha256
        else:
            proxy_request["model_frames"]["grm"]["terminal_views"]["front"][
                "image_sha256"
            ] = image_sha256
    requests_path = tmp_path / "mixed-requests.jsonl"
    proposals_path = tmp_path / "mixed-proposals.jsonl"
    output_dir = tmp_path / "mixed-assumed"
    write_jsonl(requests_path, [strict_request, proxy_request])
    write_jsonl(
        proposals_path,
        [
            *_proposal_rows(
                strict_request,
                strict_images,
                [{"query": "cup", "bbox": [2, 3, 22, 23], "score": 0.9}],
            ),
            *_proposal_rows(
                proxy_request,
                proxy_images,
                [{"query": "plate", "bbox": [4, 5, 24, 25], "score": 0.8}],
            ),
        ],
    )

    reviews_path = build_assumed_grounding_reviews(
        requests_path,
        proposals_path,
        output_dir,
        expected_count=2,
        allow_exploratory_fallbacks=True,
    )
    reviews = {
        str(row["example_id"]): row for row in read_jsonl(reviews_path)
    }
    strict = reviews["e-strict"]
    proxy = reviews["e-proxy"]
    assert (
        strict["status"],
        strict["grounding_resolution"],
        strict["grounding_status"],
    ) == (
        "assumed_valid",
        "strict",
        "auto_assumed_unreviewed",
    )
    assert (
        proxy["status"],
        proxy["grounding_resolution"],
        proxy["grounding_status"],
    ) == (
        "assumed_proxy",
        "proxy",
        "auto_proxy_unreviewed",
    )
    for model in MODELS:
        strict_target = strict["models"][model]["target"]
        proxy_target = proxy["models"][model]["target"]
        assert strict_target["fallback_used"] is False
        assert strict_target["query"] == "cup"
        assert proxy_target["fallback_used"] is True
        assert proxy_target["query"] == "plate"
        assert (
            proxy_target["selection_method"]
            == "destination_proxy_for_missing_target"
        )

    manifest = json.loads(
        (output_dir / "assumed_grounding_manifest.json").read_text()
    )
    assert manifest["status_counts"] == {
        "assumed_proxy": 1,
        "assumed_valid": 1,
    }
    assert manifest["strict_example_count"] == 1
    assert manifest["proxy_example_count"] == 1
    assert (
        manifest["grounding_status"]
        == "mixed_auto_assumed_and_proxy_unreviewed"
    )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("schema", "unsupported proposal schema"),
        ("auto_accepted", "auto_accepted must be false"),
        ("status", "candidate-bearing row must be status=ok"),
        ("sha", "image_sha256 mismatch"),
        ("query", "was not requested"),
        ("group", "group_id differs from frozen request"),
        ("partition", "partition differs from frozen request"),
    ],
)
def test_corrupt_proposal_contract_is_rejected(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    roles = {"grounding_strategy": "object_identity", "target_phrase": "cup"}
    requests, proposals, output = _grounding_files(
        tmp_path,
        roles,
        [{"query": "cup", "bbox": [1, 2, 11, 12], "score": 0.9}],
    )
    rows = list(read_jsonl(proposals))
    row = rows[0]
    if corruption == "schema":
        row["schema_version"] = "my_dataset.grounding_request.corrupt"
    elif corruption == "auto_accepted":
        row["auto_accepted"] = True
    elif corruption == "status":
        row["status"] = "invalid"
    elif corruption == "sha":
        row["image_sha256"] = "0" * 64
    elif corruption == "query":
        row["candidates"][0]["query"] = "unrequested spoon"
    elif corruption == "group":
        row["group_id"] = "g-corrupt"
    elif corruption == "partition":
        row["partition"] = "test"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(corruption)
    write_jsonl(proposals, rows)

    with pytest.raises(ValueError, match=message):
        build_assumed_grounding_reviews(
            requests, proposals, output, expected_count=1
        )


def test_proposal_model_set_must_match_every_request_exactly(
    tmp_path: Path,
) -> None:
    roles = {"grounding_strategy": "object_identity", "target_phrase": "cup"}
    requests, proposals, output = _grounding_files(
        tmp_path,
        roles,
        [{"query": "cup", "bbox": [1, 2, 11, 12], "score": 0.9}],
    )
    rows = list(read_jsonl(proposals))
    rows[1]["model_family"] = rows[0]["model_family"]
    write_jsonl(proposals, rows)
    with pytest.raises(ValueError, match="exactly one proposal per request/model"):
        build_assumed_grounding_reviews(
            requests, proposals, output, expected_count=1
        )


def test_expected_count_and_require_all_valid_are_hard_gates(
    tmp_path: Path,
) -> None:
    roles = {"grounding_strategy": "object_identity", "target_phrase": "cup"}
    requests, proposals, output = _grounding_files(tmp_path, roles, [])
    with pytest.raises(ValueError, match="Expected exactly 2 unique requests"):
        build_assumed_grounding_reviews(
            requests, proposals, output, expected_count=2
        )
    config = {
        "my_dataset_assumed_grounding": {
            "requests_path": str(requests),
            "proposals_path": str(proposals),
            "output_dir": str(output),
            "expected_count": 1,
            "require_all_valid": True,
        }
    }
    with pytest.raises(ValueError, match="require_all_valid"):
        build_assumed_grounding(config)
    assert (output / "reviews_assumed.jsonl").is_file()


def _attention_fixture(
    tmp_path: Path, status: str
) -> tuple[dict[str, Any], Path, Path]:
    roles = {"grounding_strategy": "object_identity", "target_phrase": "cup"}
    if status == "assumed_proxy":
        roles["destination"] = "plate"
    request, model_images = _request(tmp_path, roles)
    request["task_id"] = "task1_1"
    request["instruction"] = "Pick up the cup and place it in the plate."
    videos = {}
    for view in ("front", "left_wrist", "right_wrist"):
        path = tmp_path / f"{view}.mp4"
        path.write_bytes(view.encode())
        videos[view] = str(path)
    view_sha256 = {view: sha256_file(path) for view, path in videos.items()}
    group_media_sha256 = object_fingerprint(
        [(view, view_sha256[view]) for view in videos]
    )
    for model in ("roboreward", "qwen"):
        request["model_frames"][model].update(
            {
                "input_layout": "native_front_video",
                "video_path": videos["front"],
            }
        )
    request["model_frames"]["grm"]["input_layout"] = (
        "grm_native_three_view_endpoints_v1"
    )
    inputs_path = tmp_path / "inputs.jsonl"
    write_jsonl(
        inputs_path,
        [
            {
                "schema_version": "my_dataset.counterfactual.v1",
                "dataset_name": "fixture",
                "example_id": "e1",
                "group_id": "g-e1",
                "group_media_sha256": group_media_sha256,
                "task_id": "task1_1",
                "task_family": "object_identity",
                "instruction": "Pick up the cup and place it in the plate.",
                "evaluation_split": "external_test",
                "video_paths": videos,
                "view_sha256": view_sha256,
            }
        ],
    )
    requests_path = tmp_path / "attention-requests.jsonl"
    write_jsonl(requests_path, [request])
    reviews_path = tmp_path / "attention-reviews.jsonl"
    proposal_source_path = tmp_path / "proposal-source.jsonl"
    write_jsonl(proposal_source_path, [{"fixture": True}])
    proposal_source_sha256 = sha256_file(proposal_source_path)
    models = {}
    for index, model in enumerate(MODELS):
        target = {
            "image_path": model_images[model],
            "image_sha256": sha256_file(model_images[model]),
            "bbox": [index, 1, index + 10, 11],
        }
        if status in {"assumed_valid", "assumed_proxy"}:
            is_proxy = status == "assumed_proxy"
            target.update(
                {
                    "bbox_original": [index, 1, index + 10, 11],
                    "bbox_clipped_to_image": False,
                    "image_size": [100, 100],
                    "score": 0.9 - index / 10,
                    "query": "plate" if is_proxy else "cup",
                    "selection_method": (
                        "destination_proxy_for_missing_target"
                        if is_proxy
                        else "highest_score_exact_query"
                    ),
                    "fallback_used": is_proxy,
                    "proposal_source_path": str(proposal_source_path.resolve()),
                    "proposal_source_sha256": proposal_source_sha256,
                }
            )
        model_review = {
            "target": target,
            "wrong_region": None,
        }
        if status in {"assumed_valid", "assumed_proxy"}:
            model_review.update({"valid": True, "invalid_reasons": []})
        models[model] = model_review
    if status in {"assumed_valid", "assumed_proxy"}:
        is_proxy = status == "assumed_proxy"
        review_metadata = {
            "human_reviewed": False,
            "claim_status": "exploratory",
            "grounding_resolution": "proxy" if is_proxy else "strict",
            "grounding_status": (
                "auto_proxy_unreviewed"
                if is_proxy
                else "auto_assumed_unreviewed"
            ),
            "invalid_reasons": {},
        }
        review_schema = "my_dataset.assumed_grounding.v1"
    else:
        review_metadata = {}
        review_schema = "my_dataset.grounding_review.v1"
    write_jsonl(
        reviews_path,
        [
            {
                "schema_version": review_schema,
                "example_id": "e1",
                "group_id": "g-e1",
                "partition": "discovery",
                "grounding_strategy": "object_identity",
                "review_id": "review-e1",
                "status": status,
                "models": models,
                **review_metadata,
            }
        ],
    )
    split_path = tmp_path / "split.json"
    write_json(
        split_path,
        {
            "examples": {
                "discovery": ["e1"],
                "validation": [],
                "test": [],
            }
        },
    )
    config = {
        "my_dataset_attention": {
            "inputs_path": str(inputs_path),
            "grounding_requests_path": str(requests_path),
            "grounding_reviews_path": str(reviews_path),
            "split_path": str(split_path),
            "output_dir": str(tmp_path / f"attention-{status}"),
            "expected_input_count": 1,
        }
    }
    return config, reviews_path, split_path


@pytest.mark.parametrize(
    (
        "status",
        "expected_resolution",
        "expected_grounding_status",
        "expected_query",
        "expected_policy",
        "expected_fallback",
    ),
    [
        (
            "assumed_valid",
            "strict",
            "auto_assumed_unreviewed",
            "cup",
            "highest_score_exact_query",
            False,
        ),
        (
            "assumed_proxy",
            "proxy",
            "auto_proxy_unreviewed",
            "plate",
            "destination_proxy_for_missing_target",
            True,
        ),
    ],
)
def test_attention_strict_and_proxy_metadata_remain_isolated(
    tmp_path: Path,
    status: str,
    expected_resolution: str,
    expected_grounding_status: str,
    expected_query: str,
    expected_policy: str,
    expected_fallback: bool,
) -> None:
    config, _reviews, _split = _attention_fixture(tmp_path, status)
    cfg = config["my_dataset_attention"]
    cfg.update(
        {
            "accepted_review_statuses": [status],
            "include_all": True,
            "require_all_inputs": True,
        }
    )
    manifest_path = build_attention_manifests(config)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["accepted_review_statuses"] == [status]
    assert manifest["config"] == {
        "accepted_review_statuses": [status],
        "include_all": True,
        "require_all_inputs": True,
        "expected_input_count": 1,
    }
    for model in MODELS:
        assert f"{model}/all" in manifest["artifacts"]
        assert manifest["grounding_resolution_counts"][model] == {
            expected_resolution: 1
        }
        row = list(
            read_jsonl(Path(cfg["output_dir"]) / model / "all.jsonl")
        )[0]
        assert row["grounding_status"] == expected_grounding_status
        assert row["grounding_resolution"] == expected_resolution
        assert row["human_reviewed"] is False
        assert row["claim_status"] == "exploratory"
        selection = row["grounding_selection"]
        assert 0.0 < selection["proposal_score"] <= 1.0
        assert selection["proposal_query"] == expected_query
        assert selection["selection_policy"] == expected_policy
        assert selection["fallback_used"] is expected_fallback
        source_path = Path(selection["proposal_source_path"])
        assert source_path.is_file()
        assert selection["proposal_source_sha256"] == sha256_file(source_path)


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("assumed_valid", "strict assumed review contains proxy target"),
        ("assumed_proxy", "proxy assumed review has no proxy target"),
    ],
)
def test_attention_rejects_cross_contaminated_strict_proxy_flags(
    tmp_path: Path,
    status: str,
    message: str,
) -> None:
    config, reviews_path, _split = _attention_fixture(tmp_path, status)
    config["my_dataset_attention"]["accepted_review_statuses"] = [status]
    review = list(read_jsonl(reviews_path))[0]
    if status == "assumed_valid":
        review["models"]["roboreward"]["target"]["fallback_used"] = True
    else:
        for model in MODELS:
            review["models"][model]["target"]["fallback_used"] = False
    write_jsonl(reviews_path, [review])
    with pytest.raises(ValueError, match=message):
        build_attention_manifests(config)


@pytest.mark.parametrize(
    ("field", "corrupt_value", "message"),
    [
        ("human_reviewed", True, "human_reviewed=false"),
        ("claim_status", "audited", "must be exploratory"),
        ("grounding_resolution", "proxy", "grounding_resolution mismatch"),
        ("grounding_status", "auto_proxy_unreviewed", "grounding_status mismatch"),
    ],
)
def test_attention_rejects_corrupt_assumed_review_labels(
    tmp_path: Path,
    field: str,
    corrupt_value: Any,
    message: str,
) -> None:
    config, reviews_path, _split = _attention_fixture(
        tmp_path, "assumed_valid"
    )
    config["my_dataset_attention"]["accepted_review_statuses"] = [
        "assumed_valid"
    ]
    review = list(read_jsonl(reviews_path))[0]
    review[field] = corrupt_value
    write_jsonl(reviews_path, [review])
    with pytest.raises(ValueError, match=message):
        build_attention_manifests(config)


def test_attention_default_eligible_stays_audited_and_human(
    tmp_path: Path,
) -> None:
    config, reviews_path, _split = _attention_fixture(tmp_path, "eligible")
    legacy_review = list(read_jsonl(reviews_path))[0]
    for model in MODELS:
        assert "valid" not in legacy_review["models"][model]
        assert "invalid_reasons" not in legacy_review["models"][model]
    manifest_path = build_attention_manifests(config)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["accepted_review_statuses"] == ["eligible"]
    row = list(
        read_jsonl(
            Path(config["my_dataset_attention"]["output_dir"])
            / "qwen"
            / "discovery.jsonl"
        )
    )[0]
    assert row["grounding_status"] == "audited_eligible"
    assert row["human_reviewed"] is True
    assert "claim_status" not in row
    assert "grounding_selection" not in row


def test_audited_review_writes_explicit_contract_and_complete_groups(
    tmp_path: Path,
) -> None:
    config, reviews_path, split_path = _attention_fixture(
        tmp_path, "eligible"
    )
    cfg = config["my_dataset_attention"]
    inputs_path = Path(cfg["inputs_path"])
    requests_path = Path(cfg["grounding_requests_path"])
    inputs = list(read_jsonl(inputs_path))
    second_input = dict(inputs[0])
    second_input["example_id"] = "e2"
    write_jsonl(inputs_path, [inputs[0], second_input])
    requests = list(read_jsonl(requests_path))
    second_request = dict(requests[0])
    second_request["example_id"] = "e2"
    write_jsonl(requests_path, [requests[0], second_request])
    requests_sha256 = sha256_file(requests_path)
    proposals_sha256 = "a" * 64
    reviews = list(read_jsonl(reviews_path))
    eligible_review = reviews[0]
    eligible_review.update(
        {
            "review_id": "fixture-reviewer",
            "human_reviewed": True,
            "reviewed_at": "2026-08-04T00:00:00+00:00",
            "request_sha256": requests_sha256,
            "proposals_sha256": proposals_sha256,
        }
    )
    for model in MODELS:
        target = eligible_review["models"][model]["target"]
        eligible_review["models"][model]["wrong_region"] = {
            "image_path": target["image_path"],
            "image_sha256": target["image_sha256"],
            "bbox": [50, 50, 60, 60],
        }
    ineligible_review = {
        "schema_version": "my_dataset.grounding_review.v1",
        "example_id": "e2",
        "status": "ineligible",
        "review_id": "fixture-reviewer",
        "human_reviewed": True,
        "reviewed_at": "2026-08-04T00:00:00+00:00",
        "request_sha256": requests_sha256,
        "proposals_sha256": proposals_sha256,
        "ineligible_reason": "target occluded",
        "models": {},
    }
    write_jsonl(reviews_path, [eligible_review, ineligible_review])
    write_json(
        split_path,
        {
            "examples": {
                "discovery": ["e1", "e2"],
                "validation": [],
                "test": [],
            }
        },
    )
    audit_path = tmp_path / "review_audit.json"
    reviews_sha256 = sha256_file(reviews_path)
    audit = {
        "audit_schema_version": "my_dataset.grounding_review_audit.v2",
        "review_schema_version": "my_dataset.grounding_review.v1",
        "grounding_mode": "human_reviewed",
        "passed": True,
        "human_reviewed": True,
        "expected_count": 2,
        "request_count": 2,
        "review_count": 2,
        "requests_sha256": requests_sha256,
        "reviews_sha256": reviews_sha256,
        "review_sha256": reviews_sha256,
        "proposals_sha256": proposals_sha256,
        "unknown_example_ids": [],
        "missing_example_ids": [],
        "duplicate_request_ids": [],
        "duplicate_review_ids": [],
        "invalid": [],
        "dispositions": {"eligible": 1, "ineligible": 1},
        "eligible_example_count": 1,
        "ineligible_example_count": 1,
        "eligible_example_ids": ["e1"],
        "complete_group_count": 0,
        "complete_group_example_count": 0,
        "incomplete_or_ineligible_group_count": 1,
        "complete_group_ids": [],
        "complete_group_example_ids": [],
    }
    audit["fingerprint"] = object_fingerprint(audit)
    write_json(audit_path, audit)
    cfg.update(
        {
            "expected_input_count": 2,
            "include_all": True,
            "complete_groups_only": True,
            "require_review_audit": True,
            "review_audit_path": str(audit_path),
            "expected_requests_sha256": requests_sha256,
            "expected_proposals_sha256": proposals_sha256,
        }
    )
    manifest_path = build_attention_manifests(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_provenance = {
        "schema_version": "my_dataset.review_provenance.v1",
        "requests_sha256": requests_sha256,
        "proposals_sha256": proposals_sha256,
        "reviews_sha256": reviews_sha256,
        "review_audit_sha256": sha256_file(audit_path),
        "review_audit_fingerprint": audit["fingerprint"],
    }
    assert manifest["review_provenance"] == expected_provenance
    assert manifest["included_example_count"] == 1
    assert manifest["complete_group_example_count"] == 0
    assert manifest["dropped_incomplete_group_count"] == 1
    for model in MODELS:
        all_rows = list(
            read_jsonl(Path(cfg["output_dir"]) / model / "all.jsonl")
        )
        complete_rows = list(
            read_jsonl(
                Path(cfg["output_dir"]) / model / "complete_groups.jsonl"
            )
        )
        assert len(all_rows) == 1
        assert complete_rows == []
        row = all_rows[0]
        assert row["grounding_mode"] == "human_reviewed"
        assert row["grounding_resolution"] == "human_audited"
        assert row["grounding_status"] == "audited_eligible"
        assert row["human_reviewed"] is True
        assert row["claim_status"] == "reviewed_exploratory"
        assert row["review_provenance"] == expected_provenance
        assert len(row["last_image_sha256"]) == 64
        for name in ("all", "complete_groups"):
            artifact = manifest["artifacts"][f"{model}/{name}"]
            artifact_path = Path(artifact["path"])
            assert artifact["sha256"] == sha256_file(artifact_path)


def test_attention_assumed_keeps_strict_image_and_frame_checks(
    tmp_path: Path,
) -> None:
    config, reviews_path, _split = _attention_fixture(
        tmp_path, "assumed_valid"
    )
    cfg = config["my_dataset_attention"]
    cfg["accepted_review_statuses"] = ["assumed_valid"]
    request_path = Path(cfg["grounding_requests_path"])
    review = list(read_jsonl(reviews_path))[0]
    target = review["models"]["roboreward"]["target"]

    expected_sha256 = target["image_sha256"]
    target["image_sha256"] = "0" * 64
    write_jsonl(reviews_path, [review])
    with pytest.raises(ValueError, match="target image SHA differs"):
        build_attention_manifests(config)

    target["image_sha256"] = expected_sha256
    expected_query = target["query"]
    target["query"] = ""
    write_jsonl(reviews_path, [review])
    with pytest.raises(ValueError, match="assumed target query is required"):
        build_attention_manifests(config)

    target["query"] = expected_query
    original_bbox = target["bbox"]
    target["bbox"] = [0, 0, 101, 10]
    write_jsonl(reviews_path, [review])
    with pytest.raises(ValueError, match="target bbox exceeds image bounds"):
        build_attention_manifests(config)

    target["bbox"] = original_bbox
    write_jsonl(reviews_path, [review])
    request = list(read_jsonl(request_path))[0]
    request["model_frames"]["roboreward"].pop("input_layout")
    write_jsonl(request_path, [request])
    with pytest.raises(ValueError, match="frame provenance is inconsistent"):
        build_attention_manifests(config)


def test_attention_expected_count_and_related_id_sets_are_hard_gates(
    tmp_path: Path,
) -> None:
    config, reviews_path, split_path = _attention_fixture(
        tmp_path, "assumed_valid"
    )
    cfg = config["my_dataset_attention"]
    cfg.update(
        {
            "accepted_review_statuses": ["assumed_valid"],
            "require_all_inputs": True,
            "expected_input_count": 2,
        }
    )
    with pytest.raises(ValueError, match="Expected 2 model inputs, found 1"):
        build_attention_manifests(config)

    cfg["expected_input_count"] = 1
    requests_path = Path(cfg["grounding_requests_path"])
    request = list(read_jsonl(requests_path))[0]
    request["example_id"] = "e-other"
    write_jsonl(requests_path, [request])
    with pytest.raises(ValueError, match="grounding requests IDs differ from inputs"):
        build_attention_manifests(config)

    request["example_id"] = "e1"
    write_jsonl(requests_path, [request])
    review = list(read_jsonl(reviews_path))[0]
    review["example_id"] = "e-other"
    write_jsonl(reviews_path, [review])
    with pytest.raises(ValueError, match="grounding reviews IDs differ from inputs"):
        build_attention_manifests(config)

    review["example_id"] = "e1"
    write_jsonl(reviews_path, [review])
    write_json(
        split_path,
        {
            "examples": {
                "discovery": ["e-other"],
                "validation": [],
                "test": [],
            }
        },
    )
    with pytest.raises(ValueError, match="split IDs differ from inputs"):
        build_attention_manifests(config)


def test_attention_require_all_inputs_rejects_invalid_or_unpartitioned(
    tmp_path: Path,
) -> None:
    config, reviews_path, split_path = _attention_fixture(
        tmp_path, "assumed_valid"
    )
    cfg = config["my_dataset_attention"]
    cfg.update(
        {
            "accepted_review_statuses": ["assumed_valid"],
            "require_all_inputs": True,
        }
    )
    review = list(read_jsonl(reviews_path))[0]
    review["status"] = "invalid"
    write_jsonl(reviews_path, [review])
    with pytest.raises(ValueError, match="3 omitted input/model pairs"):
        build_attention_manifests(config)

    review["status"] = "assumed_valid"
    write_jsonl(reviews_path, [review])
    write_json(
        split_path,
        {
            "examples": {
                "discovery": [],
                "validation": [],
                "test": [],
            }
        },
    )
    with pytest.raises(ValueError, match="split IDs differ from inputs"):
        build_attention_manifests(config)


class _FakeTensor:
    def __init__(self, value: Any):
        self.value = value

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self.value)

    def tolist(self) -> Any:
        return self.value

    def __float__(self) -> float:
        return float(self.value)


class _FakeInputs(dict):
    def to(self, _device: str) -> "_FakeInputs":
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.query = ""

    def __call__(
        self, *, images: Any, text: str, return_tensors: str
    ) -> _FakeInputs:
        del images, return_tensors
        self.query = text
        return _FakeInputs(original_sizes=_FakeTensor([[16, 16]]))

    def post_process_instance_segmentation(
        self, outputs: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        del outputs
        scores = {"a": [0.9, 0.8], "b": [0.7, 0.6]}[self.query]
        return [
            {
                "boxes": [
                    _FakeTensor([index * 4, 0, index * 4 + 3, 3])
                    for index in range(2)
                ],
                "masks": [
                    _FakeTensor(np.ones((4, 4), dtype=np.uint8))
                    for _index in range(2)
                ],
                "scores": [_FakeTensor(score) for score in scores],
            }
        ]


class _FakeInference:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


class _FakeTorch:
    def inference_mode(self) -> _FakeInference:
        return _FakeInference()


def _fake_sam3(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> SAM3Grounder:
    grounder = SAM3Grounder(config)
    grounder._processor = _FakeProcessor()
    grounder._model = lambda **_kwargs: object()
    grounder._torch = _FakeTorch()
    grounder._device = "cpu"
    monkeypatch.setattr(grounder, "_load", lambda: None)
    return grounder


def test_sam3_top_n_per_query_prevents_query_starvation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(image_path)
    global_rows = _fake_sam3(
        monkeypatch, {"top_n": 2}
    ).candidates(str(image_path), ["a", "b"])
    per_query_rows = _fake_sam3(
        monkeypatch, {"top_n": 1, "top_n_per_query": 1}
    ).candidates(str(image_path), ["a", "b"])
    assert [row["query"] for row in global_rows] == ["a", "a"]
    assert [row["query"] for row in per_query_rows] == ["a", "b"]
    assert [row["query_priority"] for row in per_query_rows] == [0, 1]
