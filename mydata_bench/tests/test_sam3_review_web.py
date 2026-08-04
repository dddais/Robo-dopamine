from __future__ import annotations

import copy
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import pytest
from PIL import Image

from mydata_bench.io import object_fingerprint, read_jsonl, sha256_file, write_jsonl
from mydata_bench.my_dataset.tracked_grounding import (
    run_manual_retracks,
    run_tracked_grounding,
)
from mydata_bench.review_sam3_grounding_web import (
    MODELS,
    Sam3ReviewStore,
    TrackedGroundingReviewStore,
    _clip_bbox,
    make_handler,
    open_review_store,
)
from mydata_bench.tests.test_tracked_grounding import _fixture_config as _tracked_config
from http.server import ThreadingHTTPServer
from mydata_bench.my_dataset.grounding_manifest import (
    GROUNDING_REQUEST_SCHEMA,
    audit_grounding_review,
)


def _fixture(
    tmp_path: Path,
    candidates: list[dict] | None = None,
    proposal_prefix: list[dict] | None = None,
    proposal_suffix: list[dict] | None = None,
) -> Sam3ReviewStore:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "frame.png"
    Image.new("RGB", (100, 80), "white").save(image)
    digest = sha256_file(image)
    request = {
        "schema_version": GROUNDING_REQUEST_SCHEMA,
        "example_id": "example-1",
        "group_id": "group-1",
        "task_id": "task-1",
        "partition": "test",
        "instruction": "Pick up the blue block.",
        "roles": {"target_phrase": "blue block"},
        "model_frames": {
            "roboreward": {"image_path": str(image), "image_sha256": digest},
            "qwen": {"image_path": str(image), "image_sha256": digest},
            "grm": {
                "terminal_views": {"front": {"image_path": str(image), "image_sha256": digest}}
            },
        },
    }
    write_jsonl(tmp_path / "requests.jsonl", [request])
    candidates = candidates or [
        {"candidate_index": 0, "bbox": [1, 2, 20, 30], "query": "blue block", "score": 0.9},
        {"candidate_index": 1, "bbox": [50, 40, 75, 70], "query": "background", "score": 0.4},
    ]
    proposal_rows = [
            {
                "example_id": "example-1",
                "model_family": model,
                "schema_version": GROUNDING_REQUEST_SCHEMA,
                "status": "ok",
                "image_path": str(image),
                "image_sha256": digest,
                "candidates": candidates,
            }
            for model in MODELS
        ]
    write_jsonl(
        tmp_path / "proposals.jsonl",
        [*(proposal_prefix or []), *proposal_rows, *(proposal_suffix or [])],
    )
    return Sam3ReviewStore(tmp_path, "reviewer-test")


def _eligible_payload() -> dict:
    return {
        "example_id": "example-1",
        "status": "eligible",
        "models": {
            model: {
                "target": {
                    "bbox": [1, 2, 20, 30],
                    "source": "sam3_candidate",
                    "candidate_index": 0,
                },
                "wrong_region": {
                    "bbox": [50, 40, 75, 70],
                    "source": "sam3_candidate",
                    "candidate_index": 1,
                },
            }
            for model in MODELS
        },
    }


def test_review_store_keeps_history_and_materializes_unique_latest(tmp_path: Path) -> None:
    store = _fixture(tmp_path)
    state = store.submit(_eligible_payload())
    assert state["completed"] == 1
    assert state["dispositions"] == {"eligible": 1, "ineligible": 0}
    canonical = list(read_jsonl(store.reviews_path))
    assert len(canonical) == 1
    assert canonical[0]["human_reviewed"] is True
    assert canonical[0]["models"]["grm"]["target"]["candidate_index"] == 0

    store.submit(
        {
            "example_id": "example-1",
            "status": "ineligible",
            "reason": "target is fully occluded",
        }
    )
    assert len(list(read_jsonl(store.history_path))) == 2
    canonical = list(read_jsonl(store.reviews_path))
    assert len(canonical) == 1
    assert canonical[0]["status"] == "ineligible"


def test_review_store_rejects_identical_target_and_wrong_region(tmp_path: Path) -> None:
    store = _fixture(tmp_path)
    payload = _eligible_payload()
    payload["models"]["qwen"]["wrong_region"] = dict(
        payload["models"]["qwen"]["target"]
    )
    with pytest.raises(ValueError, match="overlap"):
        store.submit(payload)



def test_detector_bbox_is_clipped_with_raw_provenance(tmp_path: Path) -> None:
    raw, clipped, changed = _clip_bbox(
        [90, -2.0, 102.35, 40], width=100, height=80, identity="candidate"
    )
    assert raw == [90.0, -2.0, 102.35, 40.0]
    assert clipped == [90.0, 0.0, 100.0, 40.0]
    assert changed is True
    with pytest.raises(ValueError, match="outside the image"):
        _clip_bbox([101, 2, 120, 20], width=100, height=80, identity="candidate")


def test_review_store_recovers_canonical_from_history(tmp_path: Path) -> None:
    store = _fixture(tmp_path)
    store.submit(_eligible_payload())
    store.reviews_path.write_text("{truncated", encoding="utf-8")

    resumed = Sam3ReviewStore(tmp_path, "reviewer-test")
    assert resumed.state(position=1)["saved_review"]["status"] == "eligible"
    canonical = list(read_jsonl(resumed.reviews_path))
    assert len(canonical) == 1
    assert canonical[0]["example_id"] == "example-1"


def test_review_session_rejects_reviewer_change(tmp_path: Path) -> None:
    _fixture(tmp_path)
    with pytest.raises(ValueError, match="session differs"):
        Sam3ReviewStore(tmp_path, "another-reviewer")


def test_clipped_candidate_round_trips_through_submit(tmp_path: Path) -> None:
    candidates = [
        {
            "candidate_index": 0,
            "bbox": [90, -2.0, 102.35, 40],
            "query": "blue block",
            "score": 0.9,
        },
        {
            "candidate_index": 1,
            "bbox": [50, 40, 75, 70],
            "query": "background",
            "score": 0.4,
        },
    ]
    store = _fixture(tmp_path, candidates)
    payload = _eligible_payload()
    for model in MODELS:
        payload["models"][model]["target"]["bbox"] = [90, 0, 100, 40]
    store.submit(payload)
    target = list(read_jsonl(store.reviews_path))[0]["models"]["qwen"]["target"]
    assert target["bbox"] == [90.0, 0.0, 100.0, 40.0]
    assert target["raw_proposal_bbox"] == [90.0, -2.0, 102.35, 40.0]


def test_strict_grounding_audit_is_source_bound(tmp_path: Path) -> None:
    store = _fixture(tmp_path)
    store.submit(_eligible_payload())
    audit_dir = tmp_path / "audit"
    result = audit_grounding_review(
        tmp_path / "requests.jsonl",
        store.reviews_path,
        audit_dir,
        proposals_path=tmp_path / "proposals.jsonl",
    )
    assert result["passed"] is True
    assert result["grounding_mode"] == "human_reviewed"
    assert result["eligible_example_count"] == 1
    assert result["complete_group_example_count"] == 1
    assert result["invalid"] == []
    assert result["requests_sha256"] == store.requests_sha256
    assert result["proposals_sha256"] == store.proposals_sha256
    assert result["wrong_region_token_preflight_passed"] is False
    assert (audit_dir / "review_audit.json").is_file()
    assert (audit_dir / "complete_group_example_ids.json").is_file()


def test_proposal_history_uses_latest_attempt_for_each_valid_key(
    tmp_path: Path,
) -> None:
    failed_attempts = [
        {
            "schema_version": GROUNDING_REQUEST_SCHEMA,
            "example_id": "example-1",
            "model_family": model,
            "status": "error",
        }
        for model in MODELS
    ]
    store = _fixture(tmp_path, proposal_prefix=failed_attempts)
    store.submit(_eligible_payload())
    result = audit_grounding_review(
        tmp_path / "requests.jsonl",
        store.reviews_path,
        tmp_path / "audit",
        proposals_path=tmp_path / "proposals.jsonl",
    )
    assert result["passed"] is True
    assert result["invalid"] == []


def test_latest_failed_proposal_and_unindexable_history_are_rejected(
    tmp_path: Path,
) -> None:
    failed_latest = {
        "schema_version": GROUNDING_REQUEST_SCHEMA,
        "example_id": "example-1",
        "model_family": "qwen",
        "status": "error",
    }
    with pytest.raises(ValueError, match="latest proposal is not ok"):
        _fixture(tmp_path / "failed-latest", proposal_suffix=[failed_latest])

    invalid_key = {
        "schema_version": GROUNDING_REQUEST_SCHEMA,
        "example_id": "example-1",
        "model_family": "unknown-model",
        "status": "error",
    }
    with pytest.raises(ValueError, match="invalid proposal key"):
        _fixture(tmp_path / "invalid-key", proposal_prefix=[invalid_key])



def _tracking_fixture(
    tmp_path: Path, *, manual_strategy: bool = False
) -> tuple[dict, TrackedGroundingReviewStore]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config, _, _ = _tracked_config(tmp_path)
    if manual_strategy:
        roles_path = Path(config["my_dataset_tracked_grounding"]["roles_path"])
        role = list(read_jsonl(roles_path))[0]
        role["grounding_strategy"] = "manual_unparsed"
        write_jsonl(roles_path, [role])
    tracks_path = run_tracked_grounding(config)
    store = TrackedGroundingReviewStore(
        tracks_path.parent,
        "tracking-reviewer",
        tmp_path / "review",
    )
    return config, store


def test_tracking_default_and_alternative_have_bound_terminal_targets(
    tmp_path: Path,
) -> None:
    _, store = _tracking_fixture(tmp_path)
    state = store.state()
    default_id = state["current"]["default_candidate_id"]
    state = store.submit(
        {
            "example_id": "example-1",
            "status": "eligible",
            "candidate_id": default_id,
            "decision_source": "accept_default",
        }
    )
    review = list(read_jsonl(store.reviews_path))[0]
    assert state["completed"] == 1
    assert review["decision"]["source"] == "accept_default"
    assert set(review["models"]) == set(MODELS)
    assert all("wrong_region" not in value for value in review["models"].values())

    _, alternative_store = _tracking_fixture(tmp_path / "alternative")
    option = alternative_store.state()["current"]["options"][1]
    alternative_store.submit(
        {
            "example_id": "example-1",
            "status": "eligible",
            "candidate_id": option["candidate_id"],
            "decision_source": "select_alternative",
        }
    )
    review = list(read_jsonl(alternative_store.reviews_path))[0]
    assert review["decision"]["source"] == "select_alternative"


def test_tracking_no_default_green_preselection_is_still_alternative(
    tmp_path: Path,
) -> None:
    _, store = _tracking_fixture(tmp_path, manual_strategy=True)
    current = store.state()["current"]
    assert current["default_candidate_id"] is None
    option = current["options"][0]
    with pytest.raises(ValueError, match="does not match"):
        store.submit(
            {
                "example_id": "example-1",
                "status": "eligible",
                "candidate_id": option["candidate_id"],
                "decision_source": "accept_default",
            }
        )
    store.submit(
        {
            "example_id": "example-1",
            "status": "eligible",
            "candidate_id": option["candidate_id"],
            "decision_source": "select_alternative",
        }
    )
    assert list(read_jsonl(store.reviews_path))[0]["decision"]["source"] == "select_alternative"


def test_tracking_manual_retrack_two_phase_resume_and_freeze(tmp_path: Path) -> None:
    config, store = _tracking_fixture(tmp_path)
    state = store.submit(
        {
            "example_id": "example-1",
            "status": "needs_retrack",
            "first_bbox": [10.0, 4.0, 16.0, 10.0],
        }
    )
    assert state["phase"] == "awaiting_manual_retrack"
    assert state["completed"] == 1
    anchor = list(read_jsonl(store.manual_anchors_path))[0]
    assert anchor["schema_version"].endswith("manual_anchor.v2")
    assert anchor["note"] is None

    manual_path = store.run_dir / "manual_tracks.jsonl"
    run_manual_retracks(config, store.manual_anchors_path, manual_path)
    resumed = TrackedGroundingReviewStore(
        store.run_dir, "tracking-reviewer", store.output_dir
    )
    state = resumed.state()
    assert state["phase"] == "manual_track_review"
    assert state["completed"] == 0
    assert state["current"]["manual_track"]["candidate_id"] == anchor["manual_anchor_id"]
    with pytest.raises(ValueError, match="已冻结"):
        resumed.submit(
            {
                "example_id": "example-1",
                "status": "needs_retrack",
                "first_bbox": [11.0, 4.0, 17.0, 10.0],
            }
        )
    with pytest.raises(ValueError, match="二阶段 eligible"):
        resumed.submit(
            {
                "example_id": "example-1",
                "status": "eligible",
                "candidate_id": state["current"]["options"][0]["candidate_id"],
                "decision_source": "accept_default",
            }
        )
    resumed.submit(
        {
            "example_id": "example-1",
            "status": "eligible",
            "candidate_id": anchor["manual_anchor_id"],
            "decision_source": "accept_manual_track",
        }
    )
    assert resumed.state(position=1)["manual_artifact_frozen"] is True
    with pytest.raises(ValueError, match="已冻结"):
        resumed.submit(
            {
                "example_id": "example-1",
                "status": "needs_retrack",
                "first_bbox": [11.0, 4.0, 17.0, 10.0],
            }
        )


def test_tracking_manual_artifact_live_change_is_rejected(tmp_path: Path) -> None:
    config, store = _tracking_fixture(tmp_path)
    store.submit(
        {
            "example_id": "example-1",
            "status": "needs_retrack",
            "first_bbox": [10.0, 4.0, 16.0, 10.0],
        }
    )
    anchor = list(read_jsonl(store.manual_anchors_path))[0]
    manual_path = store.run_dir / "manual_tracks.jsonl"
    run_manual_retracks(config, store.manual_anchors_path, manual_path)
    resumed = TrackedGroundingReviewStore(
        store.run_dir, "tracking-reviewer", store.output_dir
    )
    manual_path.write_text(manual_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed during review"):
        resumed.submit(
            {
                "example_id": "example-1",
                "status": "eligible",
                "candidate_id": anchor["manual_anchor_id"],
                "decision_source": "accept_manual_track",
            }
        )


def test_tracking_skip_is_fixed_and_has_no_free_text(tmp_path: Path) -> None:
    _, store = _tracking_fixture(tmp_path)
    with pytest.raises(ValueError, match="free text"):
        store.submit(
            {
                "example_id": "example-1",
                "status": "skipped",
                "disposition_code": "reviewer_skip",
                "reason": "free text is forbidden",
            }
        )
    store.submit(
        {
            "example_id": "example-1",
            "status": "skipped",
            "disposition_code": "reviewer_skip",
        }
    )
    review = list(read_jsonl(store.reviews_path))[0]
    assert review["disposition"] == {"code": "reviewer_skip"}
    assert "reason" not in review and "note" not in review
    fingerprint = review.pop("fingerprint")
    assert fingerprint == object_fingerprint(review)


def test_tracking_attempt_identity_and_manual_anchor_identity_are_strict(
    tmp_path: Path,
) -> None:
    config, store = _tracking_fixture(tmp_path)
    base = list(read_jsonl(store.tracks_path))[0]
    duplicate = tmp_path / "duplicate_tracks.jsonl"
    write_jsonl(duplicate, [base, base])
    with pytest.raises(ValueError, match="strictly increasing"):
        store._load_artifacts(duplicate, complete=True)

    store.submit(
        {
            "example_id": "example-1",
            "status": "needs_retrack",
            "first_bbox": [10.0, 4.0, 16.0, 10.0],
        }
    )
    manual_path = tmp_path / "manual.jsonl"
    run_manual_retracks(config, store.manual_anchors_path, manual_path)
    first = list(read_jsonl(manual_path))[0]
    second = copy.deepcopy(first)
    second["selected_candidate_id"] = "manual-independent"
    second["manual_anchor"]["manual_anchor_id"] = "manual-independent"
    second["manual_anchor"].pop("fingerprint")
    second["manual_anchor"]["fingerprint"] = object_fingerprint(second["manual_anchor"])
    second.pop("fingerprint")
    second["fingerprint"] = object_fingerprint(second)
    independent = tmp_path / "independent_manual.jsonl"
    write_jsonl(independent, [first, second])
    with pytest.raises(ValueError, match="derived manual anchor identity"):
        store._load_artifacts(independent, complete=False)
    write_jsonl(independent, [first, first])
    with pytest.raises(ValueError, match="strictly increasing"):
        store._load_artifacts(independent, complete=False)


def test_auto_mode_and_http_root_work_for_v1_and_tracking(tmp_path: Path) -> None:
    v1 = _fixture(tmp_path / "v1")
    detected_v1 = open_review_store(
        v1.run_dir, "reviewer-test", v1.output_dir, mode="auto"
    )
    assert isinstance(detected_v1, Sam3ReviewStore)
    _, tracked = _tracking_fixture(tmp_path / "v2")
    detected_v2 = open_review_store(
        tracked.run_dir, "tracking-reviewer", tracked.output_dir, mode="auto"
    )
    assert isinstance(detected_v2, TrackedGroundingReviewStore)

    for store, title in ((detected_v1, "SAM3 人工审核"), (detected_v2, "Tracked grounding v2")):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=3) as response:
                body = response.read().decode("utf-8")
            assert response.status == 200
            assert title in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)



def test_tracking_review_store_rejects_tampered_latest_review_fingerprint(
    tmp_path: Path,
) -> None:
    _, store = _tracking_fixture(tmp_path)
    store.submit(
        {
            "example_id": "example-1",
            "status": "skipped",
            "disposition_code": "reviewer_skip",
        }
    )
    row = list(read_jsonl(store.reviews_path))[0]
    assert row["fingerprint"] == object_fingerprint(
        {key: value for key, value in row.items() if key != "fingerprint"}
    )
    row["models"] = {"tampered": True}
    write_jsonl(store.history_path, [row])
    write_jsonl(store.reviews_path, [row])
    with pytest.raises(ValueError, match="invalid fingerprint"):
        TrackedGroundingReviewStore(
            store.run_dir,
            "tracking-reviewer",
            store.output_dir,
        )
