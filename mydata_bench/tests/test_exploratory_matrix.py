from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mydata_bench.io import (
    object_fingerprint,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from mydata_bench.attention_eval.masking import Head
from mydata_bench.my_dataset import causal_runner
from mydata_bench.my_dataset.checkpoint_manifest import (
    freeze_checkpoint_content_manifest,
)
from mydata_bench.my_dataset.exploratory_matrix import (
    MATRIX_SCHEMA_VERSION,
    PROXY_GROUNDING_STATUS,
    STRICT_GROUNDING_STATUS,
    _assert_hook_applied,
    _runtime_contract,
    run_exploratory_matrix,
)


@dataclass
class _Prepared:
    visual_positions: list[int]


class _FakeRuntime:
    def __init__(self) -> None:
        self.mass_calls: list[str] = []
        self.prepare_calls: list[str] = []
        self.generate_calls: list[dict] = []
        self.processor = SimpleNamespace(
            video_processor=SimpleNamespace(max_frames=32)
        )

    def collect_mass(self, sample: dict) -> dict:
        self.mass_calls.append(sample["example_id"])
        order = int(sample["ranking_order"])
        # Keep every selected layer represented in the top-K while leaving an
        # equal number of lower-ranked, non-overlapping heads in that layer.
        # This makes the fixture capable of exercising the exact layer-matched
        # low-rank control even at K=64.
        values = np.zeros((16, 16), dtype=np.float64)
        for layer in range(values.shape[0]):
            for head in range(values.shape[1]):
                values[layer, head] = head + layer / 100.0 + order / 1000.0
        return {
            "example_id": sample["example_id"],
            "excess_mass": values.tolist(),
            "partition": "runtime-default-discovery",
            "status": "ok",
        }

    def prepare(self, sample: dict) -> _Prepared:
        self.prepare_calls.append(sample["example_id"])
        return _Prepared(visual_positions=[1, 2, 3, 4])

    def target_positions(self, _sample: dict, _prepared: _Prepared) -> list[int]:
        return [1]

    def visual_positions_for_scope(
        self, _prepared: _Prepared, scope: str
    ) -> list[int]:
        assert scope in {"target_slot_only", "all_visual"}
        return [1, 2, 3, 4]

    def wrong_control_positions(
        self, _prepared: _Prepared, target_positions: list[int]
    ) -> tuple[list[int], str]:
        assert target_positions == [1]
        return [2], "fixture_same_target_span"

    def generate(
        self,
        sample: dict,
        *,
        prepared: _Prepared,
        heads,
        selected_positions,
        visual_positions,
        bias,
        query_scope,
    ) -> dict:
        assert isinstance(prepared, _Prepared)
        assert selected_positions in ([1], [2])
        assert visual_positions == [1, 2, 3, 4]
        call = {
            "example_id": sample["example_id"],
            "heads": list(heads),
            "selected_positions": list(selected_positions),
            "visual_positions": list(visual_positions),
            "bias": bias,
            "query_scope": query_scope,
        }
        self.generate_calls.append(call)
        per_layer = {}
        for layer in {head.layer for head in heads}:
            layer_heads = sorted(head.head for head in heads if head.layer == layer)
            per_layer[str(layer)] = {
                "calls": 1,
                "prefill_calls": 1,
                "decode_calls": 0,
                "observed_query_rows": 4,
                "prefill_query_rows": 4,
                "decode_query_rows": 0,
                "applied_calls": 1,
                "prefill_applied_calls": 1,
                "decode_applied_calls": 0,
                "applied_query_rows": 4,
                "prefill_applied_query_rows": 4,
                "decode_applied_query_rows": 0,
                "skipped_calls": 0,
                "missing_mask_calls": 0,
                "selected_heads": layer_heads,
                "selected_token_count": len(selected_positions),
                "other_visual_token_count": len(visual_positions)
                - len(selected_positions),
                "selected_other_disjoint": True,
                "swap_bias": float(bias),
                "query_scope": query_scope,
            }
        return {
            "raw_output": "1",
            "native_prediction": 1,
            "hook_diagnostics": {
                "hook_active": bool(heads and bias),
                "per_layer": per_layer,
            },
        }


def _row(
    example_id: str,
    *,
    ranking_order: int | None = None,
    proxy_grounding: bool = False,
) -> dict:
    grounding_resolution = "proxy" if proxy_grounding else "strict"
    row = {
        "schema_version": "fixture",
        "example_id": example_id,
        "group_id": f"group-{example_id}",
        "task_id": "task1_1",
        "task_family": "object_identity",
        "partition": "test",
        "model_family": "qwen",
        "task": "Pick up the cup.",
        "grounding_status": (
            PROXY_GROUNDING_STATUS
            if proxy_grounding
            else STRICT_GROUNDING_STATUS
        ),
        "grounding_resolution": grounding_resolution,
        "grounding_selection": {
            "proposal_score": 0.75,
            "proposal_query": "cup",
            "selection_policy": f"fixture_{grounding_resolution}",
            "fallback_used": proxy_grounding,
            "fixture_marker": example_id,
        },
        "human_reviewed": False,
        "claim_status": "exploratory",
    }
    if ranking_order is not None:
        row["ranking_order"] = ranking_order
    return row


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    ranking_path = tmp_path / "ordered_max20.jsonl"
    evaluation_path = tmp_path / "all.jsonl"
    output_dir = tmp_path / "matrix"
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"fixture-model")
    write_jsonl(
        ranking_path,
        [
            _row(
                f"rank-{index:03d}",
                ranking_order=index,
                proxy_grounding=index % 2 == 0,
            )
            for index in range(1, 21)
        ],
    )
    write_jsonl(
        evaluation_path,
        [
            _row(f"eval-{index:03d}", proxy_grounding=index == 1)
            for index in range(2)
        ],
    )
    config = {
        "my_dataset_exploratory_matrix": {
            "model_family": "qwen",
            "variant_id": "fixture-qwen",
            "model_path": str(model_path),
            "output_dir": str(output_dir),
            "ranking_manifest": str(ranking_path),
            "evaluation_manifest": str(evaluation_path),
            "protocol": "roborewardbench_native",
            "content_order": "text_then_video",
            "attention_video_max_frames": 32,
            "ranking_prefix_sizes": [5, 10, 20],
            "steering_top_k": [8, 32, 64],
            "ranking_score_kind": "excess_mass",
            "skip_early_layers": 8,
            "swap_bias": 6,
            "steering_query_scope": "all",
            "expected_ranking_count": 20,
            "expected_evaluation_count": 2,
        }
    }
    return config, output_dir


def _bind_reviewed_input_chain(
    config: dict,
    tmp_path: Path,
    *,
    tracking: bool = False,
    review_source_kind: str = "tracked_grounding_v2",
    model_family: str = "qwen",
    source_content_order: str = "text_then_video",
    processor_contract: dict | None = None,
) -> dict[str, object]:
    cfg = config["my_dataset_exploratory_matrix"]
    cfg["model_family"] = model_family
    legacy_provenance = {
        "schema_version": "my_dataset.review_provenance.v1",
        "requests_sha256": "1" * 64,
        "proposals_sha256": "2" * 64,
        "reviews_sha256": "3" * 64,
        "review_audit_sha256": "4" * 64,
        "review_audit_fingerprint": "5" * 64,
    }
    tracking_provenance = {
        "schema_version": "my_dataset.tracking_review_provenance.v2",
        "requests_sha256": "1" * 64,
        "tracking_artifact_sha256": "2" * 64,
        "tracking_manifest_sha256": "3" * 64,
        "manual_tracking_artifact_sha256": None,
        "reviews_sha256": "4" * 64,
        "review_audit_sha256": "5" * 64,
        "review_audit_fingerprint": "6" * 64,
    }
    provenance = tracking_provenance if tracking else legacy_provenance

    ranking_rows = list(read_jsonl(cfg["ranking_manifest"]))
    evaluation_rows = list(read_jsonl(cfg["evaluation_manifest"]))
    for rows in (ranking_rows, evaluation_rows):
        for row in rows:
            row.update(
                {
                    "grounding_mode": "human_reviewed",
                    "grounding_resolution": "human_audited",
                    "grounding_status": "audited_eligible",
                    "human_reviewed": True,
                    "claim_status": "reviewed_exploratory",
                    "review_provenance": provenance,
                    "model_family": model_family,
                }
            )
            if tracking:
                row.pop("review_provenance")
                row.update(
                    {
                        "tracking_review_provenance": provenance,
                        "target_grounding_scope": "terminal_only",
                        "control_region_policy": "none",
                        "content_order": source_content_order,
                        "processor_frame_indices": [0, 7],
                        "processor_video_grid_thw": [[4, 8, 8]],
                    }
                )
                if processor_contract is not None:
                    row["processor_content_order_contract"] = processor_contract
            row.pop("grounding_selection", None)

    ranking_root = tmp_path / "ranking_cohort_reviewed"
    attention_root = tmp_path / "attention_reviewed"
    ranking_path = ranking_root / model_family / "ordered_max20.jsonl"
    evaluation_path = attention_root / model_family / "complete_groups.jsonl"
    write_jsonl(ranking_path, ranking_rows)
    write_jsonl(evaluation_path, evaluation_rows)
    cfg["ranking_manifest"] = str(ranking_path)
    cfg["evaluation_manifest"] = str(evaluation_path)

    evaluation_artifact = {
        "path": str(evaluation_path.resolve()),
        "count": len(evaluation_rows),
        "sha256": sha256_file(evaluation_path),
        "fingerprint": object_fingerprint(evaluation_rows),
    }
    attention_manifest = {
        "schema_version": "my_dataset.attention_input.v1",
        "require_review_audit": True,
        "accepted_review_statuses": ["eligible"],
        "grounding_requests_sha256": provenance["requests_sha256"],
        "expected_requests_sha256": provenance["requests_sha256"],
        "grounding_proposals_sha256": legacy_provenance["proposals_sha256"],
        "expected_proposals_sha256": legacy_provenance["proposals_sha256"],
        "grounding_reviews_sha256": provenance["reviews_sha256"],
        "review_audit_sha256": provenance["review_audit_sha256"],
        "review_audit_fingerprint": provenance["review_audit_fingerprint"],
        "review_audit_schema_version": "my_dataset.grounding_review_audit.v2",
        "review_provenance": legacy_provenance,
        "artifacts": {
            f"{model_family}/complete_groups": dict(evaluation_artifact)
        },
    }
    if tracking:
        for field in (
            "grounding_proposals_sha256",
            "expected_proposals_sha256",
            "review_provenance",
        ):
            attention_manifest.pop(field)
        attention_manifest.update(
            {
                "review_source_kind": review_source_kind,
                "tracking_artifact_sha256": provenance[
                    "tracking_artifact_sha256"
                ],
                "expected_tracking_artifact_sha256": provenance[
                    "tracking_artifact_sha256"
                ],
                "tracking_manifest_sha256": provenance[
                    "tracking_manifest_sha256"
                ],
                "expected_tracking_manifest_sha256": provenance[
                    "tracking_manifest_sha256"
                ],
                "manual_tracking_artifact_sha256": None,
                "expected_manual_tracking_artifact_sha256": None,
                "review_audit_schema_version": (
                    "my_dataset.tracked_grounding_review_audit.v2"
                ),
                "tracking_review_provenance": provenance,
                "target_grounding_scope": "terminal_only",
                "control_region_policy": "none",
            }
        )
    attention_manifest["fingerprint"] = object_fingerprint(attention_manifest)
    attention_manifest_path = attention_root / "manifest.json"
    write_json(attention_manifest_path, attention_manifest)

    selection_manifest = {
        "schema_version": "my_dataset.external_ranking_cohort.v1",
        "grounding_mode": "human_reviewed",
        "claim_status": "reviewed_exploratory",
        "review_provenance": provenance,
        "attention_manifest": {
            "path": str(attention_manifest_path.resolve()),
            "sha256": sha256_file(attention_manifest_path),
            "fingerprint": attention_manifest["fingerprint"],
        },
        "evaluation_population": {
            "ranking_attention_filename": "all.jsonl",
            "evaluation_attention_filename": "complete_groups.jsonl",
        },
        "attention_inputs": {
            "ranking": {},
            "evaluation": {model_family: dict(evaluation_artifact)},
        },
        "model_outputs": {
            model_family: {
                "path": str(ranking_path.resolve()),
                "count": len(ranking_rows),
                "sha256": sha256_file(ranking_path),
                "fingerprint": object_fingerprint(ranking_rows),
            }
        },
    }
    if tracking:
        selection_manifest.pop("review_provenance")
        selection_manifest.update(
            {
                "review_source_kind": review_source_kind,
                "tracking_review_provenance": provenance,
                "target_grounding_scope": "terminal_only",
                "control_region_policy": "none",
            }
        )
    selection_manifest["fingerprint"] = object_fingerprint(selection_manifest)
    write_json(ranking_root / "selection_manifest.json", selection_manifest)
    return provenance


def _processor_content_order_contract() -> dict:
    runs = {}
    for index, order in enumerate(("text_then_video", "video_then_text"), 1):
        files = [
            {
                "path": f"/fixture/{order}.jsonl",
                "sha256": str(index) * 64,
                "size_bytes": 1,
            }
        ]
        runs[order] = {
            "run_dir": f"/fixture/{order}",
            "content_order": order,
            "manifest_path": f"/fixture/{order}/manifest.json",
            "manifest_sha256": str(index + 2) * 64,
            "manifest_config_fingerprint": str(index + 4) * 64,
            "record_files": files,
            "record_files_fingerprint": object_fingerprint(files),
            "baseline_record": {
                "path": files[0]["path"],
                "file_sha256": files[0]["sha256"],
                "row_fingerprint": str(index + 6) * 64,
                "content_order_source": "run_manifest",
            },
        }
    value = {
        "schema_version": "my_dataset.processor_content_order_contract.v1",
        "model_family": "roboreward",
        "validated_orders": ["text_then_video", "video_then_text"],
        "shared_processor_frame_indices": [0, 7],
        "shared_processor_video_grid_thw": [[4, 8, 8]],
        "shared_video_metadata": {
            "frame_count": 8,
            "width": 64,
            "height": 64,
            "fps": 30.0,
        },
        "runs": runs,
    }
    value["fingerprint"] = object_fingerprint(value)
    return value


def _bind_reviewed_checkpoint(config: dict, tmp_path: Path) -> tuple[Path, Path]:
    cfg = config["my_dataset_exploratory_matrix"]
    model_path = tmp_path / "reviewed-checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"model_type":"fixture"}\n', encoding="utf-8"
    )
    (model_path / "model-00001-of-00001.safetensors").write_bytes(b"A" * 4096)
    manifest_path = tmp_path / "reviewed-checkpoint-manifest.json"
    freeze_checkpoint_content_manifest(model_path, manifest_path)
    cfg["model_path"] = str(model_path)
    cfg["checkpoint_content_manifest"] = str(manifest_path)
    return model_path, manifest_path


def test_matrix_mass_rank_grid_resume_retry_and_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output_dir = _fixture(tmp_path)
    matrix_cfg = config["my_dataset_exploratory_matrix"]
    ranking_inputs = {
        row["example_id"]: row
        for row in read_jsonl(matrix_cfg["ranking_manifest"])
    }
    evaluation_inputs = {
        row["example_id"]: row
        for row in read_jsonl(matrix_cfg["evaluation_manifest"])
    }
    first = _FakeRuntime()
    loads: list[dict] = []

    def load_runtime(cfg: dict) -> _FakeRuntime:
        loads.append(dict(cfg))
        return first

    monkeypatch.setattr(causal_runner, "_runtime", load_runtime)
    manifest_path = run_exploratory_matrix(config)

    assert len(loads) == 1
    assert loads[0]["capture_generation_attentions"] is False
    assert len(first.mass_calls) == 20
    assert len(set(first.mass_calls)) == 20
    assert first.prepare_calls == ["eval-000", "eval-001"]
    assert len(first.generate_calls) == 56

    mass = list(read_jsonl(output_dir / "ranking" / "mass.jsonl"))
    assert len(mass) == 20
    assert {row["example_id"] for row in mass} == set(first.mass_calls)
    for row in mass:
        source = ranking_inputs[row["example_id"]]
        assert row["grounding_status"] == source["grounding_status"]
        assert row["grounding_resolution"] == source["grounding_resolution"]
        assert row["grounding_selection"] == source["grounding_selection"]
        assert row["partition"] == source["partition"] == "test"
    ranking_fingerprints = {}
    expected_compositions = {
        5: {"strict_count": 3, "proxy_count": 2, "total": 5, "proxy_ratio": 0.4},
        10: {"strict_count": 5, "proxy_count": 5, "total": 10, "proxy_ratio": 0.5},
        20: {"strict_count": 10, "proxy_count": 10, "total": 20, "proxy_ratio": 0.5},
    }
    for size in (5, 10, 20):
        path = output_dir / "ranking" / f"rank_n{size:03d}.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert artifact["sample_count"] == size
        assert artifact["ranking_n"] == size
        assert len(artifact["ranking"]) == 128
        assert min(row["layer"] for row in artifact["ranking"]) >= 8
        assert artifact["grounding_status"] == "mixed"
        assert artifact["grounding_resolution"] == "mixed"
        assert artifact["grounding_composition"] == expected_compositions[size]
        ranking_fingerprints[size] = artifact["fingerprint"]
    assert len(set(ranking_fingerprints.values())) == 3

    records_path = output_dir / "steering" / "records.jsonl"
    records = list(read_jsonl(records_path))
    assert len(records) == 2 * 28
    assert {row["schema_version"] for row in records} == {MATRIX_SCHEMA_VERSION}
    for example_id in ("eval-000", "eval-001"):
        rows = [row for row in records if row["example_id"] == example_id]
        source = evaluation_inputs[example_id]
        assert len(rows) == 28
        assert {row["condition_kind"] for row in rows} == {
            "baseline",
            "candidate_target",
            "candidate_wrong_region",
            "low_rank_target",
        }
        assert {row["grounding_status"] for row in rows} == {
            source["grounding_status"]
        }
        assert {row["grounding_resolution"] for row in rows} == {
            source["grounding_resolution"]
        }
        assert all(
            row["grounding_selection"] == source["grounding_selection"]
            for row in rows
        )
    top64 = [
        row
        for row in records
        if row["condition"]
        == "candidate_target__rank_n020__top_k064"
    ]
    assert len(top64) == 2
    assert all(len(row["heads"]) == 64 for row in top64)
    assert all(row["bias"] == 6.0 and row["scope"] == "all" for row in top64)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["grounding_status"] == "mixed"
    assert manifest["grounding_resolution"] == "mixed"
    assert manifest["grounding_composition"] == {
        "ranking": expected_compositions[20],
        "evaluation": {
            "strict_count": 1,
            "proxy_count": 1,
            "total": 2,
            "proxy_ratio": 0.5,
        },
    }
    assert manifest["run_fingerprint_components"]["grounding"][
        "ranking"
    ]["composition"] == expected_compositions[20]
    assert manifest["run_fingerprint_components"]["grounding"][
        "evaluation"
    ]["composition"] == manifest["grounding_composition"]["evaluation"]
    implementation = manifest["run_fingerprint_components"]["implementation"]
    assert set(implementation["source_sha256"]) == {
            "mydata_bench/my_dataset/exploratory_matrix.py",
            "mydata_bench/my_dataset/causal_runner.py",
            "mydata_bench/my_dataset/checkpoint_manifest.py",
            "mydata_bench/my_dataset/grounding_contract.py",
        "mydata_bench/config.py",
        "mydata_bench/io.py",
        "mydata_bench/protocol.py",
        "mydata_bench/attention_eval/masking.py",
        "mydata_bench/attention_eval/runtime.py",
        "mydata_bench/qwen_eval/attention.py",
        "mydata_bench/qwen_eval/protocols.py",
        "mydata_bench/roboreward_eval/runner.py",
    }
    assert all(implementation["source_sha256"].values())
    generation = manifest["run_fingerprint_components"]["model"]["runtime"][
        "generation_contract"
    ]
    library_versions = manifest["run_fingerprint_components"]["model"][
        "runtime"
    ]["library_versions"]
    assert set(library_versions) == {
        "python",
        "torch",
        "transformers",
        "numpy",
        "pillow",
    }
    assert all(isinstance(value, str) and value for value in library_versions.values())
    assert generation["attn_implementation"] == "eager"
    assert generation["decoding"] == "greedy"
    assert generation["do_sample"] is False
    assert generation["use_cache"] is True
    assert manifest["steering"]["condition_count"] == 28
    assert manifest["steering"]["controls"] == [
        "candidate_wrong_region",
        "layer_matched_low_rank_target",
    ]

    # A partial records file resumes only its missing (example, condition)
    # without recollecting ranking mass or rerunning the other 27 conditions.
    write_jsonl(records_path, records[:-1])
    partial = _FakeRuntime()
    monkeypatch.setattr(causal_runner, "_runtime", lambda _cfg: partial)
    run_exploratory_matrix(config)
    assert partial.mass_calls == []
    assert partial.prepare_calls == ["eval-001"]
    assert len(partial.generate_calls) == 1
    resumed = list(read_jsonl(records_path))[-1]
    assert resumed["grounding_resolution"] == "proxy"
    assert resumed["grounding_status"] == PROXY_GROUNDING_STATUS
    assert resumed["grounding_selection"] == evaluation_inputs[
        "eval-001"
    ]["grounding_selection"]

    # An invalid latest attempt is a hard stop unless retry_failed is explicit.
    current = list(read_jsonl(records_path))
    current[-1]["status"] = "invalid"
    current[-1]["error"] = "fixture failure"
    write_jsonl(records_path, current)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load before invalid-resume rejection")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(RuntimeError, match="retry_failed=True"):
        run_exploratory_matrix(config)

    retried = _FakeRuntime()
    monkeypatch.setattr(causal_runner, "_runtime", lambda _cfg: retried)
    run_exploratory_matrix(config, retry_failed=True)
    assert retried.mass_calls == []
    assert retried.prepare_calls == ["eval-001"]
    assert len(retried.generate_calls) == 1
    retry_row = list(read_jsonl(records_path))[-1]
    assert retry_row["grounding_resolution"] == "proxy"
    assert retry_row["grounding_selection"]["fallback_used"] is True

    # A complete run returns after input/fingerprint validation and before
    # constructing a model runtime.
    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    assert run_exploratory_matrix(config) == manifest_path

    changed = {
        "my_dataset_exploratory_matrix": {
            **config["my_dataset_exploratory_matrix"],
            "content_order": "video_then_text",
        }
    }
    with pytest.raises(RuntimeError, match="run fingerprint mismatch"):
        run_exploratory_matrix(changed)

    # A valid grounding-resolution change is represented in provenance and
    # invalidates an existing run before any runtime can be loaded.
    evaluation_path = Path(matrix_cfg["evaluation_manifest"])
    changed_grounding = list(read_jsonl(evaluation_path))
    changed_grounding[0]["grounding_resolution"] = "proxy"
    changed_grounding[0]["grounding_status"] = PROXY_GROUNDING_STATUS
    changed_grounding[0]["grounding_selection"]["fallback_used"] = True
    write_jsonl(evaluation_path, changed_grounding)
    with pytest.raises(RuntimeError, match="run fingerprint mismatch"):
        run_exploratory_matrix(config)


def test_matrix_resume_rejects_same_fingerprint_tampered_ordered_heads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, output_dir = _fixture(tmp_path)
    monkeypatch.setattr(causal_runner, "_runtime", lambda _cfg: _FakeRuntime())
    run_exploratory_matrix(config)

    records_path = output_dir / "steering" / "records.jsonl"
    records = list(read_jsonl(records_path))
    candidate = next(
        row
        for row in records
        if row["condition"] == "candidate_target__rank_n005__top_k008"
    )
    candidate["heads"][0], candidate["heads"][1] = (
        candidate["heads"][1],
        candidate["heads"][0],
    )
    write_jsonl(records_path, records)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load before resume head validation")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(RuntimeError, match="Frozen heads mismatch"):
        run_exploratory_matrix(config)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("mismatched_pair", "grounding_resolution/grounding_status"),
        ("proxy_unstructured", "structured mapping"),
        ("proxy_without_fallback", "fallback_used must be true"),
    ],
)
def test_matrix_rejects_invalid_grounding_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    evaluation_path = Path(
        config["my_dataset_exploratory_matrix"]["evaluation_manifest"]
    )
    rows = list(read_jsonl(evaluation_path))
    if case == "mismatched_pair":
        rows[0]["grounding_status"] = PROXY_GROUNDING_STATUS
    elif case == "proxy_unstructured":
        rows[1]["grounding_selection"] = None
    elif case == "proxy_without_fallback":
        rows[1]["grounding_selection"]["fallback_used"] = False
    else:  # pragma: no cover - parameter table is frozen above.
        raise AssertionError(case)
    write_jsonl(evaluation_path, rows)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load before grounding validation")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match=message):
        run_exploratory_matrix(config)


@pytest.mark.parametrize("manifest_key", ("ranking_manifest", "evaluation_manifest"))
def test_matrix_rejects_manifest_model_family_mismatch_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_key: str,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    manifest_path = Path(cfg[manifest_key])
    rows = list(read_jsonl(manifest_path))
    rows[0]["model_family"] = "grm"
    write_jsonl(manifest_path, rows)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load before manifest validation")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match="model_family must equal configured"):
        run_exploratory_matrix(config)


def test_matrix_accepts_only_explicit_reviewed_contract_and_auto_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "reviewed-addendum.md"
    addendum.write_text("reviewed robustness rerun", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )
    provenance = _bind_reviewed_input_chain(config, tmp_path)
    _model_path, checkpoint_manifest_path = _bind_reviewed_checkpoint(
        config, tmp_path
    )

    runtime = _FakeRuntime()
    monkeypatch.setattr(causal_runner, "_runtime", lambda _cfg: runtime)
    manifest_path = run_exploratory_matrix(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["human_reviewed"] is True
    assert manifest["grounding_mode"] == "human_reviewed"
    assert manifest["claim_status"] == "reviewed_exploratory"
    assert manifest["reference_variant_id"] == "fixture-qwen-unreviewed"
    assert manifest["review_provenance"] == provenance
    assert manifest["reviewed_input_chain"]["review_provenance"] == provenance
    assert manifest["steering"]["expected_record_count"] == 56
    assert manifest["grounding_composition"]["evaluation"] == {
        "human_audited_count": 2,
        "total": 2,
        "human_audited_ratio": 1.0,
    }
    components = manifest["run_fingerprint_components"]
    assert components["grounding"]["contract"]["mode"] == "human_reviewed"
    assert components["input"]["expected_evaluation_count_mode"] == "auto"
    assert components["input"]["protocol_addendum"]["path"] == str(
        addendum.resolve()
    )
    checkpoint_verification = manifest["checkpoint_content_verification"]
    assert checkpoint_verification["passed"] is True
    assert checkpoint_verification["all_checkpoint_file_bytes_hashed"] is True
    assert checkpoint_verification["manifest_path"] == str(
        checkpoint_manifest_path.resolve()
    )
    assert (
        components["model"]["checkpoint_content_verification"]
        == checkpoint_verification
    )
    records = list(read_jsonl(output_dir / "steering" / "records.jsonl"))
    assert {row["human_reviewed"] for row in records} == {True}
    assert {row["reference_variant_id"] for row in records} == {
        "fixture-qwen-unreviewed"
    }
    assert {row["claim_status"] for row in records} == {
        "reviewed_exploratory"
    }
    assert {object_fingerprint(row["review_provenance"]) for row in records} == {
        object_fingerprint(provenance)
    }


@pytest.mark.parametrize(
    "review_source_kind", ("tracked_grounding_v2", "tracked_grounding_v3")
)
def test_tracking_reviewed_matrix_propagates_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    review_source_kind: str,
) -> None:
    config, output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "tracking-v2-addendum.md"
    addendum.write_text("tracking v2", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )
    provenance = _bind_reviewed_input_chain(
        config,
        tmp_path,
        tracking=True,
        review_source_kind=review_source_kind,
    )
    _bind_reviewed_checkpoint(config, tmp_path)
    monkeypatch.setattr(causal_runner, "_runtime", lambda _cfg: _FakeRuntime())

    manifest_path = run_exploratory_matrix(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["review_source_kind"] == review_source_kind
    assert manifest["tracking_review_provenance"] == provenance
    assert manifest["target_grounding_scope"] == "terminal_only"
    assert manifest["control_region_policy"] == "none"
    assert "review_provenance" not in manifest
    mass = list(read_jsonl(output_dir / "ranking" / "mass.jsonl"))
    steering = list(read_jsonl(output_dir / "steering" / "records.jsonl"))
    rank = json.loads(
        (output_dir / "ranking" / "rank_n005.json").read_text(
            encoding="utf-8"
        )
    )
    for value in [*mass, *steering, rank]:
        assert value["tracking_review_provenance"] == provenance
        assert value["target_grounding_scope"] == "terminal_only"
        assert value["control_region_policy"] == "none"


@pytest.mark.parametrize(
    ("contract_mode", "message"),
    (
        ("missing", "without the frozen dual-order processor contract"),
        ("tampered", "processor content-order fingerprint is invalid"),
    ),
)
def test_tracking_rr_cross_order_fails_before_model_load_without_valid_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_mode: str,
    message: str,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "tracking-v2-addendum.md"
    addendum.write_text("tracking v2", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-rr-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
            "content_order": "text_then_video",
        }
    )
    contract = None
    if contract_mode == "tampered":
        contract = _processor_content_order_contract()
        contract["fingerprint"] = "f" * 64
    _bind_reviewed_input_chain(
        config,
        tmp_path,
        tracking=True,
        model_family="roboreward",
        source_content_order="video_then_text",
        processor_contract=contract,
    )
    _bind_reviewed_checkpoint(config, tmp_path)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load before processor contract validation")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match=message):
        run_exploratory_matrix(config)


def test_tracking_rr_cross_order_accepts_frozen_processor_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "tracking-v2-addendum.md"
    addendum.write_text("tracking v2", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-rr-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
            "content_order": "text_then_video",
        }
    )
    _bind_reviewed_input_chain(
        config,
        tmp_path,
        tracking=True,
        model_family="roboreward",
        source_content_order="video_then_text",
        processor_contract=_processor_content_order_contract(),
    )
    _bind_reviewed_checkpoint(config, tmp_path)
    monkeypatch.setattr(causal_runner, "_runtime", lambda _cfg: _FakeRuntime())

    manifest = json.loads(
        run_exploratory_matrix(config).read_text(encoding="utf-8")
    )
    binding = manifest["reviewed_input_chain"][
        "processor_content_order_binding"
    ]
    assert binding["runtime_content_order"] == "text_then_video"
    assert binding["source_content_orders"] == ["video_then_text"]


@pytest.mark.parametrize("artifact_kind", ("ranking", "evaluation"))
def test_reviewed_matrix_rejects_tampered_same_id_jsonl_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "reviewed-addendum.md"
    addendum.write_text("reviewed robustness rerun", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )
    _bind_reviewed_input_chain(config, tmp_path)
    _bind_reviewed_checkpoint(config, tmp_path)
    path = Path(
        cfg[
            "ranking_manifest"
            if artifact_kind == "ranking"
            else "evaluation_manifest"
        ]
    )
    rows = list(read_jsonl(path))
    rows[0]["same_id_tamper"] = artifact_kind
    write_jsonl(path, rows)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load before reviewed artifact validation")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match="SHA-256 differs from parent manifest"):
        run_exploratory_matrix(config)


def test_reviewed_matrix_rejects_parent_provenance_mismatch_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "reviewed-addendum.md"
    addendum.write_text("reviewed robustness rerun", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )
    _bind_reviewed_input_chain(config, tmp_path)
    _bind_reviewed_checkpoint(config, tmp_path)
    ranking_path = Path(cfg["ranking_manifest"])
    selection_path = ranking_path.parent.parent / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["review_provenance"]["requests_sha256"] = "6" * 64
    selection.pop("fingerprint")
    selection["fingerprint"] = object_fingerprint(selection)
    write_json(selection_path, selection)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load before reviewed parent validation")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match="different provenance"):
        run_exploratory_matrix(config)


def test_reviewed_matrix_rejects_all_jsonl_evaluation_misbinding_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "reviewed-addendum.md"
    addendum.write_text("reviewed robustness rerun", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )
    _bind_reviewed_input_chain(config, tmp_path)
    _bind_reviewed_checkpoint(config, tmp_path)

    complete_path = Path(cfg["evaluation_manifest"])
    all_path = complete_path.with_name("all.jsonl")
    all_rows = list(read_jsonl(complete_path))
    write_jsonl(all_path, all_rows)
    cfg["evaluation_manifest"] = str(all_path)

    attention_path = all_path.parent.parent / "manifest.json"
    attention = json.loads(attention_path.read_text(encoding="utf-8"))
    attention["artifacts"]["qwen/all"] = {
        "path": str(all_path.resolve()),
        "count": len(all_rows),
        "sha256": sha256_file(all_path),
        "fingerprint": object_fingerprint(all_rows),
    }
    attention.pop("fingerprint")
    attention["fingerprint"] = object_fingerprint(attention)
    write_json(attention_path, attention)

    ranking_path = Path(cfg["ranking_manifest"])
    selection_path = ranking_path.parent.parent / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["attention_manifest"] = {
        "path": str(attention_path.resolve()),
        "sha256": sha256_file(attention_path),
        "fingerprint": attention["fingerprint"],
    }
    selection.pop("fingerprint")
    selection["fingerprint"] = object_fingerprint(selection)
    write_json(selection_path, selection)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load after all.jsonl misbinding")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match="must be complete_groups.jsonl"):
        run_exploratory_matrix(config)


def test_reviewed_matrix_rejects_selection_evaluation_artifact_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "reviewed-addendum.md"
    addendum.write_text("reviewed robustness rerun", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )
    _bind_reviewed_input_chain(config, tmp_path)
    _bind_reviewed_checkpoint(config, tmp_path)

    ranking_path = Path(cfg["ranking_manifest"])
    selection_path = ranking_path.parent.parent / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["attention_inputs"]["evaluation"]["qwen"]["sha256"] = "0" * 64
    selection.pop("fingerprint")
    selection["fingerprint"] = object_fingerprint(selection)
    write_json(selection_path, selection)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load after selection artifact drift")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(
        ValueError,
        match="reviewed selection evaluation qwen SHA-256 differs",
    ):
        run_exploratory_matrix(config)


def test_reviewed_matrix_requires_checkpoint_manifest_before_runtime_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "reviewed-addendum.md"
    addendum.write_text("reviewed robustness rerun", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load without a checkpoint manifest")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match="requires checkpoint_content_manifest"):
        run_exploratory_matrix(config)


def test_reviewed_matrix_rejects_checkpoint_model_path_mismatch_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "reviewed-addendum.md"
    addendum.write_text("reviewed robustness rerun", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )
    _bind_reviewed_input_chain(config, tmp_path)
    frozen_model, _manifest_path = _bind_reviewed_checkpoint(config, tmp_path)
    wrong_model = tmp_path / "wrong-checkpoint-path"
    wrong_model.mkdir()
    for source in frozen_model.iterdir():
        (wrong_model / source.name).write_bytes(source.read_bytes())
    cfg["model_path"] = str(wrong_model)

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load after checkpoint path mismatch")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match="model_path differs"):
        run_exploratory_matrix(config)


def test_reviewed_matrix_rejects_same_size_checkpoint_drift_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    addendum = tmp_path / "reviewed-addendum.md"
    addendum.write_text("reviewed robustness rerun", encoding="utf-8")
    cfg.update(
        {
            "grounding_mode": "human_reviewed",
            "reference_variant_id": "fixture-qwen-unreviewed",
            "protocol_addendum": str(addendum),
            "expected_evaluation_count": "auto",
        }
    )
    _bind_reviewed_input_chain(config, tmp_path)
    model_path, _manifest_path = _bind_reviewed_checkpoint(config, tmp_path)
    shard = model_path / "model-00001-of-00001.safetensors"
    frozen_size = shard.stat().st_size
    shard.write_bytes(b"B" * frozen_size)
    assert shard.stat().st_size == frozen_size

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load after same-size checkpoint drift")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_exploratory_matrix(config)


def test_matrix_auto_count_rejects_empty_evaluation_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _output_dir = _fixture(tmp_path)
    cfg = config["my_dataset_exploratory_matrix"]
    cfg["expected_evaluation_count"] = "auto"
    write_jsonl(Path(cfg["evaluation_manifest"]), [])

    def must_not_load(_cfg: dict):
        raise AssertionError("model must not load for an empty evaluation manifest")

    monkeypatch.setattr(causal_runner, "_runtime", must_not_load)
    with pytest.raises(
        ValueError,
        match="evaluation manifest must contain at least one row",
    ):
        run_exploratory_matrix(config)


def test_grm_runtime_contract_does_not_compare_unused_video_cap() -> None:
    runtime = SimpleNamespace(
        processor=SimpleNamespace(
            video_processor=SimpleNamespace(max_frames=768)
        )
    )
    grm_cfg = {
        "model_family": "grm",
        "protocol": "grm_official_eight_image",
        "content_order": "not_applicable_eight_image",
        "attention_video_max_frames": 8,
    }
    _runtime_contract(runtime, grm_cfg)

    qwen_cfg = {
        **grm_cfg,
        "model_family": "qwen",
        "protocol": "roborewardbench_native",
        "content_order": "video_then_text",
    }
    with pytest.raises(RuntimeError, match="frame cap"):
        _runtime_contract(runtime, qwen_cfg)


def test_hook_assertion_rejects_any_unapplied_scope_all_call() -> None:
    heads = [Head(layer=8, head=3)]
    evidence = {
        "hook_diagnostics": {
            "hook_active": True,
            "per_layer": {
                "8": {
                        "calls": 2,
                        "prefill_calls": 1,
                        "decode_calls": 0,
                        "observed_query_rows": 3,
                        "prefill_query_rows": 3,
                        "decode_query_rows": 0,
                        "applied_calls": 1,
                        "prefill_applied_calls": 1,
                        "decode_applied_calls": 0,
                        "applied_query_rows": 3,
                        "prefill_applied_query_rows": 3,
                        "decode_applied_query_rows": 0,
                        "skipped_calls": 1,
                    "missing_mask_calls": 1,
                    "selected_heads": [3],
                    "selected_token_count": 1,
                    "other_visual_token_count": 3,
                    "selected_other_disjoint": True,
                    "swap_bias": 6.0,
                    "query_scope": "all",
                }
            },
        }
    }
    with pytest.raises(AssertionError, match="did not apply on every"):
        _assert_hook_applied(evidence, heads)
