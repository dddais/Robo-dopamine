from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from roborewardbench.data import (
    CATEGORY_OXE_COUNTERFACTUAL,
    CATEGORY_OXE_ORIGINAL,
    CATEGORY_OXE_TEMPORAL,
    CATEGORY_ROBOARENA,
    classify_source,
    load_metadata_reference,
)
from roborewardbench.metrics import (
    DEFAULT_THRESHOLDS,
    EXPECTED_TEST_SUBSETS,
    apply_thresholds,
    calibrate_progress,
    compute_metrics,
    fit_monotonic_calibration,
    interval_ordinal_error,
    load_calibration,
    save_calibration,
)
from roborewardbench.run_benchmark import (
    aggregate_incremental,
    fingerprint_files,
    iter_local_examples,
    normalize_subset,
    parse_score,
    predict_forward_batch,
    predict_example,
    uniform_indices,
    Example,
)
from roborewardbench.score import score_records


class MetricTests(unittest.TestCase):
    def test_fixed_threshold_boundaries_are_explicit(self) -> None:
        self.assertEqual(DEFAULT_THRESHOLDS, (0.125, 0.375, 0.625, 0.875))
        self.assertEqual(apply_thresholds(0.124999), 1)
        self.assertEqual(apply_thresholds(0.125), 2)
        self.assertEqual(apply_thresholds(0.875), 5)

    def test_interval_error_does_not_penalize_inside_label_cell(self) -> None:
        self.assertEqual(interval_ordinal_error(0.125, 1), 0.0)
        self.assertEqual(interval_ordinal_error(0.25, 2), 0.0)
        self.assertAlmostEqual(interval_ordinal_error(0.0, 2), 0.5)

    def test_macro_and_micro_mae_are_both_reported(self) -> None:
        records = [{"progress": 0.0, "reward": 1, "subset": "small", "status": "ok"}]
        records.extend(
            {"progress": 0.0, "reward": 5, "subset": "large", "status": "ok"}
            for _ in range(3)
        )
        metrics = compute_metrics(records)
        official = metrics["benchmark_compatible_fixed_bin"]
        self.assertEqual(official["macro_mae"], 2.0)
        self.assertEqual(official["micro_mae"], 3.0)
        self.assertFalse(metrics["official_comparable"])

    def test_invalid_records_are_counted_not_scored(self) -> None:
        records = [
            {"progress": 0.0, "reward": 1, "subset": "a", "status": "ok"},
            {"reward": 5, "subset": "a", "status": "invalid"},
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["num_valid"], 1)
        self.assertEqual(metrics["num_invalid"], 1)
        self.assertEqual(metrics["invalid_rate"], 0.5)

    def test_discrete_accuracy_distribution_and_confusion_matrix(self) -> None:
        records = [
            {"progress": 0.0, "reward": 1, "subset": "a", "status": "ok"},
            {"progress": 0.2, "reward": 1, "subset": "a", "status": "ok"},
            {"progress": 0.9, "reward": 1, "subset": "a", "status": "ok"},
        ]
        classification = compute_metrics(records)["discrete_classification"]
        self.assertAlmostEqual(
            classification["exact_accuracy"]["micro_accuracy"], 1 / 3
        )
        self.assertAlmostEqual(
            classification["within_one_accuracy"]["micro_accuracy"], 2 / 3
        )
        self.assertEqual(
            classification["prediction_counts"],
            {"1": 1, "2": 1, "3": 0, "4": 0, "5": 1},
        )
        self.assertEqual(
            classification["confusion_matrix"]["1"],
            {"1": 1, "2": 1, "3": 0, "4": 0, "5": 1},
        )
        self.assertAlmostEqual(classification["mean_signed_error"], 5 / 3)
        self.assertAlmostEqual(classification["overprediction_rate"], 2 / 3)
        self.assertEqual(classification["underprediction_rate"], 0.0)

    def test_official_comparability_requires_exact_metadata(self) -> None:
        subsets = sorted(EXPECTED_TEST_SUBSETS)
        records = [
            {
                "id": f"fabricated-{index}",
                "progress": 0.0,
                "reward": 1,
                "task": "fabricated",
                "subset": subsets[index % len(subsets)],
                "split": "test",
                "status": "ok",
            }
            for index in range(2831)
        ]
        self.assertFalse(compute_metrics(records)["official_comparable"])
        expected = {
            row["id"]: {
                "reward": row["reward"],
                "task": row["task"],
                "subset": row["subset"],
            }
            for row in records
        }
        self.assertTrue(
            compute_metrics(records, expected_records=expected)["official_comparable"]
        )

    def test_scoring_uses_latest_retry_for_each_id(self) -> None:
        records = [
            {"id": "x", "reward": 5, "subset": "a", "status": "invalid"},
            {"id": "x", "progress": 1.0, "reward": 5, "subset": "a", "status": "ok"},
        ]
        metrics = score_records(records, bootstrap_samples=0)
        self.assertEqual(metrics["duplicate_record_count"], 1)
        self.assertEqual(metrics["raw"]["num_valid"], 1)


class CalibrationTests(unittest.TestCase):
    def test_calibration_requires_validation_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_monotonic_calibration(
                [{"progress": 0.2, "reward": 2, "subset": "a", "split": "test"}]
            )

    def test_calibration_is_monotonic_and_round_trips(self) -> None:
        rows = [
            {"progress": 0.1, "reward": 1, "split": "validation"},
            {"progress": 0.3, "reward": 3, "split": "validation"},
            {"progress": 0.6, "reward": 2, "split": "validation"},
            {"progress": 0.9, "reward": 5, "split": "validation"},
        ]
        calibration = fit_monotonic_calibration(rows)
        calibrated = [calibrate_progress(x, calibration) for x in (0.1, 0.3, 0.6, 0.9)]
        self.assertEqual(calibrated, sorted(calibrated))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_calibration(calibration, path)
            self.assertEqual(load_calibration(path), calibration)


class RunnerUtilityTests(unittest.TestCase):
    def test_terminal_sampling_includes_endpoints(self) -> None:
        self.assertEqual(uniform_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(uniform_indices(2, 8), [0, 1])

    def test_strict_score_parser(self) -> None:
        self.assertEqual(parse_score("<score>+25%</score>"), 0.25)
        self.assertEqual(parse_score("<score>+66.7%</score>"), 0.667)
        self.assertEqual(parse_score("<score>-100%</score>"), -1.0)
        with self.assertRaises(ValueError):
            parse_score("I think <score>25%</score>")
        with self.assertRaises(ValueError):
            parse_score("<score>101%</score>")

    def test_incremental_aggregation(self) -> None:
        self.assertAlmostEqual(aggregate_incremental([0.5, 0.5]), 0.75)
        self.assertAlmostEqual(aggregate_incremental([0.5, -0.5]), 0.25)

    def test_metadata_loader_preserves_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test" / "robo_arena").mkdir(parents=True)
            metadata = {
                "file_name": "robo_arena/example.mp4",
                "task": "pick object",
                "reward": 4,
            }
            with (root / "test" / "metadata.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(metadata) + "\n")
            example = next(iter_local_examples(root, "test"))
            self.assertEqual(example.subset, "robo_arena")
            self.assertEqual(example.reward, 4)
            reference = load_metadata_reference(root / "test" / "metadata.jsonl")
            self.assertEqual(reference["num_records"], 1)

    def test_metadata_loader_rejects_video_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test").mkdir()
            metadata = {
                "file_name": "../../outside.mp4",
                "task": "pick object",
                "reward": 4,
            }
            with (root / "test" / "metadata.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(metadata) + "\n")
            with self.assertRaisesRegex(ValueError, "escapes"):
                list(iter_local_examples(root, "test"))

    def test_subset_normalization_is_platform_independent(self) -> None:
        self.assertEqual(normalize_subset("droid/foo.mp4"), "droid")
        self.assertEqual(normalize_subset("robo_arena\\foo.mp4"), "robo_arena")

    def test_released_source_categories_are_explicit(self) -> None:
        self.assertEqual(classify_source("robo_arena/x.mp4", 3), CATEGORY_ROBOARENA)
        self.assertEqual(
            classify_source("droid/x_attempt_1_score_3.mp4", 3),
            CATEGORY_OXE_TEMPORAL,
        )
        self.assertEqual(classify_source("droid/x.mp4", 2), CATEGORY_OXE_COUNTERFACTUAL)
        self.assertEqual(classify_source("droid/x.mp4", 5), CATEGORY_OXE_ORIGINAL)
        with self.assertRaisesRegex(ValueError, "encodes score"):
            classify_source("droid/x_attempt_1_score_3.mp4", 2)

    def test_metrics_report_source_categories_when_ids_are_available(self) -> None:
        records = [
            {
                "id": "robo_arena/x.mp4",
                "progress": 0.0,
                "reward": 1,
                "task": "x",
                "subset": "robo_arena",
            },
            {
                "id": "droid/x.mp4",
                "progress": 1.0,
                "reward": 5,
                "task": "x",
                "subset": "droid",
            },
        ]
        categories = compute_metrics(records)["source_category_metrics"]
        self.assertEqual(categories[CATEGORY_ROBOARENA]["count"], 1)
        self.assertEqual(categories[CATEGORY_OXE_ORIGINAL]["count"], 1)

    def test_forward_prediction_uses_only_true_endpoints(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.items = []

            def inference_batch(self, items):
                self.items.extend(items)
                return [{**item, "pred": "<score>+75%</score>"} for item in items]

        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "rollout.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (16, 16),
            )
            self.assertTrue(writer.isOpened())
            for value in range(10):
                writer.write(np.full((16, 16, 3), value * 20, dtype=np.uint8))
            writer.release()

            model = FakeModel()
            prediction = predict_example(
                model,
                Example("x", video_path, "move object", 4, "droid", "test"),
                mode="forward",
                frame_sampling="1fps",
                max_states=None,
                batch_size=1,
            )
            self.assertEqual(prediction["sampled_frame_indices"], [0, 9])
            self.assertEqual(prediction["progress"], 0.75)
            self.assertEqual(len(model.items), 1)

    def test_forward_batch_uses_one_model_call_and_matches_outputs_by_id(self) -> None:
        class ReversingFakeModel:
            def __init__(self) -> None:
                self.calls = 0
                self.batch_size = 0

            def inference_batch(self, items):
                self.calls += 1
                self.batch_size = len(items)
                outputs = []
                for index, item in enumerate(items):
                    score = 25 if index == 0 else 75
                    outputs.append({**item, "pred": f"<score>+{score}%</score>"})
                return list(reversed(outputs))

        with tempfile.TemporaryDirectory() as directory:
            examples = []
            for video_index in range(2):
                video_path = Path(directory) / f"rollout_{video_index}.avi"
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    10.0,
                    (16, 16),
                )
                self.assertTrue(writer.isOpened())
                for value in range(3):
                    writer.write(np.full((16, 16, 3), value * 20, dtype=np.uint8))
                writer.release()
                examples.append(
                    Example(
                        f"x{video_index}", video_path, "move object", 4, "droid", "test"
                    )
                )

            model = ReversingFakeModel()
            outcomes = predict_forward_batch(model, examples)
            self.assertEqual(model.calls, 1)
            self.assertEqual(model.batch_size, 2)
            self.assertEqual(outcomes["x0"]["progress"], 0.25)
            self.assertEqual(outcomes["x1"]["progress"], 0.75)

    def test_forward_batch_isolates_output_parse_failures(self) -> None:
        class PartlyMalformedModel:
            def inference_batch(self, items):
                return [
                    {**items[0], "pred": "malformed"},
                    {**items[1], "pred": "<score>+75%</score>"},
                ]

        with tempfile.TemporaryDirectory() as directory:
            examples = []
            for video_index in range(2):
                video_path = Path(directory) / f"rollout_{video_index}.avi"
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    10.0,
                    (16, 16),
                )
                self.assertTrue(writer.isOpened())
                for value in range(3):
                    writer.write(np.full((16, 16, 3), value * 20, dtype=np.uint8))
                writer.release()
                examples.append(
                    Example(
                        f"x{video_index}", video_path, "move object", 4, "droid", "test"
                    )
                )

            outcomes = predict_forward_batch(PartlyMalformedModel(), examples)
            self.assertIsInstance(outcomes["x0"], ValueError)
            self.assertEqual(outcomes["x1"]["progress"], 0.75)

    def test_dataset_fingerprint_changes_with_video_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.bin"
            video.write_bytes(b"first")
            first = fingerprint_files([video], root)
            video.write_bytes(b"second")
            second = fingerprint_files([video], root)
            self.assertNotEqual(first["sha256"], second["sha256"])


if __name__ == "__main__":
    unittest.main()
