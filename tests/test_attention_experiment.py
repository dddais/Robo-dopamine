from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rewardbench.attention_eval.masking import (
    Head,
    ImageSpan,
    bbox_to_token_positions,
    matched_wrong_position_set,
)
from rewardbench.qwen_eval.attention import QwenAttentionRuntime
from rewardbench.qwen_eval.attention_experiment import (
    _summary,
    build_aligned_ranking_manifest,
    build_cohort_manifest,
    score,
)
from rewardbench.qwen_eval.protocols import ROBO_DOPAMINE_FORWARD, ROBOREWARDBENCH_NATIVE


class AttentionGeometryTests(unittest.TestCase):
    def test_native_last_temporal_span_maps_endpoint_box_and_wrong_control(self) -> None:
        # A native video span is one temporally merged plane.  The endpoint box
        # must be mapped only into the final plane, not all video-token spans.
        earlier = ImageSpan("video_t0", "video.mp4", 10, 14, (1, 4, 4))
        endpoint = ImageSpan("video_t3", "video.mp4", 30, 34, (1, 4, 4))
        positions = bbox_to_token_positions(endpoint, [10, 10, 20, 20], (40, 40))
        self.assertEqual(positions, [30])
        self.assertFalse(set(positions) & set(range(earlier.start, earlier.end)))
        wrong = matched_wrong_position_set(endpoint, positions)
        self.assertIsNotNone(wrong)
        self.assertEqual(len(wrong or []), len(positions))
        self.assertFalse(set(wrong or []) & set(positions))

    def test_forward_paths_preserve_official_eight_image_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, last, blank = root / "first.png", root / "last.png", root / "blank.png"
            for path in (first, last, blank):
                Image.new("RGB", (8, 8), "white").save(path)
            runtime = object.__new__(QwenAttentionRuntime)
            runtime.config = {"blank_goal": str(blank)}
            paths = runtime._forward_paths(
                {"first_image_path": str(first), "last_image_path": str(last)}
            )
            self.assertEqual(paths, [str(first.resolve()), str(blank.resolve()), str(first.resolve()), str(first.resolve()), str(first.resolve()), str(last.resolve()), str(last.resolve()), str(last.resolve())])


class AttentionManifestTests(unittest.TestCase):
    def _image(self, path: Path) -> None:
        Image.new("RGB", (32, 24), "white").save(path)

    def test_aligned_manifest_uses_three_labeled_final_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for name in ("carrot", "bottle", "cube"):
                first, last, video, sequence = (
                    root / f"{name}_first.png",
                    root / f"{name}_last.png",
                    root / f"{name}.mp4",
                    root / f"{name}.json",
                )
                self._image(first)
                self._image(last)
                video.write_bytes(b"placeholder")
                sequence.write_text(
                    json.dumps(
                        [
                            {"image": str(first), "index": 0},
                            {"image": str(last), "index": 9, "chosen": {"bbox": [1, 2, 12, 18]}},
                        ]
                    ),
                    encoding="utf-8",
                )
                sources.append({"name": name, "task": f"pick {name}", "video_path": str(video), "bbox_sequence_path": str(sequence)})
            config = {"attention_steer": {"output_dir": str(root / "out"), "aligned_ranking_sources": sources}}
            path = build_aligned_ranking_manifest(config)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["ranking_source"] for row in rows], ["carrot", "bottle", "cube"])
            self.assertTrue(all("reward" not in row for row in rows))
            self.assertTrue(all(row["bbox_sequence_index"] == 9 for row in rows))

    def test_progressive_manifest_matches_grm_indexing_and_exact_eight_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for name in ("carrot", "bottle", "cube"):
                source_root = root / name
                source_root.mkdir()
                fixed = []
                for slot in range(7):
                    path = source_root / f"fixed_{slot}.png"
                    self._image(path)
                    fixed.append(str(path))
                samples = []
                sequence = []
                for index in range(24):
                    after = source_root / f"frame_{(index + 1) * 20:06d}.png"
                    self._image(after)
                    images = [
                        fixed[0],
                        fixed[1],
                        fixed[2],
                        fixed[3],
                        fixed[4],
                        str(after),
                        fixed[5],
                        fixed[6],
                    ]
                    samples.append(
                        {"id": f"{name}-{index}", "task": f"pick {name}", "image": images}
                    )
                    sequence.append(
                        {
                            "index": index,
                            "image": str(after),
                            "chosen": {"bbox": [1, 2, 12, 18]},
                        }
                    )
                sample_json = source_root / "sample.json"
                bbox_json = source_root / "bbox.json"
                video = source_root / "video.mp4"
                sample_json.write_text(json.dumps(samples), encoding="utf-8")
                bbox_json.write_text(json.dumps(sequence), encoding="utf-8")
                video.write_bytes(b"placeholder")
                sources.append(
                    {
                        "name": name,
                        "task": f"pick {name}",
                        "video_path": str(video),
                        "sample_json_path": str(sample_json),
                        "bbox_sequence_path": str(bbox_json),
                    }
                )
            config = {
                "attention_steer": {
                    "output_dir": str(root / "out"),
                    "protocol": ROBO_DOPAMINE_FORWARD,
                    "ranking_num_samples": 12,
                    "aligned_ranking_sources": sources,
                }
            }
            path = build_aligned_ranking_manifest(config)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 36)
            carrot = [row for row in rows if row["ranking_source"] == "carrot"]
            self.assertEqual(
                [row["ranking_source_sample_index"] for row in carrot],
                list(range(0, 24, 2)),
            )
            self.assertTrue(
                all(row["image_paths"][5] == row["last_image_path"] for row in rows)
            )
            self.assertTrue(
                all(
                    row["media_representation"] == "exact_grm_eight_image_sample"
                    for row in rows
                )
            )

    def test_cohort_manifest_requires_audited_frozen_ids_and_omits_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            test = dataset / "test" / "toy"
            test.mkdir(parents=True)
            video = test / "episode.mp4"
            video.write_bytes(b"video")
            (dataset / "test" / "metadata.jsonl").write_text(
                json.dumps({"file_name": "toy/episode.mp4", "task": "move cube", "reward": 5}) + "\n",
                encoding="utf-8",
            )
            first, last = root / "first.png", root / "last.png"
            self._image(first)
            self._image(last)
            ids = root / "ids.json"
            ids.write_text(json.dumps(["toy/episode.mp4"]), encoding="utf-8")
            grounding = root / "grounding"
            grounding.mkdir()
            (grounding / "audit_final.jsonl").write_text(
                json.dumps({"example_id": "toy/episode.mp4", "formal_eligible": True}) + "\n",
                encoding="utf-8",
            )
            endpoint_rows = [
                {"example_id": "toy/episode.mp4", "frame": "first", "status": "ok", "provenance": {"image_path": str(first)}},
                {"example_id": "toy/episode.mp4", "frame": "last", "status": "ok", "bbox": [1, 2, 12, 18], "grounding_fingerprint": "g", "provenance": {"image_path": str(last)}},
            ]
            (grounding / "grounding.jsonl").write_text("\n".join(json.dumps(row) for row in endpoint_rows) + "\n", encoding="utf-8")
            config = {"attention_steer": {"output_dir": str(root / "out"), "example_ids_file": str(ids), "grounding_run": str(grounding), "dataset_root": str(dataset), "split": "test"}}
            path = build_cohort_manifest(config)
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["example_id"], "toy/episode.mp4")
            self.assertNotIn("reward", row)
            self.assertEqual(row["last_bbox"], [1.0, 2.0, 12.0, 18.0])


class AttentionScoringTests(unittest.TestCase):
    def test_native_and_forward_mappings_are_scored_separately(self) -> None:
        labels = {"a": 5}
        native = _summary(
            [{"example_id": "a", "status": "ok", "native_prediction": 5}],
            labels,
            ROBOREWARDBENCH_NATIVE,
        )
        forward = _summary(
            [{"example_id": "a", "status": "ok", "progress": 0.75}],
            labels,
            ROBO_DOPAMINE_FORWARD,
        )
        self.assertEqual(native["exact_accuracy"], 1)
        self.assertEqual(forward["prediction_counts"], {4: 1})
        self.assertEqual(forward["mae"], 1)

    def test_score_requires_all_four_conditions_for_formal_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            test = dataset / "test" / "toy"
            test.mkdir(parents=True)
            (test / "episode.mp4").write_bytes(b"video")
            (dataset / "test" / "metadata.jsonl").write_text(
                json.dumps({"file_name": "toy/episode.mp4", "task": "move cube", "reward": 5}) + "\n",
                encoding="utf-8",
            )
            ids = root / "ids.json"
            ids.write_text(json.dumps(["toy/episode.mp4"]), encoding="utf-8")
            output = root / "out"
            output.mkdir()
            rows = [
                {"example_id": "toy/episode.mp4", "condition": condition, "status": "ok", "native_prediction": 5}
                for condition in ("baseline", "candidate_target", "candidate_wrong", "low_rank_target")
            ]
            (output / "steering.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            config = {"attention_steer": {"output_dir": str(output), "protocol": ROBOREWARDBENCH_NATIVE, "example_ids_file": str(ids), "dataset_root": str(dataset), "split": "test"}}
            metrics = json.loads(score(config).read_text(encoding="utf-8"))
            self.assertTrue(metrics["completion"]["formal_scoring_ready"])
            self.assertTrue(metrics["official_native_discrete_output"])


if __name__ == "__main__":
    unittest.main()
