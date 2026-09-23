"""Trajectory sampling, exact-frame grounding, smooth overlays and playable MP4s."""
import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from mydata_bench.basic_method.common import file_hash
from mydata_bench.basic_method.grounding import eligibility
from mydata_bench.basic_method.visualization.__main__ import parser
from mydata_bench.basic_method.visualization.video import (
    sample_indices, trajectory_sample, render_clip, rerender,
)
from mydata_bench.basic_method.visualization.video_render import display_heat, overlay
from mydata_bench.basic_method.visualization.progress import predict_progress_frame, progress_series


class VideoGeometryTests(unittest.TestCase):
    def test_progress_only_predicts_without_attention_and_preserves_fallback_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            image = folder / 'image.png'
            Image.new('RGB', (64, 48)).save(image)
            sample = {'visualization_source_frame': 4, 'sampling': {'source_fps': 20},
                      'grm_image_paths': [str(image)] * 8}
            audit = {'prompt_sha256': 'p', 'input_ids_sha256': 'i'}
            base = {'status': 'ok', 'progress': .2, 'token_audit': audit}
            sas = {'status': 'ok', 'progress': .8, 'token_audit': audit}
            runtime = SimpleNamespace(cfg={'bias': 6.}, predict=Mock(side_effect=[base, sas]))
            args = parser().parse_args(['--progress-only'])
            with patch('mydata_bench.basic_method.visualization.progress.eligibility', return_value={'eligible': True}), \
                    patch('mydata_bench.basic_method.visualization.capture.LastPromptCapture', side_effect=AssertionError('Capture used')):
                record = predict_progress_frame(runtime, sample, args, {}, folder / 'eligible')
            self.assertEqual(runtime.predict.call_args_list[0].args, (sample, 'baseline', {}))
            self.assertEqual(runtime.predict.call_args_list[1].args, (sample, 'last_frame:target:8', {}))
            self.assertFalse(record['attention_captured'])
            self.assertNotIn('npz', record)
            self.assertEqual(record['predictions']['sas']['progress'], .8)
            self.assertFalse(list(folder.rglob('*.npz')))
            invalid = {'status': 'parse_error', 'progress': None, 'token_audit': audit}
            runtime.predict = Mock(return_value=invalid)
            with patch('mydata_bench.basic_method.visualization.progress.eligibility',
                       return_value={'eligible': False, 'reason': 'missing_exact_frames'}):
                record = predict_progress_frame(runtime, sample, args, {}, folder / 'fallback')
            runtime.predict.assert_called_once()
            self.assertTrue(record['baseline_fallback'])
            self.assertEqual(record['positive_bias'], 0.)
            self.assertEqual(record['predictions']['sas']['status'], 'parse_error')
            self.assertNotIn('baseline_fallback', invalid)

    def test_progress_curve_keeps_raw_values_and_breaks_on_invalid_predictions(self):
        frames = [{'source_time_seconds': i, 'predictions': {'baseline': row, 'sas': row}}
                  for i, row in enumerate([{'status': 'ok', 'progress': .8}, {'status': 'ok', 'progress': .2},
                                          {'status': 'parse_error', 'progress': .7}, {'status': 'ok', 'progress': None},
                                          {'status': 'ok', 'progress': True}, {'status': 'ok', 'progress': 1.2}])]
        times, values = progress_series(frames)
        np.testing.assert_array_equal(times, np.arange(6))
        np.testing.assert_array_equal(values['baseline'][:2], [.8, .2])
        self.assertTrue(np.isnan(values['baseline'][2:]).all())

    def test_uniform_and_interval_sampling_include_entire_video(self):
        self.assertEqual(sample_indices(424, 4), [0, 141, 282, 423])
        self.assertEqual(sample_indices(10, interval=4), [0, 4, 8, 9])
        self.assertEqual(sample_indices(3, 30), [0, 1, 2])
        self.assertEqual(sample_indices(1, 30), [0])
        self.assertEqual(len(sample_indices(424, 30)), 30)
        with self.assertRaises(ValueError):
            sample_indices(10, 1)

    def test_sampling_updates_all_after_views_and_preserves_original_track_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = Path(tmp) / 'track.json'
            track.write_text(json.dumps({'example_id': 'suc/e', 'video_sha256': 'v', 'terminal_frame_index': 9,
                                         'frames': [{'frame_index': i, 'bbox': [1, 2, 10, 20]} for i in (0, 4, 9)]}))
            parent = {'example_id': 'suc/e', 'video_sha256': 'v', 'grm_image_paths': [f'old{i}' for i in range(8)],
                      'sampling': {'selected_source_indices': [0, 9], 'terminal_source_index': 9},
                      'grounding': {'eligible': True, 'tracking_path': str(track), 'tracking_sha256': file_hash(track)}}
            snapshot = copy.deepcopy(parent)
            extracted = {v: {4: f'{v}4', 5: f'{v}5'} for v in ('front', 'left_wrist', 'right_wrist')}
            sample = trajectory_sample(parent, 4, extracted)
            self.assertEqual(sample['grm_image_paths'][:5], parent['grm_image_paths'][:5])
            self.assertEqual(sample['grm_image_paths'][5:], ['front4', 'left_wrist4', 'right_wrist4'])
            self.assertEqual(sample['sampling']['selected_source_indices'], [0, 4])
            self.assertEqual(sample['sampling']['terminal_source_index'], 9)
            self.assertTrue(eligibility(sample, {'model': 'grm'}, 'all_frames')['eligible'])
            missing = trajectory_sample(parent, 5, extracted)
            check = eligibility(missing, {'model': 'grm'}, 'last_frame')
            self.assertFalse(check['eligible'])
            self.assertEqual(check['missing_frames'], [5])  # Never borrow nearby frame 4.
            self.assertEqual(trajectory_sample(parent, 0, extracted)['grm_image_paths'][5:], ['old0', 'old3', 'old4'])
            self.assertEqual(trajectory_sample(parent, 9, extracted)['grm_image_paths'], parent['grm_image_paths'])
            self.assertEqual(parent, snapshot)

    def test_smooth_heat_and_transparency_preserve_empty_background_and_raw_grid(self):
        grid = np.array([[0., 1.], [0., 0.]], dtype=np.float32)
        original = grid.copy()
        smooth = display_heat(grid, (64, 48), blur_sigma=2)
        self.assertGreater(len(np.unique(smooth[20])), 20)
        self.assertGreater(smooth[10, 55], smooth[35, 10])
        np.testing.assert_array_equal(grid, original)
        source = np.full((48, 64, 3), 120, dtype=np.uint8)
        np.testing.assert_array_equal(overlay(source, grid * 0), source)
        np.testing.assert_array_equal(overlay(source, grid, alpha=0), source)
        shared = display_heat(grid, (64, 48), scale='shared', maximum=10., blur_sigma=0)
        self.assertLessEqual(shared.max(), .13)


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for MP4 integration')
class VideoEncodingTests(unittest.TestCase):
    def test_progress_only_rerender_exports_original_and_curves_without_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            original = source / 'front.mp4'
            writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*'mp4v'), 12., (320, 240))
            self.assertTrue(writer.isOpened())
            for i in range(3):
                writer.write(np.full((240, 320, 3), i * 40, dtype=np.uint8))
            writer.release()
            image = source / 'image.png'
            Image.new('RGB', (320, 240), (95, 120, 140)).save(image)
            frames = []
            for i in range(3):
                base = {'status': 'parse_error' if i == 1 else 'ok', 'progress': None if i == 1 else i / 2}
                frames.append({'source_index': i, 'source_time_seconds': i / 12, 'npz': 'absent.npz',
                               'image_path': str(image), 'image_sha256': file_hash(image), 'baseline_fallback': i == 2,
                               'predictions': {'baseline': base, 'sas': {'status': 'ok', 'progress': i / 2}}})
            manifest = {'example_id': 'suc/test', 'task': 'Pick the cup.', 'scope': 'last_frame', 'top_k': 8,
                        'frames': frames, 'trajectory': {'source_frame_count': 3, 'source_duration_seconds': .25,
                            'source_videos': {'front': {'path': str(original), 'sha256': file_hash(original)}}}}
            (source / 'attention_video_manifest.json').write_text(json.dumps(manifest))
            output = root / 'progress'
            args = parser().parse_args(['--progress-only', '--render-only', str(source), '--output-dir', str(output)])
            with patch('numpy.load', side_effect=AssertionError('Attention arrays loaded')), \
                    patch('mydata_bench.basic_method.runtime.Runtime', side_effect=AssertionError('Model loaded')), \
                    patch('mydata_bench.basic_method.visualization.video.overlay', side_effect=AssertionError('Heatmap rendered')):
                rerender(args)
            rendered = output / 'source'
            self.assertEqual({p.name for p in rendered.glob('*.mp4')}, {'original.mp4', 'video_progress.mp4'})
            self.assertEqual(file_hash(rendered / 'original.mp4'), file_hash(original))
            with (rendered / 'progress_curve.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]['grm_progress'], '')
            self.assertEqual(rows[1]['grm_status'], 'parse_error')
            self.assertEqual(rows[2]['baseline_fallback'], 'True')
            self.assertTrue((rendered / 'progress_curve.png').is_file())
            self.assertNotIn('Front-camera attention', (output / 'index.html').read_text())
            cap = cv2.VideoCapture(str(rendered / 'video_progress.mp4'))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                self.assertTrue(cap.read()[0])
            finally:
                cap.release()

    def test_render_and_model_free_rerender_keep_frame_count_fps_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'capture'
            source.mkdir()
            image = source / 'image.png'
            Image.new('RGB', (160, 120), (95, 120, 140)).save(image)
            frames = []
            for i in range(3):
                npz = source / f'{i}.npz'
                a = np.array([[[.01, .02], [.03, .04]]], dtype=np.float32)
                b = a if i == 1 else a[:, ::-1, :] * 10
                np.savez_compressed(npz, baseline=a, sas=b, heads=[[0, 1]], target=np.array([[False, True], [False, False]]))
                frames.append({'source_index': i * 10, 'source_time_seconds': i / 2, 'npz': npz.name,
                               'npz_sha256': file_hash(npz), 'image_path': str(image), 'image_sha256': file_hash(image),
                               'bbox': [80, 0, 150, 55], 'baseline_fallback': i == 1,
                               'predictions': {'baseline': {'progress': .2}, 'sas': {'progress': .2 if i == 1 else .8}}})
            manifest = {'example_id': 'suc/test', 'task': 'Pick the cup.', 'scope': 'last_frame', 'top_k': 8,
                        'observed_heads': [{'layer': 0, 'head': 1}], 'frames': frames,
                        'trajectory': {'source_frame_count': 21, 'source_duration_seconds': 1.05}}
            (source / 'attention_video_manifest.json').write_text(json.dumps(manifest))
            args = parser().parse_args(['--fps', '3'])
            output = root / 'render'
            result = render_clip(source, manifest, output, args)
            self.assertEqual(result['fallback_frames'], 1)
            for kind in ('baseline', 'sas', 'comparison'):
                cap = cv2.VideoCapture(str(output / (kind + '.mp4')))
                try:
                    self.assertTrue(cap.isOpened())
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                    self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 3)
                    self.assertTrue(cap.read()[0])
                finally:
                    cap.release()
            self.assertTrue((output / 'preview.png').is_file())
            args = parser().parse_args(['--render-only', str(source), '--output-dir', str(root / 'rerender'),
                                       '--video-scale', 'shared', '--fps', '0', '--video-head-index', '-1'])
            with patch('mydata_bench.basic_method.runtime.Runtime', side_effect=AssertionError('Model loaded during render')):
                rerender(args)
            rendered = json.loads((root / 'rerender/capture/render.json').read_text())
            self.assertAlmostEqual(rendered['duration_seconds'], 1.05)
            self.assertAlmostEqual(rendered['shared_maximum'], .4)
            summary = json.loads((root / 'rerender/summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['render_only'])


if __name__ == '__main__':
    unittest.main()
