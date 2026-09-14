"""Behavior tests for exact frames, fallback, ranking isolation and token domains."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch

from .common import create_json, file_hash, fingerprint, validate_media
from .grounding import (MissingGrounding, ControlUnavailable, UnusableGeometry, exact_boxes, eligibility,
                        required_frames, sample_identity)
from .runtime import Runtime, align_exact
from .run import cache_rows, fallback_row, predict_condition, rank
from .score import score_experiment


class GroundingContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def sample(self, eid='e', available=range(8), box=None):
        path = self.root / (eid.replace('/', '_') + '.json')
        create_json(path, {'example_id': eid, 'video_sha256': 'same-video', 'terminal_frame_index': 7,
                          'frames': [{'frame_index': i, 'bbox': box or [0, 0, 16, 16]} for i in available]})
        return {'example_id': eid, 'video_sha256': 'same-video', 'task': 'pick ' + eid,
                'holdout': True, 'sampling': {'selected_source_indices': list(range(8)), 'terminal_source_index': 7},
                'grounding': {'eligible': True, 'tracking_path': str(path), 'tracking_sha256': file_hash(path)}}

    def test_missing_middle_only_disables_required_scopes(self):
        sample = self.sample(available=[0, 1, 3, 4, 5, 6, 7])
        cfg = {'model': 'meter', 'protocol': 'text_image'}
        self.assertTrue(eligibility(sample, cfg, 'last_frame')['eligible'])
        check = eligibility(sample, cfg, 'all_frames')
        self.assertFalse(check['eligible'])
        self.assertEqual(check['missing_frames'], [2])
        with self.assertRaises(MissingGrounding):
            exact_boxes(sample, [2])

    def test_sole_preflights_every_recursive_current_frame(self):
        sample = self.sample(available=[0, 1, 3, 4, 5, 6, 7])
        cfg = {'model': 'sole', 'protocol': 'official'}
        self.assertEqual(required_frames(sample, cfg, 'last_frame'), list(range(1, 8)))
        self.assertFalse(eligibility(sample, cfg, 'last_frame')['eligible'])

    def test_native_video_last_unit_requires_both_actual_frames(self):
        sample = self.sample(available=[0, 2, 7])
        cfg = {'model': 'qwen', 'protocol': 'official'}
        check = eligibility(sample, cfg, 'last_frame', [0, 2, 5, 7])
        self.assertEqual(check['missing_frames'], [5])
        self.assertEqual(required_frames(sample, cfg, 'last_frame', [0, 2, 7]), [7])
        with self.assertRaises(ValueError):
            required_frames(sample, cfg, 'last_frame')

    def test_missing_file_and_wrong_instruction_identity_are_not_fallback(self):
        sample = self.sample()
        wrong = {**sample, 'example_id': 'another-instruction'}
        with self.assertRaisesRegex(ValueError, 'example_id mismatch'):
            exact_boxes(wrong, [7])
        sample['grounding']['tracking_path'] = str(self.root / 'missing.json')
        with self.assertRaises(FileNotFoundError):
            eligibility(sample, {'model': 'meter', 'protocol': 'official'}, 'last_frame')

    def test_changed_track_is_rejected_even_after_cache_hit(self):
        sample = self.sample()
        exact_boxes(sample, [7])
        path = Path(sample['grounding']['tracking_path'])
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Frozen track changed'):
            exact_boxes(sample, [7])

    def test_baseline_alignment_never_opens_track(self):
        payload = {'videos': [], 'span_specs': [{'sources': [7], 'size': [64, 64], 'mosaic': False}]}
        with patch('mydata_bench.basic_method.runtime.exact_boxes', side_effect=AssertionError('grounding touched')):
            mapping = align_exact([0] + [151655] * 4, [[1, 4, 4]], payload, {}, {'model': 'meter'})
        self.assertEqual(mapping['target'], {})

    def test_zero_bias_prediction_uses_original_attention_path(self):
        runtime = Runtime.__new__(Runtime)
        runtime.cfg = {'model': 'qwen', 'protocol': 'text_image', 'bias': 0., 'max_new_tokens': 2}
        runtime.prepare = Mock(return_value=({'input_ids': torch.tensor([[1, 2]])}, {'records': []}, 1, 'prompt'))
        runtime.model = SimpleNamespace(device='cpu', dtype=torch.float32,
                                        generate=Mock(return_value=torch.tensor([[1, 2, 3]])))
        runtime.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0),
                                            batch_decode=Mock(return_value=['ANSWER: 5']))
        runtime.controller = SimpleNamespace(steer=Mock(side_effect=AssertionError('Zero bias enabled a dense mask')),
                                             clear=Mock())
        row = runtime.predict({'example_id': 'e'}, 'last_frame:target:1', {'ranking': [{'layer': 0, 'head': 0}]})
        runtime.controller.steer.assert_not_called()
        self.assertEqual(row['attention_diagnostics'], {})
        self.assertEqual(row['predicted_reward'], 5)

    def test_video_target_is_token_union_without_background_gap(self):
        sample = self.sample(available=[0, 7])
        payload = {'videos': [True], 'span_specs': [{'sources': [0, 7], 'size': [64, 32], 'mosaic': False}]}
        with patch('mydata_bench.basic_method.runtime.exact_boxes', return_value={0: [0, 0, 16, 16], 7: [48, 0, 64, 16]}):
            mapping = align_exact([0] + [151656] * 8, [[1, 4, 8]], payload, sample, {'model': 'qwen'}, 'last_frame')
        self.assertEqual(mapping['target']['last_frame'], [1, 4])
        self.assertEqual(len(mapping['wrong']['last_frame']), 2)

    def test_grm_masks_only_front_slots_and_last_means_after_high(self):
        sample = self.sample(available=[0, 7])
        payload = {'videos': [], 'span_specs': [{'sources': [0 if i < 5 else 7], 'size': [64, 64], 'mosaic': False}
                                               for i in range(8)]}
        ids = [v for _ in range(8) for v in [0, 151655, 151655, 151655, 151655]]
        for scope, indices in [('last_frame', [5]), ('all_frames', [0, 2, 5])]:
            mapping = align_exact(ids, [[1, 4, 4]] * 8, payload, sample, {'model': 'grm'}, scope)
            domain = set(mapping['target'][scope]) | set(mapping['negative'][scope])
            self.assertEqual(domain, {5 * i + j for i in indices for j in (1, 2, 3, 4)})

    def test_sole_final_tile_does_not_require_unselected_tiles(self):
        sample = self.sample(available=[7])
        payload = {'videos': [], 'span_specs': [{'sources': [0, 6, 7], 'size': [1162, 384],
                                               'source_size': [64, 64], 'mosaic': True}]}
        mapping = align_exact([0] + [151655] * 444, [[1, 24, 74]], payload, sample,
                              {'model': 'sole'}, 'last_frame')
        self.assertEqual(mapping['alignment']['last_frame'][0]['tracking_frames'], [7])

    def test_sole_last_frame_excludes_keys_crossing_into_previous_tile(self):
        sample = self.sample(available=[7], box=[100, 100, 160, 160])
        payload = {'videos': [], 'span_specs': [{'sources': [0, 6, 7], 'size': [1162, 384],
                                               'source_size': [640, 480], 'mosaic': True}]}
        mapping = align_exact([0] + [151655] * 444, [[1, 24, 74]], payload, sample,
                              {'model': 'sole'}, 'last_frame')
        domain = set(mapping['target']['last_frame']) | set(mapping['negative']['last_frame'])
        self.assertEqual(len(mapping['excluded_mosaic_boundary_keys']), 10)
        self.assertEqual(len(domain), 120)
        self.assertTrue(all((p - 1) % 37 * 1162 / 37 >= 778 for p in domain))
        self.assertTrue(domain.isdisjoint(mapping['excluded_mosaic_boundary_keys']))

    def test_sole_target_only_in_a_mixed_key_is_explicitly_unusable(self):
        sample = self.sample(available=[7], box=[0, 20, 1, 30])
        payload = {'videos': [], 'span_specs': [{'sources': [0, 6, 7], 'size': [1162, 384],
                                               'source_size': [64, 64], 'mosaic': True}]}
        with self.assertRaises(UnusableGeometry):
            align_exact([0] + [151655] * 444, [[1, 24, 74]], payload, sample, {'model': 'sole'}, 'last_frame')

    def test_sole_geometry_preflight_checks_all_steps_and_is_cached(self):
        sample = self.sample()
        runtime = Runtime.__new__(Runtime)
        runtime._geometry_checks, runtime._control_checks = {}, {}
        mapping = {'control_unavailable': None, 'excluded_mosaic_boundary_keys': [1]}
        runtime.prepare = Mock(side_effect=[(None, mapping, None, None), UnusableGeometry([2])]
                               + [(None, mapping, None, None)] * 5)
        check = runtime.geometry_preflight(sample, 'last_frame')
        self.assertFalse(check['eligible'])
        self.assertEqual(check['unusable_frames'], [2])
        self.assertEqual(runtime.prepare.call_count, 7)
        self.assertEqual(runtime.geometry_preflight(sample, 'last_frame'), check)
        self.assertEqual(runtime.prepare.call_count, 7)

    def baseline(self, sample, status='ok'):
        return {'example_id': sample['example_id'], 'sample_id': sample_identity(sample), 'run_id': 'run',
                'condition': 'baseline', 'status': status, 'progress': 0.75 if status == 'ok' else None,
                'raw_output': 'baseline output', 'sas_applied': False, 'baseline_fallback': False}

    def test_fallback_keeps_prediction_and_parse_failure_and_checks_identity(self):
        sample = self.sample()
        for status in ('ok', 'parse_error'):
            base = self.baseline(sample, status)
            fallback = fallback_row(sample, 'all_frames:target:8', base, 'baseline.jsonl', {'reason': 'missing_exact_frames'}, 'run')
            self.assertEqual(fallback['status'], status)
            self.assertEqual(fallback['progress'], base['progress'])
            self.assertEqual(fallback['positive_bias'], 0)
            self.assertEqual(fallback['negative_bias'], 0)
            self.assertEqual(base['condition'], 'baseline')
        with self.assertRaisesRegex(ValueError, 'different input'):
            fallback_row({**sample, 'task': 'different'}, 'all_frames:target:8', base, 'x', {'reason': 'x'}, 'run')

    def test_entire_recursive_sample_falls_back_without_any_model_calls(self):
        sample = self.sample(available=[0, 7])
        base = self.baseline(sample)
        path = self.root / 'predictions/baseline.jsonl'
        path.parent.mkdir()
        path.write_text(json.dumps(base) + '\n')
        runtime = SimpleNamespace(cfg={'model': 'sole', 'protocol': 'official'}, predict=Mock())
        result = predict_condition(runtime, [sample], 'last_frame:target:8', {}, self.root, 'run')
        runtime.predict.assert_not_called()
        self.assertTrue(result[sample['example_id']]['baseline_fallback'])

    def test_geometry_failure_falls_back_before_starting_sole_rollout(self):
        sample = self.sample()
        base = self.baseline(sample)
        path = self.root / 'predictions/baseline.jsonl'
        path.parent.mkdir()
        path.write_text(json.dumps(base) + '\n')
        runtime = SimpleNamespace(cfg={'model': 'sole', 'protocol': 'official'}, predict=Mock(),
                    geometry_preflight=Mock(return_value={'eligible': False, 'reason': 'no_isolated_target_tokens',
                                                         'unusable_frames': [2]}))
        result = predict_condition(runtime, [sample], 'last_frame:target:8', {}, self.root, 'run')
        runtime.predict.assert_not_called()
        self.assertEqual(result[sample['example_id']]['fallback_reason'], 'no_isolated_target_tokens')

    def test_unavailable_control_has_own_status_and_never_falls_back(self):
        sample = self.sample()
        base = self.baseline(sample)
        path = self.root / 'predictions/baseline.jsonl'
        path.parent.mkdir()
        path.write_text(json.dumps(base) + '\n')
        runtime = SimpleNamespace(cfg={'model': 'meter', 'protocol': 'official'}, predict=Mock(),
                                  control_preflight=Mock(side_effect=ControlUnavailable('large target')))
        result = predict_condition(runtime, [sample], 'last_frame:wrong_region:8', {'last_frame': {}}, self.root, 'run')
        row = result[sample['example_id']]
        self.assertEqual(row['status'], 'control_unavailable')
        self.assertFalse(row['baseline_fallback'])
        runtime.predict.assert_not_called()

    def test_ranking_uses_scope_eligible_samples_without_zero_padding(self):
        samples = [self.sample('a'), self.sample('b'), self.sample('c', available=[0, 7])]
        runtime = SimpleNamespace(cfg={'model': 'meter', 'protocol': 'official', 'scopes': ['last_frame', 'all_frames'],
                                       'skip_early_layers': 0, 'top_k': [1]}, num_layers=2, num_heads=2,
                                  collect=Mock(return_value={'raw_mass': [[1., 2.], [3., 4.]],
                                                            'token_audit': {}, 'query_kind': 'last_prompt'}))
        rankings = rank(runtime, samples, {}, self.root, 'run')
        self.assertEqual(rankings['last_frame']['n'], 3)
        self.assertEqual(rankings['all_frames']['n'], 2)
        self.assertEqual(rankings['all_frames']['ranking'][0]['score'], 4.)
        self.assertEqual(rankings['all_frames']['example_ids'], ['a', 'b'])

    def test_cache_rejects_same_video_with_different_task(self):
        sample = self.sample()
        path = self.root / 'cache.jsonl'
        path.write_text(json.dumps(self.baseline(sample)) + '\n')
        with self.assertRaisesRegex(ValueError, 'Incompatible cache'):
            cache_rows(path, [{**sample, 'task': 'other task'}], 'run', 'baseline')

    def test_media_content_change_invalidates_reuse_at_the_same_path(self):
        path = self.root / 'frame.png'
        path.write_bytes(b'original pixels')
        create_json(self.root / 'input_artifacts.json', {str(path): file_hash(path)})
        cfg = {'model': 'meter', 'protocol': 'official', 'inputs': str(self.root / 'inputs.json')}
        samples = [{'image_paths': [str(path)]}]
        validate_media(cfg, samples)
        path.write_bytes(b'changed pixels')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            validate_media(cfg, samples)

    def test_full_scoring_retains_fallback_rows_and_fixed_denominator(self):
        samples = [self.sample('s'), self.sample('f', available=[0, 7])]
        create_json(self.root / 'inputs.json', samples)
        labels = {s['example_id']: {'reward': 5 if s['example_id'] == 's' else 1,
                                   'split': 'suc' if s['example_id'] == 's' else 'fail',
                                   'subset': 'task', 'source_suc_id': 's', 'video_sha256': 'same-video'} for s in samples}
        create_json(self.root / 'labels.json', labels)
        cfg = {'inputs': str(self.root / 'inputs.json'), 'labels': str(self.root / 'labels.json'),
               'output_dir': str(self.root), 'scopes': ['all_frames'], 'top_k': [8], 'controls': [],
               'model': 'meter', 'protocol': 'official'}
        identity = {'config': cfg, 'evaluation_ids': ['s', 'f']}
        create_json(self.root / 'run_config.json', cfg)
        create_json(self.root / 'run_identity.json', identity)
        base = [{**self.baseline(s), 'run_id': fingerprint(identity), 'progress': 1. if i == 0 else 0.}
                for i, s in enumerate(samples)]
        (self.root / 'predictions').mkdir()
        path = self.root / 'predictions/baseline.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in base))
        fallback = fallback_row(samples[1], 'all_frames:target:8', base[1], path,
                                {'reason': 'missing_exact_frames'}, fingerprint(identity))
        steered = {**base[0], 'condition': 'all_frames:target:8', 'sas_applied': True}
        (path.parent / 'all_frames_target_8.jsonl').write_text(json.dumps(steered) + '\n' + json.dumps(fallback) + '\n')
        with patch('mydata_bench.basic_method.score.validate_inputs'):
            result = score_experiment(cfg)
        full = result['conditions']['all_frames:target:8']['full']
        self.assertEqual((full['expected'], full['n'], full['baseline_fallback']), (2, 2, 1))
        self.assertEqual(full['mae'], 0.)
        self.assertEqual(full['accuracy']['0.125/0.875']['all']['rate_all_expected'], 1.)


if __name__ == '__main__':
    unittest.main()
