"""Behavioral checks for post-SAS capture, pass-through inference and paired scales."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from mydata_bench.addbase_eval.attention import AttentionController
from mydata_bench.basic_method.common import fingerprint
from mydata_bench.basic_method.visualization.capture import LastPromptCapture, attention_row
from mydata_bench.basic_method.visualization.__main__ import (
    choose_heads, parser, prediction_match, select_samples, visualize_sample,
)
from mydata_bench.basic_method.visualization.render import paired_maps, vector_to_grid


class AttentionCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = SimpleNamespace(is_causal=True, num_key_value_groups=2)
        previous = ALL_ATTENTION_FUNCTIONS['sdpa']
        self.addCleanup(ALL_ATTENTION_FUNCTIONS.register, 'sdpa', previous)
        self.controller = AttentionController([SimpleNamespace(self_attn=self.module)])
        self.q = torch.randn(1, 4, 6, 8)
        self.k = torch.randn(1, 2, 6, 8)
        self.v = torch.randn_like(self.k)
        self.heads = [{'layer': 0, 'head': 3}, {'layer': 0, 'head': 0}]

    def forward(self, mask=None):
        return self.controller.forward(self.module, self.q, self.k, self.v, mask)[0]

    def test_capture_preserves_baseline_and_actual_steered_outputs(self):
        plain = self.forward()
        original = self.controller.original
        with LastPromptCapture(self.controller, self.heads, 6) as base:
            captured = self.forward()
        self.assertTrue(torch.equal(plain, captured))
        self.assertIs(self.controller.original, original)
        mapping = {'visual': [1, 2, 3, 4], 'target': {'last_frame': [3]},
                   'negative': {'last_frame': [4]}}
        self.controller.steer([mapping], [self.heads[0]], 6., 'last_frame')
        steered = self.forward()
        with LastPromptCapture(self.controller, self.heads, 6) as sas:
            steered_captured = self.forward()
        self.assertTrue(torch.equal(steered, steered_captured))
        self.assertFalse(torch.equal(plain, steered))
        a, b = base.result(), sas.result()
        np.testing.assert_allclose(a.sum(-1), 1., atol=1e-6)
        np.testing.assert_allclose(b[1], a[1], atol=1e-7)  # Unsteered head is unchanged at this layer.
        self.assertGreater(b[0, 3], a[0, 3])
        self.assertLess(b[0, 4], a[0, 4])
        expected_logits = self.q[0, 3, -1].float() @ self.k[0, 1].float().T / 8 ** .5
        expected_logits[3] += 6
        expected_logits[4] -= 6
        np.testing.assert_allclose(b[0], expected_logits.softmax(-1).numpy(), rtol=1e-6)

    def test_boolean_and_additive_causal_masks_match_explicit_gqa(self):
        allowed = torch.ones(6, 6, dtype=torch.bool).tril()[None, None]
        additive = torch.zeros(1, 4, 6, 6).masked_fill(~allowed, -torch.inf)
        for mask in (None, allowed, additive):
            actual = attention_row(self.module, self.q, self.k, mask, [3, 0], 2, scaling=.25)
            expected = torch.stack([
                (self.k[0, head // 2] @ self.q[0, head, 2]) * .25 for head in [3, 0]])
            expected[:, 3:] = -torch.inf
            np.testing.assert_allclose(actual, expected.softmax(-1).numpy(), atol=1e-7)
            np.testing.assert_array_equal(actual[:, 3:], 0)

    def test_wrapper_restored_on_error_and_incomplete_capture_is_rejected(self):
        original = self.controller.original
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with LastPromptCapture(self.controller, self.heads, 6):
                raise RuntimeError('test failure')
        self.assertIs(self.controller.original, original)
        with LastPromptCapture(self.controller, self.heads, 6) as empty:
            pass
        with self.assertRaisesRegex(RuntimeError, 'Missing captured'):
            empty.result()


class VisualizationTests(unittest.TestCase):
    def test_raw_scale_is_shared_without_independent_minmax(self):
        a = np.array([[.001, .002], [.003, .004]])
        b = a * 20
        x, y, vmax = paired_maps(a, b, 'raw')
        np.testing.assert_array_equal(x, a)
        np.testing.assert_array_equal(y, b)
        self.assertEqual(vmax, .08)
        x, y, _ = paired_maps(a, b, 'image_fraction')
        np.testing.assert_allclose(x, y)
        np.testing.assert_array_equal(paired_maps(a * 0, a * 0, 'image_fraction')[0], 0)
        np.testing.assert_array_equal(vector_to_grid(np.arange(6), (1, 4, 6), 2), np.arange(6).reshape(2, 3))

    def test_explicit_ids_and_observed_heads_do_not_change_steering(self):
        samples = [{'example_id': f'suc/{i}', 'subset': 't', 'holdout': i > 0} for i in range(8)]
        args = parser().parse_args(['--example-id', 'suc/6', '--example-id', 'suc/2'])
        self.assertEqual([s['example_id'] for s in select_samples(samples, args)], ['suc/6', 'suc/2'])
        args.example_id = ['absent']
        with self.assertRaisesRegex(ValueError, 'Unknown example'):
            select_samples(samples, args)
        ranking = {'num_layers': 4, 'num_heads': 4, 'ranking': [{'layer': 1, 'head': 2}, {'layer': 2, 'head': 1}]}
        steer, observe = choose_heads(ranking, 1, 'L3H0')
        self.assertEqual(steer, ranking['ranking'][:1])
        self.assertEqual(observe, [{'layer': 3, 'head': 0}])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            choose_heads(ranking, 1, 'L3H0,L3H0')

    def test_saved_prediction_input_drift_fails_and_score_drift_is_visible(self):
        sample = {'example_id': 'suc/1'}
        row = {'status': 'ok', 'raw_output': '<score>50%</score>', 'progress': .5,
               'token_audit': {'prompt_sha256': 'p', 'input_ids_sha256': 'i'}}
        saved = {**row, 'run_id': 'r', 'condition': 'baseline', 'sample_id': fingerprint(sample)}
        self.assertTrue(prediction_match(row, saved, sample, 'r', 'baseline')['matches'])
        self.assertFalse(prediction_match({**row, 'progress': .7}, saved, sample, 'r', 'baseline')['matches'])
        with self.assertRaisesRegex(ValueError, 'input differs'):
            prediction_match({**row, 'token_audit': {**row['token_audit'], 'input_ids_sha256': 'changed'}},
                             saved, sample, 'r', 'baseline')

    def test_missing_grounding_reuses_baseline_and_writes_real_figure_and_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'source.png'
            Image.new('RGB', (64, 48), (70, 90, 110)).save(image)
            sample = {'example_id': 'fail/test/1', 'task': 'Pick the cup.'}
            audit = {'sequence_length': 6, 'query': 5, 'prompt_sha256': 'p', 'input_ids_sha256': 'i'}
            slots = {'after_cam_high': {'path': str(image), 'source_frame': 7, 'start': 1, 'end': 5,
                     'grid_thw': [1, 4, 4], 'bbox': None, 'size': [64, 48]}}
            check = {'eligible': False, 'reason': 'missing_exact_frames'}
            runtime = SimpleNamespace(cfg={'bias': 6.}, controller=Mock(), audit=lambda m: m,
                                      predict=Mock(return_value={'progress': .5, 'token_audit': audit}))
            fake_capture = Mock()
            fake_capture.__enter__ = Mock(return_value=fake_capture)
            fake_capture.__exit__ = Mock(return_value=None)
            fake_capture.result.return_value = np.array([[.2, .1, .2, .3, .1, .1]])
            args = parser().parse_args(['--per-head', '0'])
            with patch('mydata_bench.basic_method.visualization.__main__.context', return_value=(check, audit, slots)), \
                    patch('mydata_bench.basic_method.visualization.capture.LastPromptCapture', return_value=fake_capture):
                result = visualize_sample(runtime, sample, args, {}, [{'layer': 0, 'head': 0}],
                                          {'baseline': {}, 'sas': {}}, 'r', root)
            runtime.predict.assert_called_once_with(sample, 'baseline', {})
            meta = json.loads((root / result['metadata']).read_text())
            self.assertTrue(meta['baseline_fallback'])
            self.assertFalse(meta['sas_applied'])
            self.assertEqual(meta['positive_bias'], 0.)
            arrays = np.load((root / result['metadata']).parent / 'attention.npz')
            np.testing.assert_array_equal(arrays['baseline_rows'], arrays['sas_rows'])
            with Image.open(root / result['figures'][0]) as figure:
                self.assertGreater(figure.width, 1000)


if __name__ == '__main__':
    unittest.main()
