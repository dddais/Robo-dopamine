import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .head_output_contrast import compose, norm_error, HeadOutputController, HeadOutputRuntime
from .test_task_domain import mapping
from .verify_meter_port_discrete_replay import runtime


class HeadOutputTests(unittest.TestCase):
    def test_norm_zero_direction_and_degenerate_vectors(self):
        torch.manual_seed(3001)
        for dtype in [torch.float32, torch.bfloat16]:
            a, p, v, t = [torch.randn(3, 4, 16).to(dtype) for _ in range(4)]
            result = compose(a, p, v, t)
            self.assertLess(norm_error(a, result), .01)
            self.assertTrue(torch.equal(compose(a, p, p, p), a))
            self.assertTrue(torch.equal(compose(torch.zeros_like(a), p, v, t), torch.zeros_like(a)))
            self.assertTrue(torch.equal(compose(a, -a, torch.zeros_like(a), torch.zeros_like(a)), a))

    def test_orthogonal_basis_and_sample_permutation(self):
        torch.manual_seed(3011)
        parts = [torch.randn(3, 4, 8) for _ in range(4)]
        rotation, _ = torch.linalg.qr(torch.randn(8, 8))
        actual = compose(*parts)
        torch.testing.assert_close(compose(*[x @ rotation for x in parts]), actual @ rotation, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(compose(*[x[[2, 0, 1]] for x in parts]), actual[[2, 0, 1]]))

    def test_nonfinite_and_shape_rejected(self):
        a = torch.ones(2, 4)
        with self.assertRaises(ValueError): compose(a, a[:, :3], a, a)
        with self.assertRaises(ValueError): compose(a, a*float('nan'), a, a)

    def setup_controller(self):
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        cfg = dict(method='binding_transport', contrast_negative_mode='head_output',
                   task_binding_fraction=.5, task_binding_distribution='uniform', negative_task_strength=4.)
        controller = HeadOutputController([SimpleNamespace(self_attn=module)], cfg)
        return original, module, cfg, controller

    def test_controller_against_independent_probability_formula(self):
        torch.manual_seed(3021)
        original, module, cfg, controller = self.setup_controller()
        visual = torch.tensor([False, True, True, False, False, True, True, False])
        roi = torch.tensor([False, True, False, False, False, True, False, False])
        text = ~visual; task = torch.tensor([False, False, False, True, True, False, False, False])
        try:
            for qn in [8, 1]:
                q = torch.randn(2, 4, qn, 8); k = torch.randn(2, 2, 8, 8)
                v = torch.eye(8).expand(2, 2, 8, 8).clone()
                causal = torch.arange(8)[None, :] <= torch.arange(8-qn, 8)[:, None]
                for mask_type in ['none', 'boolean', 'infinity', 'finite_min']:
                    visible = causal.expand(2, 1, qn, 8).clone()
                    if mask_type != 'none': visible[0, :, :, 4] = False
                    if mask_type == 'none': mask = None
                    elif mask_type == 'boolean': mask = visible
                    else:
                        mask = torch.zeros_like(visible, dtype=torch.float32)
                        mask.masked_fill_(~visible, -torch.inf if mask_type == 'infinity' else torch.finfo(mask.dtype).min)
                    base, _ = original(module, q, k, v, mask)
                    scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2)/(8**.5)
                    scores.masked_fill_(~visible, -torch.inf)
                    probability = scores.softmax(-1)
                    def tilted(domain, target, strength):
                        part = probability*domain
                        z = torch.exp(strength*target)*part
                        return torch.where(domain, z*part.sum(-1, keepdim=True)/z.sum(-1, keepdim=True).clamp_min(1e-30), probability)
                    pos = tilted(visual, roi, 4)
                    vn = tilted(visual, roi, -4)
                    tn_text = tilted(text, task, -4)
                    tn = torch.where(text, tn_text, pos)
                    allowed = visible & task
                    count = allowed.sum(-1, keepdim=True)
                    pos = torch.where((count > 0) & text,
                        .5*pos + .5*(probability*text).sum(-1, keepdim=True)*allowed/count.clamp_min(1), pos)
                    direction = pos-.5*(vn+tn)
                    candidate = probability+direction
                    expected = candidate*torch.linalg.vector_norm(probability, dim=-1, keepdim=True)/torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
                    state = controller.steer([mapping(), mapping()], [dict(layer=0, head=1)], 4., 'all_frames')
                    actual, _ = controller.forward(module, q, k, v, mask)
                    torch.testing.assert_close(actual[:, :, 1], expected.transpose(1, 2)[:, :, 1], atol=3e-6, rtol=2e-5)
                    self.assertTrue(torch.equal(actual[:, :, [0, 2, 3]], base[:, :, [0, 2, 3]]))
                    self.assertLess(state['diagnostics']['0']['max_relative_norm_error'], .01)
                    self.assertIs(controller.cfg, cfg)
                    self.assertIs(controller.state, state)
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_masked_tokens_cannot_change_visible_outputs(self):
        torch.manual_seed(3031)
        original, module, cfg, controller = self.setup_controller()
        try:
            q = torch.randn(1, 4, 1, 8); k = torch.randn(1, 2, 8, 8); v = torch.randn_like(k)
            mask = torch.ones(1, 1, 1, 8, dtype=torch.bool); mask[..., 4] = False
            def run(keys, values):
                controller.steer([mapping()], [dict(layer=0, head=1)], 4., 'all_frames')
                return controller.forward(module, q, keys, values, mask)[0]
            first = run(k, v)
            altered_k = k.clone(); altered_v = v.clone(); altered_k[:, :, 4] *= 100; altered_v[:, :, 4] *= 100
            self.assertTrue(torch.equal(first, run(altered_k, altered_v)))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_exception_restores_outer_state_and_configuration(self):
        original, module, cfg, controller = self.setup_controller()
        try:
            q = torch.ones(1, 4, 8, 8); k = torch.ones(1, 2, 8, 8)
            state = controller.steer([mapping()], [dict(layer=0, head=1)], 4., 'all_frames')
            with patch('mydata_bench.auto_research_addbase.attention.ResearchController.forward', side_effect=RuntimeError('probe')):
                with self.assertRaises(RuntimeError): controller.forward(module, q, k, k, None)
            self.assertIs(controller.cfg, cfg); self.assertIs(controller.state, state)
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_runtime_has_single_forward_and_all_five_classes(self):
        for model in ['qwen', 'roboreward']:
            r, s = runtime(HeadOutputRuntime, model, 'head_output', 0, False)
            def forward(**kwargs):
                s['calls'].append(s['branch'])
                return SimpleNamespace(logits=(10*torch.eye(5))[:, None, :])
            r.model = forward
            rows = r.predict([dict(example_id=str(i)) for i in range(5)], 'all_frames:target:1',
                             {'all_frames': {'ranking': [dict(layer=0, head=0)]}})
            self.assertEqual(s['calls'], [('binding_transport', 4)])
            self.assertEqual({x['reward'] for x in rows}, {1, 2, 3, 4, 5})
            for row in rows:
                self.assertEqual(row['actual_forward_branches'], 1)
                self.assertEqual(row['contrast_weight'], 0)
                self.assertIsNone(row['native_class_logits_negative'])

    def test_runtime_refuses_logit_contrast(self):
        r, s = runtime(HeadOutputRuntime, 'qwen', 'head_output', 1, False)
        with self.assertRaises(ValueError): r.predict([dict(example_id='a')], 'baseline')
        self.assertEqual(s['calls'], [])


if __name__ == '__main__': unittest.main()
