import unittest
from types import SimpleNamespace

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController, uniform_redistribute, redistribute


class UniformBindingTests(unittest.TestCase):
    def test_visible_underflow_and_causal_invisibility_are_distinct(self):
        p = torch.tensor([[.2, .3, .5, 0., 0.], [.2, .3, .5, 0., 0.]], dtype=torch.float64)
        domain = torch.tensor([0, 1, 1, 1, 1], dtype=torch.bool)
        target = torch.tensor([0, 0, 0, 1, 1], dtype=torch.bool)
        visible = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
        result = uniform_redistribute(p, domain, target, visible, .5)
        expected = torch.tensor([[.2, .15, .25, .4, 0.], [.2, .3, .5, 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=0, atol=1e-15)
        self.assertTrue(torch.equal(result[:, ~domain], p[:, ~domain]))

    def test_mass_and_minimum_instruction_share(self):
        torch.manual_seed(22)
        p = torch.randn(2, 3, 5, 7, dtype=torch.float64).softmax(-1)
        domain = torch.tensor([0, 1, 1, 0, 1, 1, 0], dtype=torch.bool)
        target = torch.tensor([0, 1, 0, 0, 1, 0, 0], dtype=torch.bool)
        visible = torch.ones_like(p, dtype=torch.bool)
        for fraction in [0., .5, 1.]:
            result = uniform_redistribute(p, domain, target, visible, fraction)
            torch.testing.assert_close(result.sum(-1), p.sum(-1), rtol=0, atol=1e-15)
            torch.testing.assert_close(result[..., domain].sum(-1), p[..., domain].sum(-1), rtol=0, atol=1e-15)
            self.assertTrue(torch.equal(result[..., ~domain], p[..., ~domain]))
            minimum = fraction * p[..., domain].sum(-1, keepdim=True) / 2
            self.assertTrue(torch.all(result[..., target] >= minimum-1e-15))
            torch.testing.assert_close(result[..., domain & ~target],
                                       (1-fraction)*p[..., domain & ~target], rtol=0, atol=1e-15)

    def test_controller_matches_direct_attention_for_masks_and_decode(self):
        torch.manual_seed(82)
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        cfg = {'method':'binding_transport', 'task_binding_fraction':.5,
               'task_binding_distribution':'uniform'}
        controller = ResearchController([SimpleNamespace(self_attn=module)], cfg)
        mapping = {'visual':[1, 2], 'target':{'all_frames':[2]}, 'negative':{'all_frames':[1]},
                   'prompt_text_positions':[0, 3, 4, 5], 'task_positions':[3, 4], 'query':5}
        try:
            for qn in [6, 1]:
                q = torch.randn(2, 4, qn, 8)
                k = torch.randn(2, 2, 6, 8)
                v = torch.randn_like(k)
                causal = torch.arange(6)[None, :] <= torch.arange(6-qn, 6)[:, None]
                for kind in ['none', 'bool', 'infinity', 'finite_min']:
                    allowed = causal.expand(2, 1, qn, 6).clone()
                    if kind != 'none':allowed[0, :, :, 4] = False
                    if kind == 'none':mask = None
                    elif kind == 'bool':mask = allowed
                    else:
                        mask = torch.zeros(2, 1, qn, 6)
                        mask.masked_fill_(~allowed, -torch.inf if kind=='infinity' else torch.finfo(mask.dtype).min)
                    controller.steer([mapping, mapping], [{'layer':0, 'head':1}], 4., 'all_frames')
                    actual, _ = controller.forward(module, q, k, v, mask)
                    scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2) / (8**.5)
                    scores.masked_fill_(~allowed, -torch.inf)
                    probability = torch.nan_to_num(scores.softmax(-1), nan=0.)
                    domain = torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.bool)
                    roi = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.bool)
                    text = ~domain
                    task = torch.tensor([0, 0, 0, 1, 1, 0], dtype=torch.bool)
                    selected = redistribute(probability[:, 1:2], domain, roi, 4.)
                    # Independent probability-space implementation, including
                    # visible target counts and no-op before the instruction.
                    target = (allowed & task).to(selected.dtype)
                    count = target.sum(-1, keepdim=True)
                    text_part = selected*text
                    injected = target*text_part.sum(-1, keepdim=True)/count.clamp_min(1)
                    selected += torch.where(count>0, .5*(injected-text_part), torch.zeros_like(selected))
                    probability[:, 1:2] = selected
                    expected = (probability @ v.repeat_interleave(2, 1)).transpose(1, 2)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    baseline, _ = original(module, q, k, v, mask)
                    self.assertTrue(torch.equal(actual[:, :, [0, 2, 3]], baseline[:, :, [0, 2, 3]]))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)


if __name__ == '__main__':
    unittest.main()
