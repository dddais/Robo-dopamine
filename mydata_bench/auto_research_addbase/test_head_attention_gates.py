from types import SimpleNamespace
import unittest

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .head_attention_gates import HeadGateController, gated_probability
from .test_task_domain import mapping


class HeadGateTests(unittest.TestCase):
    def test_head_specific_doses_permutation_and_tied_layer_reduction(self):
        from .learned_attention_gates import gated_probability as layer_probability
        torch.manual_seed(3201)
        scores = torch.randn(2, 4, 3, 8)
        sign = torch.tensor([0., 1., -1., 0., 0., 1., -1., 0.])
        task = torch.tensor([0., 0., 0., 1., 1., 0., 0., 0.])
        gates = torch.rand(4, 2)
        actual = gated_probability(scores, sign, task, gates)
        for h in range(4):
            expected = layer_probability(scores[:, h], sign, task, gates[h])
            self.assertTrue(torch.equal(actual[:, h], expected))
        order = [3, 0, 2, 1]
        self.assertTrue(torch.equal(gated_probability(scores[:, order], sign, task, gates[order]), actual[:, order]))
        tied = torch.tensor([.2, .8])
        self.assertTrue(torch.equal(gated_probability(scores, sign, task, tied.expand(4, 2)),
                                    layer_probability(scores, sign, task, tied)))

    def test_worker_hash_and_policy_integrity(self):
        import hashlib
        from .head_gate_worker import POLICY, policy, sha
        self.assertEqual(sha(POLICY), hashlib.sha256(POLICY.read_bytes()).hexdigest())
        self.assertEqual(policy()['training']['trainable_parameters'], 1792)

    def test_double_precision_gradient_check(self):
        torch.manual_seed(3101)
        scores = torch.randn(2, 3, 2, 8, dtype=torch.float64, requires_grad=True)
        spatial = torch.tensor([0, 1, -1, 0, 0, 1, -1, 0], dtype=torch.float64)
        task = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0], dtype=torch.float64)
        theta = torch.full((3, 2), -2., dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda s, t: gated_probability(s, spatial, task, t.sigmoid()),
                                                (scores, theta), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_probabilities_masks_and_zero_gate(self):
        scores = torch.tensor([[[[1., -torch.inf, 2., 4.]]]])
        spatial = torch.tensor([0., 1., -1., 0.]); task = torch.tensor([0., 0., 0., 1.])
        self.assertTrue(torch.equal(gated_probability(scores, spatial, task, torch.zeros(1, 2)), scores.softmax(-1)))
        p = gated_probability(scores, spatial, task, torch.ones(1, 2))
        self.assertEqual(float(p[0, 0, 0, 1]), 0)
        self.assertAlmostEqual(float(p.sum()), 1, places=6)
        empty = gated_probability(scores*0-float('inf'), spatial, task, torch.ones(1, 2))
        self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))

    def setup_controller(self):
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        modules = [SimpleNamespace(num_key_value_groups=2, is_causal=True) for _ in range(9)]
        cfg = dict(method='learned_head_gate_bias', contrast_negative_mode='learned_head_gates')
        c = HeadGateController([SimpleNamespace(self_attn=m) for m in modules], cfg)
        c.theta = torch.nn.Parameter(torch.full((28, 32, 2), -2.))
        return original, modules[8], c

    def test_actual_controller_reference_gqa_prefill_decode(self):
        torch.manual_seed(3111)
        original, module, c = self.setup_controller()
        try:
            for qn in [8, 1]:
                q = torch.randn(2, 4, qn, 8); k = torch.randn(2, 2, 8, 8); v = torch.randn_like(k)
                visible = (torch.arange(8)[None, :] <= torch.arange(8-qn, 8)[:, None]).expand(2, 1, qn, 8).clone()
                visible[0, :, :, 4] = False
                for kind in ['boolean', 'infinity', 'finite_min']:
                    mask = visible if kind == 'boolean' else torch.zeros_like(visible, dtype=torch.float32).masked_fill(
                        ~visible, -torch.inf if kind == 'infinity' else torch.finfo(torch.float32).min)
                    base, _ = original(module, q, k, v, mask)
                    c.steer([mapping(), mapping()], [dict(layer=8, head=1)], 6., 'all_frames')
                    actual, _ = c.forward(module, q, k, v, mask)
                    sign = torch.tensor([0., 1., -1., 0., 0., 1., -1., 0.])
                    task = torch.tensor([0., 0., 0., 1., 1., 0., 0., 0.])
                    score = (q[:, 1:2] @ k[:, 0:1].transpose(-1, -2))/(8**.5)
                    score = score.masked_fill(~visible, -torch.inf)
                    g = float(torch.sigmoid(torch.tensor(-2.)))
                    expected = (score+6*g*sign+4*g*task).softmax(-1) @ v[:, 0:1]
                    torch.testing.assert_close(actual[:, :, 1:2], expected.transpose(1, 2), rtol=2e-5, atol=2e-6)
                    self.assertTrue(torch.equal(actual[:, :, [0, 2, 3]], base[:, :, [0, 2, 3]]))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_zero_gate_exact_baseline_and_masked_key_invariance(self):
        original, module, c = self.setup_controller()
        try:
            q = torch.randn(1, 4, 8, 8); k = torch.randn(1, 2, 8, 8); v = torch.randn_like(k)
            c.steer([mapping()], [dict(layer=8, head=1)], 6., 'all_frames')
            c.zero_probe = True
            actual, _ = c.forward(module, q, k, v, None)
            base, _ = original(module, q, k, v, None)
            self.assertTrue(torch.equal(actual, base))
            c.zero_probe = False
            mask = (torch.arange(8)[None, :] <= torch.arange(8)[:, None])[None, None]
            mask[..., 4] = False
            def run(keys, values):
                c.steer([mapping()], [dict(layer=8, head=1)], 6., 'all_frames')
                return c.forward(module, q, keys, values, mask)[0]
            a = run(k, v); changed_k = k.clone(); changed_v = v.clone()
            changed_k[:, :, 4] *= 1000; changed_v[:, :, 4] *= 1000
            self.assertTrue(torch.equal(a, run(changed_k, changed_v)))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_controller_gradient_finite_difference_and_inactive_layers(self):
        torch.manual_seed(3121)
        original, module, c = self.setup_controller()
        try:
            q = torch.randn(1, 4, 8, 8); k = torch.randn(1, 2, 8, 8); v = torch.randn_like(k)
            def loss():
                c.steer([mapping()], [dict(layer=8, head=1)], 6., 'all_frames')
                a, _ = c.forward(module, q, k, v, None)
                return a.square().mean()
            loss().backward()
            gradient = c.theta.grad.clone()
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient[0, 1].abs().sum()), 0)
            self.assertTrue(torch.equal(gradient[0, [0]+list(range(2,32))], torch.zeros_like(gradient[0, [0]+list(range(2,32))])))
            self.assertTrue(torch.equal(gradient[1:], torch.zeros_like(gradient[1:])))
            for j in range(2):
                with torch.no_grad(): c.theta[0, 1, j] += .001
                plus = float(loss().detach())
                with torch.no_grad(): c.theta[0, 1, j] -= .002
                minus = float(loss().detach())
                with torch.no_grad(): c.theta[0, 1, j] += .001
                self.assertAlmostEqual((plus-minus)/.002, float(gradient[0, 1, j]), delta=2e-4)
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_only_1792_parameters_update_with_toy_frozen_backbone(self):
        torch.manual_seed(3131)
        original, module, c = self.setup_controller()
        try:
            backbone = torch.nn.Linear(8, 8).requires_grad_(False)
            before = {k: v.clone() for k, v in backbone.state_dict().items()}
            old_theta = c.theta.detach().clone()
            optimizer = torch.optim.Adam([c.theta], lr=.05)
            self.assertEqual(sum(p.numel() for group in optimizer.param_groups for p in group['params']), 1792)
            q = backbone(torch.randn(1, 4, 8, 8)); k = backbone(torch.randn(1, 2, 8, 8)); v = torch.randn_like(k)
            c.steer([mapping()], [dict(layer=8, head=1)], 6., 'all_frames')
            a, _ = c.forward(module, q, k, v, None)
            a.square().mean().backward(); optimizer.step()
            self.assertFalse(torch.equal(old_theta, c.theta))
            self.assertTrue(all(torch.equal(before[k], v) for k, v in backbone.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_shared_parameters_across_scope_and_explicit_values(self):
        original, module, c = self.setup_controller()
        try:
            self.assertEqual(c.theta.shape, (28, 32, 2))
            c.theta = None; c.fixed_values = [[[.2, .8]]*32]*28
            self.assertEqual(tuple(c.values('cpu').shape), (28, 32, 2))
            c.fixed_values = [[[.2, 1.8]]*32]*28
            with self.assertRaises(ValueError): c.values('cpu')
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_global_class_agnostic_gate_batch_permutation(self):
        torch.manual_seed(3141)
        scores = torch.randn(3, 4, 2, 8)
        sign = torch.tensor([0, 1, -1, 0, 0, 1, -1, 0]); task = (sign == 0).float()
        g = torch.tensor([[.3, .7]]*4)
        expected = gated_probability(scores, sign, task, g)
        self.assertTrue(torch.equal(gated_probability(scores[[2, 0, 1]], sign, task, g), expected[[2, 0, 1]]))

    def test_frozen_training_schedule_has_complete_balanced_layout_coverage(self):
        from .head_gate_worker import schedule, policy, PROTOCOLS, SCOPES
        from collections import Counter
        spec = policy()
        for epoch in range(3):
            entries = schedule(70, epoch)
            self.assertEqual(len(entries), 1400)
            self.assertEqual(len(set(entries)), 1400)
            self.assertEqual(entries, schedule(70, epoch))
            self.assertEqual(set(Counter(i for i, p, s, k in entries).values()), {20})
            self.assertEqual(set((p, s, k) for i, p, s, k in entries), {(p, s, k) for p in PROTOCOLS for s in SCOPES for k in [32, 64]})
        self.assertNotEqual(schedule(70, 0), schedule(70, 1))
        self.assertEqual(3*1400//8, spec['training']['total_optimizer_steps'])

    def test_gpu_worker_import_preserves_explicit_device_visibility(self):
        import os
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-B', '-c',
            "import os; from mydata_bench.auto_research_addbase import head_gate_worker; "
            "assert os.environ['CUDA_VISIBLE_DEVICES']=='1'; import torch; assert not torch.cuda.is_initialized()"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inference_runtime_keeps_all_classes_and_never_contrasts_logits(self):
        from .head_attention_gates import HeadGateRuntime
        from .verify_meter_port_discrete_replay import runtime
        for model in ['qwen', 'roboreward']:
            r, state = runtime(HeadGateRuntime, model, 'learned_head_gates', 0, False)
            def forward(**kwargs):
                state['calls'].append(state['branch'])
                return SimpleNamespace(logits=(10*torch.eye(5))[:, None, :])
            forward.parameters = lambda: []
            r.model = forward
            rows = r.predict([dict(example_id=str(i)) for i in range(5)], 'all_frames:target:1',
                {'all_frames': {'ranking': [dict(layer=0, head=0)]}})
            self.assertEqual(len(state['calls']), 1)
            self.assertEqual({row['reward'] for row in rows}, {1, 2, 3, 4, 5})
            self.assertTrue(all(row['contrast_weight'] == 0 and row['native_class_logits_negative'] is None for row in rows))


if __name__ == '__main__': unittest.main()
