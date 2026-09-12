import copy
import unittest
from types import SimpleNamespace

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .task_content_block import TaskContentController, TaskContentRuntime, block_mask, compose, perturb_task_tokens


class TaskContentTests(unittest.TestCase):
    def test_mask_and_gqa_full_prefill_and_decode(self):
        torch.manual_seed(2801)
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        modules = [SimpleNamespace(num_key_value_groups=2, is_causal=True) for _ in range(3)]
        controller = TaskContentController([SimpleNamespace(self_attn=m) for m in modules], dict(method='bias'))
        maps = [dict(task_positions=[1, 3]), dict(task_positions=[2, 5])]
        try:
            for qn in [7, 1]:
                q = torch.randn(2, 4, qn, 8); k = torch.randn(2, 2, 7, 8); v = torch.randn_like(k)
                allowed = (torch.arange(7)[None, :] <= torch.arange(7-qn, 7)[:, None]).expand(2, 4, qn, 7).clone()
                allowed[0, 2, :, 0] = False
                for mask in [None, allowed, torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, -torch.inf)]:
                    state = controller.block_task(maps)
                    for module in modules:
                        out, _ = controller.forward(module, q, k, v, mask)
                        visible = (torch.ones_like(allowed) if qn == 1 else torch.ones_like(allowed).tril()) if mask is None else allowed.clone()
                        for i, m in enumerate(maps):
                            visible[i, :, :, m['task_positions']] = False
                        scores = q @ k.repeat_interleave(2, 1).transpose(-1, -2)/(8**.5)
                        weights = torch.nan_to_num(scores.masked_fill(~visible, -torch.inf).softmax(-1), nan=0.)
                        expected = (weights @ v.repeat_interleave(2, 1)).transpose(1, 2)
                        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
                    self.assertEqual(set(state['diagnostics']), {'0', '1', '2'})
                    self.assertTrue(all(d['heads'] == [0, 1, 2, 3] for d in state['diagnostics'].values()))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_padding_no_visible_keys_and_empty_task(self):
        q = torch.randn(2, 4, 3, 8); k = torch.randn(2, 2, 3, 8)
        allowed = torch.ones(2, 1, 3, 3, dtype=torch.bool).tril()
        allowed[0] = False
        original = torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, torch.finfo(q.dtype).min)
        saved = original.clone()
        mask = block_mask(q, k, original, [[], [0, 1, 2]], False)
        self.assertTrue(torch.isneginf(mask).all()); self.assertTrue(torch.equal(saved, original))
        out = torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), k.repeat_interleave(2, 1), mask)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        empty = block_mask(q, k, allowed, [[], []], False)
        self.assertTrue(torch.equal(torch.isfinite(empty), allowed))

    def test_additive_offsets_and_explicit_causal_preserved(self):
        q = torch.randn(1, 2, 4, 8); k = torch.randn(1, 1, 4, 8)
        mask = torch.randn(1, 2, 4, 4)
        result = block_mask(q, k, mask, [[1]], True)
        visible = torch.ones_like(mask, dtype=torch.bool).tril(); visible[..., 1] = False
        self.assertTrue(torch.equal(result[visible], mask[visible]))
        self.assertTrue(torch.isneginf(result[~visible]).all())

    def test_invalid_mask_positions_rejected(self):
        q = torch.randn(2, 4, 3, 8); k = torch.randn(2, 2, 3, 8)
        for positions in [[[0]], [[0, 0], []], [[-1], []], [[3], []]]:
            with self.assertRaises(ValueError): block_mask(q, k, None, positions, True)
        for mask in [torch.zeros(2, 3), torch.full((2, 1, 3, 3), float('nan')), torch.full((2, 1, 3, 3), float('inf'))]:
            with self.assertRaises(ValueError): block_mask(q, k, mask, [[], []], True)

    def test_multilayer_isolation_and_batch_independence(self):
        torch.manual_seed(2805)
        x = torch.randn(2, 7, 12); changed = x.clone(); positions = [[1, 3], [2, 5]]
        for i, ids in enumerate(positions): changed[i, ids] += torch.randn_like(changed[i, ids])*8
        weights = [torch.randn(12, 12) for _ in range(4)]
        def network(a, ps):
            for w in weights:
                q = (a @ w).view(a.shape[0], 7, 3, 4).transpose(1, 2)
                mask = block_mask(q, q, None, ps, True)
                attention = torch.nn.functional.scaled_dot_product_attention(q, q, q, mask).transpose(1, 2).reshape_as(a)
                a = a + attention
                a = a + torch.tanh(a @ w)
            return a
        actual = network(x, positions); other = network(changed, positions)
        for i, ids in enumerate(positions):
            non = [j for j in range(7) if j not in ids]
            self.assertTrue(torch.equal(actual[i, non], other[i, non]))
        torch.testing.assert_close(actual, torch.cat([network(x[i:i+1], [positions[i]]) for i in range(2)]))

    def test_unselected_vision_and_cleared_state_bypass(self):
        original = ALL_ATTENTION_FUNCTIONS['sdpa']; module = SimpleNamespace(is_causal=True)
        c = TaskContentController([SimpleNamespace(self_attn=module)], dict(method='bias'))
        calls = []
        def stub(*a, **k): calls.append((a, k)); return 'unchanged'
        c.original = stub
        try:
            c.block_task([dict(task_positions=[1])])
            self.assertEqual(c.forward(object(), None, None, None, None), 'unchanged')
            c.clear()
            self.assertEqual(c.forward(module, None, None, None, None), 'unchanged')
            self.assertEqual(len(calls), 2)
        finally: ALL_ATTENTION_FUNCTIONS.register('sdpa', original)

    def test_five_classes_symmetry_ratio_and_no_id_inputs(self):
        torch.manual_seed(2807)
        p = torch.randn(8, 5, dtype=torch.float64); n = torch.randn_like(p)
        z = compose(p, n); ratio = p.softmax(-1)**2/n.softmax(-1); ratio /= ratio.sum(-1, keepdim=True)
        torch.testing.assert_close(z.softmax(-1), ratio, rtol=1e-13, atol=1e-13)
        perm = [3, 1, 4, 0, 2]
        self.assertTrue(torch.equal(compose(p[:, perm], n[:, perm]), z[:, perm]))
        self.assertTrue(torch.equal(compose(p, p), p))
        self.assertTrue(torch.equal(compose(torch.eye(5), torch.zeros(5, 5)).argmax(-1), torch.arange(5)))

    def test_probe_only_changes_instruction_contents(self):
        tok = SimpleNamespace(encode=lambda s, **k: [11 if s == 'x' else 12], all_special_ids=[0, 99])
        inputs = dict(input_ids=torch.tensor([[99, 11, 5, 7, 9], [99, 4, 12, 7, 9]]), attention_mask=torch.ones(2, 5))
        maps = [dict(task_positions=[1, 2], visual=[3], query=4)]*2
        original = copy.deepcopy(inputs)
        changed, audit = perturb_task_tokens(inputs, maps, tok)
        self.assertTrue(torch.equal(inputs['input_ids'], original['input_ids']))
        self.assertIs(changed['attention_mask'], inputs['attention_mask'])
        self.assertTrue(torch.equal(changed['input_ids'][:, [0, 3, 4]], inputs['input_ids'][:, [0, 3, 4]]))
        self.assertTrue(torch.all(changed['input_ids'][:, [1, 2]] != inputs['input_ids'][:, [1, 2]]))
        self.assertEqual(audit['changed_token_counts'], [2, 2])

    def fake(self, fail=False):
        r = object.__new__(TaskContentRuntime)
        r.cfg = dict(model='qwen', contrast_negative_mode='task_content_block', contrast_weight=1,
                     require_task_positions=True, method='bias', bias=6)
        maps = [dict(task_positions=[1], query=3, task_span_audit=dict(literal_task_matches=1))]
        r.prepare = lambda *a: ({}, maps, [3], ['prompt'])
        r.audit = lambda m: m
        r.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s, **k: [int(s)]))
        r.layers = [object()]*2
        state = dict(kind=None, calls=[], clears=0)
        def steer(*a): state['kind'] = 'positive'; return dict(diagnostics={'0': dict(heads=[0])})
        def block(*a): state['kind'] = 'blocked'; return dict(diagnostics={'0': {}, '1': {}})
        def clear(): state['kind'] = None; state['clears'] += 1
        r.controller = SimpleNamespace(steer=steer, block_task=block, clear=clear)
        def model(**kwargs):
            state['calls'].append(state['kind'])
            if fail and state['kind'] == 'blocked': raise RuntimeError('synthetic blocked failure')
            gain = 2 if state['kind'] == 'positive' else 1
            return SimpleNamespace(logits=torch.arange(6.).view(1, 1, 6)*gain)
        r.model = model
        return r, state

    def test_runtime_two_real_branches_baseline_and_id_independence(self):
        r, state = self.fake(); rank = {'all_frames': dict(ranking=[dict(layer=0, head=0)])}
        a = r.predict([dict(example_id='a')], 'baseline')[0]
        b = r.predict([dict(example_id='a')], 'all_frames:target:1', rank)[0]
        c = r.predict([dict(example_id='different')], 'all_frames:target:1', rank)[0]
        self.assertEqual(state['calls'], [None, 'positive', 'blocked', 'positive', 'blocked'])
        self.assertEqual(state['clears'], 5); self.assertIsNone(state['kind'])
        self.assertEqual(a['actual_forward_branches'], 1); self.assertEqual(b['actual_forward_branches'], 2)
        self.assertEqual(b['native_class_probabilities'], c['native_class_probabilities'])

    def test_exception_clears_controller_and_preserves_config(self):
        r, state = self.fake(True); cfg = copy.deepcopy(r.cfg)
        with self.assertRaises(RuntimeError):
            r.predict([dict(example_id='a')], 'all_frames:target:1', {'all_frames': dict(ranking=[dict(layer=0, head=0)])})
        self.assertIsNone(state['kind']); self.assertEqual(state['clears'], 2); self.assertEqual(cfg, r.cfg)


if __name__ == '__main__': unittest.main()
