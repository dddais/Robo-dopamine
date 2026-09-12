import unittest
from types import SimpleNamespace

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController
from .interaction import CELLS, InteractionRuntime, combine_interaction


class InteractionTests(unittest.TestCase):
    def test_additive_effect_cancels_and_interaction_sign(self):
        b = torch.arange(10, dtype=torch.float64).reshape(2, 5)
        v = torch.tensor([1., 0., -2., 3., 1.], dtype=torch.float64)
        t = v.flip(0); w = torch.tensor([.5, -1., 2., 0., 1.], dtype=torch.float64)
        cells = dict(pp=b+v+t, mp=b-v+t, pm=b+v-t, mm=b-v-t)
        z, effect = combine_interaction(cells)
        self.assertTrue(torch.equal(effect, torch.zeros_like(b)))
        self.assertTrue(torch.equal(z, cells['pp']))
        for key, sign in [('pp', 1), ('mp', -1), ('pm', -1), ('mm', 1)]:
            cells[key] = cells[key]+sign*w
        z, effect = combine_interaction(cells)
        self.assertTrue(torch.equal(effect, (4*w).expand_as(b)))
        self.assertTrue(torch.equal(z, cells['pp']+4*w))

    def test_all_native_classes_and_class_permutation(self):
        pp = torch.eye(5, dtype=torch.float64)*4
        cells = {k: pp if k == 'pp' else torch.zeros_like(pp) for k in CELLS}
        z, _ = combine_interaction(cells)
        self.assertTrue(torch.equal(z.argmax(-1), torch.arange(5)))
        permutation = torch.tensor([3, 0, 4, 2, 1])
        changed, _ = combine_interaction({k:v[:, permutation] for k,v in cells.items()})
        self.assertTrue(torch.equal(changed, z[:, permutation]))
        self.assertTrue(torch.all(z.softmax(-1) > 0))

    def test_independent_branch_offsets_and_probability_ratio(self):
        torch.manual_seed(231)
        cells = {k: torch.randn(8, 5, dtype=torch.float64) for k in CELLS}
        z, _ = combine_interaction(cells)
        shifted, _ = combine_interaction({k:v+offset for (k,v),offset in zip(cells.items(), [9., -2., 4., 1.])})
        torch.testing.assert_close(shifted.softmax(-1), z.softmax(-1), rtol=1e-13, atol=1e-13)
        p = {k:v.softmax(-1) for k,v in cells.items()}
        expected = p['pp'].square()*p['mm']/(p['mp']*p['pm'])
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(z.softmax(-1), expected, rtol=1e-13, atol=1e-13)
        singles = torch.cat([combine_interaction({k:v[i:i+1] for k,v in cells.items()})[0] for i in range(8)])
        self.assertTrue(torch.equal(z, singles))

    def test_missing_or_different_class_cells_rejected(self):
        cells = {k:torch.zeros(2, 5) for k in CELLS}
        with self.assertRaises(ValueError):
            combine_interaction({k:v for k,v in cells.items() if k != 'mm'})
        with self.assertRaises(ValueError):
            combine_interaction(dict(cells, mm=torch.zeros(2, 4)))

    def test_all_four_controllers_match_independent_attention(self):
        torch.manual_seed(237)
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        cfg = dict(method='binding_transport', task_binding_fraction=.5,
            task_binding_distribution='uniform', negative_task_strength=4,
            contrast_negative_mode='joint_interaction')
        controller = ResearchController([SimpleNamespace(self_attn=module)], cfg)
        mapping = dict(visual=[1,2,5], target={'last_frame':[2]}, negative={'last_frame':[1]},
            prompt_text_positions=[0,3,4,6], task_positions=[3,4], query=6)
        visual = torch.tensor([0,1,1,0,0,0,0], dtype=torch.bool)
        roi = torch.tensor([0,0,1,0,0,0,0], dtype=torch.bool)
        text = torch.tensor([1,0,0,1,1,0,1], dtype=torch.bool)
        task = torch.tensor([0,0,0,1,1,0,0], dtype=torch.bool)
        try:
            for qn in [7,1]:
                q = torch.randn(2,4,qn,8); k = torch.randn(2,2,7,8); v = torch.randn_like(k)
                causal = torch.arange(7)[None,:] <= torch.arange(7-qn,7)[:,None]
                for kind in ['none','bool','infinity','finite_min']:
                    allowed = causal.expand(2,1,qn,7).clone()
                    if kind != 'none': allowed[0,:,:,4] = False
                    if kind == 'none': mask = None
                    elif kind == 'bool': mask = allowed
                    else:
                        mask = torch.zeros(2,1,qn,7)
                        mask.masked_fill_(~allowed, -torch.inf if kind == 'infinity' else torch.finfo(mask.dtype).min)
                    score = (q@k.repeat_interleave(2,1).transpose(-1,-2))/(8**.5)
                    score.masked_fill_(~allowed, -torch.inf)
                    before = torch.nan_to_num(score.softmax(-1), nan=0.)
                    for cell,(strength,method) in CELLS.items():
                        cfg['method'] = method
                        state = controller.steer([mapping,mapping], [{'layer':0,'head':1}], strength, 'last_frame')
                        actual,_ = controller.forward(module,q,k,v,mask)
                        selected = before[:,1:2].clone()
                        # Independent log-domain normalization at each domain.
                        for domain,target,gain in [(visual,roi,strength)]+(
                                [(text,task,-4.)] if method == 'binding_task_suppression' else []):
                            restricted = score[:,1:2].masked_fill(~domain,-torch.inf)+gain*target
                            weights = torch.nan_to_num(restricted.softmax(-1),nan=0.)
                            mass = before[:,1:2,...,domain].sum(-1,keepdim=True)
                            selected = torch.where(domain, weights*mass, selected)
                        if method == 'binding_transport':
                            visible_task = (allowed&task).to(selected.dtype)
                            n = visible_task.sum(-1,keepdim=True)
                            mass = before[:,1:2,...,text].sum(-1,keepdim=True)
                            change = .5*(visible_task*mass/n.clamp_min(1)-selected*text)
                            selected += torch.where(n>0,change,torch.zeros_like(change))
                        for domain in [visual,text,~(visual|text)]:
                            torch.testing.assert_close(selected[...,domain].sum(-1),before[:,1:2,...,domain].sum(-1),rtol=1e-5,atol=1e-6)
                        expected_p = before.clone(); expected_p[:,1:2] = selected
                        expected = (expected_p@v.repeat_interleave(2,1)).transpose(1,2)
                        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                        baseline,_ = original(module,q,k,v,mask)
                        self.assertTrue(torch.equal(actual[:,:,[0,2,3]],baseline[:,:,[0,2,3]]))
                        self.assertEqual(state['diagnostics']['0']['strength'],strength)
                        self.assertEqual(state['diagnostics']['0']['method'],method)
                        self.assertTrue(torch.equal(selected.masked_select(~allowed),torch.zeros_like(selected.masked_select(~allowed))))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def fake_runtime(self, fail_at=None):
        runtime = object.__new__(InteractionRuntime)
        runtime.cfg = dict(model='qwen', contrast_negative_mode='joint_interaction', contrast_weight=1,
            negative_strength=4, negative_task_strength=4, task_binding_fraction=.5,
            task_binding_distribution='uniform', bias=4, method='binding_transport')
        runtime.prepare = lambda *a: ({}, [{}], [0], ['literal input'])
        runtime.audit = lambda m: {'input_ids_sha256':'same'}
        runtime.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda x,**k:[int(x)]))
        state = dict(cell=None, calls=[], clears=0)
        def steer(maps,heads,strength,scope,region):
            cell = next(k for k,v in CELLS.items() if v == (strength,runtime.cfg['method']))
            state['cell'] = cell
            return dict(diagnostics={'0':dict(cell=cell, heads=[0])})
        def clear():
            state['cell'] = None; state['clears'] += 1
        runtime.controller = SimpleNamespace(steer=steer, clear=clear)
        def model(**kwargs):
            state['calls'].append(state['cell'])
            if fail_at and state['cell'] == fail_at: raise RuntimeError('synthetic branch error')
            gain = 1 if state['cell'] is None else list(CELLS).index(state['cell'])+1
            return SimpleNamespace(logits=torch.arange(6.).reshape(1,1,6)*gain)
        runtime.model = model
        return runtime,state

    def test_runtime_executes_four_cells_and_one_baseline(self):
        runtime,state = self.fake_runtime()
        sample = [dict(example_id='arbitrary-id')]
        rank = {'all_frames':{'ranking':[dict(layer=0,head=0)]}}
        base = runtime.predict(sample,'baseline')[0]
        self.assertEqual(state['calls'],[None]); self.assertEqual(base['actual_forward_branches'],1)
        out = runtime.predict(sample,'all_frames:target:1',rank)[0]
        self.assertEqual(state['calls'],[None,'pp','mp','pm','mm'])
        self.assertEqual(out['actual_forward_branches'],4)
        self.assertEqual(set(out['native_cell_logits']),set(CELLS))
        self.assertEqual(state['clears'],5); self.assertIsNone(state['cell'])
        self.assertEqual(runtime.cfg['method'],'binding_transport')
        changed = runtime.predict([dict(example_id='renamed-id')],'all_frames:target:1',rank)[0]
        self.assertEqual(out['native_class_probabilities'],changed['native_class_probabilities'])

    def test_branch_exception_clears_controller_and_restores_method(self):
        runtime,state = self.fake_runtime('pm')
        with self.assertRaises(RuntimeError):
            runtime.predict([dict(example_id='x')],'all_frames:target:1',
                {'all_frames':{'ranking':[dict(layer=0,head=0)]}})
        self.assertEqual(state['calls'],['pp','mp','pm'])
        self.assertIsNone(state['cell']); self.assertEqual(runtime.cfg['method'],'binding_transport')


if __name__ == '__main__':
    unittest.main()
