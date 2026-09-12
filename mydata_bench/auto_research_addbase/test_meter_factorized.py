import unittest
from types import SimpleNamespace

import torch

from .evidence import EvidenceRuntime,combine_native_logits,mean_native_counterfactuals


class MeterFactorizedTests(unittest.TestCase):
    def runtime(self):
        runtime=object.__new__(EvidenceRuntime)
        runtime.cfg=dict(model='meter',method='binding_transport',bias=4,contrast_weight=2,
            contrast_negative_mode='visual_and_task',negative_strength=4,negative_task_strength=4,
            task_binding_fraction=.5,task_binding_distribution='uniform')
        runtime.prepare=lambda *a: ({},[{'query':1},{'query':2}],[1,2],['literal-a','literal-b'])
        runtime.audit=lambda m:m
        state=dict(branch=None,calls=[],clears=0)
        def steer(maps,heads,strength,scope,region):
            state['branch']=(runtime.cfg['method'],strength)
            return dict(diagnostics={'0':dict(method=runtime.cfg['method'],strength=strength,heads=[0])})
        def clear():state['branch']=None;state['clears']+=1
        runtime.controller=SimpleNamespace(steer=steer,clear=clear)
        class Model:
            def __call__(self,**kwargs):
                state['calls'].append(state['branch'])
                value={None:0.,('binding_transport',4):1.,('mass_transport',-4):-1.,
                    ('binding_task_suppression',4):-2.}[state['branch']]
                h=torch.full((2,3,1),-99.)
                h[0,1,0]=value;h[1,2,0]=value+.5
                return SimpleNamespace(last_hidden_state=h)
            def progress_head(self,h):return h*torch.linspace(-1,1,10)[None,:]
            def success_head(self,h):return h
        runtime.model=Model()
        return runtime,state

    def test_three_actual_native_heads_use_task_negative_success(self):
        runtime,state=self.runtime()
        rows=runtime.predict([dict(example_id='a'),dict(example_id='b')],'all_frames:target:1',
            {'all_frames':{'ranking':[dict(layer=0,head=0)]}})
        self.assertEqual(state['calls'],[('binding_transport',4),('mass_transport',-4),('binding_task_suppression',4)])
        self.assertEqual(state['clears'],3);self.assertIsNone(state['branch'])
        self.assertEqual(runtime.cfg['method'],'binding_transport')
        for i,row in enumerate(rows):
            self.assertEqual(row['actual_forward_branches'],3)
            self.assertEqual(row['success_logit_negative_task'],-2.+i*.5)
            mean=(row['success_logit_negative']+row['success_logit_negative_task'])/2
            self.assertEqual(row['success_logit_negative_mean'],mean)
            expected=torch.tensor(3*row['success_logit_positive']-2*mean).sigmoid().item()
            self.assertAlmostEqual(row['success_probability'],expected,places=7)
            old_wrong=torch.tensor(3*row['success_logit_positive']-2*row['success_logit_negative']).sigmoid().item()
            self.assertNotAlmostEqual(row['success_probability'],old_wrong,places=5)
            p,v,t=[torch.tensor(row[k]) for k in ['native_class_logits_positive',
                'native_class_logits_negative','native_class_logits_negative_task']]
            expected_bins=(3*p-2*(.5*(v+t))).softmax(-1)
            torch.testing.assert_close(torch.tensor(row['native_class_probabilities']),expected_bins)
            self.assertAlmostEqual(row['progress'],(expected_bins*torch.linspace(0,1,10)).sum().item(),places=7)
            self.assertEqual(len(row['native_class_probabilities']),10)

    def test_native_baseline_is_one_unsteered_forward(self):
        runtime,state=self.runtime()
        rows=runtime.predict([dict(example_id='a'),dict(example_id='b')],'baseline')
        self.assertEqual(state['calls'],[None])
        for i,row in enumerate(rows):
            self.assertEqual(row['actual_forward_branches'],1)
            self.assertIsNone(row['success_logit_negative_task'])
            self.assertIsNone(row['success_logit_negative_mean'])
            self.assertAlmostEqual(row['success_probability'],torch.tensor(i*.5).sigmoid().item(),places=7)

    def test_all_ten_bins_and_binary_label_exchange(self):
        pos=torch.eye(10)*4;zero=torch.zeros_like(pos)
        combined=combine_native_logits(pos,mean_native_counterfactuals(zero,zero),2)
        self.assertTrue(torch.equal(combined.argmax(-1),torch.arange(10)))
        torch.manual_seed(25)
        p,v,t=[torch.randn(16,dtype=torch.float64) for _ in range(3)]
        a=combine_native_logits(p,mean_native_counterfactuals(v,t),2)
        b=combine_native_logits(-p,mean_native_counterfactuals(-v,-t),2)
        torch.testing.assert_close(a.sigmoid(),1-b.sigmoid(),atol=1e-14,rtol=1e-14)


if __name__=='__main__':unittest.main()
