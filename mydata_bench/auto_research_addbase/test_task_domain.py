"""Independent numeric checks for explicit visual bypass and text-only contrast."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController, redistribute
from .evidence import combine_native_logits


def mapping():
    return dict(visual=[1,2,5,6], query=7,
        target={'all_frames':[1,5]}, negative={'all_frames':[2,6]},
        wrong={'all_frames':[2,6]}, prompt_text_positions=[0,3,4,7], task_positions=[3,4])


class TaskDomainTests(unittest.TestCase):
    def test_actual_controller_independent_reference_and_exact_roi_invariance(self):
        torch.manual_seed(2203)
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        module = SimpleNamespace(num_key_value_groups=2,is_causal=True)
        cfg = dict(method='binding_transport',contrast_negative_mode='task',
            task_binding_fraction=.5,task_binding_distribution='uniform',negative_task_strength=4.)
        controller = ResearchController([SimpleNamespace(self_attn=module)],cfg)
        text = torch.tensor([i in [0,3,4,7] for i in range(8)])
        task = torch.tensor([i in [3,4] for i in range(8)])
        try:
            for qn in [8,1]:
                q=torch.randn(2,4,qn,8);k=torch.randn(2,2,8,8)
                # Identity values expose the key probability in each output coordinate.
                v=torch.eye(8).expand(2,2,8,8).clone()
                causal=torch.arange(8)[None,:]<=torch.arange(8-qn,8)[:,None]
                for kind in ['none','boolean','infinity','finite_min']:
                    visible=causal.expand(2,1,qn,8).clone()
                    if kind!='none':visible[0,:,:,4]=False
                    if kind=='none':mask=None
                    elif kind=='boolean':mask=visible
                    else:
                        mask=torch.zeros_like(visible,dtype=torch.float32)
                        mask.masked_fill_(~visible,-torch.inf if kind=='infinity' else torch.finfo(mask.dtype).min)
                    base,_=original(module,q,k,v,mask)
                    scores=(q@k.repeat_interleave(2,1).transpose(-1,-2))/(8**.5)
                    scores.masked_fill_(~visible,-torch.inf)
                    probability=torch.nan_to_num(scores.softmax(-1),nan=0.)
                    for method in ['binding_transport','binding_task_suppression']:
                        cfg['method']=method
                        outputs=[]
                        for region in ['target','wrong']:
                            state=controller.steer([mapping(),mapping()],[dict(layer=0,head=1)],0.,'all_frames',region)
                            with patch('mydata_bench.auto_research_addbase.attention.redistribute',wraps=redistribute) as transport:
                                actual,_=controller.forward(module,q,k,v,mask)
                                self.assertEqual(transport.call_count,0 if method=='binding_transport' else 1)
                                if method=='binding_task_suppression':
                                    self.assertTrue(torch.equal(transport.call_args.args[1][0,0,0],text))
                            outputs.append(actual)
                            diag=state['diagnostics']['0']
                            self.assertEqual(diag['visual_intervention'],'bypassed')
                            self.assertEqual(diag['strength'],0.)
                        self.assertTrue(torch.equal(*outputs))
                        after=probability.clone();part=probability[:,1:2]
                        mass=part[...,text].sum(-1,keepdim=True)
                        allowed_task=visible&task
                        if method=='binding_transport':
                            count=allowed_task.sum(-1,keepdim=True)
                            candidate=part.clone()
                            candidate[...,text]=.5*part[...,text]
                            candidate+=.5*mass/count.clamp_min(1)*allowed_task
                            after[:,1:2]=torch.where(count>0,candidate,part)
                        else:
                            logits=(scores[:,1:2]-4.*task).masked_fill(~text,-torch.inf)
                            conditional=torch.nan_to_num(logits.softmax(-1),nan=0.)
                            candidate=part.clone();candidate[...,text]=(conditional*mass)[...,text]
                            after[:,1:2]=candidate
                        expected=after.transpose(1,2)
                        torch.testing.assert_close(outputs[0],expected,rtol=2e-5,atol=2e-6)
                        # Unselected heads and visual output coordinates are bitwise unchanged.
                        self.assertTrue(torch.equal(outputs[0][:,:,[0,2,3]],base[:,:,[0,2,3]]))
                        self.assertTrue(torch.equal(outputs[0][:,:,1,~text],base[:,:,1,~text]))
                        torch.testing.assert_close(outputs[0][:,:,1,text].sum(-1),base[:,:,1,text].sum(-1),rtol=2e-5,atol=2e-6)
                        self.assertTrue(torch.equal(outputs[0].transpose(1,2).masked_select(~visible),
                            torch.zeros_like(outputs[0].transpose(1,2).masked_select(~visible))))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_empty_task_and_disallowed_visual_strength(self):
        torch.manual_seed(31)
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        module=SimpleNamespace(num_key_value_groups=2,is_causal=True)
        cfg=dict(method='binding_transport',contrast_negative_mode='task',
                 task_binding_fraction=.5,task_binding_distribution='uniform',negative_task_strength=4.)
        controller=ResearchController([SimpleNamespace(self_attn=module)],cfg)
        q=torch.randn(1,4,8,8);k=torch.randn(1,2,8,8);v=torch.randn_like(k)
        m=copy.deepcopy(mapping());m['task_positions']=[]
        try:
            base,_=original(module,q,k,v,None)
            for method in ['binding_transport','binding_task_suppression']:
                cfg['method']=method
                controller.steer([m],[dict(layer=0,head=1)],0.,'all_frames')
                actual,_=controller.forward(module,q,k,v,None)
                self.assertTrue(torch.equal(actual,base))
                controller.steer([mapping()],[dict(layer=0,head=1)],4.,'all_frames')
                with self.assertRaises(ValueError):controller.forward(module,q,k,v,None)
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_native_five_class_permutation_and_scalar_shift(self):
        torch.manual_seed(2)
        pos=torch.eye(5,dtype=torch.float64)*5
        neg=torch.randn(5,5,dtype=torch.float64)*.1
        combined=combine_native_logits(pos,neg,1)
        self.assertTrue(torch.equal(combined.argmax(-1),torch.arange(5)))
        p=pos.softmax(-1).square()/neg.softmax(-1);p/=p.sum(-1,keepdim=True)
        torch.testing.assert_close(combined.softmax(-1),p,rtol=1e-13,atol=1e-13)
        permutation=torch.tensor([4,1,3,0,2])
        permuted=combine_native_logits(pos[:,permutation]+9,neg[:,permutation]-4,1).softmax(-1)
        torch.testing.assert_close(permuted,combined.softmax(-1)[:,permutation],rtol=1e-13,atol=1e-13)


if __name__=='__main__':
    unittest.main()
