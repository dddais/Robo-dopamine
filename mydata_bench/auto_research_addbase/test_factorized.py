import unittest
from types import SimpleNamespace

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController
from .evidence import combine_native_logits, mean_native_counterfactuals


class FactorizedEvidenceTests(unittest.TestCase):
    def test_identical_counterfactuals_reduce_to_original_contrast(self):
        torch.manual_seed(337)
        positive = torch.randn(8,5)
        negative = torch.randn(8,5)
        mean = mean_native_counterfactuals(negative,negative)
        self.assertTrue(torch.equal(mean,negative))
        for alpha in [.5,1.,2.]:
            self.assertTrue(torch.equal(combine_native_logits(positive,mean,alpha),
                                        combine_native_logits(positive,negative,alpha)))

    def test_geometric_negative_distribution_and_all_classes(self):
        torch.manual_seed(201)
        positive = torch.eye(5,dtype=torch.float64)*4
        visual = torch.randn(5,5,dtype=torch.float64)*.1
        task = torch.randn(5,5,dtype=torch.float64)*.1
        mean = mean_native_counterfactuals(visual,task)
        combined = combine_native_logits(positive,mean,1.)
        # A logit mean is the normalized geometric mean of probabilities.
        expected = positive.softmax(-1).square() / torch.sqrt(visual.softmax(-1)*task.softmax(-1))
        expected /= expected.sum(-1,keepdim=True)
        torch.testing.assert_close(combined.softmax(-1),expected,rtol=1e-13,atol=1e-13)
        self.assertTrue(torch.equal(combined.argmax(-1),torch.arange(5)))
        torch.testing.assert_close(mean_native_counterfactuals(visual,task),mean_native_counterfactuals(task,visual))
        shifted = combine_native_logits(positive+9,mean_native_counterfactuals(visual-3,task+5),1.)
        torch.testing.assert_close(shifted.softmax(-1),combined.softmax(-1),rtol=1e-13,atol=1e-13)

    def test_task_negative_matches_independent_log_normalizers(self):
        torch.manual_seed(51)
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        module = SimpleNamespace(num_key_value_groups=2,is_causal=True)
        cfg = dict(method='binding_task_suppression',task_binding_fraction=.5,
                   task_binding_distribution='uniform',negative_task_strength=4.)
        controller = ResearchController([SimpleNamespace(self_attn=module)],cfg)
        mapping = dict(visual=[1,2,5],target={'last_frame':[2]},negative={'last_frame':[1]},
                       prompt_text_positions=[0,3,4,6],task_positions=[3,4],query=6)
        visual = torch.tensor([0,1,1,0,0,0,0],dtype=torch.bool)
        roi = torch.tensor([0,0,1,0,0,0,0],dtype=torch.bool)
        text = torch.tensor([1,0,0,1,1,0,1],dtype=torch.bool)
        task = torch.tensor([0,0,0,1,1,0,0],dtype=torch.bool)
        try:
            for qn in [7,1]:
                q = torch.randn(2,4,qn,8)
                k = torch.randn(2,2,7,8)
                v = torch.randn_like(k)
                causal = torch.arange(7)[None,:] <= torch.arange(7-qn,7)[:,None]
                for kind in ['none','bool','infinity','finite_min']:
                    allowed = causal.expand(2,1,qn,7).clone()
                    if kind!='none':allowed[0,:,:,4]=False
                    if kind=='none':mask=None
                    elif kind=='bool':mask=allowed
                    else:
                        mask=torch.zeros(2,1,qn,7)
                        mask.masked_fill_(~allowed,-torch.inf if kind=='infinity' else torch.finfo(mask.dtype).min)
                    state=controller.steer([mapping,mapping],[{'layer':0,'head':1}],4.,'last_frame')
                    actual,_=controller.forward(module,q,k,v,mask)
                    scores=(q@k.repeat_interleave(2,1).transpose(-1,-2))/(8**.5)
                    scores.masked_fill_(~allowed,-torch.inf)
                    before=torch.nan_to_num(scores.softmax(-1),nan=0.)
                    modified=scores[:,1:2].clone()
                    for domain,target,strength in [(visual,roi,4.),(text,task,-4.)]:
                        restricted=scores[:,1:2].masked_fill(~domain,-torch.inf)
                        original_log_mass=restricted.logsumexp(-1,keepdim=True)
                        changed_log_mass=(restricted+strength*target).logsumexp(-1,keepdim=True)
                        correction=torch.where(torch.isfinite(original_log_mass),changed_log_mass-original_log_mass,0.)
                        modified=modified+strength*target-correction*domain
                    after=torch.nan_to_num(modified.softmax(-1),nan=0.)
                    for domain in [visual,text,~(visual|text)]:
                        torch.testing.assert_close(after[...,domain].sum(-1),before[:,1:2,...,domain].sum(-1),rtol=1e-5,atol=1e-6)
                    self.assertTrue(torch.equal(after.masked_select(~allowed),torch.zeros_like(after.masked_select(~allowed))))
                    probability=before.clone();probability[:,1:2]=after
                    expected=(probability@v.repeat_interleave(2,1)).transpose(1,2)
                    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                    baseline,_=original(module,q,k,v,mask)
                    self.assertTrue(torch.equal(actual[:,:,[0,2,3]],baseline[:,:,[0,2,3]]))
                    self.assertEqual(state['diagnostics']['0']['task_logit_strength'],-4.)
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa',original)


if __name__=='__main__':unittest.main()
