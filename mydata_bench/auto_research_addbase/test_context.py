import unittest
from types import SimpleNamespace
import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .context import context_profile
from .attention import redistribute, ResearchController


def mapping():
    return {'visual':list(range(1,9)),'query':9,
            'target':{'all_frames':[1,6]},'negative':{'all_frames':[2,3,4,5,7,8]},
            'wrong':{'all_frames':[4,7]},
            'alignment':{'all_frames':[{'start':1,'end':5,'grid_thw':[1,4,4]},
                                       {'start':5,'end':9,'grid_thw':[1,4,4]}]}}


class ContextTests(unittest.TestCase):
    def test_geometry_and_per_span_matched_control(self):
        m=mapping()
        ids,weights,audit=context_profile(m,'all_frames','target',.15)
        wrong_ids,wrong_weights,wrong_audit=context_profile(m,'all_frames','wrong',.15)
        self.assertEqual(ids,wrong_ids)
        self.assertEqual(audit['domain_count'],8)
        for start in [0,4]:
            np.testing.assert_array_equal(np.sort(weights[start:start+4]),np.sort(wrong_weights[start:start+4]))
        for ids0,weights0,region in [(ids,weights,'target'),(wrong_ids,wrong_weights,'wrong')]:
            self.assertEqual({p for p,w in zip(ids0,weights0) if w==1.},set(m[region]['all_frames']))
        self.assertAlmostEqual(weights[1],np.exp(-.5**2/(2*.15**2)))
        self.assertEqual(wrong_audit['control'],'radial_rank_permutation_preserving_gain_multiset')

    def test_soft_kernel_probability_and_causal_invariants(self):
        torch.manual_seed(34)
        score=torch.randn(2,4,10,10,dtype=torch.float64)
        causal=torch.ones(10,10,dtype=torch.bool).tril()
        score.masked_fill_(~causal,-torch.inf)
        p=score.softmax(-1)
        domain=torch.tensor([0]+[1]*8+[0],dtype=torch.bool)
        ids,values,_=context_profile(mapping(),'all_frames','target',.15)
        profile=torch.zeros(10,dtype=torch.float64);profile[ids]=torch.tensor(values,dtype=torch.float64)
        shifted=redistribute(p,domain,profile,4.)
        torch.testing.assert_close(shifted.sum(-1),p.sum(-1),rtol=0,atol=1e-14)
        torch.testing.assert_close(shifted[...,domain].sum(-1),p[...,domain].sum(-1),rtol=0,atol=1e-14)
        self.assertTrue(torch.equal(shifted[...,~domain],p[...,~domain]))
        self.assertTrue(torch.equal(shifted[...,~causal],p[...,~causal]))
        tilted=score+4*profile
        correction=tilted[...,domain].logsumexp(-1,keepdim=True)-score[...,domain].logsumexp(-1,keepdim=True)
        # Query zero sees no domain; its baseline is the explicit no-op reference.
        expected=(tilted-domain*torch.nan_to_num(correction,nan=0.)).softmax(-1)
        torch.testing.assert_close(shifted,expected,rtol=0,atol=1e-14)

    def test_grouped_heads_match_independent_prefill_and_decode(self):
        torch.manual_seed(83)
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        module=SimpleNamespace(num_key_value_groups=2,is_causal=True)
        controller=ResearchController([SimpleNamespace(self_attn=module)],
            {'method':'context_transport','context_radius':.15})
        try:
            for qn in [10,1]:
                q=torch.randn(1,4,qn,8);k=torch.randn(1,2,10,8);v=torch.randn_like(k)
                m=mapping();controller.steer([m],[{'layer':0,'head':1}],4.,'all_frames')
                actual,_=controller.forward(module,q,k,v,None)
                score=q@k.repeat_interleave(2,1).transpose(-1,-2)/(8**.5)
                mask=torch.ones(qn,10,dtype=torch.bool).tril(10-qn)
                score.masked_fill_(~mask,-torch.inf)
                p=score.softmax(-1)
                ids,values,_=context_profile(m,'all_frames','target',.15)
                profile=torch.zeros(10);profile[ids]=torch.tensor(values)
                domain=torch.tensor([0]+[1]*8+[0],dtype=torch.bool)
                p[:,1]=redistribute(p[:,1],domain,profile,4.)
                expected=(p@v.repeat_interleave(2,1)).transpose(1,2)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                self.assertIn('context_kernel_audit',m)
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)


if __name__=='__main__':unittest.main()
