import unittest
from types import SimpleNamespace
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .attention import ResearchController, redistribute, mixture_redistribute


class TransportTests(unittest.TestCase):
    def test_dual_ranking_mass_observation_is_noninterfering(self):
        torch.manual_seed(18)
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        module=SimpleNamespace(num_key_value_groups=2,is_causal=True)
        ctrl=ResearchController([SimpleNamespace(self_attn=module)],
              {'method':'binding_transport','ranking_strategy':'dual_mass'})
        try:
            q=torch.randn(2,4,6,8);k=torch.randn(2,2,6,8);v=torch.randn_like(k)
            mapping={'visual':[1,2,3], 'target':{'last_frame':[3],'all_frames':[2,3]},
                     'task_positions':[0,4]}
            state=ctrl.rank([mapping,mapping],[3,5],num_layers=1,num_heads=4)
            actual,_=ctrl.forward(module,q,k,v,None)
            baseline,_=original(module,q,k,v,None)
            self.assertTrue(torch.equal(actual,baseline))
            scores=q@k.repeat_interleave(2,1).transpose(-1,-2)/(8**.5)
            scores.masked_fill_(~torch.ones(6,6,dtype=torch.bool).tril(),-torch.inf)
            weights=scores.softmax(-1)
            for i,j in enumerate([3,5]):
                expected=weights[i,:,j,[0,4]].sum(-1)
                torch.testing.assert_close(torch.as_tensor(state['task_mass'][i,0]),expected.double(),rtol=1e-6,atol=1e-7)
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_dual_ranking_keeps_unique_nested_budget(self):
        from .dual_rank import alternating_union
        visual=[{'layer':0,'head':i} for i in range(6)]
        instruction=[visual[i] for i in [0,5,4,3,2,1]]
        ordered=alternating_union(visual,instruction)
        self.assertEqual([row['head'] for row in ordered],[0,5,1,4,2,3])
        for k in range(1,7):self.assertEqual(len({row['head'] for row in ordered[:k]}),k)
        with self.assertRaises(ValueError):alternating_union(visual,instruction[:-1])

    def test_disjoint_text_and_visual_conservation(self):
        p=torch.tensor([[.1,.1,.2,.05,.1,.2,.15,.1]],dtype=torch.float64)
        visual=torch.tensor([0,1,1,0,0,0,0,0],dtype=torch.bool)
        target=torch.tensor([0,0,1,0,0,0,0,0],dtype=torch.bool)
        text=torch.tensor([0,0,0,0,1,1,1,0],dtype=torch.bool)
        instruction=torch.tensor([0,0,0,0,0,1,0,0],dtype=torch.bool)
        a=mixture_redistribute(redistribute(p,visual,target,4.),text,instruction,.5)
        b=redistribute(mixture_redistribute(p,text,instruction,.5),visual,target,4.)
        torch.testing.assert_close(a,b,rtol=0,atol=1e-15)
        torch.testing.assert_close(a[:,visual].sum(-1),p[:,visual].sum(-1),rtol=0,atol=1e-15)
        torch.testing.assert_close(a[:,text].sum(-1),p[:,text].sum(-1),rtol=0,atol=1e-15)
        self.assertTrue(torch.equal(a[:,~(visual|text)],p[:,~(visual|text)]))

    def test_bounded_context_retention_and_invisible_target(self):
        p=torch.tensor([[.1,.2,.3,.4],[.3,.3,0.,.4]],dtype=torch.float64)
        domain=torch.tensor([0,1,1,0],dtype=torch.bool)
        target=torch.tensor([0,0,1,0],dtype=torch.bool)
        shifted=mixture_redistribute(p,domain,target,.5)
        expected=torch.tensor([[.1,.1,.4,.4],[.3,.3,0.,.4]],dtype=torch.float64)
        torch.testing.assert_close(shifted,expected,rtol=0,atol=1e-15)
        self.assertTrue(torch.equal(shifted[:,~domain],p[:,~domain]))

    def test_probability_invariants_and_softmax_equivalence(self):
        torch.manual_seed(123)
        logits=torch.randn(2,3,7,9,dtype=torch.float64)
        causal=torch.ones(7,9,dtype=torch.bool).tril(2)
        logits.masked_fill_(~causal,-torch.inf)
        p=logits.softmax(-1)
        domain=torch.tensor([0,1,1,1,0,1,0,0,0],dtype=torch.bool)
        target=torch.tensor([0,0,1,0,0,1,0,0,0],dtype=torch.bool)
        shifted=redistribute(p,domain,target,4.)
        torch.testing.assert_close(shifted.sum(-1),p.sum(-1),rtol=0,atol=1e-14)
        torch.testing.assert_close(shifted[...,domain].sum(-1),p[...,domain].sum(-1),rtol=0,atol=1e-14)
        self.assertTrue(torch.equal(shifted[...,~domain],p[...,~domain]))
        self.assertTrue(torch.equal(shifted[... , ~causal],p[...,~causal]))
        old_z=logits[...,domain].logsumexp(-1,keepdim=True)
        new=logits+target*4.
        new_z=new[...,domain].logsumexp(-1,keepdim=True)
        equivalent=(new-domain*(new_z-old_z)).softmax(-1)
        torch.testing.assert_close(shifted,equivalent,rtol=0,atol=1e-14)

    def test_full_controller_grouped_heads_prefill_and_decode(self):
        torch.manual_seed(8)
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        module=SimpleNamespace(num_key_value_groups=2,is_causal=True)
        ctrl=ResearchController([SimpleNamespace(self_attn=module)],{'method':'mass_transport'})
        try:
            for qn in [6,1]:
                q=torch.randn(2,4,qn,8);k=torch.randn(2,2,6,8);v=torch.randn_like(k)
                mask=torch.zeros(2,1,qn,6)
                if qn>1:mask.masked_fill_(~torch.ones(qn,6,dtype=torch.bool).tril(),-torch.inf)
                mask[0,:,:,0]=-torch.inf
                mapping={'visual':[1,2,3,4],'target':{'all_frames':[2]},'negative':{'all_frames':[1,3,4]}}
                ctrl.steer([mapping,mapping],[{'layer':0,'head':1}],3.,'all_frames')
                actual,_=ctrl.forward(module,q,k,v,mask)
                logits=q@k.repeat_interleave(2,1).transpose(-1,-2)/(8**.5)+mask
                p=torch.nan_to_num(logits.softmax(-1))
                domain=torch.tensor([0,1,1,1,1,0],dtype=torch.bool)
                target=torch.tensor([0,0,1,0,0,0],dtype=torch.bool)
                p[:,1]=redistribute(p[:,1],domain,target,3.)
                expected=(p@v.repeat_interleave(2,1)).transpose(1,2)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        finally: ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_readout_scope_leaves_all_prior_queries_exactly_unchanged(self):
        torch.manual_seed(81)
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        module=SimpleNamespace(num_key_value_groups=2,is_causal=True)
        ctrl=ResearchController([SimpleNamespace(self_attn=module)],
              {'method':'mass_transport','research_query_scope':'readout'})
        try:
            q=torch.randn(1,4,6,8);k=torch.randn(1,2,6,8);v=torch.randn_like(k)
            mapping={'visual':[1,2,3,4],'query':5,'target':{'all_frames':[2]},'negative':{'all_frames':[1,3,4]}}
            baseline,_=original(module,q,k,v,None)
            for method in ['mass_transport','positive_bias']:
                ctrl.cfg['method']=method
                ctrl.steer([mapping],[{'layer':0,'head':1}],3.,'all_frames')
                actual,_=ctrl.forward(module,q,k,v,None)
                self.assertTrue(torch.equal(actual[:,:5],baseline[:,:5]))
                self.assertFalse(torch.equal(actual[:,5,1],baseline[:,5,1]))
                if method=='positive_bias':
                    score=q[:,:,5:]@k.repeat_interleave(2,1).transpose(-1,-2)/(8**.5)
                    score[:,1,:,2]+=3.
                    expected=(score.softmax(-1)@v.repeat_interleave(2,1)).transpose(1,2)
                    torch.testing.assert_close(actual[:,5:],expected,rtol=1e-5,atol=1e-6)
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':unittest.main()
