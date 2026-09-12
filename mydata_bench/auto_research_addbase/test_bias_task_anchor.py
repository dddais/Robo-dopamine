import copy
import unittest
from types import SimpleNamespace

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController
from .bias_task_anchor import CELLS, BiasTaskAnchorRuntime, compose
from .bias_task_tasks import verify_rows


class BiasTaskAnchorTests(unittest.TestCase):
    def test_equal_task_cells_recover_original_steering(self):
        torch.manual_seed(26)
        b,p=torch.randn(2,8,5),torch.randn(2,8,5)
        z,effect=compose(dict(bias=b,pp=p,pm=p))
        self.assertTrue(torch.equal(z,b));self.assertTrue(torch.equal(effect,torch.zeros_like(effect)))

    def test_probability_ratio_offsets_and_sample_independence(self):
        torch.manual_seed(2602)
        cells={k:torch.randn(8,5,dtype=torch.float64) for k in CELLS}
        z,_=compose(cells);p={k:v.softmax(-1) for k,v in cells.items()}
        expected=p['bias']*p['pp']/p['pm'];expected/=expected.sum(-1,keepdim=True)
        torch.testing.assert_close(z.softmax(-1),expected,rtol=1e-13,atol=1e-13)
        shifted,_=compose({k:v+offset for (k,v),offset in zip(cells.items(),[12.,-4.,7.])})
        torch.testing.assert_close(z.softmax(-1),shifted.softmax(-1),rtol=1e-13,atol=1e-13)
        self.assertTrue(torch.equal(z,torch.cat([compose({k:v[i:i+1] for k,v in cells.items()})[0] for i in range(8)])))

    def test_all_five_classes_and_permutation(self):
        cells=dict(bias=torch.eye(5)*4,pp=torch.zeros(5,5),pm=torch.zeros(5,5))
        z,_=compose(cells);self.assertTrue(torch.equal(z.argmax(-1),torch.arange(5)))
        perm=torch.tensor([3,0,4,2,1]);changed,_=compose({k:v[:,perm] for k,v in cells.items()})
        self.assertTrue(torch.equal(changed,z[:,perm]));self.assertTrue(torch.all(z.softmax(-1)>0))

    def test_missing_or_mismatched_cells_rejected(self):
        cells={k:torch.zeros(2,5) for k in CELLS}
        with self.assertRaises(ValueError):compose({k:v for k,v in cells.items() if k!='bias'})
        with self.assertRaises(ValueError):compose(dict(cells,pm=torch.zeros(2,4)))

    def fake(self,fail=None):
        runtime=object.__new__(BiasTaskAnchorRuntime)
        runtime.cfg=dict(model='qwen',contrast_negative_mode='bias_task_anchor',contrast_weight=1,
            negative_strength=4,negative_task_strength=4,task_binding_fraction=.5,
            task_binding_distribution='uniform',bias=4,method='binding_transport')
        runtime.prepare=lambda *a:({},[{}],[0],['literal input'])
        runtime.audit=lambda m:{'input_ids_sha256':'same'}
        runtime.processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda x,**k:[int(x)]))
        state=dict(cell=None,calls=[],clears=0)
        def steer(maps,heads,strength,scope,region):
            cell=next(k for k,v in CELLS.items() if v==(strength,runtime.cfg['method']))
            state['cell']=cell
            d=dict(heads=[0],all_query_rows=True,causal_mask_preserved=True,prefill_calls=1,
                domain_mass_preserved=True,text_domain_mass_preserved=True,method=runtime.cfg['method'],strength=strength)
            if cell=='bias':d.update(bias=6,generated_text_key_bias=0)
            elif cell=='pp':d.update(task_binding_fraction=.5,task_binding_distribution='uniform')
            else:d.update(task_logit_strength=-4,task_binding_distribution='exponential_suppression')
            return dict(diagnostics={'0':d})
        def clear():state['cell']=None;state['clears']+=1
        runtime.controller=SimpleNamespace(steer=steer,clear=clear)
        def model(**kwargs):
            state['calls'].append(state['cell'])
            if fail and state['cell']==fail:raise RuntimeError('synthetic branch failure')
            gain=1 if state['cell'] is None else list(CELLS).index(state['cell'])+1
            return SimpleNamespace(logits=torch.arange(6.).reshape(1,1,6)*gain)
        runtime.model=model
        return runtime,state

    def test_runtime_three_actual_cells_baseline_and_id_independence(self):
        r,state=self.fake();rank={'all_frames':{'ranking':[dict(layer=0,head=0)]}}
        base=r.predict([dict(example_id='a')],'baseline')[0]
        out=r.predict([dict(example_id='a')],'all_frames:target:1',rank)[0]
        self.assertEqual(state['calls'],[None,'bias','pp','pm']);self.assertEqual(state['clears'],4)
        self.assertIsNone(state['cell']);self.assertEqual(r.cfg['method'],'binding_transport')
        verify_rows({'a':base},['a'],baseline=True);verify_rows({'a':out},['a'],{(0,0)})
        changed=r.predict([dict(example_id='different-id')],'all_frames:target:1',rank)[0]
        self.assertEqual(changed['native_class_probabilities'],out['native_class_probabilities'])

    def test_exception_restores_original_method_and_clears(self):
        r,state=self.fake('pm')
        with self.assertRaises(RuntimeError):r.predict([dict(example_id='a')],'all_frames:target:1',
            {'all_frames':{'ranking':[dict(layer=0,head=0)]}})
        self.assertEqual(state['calls'],['bias','pp','pm']);self.assertIsNone(state['cell'])
        self.assertEqual(r.cfg['method'],'binding_transport')

    def test_original_bias_matches_independent_masked_attention(self):
        torch.manual_seed(2607);original=ALL_ATTENTION_FUNCTIONS['sdpa']
        module=SimpleNamespace(num_key_value_groups=2,is_causal=True)
        cfg=dict(method='bias',contrast_negative_mode='bias_task_anchor')
        controller=ResearchController([SimpleNamespace(self_attn=module)],cfg)
        mapping=dict(visual=[1,2,5],target={'last_frame':[2]},negative={'last_frame':[1]},query=6)
        try:
            for qn in [7,1]:
                q=torch.randn(2,4,qn,8);k=torch.randn(2,2,7,8);v=torch.randn_like(k)
                allowed=(torch.arange(7)[None,:]<=torch.arange(7-qn,7)[:,None]).expand(2,1,qn,7).clone()
                allowed[0,:,:,1]=False
                for mask in [allowed,torch.zeros_like(allowed,dtype=torch.float32).masked_fill(~allowed,-torch.inf)]:
                    state=controller.steer([mapping,mapping],[dict(layer=0,head=1)],6.,'last_frame')
                    actual,_=controller.forward(module,q,k,v,mask)
                    z=(q@k.repeat_interleave(2,1).transpose(-1,-2))/(8**.5)
                    z[:,1,:,2]+=6;z[:,1,:,1]-=6;z.masked_fill_(~allowed,-torch.inf)
                    expected=(torch.nan_to_num(z.softmax(-1),nan=0.)@v.repeat_interleave(2,1)).transpose(1,2)
                    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                    self.assertEqual(state['diagnostics']['0']['generated_text_key_bias'],0)
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_verifier_rejects_nonfinite_or_substituted_native_branch(self):
        r,_=self.fake();row=r.predict([dict(example_id='a')],'all_frames:target:1',
            {'all_frames':{'ranking':[dict(layer=0,head=0)]}})[0]
        for field in ['progress','native_class_probabilities','native_class_logits_combined']:
            bad=copy.deepcopy(row)
            if field=='progress':bad[field]=float('nan')
            else:bad[field][0]=float('nan')
            with self.assertRaises(ValueError):verify_rows({'a':bad},['a'])
        bad=copy.deepcopy(row);bad['cell_attention_diagnostics']['bias']['0']['bias']=4
        with self.assertRaises(ValueError):verify_rows({'a':bad},['a'])
        bad=copy.deepcopy(row);bad['derived_only']=True
        with self.assertRaises(ValueError):verify_rows({'a':bad},['a'])


if __name__=='__main__':unittest.main()
