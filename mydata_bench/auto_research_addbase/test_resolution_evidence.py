import copy
import unittest
from types import SimpleNamespace

import torch

from .resolution_evidence import ResolutionRuntime, compose, validate_views


def mapping(high=False):
    a = dict(nonvisual_input_ids_sha256='text', prompt_sha256='prompt', input_ids_sha256='high' if high else 'low',
        resolution_sources=[dict(size=[640,480], sources=[1], mosaic=False)], visual=list(range(8 if high else 2)),
        alignment={}, wrong={'all_frames':[0], 'last_frame':[0]})
    for scope in ['all_frames', 'last_frame']:
        a['alignment'][scope] = [dict(source_frames=[1], tracking_frames=[1], boxes=[[20,20,60,60]], mosaic=False,
                                    grid_thw=[1,4,8] if high else [1,2,4])]
    return a


class ResolutionTests(unittest.TestCase):
    def test_equal_views_recover_original_bias(self):
        torch.manual_seed(2901); z = torch.randn(8,5)
        self.assertTrue(torch.equal(compose(z,z),z))

    def test_probability_ratio_and_constant_offsets(self):
        torch.manual_seed(2902); h = torch.randn(8,5,dtype=torch.float64); l = torch.randn_like(h)
        z = compose(h,l); ratio = h.softmax(-1)**2/l.softmax(-1); ratio /= ratio.sum(-1,keepdim=True)
        torch.testing.assert_close(z.softmax(-1),ratio,rtol=1e-13,atol=1e-13)
        torch.testing.assert_close(z.softmax(-1),compose(h+4,l-3).softmax(-1),rtol=1e-13,atol=1e-13)

    def test_full_classes_permutation_and_shape_rejection(self):
        h,l = torch.eye(5),torch.zeros(5,5)
        z = compose(h,l);self.assertTrue(torch.equal(z.argmax(-1),torch.arange(5)))
        perm = [2,0,3,4,1];self.assertTrue(torch.equal(compose(h[:,perm],l[:,perm]),z[:,perm]))
        with self.assertRaises(ValueError):compose(h,l[:,:4])

    def test_geometry_contract_rejects_changed_semantics_and_sampling(self):
        low,high = mapping(),mapping(True);validate_views([low],[high])
        for field,value in [('nonvisual_input_ids_sha256','changed'),('visual',[0]),('input_ids_sha256','low')]:
            bad=copy.deepcopy(high);bad[field]=value
            with self.assertRaises(ValueError):validate_views([low],[bad])
        for field,value in [('source_frames',[2]),('tracking_frames',[2]),('boxes',[[0,0,20,20]]),('grid_thw',[1,4,6])]:
            bad=copy.deepcopy(high);bad['alignment']['all_frames'][0][field]=value
            with self.assertRaises(ValueError):validate_views([low],[bad])

    def fake(self,fail=False):
        r=object.__new__(ResolutionRuntime)
        r.cfg=dict(model='qwen',contrast_negative_mode='resolution',contrast_weight=1,resolution_high_max_pixels=200704,
                   max_pixels=50176,method='bias',bias=6)
        r.processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s,**k:[int(s)]))
        r.audit=lambda m:m
        r.prepare_view=lambda samples,pixels,*a:({'pixels':pixels},[mapping(pixels==200704)],[1],['same prompt'])
        state=dict(steered=False,calls=[],clears=0)
        def steer(*a):state['steered']=True;return dict(diagnostics={'0':dict(heads=[0],bias=6)})
        def clear():state['steered']=False;state['clears']+=1
        r.controller=SimpleNamespace(steer=steer,clear=clear)
        def model(**kwargs):
            pixels=kwargs['pixels'];state['calls'].append((pixels,state['steered']))
            if fail and pixels==200704:raise RuntimeError('synthetic high-resolution failure')
            gain=(pixels/50176)+(1 if state['steered'] else 0)
            return SimpleNamespace(logits=torch.arange(6.).view(1,1,6)*gain)
        r.model=model
        return r,state

    def test_runtime_executes_both_baselines_and_steered_views(self):
        r,s=self.fake();rank={'all_frames':dict(ranking=[dict(layer=0,head=0)])}
        base=r.predict([dict(example_id='a')],'baseline')[0]
        row=r.predict([dict(example_id='a')],'all_frames:target:1',rank)[0]
        self.assertEqual(s['calls'],[(50176,False),(200704,False),(50176,True),(200704,True)])
        self.assertEqual(s['clears'],4);self.assertFalse(s['steered'])
        self.assertEqual(base['native_class_logits_combined'],base['native_class_logits_low'])
        expected=compose(torch.tensor(row['native_class_logits_high']),torch.tensor(row['native_class_logits_low']))
        self.assertEqual(expected.tolist(),row['native_class_logits_combined'])
        same=r.predict([dict(example_id='changed-id')],'all_frames:target:1',rank)[0]
        self.assertEqual(row['native_class_probabilities'],same['native_class_probabilities'])

    def test_failed_high_forward_clears_state(self):
        r,s=self.fake(True);original=copy.deepcopy(r.cfg)
        with self.assertRaises(RuntimeError):r.predict([dict(example_id='a')],'all_frames:target:1',
            {'all_frames':dict(ranking=[dict(layer=0,head=0)])})
        self.assertFalse(s['steered']);self.assertEqual(s['clears'],2);self.assertEqual(r.cfg,original)

    def test_failed_prepare_restores_budget_and_prefix(self):
        r=object.__new__(ResolutionRuntime);r.cfg=dict(max_pixels=50176)
        def fail(*a):
            self.assertEqual(r.cfg['max_pixels'],200704);raise RuntimeError('synthetic processor failure')
        r.prepare=fail
        with self.assertRaises(RuntimeError):r.prepare_view([],200704,None,None)
        self.assertEqual(r.cfg['max_pixels'],50176);self.assertEqual(r.active_ranking_prefix,'')

    def test_prepare_records_nonvisual_content_independent_of_padding(self):
        r=object.__new__(ResolutionRuntime);r.cfg=dict(max_pixels=50176)
        def prep(*a):
            m=dict(visual=[2,3],records=[dict(size=[640,480],sources=[1],mosaic=False)])
            return dict(input_ids=torch.tensor([[0,8,99,99,4]]),attention_mask=torch.tensor([[0,1,1,1,1]])),[m],[4],['text']
        r.prepare=prep
        _,m,_,_=r.prepare_view([],200704,None,None)
        self.assertEqual(m[0]['resolution_sources'],[dict(size=[640,480],sources=[1],mosaic=False)])
        self.assertEqual(m[0]['resolution_max_pixels'],200704);self.assertEqual(r.cfg['max_pixels'],50176)

    def test_registered_policy_and_command_use_one_budget_and_head_family(self):
        from .resolution_tasks import check_policy, command
        spec=check_policy()
        for key,value in [('high_max_pixels',50176),('composition','low+high'),('classes',[1,5])]:
            bad=copy.deepcopy(spec);bad[key]=value
            with self.assertRaises(ValueError):check_policy(bad)
        cmd=command('qwen',['image_text'],'discovery',[8,32],probe=True)
        self.assertEqual(cmd[cmd.index('--contrast-negative-mode')+1],'resolution')
        self.assertEqual(cmd[cmd.index('--resolution-high-max-pixels')+1],'200704')
        self.assertEqual(cmd[cmd.index('--variant')+1],'resolution_evidence_a1')

    def test_native_gate_rejects_geometry_or_formula_substitution(self):
        from .resolution_tasks import verify_rows
        r,_=self.fake()
        row=r.predict([dict(example_id='a')],'all_frames:target:1',{'all_frames':dict(ranking=[dict(layer=0,head=0)])})[0]
        for field,pixels in [('token_audit',50176),('high_token_audit',200704)]:
            m=row[field];m['resolution_max_pixels']=pixels;m['target']={};m['negative']={}
            for scope,records in m['alignment'].items():
                records[0].update(span='span_0',start=0,end=len(m['visual']))
                m['target'][scope]=[0];m['negative'][scope]=list(range(1,len(m['visual'])))
        for field in ['attention_diagnostics','low_attention_diagnostics']:
            row[field]['0'].update(causal_mask_preserved=True,all_query_rows=True,generated_text_key_bias=0,prefill_calls=1)
        verify_rows({'a':row},['a'],{(0,0)})
        for field in ['native_class_logits_combined','native_class_probabilities']:
            bad=copy.deepcopy(row);bad[field][0]=float('nan')
            with self.assertRaises(ValueError):verify_rows({'a':bad},['a'],{(0,0)})
        bad=copy.deepcopy(row);bad['high_token_audit']['target']['all_frames']=[1]
        with self.assertRaises(ValueError):verify_rows({'a':bad},['a'],{(0,0)})


if __name__=='__main__':unittest.main()
