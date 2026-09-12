import copy
from types import SimpleNamespace
import unittest

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController, redistribute, uniform_redistribute
from .temporal import temporal_domains, partitioned_redistribute


def mapping():
    records = [dict(start=s, end=s+4, grid_thw=[1,4,4], mosaic=False,
                    source_frames=[2*i, 2*i+1]) for i, s in enumerate([1,5])]
    return dict(visual=list(range(1,9)), query=10,
        target=dict(all_frames=[2,6], last_frame=[6]),
        negative=dict(all_frames=[1,3,4,5,7,8], last_frame=[5,7,8]),
        alignment=dict(all_frames=records, last_frame=records[-1:]),
        prompt_text_positions=[0,9,10], task_positions=[0,9])


class TemporalTests(unittest.TestCase):
    def test_partition_is_accepted_merged_time_not_physical_frames(self):
        self.assertEqual(temporal_domains(mapping(),'all_frames'), [[1,2,3,4],[5,6,7,8]])
        self.assertEqual(temporal_domains(mapping(),'last_frame'), [[5,6,7,8]])
        for mutation in ['mosaic','multiple_times','overlap','missing']:
            m = mapping()
            if mutation == 'mosaic': m['alignment']['all_frames'][0]['mosaic'] = True
            elif mutation == 'multiple_times': m['alignment']['all_frames'][0]['grid_thw'][0] = 2
            elif mutation == 'overlap': m['alignment']['all_frames'].append(copy.deepcopy(m['alignment']['all_frames'][0]))
            else: m['alignment']['all_frames'].pop()
            with self.assertRaises(ValueError): temporal_domains(m,'all_frames')

    def test_plane_mass_external_keys_causal_and_zero_strength(self):
        torch.manual_seed(42)
        scores = torch.randn(2,4,11,11,dtype=torch.float64)
        causal = torch.ones(11,11,dtype=torch.bool).tril()
        scores.masked_fill_(~causal,-torch.inf)
        p = scores.softmax(-1)
        planes = [torch.tensor([i in ids for i in range(11)]) for ids in temporal_domains(mapping(),'all_frames')]
        domain = planes[0] | planes[1]
        target = torch.tensor([i in [2,6] for i in range(11)])
        for strength in [-4.,0.,4.]:
            shifted = partitioned_redistribute(p,planes,target,strength,redistribute)
            for plane in planes:
                torch.testing.assert_close(shifted[...,plane].sum(-1),p[...,plane].sum(-1),rtol=0,atol=1e-14)
            self.assertTrue(torch.equal(shifted[...,~domain],p[...,~domain]))
            self.assertTrue(torch.equal(shifted[...,~causal],p[...,~causal]))
            torch.testing.assert_close(shifted.sum(-1),p.sum(-1),rtol=0,atol=1e-14)
            if strength == 0: self.assertTrue(torch.equal(shifted,p))
            # Independent conditional-softmax reference within each plane.
            expected = p.clone()
            for plane in planes:
                local = scores[...,plane] + strength*target[plane]
                conditional = torch.nan_to_num(local.softmax(-1),nan=0.)
                expected[...,plane] = conditional*p[...,plane].sum(-1,keepdim=True)
            torch.testing.assert_close(shifted,expected,rtol=0,atol=1e-14)

    def test_actual_controller_prefill_decode_masks_grouped_heads(self):
        torch.manual_seed(18)
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        module = SimpleNamespace(num_key_value_groups=2,is_causal=True)
        cfg = dict(method='binding_transport',task_binding_fraction=.5,
                   task_binding_distribution='uniform',visual_mass_partition='temporal_planes')
        controller = ResearchController([SimpleNamespace(self_attn=module)],cfg)
        try:
            for qn in [11,1]:
                q = torch.randn(2,4,qn,8); k = torch.randn(2,2,11,8); v = torch.randn_like(k)
                visible = torch.ones(qn,11,dtype=torch.bool).tril(11-qn)
                for kind in ['none','boolean','additive']:
                    if kind == 'none': mask = None
                    elif kind == 'boolean': mask = visible[None,None]
                    else: mask = torch.where(visible,0.,torch.finfo(torch.float32).min)[None,None]
                    for method,strength in [('binding_transport',4.),('mass_transport',-4.)]:
                        cfg['method'] = method
                        maps = [mapping(),mapping()]
                        # A batch member may have fewer selected time planes.
                        for field in ['target','negative','alignment']:
                            maps[1][field]['all_frames'] = maps[1][field]['last_frame']
                        controller.steer(maps,[dict(layer=0,head=1)],strength,'all_frames')
                        actual,_ = controller.forward(module,q,k,v,mask)
                        scores = (q @ k.repeat_interleave(2,1).transpose(-1,-2))/(8**.5)
                        scores.masked_fill_(~visible,-torch.inf); p = scores.softmax(-1)
                        for i,m in enumerate(maps):
                            for ids in temporal_domains(m,'all_frames'):
                                plane = torch.tensor([j in ids for j in range(11)])
                                local = scores[i,1,:,plane] + strength*torch.tensor([j in m['target']['all_frames'] for j in ids])
                                mass = p[i,1,:,plane].sum(-1,keepdim=True)
                                p[i,1,:,plane] = torch.nan_to_num(local.softmax(-1),nan=0.)*mass
                            if method == 'binding_transport':
                                domain = torch.tensor([j in m['prompt_text_positions'] for j in range(11)])
                                task = torch.tensor([j in m['task_positions'] for j in range(11)])
                                p[i,1] = uniform_redistribute(p[i,1],domain,task,visible,.5)
                        expected = (p @ v.repeat_interleave(2,1)).transpose(1,2)
                        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_single_plane_controller_exactly_matches_original(self):
        torch.manual_seed(20)
        original = ALL_ATTENTION_FUNCTIONS['sdpa']
        module = SimpleNamespace(num_key_value_groups=2,is_causal=True)
        cfg = dict(method='binding_transport',task_binding_fraction=.5,task_binding_distribution='uniform')
        controller = ResearchController([SimpleNamespace(self_attn=module)],cfg)
        try:
            q = torch.randn(1,4,11,8);k = torch.randn(1,2,11,8);v = torch.randn_like(k)
            for method,strength in [('binding_transport',4.),('mass_transport',-4.)]:
                cfg.update(method=method,visual_mass_partition='global')
                controller.steer([mapping()],[dict(layer=0,head=1)],strength,'last_frame')
                expected,_ = controller.forward(module,q,k,v,None)
                cfg['visual_mass_partition']='temporal_planes'
                controller.steer([mapping()],[dict(layer=0,head=1)],strength,'last_frame')
                actual,_ = controller.forward(module,q,k,v,None)
                self.assertTrue(torch.equal(actual,expected))
        finally:
            ALL_ATTENTION_FUNCTIONS.register('sdpa',original)


if __name__ == '__main__':
    unittest.main()
