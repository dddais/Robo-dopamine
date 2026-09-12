from types import SimpleNamespace
import unittest

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .head_attention_gates import HeadGateController
from .head_delta_reft import HeadDeltaController, initial_matrices, increment_correction, operator_penalty
from .test_task_domain import mapping


class HeadDeltaTests(unittest.TestCase):
    def test_rank4_linear_formula_zero_increment_and_sample_independence(self):
        torch.manual_seed(3301)
        d=torch.randn(3,2,5,128);A=torch.randn(4,128);B=torch.randn(128,4)
        actual=increment_correction(d,A,B)
        torch.testing.assert_close(actual,d@(B@A).T,rtol=1e-4,atol=1e-4)
        self.assertTrue(torch.equal(increment_correction(torch.zeros_like(d),A,B),torch.zeros_like(d)))
        self.assertTrue(torch.equal(increment_correction(d,A,B*0),torch.zeros_like(d)))
        for i in range(3):torch.testing.assert_close(increment_correction(d[i:i+1],A,B),actual[i:i+1])
        torch.testing.assert_close(increment_correction(2*d,A,B),2*actual)

    def test_matrix_gradient_finite_differences_and_initial_zero_A_gradient(self):
        torch.manual_seed(3302)
        d=torch.randn(2,3,8,dtype=torch.float64,requires_grad=True)
        A=torch.randn(2,8,dtype=torch.float64,requires_grad=True)
        B=torch.randn(8,2,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(increment_correction,(d,A,B),eps=1e-6,atol=1e-5,rtol=1e-4))
        B0=torch.zeros_like(B,requires_grad=True)
        (increment_correction(d,A,B0).sum()).backward()
        self.assertEqual(float(A.grad.abs().sum()),0)
        self.assertGreater(float(B0.grad.abs().sum()),0)

    def test_fixed_seed_initialization_and_parameter_count(self):
        A,B=initial_matrices();C,D=initial_matrices()
        self.assertTrue(torch.equal(A,C));self.assertTrue(torch.equal(B,D))
        self.assertEqual(A.numel()+B.numel(),28672)
        self.assertEqual(float(B.abs().sum()),0)
        self.assertEqual(float(operator_penalty(A,B)),0)
        self.assertGreater(float(operator_penalty(A,B+1)),0)

    def controllers(self):
        original=ALL_ATTENTION_FUNCTIONS['sdpa']
        modules=[SimpleNamespace(num_key_value_groups=2,is_causal=True) for _ in range(9)]
        layers=[SimpleNamespace(self_attn=m) for m in modules]
        old=HeadGateController(layers,dict(method='learned_head_gate_bias',contrast_negative_mode='learned_head_gates'))
        # Each controller installs a global hook; both references must start from native SDPA.
        ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
        new=HeadDeltaController(layers,dict(method='learned_head_delta_reft',contrast_negative_mode='learned_head_gates'))
        values=torch.sigmoid(torch.randn(28,32,2)).tolist();old.fixed_values=values;new.fixed_values=values
        new.A,new.B=initial_matrices();return original,modules[8],old,new

    def test_zero_adapter_exact_round32_replay_gqa_masks_prefill_decode(self):
        torch.manual_seed(3303)
        original,module,old,new=self.controllers()
        try:
            for dtype in [torch.float32,torch.bfloat16]:
                for qn in [8,1]:
                    q=torch.randn(2,4,qn,128).to(dtype);k=torch.randn(2,2,8,128).to(dtype);v=torch.randn_like(k)
                    visible=(torch.arange(8)[None,:]<=torch.arange(8-qn,8)[:,None]).expand(2,1,qn,8).clone()
                    visible[0,:,:,4]=False
                    for mask in [visible,torch.zeros_like(visible,dtype=torch.float32).masked_fill(~visible,-torch.inf)]:
                        old.steer([mapping(),mapping()],[dict(layer=8,head=1),dict(layer=8,head=3)],6.,'all_frames')
                        new.steer([mapping(),mapping()],[dict(layer=8,head=1),dict(layer=8,head=3)],6.,'all_frames')
                        a,_=old.forward(module,q,k,v,mask);b,_=new.forward(module,q,k,v,mask)
                        self.assertTrue(torch.equal(a,b))
                        base,_=original(module,q,k,v,mask)
                        self.assertTrue(torch.equal(b[:,:,[0,2]],base[:,:,[0,2]]))
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_nonzero_adapter_zero_gates_exact_baseline_and_masked_key_invariance(self):
        torch.manual_seed(3304)
        original,module,old,new=self.controllers()
        try:
            new.B=torch.randn_like(new.B)*.01
            q=torch.randn(1,4,8,128);k=torch.randn(1,2,8,128);v=torch.randn_like(k)
            mask=(torch.arange(8)[None,:]<=torch.arange(8)[:,None])[None,None];mask[...,4]=False
            def run(keys,values):
                new.steer([mapping()],[dict(layer=8,head=1)],6.,'all_frames')
                return new.forward(module,q,keys,values,mask)[0]
            new.zero_probe=True
            self.assertTrue(torch.equal(run(k,v),original(module,q,k,v,mask)[0]))
            new.zero_probe=False;a=run(k,v);changed_k=k.clone();changed_v=v.clone()
            changed_k[:,:,4]=999;changed_v[:,:,4]=-999
            torch.testing.assert_close(run(changed_k,changed_v),a)
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_controller_gradient_support_and_reconstructed_increment(self):
        torch.manual_seed(3305)
        original,module,old,new=self.controllers()
        try:
            new.A=torch.nn.Parameter(new.A);new.B=torch.nn.Parameter(torch.randn_like(new.B)*.001)
            new.capture_probes=True
            q=torch.randn(1,4,8,128);k=torch.randn(1,2,8,128);v=torch.randn_like(k)
            state=new.steer([mapping()],[dict(layer=8,head=1)],6.,'all_frames')
            result,_=new.forward(module,q,k,v,None)
            result.square().mean().backward()
            for parameter in [new.A,new.B]:
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad[0].abs().sum()),0)
                self.assertEqual(float(parameter.grad[1:].abs().sum()),0)
            probe=state['diagnostics']['8']['increment_probe']
            expected=increment_correction(torch.tensor(probe['raw_delta']),new.A[0].detach(),new.B[0].detach())
            torch.testing.assert_close(expected,torch.tensor(probe['correction']),atol=1e-6,rtol=1e-5)
            self.assertIsNone(new.theta)
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)


if __name__=='__main__':unittest.main()
