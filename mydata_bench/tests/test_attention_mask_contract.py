"""Test the mydata runtime hook; rewardbench owns a separate implementation."""
import pytest
import torch

from mydata_bench.attention_eval.masking import make_attention_mask_hook


def test_hook_rejects_masks_that_would_silently_disable_or_invert_steering():
    hook = make_attention_mask_hook([1], [2], [3], 3, 6)
    for mask in [None, torch.ones(1,1,4,6,dtype=torch.bool), torch.zeros(1,6)]:
        with pytest.raises(RuntimeError):
            hook(None, (), {'attention_mask': mask})
    with pytest.raises(ValueError,match='disjoint'):
        make_attention_mask_hook([1], [2], [2,3], 3, 6)


def test_hook_matches_independent_causal_attention_and_preserves_text_keys():
    torch.manual_seed(917)
    q,k,v = [torch.randn(1,3,7,4) for _ in range(3)]
    mask = torch.zeros(1,1,7,7).masked_fill(~torch.ones(7,7,dtype=torch.bool).tril(),-torch.inf)
    for strength in [0.,6.]:
        hook=make_attention_mask_hook([1],[2],[3],3,strength)
        for start,end in [(0,5),(5,6),(6,7)]:
            causal=mask[:,:,start:end,:end]
            _,kwargs=hook(None,(),{'attention_mask':causal})
            logits=q[:,:,start:end] @ k[:,:,:end].transpose(-1,-2) / 2
            expected=logits+causal
            expected[:,1,:,2]+=strength
            expected[:,1,:,3]-=strength
            torch.testing.assert_close((logits+kwargs['attention_mask']).softmax(-1) @ v[:,:,:end],
                                       expected.softmax(-1) @ v[:,:,:end],rtol=0,atol=0)
            assert torch.equal(kwargs['attention_mask'][...,4:],causal[...,4:].expand(1,3,end-start,end-4))


def test_declared_query_scope_applies_only_to_its_rows():
    for scope in ['all','prefill','last_prompt','decode']:
        diag={};hook=make_attention_mask_hook([1],[2],[3],3,6,diag,query_scope=scope)
        prefill=hook(None,(),{'attention_mask':torch.zeros(1,1,5,5)})
        decode=hook(None,(),{'attention_mask':torch.zeros(1,1,1,6)})
        assert (prefill is None)==(scope=='decode')
        assert (decode is None)==(scope in {'prefill','last_prompt'})
        if scope=='last_prompt':
            assert prefill[1]['attention_mask'][...,:-1,:].count_nonzero()==0
        assert diag['calls']==2
