"""Auditable pre-softmax key bias using the existing Qwen SDPA implementation.

Unlike a pre-hook that returns when attention_mask is None, this implementation
also handles SDPA's implicit causal mask. Ranking materializes one query row,
not all quadratic attention matrices, and always uses unsteered activations.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class AttentionController:
    def __init__(self, layers):
        self.layer_ids = {id(l.self_attn): i for i,l in enumerate(layers)}
        self.original = ALL_ATTENTION_FUNCTIONS['sdpa']
        self.state = None
        ALL_ATTENTION_FUNCTIONS.register('sdpa', self.forward)

    def clear(self):
        self.state = None

    def rank(self, maps, queries, num_layers=36, num_heads=32):
        self.state = {'kind': 'rank', 'maps': maps, 'queries': queries,
                      'raw': {s: np.zeros((len(maps),num_layers,num_heads)) for s in ['last_frame','all_frames']},
                      'visual': np.zeros((len(maps),num_layers,num_heads)), 'seen': set()}
        return self.state

    def steer(self, maps, heads, bias, scope, region='target'):
        grouped = {}
        for h in heads: grouped.setdefault(int(h['layer']),[]).append(int(h['head']))
        self.state = {'kind': 'steer', 'maps': maps, 'heads': grouped, 'bias': bias,
                      'scope': scope, 'region': region, 'cache': {}, 'diagnostics': {}}
        return self.state

    def forward(self, module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        layer = self.layer_ids.get(id(module))
        state = self.state
        if state is None or layer is None:
            return self.original(module, query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        b,h,q,d = query.shape
        k = key.shape[-2]
        if state['kind'] == 'rank':
            group = h//key.shape[1]
            repeated = key.repeat_interleave(group,dim=1)
            qs = torch.stack([query[i,:,j,:] for i,j in enumerate(state['queries'])])
            scores = torch.einsum('bhd,bhkd->bhk', qs.float(),repeated.float())*(scaling or d**-0.5)
            if attention_mask is not None:
                for i,j in enumerate(state['queries']):
                    mask = attention_mask[i if attention_mask.shape[0]>1 else 0,:,j,:k]
                    if mask.dtype == torch.bool: scores[i].masked_fill_(~mask, -torch.inf)
                    else: scores[i] += mask.float()
            else:
                for i,j in enumerate(state['queries']): scores[i,:,j+1:] = -torch.inf
            weights = scores.softmax(-1)
            for i,m in enumerate(state['maps']):
                for s in state['raw']:
                    state['raw'][s][i,layer] = weights[i,:,m['target'][s]].sum(-1).cpu().numpy()
                state['visual'][i,layer] = weights[i,:,m['visual']].sum(-1).cpu().numpy()
            state['seen'].add(layer)
        elif layer in state['heads']:
            if layer not in state['cache']:
                max_key = max(max(m['visual']) for m in state['maps'])+1
                offset = torch.zeros((b,h,1,max_key),device=query.device,dtype=query.dtype)
                hs = state['heads'][layer]
                for i,m in enumerate(state['maps']):
                    selected = m[state['region']][state['scope']]
                    domain = set(m['target'][state['scope']]) | set(m['negative'][state['scope']])
                    negative = sorted(domain-set(selected))
                    if not selected or set(selected)&set(negative): raise ValueError('Invalid bias domains')
                    for head in hs:
                        offset[i,head,0,selected] = state['bias']
                        offset[i,head,0,negative] = -state['bias']
                state['cache'][layer] = offset
            offset = state['cache'][layer]
            if k > offset.shape[-1]: offset = torch.nn.functional.pad(offset,(0,k-offset.shape[-1]))
            else: offset = offset[...,:k]
            if attention_mask is None:
                if q > 1:
                    positions_q = torch.arange(k-q,k,device=query.device)[:,None]
                    positions_k = torch.arange(k,device=query.device)[None,:]
                    mask = torch.zeros((1,1,q,k),device=query.device,dtype=query.dtype)
                    mask.masked_fill_(positions_k>positions_q, -torch.inf)
                    attention_mask = mask+offset
                else:
                    attention_mask = offset
            elif attention_mask.dtype == torch.bool:
                converted = torch.zeros_like(attention_mask,dtype=query.dtype).masked_fill_(~attention_mask,-torch.inf)
                attention_mask = converted+offset
            else:
                attention_mask = attention_mask[...,:k]+offset
            kwargs['is_causal'] = False
            diag = state['diagnostics'].setdefault(str(layer),{'prefill_calls':0,'decode_calls':0,'query_rows':0,
                          'heads':state['heads'][layer], 'bias':state['bias'], 'all_query_rows':True,
                          'causal_mask_preserved':True, 'generated_text_key_bias':0})
            diag['prefill_calls' if q>1 else 'decode_calls'] += 1
            diag['query_rows'] += q
        return self.original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
