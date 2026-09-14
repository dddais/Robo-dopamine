"""Observe the last prompt query after the experiment's attention controller.

The SDPA output is passed through unchanged. Only selected heads' probability
rows are reconstructed in float32, as in the independent ranking. These are
diagnostic probabilities, not bitwise attention weights from the fused kernel.
"""
from __future__ import annotations

import numpy as np
import torch


def attention_row(module, query, key, mask, heads, query_index, scaling=None, is_causal=None):
    """Reconstruct selected GQA heads with the actual post-bias SDPA mask."""
    batch, count, queries, dim = query.shape
    keys = key.shape[-2]
    if batch != 1 or count % key.shape[1]:
        raise ValueError('Capture requires batch size one and integral GQA groups')
    if not 0 <= query_index < queries or not heads or any(h < 0 or h >= count for h in heads):
        raise ValueError('Invalid query/head selection')
    kv_heads = [h // (count // key.shape[1]) for h in heads]
    q = query[0, heads, query_index].float()
    k = key[0, kv_heads].float()
    scores = torch.einsum('hd,hkd->hk', q, k) * (dim ** -0.5 if scaling is None else scaling)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None]
        if mask.ndim != 4 or mask.shape[0] != 1:
            raise ValueError(f'Unsupported attention mask: {tuple(mask.shape)}')
        mh = [0] * len(heads) if mask.shape[1] == 1 else heads
        mq = 0 if mask.shape[2] == 1 else query_index
        row = mask[0, mh, mq, :keys]
        if row.dtype == torch.bool:
            scores = scores.masked_fill(~row, -torch.inf)
        else:
            scores = scores + row.float()
    if is_causal is None:
        is_causal = queries > 1 and mask is None and getattr(module, 'is_causal', True)
    if is_causal:
        # Match torch SDPA's upper-left causal alignment, including non-square inputs.
        scores[:, query_index + 1:] = -torch.inf
    weights = scores.softmax(-1)
    if not torch.isfinite(weights).all():
        raise ValueError('Captured query has non-finite attention probabilities')
    return weights.detach().cpu().numpy()


class LastPromptCapture:
    """Temporarily wrap controller.original, downstream of its SAS mask edits.

    Runtime.predict still performs its usual greedy generation. Capture only
    the prefill query; subsequent score-token decoding is deliberately ignored.
    The wrapper is process-local and must not be shared by concurrent threads.
    """

    def __init__(self, controller, heads, prompt_length):
        self.controller = controller
        self.heads = [(int(h['layer']), int(h['head'])) for h in heads]
        if not self.heads or len(set(self.heads)) != len(self.heads):
            raise ValueError('Capture heads must be nonempty and unique')
        self.grouped = {}
        for layer, head in self.heads:
            self.grouped.setdefault(layer, []).append(head)
        self.prompt_length = prompt_length
        self.rows = {}
        self.original = None

    def __enter__(self):
        self.original = self.controller.original
        self.controller.original = self.forward
        return self

    def __exit__(self, *exc):
        self.controller.original = self.original

    def forward(self, module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        layer = self.controller.layer_ids.get(id(module))
        if layer in self.grouped and layer not in self.rows:
            if query.shape[-2] != self.prompt_length or key.shape[-2] != self.prompt_length:
                raise ValueError('Expected an uncached, complete prompt prefill')
            if dropout:
                raise ValueError('Attention visualization requires evaluation without dropout')
            self.rows[layer] = attention_row(
                module, query, key, attention_mask, self.grouped[layer], self.prompt_length - 1,
                scaling, kwargs.get('is_causal'))
        return self.original(module, query, key, value, attention_mask,
                             dropout=dropout, scaling=scaling, **kwargs)

    def result(self):
        if set(self.rows) != set(self.grouped):
            raise RuntimeError(f'Missing captured layers: {set(self.grouped) - set(self.rows)}')
        return np.stack([self.rows[layer][self.grouped[layer].index(head)] for layer, head in self.heads])
