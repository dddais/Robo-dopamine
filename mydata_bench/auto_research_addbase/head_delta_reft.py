"""Rank4 linear adaptation of a fixed round32 attention increment; no output-head change."""
import math
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .head_attention_gates import HeadGateController, HeadGateRuntime, gated_probability


def initial_matrices(device='cpu'):
    generator = torch.Generator(device='cpu').manual_seed(20260912)
    A = torch.randn((28,4,128), generator=generator, dtype=torch.float32)/math.sqrt(128)
    B = torch.zeros((28,128,4), dtype=torch.float32)
    return A.to(device), B.to(device)


def increment_correction(delta, A, B):
    return (delta @ A.transpose(-1,-2)) @ B.transpose(-1,-2)


def operator_penalty(A, B):
    return .01*(B@A).square().sum((-2,-1)).mean()/128


class HeadDeltaController(HeadGateController):
    def __init__(self, layers, cfg):
        super().__init__(layers, cfg)
        self.A = None
        self.B = None
        self.capture_probes = False

    def matrices(self, device):
        if (self.A is None or self.B is None or self.A.shape != (28,4,128) or self.B.shape != (28,128,4)
                or self.A.device != device or self.B.device != device
                or self.A.dtype != torch.float32 or self.B.dtype != torch.float32):
            raise ValueError('Explicit rank4 float32 matrices on the actual attention device required')
        return self.A, self.B

    def forward(self, module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        state = self.state; layer = self.layer_ids.get(id(module))
        if state is None or state['kind'] != 'steer' or layer not in state['heads']:
            return super().forward(module, query, key, value, attention_mask,
                                   dropout=dropout, scaling=scaling, **kwargs)
        if (not 8 <= layer <= 35 or dropout or self.cfg.get('method') != 'learned_head_delta_reft'
                or self.cfg.get('contrast_negative_mode') != 'learned_head_gates' or state['bias'] != 6):
            raise ValueError('Require registered rank4 increment adaptation on stage8 heads')
        if self.theta is not None:
            raise ValueError('Round33 gates must stay fixed')
        A, B = self.matrices(query.device)
        base, weights = self.original(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
        b, h, q, d = query.shape; k = key.shape[-2]; selected = state['heads'][layer]
        kv = torch.tensor([j//(h//key.shape[1]) for j in selected], device=query.device)
        kh = key.index_select(1, kv).float(); vh = value.index_select(1, kv).float()
        cache_key = ('learned_gate_domains', k, str(query.device))
        if cache_key not in state['cache']:
            spatial = torch.zeros((b, 1, 1, k), dtype=torch.float32, device=query.device)
            task = torch.zeros_like(spatial)
            for i, mapping in enumerate(state['maps']):
                chosen = set(mapping[state['region']][state['scope']])
                domain = set(mapping['target'][state['scope']]) | set(mapping['negative'][state['scope']])
                tasks = set(mapping['task_positions'])
                if not chosen or not chosen <= domain or tasks & domain:
                    raise ValueError('Invalid disjoint visual and instruction key domains')
                spatial[i, 0, 0, sorted(chosen)] = 1
                spatial[i, 0, 0, sorted(domain-chosen)] = -1
                task[i, 0, 0, sorted(tasks)] = 1
            state['cache'][cache_key] = spatial, task
        spatial, task = state['cache'][cache_key]
        gates = self.values(query.device)[layer-8, selected]
        result = base.clone(); probe = None
        for start in range(0, q, 128):
            stop = min(start+128, q)
            scores = query[:, selected, start:stop].float() @ kh.transpose(-1, -2)*(scaling or d**-.5)
            if attention_mask is None:
                if q > 1:
                    visible = torch.arange(k, device=query.device)[None, :] <= torch.arange(k-q+start, k-q+stop, device=query.device)[:, None]
                    scores = scores.masked_fill(~visible, -torch.inf)
            else:
                mask = attention_mask[..., :k]
                if mask.shape[1] > 1: mask = mask[:, selected]
                if mask.shape[-2] > 1: mask = mask[..., start:stop, :]
                scores = scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores+mask.float()
            original = torch.nan_to_num(scores.softmax(-1), nan=0.)
            shifted = gated_probability(scores, spatial, task, gates)
            raw = (shifted-original) @ vh
            correction = increment_correction(raw, A[layer-8], B[layer-8])
            delta = raw.transpose(1, 2).to(base.dtype)
            corrected = correction.transpose(1, 2).to(base.dtype)
            # Preserve round32's first BF16 addition exactly when B is zero.
            result[:, start:stop, selected, :] = (result[:, start:stop, selected, :]+delta)+corrected
            if self.capture_probes:
                probe = dict(raw_delta=raw[0,0,-1].detach().float().cpu().tolist(),
                    correction=correction[0,0,-1].detach().float().cpu().tolist())
        diag = state['diagnostics'].setdefault(str(layer), dict(method='learned_head_delta_reft', heads=selected,
            visual_cap=6., task_cap=4., all_query_rows=True, causal_mask_preserved=True,
            modality_mass_preserved=False, prefill_calls=0, decode_calls=0,
            gates=gates.detach().float().cpu().tolist(), zero_gate_probe=self.zero_probe,
            global_parameters_shared_across_inputs=True, head_specific_gates=True, selected_head_difference_from_original_sdpa=True))
        diag.update(adapter_rank=4, adapter_parameter_count=28672,
            layer_shared_across_heads=True, no_constant_term=True,
            A_l2=float(A[layer-8].detach().norm()), B_l2=float(B[layer-8].detach().norm()),
            separate_native_dtype_additions=True)
        if probe is not None: diag['increment_probe'] = probe
        diag['prefill_calls' if q > 1 else 'decode_calls'] += 1
        return result, weights


class HeadDeltaRuntime(HeadGateRuntime):
    def __init__(self, cfg):
        super().__init__(cfg)
        ALL_ATTENTION_FUNCTIONS.register('sdpa', self.controller.original)
        self.controller = HeadDeltaController(self.layers, self.cfg)

    def predict(self, samples, condition, rankings=None, step=None, previous=None):
        if self.cfg.get('method') != 'learned_head_delta_reft':
            raise ValueError('Explicit round33 increment-adaptation method required')
        rows = super().predict(samples, condition, rankings, step, previous)
        for row in rows:
            row.update(readout='five_way_answer_likelihood_attention_increment_reft',
                adapter_rank=4, adapter_parameter_count=28672, fixed_gate_parameter_count=1792,
                learned_adapter_file=self.cfg.get('learned_adapter_file'),
                learned_adapter_sha256=self.cfg.get('learned_adapter_sha256'),
                adapter_mode=self.cfg.get('adapter_mode','learned'), no_constant_adapter_term=True)
        return rows
