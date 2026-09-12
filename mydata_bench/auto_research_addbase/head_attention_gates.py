"""Frozen-backbone attention with1792 input-independent, per-head sigmoid gates."""
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController
from .evidence import EvidenceRuntime


def gated_probability(scores, spatial_sign, task_keys, gates):
    """Masking is already in scores; bounded finite offsets cannot unmask keys."""
    offset = 6*gates[None, :, 0, None, None]*spatial_sign+4*gates[None, :, 1, None, None]*task_keys
    return torch.nan_to_num((scores+offset).softmax(-1), nan=0.)


class HeadGateController(ResearchController):
    def __init__(self, layers, cfg):
        super().__init__(layers, cfg)
        self.theta = None
        self.fixed_values = None
        self.zero_probe = False

    def values(self, device):
        if self.zero_probe:
            return torch.zeros((28, 32, 2), dtype=torch.float32, device=device)
        if self.theta is not None:
            return self.theta.sigmoid()
        if self.fixed_values is None:
            raise ValueError('Explicit fixed trained gates or training parameters are required')
        result = torch.as_tensor(self.fixed_values, dtype=torch.float32, device=device)
        if result.shape != (28, 32, 2) or not torch.isfinite(result).all() or (result < 0).any() or (result > 1).any():
            raise ValueError('Require exactly 1792 bounded finite global gates')
        return result

    def forward(self, module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        state = self.state; layer = self.layer_ids.get(id(module))
        if state is None or state['kind'] != 'steer' or layer not in state['heads']:
            return super().forward(module, query, key, value, attention_mask,
                                   dropout=dropout, scaling=scaling, **kwargs)
        if (not 8 <= layer <= 35 or dropout or self.cfg.get('method') != 'learned_head_gate_bias'
                or self.cfg.get('contrast_negative_mode') != 'learned_head_gates' or state['bias'] != 6):
            raise ValueError('Require registered head-specific gated bias on stage8 layers')
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
        result = base.clone()
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
            delta = ((shifted-original) @ vh).transpose(1, 2).to(base.dtype)
            result[:, start:stop, selected, :] = result[:, start:stop, selected, :]+delta
        diag = state['diagnostics'].setdefault(str(layer), dict(method='learned_head_gate_bias', heads=selected,
            visual_cap=6., task_cap=4., all_query_rows=True, causal_mask_preserved=True,
            modality_mass_preserved=False, prefill_calls=0, decode_calls=0,
            gates=gates.detach().float().cpu().tolist(), zero_gate_probe=self.zero_probe,
            global_parameters_shared_across_inputs=True, head_specific_gates=True, selected_head_difference_from_original_sdpa=True))
        diag['prefill_calls' if q > 1 else 'decode_calls'] += 1
        return result, weights


class HeadGateRuntime(EvidenceRuntime):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model.requires_grad_(False)
        self.model.eval()
        ALL_ATTENTION_FUNCTIONS.register('sdpa', self.controller.original)
        self.controller = HeadGateController(self.layers, self.cfg)

    def predict(self, samples, condition, rankings=None, step=None, previous=None):
        if (self.cfg.get('contrast_negative_mode') != 'learned_head_gates' or self.cfg.get('contrast_weight') != 0
                or self.cfg['model'] not in {'qwen', 'roboreward'} or any(p.requires_grad for p in self.model.parameters())):
            raise ValueError('Require frozen native model and no final-logit contrast')
        rows = super().predict(samples, condition, rankings, step, previous)
        for row in rows:
            row.update(readout='five_way_answer_likelihood_head_attention_gates',
                contrast_negative_mode='learned_head_gates', actual_forward_branches=1,
                learned_gate_file=self.cfg.get('learned_gate_file'),
                learned_gate_sha256=self.cfg.get('learned_gate_sha256'),
                gate_parameter_count=1792, backbone_frozen=True)
        return rows

    def differentiable_logits(self, samples, condition, rankings):
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Backbone weights must never enter the optimizer')
        self.active_ranking_prefix = 'ANSWER: '
        try:
            inputs, maps, queries, texts = self.prepare(samples)
        finally:
            self.active_ranking_prefix = ''
        scope, kind, k = condition.split(':'); k = int(k)
        if kind != 'target': raise ValueError('Training/probe uses only original target')
        heads = rankings[scope]['ranking'][:k]
        if len({(h['layer'], h['head']) for h in heads}) != k: raise ValueError('Wrong training top-k')
        class_ids = [self.processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
        if any(len(x) != 1 for x in class_ids): raise ValueError('All five native single-token classes required')
        state = self.controller.steer(maps, heads, 6., scope)
        try:
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, [x[0] for x in class_ids]].float()
            return logits, state['diagnostics'], [self.audit(m) for m in maps]
        finally:
            self.controller.clear()
