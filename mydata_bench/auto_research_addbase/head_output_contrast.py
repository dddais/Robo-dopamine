"""Sample-local head output contrast at fixed QKV, before downstream nonlinearities."""
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController
from .evidence import EvidenceRuntime


def compose(original, positive, visual_negative, task_negative):
    if any(x.shape != original.shape for x in [positive, visual_negative, task_negative]):
        raise ValueError('Local attention output shapes must match')
    if not original.is_floating_point() or any(not torch.isfinite(x).all() for x in [original, positive, visual_negative, task_negative]):
        raise ValueError('Finite floating point head outputs required')
    base = original.float()
    direction = positive.float() - .5*(visual_negative.float()+task_negative.float())
    candidate = base+direction
    base_norm = torch.linalg.vector_norm(base, dim=-1, keepdim=True)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
    normalized = candidate*(base_norm/candidate_norm.clamp_min(torch.finfo(base.dtype).tiny))
    unchanged = (base_norm == 0) | (candidate_norm == 0) | (direction == 0).all(-1, keepdim=True)
    return torch.where(unchanged, base, normalized).to(original.dtype)


def norm_error(original, output):
    a = torch.linalg.vector_norm(original.float(), dim=-1)
    b = torch.linalg.vector_norm(output.float(), dim=-1)
    return float(((a-b).abs()/a.clamp_min(torch.finfo(a.dtype).tiny)).max())


class HeadOutputController(ResearchController):
    def forward(self, module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        outer = self.state; layer = self.layer_ids.get(id(module))
        if outer is None or outer['kind'] != 'steer' or layer not in outer['heads']:
            return super().forward(module, query, key, value, attention_mask,
                                   dropout=dropout, scaling=scaling, **kwargs)
        cfg = self.cfg
        if (cfg.get('contrast_negative_mode') != 'head_output' or cfg.get('method') != 'binding_transport'
                or cfg.get('task_binding_fraction') != .5 or cfg.get('task_binding_distribution') != 'uniform'
                or outer['bias'] != 4 or cfg.get('negative_task_strength') != 4
                or cfg.get('research_query_scope', 'all') != 'all' or cfg.get('visual_mass_partition', 'global') != 'global'
                or dropout):
            raise ValueError('Require the frozen one-forward local head output design')
        base, weights = self.original(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
        outputs = {}; diagnostics = {}
        try:
            for name, method, strength in [('positive', 'binding_transport', 4.),
                    ('visual_negative', 'mass_transport', -4.), ('task_negative', 'binding_task_suppression', 4.)]:
                cache_key = ('head_output_local_state', name)
                if cache_key not in outer['cache']:
                    outer['cache'][cache_key] = dict(outer, bias=strength, cache={}, diagnostics={})
                local = outer['cache'][cache_key]
                self.state = local
                self.cfg = dict(cfg, method=method, contrast_negative_mode='visual')
                outputs[name], _ = super().forward(module, query, key, value, attention_mask,
                                                  dropout=dropout, scaling=scaling, **kwargs)
                diagnostics[name] = dict(local['diagnostics'][str(layer)])
        finally:
            self.cfg = cfg
            self.state = outer
        selected = outer['heads'][layer]
        raw = base[:, :, selected, :]
        local_parts = [outputs[n][:, :, selected, :] for n in ['positive', 'visual_negative', 'task_negative']]
        updated = compose(raw, *local_parts)
        error = norm_error(raw, updated)
        if error > .01:
            raise ValueError('Actual selected head norm exceeds preregistered dtype tolerance')
        result = base.clone()
        result[:, :, selected, :] = updated
        unselected = [h for h in range(base.shape[2]) if h not in selected]
        if unselected and not torch.equal(result[:, :, unselected, :], base[:, :, unselected, :]):
            raise ValueError('Unselected local head outputs changed')
        q_indices = sorted({0, base.shape[1]-1})
        # First selected head at first/last actual query, every batch row.
        probes = {name: tensor[:, q_indices, 0, :].detach().float().cpu().tolist()
                  for name, tensor in zip(['original', 'positive', 'visual_negative', 'task_negative', 'output'],
                                          [raw, *local_parts, updated])}
        diag = outer['diagnostics'].setdefault(str(layer), dict(method='head_output_contrast_norm',
            heads=selected, strength=4., gain=1., all_query_rows=True, causal_mask_preserved=True,
            actual_full_model_forwards=1, local_attention_evaluations=4, local_QKV_shared=True,
            domain_mass_preserved=False, signed_output_intervention=True,
            norm_location='per selected head, before output projection', prefill_calls=0, decode_calls=0,
            max_relative_norm_error=0., unselected_head_outputs_exact=True, probes=[]))
        diag['prefill_calls' if query.shape[2] > 1 else 'decode_calls'] += 1
        diag['max_relative_norm_error'] = max(diag['max_relative_norm_error'], error)
        diag['local_branches'] = diagnostics
        diag['probes'].append(dict(query_indices=q_indices, selected_head=selected[0],
                                   output_dtype=str(base.dtype), vectors=probes))
        return result, weights


class HeadOutputRuntime(EvidenceRuntime):
    def __init__(self, cfg):
        super().__init__(cfg)
        ALL_ATTENTION_FUNCTIONS.register('sdpa', self.controller.original)
        self.controller = HeadOutputController(self.layers, self.cfg)

    def predict(self, samples, condition, rankings=None, step=None, previous=None):
        if (self.cfg.get('contrast_negative_mode') != 'head_output' or self.cfg.get('contrast_weight') != 0
                or self.cfg['model'] not in {'qwen', 'roboreward'}):
            raise ValueError('No final-logit contrast is allowed for the local head output candidate')
        rows = super().predict(samples, condition, rankings, step, previous)
        for row in rows:
            row.update(readout='five_way_answer_likelihood_head_output_contrast',
                       contrast_negative_mode='head_output', actual_forward_branches=1,
                       local_composition='norm(o0)*(o0+opp-(ovn+otn)/2)/norm(o0+opp-(ovn+otn)/2)',
                       head_output_gain=1., norm_relative_tolerance=.01)
        return rows
