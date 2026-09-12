"""Fixed-layout instruction isolation; no ROI is used in the negative branch."""
import hashlib
import time

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention import ResearchController
from .runtime import ResearchRuntime


def block_mask(query, key, attention_mask, task_positions, is_causal):
    b, h, q, _ = query.shape
    k = key.shape[-2]
    if len(task_positions) != b or key.shape[0] != b:
        raise ValueError('Task masks must match each batch row')
    blocked = torch.zeros((b, 1, 1, k), dtype=torch.bool, device=query.device)
    for i, positions in enumerate(task_positions):
        if len(set(positions)) != len(positions) or any(p < 0 or p >= k for p in positions):
            raise ValueError('Task keys must be distinct positions in the current key sequence')
        blocked[i, 0, 0, positions] = True
    if attention_mask is None:
        result = torch.zeros((1, 1, q, k), dtype=query.dtype, device=query.device)
    else:
        m = attention_mask[..., :k]
        if (m.ndim != 4 or m.shape[0] not in [1, b] or m.shape[1] not in [1, h]
                or m.shape[2] not in [1, q] or m.shape[3] != k):
            raise ValueError('Require the original broadcastable four-dimensional SDPA mask')
        if m.dtype == torch.bool:
            result = torch.zeros_like(m, dtype=query.dtype).masked_fill(~m, -torch.inf)
        else:
            if not m.is_floating_point() or torch.isnan(m).any() or torch.isposinf(m).any():
                raise ValueError('Invalid original additive mask')
            # Transformers uses both -inf and finfo.min for invisible keys.
            # Canonicalize fully masked rows so SDPA returns exactly zero.
            result = m.masked_fill(m <= torch.finfo(m.dtype).min, -torch.inf)
    if is_causal:
        # Exactly SDPA's upper-left causal relation; prefill has q == k.
        allowed = torch.arange(k, device=query.device)[None, :] <= torch.arange(q, device=query.device)[:, None]
        result = result.masked_fill(~allowed, -torch.inf)
    return result.masked_fill(blocked, -torch.inf)


class TaskContentController(ResearchController):
    def block_task(self, maps):
        if any('task_positions' not in m for m in maps):
            raise ValueError('Explicit instruction alignment is required')
        self.state = dict(kind='block_task', maps=maps, diagnostics={})
        return self.state

    def forward(self, module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        state = self.state
        layer = self.layer_ids.get(id(module))
        if state is None or state['kind'] != 'block_task' or layer is None:
            return super().forward(module, query, key, value, attention_mask,
                                   dropout=dropout, scaling=scaling, **kwargs)
        if dropout:
            raise ValueError('Task-content isolation is deterministic inference only')
        causal = kwargs.get('is_causal')
        if causal is None:
            causal = query.shape[2] > 1 and attention_mask is None and getattr(module, 'is_causal', True)
        positions = [m['task_positions'] for m in state['maps']]
        mask = block_mask(query, key, attention_mask, positions, causal)
        kwargs['is_causal'] = False
        result = self.original(module, query, key, value, mask, dropout=dropout, scaling=scaling, **kwargs)
        d = state['diagnostics'].setdefault(str(layer), dict(method='task_key_block',
            heads=list(range(query.shape[1])), all_query_rows=True, causal_mask_preserved=True,
            roi_intervention=False, task_token_counts=list(map(len, positions)), prefill_calls=0, decode_calls=0,
            original_mask_kind='implicit' if attention_mask is None else str(attention_mask.dtype),
            fully_masked_rows_return_zero=True))
        d['prefill_calls' if query.shape[2] > 1 else 'decode_calls'] += 1
        return result


def compose(positive, blocked):
    if positive.shape != blocked.shape or positive.shape[-1] != 5:
        raise ValueError('Require matching unrestricted five-class native logits')
    return positive + (positive - blocked)


def tensor_fingerprint(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [tensor_fingerprint(v) for v in value]
    if not torch.is_tensor(value):
        raise ValueError('Expected an actual model tensor')
    a = value.detach().contiguous().cpu()
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def perturb_task_tokens(inputs, maps, tokenizer):
    candidates = [tokenizer.encode(s, add_special_tokens=False) for s in ['x', 'y']]
    if any(len(t) != 1 for t in candidates) or candidates[0] == candidates[1]:
        raise ValueError('Probe needs two distinct single ordinary tokens')
    tokens = [c[0] for c in candidates]
    special = set(tokenizer.all_special_ids)
    if special.intersection(tokens):
        raise ValueError('Probe replacement must not introduce special tokens')
    changed = dict(inputs)
    changed['input_ids'] = inputs['input_ids'].clone()
    counts = []
    for i, m in enumerate(maps):
        positions = m['task_positions']
        if not positions or set(positions).intersection(m['visual']) or m['query'] in positions:
            raise ValueError('Probe must only replace instruction text outside the visual/readout tokens')
        old = changed['input_ids'][i, positions]
        if special.intersection(old.tolist()):
            raise ValueError('Instruction unexpectedly contains special tokens')
        replacement = torch.where(old == tokens[0], tokens[1], tokens[0])
        changed['input_ids'][i, positions] = replacement
        if not torch.all(old != replacement):
            raise ValueError('Every instruction token must change')
        counts.append(len(positions))
    return changed, dict(replacement_token_ids=tokens, changed_token_counts=counts,
        original_input_ids=tensor_fingerprint(inputs['input_ids']),
        perturbed_input_ids=tensor_fingerprint(changed['input_ids']),
        unchanged_other_inputs={k: tensor_fingerprint(v) for k, v in inputs.items() if k != 'input_ids'})


class TaskContentRuntime(ResearchRuntime):
    def __init__(self, cfg):
        super().__init__(cfg)
        ALL_ATTENTION_FUNCTIONS.register('sdpa', self.controller.original)
        self.controller = TaskContentController(self.layers, self.cfg)

    def predict(self, samples, condition, rankings=None, step=None, previous=None):
        cfg = self.cfg
        if (cfg['model'] not in {'qwen', 'roboreward'} or cfg.get('contrast_negative_mode') != 'task_content_block'
                or cfg.get('contrast_weight') != 1 or not cfg.get('require_task_positions')
                or cfg.get('task_binding_fraction') is not None or cfg.get('contrast_kl_budget') is not None
                or cfg.get('method') != 'bias' or (condition != 'baseline' and cfg['bias'] != 6)):
            raise ValueError('Require the frozen original-bias/task-content-block design')
        self.active_ranking_prefix = 'ANSWER: '
        try:
            inputs, maps, queries, texts = self.prepare(samples, step, previous)
        finally:
            self.active_ranking_prefix = ''
        for m in maps:
            if (not m['task_positions'] or m['query'] in m['task_positions']
                    or m['task_span_audit']['literal_task_matches'] != 1):
                raise ValueError('Require complete single literal instruction outside the readout')
        heads = None; scope = None; region = 'target'
        if condition != 'baseline':
            scope, kind, count = condition.split(':'); count = int(count)
            if kind not in {'target', 'wrong_region', 'low_rank'} or count <= 0:
                raise ValueError('Unknown task-content condition')
            ranking = rankings[scope]['ranking']
            heads = ranking[-count:] if kind == 'low_rank' else ranking[:count]
            if len({(h['layer'], h['head']) for h in heads}) != count:
                raise ValueError('Require exactly k unique frozen positive heads')
            region = 'wrong' if kind == 'wrong_region' else 'target'
            if region == 'wrong' and any(not m['wrong'][scope] for m in maps):
                raise ValueError('Wrong-region control unavailable: insufficient disjoint cells')
        ids = [self.processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
        if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 5:
            raise ValueError('Require all five distinct native reward tokens')
        class_ids = [x[0] for x in ids]
        probe_enabled = bool(cfg.get('content_block_probe')) and condition != 'baseline'

        def branch(kind, branch_inputs, observe=False):
            state = None; hook = None; observations = []
            def capture(module, args, kwargs):
                observations.append({k: tensor_fingerprint(kwargs.get(k)) for k in
                    ['position_ids', 'position_embeddings', 'attention_mask', 'cache_position']})
            try:
                if kind == 'positive':
                    state = self.controller.steer(maps, heads, 6., scope, region)
                elif kind == 'blocked':
                    state = self.controller.block_task(maps)
                if observe:
                    hook = self.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
                with torch.inference_mode():
                    z = self.model(**branch_inputs, use_cache=False, logits_to_keep=1).logits[:, -1, class_ids].float()
                d = state['diagnostics'] if state else {}
                if kind == 'blocked' and set(d) != {str(i) for i in range(len(self.layers))}:
                    raise ValueError('Every actual language layer must execute instruction blocking')
                if observe and (len(observations) != 1 or observations[0]['position_embeddings'] is None):
                    raise ValueError('Probe must observe the actual rotary position tensors exactly once')
                return z, d, observations
            finally:
                if hook is not None:
                    hook.remove()
                self.controller.clear()

        started = time.monotonic(); probe = None; blocked = None; bd = {}; pd = {}
        if condition == 'baseline':
            positive, _, _ = branch(None, inputs)
            combined = positive
        else:
            positive, pd, _ = branch('positive', inputs)
            blocked, bd, observation = branch('blocked', inputs, probe_enabled)
            combined = compose(positive, blocked)
            if probe_enabled:
                changed, audit = perturb_task_tokens(inputs, maps, self.processor.tokenizer)
                probe_z, probe_diag, probe_obs = branch('blocked', changed, True)
                if not torch.equal(blocked, probe_z) or observation != probe_obs or bd != probe_diag:
                    raise ValueError('Actual task-content invariance or fixed position/mask probe failed')
                probe = dict(audit, actual_layout_observation=observation, negative_logits_exact=True,
                    original_masks_and_rotary_positions_exact=True, negative_head_diagnostics=probe_diag,
                    native_probe_logits=probe_z.cpu().tolist(), probe_forward_branches=1)
        probabilities = combined.softmax(-1)
        rows = []
        for i, sample in enumerate(samples):
            reward = int(probabilities[i].argmax()) + 1
            rows.append(dict(example_id=sample['example_id'], condition=condition, status='ok',
                readout='five_way_answer_likelihood_task_content_block', contrast_negative_mode='task_content_block',
                contrast_weight=0 if blocked is None else 1, actual_forward_branches=1 if blocked is None else 2,
                actual_probe_forward_branches=int(probe_enabled),
                native_class_logits_positive=positive[i].cpu().tolist(),
                native_class_logits_blocked=blocked[i].cpu().tolist() if blocked is not None else None,
                native_class_logits_combined=combined[i].cpu().tolist(),
                native_class_probabilities=probabilities[i].cpu().tolist(),
                attention_diagnostics=pd, blocked_attention_diagnostics=bd,
                invariance_probe=dict(probe, batch_row=i) if probe else None,
                prompt=texts[i], token_audit=self.audit(maps[i]), candidate_token_ids=class_ids,
                reward=reward, progress=(reward - 1)/4, raw_output=None,
                duration_seconds_per_batch=time.monotonic()-started))
        return rows
