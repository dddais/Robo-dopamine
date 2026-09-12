"""Two full-frame resolutions, the same original attention plan, five native classes."""
import hashlib
import json
import time

import torch

from .runtime import ResearchRuntime


LOW_PIXELS = 50176
HIGH_PIXELS = 200704


def compose(high, low):
    if high.shape != low.shape or high.shape[-1] != 5:
        raise ValueError('Require the same complete native five-class space in both views')
    return high+(high-low)


def validate_views(low, high):
    if len(low) != len(high): raise ValueError('View batch counts differ')
    for a, b in zip(low, high):
        if (a['nonvisual_input_ids_sha256'] != b['nonvisual_input_ids_sha256']
                or a['prompt_sha256'] != b['prompt_sha256'] or a['resolution_sources'] != b['resolution_sources']
                or len(b['visual']) != 4*len(a['visual'])
                or a['input_ids_sha256'] == b['input_ids_sha256']):
            raise ValueError('Require identical nonvisual content and full-frame sampling with fourfold actual visual keys')
        if any(r['size'] != [640, 480] or r['mosaic'] for r in a['resolution_sources']):
            raise ValueError('Resolution registration covers the audited original 640x480 full frames only')
        for scope in ['all_frames', 'last_frame']:
            x, y = a['alignment'][scope], b['alignment'][scope]
            if len(x) != len(y): raise ValueError('View span count differs')
            for l, h in zip(x, y):
                if (any(l[k] != h[k] for k in ['source_frames', 'tracking_frames', 'boxes', 'mosaic'])
                        or h['grid_thw'] != [l['grid_thw'][0], 2*l['grid_thw'][1], 2*l['grid_thw'][2]]):
                    raise ValueError('Actual temporal sampling, full-frame boxes or spatial grid scaling differs')


class ResolutionRuntime(ResearchRuntime):
    def prepare_view(self, samples, pixels, step, previous):
        old = self.cfg['max_pixels']
        self.active_ranking_prefix = 'ANSWER: '
        try:
            self.cfg['max_pixels'] = pixels
            inputs, maps, queries, texts = self.prepare(samples, step, previous)
        finally:
            self.cfg['max_pixels'] = old
            self.active_ranking_prefix = ''
        for i, m in enumerate(maps):
            ids = inputs['input_ids'][i].tolist(); valid = inputs['attention_mask'][i].tolist()
            visual = set(m['visual'])
            nonvisual = [token for j, token in enumerate(ids) if valid[j] and j not in visual]
            m['nonvisual_input_ids_sha256'] = hashlib.sha256(json.dumps(nonvisual).encode()).hexdigest()
            m['resolution_sources'] = [{k: r[k] for k in ['size', 'sources', 'mosaic']} for r in m['records']]
            m['resolution_max_pixels'] = pixels
        return inputs, maps, queries, texts

    def predict(self, samples, condition, rankings=None, step=None, previous=None):
        cfg = self.cfg
        if (cfg['model'] not in {'qwen', 'roboreward'} or cfg.get('contrast_negative_mode') != 'resolution'
                or cfg.get('contrast_weight') != 1 or cfg.get('resolution_high_max_pixels') != HIGH_PIXELS
                or cfg.get('max_pixels') != LOW_PIXELS or cfg.get('task_binding_fraction') is not None
                or cfg.get('contrast_kl_budget') is not None or cfg.get('method') != 'bias'
                or (condition != 'baseline' and cfg['bias'] != 6)):
            raise ValueError('Require the frozen full-frame budgets, original bias6 and single weight1')
        views = {name: self.prepare_view(samples, pixels, step, previous)
                 for name, pixels in [('low', LOW_PIXELS), ('high', HIGH_PIXELS)]}
        validate_views(views['low'][1], views['high'][1])
        heads = None; scope = None; region = 'target'
        if condition != 'baseline':
            scope, kind, count = condition.split(':'); count = int(count)
            if kind not in {'target', 'wrong_region', 'low_rank'} or count <= 0: raise ValueError('Unknown resolution condition')
            rank = rankings[scope]['ranking']
            heads = rank[-count:] if kind == 'low_rank' else rank[:count]
            if len({(h['layer'], h['head']) for h in heads}) != count: raise ValueError('Changed positive heads')
            region = 'wrong' if kind == 'wrong_region' else 'target'
            if region == 'wrong' and any(not m['wrong'][scope] for v in views.values() for m in v[1]):
                raise ValueError('Wrong-region control unavailable: insufficient disjoint cells')
        candidates = [self.processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
        if any(len(c) != 1 for c in candidates) or len({c[0] for c in candidates}) != 5:
            raise ValueError('All five native distinct classes must be available')
        class_ids = [c[0] for c in candidates]; logits = {}; diagnostics = {}; started = time.monotonic()
        for name, (inputs, maps, queries, texts) in views.items():
            state = None
            try:
                if condition != 'baseline': state = self.controller.steer(maps, heads, 6., scope, region)
                with torch.inference_mode():
                    logits[name] = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, class_ids].float()
                diagnostics[name] = state['diagnostics'] if state else {}
            finally:
                self.controller.clear()
        combined = logits['low'] if condition == 'baseline' else compose(logits['high'], logits['low'])
        probabilities = combined.softmax(-1); rows = []
        for i, sample in enumerate(samples):
            reward = int(probabilities[i].argmax())+1
            rows.append(dict(example_id=sample['example_id'], condition=condition, status='ok',
                readout='five_way_answer_likelihood_resolution', contrast_negative_mode='resolution',
                contrast_weight=0 if condition == 'baseline' else 1, actual_forward_branches=2,
                low_max_pixels=LOW_PIXELS, high_max_pixels=HIGH_PIXELS,
                native_class_logits_low=logits['low'][i].cpu().tolist(),
                native_class_logits_high=logits['high'][i].cpu().tolist(),
                native_class_logits_positive=(logits['low'] if condition == 'baseline' else logits['high'])[i].cpu().tolist(),
                native_class_logits_combined=combined[i].cpu().tolist(), native_class_probabilities=probabilities[i].cpu().tolist(),
                low_attention_diagnostics=diagnostics['low'], attention_diagnostics=diagnostics['high'],
                token_audit=self.audit(views['low'][1][i]), high_token_audit=self.audit(views['high'][1][i]),
                prompt=views['low'][3][i], high_prompt=views['high'][3][i], candidate_token_ids=class_ids,
                high_resolution_progress=int(logits['high'][i].argmax())/4,
                reward=reward, progress=(reward-1)/4, raw_output=None,
                duration_seconds_per_batch=time.monotonic()-started))
        return rows
