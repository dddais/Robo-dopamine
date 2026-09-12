"""Original visual steering plus class-symmetric conditional task evidence."""
import time

import torch

from .runtime import ResearchRuntime


CELLS = {'bias': (6., 'bias'), 'pp': (4., 'binding_transport'),
         'pm': (4., 'binding_task_suppression')}


def compose(cells):
    if set(cells) != set(CELLS):
        raise ValueError('All three actual native branches are required')
    b, p, n = [cells[k] for k in ['bias', 'pp', 'pm']]
    if any(z.shape != b.shape or z.shape[-1] != 5 for z in [b, p, n]):
        raise ValueError('Same five unrestricted native classes required')
    effect = p - n
    return b + effect, effect


class BiasTaskAnchorRuntime(ResearchRuntime):
    def predict(self, samples, condition, rankings=None, step=None, previous=None):
        cfg = self.cfg
        if (cfg['model'] not in {'qwen','roboreward'} or cfg.get('contrast_negative_mode') != 'bias_task_anchor'
                or cfg.get('contrast_reference','positive') != 'positive' or cfg.get('contrast_weight') != 1
                or cfg.get('negative_strength') != 4 or cfg.get('negative_task_strength') != 4
                or cfg.get('task_binding_fraction') != .5 or cfg.get('task_binding_distribution') != 'uniform'
                or cfg.get('visual_mass_partition','global') != 'global'
                or (condition != 'baseline' and cfg['bias'] != 4)):
            raise ValueError('Actual original-bias/task composition differs from frozen design')
        self.active_ranking_prefix = 'ANSWER: '
        try: inputs, maps, queries, texts = self.prepare(samples, step, previous)
        finally: self.active_ranking_prefix = ''
        started = time.monotonic()
        heads = None; scope = None; region = 'target'
        if condition != 'baseline':
            scope, kind, k = condition.split(':'); k = int(k)
            if kind not in {'target','wrong_region','low_rank'} or k <= 0:
                raise ValueError('Unknown original-bias/task condition')
            ranked = rankings[scope]['ranking']
            heads = ranked[-k:] if kind == 'low_rank' else ranked[:k]
            if len({(h['layer'],h['head']) for h in heads}) != k:
                raise ValueError('Require exactly k unique frozen heads')
            region = 'wrong' if kind == 'wrong_region' else 'target'
            if region == 'wrong' and any(not m['wrong'][scope] for m in maps):
                raise ValueError('Wrong-region control unavailable: insufficient disjoint cells')
        ids = [self.processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1,6)]
        if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 5:
            raise ValueError('Require all five distinct native reward tokens')
        class_ids = [x[0] for x in ids]

        def branch(cell=None):
            old_method = cfg['method']; state = None
            try:
                if cell is not None:
                    strength, cfg['method'] = CELLS[cell]
                    state = self.controller.steer(maps, heads, strength, scope, region)
                with torch.inference_mode():
                    logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:,-1,class_ids].float()
                return logits, state['diagnostics'] if state is not None else {}
            finally:
                self.controller.clear(); cfg['method'] = old_method

        cells = {}; diagnostics = {}
        if condition == 'baseline':
            combined, baseline_diagnostics = branch()
            if baseline_diagnostics: raise ValueError('Require an unsteered baseline')
            effect = None
        else:
            for cell in CELLS: cells[cell], diagnostics[cell] = branch(cell)
            combined, effect = compose(cells)
        probabilities = combined.softmax(-1)
        rows = []
        for i, sample in enumerate(samples):
            reward = int(probabilities[i].argmax()) + 1
            rows.append(dict(example_id=sample['example_id'], condition=condition, status='ok',
                readout='five_way_answer_likelihood_bias_task_anchor', contrast_negative_mode='bias_task_anchor',
                contrast_weight=1 if cells else 0, native_cell_logits={k:z[i].cpu().tolist() for k,z in cells.items()},
                cell_attention_diagnostics=diagnostics,
                native_class_logits_positive=(cells['pp'] if cells else combined)[i].cpu().tolist(),
                native_task_difference_logits=effect[i].cpu().tolist() if effect is not None else None,
                native_class_logits_combined=combined[i].cpu().tolist(),
                native_class_probabilities=probabilities[i].cpu().tolist(), actual_forward_branches=3 if cells else 1,
                composition='bias+(pp-pm)' if cells else 'single unsteered forward',
                prompt=texts[i], token_audit=self.audit(maps[i]), candidate_token_ids=class_ids,
                reward=reward, progress=(reward-1)/4, raw_output=None,
                duration_seconds_per_batch=time.monotonic()-started))
        return rows
