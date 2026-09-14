"""Compare official sources and intercept real SDPA calls independently of diagnostics.

This is a small verification run, not an accuracy experiment. Generation is
limited to two tokens for the attention checks; SOLE feedback is checked
separately with complete seven-step baseline predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import yaml

from mydata_bench.addbase_eval.verify_execution import (
    REF, official_meter_inputs, official_meter_readout, official_sole_inputs, assert_inputs_equal,
)
from mydata_bench.top_eval.versioning import is_official_sole
from .common import ROOT, OUT, MODELS, PROTOCOLS, create_json, file_hash, conditions
from .runtime import Runtime


class AttentionAudit:
    """Check the actual mask delivered to SDPA against independently selected heads/keys."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.original = runtime.controller.original
        self.pending = None
        self.mapping = None
        self.heads = []
        self.scope = None
        self.region = 'target'
        self.bias = 0.0
        self.calls = Counter()
        self.rank_max_error = 0.0
        runtime.controller.original = self.checked_sdpa
        ALL_ATTENTION_FUNCTIONS.register('sdpa', self.entry)

    def begin(self, mapping=None, heads=(), scope=None, region='target', bias=0.0):
        self.mapping, self.heads, self.scope = mapping, list(heads), scope
        self.region, self.bias = region, bias
        self.calls.clear()

    @staticmethod
    def rows(mask, q, k, heads, selected, device):
        if mask is None:
            out = torch.zeros(heads, len(selected), k, device=device)
            future = torch.arange(k, device=device)[None, :] > (k - q + torch.tensor(selected, device=device))[:, None]
            return out.masked_fill(future[None], -torch.inf)
        selected = selected if mask.shape[-2] != 1 else [0] * len(selected)
        raw = mask[0, :, selected, :k]
        if raw.dtype == torch.bool:
            raw = torch.zeros_like(raw, dtype=torch.float32).masked_fill(~raw, -torch.inf)
        return raw.float().expand(heads, -1, -1)

    def entry(self, module, query, key, value, attention_mask, **kwargs):
        layer = self.runtime.controller.layer_ids.get(id(module))
        self.pending = (module, attention_mask, layer)
        try:
            result = self.runtime.controller.forward(module, query, key, value, attention_mask, **kwargs)
            state = self.runtime.controller.state
            if layer is not None and state is not None and state['kind'] == 'rank':
                qpos = state['queries'][0]
                h, k = query.shape[1], key.shape[-2]
                repeated = key[0].repeat_interleave(h // key.shape[1], dim=0).float()
                scores = torch.bmm(query[0, :, qpos:qpos + 1].float(), repeated.transpose(1, 2)).squeeze(1)
                scores *= kwargs.get('scaling') or query.shape[-1] ** -0.5
                scores += self.rows(attention_mask, query.shape[-2], k, h, [qpos], query.device)[:, 0]
                weights = scores.softmax(-1)
                for scope in state['raw']:
                    expected = weights[:, self.mapping['target'][scope]].sum(-1).cpu().numpy()
                    actual = state['raw'][scope][0, layer]
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                    self.rank_max_error = max(self.rank_max_error, float(np.max(np.abs(actual - expected))))
            return result
        finally:
            self.pending = None

    def checked_sdpa(self, module, query, key, value, attention_mask, **kwargs):
        pending_module, original_mask, layer = self.pending
        assert pending_module is module
        if layer is not None:
            q, k, h = query.shape[-2], key.shape[-2], query.shape[1]
            selected_rows = sorted({0, min(2, q - 1), q // 2, q - 1})
            expected = self.rows(original_mask, q, k, h, selected_rows, query.device).clone()
            chosen_heads = [int(x['head']) for x in self.heads if int(x['layer']) == layer]
            if chosen_heads:
                positive = set(self.mapping[self.region][self.scope])
                domain = set(self.mapping['target'][self.scope]) | set(self.mapping['negative'][self.scope])
                assert positive <= domain
                for head in chosen_heads:
                    expected[head, :, sorted(p for p in positive if p < k)] += self.bias
                    expected[head, :, sorted(p for p in domain - positive if p < k)] -= self.bias
            actual = self.rows(attention_mask, q, k, h, selected_rows, query.device)
            # Compare blocked positions separately; finite min-dtype masks preserve causality too.
            torch.testing.assert_close(actual < -10000, expected < -10000, rtol=0, atol=0)
            visible = expected >= -10000
            torch.testing.assert_close(actual[visible], expected[visible], rtol=0, atol=0)
            self.calls['language_prefill' if q > 1 else 'language_decode'] += 1
            if chosen_heads:
                self.calls['biased_prefill' if q > 1 else 'biased_decode'] += 1
        else:
            assert attention_mask is original_mask, 'Vision encoder mask must remain untouched'
            self.calls['vision_unmodified'] += 1
        return self.original(module, query, key, value, attention_mask, **kwargs)

    def close(self):
        self.runtime.controller.original = self.original
        ALL_ATTENTION_FUNCTIONS.register('sdpa', self.original)


def official_input_checks(runtime, samples):
    records = []
    for sample in samples:
        for step in range(1, 8) if runtime.cfg['model'] == 'sole' else [None]:
            prior = '37.125' if step else 0
            actual, mapping, query, _ = runtime.prepare(sample, step=step, previous=prior)
            reference = (official_meter_inputs(runtime.processor, [sample]) if runtime.cfg['model'] == 'meter'
                         else official_sole_inputs(runtime.processor, [sample], step, [prior]))
            assert_inputs_equal(actual, reference)
            records.append({'example_id': sample['example_id'], 'step': step, 'all_tensors_exact': True,
                            'shapes': {k: list(v.shape) for k, v in actual.items()}, 'query_position': query})
    return records


def mosaic_boundary_count(mapping, scope):
    if scope != 'last_frame' or not mapping['records'][-1].get('mosaic'):
        return 0
    record = mapping['records'][-1]
    span = record['span']
    cols = span.grid_thw[2] // 2
    width = record['size'][0]
    # Previous tile ends at 773; current tile begins at 778 (5px separator).
    domain = set(mapping['target'][scope]) | set(mapping['negative'][scope])
    return sum((p - span.start) % cols * width / cols < 773 for p in domain)


def run_model(model, samples, output):
    configs = {p: yaml.safe_load((ROOT / f'mydata_bench/configs/v2_basic_method/{model}_{p}.yaml').read_text()) for p in PROTOCOLS}
    runtime = Runtime(configs['official'])
    report = {'model': model, 'conditions': {}, 'num_layers': runtime.num_layers, 'num_heads': runtime.num_heads,
              'loading_audit': runtime.model.loading_audit}
    if model in ('meter', 'sole'):
        report['official_input_parity'] = official_input_checks(runtime, samples)
    sample = samples[0]
    if model == 'meter':
        batch, _, _, _ = runtime.prepare(sample)
        inputs = runtime._move(batch)
        with torch.inference_mode():
            hidden = runtime.model(**inputs, use_cache=False).last_hidden_state
            actual, success, _ = runtime.model.read_progress(hidden, inputs['input_ids'], runtime.prog_id)
            expected, expected_success, _ = official_meter_readout(runtime, hidden, inputs['input_ids'])
        torch.testing.assert_close(torch.tensor(actual, device=expected.device), expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.tensor(success, device=expected.device), expected_success, rtol=0, atol=0)
        report['official_progress_and_success_heads_exact'] = True
        del hidden, inputs, batch
    if model == 'sole':
        previous, trace = '0', []
        for step in range(1, 8):
            row = runtime.predict(sample, 'baseline', step=step, previous=previous)
            assert row['status'] == 'ok', row
            ref = official_sole_inputs(runtime.processor, [sample], step, [previous])
            with torch.inference_mode():
                generated = runtime.model.generate(**runtime._move(ref), do_sample=False, max_new_tokens=512,
                    temperature=None, top_p=None, top_k=None, use_cache=True, logits_to_keep=1,
                    pad_token_id=runtime.processor.tokenizer.pad_token_id)
            text = runtime.processor.batch_decode(generated[:, ref['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
            assert text == row['raw_output'], f'Official direct generation differs at step {step}'
            assert f'previous timestep is {previous}%' in row['prompt']
            trace.append({'step': step, 'previous_percentage': previous, 'progress': row['progress'], 'raw_output': text})
            previous = row['percentage_text']
        report['seven_step_reference_decode_and_feedback'] = trace
    audit = AttentionAudit(runtime)
    try:
        for protocol, original_cfg in configs.items():
            cfg = copy.deepcopy(original_cfg)
            cfg['max_new_tokens'] = 2  # Inspect prefill and cache decode, not task accuracy.
            runtime.cfg = cfg
            step, prior = (7, '37.125') if is_official_sole(cfg) else (None, 0)
            audit.begin()
            base = runtime.predict(sample, 'baseline', step=step, previous=prior)
            maps, rankings = {}, {}
            for scope in cfg['scopes']:
                _, mapping, _, _ = runtime.prepare(sample, scope, step, prior)
                assert base['token_audit']['input_ids_sha256'] == mapping['input_ids_sha256']
                maps[scope] = mapping
                audit.begin(mapping)
                observation = runtime.collect(sample, scope, step, prior)
                masses = np.asarray(observation['raw_mass'])
                ordered = [{'layer': l, 'head': h, 'score': float(masses[l, h])}
                           for l in range(cfg['skip_early_layers'], runtime.num_layers) for h in range(runtime.num_heads)]
                ordered.sort(key=lambda h: (-h['score'], h['layer'], h['head']))
                rankings[scope] = {'ranking': ordered}
            for condition in conditions(cfg)[1:]:
                scope, kind, k = condition.split(':')
                mapping = maps[scope]
                if kind == 'wrong_region' and mapping['control_unavailable']:
                    raise AssertionError('Choose a smoke example supporting wrong-region controls')
                ordered = rankings[scope]['ranking']
                heads = ordered[-int(k):] if kind == 'low_rank' else ordered[:int(k)]
                region = 'wrong' if kind == 'wrong_region' else 'target'
                audit.begin(mapping, heads, scope, region, cfg['bias'])
                runtime.predict(sample, condition, rankings[scope], step, prior)
                assert audit.calls['language_prefill'] == runtime.num_layers
                assert audit.calls['biased_prefill'] == len({h['layer'] for h in heads})
                if model != 'meter':
                    assert audit.calls['biased_decode'] > 0
                report['conditions'][f'{protocol}/{condition}'] = {
                    'actual_sdpa_offsets_exact': True, 'calls': dict(audit.calls),
                    'target_keys': len(mapping['target'][scope]),
                    'negative_keys': len(mapping['negative'][scope]),
                    'selected_heads': len(heads), 'mosaic_previous_tile_boundary_keys': mosaic_boundary_count(mapping, scope)}
            cfg['bias'] = 0.0
            for scope in cfg['scopes']:
                mapping, heads = maps[scope], rankings[scope]['ranking'][:8]
                audit.begin(mapping, heads, scope, 'target', 0.0)
                zero = runtime.predict(sample, f'{scope}:target:8', rankings[scope], step, prior)
                # Explicit-mask zero bias can change BF16 rounding, but cannot change the mask.
                report.setdefault('zero_bias', {})[f'{protocol}/{scope}'] = {
                    'sdpa_mask_exact': True, 'same_output': zero.get('raw_output') == base.get('raw_output')
                    if model != 'meter' else abs(zero['progress'] - base['progress']) < 1e-5}
            audit.begin()
            again = runtime.predict(sample, 'baseline', step=step, previous=prior)
            if model == 'meter':
                assert again['progress'] == base['progress'] and again['success_probability'] == base['success_probability']
            else:
                assert again['raw_output'] == base['raw_output']
            print(f'VERIFIED {model}/{protocol}: 18 conditions, ranking and baseline isolation', flush=True)
        report['ranking_independent_max_abs_error'] = audit.rank_max_error
        report['status'] = 'verified'
        create_json(output / f'{model}.json', report)
    finally:
        audit.close()
        del audit, runtime
        gc.collect()
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; audit reports are immutable')
    samples = json.loads((OUT / 'inputs.json').read_text())[:2]
    sources = [REF / 'robometer/robometer/data/collators/rbm_heads.py',
               REF / 'robometer/robometer/models/rbm.py', REF / 'robometer/robometer/models/heads.py',
               REF / 'rewardgen/rewardgen/sole.py', ROOT / 'mydata_bench/basic_method/runtime.py',
               ROOT / 'mydata_bench/addbase_eval/attention.py']
    create_json(args.output / 'manifest.json', {'purpose': __doc__, 'sample_ids': [s['example_id'] for s in samples],
                 'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                 'sdpa_query_rows_checked': 'first, third, middle, final; every cached decode query',
                 'official_sole_decoding_comparison': 'same HF greedy/512 setting, not upstream vLLM sampling'})
    for model in args.models:
        run_model(model, samples, args.output)
    create_json(args.output / 'complete.json', {'models': args.models, 'status': 'verified'})
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
