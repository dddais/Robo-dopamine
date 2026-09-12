"""Label-free full-neighborhood audit for frozen round24/25 alpha2 targets."""
import argparse
import json
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .factorized_gain_tasks import verify_native
from .compare_native_bias import same_input as discrete_same_input
from .meter_factorized import verify_rows as meter_verify, same_input as meter_same_input


VARIANT = 'uniform_factorized_evidence_a2'


def verify(model, protocol):
    selection = OUT / ('selection_meter_factorized_a2_full_v1.json' if model == 'meter'
                       else 'selection_factorized_global_gain_full_v1.json')
    record = json.loads(selection.read_text())
    point, = [p for p in record['selected'] if p['model'] == model and p['protocol'] == protocol]
    verify_sources(record['sources_sha256'])
    sources = {str(selection): sha(selection)}

    def read(path, rows=False):
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())

    root = OUT / 'experiments' / f'{model}_{protocol}_{VARIANT}' / 'full_cohort'
    ids = read(OUT / 'splits.json')['full_cohort']
    if len(ids) != 846 or len(set(ids)) != 846 or read(root / 'requested_ids.json') != ids:
        raise ValueError('All and only the frozen full846 population is required')
    rank_root = OUT / 'functional_selections' / f'stage8_{model}_v1' / f'{model}_{protocol}'
    expected = dict(model=model, contrast_negative_mode='visual_and_task', contrast_weight=2,
                    negative_strength=4, negative_task_strength=4, task_binding_fraction=.5,
                    task_binding_distribution='uniform', ranking_prefix='' if model == 'meter' else 'ANSWER: ',
                    frozen_ranking_source=str(rank_root))
    cfg = read(root / 'runtime_config.json')
    if any(cfg.get(k) != v for k, v in expected.items()) or cfg.get('visual_mass_partition', 'global') != 'global':
        raise ValueError('Actual configuration differs from frozen shared alpha2')
    scope = point['scope']
    ranking = read(rank_root / f'ranking_{scope}.json')['ranking']
    actual_ranking = read(root.parent / 'ranking' / f'ranking_{scope}.json')['ranking']
    if actual_ranking != ranking:
        raise ValueError('Actual copied ranking differs from the fixed source')
    same_input = meter_same_input if model == 'meter' else discrete_same_input

    def native(rows, heads=None, baseline=False):
        if model == 'meter':
            return meter_verify(rows, ids, heads, baseline)
        return verify_native(rows, ids, 2., heads, baseline)

    base = native(read(root / 'predictions/baseline.jsonl', True), baseline=True)
    if any(r['condition'] != 'baseline' or r['attention_diagnostics'] or r['negative_attention_diagnostics']
           or r['task_negative_attention_diagnostics'] for r in base.values()):
        raise ValueError('Baseline must be an actual unsteered forward')
    checks = []
    for k in point['ks']:
        h = {(p['layer'], p['head']) for p in ranking[:k]}
        if len(h) != k:
            raise ValueError('Duplicate or missing selected heads')
        rows = native(read(root / 'binding_transport_s4/predictions' / f'{scope}_target_{k}.jsonl', True), h)
        for e, row in rows.items():
            if row['condition'] != f'{scope}:target:{k}' or not same_input(row, base[e]):
                raise ValueError('Actual condition or same-batch input differs')
            for field in ['attention_diagnostics', 'negative_attention_diagnostics', 'task_negative_attention_diagnostics']:
                if not all(d['causal_mask_preserved'] and d['all_query_rows'] and d['prefill_calls'] >= 1
                           for d in row[field].values()):
                    raise ValueError('Actual branch masking/query contract differs')
        checks.append(dict(k=k, scope=scope, n=len(rows), actual_three_branches=True,
                           native_formula_heads_and_same_input_verified=True))
    destination = OUT / 'audit' / time.strftime(f'factorized_a2_full_{model}_{protocol}_%Y%m%d_%H%M%S.json')
    create_json(destination, dict(status='pass', labels_read=False, model=model, protocol=protocol,
        frozen_point=point, checks=checks, sources_sha256=sources,
        actual_intervention_rows=sum(c['n'] for c in checks),
        interpretation='Every explicitly frozen adjacent k file has all846 valid actual three-branch observations. '
                       'This verifies implementation and coverage only, not efficacy or family completion.'))
    print(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['qwen', 'roboreward', 'meter'])
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    verify(args.model, args.protocol)


if __name__ == '__main__':
    main()
