"""Frozen joint two-model, five-input discovery-only test of KL-limited contrast."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .kl_contrast import kl_limited_contrast
from .prepare import OUT
from .select_robust_discovery import PROTOCOLS, NEIGHBORHOODS, verify_native


def main():
    policy = OUT / 'selection_kl_limited_discovery_v1.json'
    spec = json.loads(policy.read_text())
    if spec['kl_budgets_nats'] != [.05, .2, .8] or spec['alpha_cap'] != 2 or spec['protocols'] != PROTOCOLS:
        raise ValueError('Frozen KL policy differs')
    ids = json.loads((OUT / 'splits.json').read_text())['discovery']
    if len(ids) != 70:
        raise ValueError('Discovery population differs')
    sources = {str(policy): hashlib.sha256(policy.read_bytes()).hexdigest()}
    def record(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    matrices = {}
    for model in ['qwen', 'roboreward']:
        for protocol in PROTOCOLS:
            folder = OUT / 'experiments' / f'{model}_{protocol}_uniform_binding_evidence_a1' / 'discovery'
            events = [json.loads(line) for line in record(folder / 'worker_events.jsonl').read_text().splitlines()]
            if not events or events[-1]['event'] != 'complete':
                raise ValueError('Require complete original five-input discovery')
            if set(json.loads(record(folder / 'requested_ids.json').read_text())) != set(ids):
                raise ValueError('Source population differs')
            cfg = json.loads(record(folder / 'runtime_config.json').read_text())
            expected = dict(contrast_weight=1, task_binding_distribution='uniform', task_binding_fraction=.5,
                negative_strength=4, bias=4, frozen_ranking_source=str(OUT / 'functional_selections' /
                    f'stage8_{model}_v1' / f'{model}_{protocol}'))
            if any(cfg.get(key) != value for key, value in expected.items()):
                raise ValueError('Unexpected original branch source')
            baseline = verify_native(latest(record(folder / 'predictions/baseline.jsonl')), ids, True)
            paths = list(folder.glob('binding_transport_s4/predictions/*.jsonl'))
            expected_files = {f'{s}_target_{k}.jsonl' for s in ['all_frames', 'last_frame'] for k in [8,32,64]}
            if {p.name for p in paths} != expected_files:
                raise ValueError('Expected exactly six source conditions')
            conditions = {}
            for path in paths:
                rows = verify_native(latest(record(path)), ids)
                if any(rows[e]['token_audit']['input_ids_sha256'] != baseline[e]['token_audit']['input_ids_sha256'] for e in ids):
                    raise ValueError('Source condition differs from baseline input')
                conditions[path.stem] = rows
            matrices[(model, protocol)] = (baseline, conditions)
    # Every source across both models is complete before discovery labels are used.
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {eid: all_labels[eid] for eid in ids}; del all_labels
    destination = OUT / 'derived_candidates' / time.strftime('kl_limited_%Y%m%d_%H%M%S')
    points, details = [], {}
    for (model, protocol), (baseline, conditions) in matrices.items():
        base = summary(baseline, labels, ids)
        for condition, rows in sorted(conditions.items()):
            positive = np.asarray([rows[e]['native_class_logits_positive'] for e in ids])
            negative = np.asarray([rows[e]['native_class_logits_negative'] for e in ids])
            scope, k = condition.split('_target_'); k = int(k)
            for budget in spec['kl_budgets_nats']:
                out = kl_limited_contrast(positive, negative, budget, spec['alpha_cap'])
                derived = {eid: dict(example_id=eid, status='ok', reward=int(out['probabilities'][i].argmax())+1,
                    progress=int(out['probabilities'][i].argmax())/4, derived_only=True,
                    native_class_probabilities=out['probabilities'][i].tolist(),
                    alpha=float(out['alpha'][i]), kl=float(out['kl'][i])) for i, eid in enumerate(ids)}
                score = summary(derived, labels, ids)
                delta = {c: score['accuracy']['0.125/0.875'][c]['rate_all_expected']-
                         base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all', 'suc', 'fail']}
                passing = score['mae'] < base['mae'] and delta['all'] >= .1-1e-12 and min(delta['suc'], delta['fail']) > 0
                point = dict(model=model, protocol=protocol, budget=budget, scope=scope, center_k=k,
                    mae=score['mae'], baseline_mae=base['mae'], deltas=delta, passes=passing,
                    accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'],
                    alpha_quantiles=np.quantile(out['alpha'], [0,.25,.5,.75,1]).tolist(),
                    fraction_capped=float(np.mean(out['alpha'] < 2)), max_kl=float(out['kl'].max()))
                points.append(point)
                key = f'{model}/{protocol}/{condition}/kl{budget:g}'
                details[key] = dict(metrics=score, rows=derived)
    if len(points) != 180:
        raise ValueError('Require the full 180 independent conditions')
    candidates = {}
    for budget in spec['kl_budgets_nats']:
        selected = []
        for model in ['qwen', 'roboreward']:
            for protocol in PROTOCOLS:
                group = [p for p in points if p['model'] == model and p['protocol'] == protocol and
                         p['budget'] == budget and p['passes']]
                if group:
                    best = min(group, key=lambda p: (p['mae'], -p['accuracy_all'], p['center_k'], p['scope']))
                    selected.append(dict(best, ks=NEIGHBORHOODS[best['center_k']]))
        counts = {model: sum(p['model'] == model for p in selected) for model in ['qwen', 'roboreward']}
        candidates[str(budget)] = dict(selected=selected, input_counts=counts, eligible=min(counts.values()) >= 3)
    eligible = [b for b in spec['kl_budgets_nats'] if candidates[str(b)]['eligible']]
    chosen = min(eligible) if eligible else None
    create_json(destination / 'points.json', points)
    create_json(destination / 'details.json', details)
    create_json(destination / 'provenance.json', dict(created_at=time.time(), sources_sha256=sources,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), validation_labels_used=[],
        independent_conditions=180, actual_source_intervention_rows_verified=4200, derived_predictions=12600,
        selection_modified=False, interpretation='Frozen joint derived discovery, adaptive exploration; not actual new GPU inference or final efficacy.'))
    create_json(OUT / 'selection_kl_limited_full_v1.json', dict(created_at=time.time(),
        policy=str(policy), policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(), source=str(destination),
        candidates_by_budget=candidates, selected_budget=chosen,
        selected=candidates[str(chosen)]['selected'] if chosen is not None else [],
        actual_smoke_required=chosen is not None, full_gpu_expansion_authorized_by_rule=chosen is not None,
        interpretation='No eligible budget means no full GPU expansion for this frozen stage. '
            'All budgets and failures are retained; no mixed-budget or cross-method coverage.'))
    print(destination)
    for budget, value in candidates.items():
        print('budget', budget, 'inputs', value['input_counts'], 'eligible', value['eligible'])
        for point in value['selected']:
            print(point['model'], point['protocol'], point['scope'], point['center_k'], point['mae'], point['deltas'])
    print('selected global budget', chosen)


if __name__ == '__main__':
    main()
