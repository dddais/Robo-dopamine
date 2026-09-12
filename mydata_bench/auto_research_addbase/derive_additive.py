"""Discovery-only additive native logits from frozen independent head probes.

These are derived predictions, not fresh GPU inference. This module does not
change ranking, fit coefficients, or read validation labels for computation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .analyze import for_field
from .branch_ablation import native_branch
from .prepare import OUT


def additive_logits(baseline, groups):
    """Fixed coefficient 1, with exact identity for the single-group case."""
    if not groups:
        raise ValueError('At least one independent head group is required')
    baseline = np.asarray(baseline, dtype=np.float32)
    groups = [np.asarray(z, dtype=np.float32) for z in groups]
    if any(z.shape != baseline.shape for z in groups):
        raise ValueError('Native class shapes must agree')
    if not all(np.isfinite(z).all() for z in [baseline, *groups]):
        raise ValueError('Nonfinite native logits')
    result = groups[0].copy()
    for group in groups[1:]:
        result = result + (group - baseline)
    return result


def derived_row(baseline, groups, model):
    if any(r['status'] != 'ok' for r in [baseline, *groups]):
        raise ValueError('Only complete native probes can be combined')
    for group in groups:
        if (group['prompt'] != baseline['prompt'] or
            group['token_audit']['input_ids_sha256'] != baseline['token_audit']['input_ids_sha256']):
            raise ValueError('Group input or batch padding differs from baseline')
    logits = additive_logits(baseline['native_class_logits_positive'],
                             [r['native_class_logits_positive'] for r in groups])
    expected = 10 if model == 'meter' else 5
    if logits.shape != (expected,):
        raise ValueError('All native reward classes are required')
    row = {'status': 'ok', 'example_id': baseline['example_id'],
           'native_class_logits_positive': logits.tolist()}
    if model == 'meter':
        row['success_logit_positive'] = float(additive_logits(
            baseline['success_logit_positive'], [r['success_logit_positive'] for r in groups]))
    prediction = native_branch(row, 'positive', model)
    prediction.update(combined_native_logits=logits.tolist(),
                      derived_from_recorded_branch='baseline + sum(independent group - baseline)',
                      independent_group_count=len(groups), unique_intervened_heads=8*len(groups),
                      requires_actual_forward_verification=True)
    if model == 'meter':
        prediction['combined_success_logit'] = row['success_logit_positive']
    return prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', choices=['qwen', 'roboreward', 'meter'],
                   default=['qwen', 'roboreward'])
    p.add_argument('--protocols', nargs='+', default=['image_text', 'text_image', 'interleaved',
                                                      'text_video', 'video_text'])
    args = p.parse_args()
    splits = json.loads((OUT/'splits.json').read_text())
    ids = splits['discovery']
    if set(ids) & set(splits['validation']):
        raise ValueError('Training and validation overlap')
    all_labels = json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels = {eid: all_labels[eid] for eid in ids}
    del all_labels
    destination = OUT/'derived_candidates'/time.strftime('additive_%Y%m%d_%H%M%S')
    points, provenance, identity = [], {}, []
    for model in args.models:
        for protocol in args.protocols:
            selection = OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'
            folder = OUT/'experiments'/f'{model}_{protocol}_functional_profile'/'discovery'
            if set(json.loads((folder/'requested_ids.json').read_text())) != set(ids):
                raise ValueError('Discovery probes only')
            baseline_path = folder/'predictions/baseline.jsonl'
            baseline = latest(baseline_path)
            if set(baseline) != set(ids):
                raise ValueError('Incomplete discovery baseline')
            digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            provenance[str(baseline_path)] = digest
            for scope in ['all_frames', 'last_frame']:
                ranking_path = selection/f'ranking_{scope}.json'
                ranking = json.loads(ranking_path.read_text())
                if (ranking['validation_ids_used'] or set(ranking['example_ids']) != set(ids)
                    or ranking['baseline_sha256'] != digest):
                    raise ValueError('Frozen ranking or baseline provenance mismatch')
                provenance[str(ranking_path)] = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
                head_set = set()
                all_groups = []
                for number, group in enumerate(ranking['layer_profiles'][:8], 1):
                    heads = [(h['layer'], h['head']) for h in group['heads']]
                    expected = [(h['layer'], h['head']) for h in ranking['ranking'][(number-1)*8:number*8]]
                    if (len(heads) != 8 or len(set(heads)) != 8 or set(heads) & head_set
                        or heads != expected or {h[0] for h in heads} != {group['layer']}):
                        raise ValueError('Frozen unique head group differs')
                    head_set.update(heads)
                    path = Path(group['source'])
                    if not path.resolve().is_relative_to(folder.resolve()):
                        raise ValueError('Only this input discovery profile may be read')
                    digest_group = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest_group != group['sha256']:
                        raise ValueError('Frozen probe source changed')
                    provenance[str(path)] = digest_group
                    rows = latest(path)
                    if set(rows) != set(ids):
                        raise ValueError('Incomplete group probe')
                    all_groups.append(rows)
                    if number not in [1, 2, 4, 8]:
                        continue
                    derived = {eid: derived_row(baseline[eid], [g[eid] for g in all_groups], model)
                               for eid in ids}
                    condition = f'{scope}_target_{8*number}'
                    create_json(destination/'derived_rows'/f'{model}_{protocol}'/f'{condition}.json', derived)
                    if number == 1:
                        for eid in ids:
                            original = native_branch(rows[eid], 'positive', model)
                            if derived[eid]['progress'] != original['progress']:
                                raise ValueError('One-group output is not exactly the original probe')
                            if model == 'meter' and derived[eid]['success_probability'] != original['success_probability']:
                                raise ValueError('One-group native success identity failed')
                        identity.append({'model': model, 'protocol': protocol, 'scope': scope, 'n': len(ids),
                                         'one_group_prediction_exact': True})
                    for field in ['progress'] + (['success_probability'] if model == 'meter' else []):
                        base = summary(for_field(baseline, field), labels, ids)
                        result = summary(for_field(derived, field), labels, ids)
                        for threshold in ['0.125/0.875', '0.2/0.8']:
                            point = dict(experiment=f'{model}_{protocol}_functional_additive',
                                         condition=condition, field=field, threshold=threshold,
                                         method='independent_binding_s4', population='discovery',
                                         derived_only=True, expected=len(ids), valid=result['n'],
                                         mae=result['mae'], baseline_mae=base['mae'],
                                         delta_mae=result['mae']-base['mae'], independent_group_count=number)
                            for label in ['all', 'suc', 'fail']:
                                point['delta_'+label] = (result['accuracy'][threshold][label]['rate_all_expected']
                                                         -base['accuracy'][threshold][label]['rate_all_expected'])
                            point['meets_descriptive_gate'] = (point['delta_mae'] < 0 and point['delta_all'] >= .1-1e-12
                                                              and point['delta_suc'] > 0 and point['delta_fail'] > 0)
                            points.append(point)
    create_json(destination/'points.json', points)
    create_json(destination/'provenance.json', {'sources': provenance, 'identity_checks': identity,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pre_registered': 'selection_additive_functional_discovery_v1.json',
        'interpretation': 'Derived discovery-only candidates. Existing supervised ranking, no new fit. '
                          'One baseline plus k/8 independent group forwards. Actual GPU verification still required.'})
    print(destination)
    for experiment in sorted({p['experiment'] for p in points}):
        candidates = [p for p in points if p['experiment'] == experiment and p['threshold'] == '0.125/0.875']
        for item in sorted(candidates, key=lambda p: (not p['meets_descriptive_gate'], p['mae']))[:3]:
            print(experiment, item['field'], item['condition'], 'MAE', round(item['baseline_mae'], 3),
                  '->', round(item['mae'], 3), 'delta pp',
                  *[round(100*item['delta_'+s], 2) for s in ['all', 'suc', 'fail']],
                  'derived PASS', item['meets_descriptive_gate'])


if __name__ == '__main__':
    main()
