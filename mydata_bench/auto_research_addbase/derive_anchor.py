"""Discovery-only, explicitly derived three-branch attention evidence candidates."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .analyze import for_field
from .branch_ablation import native_branch


def derived_row(baseline, steered, model, beta):
    if baseline['status'] != 'ok' or steered['status'] != 'ok':
        raise ValueError('Incomplete native branch')
    b = np.asarray(baseline['native_class_logits_positive'], dtype=np.float32)
    p = np.asarray(steered['native_class_logits_positive'], dtype=np.float32)
    n = np.asarray(steered['native_class_logits_negative'], dtype=np.float32)
    if b.shape != p.shape or b.shape != n.shape:
        raise ValueError('Native branch shape mismatch')
    if baseline['prompt'] != steered['prompt']:
        raise ValueError('Baseline and evidence prompts differ')
    record = {'status': 'ok', 'example_id': baseline['example_id'],
              'native_class_logits_positive': (b + beta * (p - n)).tolist()}
    if model == 'meter':
        b = np.float32(baseline['success_logit_positive'])
        p = np.float32(steered['success_logit_positive'])
        n = np.float32(steered['success_logit_negative'])
        record['success_logit_positive'] = float(b + beta * (p - n))
    result = native_branch(record, 'positive', model)
    result.update(derived_from_recorded_branch='baseline + beta * (positive - negative)',
                  beta=beta, requires_actual_forward_verification=True,
                  combined_native_logits=record['native_class_logits_positive'])
    return result


def main():
    requested = json.loads((OUT / 'splits.json').read_text())['discovery']
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {eid: all_labels[eid] for eid in requested}
    del all_labels
    points = []; sources = {}
    destination = OUT / 'derived_candidates' / time.strftime('anchor_%Y%m%d_%H%M%S')
    for folder in sorted((OUT / 'experiments').glob('*_evidence_a1/discovery')):
        if set(json.loads((folder / 'requested_ids.json').read_text())) != set(requested):
            raise ValueError('Discovery only')
        model = json.loads((folder / 'runtime_config.json').read_text())['model']
        baseline_path = folder / 'predictions/baseline.jsonl'
        baseline = latest(baseline_path)
        if set(baseline) != set(requested):
            continue
        sources[str(baseline_path)] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        for path in sorted(folder.glob('mass_transport_s*/predictions/*_target_*.jsonl')):
            if path.parents[1].name not in {'mass_transport_s2', 'mass_transport_s4', 'mass_transport_s8'}:
                continue
            if int(path.stem.rsplit('_', 1)[1]) not in [8, 32, 64]:
                continue
            observed = latest(path)
            if set(observed) != set(requested) or any(r['status'] != 'ok' for r in observed.values()):
                continue
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for beta in [1., .5, 2.]:
                rows = {eid: derived_row(baseline[eid], observed[eid], model, beta) for eid in requested}
                method = f'{path.parents[1].name}_beta{beta:g}'
                create_json(destination / 'derived_rows' / folder.parent.name / method / f'{path.stem}.json', rows)
                for field in ['progress'] + (['success_probability'] if model == 'meter' else []):
                    base = summary(for_field(baseline, field), labels, requested)
                    result = summary(for_field(rows, field), labels, requested)
                    for threshold in ['0.125/0.875', '0.2/0.8']:
                        point = {'experiment': folder.parent.name, 'condition': path.stem,
                                 'field': field, 'method': method, 'beta': beta,
                                 'threshold': threshold, 'population': 'discovery',
                                 'derived_only': True, 'expected': len(requested),
                                 'mae': result['mae'], 'baseline_mae': base['mae'],
                                 'delta_mae': result['mae'] - base['mae']}
                        for split in ['all', 'suc', 'fail']:
                            point['delta_' + split] = (result['accuracy'][threshold][split]['rate_all_expected']
                                                      - base['accuracy'][threshold][split]['rate_all_expected'])
                        point['meets_descriptive_gate'] = (point['delta_mae'] < 0 and point['delta_all'] >= .1 - 1e-12
                                                          and point['delta_suc'] > 0 and point['delta_fail'] > 0)
                        points.append(point)
    create_json(destination / 'points.json', points)
    create_json(destination / 'provenance.json', {
        'sources': sources, 'formula': 'baseline logits + beta * (positive logits - negative logits)',
        'primary_beta': 1., 'sensitivity_betas': [.5, 2.],
        'population': 'discovery70 only; validation not scored',
        'interpretation': 'Derived candidates, not actual new GPU inference or confirmed effects.',
        'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(destination)
    groups = defaultdict(list)
    for point in points:
        if point['threshold'] == '0.125/0.875':
            groups[(point['experiment'], point['field'])].append(point)
    for name, items in groups.items():
        print(name)
        for point in sorted(items, key=lambda p: (not p['meets_descriptive_gate'], p['beta'] != 1, p['delta_mae']))[:3]:
            print(point['method'], point['condition'], 'MAE', round(point['baseline_mae'], 3), '->', round(point['mae'], 3),
                  'delta pp', *[round(point['delta_' + s] * 100, 2) for s in ['all', 'suc', 'fail']],
                  'derived PASS', point['meets_descriptive_gate'])


if __name__ == '__main__':
    main()
