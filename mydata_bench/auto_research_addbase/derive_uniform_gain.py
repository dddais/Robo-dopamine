"""Frozen discovery-only contrast gain sensitivity; no new GPU predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .prepare import OUT
from .branch_ablation import native_branch
from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary


def derive(row, alpha):
    positive = np.asarray(row['native_class_logits_positive'], dtype=np.float32)
    negative = np.asarray(row['native_class_logits_negative'], dtype=np.float32)
    if row['status'] != 'ok' or positive.shape != (5,) or negative.shape != (5,):
        raise ValueError('Requires all five original native logits')
    logits = (1 + alpha) * positive - alpha * negative
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    result = native_branch(dict(status='ok', example_id=row['example_id'], native_class_logits_positive=logits), 'positive', 'qwen')
    result.update(combined_native_logits=logits.tolist(), native_class_probabilities=probabilities.tolist(),
                  alpha=alpha, derived_only=True, requires_actual_forward_verification=True,
                  composition='(1+alpha)*positive-alpha*negative; equal gain for all five classes')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    args = parser.parse_args()
    frozen_path = OUT / 'selection_uniform_contrast_gain_discovery_v1.json'
    frozen = json.loads(frozen_path.read_text())
    requested = json.loads((OUT / 'splits.json').read_text())['discovery']
    folders = [OUT / 'experiments' / f'{args.model}_{p}_uniform_binding_evidence_a1' / 'discovery'
               for p in frozen['protocols']]
    # Refuse partial per-model screening, including selection biased by job order.
    for folder in folders:
        if not (folder / 'worker_events.jsonl').exists():
            raise ValueError('Wait for all five input protocols to complete: ' + str(folder))
        if set(json.loads((folder / 'requested_ids.json').read_text())) != set(requested):
            raise ValueError('Wrong discovery population')
        files = list(folder.glob('binding_transport_s4/predictions/*_target_*.jsonl'))
        if len(files) != 6:
            raise ValueError('Expected all six discovery conditions')
        if any(set(latest(path)) != set(requested) for path in files):
            raise ValueError('Incomplete source condition')
    labels_all = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {eid: labels_all[eid] for eid in requested}
    del labels_all
    destination = OUT / 'derived_candidates' / time.strftime(f'uniform_gain_{args.model}_%Y%m%d_%H%M%S')
    sources = {str(frozen_path): hashlib.sha256(frozen_path.read_bytes()).hexdigest()}
    points = []
    max_original_error = 0.
    original_checked = 0
    for folder in folders:
        baseline_path = folder / 'predictions/baseline.jsonl'
        baseline = latest(baseline_path)
        if set(baseline) != set(requested) or any(row['status'] != 'ok' for row in baseline.values()):
            raise ValueError('Incomplete baseline')
        base_score = summary(baseline, labels, requested)
        sources[str(baseline_path)] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        for path in sorted(folder.glob('binding_transport_s4/predictions/*_target_*.jsonl')):
            rows = latest(path)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for eid, row in rows.items():
                if row['token_audit']['input_ids_sha256'] != baseline[eid]['token_audit']['input_ids_sha256']:
                    raise ValueError('Mismatched baseline input tokens')
                reconstructed = derive(row, 1.)
                if reconstructed['reward'] != row['reward']:
                    raise ValueError('Unit-gain reconstruction changed class')
                error = np.max(np.abs(np.asarray(reconstructed['native_class_probabilities']) - row['native_class_probabilities']))
                max_original_error = max(max_original_error, float(error))
                if error > 1e-5:
                    raise ValueError('Unit-gain reconstruction changed probabilities')
                original_checked += 1
            for alpha in frozen['sensitivity_alphas']:
                derived = {eid: derive(rows[eid], alpha) for eid in requested}
                method = f'binding_transport_s4_alpha{alpha:g}'
                create_json(destination / 'derived_rows' / folder.parent.name / method / f'{path.stem}.json', derived)
                score = summary(derived, labels, requested)
                for threshold in ['0.125/0.875', '0.2/0.8']:
                    p = dict(experiment=folder.parent.name, method=method, condition=path.stem, field='progress',
                             alpha=alpha, population='discovery', derived_only=True, threshold=threshold,
                             expected=len(requested), valid=score['n'], baseline_valid=base_score['n'],
                             mae=score['mae'], baseline_mae=base_score['mae'], delta_mae=score['mae']-base_score['mae'])
                    for subset in ['all', 'suc', 'fail']:
                        p['accuracy_' + subset] = score['accuracy'][threshold][subset]['rate_all_expected']
                        p['baseline_accuracy_' + subset] = base_score['accuracy'][threshold][subset]['rate_all_expected']
                        p['delta_' + subset] = p['accuracy_' + subset] - p['baseline_accuracy_' + subset]
                    p['meets_descriptive_gate'] = (p['delta_mae'] < 0 and p['delta_all'] >= .1-1e-12 and p['delta_suc'] > 0 and p['delta_fail'] > 0)
                    points.append(p)
    # Same-branch identity preserves intermediate categories as well as endpoints.
    for reward in range(1, 6):
        logits = [0.] * 5
        logits[reward - 1] = 4.
        row = dict(status='ok', example_id='math-only', native_class_logits_positive=logits, native_class_logits_negative=logits)
        for alpha in [.5, 1., 2.]:
            assert derive(row, alpha)['reward'] == reward
    create_json(destination / 'points.json', points)
    create_json(destination / 'provenance.json', dict(model=args.model, sources_sha256=sources,
        frozen_plan=str(frozen_path), original_alpha1_checked=original_checked, max_original_probability_error=max_original_error,
        same_branch_all_five_class_identity='passed', source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Discovery-only derived sensitivity, not actual new forward or validated efficacy'))
    print(destination)
    for protocol in frozen['protocols']:
        group = [p for p in points if p['experiment']==f'{args.model}_{protocol}_uniform_binding_evidence_a1' and p['threshold']=='0.125/0.875']
        print(protocol, 'independent sensitivity conditions', len(group), 'derived passing', sum(p['meets_descriptive_gate'] for p in group))
        for p in sorted(group, key=lambda x: (not x['meets_descriptive_gate'],x['mae'],-x['accuracy_all'],abs(x['alpha']-1),int(x['condition'].rsplit('_',1)[1])))[:3]:
            print(p['condition'], 'alpha',p['alpha'], 'MAE', round(p['mae'],3), 'delta pp', *[round(p['delta_'+s]*100,2) for s in ['all','suc','fail']], 'derived PASS',p['meets_descriptive_gate'])


if __name__ == '__main__':
    main()
