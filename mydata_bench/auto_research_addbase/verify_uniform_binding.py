"""Verify uniform task injection against the frozen proportional counterpart."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', required=True)
    args = parser.parse_args()
    checks, sources = [], {}

    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path)

    for experiment in args.experiments:
        folder = OUT/'experiments'/experiment/'discovery_smoke'
        events = [json.loads(line) for line in (folder/'worker_events.jsonl').read_text().splitlines()]
        if not events or events[-1]['event'] != 'complete':
            raise ValueError('Wait for completed smoke')
        cfg = json.loads((folder/'runtime_config.json').read_text())
        ids = json.loads((folder/'requested_ids.json').read_text())
        if (len(ids) != 8 or cfg['task_binding_distribution'] != 'uniform' or
            cfg['task_binding_fraction'] != .5 or cfg['contrast_weight'] != 1):
            raise ValueError('Smoke differs from frozen setting')
        reference = OUT/'experiments'/f"{cfg['model']}_{cfg['protocol']}_functional_evidence_a1"/'discovery_smoke'
        baseline, old_baseline = read(folder/'predictions/baseline.jsonl'), read(reference/'predictions/baseline.jsonl')
        if set(baseline) != set(ids) or set(old_baseline) != set(ids):
            raise ValueError('Matched baseline coverage differs')
        paths = sorted(folder.glob('binding_transport_s4/predictions/*_target_*.jsonl'))
        expected_conditions = {f'{s}_target_{k}' for s in ['all_frames', 'last_frame'] for k in [8, 32]}
        if {path.stem for path in paths} != expected_conditions:
            raise ValueError('Four frozen smoke conditions required')
        changed = 0
        for path in paths:
            rows = read(path)
            old = read(reference/'binding_transport_s4/predictions'/path.name)
            if set(rows) != set(ids) or set(old) != set(ids):
                raise ValueError('Missing current or reference predictions')
            scope, _, count = next(iter(rows.values()))['condition'].split(':')
            k = int(count)
            ranking = json.loads((folder.parent/'ranking'/f'ranking_{scope}.json').read_text())
            expected_heads = {(h['layer'], h['head']) for h in ranking['ranking'][:k]}
            if ranking['validation_ids_used']:
                raise ValueError('Validation labels cannot select heads')
            error = change = 0.
            for eid in ids:
                a, b = rows[eid], old[eid]
                if any(x['status'] != 'ok' for x in [a, b, baseline[eid], old_baseline[eid]]):
                    raise ValueError('Native forward did not succeed')
                for left, right in [(a, b), (baseline[eid], old_baseline[eid])]:
                    if (left['prompt'] != right['prompt'] or
                        left['token_audit']['input_ids_sha256'] != right['token_audit']['input_ids_sha256']):
                        raise ValueError('Prompt or batch padding changed')
                if baseline[eid]['native_class_logits_positive'] != old_baseline[eid]['native_class_logits_positive']:
                    raise ValueError('Baseline changed')
                if a['native_class_logits_negative'] != b['native_class_logits_negative']:
                    raise ValueError('Negative branch changed')
                for field in ['attention_diagnostics', 'negative_attention_diagnostics']:
                    actual = {(int(layer), head) for layer, item in a[field].items() for head in item['heads']}
                    if actual != expected_heads:
                        raise ValueError('Head budget differs')
                for item in a['attention_diagnostics'].values():
                    if not (item['method']=='binding_transport' and item['task_binding_distribution']=='uniform'
                            and item['explicit_mask_visibility'] and item['text_domain_mass_preserved']
                            and item['domain_mass_preserved'] and item['all_query_rows']):
                        raise ValueError('Positive uniform hook differs')
                if any(item['method'] != 'mass_transport' for item in a['negative_attention_diagnostics'].values()):
                    raise ValueError('Negative branch has instruction intervention')
                positive = np.asarray(a['native_class_logits_positive'], dtype=np.float32)
                negative = np.asarray(a['native_class_logits_negative'], dtype=np.float32)
                logits = (2*positive-negative).astype(np.float64)
                if logits.shape not in [(5,), (10,)]:
                    raise ValueError('Native class set changed')
                probability = np.exp(logits-logits.max()); probability /= probability.sum()
                error = max(error, float(np.max(np.abs(probability-a['native_class_probabilities']))))
                change = max(change, float(np.max(np.abs(positive-np.asarray(b['native_class_logits_positive'])))))
                if cfg['model'] != 'meter':
                    if int(probability.argmax())+1 != a['reward']:
                        raise ValueError('Reward no longer comes from all five native classes')
                else:
                    if a['success_logit_negative'] != b['success_logit_negative']:
                        raise ValueError('Native negative success changed')
                    if baseline[eid]['success_logit_positive'] != old_baseline[eid]['success_logit_positive']:
                        raise ValueError('Native baseline success changed')
                    z = float(np.float32(2*np.float32(a['success_logit_positive'])-np.float32(a['success_logit_negative'])))
                    error = max(error, abs(a['success_probability']-np.exp(-np.logaddexp(0, -z))),
                                abs(a['progress']-probability@np.linspace(0, 1, 10)))
            if error >= 1e-5:
                raise ValueError('Native probability reconstruction failed')
            changed += change > 0
            checks.append({'experiment': experiment, 'condition': path.stem, 'n': 8,
                           'baseline_and_negative_exact': True, 'unique_heads_per_branch': k,
                           'max_positive_logit_change': change, 'max_probability_error': error})
        if not changed:
            raise ValueError('Uniform intervention had no observable positive effect')
    destination = OUT/'audit'/time.strftime('uniform_binding_actual_%Y%m%d_%H%M%S.json')
    create_json(destination, {'status':'pass', 'checks':checks, 'sources':sources, 'labels_read':False,
                             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(destination)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
