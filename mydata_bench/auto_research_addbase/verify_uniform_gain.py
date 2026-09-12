"""Match actual gain sensitivity to recorded discovery branches, without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .prepare import OUT
from .derive_uniform_gain import derive
from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', required=True)
    args = parser.parse_args()
    sources = {}
    checks = []

    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path)

    for experiment in args.experiments:
        folder = OUT / 'experiments' / experiment / 'discovery_smoke'
        events = [json.loads(line) for line in (folder / 'worker_events.jsonl').read_text().splitlines()]
        if not events or events[-1]['event'] != 'complete':
            raise ValueError('Wait for completed actual smoke')
        cfg = json.loads((folder / 'runtime_config.json').read_text())
        alpha = cfg['contrast_weight']
        if alpha not in [.5, 2.] or cfg['task_binding_distribution'] != 'uniform' or cfg['task_binding_fraction'] != .5:
            raise ValueError('Smoke differs from frozen sensitivity settings')
        ids = json.loads((folder / 'requested_ids.json').read_text())
        if len(ids) != 8:
            raise ValueError('Expected eight matched samples')
        reference = OUT / 'experiments' / f"{cfg['model']}_{cfg['protocol']}_uniform_binding_evidence_a1" / 'discovery'
        baseline = read(folder / 'predictions/baseline.jsonl')
        old_baseline = read(reference / 'predictions/baseline.jsonl')
        if set(baseline) != set(ids) or not set(ids) <= set(old_baseline):
            raise ValueError('Baseline coverage differs')
        paths = sorted(folder.glob('binding_transport_s4/predictions/*_target_*.jsonl'))
        if {p.stem for p in paths} != {f'{scope}_target_{k}' for scope in ['all_frames','last_frame'] for k in [8,32]}:
            raise ValueError('Expected all four smoke conditions')
        for path in paths:
            rows = read(path)
            source = read(reference / 'binding_transport_s4/predictions' / path.name)
            if set(rows) != set(ids) or not set(ids) <= set(source):
                raise ValueError('Actual or source coverage differs')
            scope, k = path.stem.split('_target_')
            ranking_path = folder.parent / 'ranking' / f'ranking_{scope}.json'
            ranking = json.loads(ranking_path.read_text())
            sources[str(ranking_path)] = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
            if ranking['validation_ids_used']:
                raise ValueError('Validation labels used for head selection')
            expected_heads = {(h['layer'],h['head']) for h in ranking['ranking'][:int(k)]}
            error = 0.
            changed = 0
            for eid in ids:
                row, previous = rows[eid], source[eid]
                for left, right in [(row,previous),(baseline[eid],old_baseline[eid])]:
                    if left['status'] != 'ok' or right['status'] != 'ok':
                        raise ValueError('Failed native prediction')
                    if left['prompt'] != right['prompt'] or left['token_audit']['input_ids_sha256'] != right['token_audit']['input_ids_sha256']:
                        raise ValueError('Prompt or padding differs')
                    for key in ['native_class_logits_positive','native_class_logits_negative']:
                        if left.get(key) != right.get(key):
                            raise ValueError('Changing scalar gain changed a native forward branch')
                for field in ['attention_diagnostics','negative_attention_diagnostics']:
                    actual = {(int(layer),head) for layer,item in row[field].items() for head in item['heads']}
                    if actual != expected_heads or len(actual) != int(k):
                        raise ValueError('Actual head budget differs')
                reconstructed = derive(previous,alpha)
                error = max(error,float(np.max(np.abs(np.asarray(reconstructed['native_class_probabilities'])-row['native_class_probabilities']))))
                if reconstructed['reward'] != row['reward'] or row['contrast_weight'] != alpha:
                    raise ValueError('Actual native reward differs from frozen gain formula')
                changed += bool(np.max(np.abs(np.asarray(row['native_class_probabilities'])-previous['native_class_probabilities'])) > 1e-6)
            if error >= 1e-5:
                raise ValueError('Probability reconstruction exceeds frozen tolerance')
            checks.append(dict(experiment=experiment,condition=path.stem,alpha=alpha,n=len(ids),
                               baseline_positive_negative_exact=True,head_budget=int(k),max_probability_error=error,
                               rows_with_changed_combined_probabilities=changed))
    destination = OUT / 'audit' / time.strftime('uniform_gain_actual_%Y%m%d_%H%M%S.json')
    create_json(destination,dict(status='pass',checks=checks,sources_sha256=sources,labels_read=False,
                                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(destination)
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
