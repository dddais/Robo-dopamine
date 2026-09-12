"""Round18 same-group actual-forward audit before full profiling expansion."""
import argparse
import hashlib
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['qwen', 'roboreward'], required=True)
    args = parser.parse_args()
    checks, sources = [], {}
    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path)
    for model in args.models:
        folder = OUT / 'experiments' / f'{model}_image_text_uniform_method_profile' / 'discovery_smoke'
        reference = OUT / 'experiments' / f'{model}_image_text_uniform_binding_evidence_a1' / 'discovery_smoke'
        events = [json.loads(l) for l in (folder / 'worker_events.jsonl').read_text().splitlines()]
        if not events or events[-1]['event'] != 'complete':
            raise ValueError('Wait for the actual smoke to complete')
        ids = json.loads((folder / 'requested_ids.json').read_text())
        if len(ids) != 8:
            raise ValueError('Require exactly eight matched samples')
        base = read(folder / 'predictions/baseline.jsonl')
        old_base = read(reference / 'predictions/baseline.jsonl')
        if set(base) != set(ids) or set(old_base) != set(ids):
            raise ValueError('Baseline sample coverage differs')
        for eid in ids:
            for field in ['native_class_logits_positive', 'native_class_logits_negative', 'prompt', 'candidate_token_ids']:
                if base[eid][field] != old_base[eid][field]:
                    raise ValueError('Unsteered native baseline differs')
            if base[eid]['token_audit']['input_ids_sha256'] != old_base[eid]['token_audit']['input_ids_sha256']:
                raise ValueError('Baseline padding differs')
        for scope in ['all_frames', 'last_frame']:
            rank_path = OUT / 'functional_selections' / f'stage8_{model}_v1' / f'{model}_image_text' / f'ranking_{scope}.json'
            sources[str(rank_path)] = hashlib.sha256(rank_path.read_bytes()).hexdigest()
            original = json.loads(rank_path.read_text())
            layer = original['layer_profiles'][0]['layer']
            rows = read(folder / f'binding_transport_s4_layer{layer}' / f'predictions/{scope}_target_8.jsonl')
            previous = read(reference / f'binding_transport_s4/predictions/{scope}_target_8.jsonl')
            expected_heads = {(h['layer'], h['head']) for h in original['ranking'][:8]}
            if set(rows) != set(ids) or set(previous) != set(ids):
                raise ValueError('Intervention sample coverage differs')
            error = 0.
            for eid in ids:
                row, old = rows[eid], previous[eid]
                if row['status'] != 'ok' or old['status'] != 'ok':
                    raise ValueError('Failed actual output')
                for field in ['native_class_logits_positive', 'native_class_logits_negative', 'prompt', 'candidate_token_ids']:
                    if row[field] != old[field]:
                        raise ValueError('Same-layer true contrast no longer matches original smoke')
                if row['token_audit']['input_ids_sha256'] != old['token_audit']['input_ids_sha256']:
                    raise ValueError('Matched input tokens differ')
                for field in ['attention_diagnostics', 'negative_attention_diagnostics']:
                    actual = {(int(l), h) for l, d in row[field].items() for h in d['heads']}
                    if actual != expected_heads:
                        raise ValueError('Actual selected head group differs')
                positive = np.asarray(row['native_class_logits_positive'], dtype=np.float32)
                negative = np.asarray(row['native_class_logits_negative'], dtype=np.float32)
                if positive.shape != (5,) or negative.shape != (5,):
                    raise ValueError('Require all five native classes')
                z = (2 * positive - negative).astype(float)
                p = np.exp(z - z.max()); p /= p.sum()
                error = max(error, float(np.max(np.abs(p - row['native_class_probabilities']))))
                if row['reward'] != int(p.argmax()) + 1 or row['progress'] != (row['reward'] - 1) / 4:
                    raise ValueError('Output changed from native five-way argmax')
            if error >= 1e-5:
                raise ValueError('Native probability composition differs')
            checks.append(dict(model=model, scope=scope, original_best_layer=layer, n=8,
                baseline_positive_negative_exact=True, actual_heads_exact=True, max_probability_error=error))
    output = OUT / 'audit' / time.strftime('matched_profile_actual_%Y%m%d_%H%M%S.json')
    create_json(output, dict(status='pass', labels_read=False, checks=checks, sources_sha256=sources,
        scope='Original best eight-head group in each scope; auxiliary smoke layers not claimed as matched-reference comparisons'))
    print(output)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
