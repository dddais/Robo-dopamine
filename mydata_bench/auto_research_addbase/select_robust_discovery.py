"""Execute the already frozen round18 full-selection rule after five inputs finish."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .durable_scheduler import include_additions, validate_plan


PROTOCOLS = ['image_text', 'text_image', 'interleaved', 'text_video', 'video_text']
VARIANT = 'uniform_evidence_robustprofile_a1'
NEIGHBORHOODS = {8: [8, 16, 24], 32: [24, 32, 40], 64: [48, 64, 80]}


def verify_native(rows, ids, baseline=False, expected_heads=None, negative_mode='visual'):
    if set(rows) != set(ids) or any(row['status'] != 'ok' for row in rows.values()):
        raise ValueError('Require exact complete prediction coverage')
    for row in rows.values():
        pos = np.asarray(row['native_class_logits_positive'], dtype=np.float32)
        neg = np.asarray(row['native_class_logits_negative'], dtype=np.float32) if not baseline else None
        if any(x.shape != (5,) or not np.isfinite(x).all() for x in ([pos] if baseline else [pos, neg])):
            raise ValueError('Five finite native classes required')
        if row.get('contrast_negative_mode', 'visual') != negative_mode:
            raise ValueError('Actual native negative branch mode differs from the requested verifier')
        z = (pos if baseline else 2*pos-neg).astype(float)
        p = np.exp(z-z.max()); p /= p.sum()
        recorded = np.asarray(row['native_class_probabilities'], dtype=float)
        if recorded.shape != (5,) or not np.isfinite(recorded).all():
            raise ValueError('Recorded probabilities must be finite five-way output')
        if (np.max(np.abs(p-recorded)) >= 1e-5 or
                row['reward'] != int(p.argmax())+1 or row['progress'] != (row['reward']-1)/4):
            raise ValueError('Recorded output differs from the frozen native formula')
        if expected_heads is not None:
            for field in ['attention_diagnostics', 'negative_attention_diagnostics']:
                actual = {(int(layer), h) for layer, item in row[field].items() for h in item['heads']}
                if actual != expected_heads:
                    raise ValueError('Actual forward heads differ from robust ranking')
    return rows


def matrix_complete(model):
    for protocol in PROTOCOLS:
        folder = OUT / 'experiments' / f'{model}_{protocol}_{VARIANT}' / 'discovery'
        events = folder / 'worker_events.jsonl'
        if not events.exists():
            return False
        records = [json.loads(line) for line in events.read_text().splitlines()]
        if not records or records[-1]['event'] != 'complete':
            return False
    return True


def select(model, destination):
    if not matrix_complete(model):
        raise ValueError('Wait for all five complete round18 discovery inputs')
    splits = json.loads((OUT / 'splits.json').read_text())
    ids = splits['discovery']
    if len(ids) != 70 or set(ids) & set(splits['validation']):
        raise ValueError('Frozen discovery boundary differs')
    sources = {}
    def record(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    policy = record(OUT / 'selection_method_matched_profile_discovery_v1.json')
    spec = json.loads(policy.read_text())
    if spec['variant'] != VARIANT or spec['discovery_ks'] != [8, 32, 64] or spec['protocols'] != PROTOCOLS:
        raise ValueError('Frozen round18 policy differs')
    matrices = {}
    for protocol in PROTOCOLS:
        folder = OUT / 'experiments' / f'{model}_{protocol}_{VARIANT}' / 'discovery'
        cfg = json.loads(record(folder / 'runtime_config.json').read_text())
        rank_root = OUT / 'functional_selections' / f'stage18_{model}_robust_v1' / f'{model}_{protocol}'
        expected = dict(contrast_weight=1, negative_strength=4, task_binding_fraction=.5,
                        task_binding_distribution='uniform', ranking_prefix='ANSWER: ',
                        frozen_ranking_source=str(rank_root))
        if any(cfg.get(key) != value for key, value in expected.items()):
            raise ValueError('Frozen round18 settings differ')
        if set(json.loads(record(folder / 'requested_ids.json').read_text())) != set(ids):
            raise ValueError('Population differs')
        record(folder / 'worker_events.jsonl')
        expected_files = {f'{scope}_target_{k}.jsonl' for scope in ['all_frames', 'last_frame'] for k in NEIGHBORHOODS}
        paths = list(folder.glob('binding_transport_s4/predictions/*.jsonl'))
        if {p.name for p in paths} != expected_files:
            raise ValueError('Require exactly six prespecified conditions per input')
        rankings = {scope: json.loads(record(rank_root / f'ranking_{scope}.json').read_text())
                    for scope in ['all_frames', 'last_frame']}
        baseline = verify_native(latest(record(folder / 'predictions/baseline.jsonl')), ids, True)
        conditions = {}
        for path in paths:
            scope, k = path.stem.split('_target_'); k = int(k)
            rank = rankings[scope]
            if rank['validation_ids_used'] or set(rank['example_ids']) != set(ids) or len(rank['ranking']) != 224:
                raise ValueError('Rank selection population or head budget differs')
            heads = {(h['layer'], h['head']) for h in rank['ranking'][:k]}
            if len(heads) != k:
                raise ValueError('Duplicate selected head')
            conditions[(scope, k)] = verify_native(latest(record(path)), ids, expected_heads=heads)
        matrices[protocol] = (baseline, conditions)
    # No scoring labels are opened until every input and actual condition is verified.
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {eid: all_labels[eid] for eid in ids}; del all_labels
    points, selected = [], []
    for protocol, (baseline, conditions) in matrices.items():
        base = summary(baseline, labels, ids)
        candidates = []
        for (scope, k), rows in sorted(conditions.items()):
            score = summary(rows, labels, ids)
            delta = {c: score['accuracy']['0.125/0.875'][c]['rate_all_expected'] -
                     base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all', 'suc', 'fail']}
            passing = score['mae'] < base['mae'] and delta['all'] >= .1-1e-12 and min(delta['suc'], delta['fail']) > 0
            point = dict(model=model, protocol=protocol, variant=VARIANT, method='binding_transport_s4', scope=scope,
                center_k=k, mae=score['mae'], baseline_mae=base['mae'], deltas=delta, passes=passing,
                accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'])
            points.append(point)
            if passing:
                candidates.append(point)
        if candidates:
            best = min(candidates, key=lambda p: (p['mae'], -p['accuracy_all'], p['center_k'], p['scope']))
            selected.append(dict(best, ks=NEIGHBORHOODS[best['center_k']]))
    create_json(destination, dict(created_at=time.time(), model=model, variant=VARIANT, selected=selected,
        all_discovery_points=points, sources_sha256=sources, selection_rule=str(policy),
        validation_labels_used=[], independent_conditions=30, actual_predictions_verified=2100,
        interpretation='Prespecified round18 selection after all five discovery inputs. '
            'Adaptive dataset-internal exploration; neither bootstrap ranking nor selection guarantees full validation.'))


def register(model, destination):
    record = json.loads(destination.read_text())
    audit_path = OUT / 'audit/matched_profile_actual_20260911_184717.json'
    audit = json.loads(audit_path.read_text())
    checks = [c for c in audit['checks'] if c['model'] == model]
    if audit['status'] != 'pass' or audit['labels_read'] or len(checks) != 2:
        raise ValueError('Actual round18 smoke is not verified')
    for path, sha in dict(record['sources_sha256'], **audit['sources_sha256']).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('A frozen source changed')
    queue = OUT / 'queue_gpu01_20260911_1700'
    jobs, previous = [], None
    for point in record['selected']:
        name = f"validate_{model}_{point['protocol']}_robustprofile_evidence"
        command = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
            'mydata_bench.auto_research_addbase.worker', '--model', model, '--protocols', point['protocol'],
            '--variant', VARIANT, '--methods', 'binding_transport', '--task-binding-fraction', '.5',
            '--task-binding-distribution', 'uniform', '--ranking-prefix', 'ANSWER: ', '--contrast-weight', '1',
            '--negative-strength', '4', '--strengths', '4', '--ks', *map(str, point['ks']), '--scopes', point['scope'],
            '--population', 'full_cohort', '--frozen-ranking-root', str(OUT / 'functional_selections' / f'stage18_{model}_robust_v1')]
        jobs.append(dict(name=name, command=command, gpu=0 if model == 'qwen' else 1, min_free_mb=23000,
                         depends_on=[previous] if previous else []))
        previous = name
    addition = queue / 'additions' / f'stage18_{model}_robustprofile_validation.json'
    existing = include_additions(json.loads((queue / 'plan.json').read_text()), queue)
    validate_plan(dict(jobs=list(existing.values()) + jobs))
    create_json(addition, dict(created_at=time.time(), selection=str(destination),
        selection_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), actual_audit=str(audit_path), jobs=jobs))
    with addition.with_suffix('.ready').open('x') as handle:
        handle.write('Complete five-input discovery, frozen selection and actual smoke verified; physical GPU0/1 only\n')
    print(addition, 'jobs', len(jobs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    destination = OUT / f'selection_robustprofile_{args.model}_full_v1.json'
    if not destination.exists():
        select(args.model, destination)
    print(destination)
    for point in json.loads(destination.read_text())['selected']:
        print(point['protocol'], point['scope'], point['ks'], point['mae'], point['deltas'])
    if args.register:
        register(args.model, destination)


if __name__ == '__main__':
    main()
