"""Select round17 only after its full, frozen five-input discovery matrix.

CPU scoring reads discovery labels only. Registration is a separate explicit
option and always uses the immutable selection and smoke audit.
"""
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
VARIANT = 'uniform_factorized_evidence_a1'


def verified_rows(path, ids, sources, baseline=False):
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = latest(path)
    if set(rows) != set(ids) or any(r['status'] != 'ok' for r in rows.values()):
        raise ValueError(f'Incomplete or extra observations: {path}')
    for row in rows.values():
        if row['actual_forward_branches'] != (1 if baseline else 3):
            raise ValueError('Actual forward count differs')
        positive = np.asarray(row['native_class_logits_positive'], dtype=np.float32)
        if positive.shape != (5,) or not np.isfinite(positive).all():
            raise ValueError('Five finite native logits required')
        if baseline:
            z = positive
        else:
            visual = np.asarray(row['native_class_logits_negative'], dtype=np.float32)
            task = np.asarray(row['native_class_logits_negative_task'], dtype=np.float32)
            if any(v.shape != (5,) or not np.isfinite(v).all() for v in [visual, task]):
                raise ValueError('Both actual negative branches required')
            mean = .5 * (visual + task)
            if not row['negative_mean_is_derived'] or not np.array_equal(mean, row['native_class_logits_negative_mean']):
                raise ValueError('Incorrect derived mean')
            z = 2 * positive - mean
        z = z.astype(float)
        p = np.exp(z - z.max()); p /= p.sum()
        if (np.max(np.abs(p - row['native_class_probabilities'])) >= 1e-5 or
                row['reward'] != int(p.argmax()) + 1 or row['progress'] != (row['reward'] - 1) / 4):
            raise ValueError('Recorded five-class output does not reconstruct')
    return rows


def select(model, destination):
    splits = json.loads((OUT / 'splits.json').read_text())
    ids = splits['discovery']
    if len(ids) != 70 or set(ids) & set(splits['validation']):
        raise ValueError('Frozen discovery boundary differs')
    sources = {}
    roots = [OUT / 'experiments' / f'{model}_{p}_{VARIANT}' / 'discovery' for p in PROTOCOLS]
    # Check every completion and setting before opening scoring labels.
    for folder in roots:
        for name in ['worker_events.jsonl', 'runtime_config.json', 'requested_ids.json']:
            path = folder / name
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        events = [json.loads(l) for l in (folder / 'worker_events.jsonl').read_text().splitlines()]
        if not events or events[-1]['event'] != 'complete':
            raise ValueError('Wait for all five complete discovery inputs')
        if set(json.loads((folder / 'requested_ids.json').read_text())) != set(ids):
            raise ValueError('Discovery coverage differs')
        cfg = json.loads((folder / 'runtime_config.json').read_text())
        expected = dict(contrast_weight=1, negative_strength=4, negative_task_strength=4,
                        contrast_negative_mode='visual_and_task', task_binding_fraction=.5,
                        task_binding_distribution='uniform')
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError('Frozen operator differs')
        expected_conditions = {f'{s}_target_{k}.jsonl' for s in ['all_frames', 'last_frame'] for k in [8,32,64]}
        if {p.name for p in folder.glob('binding_transport_s4/predictions/*.jsonl')} != expected_conditions:
            raise ValueError('Require exactly the six prespecified independent conditions')
    policy = OUT / 'selection_factorized_neighborhood_rule_v1.json'
    sources[str(policy)] = hashlib.sha256(policy.read_bytes()).hexdigest()
    neighborhoods = json.loads(policy.read_text())['neighborhoods']
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {e: all_labels[e] for e in ids}
    del all_labels
    points, selected = [], []
    for protocol, folder in zip(PROTOCOLS, roots):
        baseline = summary(verified_rows(folder / 'predictions/baseline.jsonl', ids, sources, True), labels, ids)
        candidates = []
        for path in sorted(folder.glob('binding_transport_s4/predictions/*.jsonl')):
            score = summary(verified_rows(path, ids, sources), labels, ids)
            scope, k = path.stem.split('_target_'); k = int(k)
            deltas = {c: score['accuracy']['0.125/0.875'][c]['rate_all_expected'] -
                         baseline['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all', 'suc', 'fail']}
            passing = score['mae'] < baseline['mae'] and deltas['all'] >= .1 - 1e-12 and min(deltas['suc'], deltas['fail']) > 0
            point = dict(model=model, protocol=protocol, variant=VARIANT, method='binding_transport_s4', scope=scope,
                         center_k=k, mae=score['mae'], baseline_mae=baseline['mae'], deltas=deltas,
                         accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'],
                         passes=passing, source=str(path))
            points.append(point)
            if passing:
                candidates.append(point)
        if candidates:
            best = min(candidates, key=lambda p:(p['mae'], -p['accuracy_all'], p['center_k'], p['scope']))
            selected.append(dict(best, ks=neighborhoods[str(best['center_k'])]))
    create_json(destination, dict(created_at=time.time(), model=model, variant=VARIANT, selected=selected,
        all_discovery_points=points, sources_sha256=sources, selection_rule=str(policy),
        validation_labels_used=[], independent_conditions=30, actual_predictions_verified=2100,
        interpretation='Prespecified within-discovery selection; adaptive dataset-internal exploration, not a held-out final claim'))


def register(model, destination):
    record = json.loads(destination.read_text())
    audit_path = OUT / 'audit/factorized_actual_20260911_180233.json'
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'pass' or audit['labels_read'] or len([c for c in audit['checks'] if c['model'] == model]) != 4:
        raise ValueError('Actual model smoke not verified')
    for source, sha in dict(record['sources_sha256'], **audit['sources_sha256']).items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != sha:
            raise ValueError('A frozen source changed')
    queue = OUT / 'queue_gpu01_20260911_1700'
    addition = queue / 'additions' / f'stage17_{model}_factorized_validation.json'
    jobs, previous = [], None
    for point in record['selected']:
        name = f"validate_{model}_{point['protocol']}_factorized_evidence"
        command = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
            'mydata_bench.auto_research_addbase.worker', '--model', model, '--protocols', point['protocol'],
            '--variant', VARIANT, '--methods', 'binding_transport', '--task-binding-fraction', '.5',
            '--task-binding-distribution', 'uniform', '--ranking-prefix', 'ANSWER: ', '--contrast-weight', '1',
            '--negative-strength', '4', '--strengths', '4', '--contrast-negative-mode', 'visual_and_task',
            '--negative-task-strength', '4', '--ks', *map(str, point['ks']), '--scopes', point['scope'],
            '--population', 'full_cohort', '--frozen-ranking-root', str(OUT / 'functional_selections' / f'stage8_{model}_v1')]
        jobs.append(dict(name=name, command=command, gpu=0 if model == 'qwen' else 1, min_free_mb=23000,
                         depends_on=[previous] if previous else []))
        previous = name
    existing = include_additions(json.loads((queue / 'plan.json').read_text()), queue)
    validate_plan(dict(jobs=list(existing.values()) + jobs))
    create_json(addition, dict(created_at=time.time(), selection=str(destination),
        selection_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), actual_audit=str(audit_path), jobs=jobs))
    with addition.with_suffix('.ready').open('x') as f:
        f.write('Complete discovery, fixed selection, actual smoke verified; physical GPU0/1 only\n')
    print(addition, 'jobs', len(jobs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    destination = OUT / f'selection_factorized_{args.model}_full_v1.json'
    if not destination.exists():
        select(args.model, destination)
    print(destination)
    for point in json.loads(destination.read_text())['selected']:
        print(point['protocol'], point['scope'], point['ks'], point['mae'], point['deltas'])
    if args.register:
        register(args.model, destination)


if __name__ == '__main__':
    main()
