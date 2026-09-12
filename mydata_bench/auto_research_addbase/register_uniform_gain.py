"""Register only previously frozen full candidates whose gain smoke passed."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .prepare import OUT
from .durable_scheduler import include_additions, validate_plan
from mydata_bench.addbase_eval.prepare import create_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen','roboreward'], required=True)
    parser.add_argument('--audits', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    selection_path = OUT / f'selection_uniform_gain_{args.model}_full_v1.json'
    selected = json.loads(selection_path.read_text())
    source = Path(selected['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != selected['source_sha256']:
        raise ValueError('Frozen discovery selection source changed')
    passed = set()
    for path in args.audits:
        if not path.resolve().is_relative_to((OUT / 'audit').resolve()):
            raise ValueError('Audit outside research session')
        audit = json.loads(path.read_text())
        if audit['status'] != 'pass' or audit['labels_read']:
            raise ValueError('Invalid actual-forward audit')
        for source, sha in audit['sources_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != sha:
                raise ValueError('Smoke source changed after verification')
        by_gain = {}
        for check in audit['checks']:
            if check['experiment'].startswith(args.model + '_') and check['n'] == 8 and check['baseline_positive_negative_exact']:
                by_gain.setdefault(check['alpha'],set()).add(check['condition'])
        expected = {f'{scope}_target_{k}' for scope in ['all_frames','last_frame'] for k in [8,32]}
        passed.update(alpha for alpha, conditions in by_gain.items() if conditions == expected)
    if {p['alpha'] for p in selected['selected']} - passed:
        raise ValueError('Wait for every selected model/gain to pass all four actual smoke conditions')
    queue = OUT / 'queue_gpu01_20260911_1700'
    existing = include_additions(json.loads((queue / 'plan.json').read_text()),queue)
    jobs = []
    previous = None
    for point in selected['selected']:
        command = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m','mydata_bench.auto_research_addbase.worker',
                   '--model',args.model,'--protocols',point['protocol'],'--variant',point['variant'],'--methods','binding_transport',
                   '--task-binding-fraction','.5','--task-binding-distribution','uniform','--ranking-prefix','ANSWER: ',
                   '--contrast-weight',str(point['alpha']),'--negative-strength','4','--strengths','4',
                   '--ks',*map(str,point['ks']),'--scopes',point['scope'],'--population','full_cohort',
                   '--frozen-ranking-root',str(OUT / 'functional_selections' / f'stage8_{args.model}_v1')]
        name = f"validate_{args.model}_{point['protocol']}_uniform_gain_{str(point['alpha']).replace('.','p')}"
        jobs.append(dict(name=name,command=command,gpu=0 if args.model=='qwen' else 1,min_free_mb=23000,
                         depends_on=[previous] if previous else []))
        previous = name
    validate_plan(dict(jobs=list(existing.values())+jobs))
    destination = queue / 'additions' / f'stage16_{args.model}_uniform_gain_validation.json'
    create_json(destination,dict(created_at=time.time(),selection=str(selection_path),
        selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        actual_audits_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.audits},jobs=jobs))
    with destination.with_suffix('.ready').open('x') as f:
        f.write('Actual smoke verified; original frozen selection unchanged; GPU0/1 only\n')
    print(destination)
    print('Registered',len(jobs),'previously frozen full-cohort jobs')


if __name__ == '__main__':
    main()
