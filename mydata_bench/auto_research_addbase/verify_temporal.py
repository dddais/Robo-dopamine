"""Actual round20 two-model smoke gate; optional frozen discovery registration."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .select_robust_discovery import PROTOCOLS, verify_native
from .temporal import temporal_domains
from .durable_scheduler import include_additions, validate_plan


def verify(destination):
    checks, sources = [], {}
    def record(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    for model in ['qwen','roboreward']:
        for protocol in ['image_text','text_video']:
            folder = OUT / 'experiments' / f'{model}_{protocol}_uniform_temporal_evidence_a1' / 'discovery_smoke'
            reference = OUT / 'experiments' / f'{model}_{protocol}_uniform_binding_evidence_a1' / 'discovery'
            events = [json.loads(line) for line in record(folder / 'worker_events.jsonl').read_text().splitlines()]
            if not events or events[-1]['event'] != 'complete':
                raise ValueError('Wait for both models and both smoke inputs to finish')
            ids = json.loads(record(folder / 'requested_ids.json').read_text())
            original_ids = json.loads(record(reference / 'requested_ids.json').read_text())
            if len(ids) != 8 or ids != original_ids[:8]:
                raise ValueError('Require the exact first original discovery batch')
            cfg = json.loads(record(folder / 'runtime_config.json').read_text())
            expected = dict(visual_mass_partition='temporal_planes',contrast_weight=1,negative_strength=4,
                            task_binding_fraction=.5,task_binding_distribution='uniform')
            if any(cfg.get(k) != value for k,value in expected.items()):
                raise ValueError('Frozen temporal operator differs')
            baseline = verify_native(latest(record(folder / 'predictions/baseline.jsonl')), ids, True)
            old_baseline = latest(record(reference / 'predictions/baseline.jsonl'))
            def matched_input(a,b):
                return all(a[k] == b[k] for k in ['prompt','candidate_token_ids']) and (
                    a['token_audit']['input_ids_sha256'] == b['token_audit']['input_ids_sha256'])
            for eid in ids:
                if (not matched_input(baseline[eid],old_baseline[eid]) or
                        baseline[eid]['native_class_logits_positive'] != old_baseline[eid]['native_class_logits_positive']):
                    raise ValueError('Same-input unsteered baseline differs from original discovery batch')
            for scope in ['all_frames','last_frame']:
                rank_path = OUT / 'functional_selections' / f'stage8_{model}_v1' / f'{model}_{protocol}' / f'ranking_{scope}.json'
                ranking = json.loads(record(rank_path).read_text())['ranking']
                for k in [8,32]:
                    name = f'{scope}_target_{k}.jsonl'
                    heads = {(h['layer'],h['head']) for h in ranking[:k]}
                    rows = verify_native(latest(record(folder / 'binding_transport_s4/predictions' / name)), ids, expected_heads=heads)
                    previous = latest(record(reference / 'binding_transport_s4/predictions' / name))
                    changed = 0; maximum = 0.; counts = []
                    for eid in ids:
                        row, old = rows[eid], previous[eid]
                        if not matched_input(row,old):
                            raise ValueError('Actual smoke input differs from its original batch')
                        n_planes = len(temporal_domains(row['token_audit'],scope)); counts.append(n_planes)
                        for field in ['attention_diagnostics','negative_attention_diagnostics']:
                            for diag in row[field].values():
                                if (diag.get('visual_mass_partition') != 'temporal_planes' or
                                        not diag['per_plane_mass_preserved_by_operator'] or
                                        diag['plane_counts'] != [n_planes]*len(ids) or
                                        diag['method'] != ('binding_transport' if field=='attention_diagnostics' else 'mass_transport') or
                                        diag['strength'] != (4 if field=='attention_diagnostics' else -4)):
                                    raise ValueError('Actual controller lacks the specified partition')
                                if field=='attention_diagnostics' and (diag.get('task_binding_fraction')!=.5 or
                                        diag.get('task_binding_distribution')!='uniform'):
                                    raise ValueError('Actual positive task binding differs')
                        same = True
                        for field in ['native_class_logits_positive','native_class_logits_negative']:
                            maximum = max(maximum,float(np.max(np.abs(np.asarray(row[field])-old[field]))))
                            same = same and row[field] == old[field]
                        changed += not same
                        if scope == 'last_frame' and (n_planes != 1 or not same):
                            raise ValueError('Single-plane actual arithmetic differs from original uniform operator')
                    checks.append(dict(model=model,protocol=protocol,scope=scope,k=k,n=8,
                        baseline_and_input_exact=True,actual_heads_and_partition_verified=True,
                        plane_counts=counts,single_plane_branches_exact=True if scope=='last_frame' else None,
                        branch_changed_rows=changed,max_native_logit_difference_from_global=maximum))
    create_json(destination,dict(status='pass',labels_read=False,checks=checks,sources_sha256=sources,
        interpretation='Two models, two input types, four conditions, 128 actual intervention rows. '
            'Single-plane scope must match old positive and negative logits exactly. '
            'Partition diagnostics identify the executed operator; numeric conservation was tested on CPU, '
            'not separately measured at every actual model query. No efficacy scoring.'))


def register(audit_path):
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'pass' or audit['labels_read'] or len(audit['checks']) != 16:
        raise ValueError('All actual smoke comparisons must pass')
    for path,sha in audit['sources_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('Actual smoke source changed')
    queue = OUT / 'queue_gpu01_20260911_1700'
    jobs = []
    for model,gpu in [('qwen',0),('roboreward',1)]:
        command = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m',
            'mydata_bench.auto_research_addbase.worker','--model',model,'--protocols',*PROTOCOLS,
            '--variant','uniform_temporal_evidence_a1','--methods','binding_transport',
            '--task-binding-fraction','.5','--task-binding-distribution','uniform',
            '--visual-mass-partition','temporal_planes','--ranking-prefix','ANSWER: ',
            '--contrast-weight','1','--negative-strength','4','--strengths','4','--ks','8','32','64',
            '--scopes','all_frames','last_frame','--population','discovery',
            '--frozen-ranking-root',str(OUT / 'functional_selections' / f'stage8_{model}_v1')]
        jobs.append(dict(name=f'round20_{model}_temporal_discovery',gpu=gpu,min_free_mb=23000,command=command,
                         depends_on=[f'smoke_{model}_temporal_planes']))
    existing = include_additions(json.loads((queue / 'plan.json').read_text()),queue)
    validate_plan(dict(jobs=list(existing.values())+jobs))
    path = queue / 'additions/stage20_temporal_planes_discovery.json'
    create_json(path,dict(created_at=time.time(),actual_audit=str(audit_path),
        actual_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),jobs=jobs))
    with path.with_suffix('.ready').open('x') as handle:
        handle.write('Both models and both actual smoke inputs verified; discovery only; physical GPU0/1\n')
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register',action='store_true')
    args = parser.parse_args()
    destination = OUT / 'audit' / time.strftime('temporal_planes_actual_%Y%m%d_%H%M%S.json')
    verify(destination)
    print(destination)
    if args.register:
        register(destination)


if __name__ == '__main__':
    main()
