"""Joint round20 selection; both five-input discovery matrices must be complete."""
import hashlib
import json
from pathlib import Path
import time

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .select_robust_discovery import PROTOCOLS, NEIGHBORHOODS, verify_native
from .durable_scheduler import include_additions, validate_plan


VARIANT = 'uniform_temporal_evidence_a1'


def completed(folder):
    path = folder / 'worker_events.jsonl'
    if not path.exists():
        return False
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            return False
    return bool(events and events[-1]['event'] == 'complete')


def matrix_complete():
    return all(completed(OUT / 'experiments' / f'{model}_{protocol}_{VARIANT}' / 'discovery')
               for model in ['qwen','roboreward'] for protocol in PROTOCOLS)


def select(destination):
    if not matrix_complete():
        raise ValueError('Wait for both complete five-input temporal discovery matrices')
    splits = json.loads((OUT / 'splits.json').read_text())
    ids = splits['discovery']
    if len(ids) != 70 or set(ids) & set(splits['validation']):
        raise ValueError('Frozen discovery boundary differs')
    sources = {}
    def record(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    policy = record(OUT / 'selection_temporal_planes_discovery_v1.json')
    spec = json.loads(policy.read_text())
    if spec['variant'] != VARIANT or spec['discovery_ks'] != [8,32,64] or spec['protocols'] != PROTOCOLS:
        raise ValueError('Frozen temporal selection policy differs')
    matrices = {}
    for model in ['qwen','roboreward']:
        for protocol in PROTOCOLS:
            folder = OUT / 'experiments' / f'{model}_{protocol}_{VARIANT}' / 'discovery'
            cfg = json.loads(record(folder / 'runtime_config.json').read_text())
            rank_root = OUT / 'functional_selections' / f'stage8_{model}_v1' / f'{model}_{protocol}'
            expected = dict(visual_mass_partition='temporal_planes',contrast_weight=1,negative_strength=4,
                task_binding_fraction=.5,task_binding_distribution='uniform',ranking_prefix='ANSWER: ',
                frozen_ranking_source=str(rank_root))
            if any(cfg.get(key) != value for key,value in expected.items()):
                raise ValueError('Frozen temporal settings differ')
            if set(json.loads(record(folder / 'requested_ids.json').read_text())) != set(ids):
                raise ValueError('Discovery population differs')
            record(folder / 'worker_events.jsonl')
            baseline = verify_native(latest(record(folder / 'predictions/baseline.jsonl')),ids,True)
            paths = list(folder.glob('binding_transport_s4/predictions/*.jsonl'))
            if {p.name for p in paths} != {f'{s}_target_{k}.jsonl' for s in ['all_frames','last_frame'] for k in NEIGHBORHOODS}:
                raise ValueError('Expected exactly six independent conditions')
            rankings = {scope:json.loads(record(rank_root / f'ranking_{scope}.json').read_text())['ranking']
                        for scope in ['all_frames','last_frame']}
            conditions = {}
            for path in paths:
                scope,k = path.stem.split('_target_'); k=int(k)
                heads = {(h['layer'],h['head']) for h in rankings[scope][:k]}
                if len(heads) != k:
                    raise ValueError('Duplicate head')
                rows = verify_native(latest(record(path)),ids,expected_heads=heads)
                for row in rows.values():
                    for field in ['attention_diagnostics','negative_attention_diagnostics']:
                        if any(d.get('visual_mass_partition') != 'temporal_planes' for d in row[field].values()):
                            raise ValueError('Actual controller lacks the registered partition')
                conditions[(scope,k)] = rows
            matrices[(model,protocol)] = (baseline,conditions)
    # All ten inputs are complete and structurally verified before scoring labels.
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {eid:all_labels[eid] for eid in ids}; del all_labels
    points,proposed = [],[]
    for (model,protocol),(baseline,conditions) in matrices.items():
        base = summary(baseline,labels,ids); passing = []
        for (scope,k),rows in sorted(conditions.items()):
            score = summary(rows,labels,ids)
            delta = {c:score['accuracy']['0.125/0.875'][c]['rate_all_expected'] -
                     base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all','suc','fail']}
            gate = score['mae'] < base['mae'] and delta['all'] >= .1-1e-12 and min(delta['suc'],delta['fail']) > 0
            point = dict(model=model,protocol=protocol,variant=VARIANT,method='binding_transport_s4',
                scope=scope,center_k=k,mae=score['mae'],baseline_mae=base['mae'],deltas=delta,passes=gate,
                accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'])
            points.append(point)
            if gate:
                passing.append(point)
        if passing:
            best = min(passing,key=lambda p:(p['mae'],-p['accuracy_all'],p['center_k'],p['scope']))
            proposed.append(dict(best,ks=NEIGHBORHOODS[best['center_k']]))
    counts = {model:sum(p['model']==model for p in proposed) for model in ['qwen','roboreward']}
    eligible = min(counts.values()) >= 3
    create_json(destination,dict(created_at=time.time(),variant=VARIANT,all_discovery_points=points,
        proposed_input_candidates=proposed,input_counts=counts,family_eligible_for_full=eligible,
        selected=proposed if eligible else [],sources_sha256=sources,validation_labels_used=[],
        independent_conditions=60,actual_intervention_rows_verified=4200,
        interpretation='Frozen joint family screen after both five-input matrices. No eligible family means no full GPU expansion. '
            'Last-frame single-plane results can equal original uniform results; they are not new independent replications. '
            'Dataset-internal adaptive exploration, no final efficacy guarantee.'))


def register(destination,audit_path):
    selection=json.loads(destination.read_text()); audit=json.loads(audit_path.read_text())
    if (selection['family_eligible_for_full'] != (min(selection['input_counts'].values())>=3) or
            selection['selected'] != (selection['proposed_input_candidates'] if selection['family_eligible_for_full'] else [])):
        raise ValueError('Frozen joint coverage gate and selected jobs disagree')
    if audit['status']!='pass' or audit['labels_read'] or len(audit['checks'])!=16:
        raise ValueError('Both-model actual smoke must have passed')
    for source,sha in dict(selection['sources_sha256'],**audit['sources_sha256']).items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=sha:
            raise ValueError('A frozen source changed')
    queue=OUT/'queue_gpu01_20260911_1700';jobs=[];previous={}
    for point in selection['selected']:
        model=point['model'];name=f"validate_{model}_{point['protocol']}_temporal_planes"
        command=['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m',
            'mydata_bench.auto_research_addbase.worker','--model',model,'--protocols',point['protocol'],
            '--variant',VARIANT,'--methods','binding_transport','--task-binding-fraction','.5',
            '--task-binding-distribution','uniform','--visual-mass-partition','temporal_planes',
            '--ranking-prefix','ANSWER: ','--contrast-weight','1','--negative-strength','4','--strengths','4',
            '--ks',*map(str,point['ks']),'--scopes',point['scope'],'--population','full_cohort',
            '--frozen-ranking-root',str(OUT/'functional_selections'/f'stage8_{model}_v1')]
        jobs.append(dict(name=name,command=command,gpu=0 if model=='qwen' else 1,min_free_mb=23000,
            depends_on=[previous[model]] if model in previous else [f'round20_{model}_temporal_discovery']))
        previous[model]=name
    existing=include_additions(json.loads((queue/'plan.json').read_text()),queue)
    validate_plan(dict(jobs=list(existing.values())+jobs))
    path=queue/'additions/stage20_temporal_planes_validation.json'
    create_json(path,dict(created_at=time.time(),selection=str(destination),
        selection_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),actual_audit=str(audit_path),jobs=jobs))
    with path.with_suffix('.ready').open('x') as handle:
        handle.write('Both complete matrices, joint frozen screen and actual smoke verified; GPU0/1 only\n')
    return path
