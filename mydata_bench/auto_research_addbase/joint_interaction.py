"""Frozen, complete-data gates for round23 actual 2x2 interaction evidence."""
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .durable_scheduler import include_additions, validate_plan
from .compare_native_bias import same_input


MODELS = ['qwen', 'roboreward']
PROTOCOLS = ['image_text', 'text_image', 'interleaved', 'video_text', 'text_video']
SCOPES = ['all_frames', 'last_frame']
NEIGHBORHOODS = {8:[4,8,12], 32:[24,32,40], 64:[48,64,80]}
VARIANT = 'joint_interaction_evidence_a1'
POLICY = OUT / 'selection_joint_interaction_discovery_v1.json'
QUEUE = OUT / 'queue_gpu01_20260911_1700'
CELLS = {'pp':(4., 'binding_transport'), 'mp':(-4., 'binding_transport'),
         'pm':(4., 'binding_task_suppression'), 'mm':(-4., 'binding_task_suppression')}


def folder(model, protocol, population):
    return OUT / 'experiments' / f'{model}_{protocol}_{VARIANT}' / population


def rank_path(model, protocol, scope):
    return OUT / 'functional_selections' / f'stage8_{model}_v1' / f'{model}_{protocol}' / f'ranking_{scope}.json'


def config_check(model, protocol, cfg):
    expected = dict(contrast_negative_mode='joint_interaction', contrast_weight=1,
        negative_strength=4, negative_task_strength=4, task_binding_fraction=.5,
        task_binding_distribution='uniform', ranking_prefix='ANSWER: ',
        frozen_ranking_source=str(rank_path(model, protocol, 'all_frames').parent))
    if any(cfg.get(k) != v for k,v in expected.items()) or cfg.get('visual_mass_partition','global') != 'global':
        raise ValueError('Frozen joint interaction configuration differs')


def verify_rows(rows, ids, expected_heads=None, baseline=False):
    if set(rows) != set(ids) or any(r['status'] != 'ok' for r in rows.values()):
        raise ValueError('Exact complete valid population required')
    def vector(x):
        a = np.asarray(x, dtype=np.float32)
        if a.shape != (5,) or not np.isfinite(a).all():
            raise ValueError('Require finite five-class native logits')
        return a
    for row in rows.values():
        if (row['contrast_negative_mode'] != 'joint_interaction'
                or row['readout'] != 'five_way_answer_likelihood_joint_interaction'
                or len(set(row['candidate_token_ids'])) != 5):
            raise ValueError('Wrong native likelihood readout')
        positive = vector(row['native_class_logits_positive'])
        if baseline:
            if (row['actual_forward_branches'] != 1 or row['interaction_weight'] != 0
                    or row['native_cell_logits'] or row['cell_attention_diagnostics']
                    or row['native_interaction_logits'] is not None or row['condition'] != 'baseline'):
                raise ValueError('Baseline must be one unsteered actual forward')
            z = positive
        else:
            if (row['actual_forward_branches'] != 4 or row['interaction_weight'] != 1
                    or set(row['native_cell_logits']) != set(CELLS)
                    or set(row['cell_attention_diagnostics']) != set(CELLS)
                    or row['composition'] != 'pp+(pp-mp-pm+mm)'):
                raise ValueError('Missing actual 2x2 cell or changed composition')
            cells = {k:vector(v) for k,v in row['native_cell_logits'].items()}
            effect = cells['pp']-cells['mp']-cells['pm']+cells['mm']
            z = cells['pp']+effect
            if (not np.array_equal(cells['pp'],positive)
                    or not np.array_equal(effect,vector(row['native_interaction_logits']))):
                raise ValueError('Recorded interaction differs from actual cell arithmetic')
            for cell,(strength,method) in CELLS.items():
                diagnostics = row['cell_attention_diagnostics'][cell]
                heads = {(int(layer),h) for layer,d in diagnostics.items() for h in d['heads']}
                if not heads or (expected_heads is not None and heads != expected_heads):
                    raise ValueError('Actual four-cell heads differ from frozen ranking')
                for d in diagnostics.values():
                    if (d['method'] != method or d['strength'] != strength
                            or not d['domain_mass_preserved'] or not d['text_domain_mass_preserved']
                            or not d['causal_mask_preserved'] or not d['all_query_rows'] or d['prefill_calls'] < 1):
                        raise ValueError('Actual cell operator is not the registered intervention')
                    if method == 'binding_transport':
                        if d['task_binding_fraction'] != .5 or d['task_binding_distribution'] != 'uniform':
                            raise ValueError('Task positive cell differs')
                    elif d['task_logit_strength'] != -4 or d['task_binding_distribution'] != 'exponential_suppression':
                        raise ValueError('Task negative cell differs')
        if not np.array_equal(z,vector(row['native_class_logits_combined'])):
            raise ValueError('Combined logits differ')
        p = np.exp(z.astype(float)-float(z.max())); p /= p.sum()
        recorded = vector(row['native_class_probabilities'])
        if (np.max(np.abs(p-recorded)) >= 1e-5 or row['reward'] != int(p.argmax())+1
                or row['progress'] != (row['reward']-1)/4):
            raise ValueError('Recorded native probabilities/reward differ from four-cell formula')
    return rows


def complete(root, ks, scopes, n):
    events = root / 'worker_events.jsonl'
    if not events.exists(): return False
    last = json.loads(events.read_text().splitlines()[-1])
    args = last.get('arguments',{})
    if (last.get('event') != 'complete' or args.get('ks') != ks or args.get('scopes') != scopes
            or args.get('controls') != ['target'] or args.get('variant') != VARIANT):
        return False
    ids = json.loads((root / 'requested_ids.json').read_text())
    if len(ids) != n or len(set(ids)) != n: return False
    paths = [root/'predictions/baseline.jsonl'] + [root/'binding_transport_s4/predictions'/f'{s}_target_{k}.jsonl'
        for s in scopes for k in ks]
    return all(p.exists() and set(latest(p)) == set(ids) for p in paths)


def command(model, protocols, population, ks, scopes=SCOPES, smoke=False):
    result = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m',
        'mydata_bench.auto_research_addbase.worker','--model',model,'--protocols',*protocols,
        '--variant',VARIANT,'--methods','binding_transport','--task-binding-fraction','.5',
        '--task-binding-distribution','uniform','--ranking-prefix','ANSWER: ','--contrast-weight','1',
        '--contrast-negative-mode','joint_interaction','--negative-strength','4','--negative-task-strength','4',
        '--strengths','4','--ks',*map(str,ks),'--scopes',*scopes,'--population',population,
        '--frozen-ranking-root',str(rank_path(model,protocols[0],'all_frames').parents[1])]
    if smoke: result += ['--limit','8']
    return result


def write_addition(name, jobs, sources):
    existing = include_additions(json.loads((QUEUE/'plan.json').read_text()),QUEUE)
    validate_plan(dict(jobs=list(existing.values())+jobs))
    path = QUEUE/'additions'/name
    create_json(path,dict(jobs=jobs,sources_sha256=sources,
        interpretation='Registered actual four-cell interaction; physical GPU0/1 only'))
    if not path.with_suffix('.ready').exists():
        with path.with_suffix('.ready').open('x') as f:
            f.write('Frozen round23 gates and complete sources; physical GPU0/1 only\n')
    return path


def register_smoke(cpu_audit):
    audit = json.loads(cpu_audit.read_text())
    if audit['status'] != 'pass' or audit['labels_read'] or audit['interaction_tests'] != 7:
        raise ValueError('Require numerical, executed-cell and prior method regression tests before GPU')
    for path,digest in audit['source_code_sha256'].items():
        if sha(path) != digest: raise ValueError('Implementation changed after the CPU audit')
    jobs = [dict(name=f'smoke_{m}_joint_interaction',gpu=i,min_free_mb=23000,depends_on=[],
        command=command(m,['image_text','text_video'],'discovery',[8,32],smoke=True)) for i,m in enumerate(MODELS)]
    return write_addition('stage23_joint_interaction_smoke.json',jobs,{str(cpu_audit):sha(cpu_audit),str(POLICY):sha(POLICY)})


def smoke_ready():
    return all(complete(folder(m,p,'discovery_smoke'),[8,32],SCOPES,8) for m in MODELS for p in ['image_text','text_video'])


def verify_smoke(destination):
    if not smoke_ready(): raise ValueError('Wait for all actual four-cell smoke conditions')
    sources = {str(POLICY):sha(POLICY)}; checks = []
    def read(path, rows=False):
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    for m in MODELS:
        for protocol in ['image_text','text_video']:
            root = folder(m,protocol,'discovery_smoke')
            reference = OUT/'experiments'/f'{m}_{protocol}_uniform_factorized_evidence_a1'/'discovery'
            ids = read(root/'requested_ids.json')
            if ids != read(reference/'requested_ids.json')[:8]: raise ValueError('Changed smoke batch')
            config_check(m,protocol,read(root/'runtime_config.json'))
            baseline = verify_rows(read(root/'predictions/baseline.jsonl',True),ids,baseline=True)
            old_baseline = read(reference/'predictions/baseline.jsonl',True)
            for e in ids:
                if (not same_input(baseline[e],old_baseline[e]) or
                        baseline[e]['native_class_logits_positive'] != old_baseline[e]['native_class_logits_positive']):
                    raise ValueError('Baseline does not exactly replay the original same input batch')
            for scope in SCOPES:
                rank = read(rank_path(m,protocol,scope))['ranking']
                for k in [8,32]:
                    condition = f'{scope}_target_{k}'
                    rows = verify_rows(read(root/'binding_transport_s4/predictions'/f'{condition}.jsonl',True),ids,
                        {(h['layer'],h['head']) for h in rank[:k]})
                    old = read(reference/'binding_transport_s4/predictions'/f'{condition}.jsonl',True)
                    for e in ids:
                        a,b = rows[e],old[e]
                        if not same_input(a,b) or not same_input(a,baseline[e]): raise ValueError('Cell input differs')
                        if (a['native_cell_logits']['pp'] != b['native_class_logits_positive'] or
                                a['native_cell_logits']['pm'] != b['native_class_logits_negative_task']):
                            raise ValueError('Shared pp/pm cells do not exactly replay actual factorized forwards')
                        if a['condition'] != f'{scope}:target:{k}': raise ValueError('Wrong recorded condition')
                    checks.append(dict(model=m,protocol=protocol,scope=scope,k=k,n=8,
                        pp_pm_actual_replays_exact=True,actual_four_cells_heads_and_formula_verified=True,baseline_exact=True))
    create_json(destination,dict(status='pass',labels_read=False,checks=checks,sources_sha256=sources,
        actual_intervention_rows_verified=128,actual_cell_forward_observations=512,
        interpretation='Actual four-cell execution and pp/pm replay; local attention invariants verified on CPU. No efficacy labels read.'))


def register_discovery(audit_path):
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'pass' or audit['labels_read'] or len(audit['checks']) != 16:
        raise ValueError('All actual smoke checks must pass')
    verify_sources(audit['sources_sha256'])
    jobs = [dict(name=f'round23_{m}_joint_interaction_discovery',gpu=i,min_free_mb=23000,
        depends_on=[f'smoke_{m}_joint_interaction'],command=command(m,PROTOCOLS,'discovery',[8,32,64]))
        for i,m in enumerate(MODELS)]
    return write_addition('stage23_joint_interaction_discovery.json',jobs,{str(audit_path):sha(audit_path),str(POLICY):sha(POLICY)})


def matrix_ready():
    return all(complete(folder(m,p,'discovery'),[8,32,64],SCOPES,70) for m in MODELS for p in PROTOCOLS)


def select(destination):
    if not matrix_ready(): raise ValueError('Wait for all complete 60 independent conditions')
    splits = json.loads((OUT/'splits.json').read_text()); ids = splits['discovery']
    if len(ids) != 70 or set(ids)&set(splits['validation']): raise ValueError('Changed discovery boundary')
    spec = json.loads(POLICY.read_text())
    if (spec['variant'] != VARIANT or spec['protocols'] != PROTOCOLS or spec['scopes'] != SCOPES
            or spec['formula'] != 'z_pp + (z_pp - z_mp - z_pm + z_mm)'):
        raise ValueError('Frozen policy differs')
    matrices = {}; sources = {str(POLICY):sha(POLICY)}
    def read(path,rows=False):
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    for m in MODELS:
        for p in PROTOCOLS:
            root = folder(m,p,'discovery')
            config_check(m,p,read(root/'runtime_config.json'))
            if set(read(root/'requested_ids.json')) != set(ids): raise ValueError('Changed actual population')
            base = verify_rows(read(root/'predictions/baseline.jsonl',True),ids,baseline=True)
            expected = {f'{s}_target_{k}.jsonl' for s in SCOPES for k in NEIGHBORHOODS}
            if {f.name for f in (root/'binding_transport_s4/predictions').glob('*.jsonl')} != expected:
                raise ValueError('Require exactly six intervention files per input')
            conditions = {}
            for scope in SCOPES:
                rank = read(rank_path(m,p,scope))['ranking']
                for k in NEIGHBORHOODS:
                    heads = {(h['layer'],h['head']) for h in rank[:k]}
                    if len(heads) != k: raise ValueError('Duplicate fixed heads')
                    rows = verify_rows(read(root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True),ids,heads)
                    if any(not same_input(rows[e],base[e]) or rows[e]['condition'] != f'{scope}:target:{k}' for e in ids):
                        raise ValueError('Input or recorded condition differs')
                    conditions[(scope,k)] = rows
            matrices[(m,p)] = base,conditions
    # All actual rows and formula checks precede reading discovery scoring labels.
    all_labels = json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels = {e:all_labels[e] for e in ids}; del all_labels
    points = []; proposed = []
    for (m,p),(baseline,conditions) in matrices.items():
        base = summary(baseline,labels,ids); passing = []
        for (scope,k),rows in conditions.items():
            score = summary(rows,labels,ids)
            delta = {c:score['accuracy']['0.125/0.875'][c]['rate_all_expected']-
                base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all','suc','fail']}
            gate = score['mae'] < base['mae'] and delta['all'] >= .1-1e-12 and min(delta['suc'],delta['fail']) > 0
            point = dict(model=m,protocol=p,scope=scope,center_k=k,variant=VARIANT,method='binding_transport_s4',
                mae=score['mae'],baseline_mae=base['mae'],deltas=delta,passes=gate,
                accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'])
            points.append(point)
            if gate: passing.append(point)
        if passing:
            best = min(passing,key=lambda x:(x['mae'],-x['accuracy_all'],x['scope']!='all_frames',x['center_k']))
            proposed.append(dict(best,ks=NEIGHBORHOODS[best['center_k']]))
    counts = {m:sum(p['model']==m for p in proposed) for m in MODELS}; eligible = min(counts.values()) >= 3
    create_json(destination,dict(created_at=time.time(),variant=VARIANT,all_discovery_points=points,
        proposed_input_candidates=proposed,input_counts=counts,family_eligible_for_full=eligible,
        selected=proposed if eligible else [],sources_sha256=sources,validation_labels_used=[],
        actual_intervention_rows_verified=4200,independent_conditions=60,actual_cell_forward_observations=16800,
        interpretation='One fixed four-cell positive-anchored interaction; shared family gate; no cross-method pooling. Adaptive dataset-internal exploration.'))


def register_full(selection,audit_path):
    record = json.loads(selection.read_text()); audit = json.loads(audit_path.read_text())
    eligible = min(record['input_counts'].values()) >= 3
    if (record['family_eligible_for_full'] != eligible or
            record['selected'] != (record['proposed_input_candidates'] if eligible else [])):
        raise ValueError('Changed frozen family selection')
    if audit['status'] != 'pass' or audit['labels_read'] or len(audit['checks']) != 16:
        raise ValueError('Missing actual smoke gate')
    verify_sources(dict(record['sources_sha256'],**audit['sources_sha256']))
    jobs = []; previous = {}
    for p in record['selected']:
        m = p['model']; name = f"validate_{m}_{p['protocol']}_joint_interaction"
        jobs.append(dict(name=name,gpu=MODELS.index(m),min_free_mb=23000,
            depends_on=[previous.get(m,f'round23_{m}_joint_interaction_discovery')],
            command=command(m,[p['protocol']],'full_cohort',p['ks'],[p['scope']])))
        previous[m] = name
    return write_addition('stage23_joint_interaction_validation.json',jobs,{str(selection):sha(selection),str(audit_path):sha(audit_path)})
