"""Actual replay gate and frozen full jobs for round24 shared alpha selection."""
import json

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .factorized_global_gain import POLICY,MODELS,SCOPES,compose
from .empirical_profile import sha,verify_sources
from .compare_native_bias import heads,same_input
from .durable_scheduler import include_additions,validate_plan


QUEUE=OUT/'queue_gpu01_20260911_1700'
SELECTION=OUT/'selection_factorized_global_gain_full_v1.json'


def selected():
    record=json.loads(SELECTION.read_text())
    gains=[a for a in [.5,2.] if min(record['input_counts'][str(a)].values())>=3]
    gain=min(gains) if gains else None
    if record['selected_gain']!=gain or record['selected']!=(record['proposed_by_gain'][str(gain)] if gain is not None else []):
        raise ValueError('Frozen global gain selection differs')
    verify_sources(record['sources_sha256'])
    return record,gain


def variant(gain):
    if gain not in [.5,2.]:raise ValueError('Unregistered contrast weight')
    return 'uniform_factorized_evidence_a'+('05' if gain==.5 else '2')


def command(model,protocols,population,ks,scopes,gain,smoke=False):
    result=['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m',
        'mydata_bench.auto_research_addbase.worker','--model',model,'--protocols',*protocols,
        '--variant',variant(gain),'--methods','binding_transport','--task-binding-fraction','.5',
        '--task-binding-distribution','uniform','--ranking-prefix','ANSWER: ','--contrast-weight',str(gain),
        '--contrast-negative-mode','visual_and_task','--negative-strength','4','--negative-task-strength','4',
        '--strengths','4','--ks',*map(str,ks),'--scopes',*scopes,'--population',population,
        '--frozen-ranking-root',str(OUT/'functional_selections'/f'stage8_{model}_v1')]
    if smoke:result+=['--limit','8']
    return result


def write_addition(name,jobs,sources):
    previous=include_additions(json.loads((QUEUE/'plan.json').read_text()),QUEUE)
    validate_plan(dict(jobs=list(previous.values())+jobs))
    path=QUEUE/'additions'/name
    create_json(path,dict(jobs=jobs,sources_sha256=sources,
        interpretation='One shared round24 alpha from complete discovery; physicalGPU0/1 only'))
    if not path.with_suffix('.ready').exists():
        with path.with_suffix('.ready').open('x') as f:f.write('Frozen global gain; actual replay required before full; GPU0/1 only\n')
    return path


def register_smoke():
    record,gain=selected()
    jobs=[] if gain is None else [dict(name=f'smoke_{m}_factorized_global_gain',gpu=i,min_free_mb=23000,
        depends_on=[],command=command(m,['image_text','text_video'],'discovery',[8,32],SCOPES,gain,True))
        for i,m in enumerate(MODELS)]
    return write_addition('stage24_factorized_global_gain_smoke.json',jobs,{str(SELECTION):sha(SELECTION),str(POLICY):sha(POLICY)})


def verify_native(rows,ids,gain,expected_heads=None,baseline=False):
    if set(rows)!=set(ids) or any(r['status']!='ok' for r in rows.values()):raise ValueError('Incomplete native observations')
    for row in rows.values():
        if row.get('derived_only') or row['actual_forward_branches']!=(1 if baseline else 3):
            raise ValueError('Require actually executed branches, not copied derived predictions')
        if row['contrast_negative_mode']!='visual_and_task' or row['contrast_weight']!=(0 if baseline else gain):
            raise ValueError('Wrong actual contrast composition')
        pos=np.asarray(row['native_class_logits_positive'],dtype=np.float32)
        if pos.shape!=(5,) or not np.isfinite(pos).all():raise ValueError('Five finite native classes required')
        if baseline:z=pos
        else:
            visual=np.asarray(row['native_class_logits_negative'],dtype=np.float32)
            task=np.asarray(row['native_class_logits_negative_task'],dtype=np.float32)
            if any(x.shape!=(5,) or not np.isfinite(x).all() for x in [visual,task]):raise ValueError('Missing actual negative cell')
            if not row['negative_mean_is_derived'] or not np.array_equal(.5*(visual+task),row['native_class_logits_negative_mean']):
                raise ValueError('Wrong actual mean')
            z=compose(pos,visual,task,gain)
            for field,method,strength in [('attention_diagnostics','binding_transport',4),
                    ('negative_attention_diagnostics','mass_transport',-4),
                    ('task_negative_attention_diagnostics','binding_task_suppression',4)]:
                if expected_heads is not None and heads(row,field)!=expected_heads:raise ValueError('Actual heads differ')
                for d in row[field].values():
                    if d['method']!=method or d['strength']!=strength:raise ValueError('Actual branch operator differs')
                    if method=='binding_transport' and (d['task_binding_fraction']!=.5 or d['task_binding_distribution']!='uniform'):
                        raise ValueError('Actual positive task distribution differs')
                    if method=='binding_task_suppression' and d['task_logit_strength']!=-4:raise ValueError('Actual negative task differs')
        p=np.exp(z.astype(float)-float(z.max()));p/=p.sum()
        recorded=np.asarray(row['native_class_probabilities'],dtype=float)
        if recorded.shape!=(5,) or not np.isfinite(recorded).all() or not np.isfinite(row['progress']):
            raise ValueError('Recorded native probabilities must be finite')
        if (np.max(np.abs(p-recorded))>=1e-5 or row['reward']!=int(p.argmax())+1
                or row['progress']!=(row['reward']-1)/4):raise ValueError('Actual output differs from selected three-branch gain')
    return rows


def smoke_ready():
    record,gain=selected()
    if gain is None:return False
    for m in MODELS:
        for p in ['image_text','text_video']:
            root=OUT/'experiments'/f'{m}_{p}_{variant(gain)}'/'discovery_smoke'
            events=root/'worker_events.jsonl'
            if not events.exists():return False
            record=json.loads(events.read_text().splitlines()[-1]);args=record.get('arguments',{})
            if (record['event']!='complete' or args.get('ks')!=[8,32] or args.get('scopes')!=SCOPES
                    or args.get('contrast_weight')!=gain or args.get('limit')!=8):return False
    return True


def verify_smoke(destination):
    if not smoke_ready():raise ValueError('Wait for both complete selected-gain smoke matrices')
    record,gain=selected();sources={str(SELECTION):sha(SELECTION)};checks=[]
    def read(path,rows=False):
        sources[str(path)]=sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    for m in MODELS:
        for p in ['image_text','text_video']:
            root=OUT/'experiments'/f'{m}_{p}_{variant(gain)}'/'discovery_smoke'
            old=OUT/'experiments'/f'{m}_{p}_uniform_factorized_evidence_a1'/'discovery'
            ids=read(root/'requested_ids.json')
            if len(ids)!=8 or ids!=read(old/'requested_ids.json')[:8]:raise ValueError('Wrong actual smoke batch')
            cfg=read(root/'runtime_config.json')
            if cfg['contrast_weight']!=gain or cfg['contrast_negative_mode']!='visual_and_task':raise ValueError('Wrong actual selected gain')
            base=verify_native(read(root/'predictions/baseline.jsonl',True),ids,gain,baseline=True)
            old_base=read(old/'predictions/baseline.jsonl',True)
            for e in ids:
                if not same_input(base[e],old_base[e]) or base[e]['native_class_logits_positive']!=old_base[e]['native_class_logits_positive']:
                    raise ValueError('Same-batch baseline replay differs')
            for scope in SCOPES:
                rank=read(OUT/'functional_selections'/f'stage8_{m}_v1'/f'{m}_{p}'/f'ranking_{scope}.json')['ranking']
                for k in [8,32]:
                    actual=verify_native(read(root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True),ids,gain,
                        {(h['layer'],h['head']) for h in rank[:k]})
                    source=read(old/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True)
                    for e in ids:
                        if not same_input(actual[e],source[e]):raise ValueError('Actual smoke input differs from derivation source')
                        for field in ['native_class_logits_positive','native_class_logits_negative','native_class_logits_negative_task']:
                            if actual[e][field]!=source[e][field]:raise ValueError('Actual branch does not replay stored source')
                    checks.append(dict(model=m,protocol=p,scope=scope,k=k,n=8,all_three_actual_cells_replay_exact=True,
                        native_formula_and_heads_verified=True,baseline_exact=True))
    create_json(destination,dict(status='pass',labels_read=False,gain=gain,checks=checks,sources_sha256=sources,
        actual_intervention_rows_verified=128,actual_branch_forward_observations=384,
        interpretation='Three actual forwards exactly replay source cells; selected-gain probabilities reconstructed. No copied derived output is accepted as actual inference.'))


def register_full(audit_path):
    record,gain=selected();audit=json.loads(audit_path.read_text())
    if audit['status']!='pass' or audit['labels_read'] or audit['gain']!=gain or len(audit['checks'])!=16:
        raise ValueError('Selected gain actual smoke did not pass')
    verify_sources(audit['sources_sha256']);jobs=[];previous={}
    for point in record['selected']:
        m=point['model'];name=f"validate_{m}_{point['protocol']}_factorized_global_gain"
        jobs.append(dict(name=name,gpu=MODELS.index(m),min_free_mb=23000,
            depends_on=[previous.get(m,f'smoke_{m}_factorized_global_gain')],
            command=command(m,[point['protocol']],'full_cohort',point['ks'],[point['scope']],gain)))
        previous[m]=name
    return write_addition('stage24_factorized_global_gain_validation.json',jobs,{str(SELECTION):sha(SELECTION),str(audit_path):sha(audit_path)})
