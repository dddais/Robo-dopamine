"""Round25: same alpha2 three-branch operator, fixed native Robometer head."""
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD,create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .empirical_profile import sha,verify_sources
from .compare_native_bias import heads
from .factorized_global_gain import compose,NEIGHBORHOODS
from .durable_scheduler import include_additions,validate_plan


VARIANT='uniform_factorized_evidence_a2'
POLICY=OUT/'selection_meter_factorized_a2_discovery_v1.json'
PROTOCOLS=['image_text','text_image','interleaved','text_video','video_text','official']
SCOPES=['all_frames','last_frame']
THRESHOLDS=['0.125/0.875','0.2/0.8']
QUEUE=OUT/'queue_gpu01_20260911_1700'


def folder(protocol,population):
    return OUT/'experiments'/f'meter_{protocol}_{VARIANT}'/population


def rank_path(protocol,scope):
    return OUT/'functional_selections/stage8_meter_v1'/f'meter_{protocol}'/f'ranking_{scope}.json'


def same_input(a,b):
    return (a['prompt']==b['prompt'] and a['token_audit']['input_ids_sha256']==b['token_audit']['input_ids_sha256']
        and a['token_audit']['query']==b['token_audit']['query'])


def config_check(cfg,protocol):
    expected=dict(model='meter',contrast_negative_mode='visual_and_task',contrast_weight=2,
        task_binding_fraction=.5,task_binding_distribution='uniform',negative_strength=4,negative_task_strength=4,
        ranking_prefix='',frozen_ranking_source=str(rank_path(protocol,'all_frames').parent))
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen meter factorized method differs')


def verify_rows(rows,ids,expected_heads=None,baseline=False):
    if set(rows)!=set(ids) or any(r['status']!='ok' for r in rows.values()):raise ValueError('Complete valid native observations required')
    for row in rows.values():
        if (row['readout']!='native_bin_attention_contrast' or row.get('derived_only')
                or row['actual_forward_branches']!=(1 if baseline else 3)
                or row['contrast_negative_mode']!='visual_and_task' or row['contrast_weight']!=(0 if baseline else 2)):
            raise ValueError('Wrong actual native head/branch composition')
        pos=np.asarray(row['native_class_logits_positive'],dtype=np.float32)
        sp=np.asarray([row['success_logit_positive']],dtype=np.float32)
        if pos.shape!=(10,) or not np.isfinite(pos).all() or not np.isfinite(sp).all():raise ValueError('Native ten-bin and success logits required')
        if baseline:z=pos;sz=sp
        else:
            vis=np.asarray(row['native_class_logits_negative'],dtype=np.float32)
            task=np.asarray(row['native_class_logits_negative_task'],dtype=np.float32)
            sv=np.asarray([row['success_logit_negative']],dtype=np.float32)
            st=np.asarray([row['success_logit_negative_task']],dtype=np.float32)
            if any(x.shape!=(10,) or not np.isfinite(x).all() for x in [vis,task]) or not np.isfinite([sv,st]).all():
                raise ValueError('Both actual negative native heads required')
            if (not row['negative_mean_is_derived'] or not np.array_equal(.5*(vis+task),row['native_class_logits_negative_mean'])
                    or not np.array_equal(.5*(sv+st),[row['success_logit_negative_mean']])):
                raise ValueError('Both actual native counterfactual means must reconstruct')
            z=compose(pos,vis,task,2);sz=compose(sp,sv,st,2)
            for field,method,strength in [('attention_diagnostics','binding_transport',4),
                    ('negative_attention_diagnostics','mass_transport',-4),
                    ('task_negative_attention_diagnostics','binding_task_suppression',4)]:
                if expected_heads is not None and heads(row,field)!=expected_heads:raise ValueError('Actual selected heads differ')
                for d in row[field].values():
                    if d['method']!=method or d['strength']!=strength or not d['causal_mask_preserved']:
                        raise ValueError('Actual meter branch differs')
                    if method=='binding_transport' and (d['task_binding_distribution']!='uniform' or d['task_binding_fraction']!=.5):
                        raise ValueError('Actual positive instruction operator differs')
                    if method=='binding_task_suppression' and d['task_logit_strength']!=-4:raise ValueError('Actual negative instruction operator differs')
        p=np.exp(z.astype(float)-float(z.max()));p/=p.sum()
        success=float(np.exp(-np.logaddexp(0,-float(sz[0]))))
        progress=float((p*np.linspace(0,1,10)).sum())
        recorded=np.asarray(row['native_class_probabilities'],dtype=float)
        if (recorded.shape!=(10,) or not np.isfinite(recorded).all()
                or not np.isfinite([row['progress'],row['success_probability']]).all()):
            raise ValueError('Recorded native probabilities must be finite')
        if (np.max(np.abs(p-recorded))>=1e-5
                or abs(progress-row['progress'])>=2e-6 or abs(success-row['success_probability'])>=2e-6):
            raise ValueError('Native bin expectation or actual three-branch success sigmoid differs')
    return rows


def command(protocols,population,ks,scopes=SCOPES,smoke=False):
    result=['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m',
        'mydata_bench.auto_research_addbase.worker','--model','meter','--protocols',*protocols,
        '--variant',VARIANT,'--methods','binding_transport','--strengths','4','--contrast-weight','2',
        '--contrast-negative-mode','visual_and_task','--negative-strength','4','--negative-task-strength','4',
        '--task-binding-fraction','.5','--task-binding-distribution','uniform',
        '--ks',*map(str,ks),'--scopes',*scopes,'--population',population,
        '--frozen-ranking-root',str(OUT/'functional_selections/stage8_meter_v1')]
    if smoke:result+=['--limit','8']
    return result


def write_addition(name,jobs,sources):
    old=include_additions(json.loads((QUEUE/'plan.json').read_text()),QUEUE)
    validate_plan(dict(jobs=list(old.values())+jobs))
    path=QUEUE/'additions'/name
    create_json(path,dict(jobs=jobs,sources_sha256=sources,
        interpretation='Robometer native-head port of fixed shared alpha2 three-branch operator; primary success head fixed. GPU0/1 only'))
    if not path.with_suffix('.ready').exists():
        with path.with_suffix('.ready').open('x') as f:f.write('Frozen round25 primary head and shared alpha2; physical GPU0/1 only\n')
    return path


def register_smoke(cpu_audit):
    audit=json.loads(cpu_audit.read_text())
    if audit['status']!='pass' or audit['labels_read'] or audit['tests_run']!=37 or audit['meter_native_head_tests']!=3:
        raise ValueError('Native scalar/vector and all prior-method regressions must pass')
    for path,digest in audit['source_code_sha256'].items():
        if sha(path)!=digest:raise ValueError('Inference code changed after CPU audit')
    jobs=[dict(name='smoke_meter_factorized_a2',gpu=0,min_free_mb=16000,depends_on=[],
        command=command(['text_video','official'],'discovery',[8,32],smoke=True))]
    return write_addition('stage25_meter_factorized_a2_smoke.json',jobs,{str(cpu_audit):sha(cpu_audit),str(POLICY):sha(POLICY)})


def complete(root,ks,scopes,n):
    path=root/'worker_events.jsonl'
    if not path.exists():return False
    last=json.loads(path.read_text().splitlines()[-1]);args=last.get('arguments',{})
    if (last.get('event')!='complete' or args.get('ks')!=ks or args.get('scopes')!=scopes
            or args.get('variant')!=VARIANT or args.get('contrast_weight')!=2):return False
    ids=json.loads((root/'requested_ids.json').read_text())
    paths=[root/'predictions/baseline.jsonl']+[root/'binding_transport_s4/predictions'/f'{s}_target_{k}.jsonl' for s in scopes for k in ks]
    return len(ids)==n and all(p.exists() and set(latest(p))==set(ids) for p in paths)


def smoke_ready():
    return all(complete(folder(p,'discovery_smoke'),[8,32],SCOPES,8) for p in ['text_video','official'])


def verify_smoke(destination):
    if not smoke_ready():raise ValueError('Wait for both complete native meter smoke inputs')
    sources={str(POLICY):sha(POLICY)};checks=[]
    def read(path,rows=False):
        sources[str(path)]=sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    for protocol in ['text_video','official']:
        root=folder(protocol,'discovery_smoke');old=OUT/'experiments'/f'meter_{protocol}_functional_evidence_a1'/'discovery'
        ids=read(root/'requested_ids.json')
        if ids!=read(old/'requested_ids.json')[:8]:raise ValueError('Wrong actual smoke batch')
        cfg=read(root/'runtime_config.json');config_check(cfg,protocol)
        if cfg['batch_size']!=read(old/'runtime_config.json')['batch_size']:raise ValueError('Reference batch sizes differ')
        base=verify_rows(read(root/'predictions/baseline.jsonl',True),ids,baseline=True)
        old_base=read(old/'predictions/baseline.jsonl',True)
        for e in ids:
            if (not same_input(base[e],old_base[e])
                    or base[e]['native_class_logits_positive']!=old_base[e]['native_class_logits_positive']
                    or base[e]['success_logit_positive']!=old_base[e]['success_logit_positive']):
                raise ValueError('Final native-query baseline heads do not exactly replay the source')
        for scope in SCOPES:
            rank=read(rank_path(protocol,scope))['ranking']
            for k in [8,32]:
                rows=verify_rows(read(root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True),ids,
                    {(h['layer'],h['head']) for h in rank[:k]})
                if any(not same_input(rows[e],base[e]) or rows[e]['condition']!=f'{scope}:target:{k}' for e in ids):
                    raise ValueError('Actual intervention native query/input differs')
                checks.append(dict(protocol=protocol,scope=scope,k=k,n=8,
                    actual_ten_bins_and_binary_success_formula=True,actual_three_branches_and_heads=True,
                    same_input_and_final_native_query_baseline_replay_exact=True))
    create_json(destination,dict(status='pass',labels_read=False,checks=checks,sources_sha256=sources,
        actual_three_branch_rows=64,actual_native_branch_observations=192,
        interpretation='Both trained native heads and same final-query baselines verified. No endpoint transform or head-based coverage pooling.'))


def register_discovery(audit_path):
    audit=json.loads(audit_path.read_text())
    if audit['status']!='pass' or audit['labels_read'] or len(audit['checks'])!=8:raise ValueError('All actual meter smoke checks required')
    verify_sources(audit['sources_sha256'])
    jobs=[dict(name='round25_meter_factorized_a2_discovery',gpu=0,min_free_mb=16000,
        depends_on=['smoke_meter_factorized_a2'],command=command(PROTOCOLS,'discovery',[8,32,64]))]
    return write_addition('stage25_meter_factorized_a2_discovery.json',jobs,{str(audit_path):sha(audit_path),str(POLICY):sha(POLICY)})


def matrix_ready():
    return all(complete(folder(p,'discovery'),[8,32,64],SCOPES,70) for p in PROTOCOLS)


def select(destination):
    if not matrix_ready():raise ValueError('Wait for all six complete discovery inputs')
    spec=json.loads(POLICY.read_text())
    if spec['primary_field']!='success_probability' or spec['contrast_weight']!=2 or spec['protocols']!=PROTOCOLS:
        raise ValueError('Frozen primary native head or gain differs')
    splits=json.loads((OUT/'splits.json').read_text());ids=splits['discovery']
    if len(ids)!=70 or set(ids)&set(splits['validation']):raise ValueError('Wrong discovery boundary')
    sources={str(POLICY):sha(POLICY)};matrices={}
    def read(path,rows=False):
        sources[str(path)]=sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    for protocol in PROTOCOLS:
        root=folder(protocol,'discovery');config_check(read(root/'runtime_config.json'),protocol)
        if set(read(root/'requested_ids.json'))!=set(ids):raise ValueError('Wrong actual population')
        base=verify_rows(read(root/'predictions/baseline.jsonl',True),ids,baseline=True);conditions={}
        for scope in SCOPES:
            rank=read(rank_path(protocol,scope))['ranking']
            for k in NEIGHBORHOODS:
                h={(r['layer'],r['head']) for r in rank[:k]}
                if len(h)!=k:raise ValueError('Duplicate heads')
                rows=verify_rows(read(root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True),ids,h)
                if any(not same_input(rows[e],base[e]) for e in ids):raise ValueError('Native input/query differs')
                conditions[(scope,k)]=rows
        matrices[protocol]=(base,conditions)
    all_labels=json.loads((OLD/'labels_for_scoring_only.json').read_text());labels={e:all_labels[e] for e in ids};del all_labels
    def primary(rows):return {e:dict(r,progress=r['success_probability']) for e,r in rows.items()}
    points=[];proposed=[]
    for protocol,(base,conditions) in matrices.items():
        baseline=summary(primary(base),labels,ids);passing=[]
        for (scope,k),rows in conditions.items():
            score=summary(primary(rows),labels,ids)
            deltas={t:{c:score['accuracy'][t][c]['rate_all_expected']-baseline['accuracy'][t][c]['rate_all_expected']
                for c in ['all','suc','fail']} for t in THRESHOLDS}
            gate=score['mae']<baseline['mae'] and all(v['all']>=.1-1e-12 and min(v['suc'],v['fail'])>0 for v in deltas.values())
            point=dict(model='meter',protocol=protocol,scope=scope,center_k=k,variant=VARIANT,
                primary_field='success_probability',mae=score['mae'],baseline_mae=baseline['mae'],deltas=deltas,passes=gate,
                worst_threshold_accuracy=min(score['accuracy'][t]['all']['rate_all_expected'] for t in THRESHOLDS))
            points.append(point)
            if gate:passing.append(point)
        if passing:
            best=min(passing,key=lambda x:(x['mae'],-x['worst_threshold_accuracy'],x['center_k'],x['scope']))
            proposed.append(dict(best,ks=NEIGHBORHOODS[best['center_k']]))
    eligible=len(proposed)>=3
    create_json(destination,dict(created_at=time.time(),all_discovery_points=points,proposed=proposed,
        input_count=len(proposed),eligible_for_full=eligible,selected=proposed if eligible else [],
        sources_sha256=sources,actual_three_branch_rows_verified=2520,independent_conditions=36,
        validation_labels_used=[],primary_field='success_probability',both_thresholds_required=True,
        interpretation='Fixed native primary head and same global alpha2 as round24; cannot pool native heads or thresholds to pass coverage.'))


def register_full(selection,audit_path):
    record=json.loads(selection.read_text());audit=json.loads(audit_path.read_text())
    if record['eligible_for_full']!=(record['input_count']>=3) or record['selected']!=(record['proposed'] if record['eligible_for_full'] else []):
        raise ValueError('Changed frozen meter coverage rule')
    if audit['status']!='pass' or audit['labels_read'] or len(audit['checks'])!=8:raise ValueError('Missing actual meter gate')
    verify_sources(dict(record['sources_sha256'],**audit['sources_sha256']));jobs=[];previous='round25_meter_factorized_a2_discovery'
    for point in record['selected']:
        name=f"validate_meter_{point['protocol']}_factorized_a2"
        jobs.append(dict(name=name,gpu=0,min_free_mb=16000,depends_on=[previous],
            command=command([point['protocol']],'full_cohort',point['ks'],[point['scope']])))
        previous=name
    return write_addition('stage25_meter_factorized_a2_validation.json',jobs,{str(selection):sha(selection),str(audit_path):sha(audit_path)})
