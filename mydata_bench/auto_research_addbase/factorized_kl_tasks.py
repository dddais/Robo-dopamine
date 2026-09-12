"""Actual source-replay gate and frozen full jobs for round27 shared KL budget."""
import json
import numpy as np
from pathlib import Path

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .empirical_profile import sha,verify_sources
from .factorized_gain_tasks import verify_native as verify_uncapped
from .compare_native_bias import same_input
from .factorized_global_gain import MODELS,SCOPES
from .factorized_kl import factorized_kl
from .derive_factorized_kl import POLICY,BUDGETS
from .durable_scheduler import include_additions,validate_plan


QUEUE=OUT/'queue_gpu01_20260911_1700'
SELECTION=OUT/'selection_factorized_kl_full_v1.json'


def selected():
    record=json.loads(SELECTION.read_text())
    eligible=[b for b in BUDGETS if min(record['candidates_by_budget'][str(b)]['input_counts'].values())>=3]
    budget=min(eligible) if eligible else None
    expected=record['candidates_by_budget'][str(budget)]['selected'] if budget is not None else []
    if record['selected_budget']!=budget or record['selected']!=expected:raise ValueError('Changed shared-budget selection')
    verify_sources(record['sources_sha256'])
    return record,budget


def variant(budget):
    return 'uniform_factorized_kl_b'+{.05:'005',.2:'02',.8:'08'}[budget]


def command(model,protocols,population,ks,scopes,budget,smoke=False):
    cmd=['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m','mydata_bench.auto_research_addbase.worker',
        '--model',model,'--protocols',*protocols,'--variant',variant(budget),'--methods','binding_transport',
        '--task-binding-fraction','.5','--task-binding-distribution','uniform','--ranking-prefix','ANSWER: ',
        '--contrast-weight','2','--contrast-kl-budget',str(budget),'--contrast-negative-mode','visual_and_task',
        '--negative-strength','4','--negative-task-strength','4','--strengths','4','--ks',*map(str,ks),
        '--scopes',*scopes,'--population',population,'--frozen-ranking-root',str(OUT/'functional_selections'/f'stage8_{model}_v1')]
    return cmd+(['--limit','8'] if smoke else [])


def verify_rows(rows,ids,budget,heads=None,baseline=False):
    if set(rows)!=set(ids):raise ValueError('All requested actual rows are required')
    uncapped={}
    for e,r in rows.items():
        if (r['readout']!='five_way_answer_likelihood_factorized_kl' or r['kl_budget']!=budget
                or r['contrast_cap']!=2 or r['kl_solver_iterations']!=(0 if baseline else 64)):
            raise ValueError('Actual KL runtime contract differs')
        cap_p=np.asarray(r['fixed_cap_native_class_probabilities'],dtype=float)
        if cap_p.shape!=(5,) or not np.isfinite(cap_p).all():raise ValueError('Missing actual uncapped native output')
        reward=int(cap_p.argmax())+1
        uncapped[e]=dict(r,contrast_weight=0 if baseline else 2,native_class_probabilities=cap_p,
            reward=reward,progress=(reward-1)/4)
    verify_uncapped(uncapped,ids,2.,heads,baseline)
    for r in rows.values():
        if baseline:
            z=np.asarray(r['native_class_logits_positive'],dtype=float)
            p=np.exp(z-z.max());p/=p.sum();alpha=0.;kl=0.
        else:
            result=factorized_kl(*[r[f] for f in ['native_class_logits_positive','native_class_logits_negative',
                'native_class_logits_negative_task']],budget)
            z=result['logits'];p=result['probabilities'];alpha=float(result['alpha']);kl=float(result['kl'])
        recorded=np.asarray(r['native_class_probabilities'],dtype=float)
        actual_z=np.asarray(r['native_class_logits_combined'],dtype=float)
        if (recorded.shape!=(5,) or actual_z.shape!=(5,) or not np.isfinite([*recorded,*actual_z,
                r['contrast_weight'],r['kl_divergence_from_positive'],r['progress']]).all()):
            raise ValueError('Finite actual KL outputs required')
        if (np.max(np.abs(recorded-p))>=1e-5 or np.max(np.abs(actual_z-z))>=1e-10
                or abs(r['contrast_weight']-alpha)>=1e-10 or abs(r['kl_divergence_from_positive']-kl)>=1e-10
                or not 0<=alpha<=2 or kl>budget+1e-12 or r['reward']!=int(p.argmax())+1
                or r['progress']!=(r['reward']-1)/4):
            raise ValueError('Actual same-sample KL formula differs')
    return rows


def write_addition(name,jobs,sources):
    old=include_additions(json.loads((QUEUE/'plan.json').read_text()),QUEUE)
    validate_plan(dict(jobs=list(old.values())+jobs))
    path=QUEUE/'additions'/name
    create_json(path,dict(jobs=jobs,sources_sha256=sources,interpretation='Round27 one shared KL budget; actual three-forward replay required; physical GPU0/1 only'))
    if not path.with_suffix('.ready').exists():
        with path.with_suffix('.ready').open('x') as f:f.write('Frozen shared KL budget; actual gate before full; GPU0/1 only\n')
    return path


def register_smoke(cpu):
    record,budget=selected();audit=json.loads(cpu.read_text())
    if audit['status']!='pass' or audit['labels_read'] or audit['kl_runtime_tests']!=3 or audit['tests_run']!=57:
        raise ValueError('Complete KL math/runtime and prior-method regression required')
    for path,digest in audit['source_code_sha256'].items():
        if not Path(path).resolve().is_relative_to(Path(__file__).resolve().parent) or sha(path)!=digest:
            raise ValueError('Research source code changed after CPU audit')
    jobs=[] if budget is None else [dict(name=f'smoke_{m}_factorized_kl',gpu=i,min_free_mb=23000,depends_on=[],
        command=command(m,['image_text','text_video'],'discovery',[8,32],SCOPES,budget,True)) for i,m in enumerate(MODELS)]
    return write_addition('stage27_factorized_kl_smoke.json',jobs,{str(cpu):sha(cpu),str(SELECTION):sha(SELECTION)})


def smoke_ready():
    record,budget=selected()
    if budget is None:return False
    for m in MODELS:
        for p in ['image_text','text_video']:
            root=OUT/'experiments'/f'{m}_{p}_{variant(budget)}'/'discovery_smoke'
            events=root/'worker_events.jsonl'
            if not events.exists():return False
            last=json.loads(events.read_text().splitlines()[-1]);args=last.get('arguments',{})
            if (last['event']!='complete' or args.get('contrast_kl_budget')!=budget or args.get('ks')!=[8,32]
                    or args.get('scopes')!=SCOPES or args.get('limit')!=8):return False
            ids=json.loads((root/'requested_ids.json').read_text())
            paths=[root/'predictions/baseline.jsonl']+[root/'binding_transport_s4/predictions'/f'{s}_target_{k}.jsonl' for s in SCOPES for k in [8,32]]
            if len(ids)!=8 or any(not f.exists() or set(latest(f))!=set(ids) for f in paths):return False
    return True


def verify_smoke(destination):
    if not smoke_ready():raise ValueError('Wait for all actual KL smoke inputs')
    record,budget=selected();sources={str(SELECTION):sha(SELECTION)};checks=[]
    def read(path,rows=False):
        sources[str(path)]=sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    for m in MODELS:
        for p in ['image_text','text_video']:
            root=OUT/'experiments'/f'{m}_{p}_{variant(budget)}'/'discovery_smoke'
            old=OUT/'experiments'/f'{m}_{p}_uniform_factorized_evidence_a1'/'discovery'
            ids=read(root/'requested_ids.json')
            if ids!=read(old/'requested_ids.json')[:8]:raise ValueError('Changed same-batch replay population')
            cfg=read(root/'runtime_config.json')
            if cfg['contrast_kl_budget']!=budget or cfg['contrast_weight']!=2:raise ValueError('Changed actual budget/cap')
            base=verify_rows(read(root/'predictions/baseline.jsonl',True),ids,budget,baseline=True)
            original=read(old/'predictions/baseline.jsonl',True)
            for e in ids:
                if not same_input(base[e],original[e]) or base[e]['native_class_logits_positive']!=original[e]['native_class_logits_positive']:
                    raise ValueError('Native baseline does not replay actual source')
            for scope in SCOPES:
                rank=read(OUT/'functional_selections'/f'stage8_{m}_v1'/f'{m}_{p}'/f'ranking_{scope}.json')['ranking']
                for k in [8,32]:
                    rows=verify_rows(read(root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True),ids,budget,
                        {(h['layer'],h['head']) for h in rank[:k]})
                    source=read(old/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True)
                    for e in ids:
                        if not same_input(rows[e],base[e]) or not same_input(rows[e],source[e]):raise ValueError('Actual source input differs')
                        for field in ['native_class_logits_positive','native_class_logits_negative','native_class_logits_negative_task']:
                            if rows[e][field]!=source[e][field]:raise ValueError('Actual KL branch does not replay stored source')
                        if rows[e]['condition']!=f'{scope}:target:{k}':raise ValueError('Wrong actual condition')
                    checks.append(dict(model=m,protocol=p,scope=scope,k=k,n=8,
                        all_three_actual_source_cells_exact=True,kl_formula_and_heads_verified=True,baseline_exact=True))
    create_json(destination,dict(status='pass',labels_read=False,budget=budget,checks=checks,sources_sha256=sources,
        actual_intervention_rows_verified=128,actual_model_branch_observations=384,
        interpretation='New actual three-forward runtime, same source cells and KL composition; not copied derived inference. No efficacy labels read.'))


def register_full(audit_path):
    record,budget=selected();audit=json.loads(audit_path.read_text())
    if audit['status']!='pass' or audit['labels_read'] or audit['budget']!=budget or len(audit['checks'])!=16:
        raise ValueError('Actual selected-budget replay must pass')
    verify_sources(audit['sources_sha256']);jobs=[];previous={}
    for p in record['selected']:
        m=p['model'];name=f"validate_{m}_{p['protocol']}_factorized_kl"
        jobs.append(dict(name=name,gpu=MODELS.index(m),min_free_mb=23000,
            depends_on=[previous.get(m,f'smoke_{m}_factorized_kl')],
            command=command(m,[p['protocol']],'full_cohort',p['ks'],[p['scope']],budget)))
        previous[m]=name
    return write_addition('stage27_factorized_kl_validation.json',jobs,{str(SELECTION):sha(SELECTION),str(audit_path):sha(audit_path)})
