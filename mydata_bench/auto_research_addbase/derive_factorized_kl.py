"""Round27 complete-source derivation; global KL choice precedes actual smoke."""
import io
import json
import time
import unittest
from pathlib import Path

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest, append
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .empirical_profile import sha
from .factorized_gain_tasks import verify_native
from .compare_native_bias import same_input
from .factorized_global_gain import MODELS, SCOPES, NEIGHBORHOODS
from .factorized_kl import factorized_kl


POLICY=OUT/'selection_factorized_kl_discovery_v1.json'
BUDGETS=[.05,.2,.8]
PROTOCOLS=['image_text','text_image','interleaved','video_text','text_video']


def main():
    spec=json.loads(POLICY.read_text())
    if (spec['kl_budgets_nats']!=BUDGETS or spec['alpha_cap']!=2 or spec['models']!=MODELS
            or spec['protocols']!=PROTOCOLS or spec['scopes']!=SCOPES or spec['ks']!=list(NEIGHBORHOODS)):
        raise ValueError('Frozen three-branch KL policy differs')
    modules=['mydata_bench.auto_research_addbase.'+n for n in ['test_kl_contrast','test_factorized_kl']]
    stream=io.StringIO()
    test=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    if not test.wasSuccessful():raise ValueError(stream.getvalue())
    cpu=OUT/'audit'/time.strftime('factorized_kl_cpu_%Y%m%d_%H%M%S.json')
    create_json(cpu,dict(status='pass',labels_read=False,tests_run=test.testsRun,output=stream.getvalue(),
        source_code_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('factorized_kl.py'),
            Path(__file__).with_name('kl_contrast.py'),Path(__file__).with_name('test_factorized_kl.py')]},policy_sha256=sha(POLICY)))
    sources={str(POLICY):sha(POLICY),str(cpu):sha(cpu)}
    def read(path,rows=False):
        sources[str(path)]=sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    splits=read(OUT/'splits.json');ids=splits['discovery']
    if len(ids)!=70 or set(ids)&set(splits['validation']):raise ValueError('Changed discovery boundary')
    matrices={}
    for model in MODELS:
        for protocol in PROTOCOLS:
            root=OUT/'experiments'/f'{model}_{protocol}_uniform_factorized_evidence_a1'/'discovery'
            rank_root=OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'
            if read(root/'requested_ids.json')!=ids:raise ValueError('Changed source population')
            cfg=read(root/'runtime_config.json')
            expected=dict(model=model,contrast_negative_mode='visual_and_task',contrast_weight=1,
                negative_strength=4,negative_task_strength=4,task_binding_fraction=.5,
                task_binding_distribution='uniform',ranking_prefix='ANSWER: ',frozen_ranking_source=str(rank_root))
            if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Changed actual source operator')
            base=verify_native(read(root/'predictions/baseline.jsonl',True),ids,1,baseline=True)
            conditions={}
            for scope in SCOPES:
                ranking=read(rank_root/f'ranking_{scope}.json')['ranking']
                for k in NEIGHBORHOODS:
                    heads={(h['layer'],h['head']) for h in ranking[:k]}
                    if len(heads)!=k:raise ValueError('Wrong head count')
                    rows=verify_native(read(root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True),ids,1,heads)
                    if any(not same_input(rows[e],base[e]) or rows[e]['condition']!=f'{scope}:target:{k}' for e in ids):
                        raise ValueError('Changed actual input or condition')
                    conditions[(scope,k)]=rows
            matrices[(model,protocol)]=(base,conditions)
    destination=OUT/'derived_candidates'/time.strftime('factorized_kl_%Y%m%d_%H%M%S')
    derived={}
    for budget in BUDGETS:
        for (model,protocol),(base,conditions) in matrices.items():
            folder=destination/f'budget_{budget:g}'/f'{model}_{protocol}'
            create_json(folder/'requested_ids.json',ids)
            append(folder/'predictions/baseline.jsonl',[dict(base[e],derived_only=True,new_model_forwards=0) for e in ids])
            for (scope,k),rows in conditions.items():
                vectors=[np.asarray([rows[e][field] for e in ids],dtype=np.float32) for field in
                    ['native_class_logits_positive','native_class_logits_negative','native_class_logits_negative_task']]
                r=factorized_kl(*vectors,budget)
                result={}
                for i,e in enumerate(ids):
                    reward=int(r['probabilities'][i].argmax())+1
                    result[e]=dict(rows[e],derived_only=True,new_model_forwards=0,actual_forward_branches=0,
                        source_actual_forward_branches=3,source_contrast_weight=1,contrast_weight=float(r['alpha'][i]),
                        kl_budget=budget,kl=float(r['kl'][i]),native_class_probabilities=r['probabilities'][i].tolist(),
                        native_class_logits_combined=r['logits'][i].tolist(),reward=reward,progress=(reward-1)/4,
                        composition='Derived KL-limited positive versus actual visual/task negative mean')
                path=folder/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl'
                append(path,[result[e] for e in ids]);derived[(budget,model,protocol,scope,k)]=(result,r,str(path))
    # All source checks and all derived outputs precede reading scoring labels.
    all_labels=json.loads((OLD/'labels_for_scoring_only.json').read_text());labels={e:all_labels[e] for e in ids};del all_labels
    points=[];details={};export_points=[];candidates={}
    for budget in BUDGETS:
        selected=[]
        for (model,protocol),(base,conditions) in matrices.items():
            baseline=summary(base,labels,ids);passing=[]
            tasks=sorted({labels[e]['subset'] for e in ids})
            baseline_tasks={t:summary(base,labels,[e for e in ids if labels[e]['subset']==t]) for t in tasks}
            for scope,k in conditions:
                rows,r,path=derived[(budget,model,protocol,scope,k)];score=summary(rows,labels,ids)
                delta={c:score['accuracy']['0.125/0.875'][c]['rate_all_expected']-baseline['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all','suc','fail']}
                gate=score['mae']<baseline['mae'] and delta['all']>=.1-1e-12 and min(delta['suc'],delta['fail'])>0
                point=dict(model=model,protocol=protocol,budget=budget,scope=scope,center_k=k,mae=score['mae'],
                    baseline_mae=baseline['mae'],deltas=delta,passes=gate,
                    accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'],
                    alpha_quantiles=np.quantile(r['alpha'],[0,.25,.5,.75,1]).tolist(),
                    fraction_below_cap=float(np.mean(r['alpha']<2)),max_kl=float(r['kl'].max()),source=path)
                points.append(point)
                if gate:passing.append(point)
                experiment=f'{model}_{protocol}_factorized_kl_b{budget:g}'
                key=f'{experiment}/binding_transport_s4/{scope}_target_{k}/progress'
                details[key]=dict(baseline=baseline,intervention=score,baseline_by_task=baseline_tasks,
                    by_task={t:summary(rows,labels,[e for e in ids if labels[e]['subset']==t]) for t in tasks})
                for threshold in ['0.125/0.875','0.2/0.8']:
                    export_points.append(dict(experiment=experiment,method='binding_transport_s4',population='discovery',
                        condition=f'{scope}_target_{k}',field='progress',threshold=threshold,expected=70,valid=70,baseline_valid=70,
                        mae=score['mae'],baseline_mae=baseline['mae'],delta_mae=score['mae']-baseline['mae'],
                        **{f'delta_{c}':d for c,d in delta.items()},meets_descriptive_gate=gate,derived_only=True))
            if passing:
                best=min(passing,key=lambda p:(p['mae'],-p['accuracy_all'],p['center_k'],p['scope']))
                selected.append(dict(best,ks=NEIGHBORHOODS[best['center_k']]))
        counts={m:sum(p['model']==m for p in selected) for m in MODELS}
        candidates[str(budget)]=dict(selected=selected,input_counts=counts,eligible=min(counts.values())>=3)
    eligible=[b for b in BUDGETS if candidates[str(b)]['eligible']];chosen=min(eligible) if eligible else None
    create_json(destination/'points.json',export_points);create_json(destination/'details.json',details)
    create_json(destination/'selection_points.json',points)
    output=OUT/'selection_factorized_kl_full_v1.json'
    create_json(output,dict(created_at=time.time(),sources_sha256=sources,derived_root=str(destination),
        all_discovery_points=points,candidates_by_budget=candidates,selected_budget=chosen,
        selected=candidates[str(chosen)]['selected'] if chosen is not None else [],
        actual_source_intervention_rows_verified=4200,source_baselines_verified=700,
        independent_conditions=180,derived_predictions=12600,new_model_forwards=0,
        actual_smoke_required=chosen is not None,validation_labels_used=[],
        interpretation='Separate three-branch KL mechanism; one shared budget only. Derived exploration, not actual new inference or independent confirmation.'))
    print(cpu);print(destination);print(output)
    print({k:v['input_counts'] for k,v in candidates.items()},'selected_budget',chosen)


if __name__=='__main__':main()
