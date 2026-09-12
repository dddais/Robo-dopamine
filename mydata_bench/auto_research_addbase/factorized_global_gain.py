"""Explicit global gain sensitivity from complete actual three-branch sources."""
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import append
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .select_factorized import PROTOCOLS, verified_rows
from .compare_native_bias import heads,same_input
from .empirical_profile import sha


POLICY = OUT/'selection_factorized_global_gain_discovery_v1.json'
MODELS = ['qwen','roboreward']
GAINS = [.5,2.]
SCOPES = ['all_frames','last_frame']
NEIGHBORHOODS = {8:[8,16,24],32:[24,32,40],64:[48,64,80]}


def compose(positive,visual,task,gain):
    return (1+gain)*positive-gain*(.5*(visual+task))


def derive(row,gain):
    positive,visual,task=[np.asarray(row[key],dtype=np.float32) for key in
        ['native_class_logits_positive','native_class_logits_negative','native_class_logits_negative_task']]
    z=compose(positive,visual,task,gain)
    p=np.exp(z.astype(float)-float(z.max()));p/=p.sum();reward=int(p.argmax())+1
    return dict(row,contrast_weight=gain,native_class_probabilities=p.tolist(),reward=reward,progress=(reward-1)/4,
        derived_only=True,derivation='Frozen additional global gain; unchanged three actual source forwards',
        new_model_forwards=0)


def cpu_check():
    rng=np.random.default_rng(24)
    p,v,t=[rng.normal(size=(8,5)) for _ in range(3)]
    def soft(z):
        a=np.exp(z-z.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)
    for a in GAINS:
        z=compose(p,v,t,a)
        expected=soft(p)**(1+a)/(soft(v)*soft(t))**(a/2);expected/=expected.sum(-1,keepdims=True)
        np.testing.assert_allclose(soft(z),expected,atol=1e-13,rtol=1e-13)
        np.testing.assert_allclose(soft(compose(p+8,v-3,t+1,a)),soft(z),atol=1e-13,rtol=1e-13)
        order=[4,1,3,0,2]
        np.testing.assert_array_equal(compose(p[:,order],v[:,order],t[:,order],a),z[:,order])
        np.testing.assert_array_equal(compose(p,v,v,a),(1+a)*p-a*v)
        eye=np.eye(5)*5
        assert np.array_equal(compose(eye,np.zeros_like(eye),np.zeros_like(eye),a).argmax(-1),np.arange(5))
        np.testing.assert_array_equal(np.concatenate([compose(p[i:i+1],v[i:i+1],t[i:i+1],a) for i in range(8)]),z)
    path=OUT/'audit'/time.strftime('factorized_global_gain_cpu_%Y%m%d_%H%M%S.json')
    create_json(path,dict(status='pass',labels_read=False,checks_per_gain=6,total_checks=12,
        checks=['normalized probability ratio','branch constant offsets','class permutation',
            'identical negatives reduce to two-branch formula','all five classes retained','sample independence'],
        selector_sha256=sha(__file__),policy_sha256=sha(POLICY)))
    return path


def main():
    cpu=cpu_check()
    spec=json.loads(POLICY.read_text())
    if spec['gains']!=GAINS or spec['protocols']!=PROTOCOLS or spec['ks']!=list(NEIGHBORHOODS):
        raise ValueError('Registered sensitivity changed')
    splits=json.loads((OUT/'splits.json').read_text());ids=splits['discovery']
    if len(ids)!=70 or set(ids)&set(splits['validation']):raise ValueError('Wrong discovery boundary')
    sources={str(POLICY):sha(POLICY),str(cpu):sha(cpu)};matrices={}
    for m in MODELS:
        for p in PROTOCOLS:
            root=OUT/'experiments'/f'{m}_{p}_uniform_factorized_evidence_a1'/'discovery'
            for name in ['runtime_config.json','requested_ids.json','worker_events.jsonl']:
                sources[str(root/name)]=sha(root/name)
            if set(json.loads((root/'requested_ids.json').read_text()))!=set(ids):raise ValueError('Wrong actual source population')
            event=json.loads((root/'worker_events.jsonl').read_text().splitlines()[-1])
            if event['event']!='complete':raise ValueError('All source inputs must be complete')
            cfg=json.loads((root/'runtime_config.json').read_text())
            expected=dict(contrast_weight=1,contrast_negative_mode='visual_and_task',negative_strength=4,
                negative_task_strength=4,task_binding_fraction=.5,task_binding_distribution='uniform')
            if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong actual source operator')
            base=verified_rows(root/'predictions/baseline.jsonl',ids,sources,True)
            conditions={}
            for scope in SCOPES:
                rank_path=OUT/'functional_selections'/f'stage8_{m}_v1'/f'{m}_{p}'/f'ranking_{scope}.json'
                sources[str(rank_path)]=sha(rank_path);rank=json.loads(rank_path.read_text())['ranking']
                for k in NEIGHBORHOODS:
                    rows=verified_rows(root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',ids,sources)
                    expected_heads={(h['layer'],h['head']) for h in rank[:k]}
                    if len(expected_heads)!=k:raise ValueError('Duplicate frozen heads')
                    for e,row in rows.items():
                        if not same_input(row,base[e]):raise ValueError('Source native inputs differ')
                        for field in ['attention_diagnostics','negative_attention_diagnostics','task_negative_attention_diagnostics']:
                            if heads(row,field)!=expected_heads:raise ValueError('Source heads differ')
                    conditions[(scope,k)]=rows
            matrices[(m,p)]=(base,conditions)
    # Freeze all derived outputs only after every source has been verified.
    destination=OUT/'derived_candidates'/time.strftime('factorized_global_gain_%Y%m%d_%H%M%S')
    derived={}
    for gain in GAINS:
        for (m,p),(base,conditions) in matrices.items():
            folder=destination/f'gain_{gain:g}'/f'{m}_{p}'
            create_json(folder/'requested_ids.json',ids)
            append(folder/'predictions/baseline.jsonl',[dict(base[e],derived_only=True,new_model_forwards=0) for e in ids])
            for (scope,k),rows in conditions.items():
                result={e:derive(row,gain) for e,row in rows.items()}
                path=folder/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl'
                append(path,[result[e] for e in ids]);derived[(gain,m,p,scope,k)]=(result,str(path))
    all_labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels={e:all_labels[e] for e in ids};del all_labels
    points=[];proposed={};counts={};details={}
    for gain in GAINS:
        proposed[gain]=[]
        for (m,p),(base,conditions) in matrices.items():
            reference=summary(base,labels,ids);passing=[]
            tasks=sorted({labels[e]['subset'] for e in ids})
            baseline_by_task={task:summary(base,labels,[e for e in ids if labels[e]['subset']==task]) for task in tasks}
            for scope,k in conditions:
                rows,path=derived[(gain,m,p,scope,k)];score=summary(rows,labels,ids)
                delta={c:score['accuracy']['0.125/0.875'][c]['rate_all_expected']-
                    reference['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all','suc','fail']}
                gate=score['mae']<reference['mae'] and delta['all']>=.1-1e-12 and min(delta['suc'],delta['fail'])>0
                point=dict(gain=gain,model=m,protocol=p,scope=scope,center_k=k,mae=score['mae'],
                    baseline_mae=reference['mae'],deltas=delta,passes=gate,source=path,
                    accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'],derived_only=True)
                points.append(point)
                details[f'{gain:g}/{m}_{p}/{scope}_{k}']=dict(baseline=reference,intervention=score,
                    baseline_by_task=baseline_by_task,
                    by_task={task:summary(rows,labels,[e for e in ids if labels[e]['subset']==task]) for task in tasks})
                if gate:passing.append(point)
            if passing:
                best=min(passing,key=lambda x:(x['mae'],-x['accuracy_all'],x['center_k'],x['scope']))
                proposed[gain].append(dict(best,ks=NEIGHBORHOODS[best['center_k']]))
        counts[gain]={m:sum(x['model']==m for x in proposed[gain]) for m in MODELS}
    qualifying=[a for a in GAINS if min(counts[a].values())>=3]
    chosen=min(qualifying) if qualifying else None
    create_json(destination/'details.json',details)
    output=OUT/'selection_factorized_global_gain_full_v1.json'
    create_json(output,dict(created_at=time.time(),all_discovery_points=points,input_counts=counts,
        selected_gain=chosen,selected=proposed[chosen] if chosen is not None else [],
        proposed_by_gain=proposed,sources_sha256=sources,derived_root=str(destination),validation_labels_used=[],
        actual_source_intervention_rows_verified=4200,additional_independent_gain_conditions=120,derived_rows=8400,
        new_model_forwards=0,actual_smoke_required_before_full=True,
        interpretation='Shared scalar only; no cross-gain or cross-method pooling. Explicit extra adaptive search, not independent efficacy replication.'))
    print(output);print('counts',counts,'selected_gain',chosen)


if __name__=='__main__':main()
