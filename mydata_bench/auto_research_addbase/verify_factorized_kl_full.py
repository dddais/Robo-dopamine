"""Label-free audit of every frozen full-cohort KL-neighborhood observation."""
import argparse
import time
import json

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .empirical_profile import sha
from .compare_native_bias import same_input
from .factorized_kl_tasks import selected,variant,verify_rows,SELECTION


def verify(model,protocol):
    record,budget=selected()
    point,=[p for p in record['selected'] if p['model']==model and p['protocol']==protocol]
    sources={str(SELECTION):sha(SELECTION)}
    def read(path,rows=False):
        sources[str(path)]=sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    ids=read(OUT/'splits.json')['full_cohort']
    root=OUT/'experiments'/f'{model}_{protocol}_{variant(budget)}'/'full_cohort'
    if len(ids)!=846 or len(set(ids))!=846 or read(root/'requested_ids.json')!=ids:
        raise ValueError('Require the complete frozen full846 population')
    rank_root=OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'
    expected=dict(model=model,contrast_kl_budget=budget,contrast_negative_mode='visual_and_task',contrast_weight=2,
        negative_strength=4,negative_task_strength=4,task_binding_fraction=.5,task_binding_distribution='uniform',
        ranking_prefix='ANSWER: ',frozen_ranking_source=str(rank_root))
    cfg=read(root/'runtime_config.json')
    if any(cfg.get(k)!=v for k,v in expected.items()) or cfg.get('visual_mass_partition','global')!='global':
        raise ValueError('Actual shared KL budget or branch configuration differs')
    scope=point['scope'];ranking=read(rank_root/f'ranking_{scope}.json')['ranking']
    if read(root.parent/'ranking'/f'ranking_{scope}.json')['ranking']!=ranking:
        raise ValueError('Actual copied ranking differs')
    base=verify_rows(read(root/'predictions/baseline.jsonl',True),ids,budget,baseline=True)
    if any(r['condition']!='baseline' or r['attention_diagnostics'] or r['negative_attention_diagnostics']
           or r['task_negative_attention_diagnostics'] for r in base.values()):
        raise ValueError('Baseline is not actually unsteered')
    checks=[]
    for k in point['ks']:
        h={(r['layer'],r['head']) for r in ranking[:k]}
        if len(h)!=k:raise ValueError('Duplicate or missing selected heads')
        rows=verify_rows(read(root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True),ids,budget,h)
        for e,r in rows.items():
            if not same_input(r,base[e]) or r['condition']!=f'{scope}:target:{k}':
                raise ValueError('Actual same-batch input or condition differs')
            for field in ['attention_diagnostics','negative_attention_diagnostics','task_negative_attention_diagnostics']:
                if not all(d['causal_mask_preserved'] and d['all_query_rows'] and d['prefill_calls']>=1 for d in r[field].values()):
                    raise ValueError('Actual masking/query contract differs')
        checks.append(dict(k=k,scope=scope,n=len(rows),actual_three_branches_and_kl_verified=True,
                           heads_and_same_batch_input_verified=True))
    path=OUT/'audit'/time.strftime(f'factorized_kl_full_{model}_{protocol}_%Y%m%d_%H%M%S.json')
    create_json(path,dict(status='pass',labels_read=False,model=model,protocol=protocol,budget=budget,
        checks=checks,frozen_point=point,sources_sha256=sources,
        interpretation='All frozen neighboring k values have full846 valid actual three-forward KL predictions. Implementation/coverage audit, not efficacy.'))
    print(path)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['qwen','roboreward'])
    parser.add_argument('--protocol',required=True)
    args=parser.parse_args();verify(args.model,args.protocol)
