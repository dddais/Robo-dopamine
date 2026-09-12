"""Score actually recorded evidence branches; no new inference or fitted transform."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .analyze import for_field
from .statistics import paired_statistics


def native_branch(row,branch,model):
    if row['status']!='ok':return dict(row)
    logits=np.asarray(row[f'native_class_logits_{branch}'],dtype=np.float64)
    if logits.ndim!=1:raise ValueError('Missing recorded native branch')
    out={'example_id':row['example_id'],'status':'ok','derived_from_recorded_branch':branch,
         'raw_output':None,'readout':'native reward classes, recorded branch ablation'}
    if model=='meter':
        if logits.shape!=(10,):raise ValueError('All ten trained progress bins are required')
        probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum()
        out['progress']=float(probabilities@np.linspace(0,1,10))
        z=float(row[f'success_logit_{branch}'])
        out['success_probability']=1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
    else:
        if logits.shape!=(5,):raise ValueError('All five native reward classes are required')
        out['reward']=int(logits.argmax())+1
        out['progress']=(out['reward']-1)/4
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--conditions',nargs='+',required=True)
    args=p.parse_args()
    folder=OUT/'experiments'/args.experiment/'full_cohort'
    cfg=json.loads((folder/'runtime_config.json').read_text())
    requested=set(json.loads((folder/'requested_ids.json').read_text()))
    baseline_path=folder/'predictions/baseline.jsonl'
    baseline=latest(baseline_path)
    if set(baseline)!=requested:raise ValueError('Incomplete baseline')
    splits=json.loads((OUT/'splits.json').read_text())
    labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    results={};sources={str(baseline_path):hashlib.sha256(baseline_path.read_bytes()).hexdigest()}
    for condition in args.conditions:
        path=folder/args.method/'predictions'/f'{condition}.jsonl'
        rows=latest(path)
        if set(rows)!=requested:raise ValueError('Wait for complete condition '+condition)
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        branches={'baseline':baseline,'combined':rows}
        for branch in ['positive','negative']:
            branches[branch]={eid:native_branch(row,branch,cfg['model']) for eid,row in rows.items()}
        for field in ['progress']+(['success_probability'] if cfg['model']=='meter' else []):
            converted={name:for_field(value,field) for name,value in branches.items()}
            for population in ['full_cohort','old_holdout','validation']:
                ids=[eid for eid in splits[population] if eid in requested]
                entry={'metrics':{name:summary(value,labels,ids) for name,value in converted.items()},
                    'combined_minus_positive':paired_statistics(converted['positive'],converted['combined'],labels,ids),
                    'positive_minus_baseline':paired_statistics(converted['baseline'],converted['positive'],labels,ids)}
                results[f'{condition}/{field}/{population}']=entry
    output=OUT/'analysis'/time.strftime('branch_ablation_%Y%m%d_%H%M%S.json')
    create_json(output,{'arguments':vars(args),'sources':sources,
        'interpretation':'Ablation of the stored positive and negative forward passes. Combined uses the original recorded result. No alpha tuning, threshold fitting, endpoint remapping, or label-conditioned inference. CIs are exploratory.',
        'results':results})
    print(output)
    for key,entry in results.items():
        if key.endswith('/validation'):
            print(key)
            for name,value in entry['metrics'].items():
                print(name,'MAE',round(value['mae'],3),'accuracy',*[round(value['accuracy']['0.125/0.875'][s]['rate_all_expected'],4) for s in ['all','suc','fail']])


if __name__=='__main__':main()
