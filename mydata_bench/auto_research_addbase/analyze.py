"""Labels are read only here, after inference. Immutable timestamped reports."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from .prepare import OUT
from .statistics import paired_statistics, holm


def for_field(rows,field):
    if field=='progress':return rows
    return {k:dict(r,progress=r.get(field)) for k,r in rows.items()}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--population',default='discovery')
    parser.add_argument('--name')
    parser.add_argument('--score-split',help='Score this frozen split within the source population')
    parser.add_argument('--uncertainty',action='store_true')
    parser.add_argument('--experiments',nargs='+',help='Optional explicit experiment directories for a focused checkpoint')
    parser.add_argument('--condition-kind',choices=['target','wrong_region','low_rank'],
                        help='Optionally restrict a checkpoint to one explicitly named intervention/control kind')
    args=parser.parse_args()
    labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    splits=json.loads((OUT/'splits.json').read_text())
    points=[];details={}
    for folder in sorted((OUT/'experiments').glob(f'*/{args.population}')):
        if args.experiments and folder.parent.name not in args.experiments:continue
        folder_requested=json.loads((folder/'requested_ids.json').read_text())
        requested=[k for k in folder_requested if not args.score_split or k in splits[args.score_split]]
        base=latest(folder/'predictions/baseline.jsonl')
        for field in ['progress']+(['success_probability'] if folder.parent.name.startswith('meter_') else []):
            baseline=summary(for_field(base,field),labels,requested)
            tasks=sorted({labels[k]['subset'] for k in requested})
            base_tasks={t:summary(for_field(base,field),labels,[k for k in requested if labels[k]['subset']==t]) for t in tasks}
            for prediction in sorted(folder.glob('*/predictions/*.jsonl')):
                if args.condition_kind and f'_{args.condition_kind}_' not in prediction.stem:continue
                rows=latest(prediction)
                if set(folder_requested)-set(rows):continue
                scores=summary(for_field(rows,field),labels,requested)
                task_scores={t:summary(for_field(rows,field),labels,[k for k in requested if labels[k]['subset']==t]) for t in tasks}
                for threshold in ['0.125/0.875','0.2/0.8']:
                    if scores.get('n',0)==0:continue
                    point={'experiment':folder.parent.name,'method':prediction.parents[1].name,
                           'population':args.score_split or args.population,
                           'condition':prediction.stem,'field':field,'threshold':threshold,
                           'expected':len(requested),'valid':scores['n'],'baseline_valid':baseline['n'],
                           'mae':scores['mae'],'baseline_mae':baseline['mae'],
                           'delta_mae':scores['mae']-baseline['mae']}
                    point['task_macro_mae']=float(np.mean([v['mae'] for v in task_scores.values()])) if all(v.get('n') for v in task_scores.values()) else None
                    point['baseline_task_macro_mae']=float(np.mean([v['mae'] for v in base_tasks.values()])) if all(v.get('n') for v in base_tasks.values()) else None
                    point['delta_task_macro_mae']=(point['task_macro_mae']-point['baseline_task_macro_mae']
                          if point['task_macro_mae'] is not None and point['baseline_task_macro_mae'] is not None else None)
                    for s in ['all','suc','fail']:
                        a=scores['accuracy'][threshold][s]['rate_all_expected']
                        b=baseline['accuracy'][threshold][s]['rate_all_expected']
                        point[f'accuracy_{s}']=a;point[f'baseline_accuracy_{s}']=b
                        point[f'delta_{s}']=a-b if a is not None and b is not None else None
                    point['meets_descriptive_gate']=(scores['n']==baseline['n']==len(requested)
                          and point['delta_mae']<0 and point['delta_all']>=.1-1e-12
                          and point['delta_suc']>0 and point['delta_fail']>0)
                    if args.uncertainty:
                        statistic=paired_statistics(for_field(base,field),for_field(rows,field),labels,requested,
                                      tuple(map(float,threshold.split('/'))))
                        for name,value in statistic.get('metrics',{}).items():
                            point[name+'_ci_low'],point[name+'_ci_high']=value['ci95']
                            point[name+'_p']=value['one_sided_cluster_signflip_p']
                    points.append(point)
                key=f'{folder.parent.name}/{prediction.parents[1].name}/{prediction.stem}/{field}'
                details[key]={'baseline':baseline,'intervention':scores,
                             'by_task':task_scores,'baseline_by_task':base_tasks}
    name=args.name or time.strftime('checkpoint_%Y%m%d_%H%M%S')+'_'+(args.score_split or args.population)
    output=OUT/'analysis'/name
    if args.uncertainty:
        for metric in ['delta_mae','delta_accuracy_all','delta_accuracy_suc','delta_accuracy_fail']:
            eligible=[p for p in points if metric+'_p' in p]
            corrected=holm([p[metric+'_p'] for p in eligible])
            for p,value in zip(eligible,corrected):p[metric+'_holm_p']=value
    create_json(output/'points.json',points)
    create_json(output/'details.json',details)
    if points:
        with (output/'points.csv').open('x') as f:
            keys=list(dict.fromkeys(k for p in points for k in p))
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(points)
    print('COMPLETE POINTS',len(points),'PASS',sum(p['meets_descriptive_gate'] for p in points))
    grouped=defaultdict(list)
    for p in points:
        if p['threshold']=='0.125/0.875':grouped[(p['experiment'],p['field'])].append(p)
    for key,group in grouped.items():
        print(key)
        for p in sorted(group,key=lambda p:(not p['meets_descriptive_gate'],p['mae']))[:3]:
            print(p['method'],p['condition'], 'n',p['valid'], 'MAE',round(p['baseline_mae'],3),'->',round(p['mae'],3),
                  'Δacc(all/suc/fail)',*[round(p[f'delta_{s}']*100,2) for s in ['all','suc','fail']],
                  'PASS',p['meets_descriptive_gate'])
    print(output)

if __name__=='__main__':main()
