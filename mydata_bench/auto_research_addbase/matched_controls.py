"""Compare region and rank controls on the identical valid instruction set."""
import argparse
import json
import time
from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .analyze import for_field
from .statistics import paired_statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--scope',default='all_frames')
    p.add_argument('--ks',nargs='+',type=int,required=True)
    p.add_argument('--score-split',default='validation')
    args=p.parse_args()
    folder=OUT/'experiments'/args.experiment/'full_cohort'
    requested=set(json.loads((OUT/'splits.json').read_text())[args.score_split])
    labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    full_requested=set(json.loads((folder/'requested_ids.json').read_text()))
    baseline=latest(folder/'predictions/baseline.jsonl')
    result={}
    for k in args.ks:
        rows={'baseline':baseline}
        for kind in ['target','wrong_region','low_rank']:
            rows[kind]=latest(folder/args.method/'predictions'/f'{args.scope}_{kind}_{k}.jsonl')
        if any(set(r)!=full_requested for r in rows.values()):
            raise ValueError(f'Wait for all complete conditions at k{k}')
        for field in ['progress']+(['success_probability'] if args.experiment.startswith('meter_') else []):
            converted={name:for_field(value,field) for name,value in rows.items()}
            common=sorted(eid for eid in requested if all(
                value[eid]['status']=='ok' and value[eid].get('progress') is not None for value in converted.values()))
            entry={'expected':len(requested),'common_n':len(common),'excluded_ids':sorted(requested-set(common)),
                   'metrics':{name:summary(value,labels,common) for name,value in converted.items()},'paired_changes':{}}
            for reference in ['baseline','wrong_region','low_rank']:
                entry['paired_changes']['target_minus_'+reference]={
                    str(threshold):paired_statistics(converted[reference],converted['target'],labels,common,threshold)
                    for threshold in [(.125,.875),(.2,.8)]}
            result[f'k{k}/{field}']=entry
    path=OUT/'analysis'/time.strftime('matched_controls_%Y%m%d_%H%M%S.json')
    create_json(path,{'arguments':vars(args),'interpretation':'Identical valid set for all four arms; unavailable equal-area controls are listed, never imputed. Pairwise CIs are exploratory and not multiplicity corrected.',
                     'results':result})
    print(path)
    for key,entry in result.items():
        print(key,'common',entry['common_n'])
        for name,values in entry['metrics'].items():
            print(name,'MAE',round(values['mae'],3),'accuracy',*[round(values['accuracy']['0.125/0.875'][s]['rate_all_expected'],4) for s in ['all','suc','fail']])


if __name__=='__main__':main()
