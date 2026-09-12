"""Descriptive mechanism checks; masks never select or alter inference outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT


def aggregate(pairs):
    counts=Counter('<0' if row['ordinal_delta']<0 else str(row['ordinal_delta']) for row in pairs)
    return {'n':len(pairs),
        'ordinal_difference_counts':{k:counts[k] for k in ['<0','0','1','2','3','4']},
        'continuous_mean_delta':float(np.mean([p['continuous_delta'] for p in pairs])) if pairs else None,
        'strict_positive_fraction':sum(p['ordinal_delta']>0 for p in pairs)/len(pairs) if pairs else None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    if not args.checkpoint.resolve().is_relative_to(OUT.resolve()):raise ValueError('Wrong research session')
    geometry=json.loads((OUT/'analysis/roi_quantization_v1.json').read_text())
    details=json.loads((args.checkpoint/'details.json').read_text())
    results={}
    for key,record in details.items():
        experiment,method,condition,field=key.split('/')
        scope='last_frame' if condition.startswith('last_frame') else 'all_frames'
        # All nonofficial independent-image layouts share visual-relative masks.
        protocol=('official' if experiment.startswith('meter_official') else
                  'video_text' if any(p in experiment for p in ['text_video','video_text']) else 'image_text')
        identical={(p['suc_id'],p['fail_id']):p['identical']
                   for p in geometry[protocol]['scopes'][scope]['pair_details']}
        if experiment.startswith('sole_official'):continue  # Mosaic geometry differs.
        entry={}
        for group,is_identical in [('identical_roi',True),('different_roi',False)]:
            entry[group]={}
            for branch in ['baseline','intervention']:
                pairs=record[branch].get('pairwise',{}).get('pairs',[])
                selected=[p for p in pairs if identical.get((p['suc_id'],p['fail_id'])) is is_identical]
                entry[group][branch]=aggregate(selected)
            a,b=entry[group]['baseline'],entry[group]['intervention']
            if a['n']!=b['n']:raise ValueError('Mechanism comparison requires matched valid pairs')
            entry[group]['delta_mean_separation']=(b['continuous_mean_delta']-a['continuous_mean_delta'] if a['n'] else None)
        results[key]=entry
    path=OUT/'analysis'/time.strftime('paired_mask_%Y%m%d_%H%M%S.json')
    create_json(path,{'source_checkpoint':str(args.checkpoint.resolve()),
        'interpretation':'Post-inference descriptive stratification. Identical masks do not imply identical activations because instructions differ. Not a randomized causal test of grounding quality.',
        'results':results})
    print(path)
    for key,entry in results.items():
        if key.endswith('/progress') and key.startswith('roboreward'):
            print(key,{group:{'n':v['baseline']['n'],'delta_mean_separation':v['delta_mean_separation']} for group,v in entry.items()})


if __name__=='__main__':main()
