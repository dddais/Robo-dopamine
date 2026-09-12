"""CPU-only supervised head selection restricted to frozen discovery labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT


def nll(row,reward,model):
    if row['status']!='ok':raise ValueError('Incomplete profiling output')
    if model=='meter':
        # Native success logit; use a stable logistic loss, with no thresholding.
        z=row['success_logit_positive']
        target=(reward-1)/4
        return max(z,0)-target*z+math.log1p(math.exp(-abs(z)))
    # All five native classes remain available during both training and inference.
    logits=np.asarray(row['native_class_logits_positive'],dtype=np.float64)
    if logits.shape!=(5,):raise ValueError('Expected all five native reward classes')
    return float(np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[reward-1])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['qwen','roboreward','meter'],required=True)
    p.add_argument('--protocols',nargs='+',required=True)
    p.add_argument('--name',required=True,help='New, immutable selection directory')
    args=p.parse_args()
    if Path(args.name).name!=args.name:raise ValueError('Selection name must be one path component')
    split=json.loads((OUT/'splits.json').read_text())
    discovery=set(split['discovery'])
    if discovery&set(split['validation']):raise ValueError('Training/validation overlap')
    all_labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels={eid:all_labels[eid] for eid in discovery}
    del all_labels
    destination=OUT/'functional_selections'/args.name
    for protocol in args.protocols:
        folder=OUT/'experiments'/f'{args.model}_{protocol}_functional_profile'/'discovery'
        requested=json.loads((folder/'requested_ids.json').read_text())
        if set(requested)!=discovery:raise ValueError('Head selection is restricted to exactly the frozen discovery set')
        baseline_path=folder/'predictions/baseline.jsonl'
        baseline=latest(baseline_path)
        if set(baseline)!=discovery:raise ValueError('Incomplete baseline')
        reference={eid:nll(baseline[eid],labels[eid]['reward'],args.model) for eid in requested}
        for scope in ['all_frames','last_frame']:
            profiles=[]
            for layer in range(8,36):
                source=folder/f'binding_transport_s4_layer{layer}/predictions/{scope}_target_8.jsonl'
                rows=latest(source)
                if set(rows)!=discovery:raise ValueError(f'Incomplete profile: {source}')
                change={eid:nll(rows[eid],labels[eid]['reward'],args.model)-reference[eid] for eid in requested}
                means={name:float(np.mean([v for eid,v in change.items() if labels[eid]['split']==name]))
                       for name in ['suc','fail']}
                cfg=json.loads((source.parents[1]/'intervention_config.json').read_text())
                profiles.append({'layer':layer,'delta_nll_suc':means['suc'],'delta_nll_fail':means['fail'],
                    'worst_class_delta_nll':max(means.values()),'balanced_delta_nll':sum(means.values())/2,
                    'heads':cfg['profiled_heads'][scope],'source':str(source),
                    'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
            profiles.sort(key=lambda row:(row['worst_class_delta_nll'],row['balanced_delta_nll'],row['layer']))
            ranking=[]
            for profile in profiles:
                ranking += [dict(h,score=-profile['worst_class_delta_nll'],profile_layer=profile['layer'],
                                 selection_source='discovery_functional_layer_profile') for h in profile['heads']]
            if len({(r['layer'],r['head']) for r in ranking})!=224:raise ValueError('Head budget mismatch')
            artifact={'scope':scope,'num_layers':36,'num_heads':32,'skip_early_layers':8,
                'ranking_score':'negative worst-class discovery NLL change, profiled in groups of 8 heads',
                'n':len(requested),'example_ids':requested,
                'query_kind':'final_prog_token' if args.model=='meter' else 'answer_format_prefix',
                'ranking':ranking,'layer_profiles':profiles,
                'supervision':'few-shot head selection; frozen model weights; discovery labels only',
                'loss_target':'native_success_head' if args.model=='meter' else 'all_five_native_reward_classes',
                'baseline_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                'candidate_pool':'8 independently dual-selected heads per layer 8..35; 224 total',
                'frozen_strength':4.,'task_binding_fraction':.5,'validation_ids_used':[]}
            artifact['selector_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            create_json(destination/f'{args.model}_{protocol}'/f'ranking_{scope}.json',artifact)
        print(destination/f'{args.model}_{protocol}')


if __name__=='__main__':main()
