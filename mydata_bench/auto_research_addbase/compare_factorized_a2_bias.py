"""Actual round24 alpha2 three-branch versus same-head original +/-6 comparison."""
import argparse
import hashlib
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .statistics import paired_statistics


from .compare_native_bias import native, heads, same_input
from .factorized_gain_tasks import verify_native


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--bias-experiment',required=True)
    parser.add_argument('--method',default='binding_transport_s4')
    parser.add_argument('--scope',required=True,choices=['all_frames','last_frame'])
    parser.add_argument('--ks',nargs='+',type=int,required=True)
    parser.set_defaults(factorized=True)
    args=parser.parse_args()
    if not args.experiment.endswith('_uniform_factorized_evidence_a2'):
        raise ValueError('This comparator is explicitly for round24 three-branch alpha2')
    roots=[OUT/'experiments'/n/'full_cohort' for n in [args.experiment,args.bias_experiment]]
    if any(not p.resolve().is_relative_to((OUT/'experiments').resolve()) for p in roots):
        raise ValueError('Only this session can be compared')
    if args.experiment.split('_')[0] != args.bias_experiment.split('_')[0]:
        raise ValueError('Require the same model')
    splits=json.loads((OUT/'splits.json').read_text());ids=splits['full_cohort'];sources={}
    def read(path, factorized=False):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows=latest(path)
        return verify_native(rows,ids,2.,baseline=path.name=='baseline.jsonl') if factorized else native(rows,ids)
    baselines=[read(root/'predictions/baseline.jsonl',args.factorized and i==0) for i,root in enumerate(roots)]
    if any(not same_input(baselines[0][e],baselines[1][e]) or
           baselines[0][e]['native_class_logits_positive'] != baselines[1][e]['native_class_logits_positive'] for e in ids):
        raise ValueError('The two same-batch native baselines do not match exactly')
    matrices={}
    for k in args.ks:
        left=read(roots[0]/args.method/'predictions'/f'{args.scope}_target_{k}.jsonl',args.factorized)
        right=read(roots[1]/'bias_s6/predictions'/f'{args.scope}_target_{k}.jsonl')
        for e in ids:
            a,b=left[e],right[e];h=heads(a,'attention_diagnostics')
            if (len(h)!=k or h!=heads(a,'negative_attention_diagnostics') or h!=heads(b,'attention_diagnostics')
                    or not same_input(a,b) or not same_input(a,baselines[0][e])):
                raise ValueError('Actual heads or input batches differ')
            if a['contrast_weight']<=0 or b['contrast_weight']!=0:
                raise ValueError('Require contrast against one uncontrasted original-bias forward')
            if args.factorized:
                if h != heads(a,'task_negative_attention_diagnostics'):
                    raise ValueError('Actual task-negative heads differ')
                for field,method,strength in [('attention_diagnostics','binding_transport',4),
                        ('negative_attention_diagnostics','mass_transport',-4),
                        ('task_negative_attention_diagnostics','binding_task_suppression',4)]:
                    for d in a[field].values():
                        if d['method'] != method or d['strength'] != strength:
                            raise ValueError('Actual three-branch intervention differs')
                        if method == 'binding_transport' and (d['task_binding_fraction'] != .5 or d['task_binding_distribution'] != 'uniform'):
                            raise ValueError('Actual positive task intervention differs')
                        if method == 'binding_task_suppression' and d['task_logit_strength'] != -4:
                            raise ValueError('Actual task-negative strength differs')
            if any(d.get('bias')!=6 or d.get('generated_text_key_bias')!=0 for d in b['attention_diagnostics'].values()):
                raise ValueError('The actual reference is not original +/-6 steering')
        matrices[k]=(left,right)
    all_labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels={e:all_labels[e] for e in ids};del all_labels
    results={}
    for k,(left,right) in matrices.items():
        for population in ['validation','full_cohort']:
            requested=splits[population]
            results[f'k{k}/{population}']=dict(metrics={name:summary(rows,labels,requested)
                for name,rows in [('baseline',baselines[0]),('contrast',left),('original_bias',right)]},
                contrast_minus_original_bias=paired_statistics(right,left,labels,requested))
    path=OUT/'analysis'/time.strftime(f'factorized_a2_native_bias_comparison_{args.experiment}_%Y%m%d_%H%M%S.json')
    create_json(path,dict(arguments=vars(args),sources_sha256=sources,results=results,
        exact_same_batch_baseline_and_head_checks=len(ids)*len(args.ks),
        interpretation='Actual same-head and same-input native-readout comparison; all requested conditions complete. '
            'Paired video-cluster intervals remain descriptive after adaptive exploration; no inference or parameter change.'))
    print(path)
    for key,r in results.items():
        print(key,{n:dict(estimate=v['estimate'],ci95=v['ci95']) for n,v in r['contrast_minus_original_bias']['metrics'].items()})


if __name__=='__main__':
    main()
