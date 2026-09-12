"""Complete, same-head native visual-contrast versus original +/-6 comparison."""
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


def native(rows, ids, factorized=False):
    if set(rows) != set(ids) or any(r['status'] != 'ok' for r in rows.values()):
        raise ValueError('All and only the complete requested observations are required')
    for r in rows.values():
        if r.get('contrast_negative_mode', 'visual') != ('visual_and_task' if factorized else 'visual') or 'native_class_logits_reference' in r:
            raise ValueError('This comparator requires the explicitly requested unanchored contrast mode')
        pos=np.asarray(r['native_class_logits_positive'],dtype=np.float32);alpha=r['contrast_weight']
        neg=np.asarray(r['native_class_logits_negative'],dtype=np.float32) if alpha else None
        if any(z.shape != (5,) or not np.isfinite(z).all() for z in ([pos,neg] if alpha else [pos])):
            raise ValueError('Require finite native five-class branches')
        if factorized:
            if r['actual_forward_branches'] != (3 if alpha else 1):
                raise ValueError('Require three actual factorized branches or one baseline')
            if alpha:
                task=np.asarray(r['native_class_logits_negative_task'],dtype=np.float32)
                if task.shape != (5,) or not np.isfinite(task).all() or alpha != 1:
                    raise ValueError('Require the frozen actual task-negative cell and alpha1')
                mean=.5*(neg+task)
                if not r['negative_mean_is_derived'] or not np.array_equal(mean,r['native_class_logits_negative_mean']):
                    raise ValueError('The factorized negative mean does not reconstruct')
                neg=mean
        z=((1+alpha)*pos-alpha*neg if alpha else pos).astype(float)
        p=np.exp(z-z.max());p/=p.sum()
        if (np.max(np.abs(p-r['native_class_probabilities'])) >= 1e-5 or r['reward'] != int(p.argmax())+1
                or r['progress'] != (r['reward']-1)/4):
            raise ValueError('Recorded output differs from the actual native formula')
    return rows


def heads(row, field):
    return {(int(layer),h) for layer,d in row[field].items() for h in d['heads']}


def same_input(a,b):
    return all(a[k] == b[k] for k in ['prompt','candidate_token_ids']) and (
        a['token_audit']['input_ids_sha256'] == b['token_audit']['input_ids_sha256'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--bias-experiment',required=True)
    parser.add_argument('--method',default='binding_transport_s4')
    parser.add_argument('--scope',required=True,choices=['all_frames','last_frame'])
    parser.add_argument('--ks',nargs='+',type=int,required=True)
    parser.add_argument('--factorized',action='store_true',help='Explicitly verify the three actual factorized branches')
    args=parser.parse_args()
    roots=[OUT/'experiments'/n/'full_cohort' for n in [args.experiment,args.bias_experiment]]
    if any(not p.resolve().is_relative_to((OUT/'experiments').resolve()) for p in roots):
        raise ValueError('Only this session can be compared')
    if args.experiment.split('_')[0] != args.bias_experiment.split('_')[0]:
        raise ValueError('Require the same model')
    splits=json.loads((OUT/'splits.json').read_text());ids=splits['full_cohort'];sources={}
    def read(path, factorized=False):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return native(latest(path),ids,factorized)
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
    path=OUT/'analysis'/time.strftime(f'native_bias_comparison_{args.experiment}_%Y%m%d_%H%M%S.json')
    create_json(path,dict(arguments=vars(args),sources_sha256=sources,results=results,
        exact_same_batch_baseline_and_head_checks=len(ids)*len(args.ks),
        interpretation='Actual same-head and same-input native-readout comparison; all requested conditions complete. '
            'Paired video-cluster intervals remain descriptive after adaptive exploration; no inference or parameter change.'))
    print(path)
    for key,r in results.items():
        print(key,{n:dict(estimate=v['estimate'],ci95=v['ci95']) for n,v in r['contrast_minus_original_bias']['metrics'].items()})


if __name__=='__main__':
    main()
