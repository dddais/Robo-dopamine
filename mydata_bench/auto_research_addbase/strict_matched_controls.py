"""Paired control sensitivity requiring identical padded native input tokens.

Individual retry after an unavailable ROI can alter batch padding. Keep the
original results and separately report the common exact-input subset.
"""
import argparse
import json
from pathlib import Path
import time

from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .compare_native_bias import native,heads,same_input
from .empirical_profile import sha
from .statistics import paired_statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--scope',required=True,choices=['all_frames','last_frame'])
    parser.add_argument('--ks',nargs='+',type=int,required=True)
    args=parser.parse_args()
    root=OUT/'experiments'/args.experiment/'full_cohort'
    if not root.resolve().is_relative_to((OUT/'experiments').resolve()): raise ValueError('Wrong session')
    sources={}
    def read(path,rows=False):
        sources[str(path)]=sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    splits=json.loads((OUT/'splits.json').read_text());ids=splits['full_cohort']
    if set(read(root/'requested_ids.json'))!=set(ids): raise ValueError('Require all full-cohort observations')
    cfg=read(root/'runtime_config.json')
    alpha=cfg['contrast_weight'];distribution=cfg.get('task_binding_distribution','attention')
    if (cfg.get('contrast_negative_mode','visual')!='visual' or cfg.get('contrast_reference','positive')!='positive'
            or cfg.get('negative_strength')!=4 or cfg.get('task_binding_fraction')!=.5):
        raise ValueError('This audit requires the original two visual-contrast branches')
    ranking_file=Path(cfg['frozen_ranking_source'])/f'ranking_{args.scope}.json'
    if not ranking_file.resolve().is_relative_to(OUT.resolve()): raise ValueError('Wrong head source')
    rank=read(ranking_file)['ranking']
    baseline=native(read(root/'predictions/baseline.jsonl',True),ids)
    matrices={};excluded={};audit=[]
    for k in args.ks:
        arms={'baseline':baseline};invalid={};different={}
        for kind in ['target','wrong_region','low_rank']:
            rows=read(root/'binding_transport_s4/predictions'/f'{args.scope}_{kind}_{k}.jsonl',True)
            if set(rows)!=set(ids): raise ValueError('Wait for every explicitly requested condition')
            valid={e:r for e,r in rows.items() if r['status']=='ok'}
            invalid[kind]=sorted(set(ids)-set(valid))
            for e in invalid[kind]:
                if kind!='wrong_region' or 'Wrong-region control unavailable: insufficient disjoint cells' not in str(rows[e]):
                    raise ValueError('Unexpected inference error')
            native(valid,list(valid))
            selected=rank[-k:] if kind=='low_rank' else rank[:k]
            expected={(h['layer'],h['head']) for h in selected}
            if len(expected)!=k: raise ValueError('Duplicate heads')
            different[kind]=[]
            for e,row in valid.items():
                if row['contrast_weight']!=alpha or row['condition']!=f'{args.scope}:{kind}:{k}':
                    raise ValueError('Wrong actual contrast/condition')
                if not same_input(row,baseline[e]): different[kind].append(e)
                for field,method,strength in [('attention_diagnostics','binding_transport',4),
                        ('negative_attention_diagnostics','mass_transport',-4)]:
                    if heads(row,field)!=expected: raise ValueError('Wrong actual heads')
                    for d in row[field].values():
                        if d['method']!=method or d['strength']!=strength: raise ValueError('Wrong actual operator')
                        if method=='binding_transport' and (d['task_binding_fraction']!=.5 or
                                d.get('task_binding_distribution','attention')!=distribution):
                            raise ValueError('Wrong actual task distribution')
                if kind=='wrong_region':
                    a=row['token_audit'];t=set(a['target'][args.scope]);w=set(a['wrong'][args.scope])
                    if not t or t&w or len(t)!=len(w): raise ValueError('Require equal-size disjoint ROI')
            arms[kind]=rows
            audit.append(dict(k=k,control=kind,valid=len(valid),expected=len(ids),
                input_mismatch_count=len(different[kind]),actual_native_heads_formula_and_roi_verified=True))
        matrices[k]=arms;excluded[k]=dict(unavailable=invalid,input_mismatch=different)
    # Dataset filtering for this sensitivity is fixed only by validity and exact inputs.
    all_labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels={e:all_labels[e] for e in ids};del all_labels
    results={}
    for k,arms in matrices.items():
        omitted=set(e for values in excluded[k].values() for members in values.values() for e in members)
        for population in ['validation','full_cohort']:
            requested=splits[population];common=[e for e in requested if e not in omitted]
            results[f'k{k}/{population}']=dict(expected=len(requested),common_n=len(common),
                excluded_ids=[e for e in requested if e in omitted],
                metrics={name:summary(rows,labels,common) for name,rows in arms.items()},
                paired_changes={f'target_minus_{name}':paired_statistics(rows,arms['target'],labels,common)
                    for name,rows in arms.items() if name!='target'})
    destination=OUT/'analysis'/time.strftime(f'strict_matched_controls_{args.experiment}_%Y%m%d_%H%M%S.json')
    create_json(destination,dict(arguments=vars(args),sources_sha256=sources,actual_audit=audit,
        exclusions_before_labels=excluded,results=results,
        interpretation='Sensitivity on identical padded input token hashes and valid common observations. '
            'Original full 846/validation660 primary metrics and earlier nominal-valid controls remain unchanged. '
            'This restricted set is not a replacement cohort; CIs remain exploratory.'))
    print(destination)
    for key,result in results.items():
        print(key,'common',result['common_n'],{name:{m:dict(estimate=v['estimate'],ci95=v['ci95'])
            for m,v in stat['metrics'].items()} for name,stat in result['paired_changes'].items() if name!='target_minus_baseline'})


if __name__=='__main__':main()
