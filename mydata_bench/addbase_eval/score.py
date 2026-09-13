"""Independent scoring from immutable raw outputs and label-only manifests."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from numbers import Real
from pathlib import Path
import numpy as np

from .prepare import OUT, create_json
from .run import latest
from mydata_bench.top_eval.versioning import validate_records, validate_protocol_config

ORDINAL=(.125,.375,.625,.875)
EQUAL=(.2,.4,.6,.8)


def ordinal(p): return 1+sum(p>=t for t in ORDINAL)
def distribution(p): return 1+sum(p>=t for t in EQUAL)
def endpoint(p,low,high): return 1 if p<=low else 5 if p>=high else 0


def valid_prediction(row):
    value = row.get('progress')
    return (row.get('status') == 'ok' and isinstance(value, Real)
            and not isinstance(value, bool) and math.isfinite(value))


def summary(rows, labels, expected):
    ids=set(expected)
    good={k:r for k,r in rows.items() if k in ids and valid_prediction(r)}
    n=len(good)
    out={'expected':len(ids),'n':n,'invalid':len(ids)-n,'coverage':n/len(ids) if ids else None}
    out['missing_ids']=sorted(ids-set(rows))
    out['invalid_ids']=sorted(ids-set(good))
    if not n:return out
    out['mae']=float(np.mean([abs(ordinal(r['progress'])-labels[k]['reward']) for k,r in good.items()]))
    absolute_sum=sum(abs(ordinal(r['progress'])-labels[k]['reward']) for k,r in good.items())
    out['mae_all_expected_bounds']=[absolute_sum/len(ids),(absolute_sum+4*(len(ids)-n))/len(ids)]
    out['continuous_ordinal_mae']=float(np.mean([abs(1+4*np.clip(r['progress'],0,1)-labels[k]['reward']) for k,r in good.items()]))
    out['mean_progress']=float(np.mean([r['progress'] for r in good.values()]))
    out['mean_progress_by_split']={s:float(np.mean([r['progress'] for k,r in good.items() if labels[k]['split']==s]))
                                  for s in ['suc','fail'] if any(labels[k]['split']==s for k in good)}
    out['ordinal_exact_accuracy']=sum(ordinal(r['progress'])==labels[k]['reward'] for k,r in good.items())/n
    out['accuracy']={}
    for low,high in [(.125,.875),(.2,.8)]:
        subsets={'all':set(ids),'suc':{k for k in ids if labels[k]['split']=='suc'},
                 'fail':{k for k in ids if labels[k]['split']=='fail'}}
        record={}
        for split,requested in subsets.items():
            valid=requested&set(good)
            correct=sum(endpoint(good[k]['progress'],low,high)==labels[k]['reward'] for k in valid)
            record[split]={'correct':correct,'valid':len(valid),'expected':len(requested),
                            'rate_valid':correct/len(valid) if valid else None,
                            'rate_all_expected':correct/len(requested) if requested else None}
        if record['suc']['rate_valid'] is not None and record['fail']['rate_valid'] is not None:
            record['balanced_accuracy']=(record['suc']['rate_valid']+record['fail']['rate_valid'])/2
        out['accuracy'][f'{low}/{high}']=record
    for field,classify in [('prediction_distributions',distribution),('ordinal_prediction_distributions',ordinal)]:
        out[field]={}
        for split in ['all','suc','fail']:
            selected={k:r for k,r in good.items() if split=='all' or labels[k]['split']==split}
            counts=Counter(classify(r['progress']) for r in selected.values())
            out[field][split]={'n':len(selected),'counts':{str(i):counts[i] for i in range(1,6)},
                    'rates':{str(i):counts[i]/len(selected) if selected else None for i in range(1,6)}}
    out['pairwise']=pairwise(good,labels)
    return out


def pairwise(rows,labels):
    pairs=[]
    for fid,r in rows.items():
        label=labels[fid]
        if label['split']!='fail':continue
        sid=label['source_suc_id']
        if sid not in rows:continue
        if label['video_sha256']!=labels[sid]['video_sha256']:raise ValueError('Source pair videos differ')
        dp=rows[sid]['progress']-r['progress']
        di=ordinal(rows[sid]['progress'])-ordinal(r['progress'])
        pairs.append({'suc_id':sid,'fail_id':fid,'continuous_delta':dp,'ordinal_delta':di})
    bins=Counter('<0' if p['ordinal_delta']<0 else str(p['ordinal_delta']) for p in pairs)
    continuous_bins=['<0','0','(0,0.1)']+[f'[{i/10:.1f},{(i+1)/10:.1f})' for i in range(1,9)]+['[0.9,1.0]','>1.0']
    continuous_counts=Counter()
    for pair in pairs:
        delta=pair['continuous_delta']
        if delta < -1e-9:key='<0'
        elif abs(delta)<=1e-9:key='0'
        elif delta<.1:key='(0,0.1)'
        elif delta>1:key='>1.0'
        elif delta>=.9:key='[0.9,1.0]'
        else:
            i=min(8,int(np.floor(delta*10)))
            key=f'[{i/10:.1f},{(i+1)/10:.1f})'
        continuous_counts[key]+=1
    n=len(pairs)
    return {'n':n,'unique_suc_videos':len({p['suc_id'] for p in pairs}),
            'continuous_mean_delta':float(np.mean([p['continuous_delta'] for p in pairs])) if n else None,
            'continuous_positive_rate':sum(p['continuous_delta']>1e-9 for p in pairs)/n if n else None,
            'continuous_tie_rate':sum(abs(p['continuous_delta'])<=1e-9 for p in pairs)/n if n else None,
            'ordinal_difference_counts':{k:bins[k] for k in ['<0','0','1','2','3','4']},
            'ordinal_difference_rates':{k:bins[k]/n if n else None for k in ['<0','0','1','2','3','4']},
            'continuous_difference_counts':{k:continuous_counts[k] for k in continuous_bins},
            'continuous_difference_rates':{k:continuous_counts[k]/n if n else None for k in continuous_bins},
            'continuous_negative_rate':continuous_counts['<0']/n if n else None,
            'continuous_strong_positive_ge_0p5_rate':sum(p['continuous_delta']>=.5 for p in pairs)/n if n else None,
            'pairs':pairs}


def paired_change(base,rows,labels,expected,seed=20260909):
    valid=[k for k in expected if valid_prediction(base.get(k,{})) and valid_prediction(rows.get(k,{}))]
    if not valid:return {'n':0}
    groups=defaultdict(list)
    for k in valid:groups[labels[k]['video_sha256']].append(k)
    def delta(k):return abs(ordinal(rows[k]['progress'])-labels[k]['reward'])-abs(ordinal(base[k]['progress'])-labels[k]['reward'])
    totals=np.array([sum(delta(k) for k in ks) for ks in groups.values()],dtype=float)
    sizes=np.array([len(ks) for ks in groups.values()])
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(totals),size=(5000,len(totals)))
    boot=totals[draws].sum(1)/sizes[draws].sum(1)
    signs=rng.choice([-1,1],size=(10000,len(totals)))
    null=(signs*totals).sum(1)/sizes.sum()
    estimate=totals.sum()/sizes.sum()
    return {'n':len(valid),'expected':len(expected),'complete':len(valid)==len(expected),'video_clusters':len(groups),
            'mae_delta':float(estimate),'mae_delta_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'paired_cluster_signflip_p':float((1+sum(np.abs(null)>=abs(estimate)-1e-12))/(len(null)+1)),
            'bootstrap_samples':5000,'signflip_samples':10000,'seed':seed,
            'accuracy_delta_0125':{s:float(np.mean([int(endpoint(rows[k]['progress'],.125,.875)==labels[k]['reward'])-
                                                        int(endpoint(base[k]['progress'],.125,.875)==labels[k]['reward'])
                                                        for k in valid if labels[k]['split']==s]))
                                     for s in ['suc','fail'] if any(labels[k]['split']==s for k in valid)}}


def score_experiment(folder, inputs, labels, output):
    baseline=latest(folder/'predictions/baseline.jsonl')
    cfg=json.loads((folder/'run_config.json').read_text())
    validate_protocol_config(cfg)
    if cfg.get('inputs') and json.loads(Path(cfg['inputs']).read_text()) != inputs:
        raise ValueError('Scoring inputs differ from the run manifest; do not score a custom run against the frozen population')
    all_ids=[s['example_id'] for s in inputs]
    cohort=[s['example_id'] for s in inputs if s['cohort']]
    holdout=[s['example_id'] for s in inputs if s['cohort'] and s['holdout']]
    result={'model':cfg['model'],'protocol':cfg['protocol'],'config':cfg,'conditions':{}}
    loaded={}
    for path in sorted((folder/'predictions').glob('*.jsonl')):
        rows=latest(path)
        validate_records(cfg, rows.values(), path)
        condition=next(iter(rows.values()))['condition'] if rows else path.stem
        if any(r.get('condition') != condition for r in rows.values()):
            raise ValueError(f'Mixed prediction conditions in {path}')
        expected_ids = set(all_ids if condition=='baseline' else cohort)
        if set(rows)-expected_ids:
            raise ValueError(f'Predictions outside the expected population in {path}')
        loaded[condition]=rows
        groups={'cohort':cohort,'holdout':holdout}
        if condition=='baseline':groups['full']=all_ids
        record={}
        for name,ids in groups.items():
            record[name]=summary(rows,labels,ids)
            record[name]['by_task']={task:summary(rows,labels,[k for k in ids if labels[k]['subset']==task])
                                    for task in sorted({labels[k]['subset'] for k in ids})}
            if condition!='baseline':record[name]['paired_change']=paired_change(baseline,rows,labels,ids)
        result['conditions'][condition]=record
    result['matched_controls']={}
    for scope in cfg['scopes']:
        for k in cfg['top_k']:
            names=['baseline']+[f'{scope}:{kind}:{k}' for kind in ['target','wrong_region','low_rank']]
            if not all(name in loaded for name in names):continue
            record={}
            for population,requested in [('cohort',cohort),('holdout',holdout)]:
                common=[eid for eid in requested if all(valid_prediction(loaded[name].get(eid,{})) for name in names)]
                record[population]={'expected_population':len(requested),'common_n':len(common),
                                    'excluded_ids':sorted(set(requested)-set(common)),
                                    'conditions':{name:summary(loaded[name],labels,common) for name in names}}
            result['matched_controls'][f'{scope}:{k}']=record
    if cfg['model']=='meter':
        # The official interface returns BOTH heads. Progress stays the primary
        # endpoint; publish success-head diagnostics without choosing between them.
        result['secondary_success_head']={}
        for condition,rows in loaded.items():
            converted={eid:{**r,'progress':r.get('success_probability')} for eid,r in rows.items()}
            groups={'cohort':cohort,'holdout':holdout}
            if condition=='baseline':groups['full']=all_ids
            result['secondary_success_head'][condition]={pop:summary(converted,labels,ids) for pop,ids in groups.items()}
    create_json(output/(folder.name+'.json'),result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-name',default='analysis_v1')
    p.add_argument('--configs',nargs='+',help='Explicit config matrix defining the Holm family; defaults to the historical frozen matrix')
    p.add_argument('--experiments',nargs='+',help='Optional explicit checkpoint subset; Holm family still includes the full frozen matrix')
    args=p.parse_args()
    matrix=args.configs or json.loads((OUT/'matrix.json').read_text())
    requested=set(args.experiments) if args.experiments else None
    if requested is not None and not requested<={Path(path).stem for path in matrix}:
        p.error('Unknown experiment name in --experiments')
    destination=OUT/args.output_name;destination.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((OUT/'inputs.json').read_text());labels=json.loads((OUT/'labels_for_scoring_only.json').read_text())
    references={}
    for progress in [0.,.5,1.]:
        rows={s['example_id']:{'status':'ok','progress':progress} for s in inputs}
        references[str(progress)]={pop:summary(rows,labels,[s['example_id'] for s in inputs if
                pop=='full' or (s['cohort'] and (pop=='cohort' or s['holdout']))]) for pop in ['full','cohort','holdout']}
    create_json(destination/'constant_reference_metrics.json',references)
    results={}
    for cfg_path in matrix:
        import yaml
        cfg=yaml.safe_load(Path(cfg_path).read_text());folder=Path(cfg['output_dir'])
        if requested is not None and folder.name not in requested:continue
        if not (folder/'run_config.json').exists():continue
        results[folder.name]=score_experiment(folder,inputs,labels,destination)
        print(folder.name,'scored',len(results[folder.name]['conditions']),flush=True)
    # Adjust all target-condition tests as one family per analysis population.
    holm={}
    for population in ['cohort','holdout']:
        entries=[]
        eligibility={}
        for cfg_path in matrix:
            cfg=yaml.safe_load(Path(cfg_path).read_text());experiment=Path(cfg['output_dir']).name
            for scope in cfg['scopes']:
                for k in cfg['top_k']:
                    condition=f'{scope}:target:{k}';key=f'{experiment}/{condition}'
                    change=results.get(experiment,{}).get('conditions',{}).get(condition,{}).get(population,{}).get('paired_change',{})
                    complete=change.get('complete',False)
                    eligibility[key]=complete
                    # Retain ALL frozen comparisons in the family, assigning
                    # p=1 to incomplete comparisons; missing output never shrinks it.
                    entries.append((key,change['paired_cluster_signflip_p'] if complete else 1.0))
        entries.sort(key=lambda x:x[1]);acc=0;corrected={}
        for i,(key,pvalue) in enumerate(entries):
            acc=max(acc,min(1,pvalue*(len(entries)-i)));corrected[key]=acc
        holm[population]={'family_size':len(entries),'adjusted_p':corrected,'complete_comparison':eligibility,
                          'incomplete_handling':'p=1 for Holm; matched-subset descriptive CI retained separately'}
    create_json(destination/'holm.json',holm)
    # Flat tables permit independent numerical checks without reading Markdown.
    flat=[]
    for experiment,result in results.items():
        for condition,record in result['conditions'].items():
            for pop,data in record.items():
                row={'experiment':experiment,'condition':condition,'population':pop,
                     **{k:data.get(k) for k in ['expected','n','invalid','coverage','mae','continuous_ordinal_mae','mean_progress']}}
                for threshold in ['0.125/0.875','0.2/0.8']:
                    for split in ['all','suc','fail']:
                        accuracy=data.get('accuracy',{}).get(threshold,{}).get(split,{})
                        # Preserve the legacy valid-denominator columns, while
                        # making both denominator choices explicit for CSV users.
                        row[f'acc_{threshold}_{split}']=accuracy.get('rate_valid')
                        row[f'acc_valid_{threshold}_{split}']=accuracy.get('rate_valid')
                        row[f'acc_fixed_{threshold}_{split}']=accuracy.get('rate_all_expected')
                        row[f'correct_{threshold}_{split}']=accuracy.get('correct')
                        row[f'expected_{split}']=accuracy.get('expected')
                flat.append(row)
    if flat:
        with (destination/'metrics.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    create_json(destination/'index.json',{'experiments':list(results),'rows':len(flat), 'config_matrix':matrix,
                'accuracy_columns':{'acc_*':'legacy valid-output denominator',
                                    'acc_valid_*':'valid-output denominator',
                                    'acc_fixed_*':'fixed expected denominator; invalid outputs not correct',
                                    'correct_*':'correct prediction count',
                                    'expected_*':'fixed expected split count'}})


if __name__=='__main__':main()
