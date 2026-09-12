"""Paired uncertainty with videos, not instruction variants, as clusters."""
from collections import defaultdict
import numpy as np
from mydata_bench.addbase_eval.score import ordinal, endpoint


def paired_statistics(baseline, intervention, labels, requested, threshold=(.125,.875), draws=5000, seed=20260911):
    ids=list(requested)
    invalid=[k for k in ids if k not in baseline or k not in intervention
             or baseline[k]['status']!='ok' or intervention[k]['status']!='ok'
             or baseline[k].get('progress') is None or intervention[k].get('progress') is None]
    if invalid:return {'status':'incomplete','invalid_pairs':len(invalid),'expected':len(ids)}
    grouped=defaultdict(list)
    for k in ids:grouped[labels[k]['video_sha256']].append(k)
    totals=[];denominators=[]
    for members in grouped.values():
        total=np.zeros(4);counts=np.zeros(4)
        for k in members:
            y=labels[k]['reward'];a=baseline[k]['progress'];b=intervention[k]['progress']
            error_change=abs(ordinal(b)-y)-abs(ordinal(a)-y)
            correct_change=int(endpoint(b,*threshold)==y)-int(endpoint(a,*threshold)==y)
            split=2 if labels[k]['split']=='suc' else 3
            total[0]+=error_change;total[1]+=correct_change;total[split]+=correct_change
            counts[0]+=1;counts[1]+=1;counts[split]+=1
        totals.append(total);denominators.append(counts)
    totals=np.asarray(totals);denominators=np.asarray(denominators)
    rng=np.random.default_rng(seed)
    index=rng.integers(0,len(totals),size=(draws,len(totals)))
    sampled_counts=denominators[index].sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        boot=totals[index].sum(1)/sampled_counts
        estimate=totals.sum(0)/denominators.sum(0)
    signs=rng.choice([-1,1],size=(draws,len(totals)))
    perm=(signs@totals)/denominators.sum(0)
    metrics={}
    for i,name in enumerate(['delta_mae','delta_accuracy_all','delta_accuracy_suc','delta_accuracy_fail']):
        finite=boot[:,i][np.isfinite(boot[:,i])]
        if not len(finite):continue
        p=(1+np.sum(perm[:,i]<=estimate[i]+1e-12 if i==0 else perm[:,i]>=estimate[i]-1e-12))/(draws+1)
        metrics[name]={'estimate':float(estimate[i]),'ci95':np.quantile(finite,[.025,.975]).tolist(),
                       'one_sided_cluster_signflip_p':float(p),'bootstrap_valid_draws':len(finite)}
    return {'status':'complete','n':len(ids),'video_clusters':len(grouped),'draws':draws,'seed':seed,
            'resampling_unit':'video_sha256; all instruction variants stay together','metrics':metrics}


def holm(values):
    """Familywise adjusted p-values, preserving caller order."""
    order=np.argsort(values);out=np.zeros(len(values));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(values)-rank)*values[index]))
        out[index]=previous
    return out.tolist()
