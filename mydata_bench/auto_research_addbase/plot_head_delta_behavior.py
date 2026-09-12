"""Task heterogeneity and native distributions at the original discovery-selected centers."""
import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .head_delta_worker import BASE, sha, verify_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    args = p.parse_args(); sources = {}; data = {}
    selection_file = BASE/'selection_complete_discovery_v1.json'
    selection = json.loads(selection_file.read_text()); sources[str(selection_file)] = sha(selection_file)
    for checkpoint in args.checkpoints:
        path = checkpoint.resolve()/'details.json'
        if not path.is_relative_to(OUT.resolve()): raise ValueError('Only current research session')
        sources[str(path)] = sha(path)
        for key, value in json.loads(path.read_text()).items():
            if key in data: raise ValueError('Overlapping checkpoint conditions')
            data[key] = value
    records = []
    for point in selection['selected']:
        key = f'{point["model"]}_{point["protocol"]}_learned_head_delta_reft_v1/learned_head_delta_reft_s6/{point["scope"]}_target_{point["center_k"]}/progress'
        row = data[key]
        if row['baseline']['n'] != 846 or row['intervention']['n'] != 846:
            raise ValueError('This figure is restricted to complete full846 centers')
        records.append((point, row))
    tasks = sorted(records[0][1]['by_task'])
    if any(sorted(r['by_task']) != tasks for _, r in records): raise ValueError('Common task support required')
    dest = OUT/'analysis'/time.strftime('figures_head_delta_frozen_centers_behavior_%Y%m%d_%H%M%S'); dest.mkdir()
    changes = np.asarray([[r['by_task'][t]['mae']-r['baseline_by_task'][t]['mae'] for t in tasks] for _,r in records])
    labels = [f'{p["model"]} / {p["protocol"]} / k{p["center_k"]}' for p,_ in records]
    fig, ax = plt.subplots(figsize=(16,5.7), constrained_layout=True)
    limit = max(abs(changes.min()), abs(changes.max()))
    im = ax.imshow(changes, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set(xticks=range(len(tasks)), xticklabels=tasks, yticks=range(len(records)), yticklabels=labels,
        title='Frozen round33 centers: per-task MAE change, full n=846')
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=8)
    fig.colorbar(im, ax=ax, label='Intervention minus baseline MAE; blue is lower error')
    fig.supxlabel('One fixed discovery-selected center per input. Tasks vary in sample size; these are descriptive changes, not independent tests.',fontsize=9)
    fig.savefig(dest/'task_mae_changes.png',dpi=160);fig.savefig(dest/'task_mae_changes.pdf');plt.close(fig)
    distributions=[]; pairs=[]; task_rows=[]
    fig, axes = plt.subplots(len(records),2,figsize=(12,2.8*len(records)), constrained_layout=True,squeeze=False)
    for i,(point,r) in enumerate(records):
        for j,split in enumerate(['suc','fail']):
            ax=axes[i,j]; x=np.arange(1,6)
            for side,offset,color in [('baseline',-.18,'#999999'),('intervention',.18,'#265a7f')]:
                d=r[side]['ordinal_prediction_distributions'][split]
                if sum(d['counts'].values()) != d['n']:raise ValueError('Invalid native distribution count')
                y=[100*d['counts'][str(k)]/d['n'] for k in x]
                ax.bar(x+offset,y,width=.35,color=color,label=side)
                for k,count in d['counts'].items(): distributions.append(dict(model=point['model'],protocol=point['protocol'],k=point['center_k'],split=split,arm=side,reward=k,count=count,n=d['n']))
            ax.set(xticks=x,ylim=(0,100),ylabel='Prediction share (%)',title=f'{labels[i]} / {split}, n={d["n"]}')
            ax.grid(axis='y',alpha=.15)
            if i==0 and j==0:ax.legend()
        for t in tasks:
            a=r['by_task'][t];b=r['baseline_by_task'][t]
            if a['n'] != b['n']:raise ValueError('Changed task membership')
            task_rows.append(dict(model=point['model'],protocol=point['protocol'],k=point['center_k'],task=t,n=a['n'],baseline_mae=b['mae'],intervention_mae=a['mae'],delta_mae=a['mae']-b['mae']))
    fig.suptitle('Frozen round33 centers: all five native reward classes remain available')
    fig.supxlabel('Full846; native argmax over classes1–5. Endpoint-only training labels influence class frequencies. Center plots do not establish family coverage; complete measured-k intervals are separate.',fontsize=9)
    fig.savefig(dest/'native_class_distributions.png',dpi=140);fig.savefig(dest/'native_class_distributions.pdf');plt.close(fig)
    fig,axes=plt.subplots((len(records)+1)//2,2,figsize=(13,3.3*((len(records)+1)//2)),constrained_layout=True,squeeze=False)
    bins=['<0','0','1','2','3','4'];x=np.arange(6)
    for ax,(point,r),label in zip(axes.flat,records,labels):
        for side,offset,color in [('baseline',-.18,'#999999'),('intervention',.18,'#265a7f')]:
            d=r[side]['pairwise'];counts=d['ordinal_difference_counts']
            if sum(counts.values()) != d['n']:raise ValueError('Invalid native pair count')
            ax.bar(x+offset,[100*counts[k]/d['n'] for k in bins],width=.35,color=color,label=side)
            for k in bins:pairs.append(dict(model=point['model'],protocol=point['protocol'],k=point['center_k'],arm=side,ordinal_difference=k,count=counts[k],pairs=d['n']))
        ax.set(xticks=x,xticklabels=bins,ylabel='Pair share (%)',title=label,xlabel='Success minus failure native reward')
        ax.grid(axis='y',alpha=.15)
    for ax in list(axes.flat)[len(records):]:ax.set_visible(False)
    axes[0,0].legend()
    fig.suptitle('Same video, different instructions: original-center paired reward gaps')
    fig.supxlabel('All available instruction pairs stay with their video. Pair bins are descriptive; identical-ROI intervals are reported separately.',fontsize=9)
    fig.savefig(dest/'paired_ordinal_gaps.png',dpi=150);fig.savefig(dest/'paired_ordinal_gaps.pdf');plt.close(fig)
    for name,rows in [('class_distributions',distributions),('paired_gap_distributions',pairs),('task_mae_changes',task_rows)]:
        with (dest/f'{name}.csv').open('x') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verify_sources(sources)
    create_json(dest/'manifest.json',dict(sources_sha256=sources,centers=selection['selected'],n=846,
        interpretation='Discovery centers fixed before full inference; all frozen neighborhood k and shared coverage are reported separately. Dataset-internal adaptive exploration, not independent confirmation.'))
    print(dest,flush=True)


if __name__ == '__main__':main()
