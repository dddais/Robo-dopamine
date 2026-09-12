"""Measured neighboring-k curves from an immutable primary-target checkpoint."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .empirical_profile import sha, verify_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--note', default='Dataset-internal adaptive comparison; no external generalization claim.')
    args = p.parse_args(); source = args.checkpoint.resolve()/'points.json'
    if not source.is_relative_to(OUT.resolve()): raise ValueError('Wrong research session')
    sources = {str(source):sha(source)}
    points = [r for r in json.loads(source.read_text()) if r['threshold']=='0.125/0.875' and r['field']=='progress' and '_target_' in r['condition']]
    decision_path=OUT/'learned_head_delta_reft_v1/coverage.json'
    decision=json.loads(decision_path.read_text());verify_sources(decision['sources_sha256'])
    populations={r['population'] for r in points}
    if len(populations)!=1:raise ValueError('One complete audited population required')
    population=next(iter(populations))
    if Path(decision['artifacts'][population]['checkpoint']).resolve()!=source.parent:
        raise ValueError('Only the fully audited target checkpoint may be plotted')
    sources[str(decision_path)]=sha(decision_path)
    expected={(f"{e['point']['model']}_{e['point']['protocol']}_learned_head_delta_reft_v1",f"{e['point']['scope']}_target_{k}")
        for e in decision['inputs'] for k in e['point']['ks']}
    if {(r['experiment'],r['condition']) for r in points}!=expected or len(points)!=len(expected):
        raise ValueError('Require every frozen measured k, without cherry-picking')
    if not points or any(not r['experiment'].endswith('learned_head_delta_reft_v1') or r['valid'] != r['expected'] or r['baseline_valid'] != r['expected'] for r in points):
        raise ValueError('Only complete round33 primary native target observations')
    experiments = sorted({r['experiment'] for r in points})
    fig, axes = plt.subplots(len(experiments),4,figsize=(17,3.1*len(experiments)),squeeze=False,constrained_layout=True)
    metrics = [('delta_mae','delta_mae','MAE change',1),
        ('delta_all','delta_accuracy_all','Total accuracy change (pp)',100),
        ('delta_suc','delta_accuracy_suc','Success accuracy change (pp)',100),
        ('delta_fail','delta_accuracy_fail','Failure accuracy change (pp)',100)]
    for i, name in enumerate(experiments):
        rows = sorted([r for r in points if r['experiment']==name],key=lambda r:int(r['condition'].rsplit('_',1)[1]))
        if len({r['condition'].split('_target_')[0] for r in rows}) != 1: raise ValueError('Each frozen input uses one scope')
        ks = [int(r['condition'].rsplit('_',1)[1]) for r in rows]
        model, protocol = name.removesuffix('_learned_head_delta_reft_v1').split('_',1)
        scope = rows[0]['condition'].split('_target_')[0]
        for j,(estimate,ci,label,scale) in enumerate(metrics):
            ax = axes[i,j]
            y = [r[estimate]*scale for r in rows]
            low = [(r[estimate]-r[ci+'_ci_low'])*scale for r in rows]
            high = [(r[ci+'_ci_high']-r[estimate])*scale for r in rows]
            ax.errorbar(ks,y,yerr=[low,high],fmt='o-',capsize=4,lw=1.7,color='#265a7f')
            ax.axhline(0,color='gray',lw=.8)
            if j==1: ax.axhline(10,color='#a45524',ls='--',lw=1,label='10 pp threshold');ax.legend(fontsize=8)
            ax.set(xticks=ks,xlabel='Measured top-k',ylabel=label,title=f'{model} / {protocol} / {scope}')
            ax.grid(alpha=.2)
    population = {r['population'] for r in points}
    if len(population)!=1: raise ValueError('One population per checkpoint figure')
    fig.suptitle(f'Round33 rank4 attention increment: {next(iter(population))}, n={points[0]["expected"]}; video-cluster 95% intervals')
    fig.supxlabel(args.note+'\nLines connect measured k only. Full family coverage is assessed across all frozen inputs.',fontsize=9)
    verify_sources(sources)
    dest = OUT/'analysis'/time.strftime('figures_head_delta_'+next(iter(population))+'_%Y%m%d_%H%M%S');dest.mkdir()
    fig.savefig(dest/'neighboring_k.png',dpi=150);fig.savefig(dest/'neighboring_k.pdf');plt.close(fig)
    create_json(dest/'plotted_points.json',points)
    create_json(dest/'manifest.json',dict(sources_sha256=sources,note=args.note,experiments=experiments,source_checkpoint=str(args.checkpoint.resolve())))
    print(dest,flush=True)


if __name__ == '__main__': main()
