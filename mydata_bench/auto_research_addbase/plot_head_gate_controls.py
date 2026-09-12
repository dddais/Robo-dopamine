"""Paired original-bias and strict ROI/head comparisons from completed audits only."""
import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .head_gate_worker import sha, verify_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reports', type=Path, nargs='+', required=True)
    p.add_argument('--population', choices=['validation','full_cohort','old_holdout'], default='validation')
    args = p.parse_args(); sources={}; reports=[]; rows=[]
    for source in args.reports:
        source=source.resolve()
        if not source.is_relative_to(OUT.resolve()):raise ValueError('Only this session')
        data=json.loads(source.read_text())
        if data['status']!='pass':raise ValueError('Completed row audit and paired comparison required')
        verify_sources(data['sources_sha256']);sources[str(source)]=sha(source)
        reports.append(data)
    reports.sort(key=lambda r:(r['point']['model'],r['point']['protocol']))
    dest=OUT/'analysis'/time.strftime(f'figures_head_gates_controls_{args.population}_%Y%m%d_%H%M%S');dest.mkdir()
    metrics=[('delta_mae','MAE change',1),('delta_accuracy_all','Total accuracy change (pp)',100),
             ('delta_accuracy_suc','Success accuracy change (pp)',100),('delta_accuracy_fail','Failure accuracy change (pp)',100)]
    for arm in ['original_bias','wrong_region','low_rank']:
        fig,axes=plt.subplots(len(reports),4,figsize=(17,3.2*len(reports)),squeeze=False,constrained_layout=True)
        for i,report in enumerate(reports):
            point=report['point'];ks=sorted(point['ks']);statistics=[];denominators=[]
            for k in ks:
                r=report['results'][f'k{k}/{args.population}']
                st=(r['target_minus_original_bias_full_population'] if arm=='original_bias' else
                    r['strict_paired_changes'][f'target_minus_{arm}'])
                n=r['expected'] if arm=='original_bias' else r['strict_common_n']
                if st['status']!='complete' or st['n']!=n:raise ValueError('Incomplete paired statistic')
                statistics.append(st);denominators.append(n)
                for metric,_,scale in metrics:
                    v=st['metrics'][metric]
                    rows.append(dict(model=point['model'],protocol=point['protocol'],scope=point['scope'],k=k,
                        population=args.population,comparator=arm,n=n,video_clusters=st['video_clusters'],metric=metric,
                        estimate=v['estimate'],ci_low=v['ci95'][0],ci_high=v['ci95'][1]))
            for j,(metric,label,scale) in enumerate(metrics):
                ax=axes[i,j];y=[s['metrics'][metric]['estimate']*scale for s in statistics]
                lower=[(s['metrics'][metric]['estimate']-s['metrics'][metric]['ci95'][0])*scale for s in statistics]
                upper=[(s['metrics'][metric]['ci95'][1]-s['metrics'][metric]['estimate'])*scale for s in statistics]
                ax.errorbar(ks,y,yerr=[lower,upper],fmt='o-',capsize=4,lw=1.7,color='#265a7f')
                ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.2)
                n=str(denominators[0]) if len(set(denominators))==1 else '/'.join(map(str,denominators))
                ax.set(xticks=ks,xlabel='Measured top-k',ylabel=label,
                    title=f'{point["model"]} / {point["protocol"]}, n={n}')
        fig.suptitle(f'Learned head gates minus {arm}: {args.population}; video-cluster95% intervals')
        note=('Complete population; exact same heads/input and original native readout.' if arm=='original_bias' else
              'Strict same-input common subset; exclusions fixed before scoring. '+
              ('Equal-size disjoint wrong ROI.' if arm=='wrong_region' else 'Low-ranked heads also differ in gradient-training exposure.'))
        fig.supxlabel(note+'\nAdaptive dataset-internal comparison; uncorrected intervals. Lines connect measured k only.',fontsize=9)
        fig.savefig(dest/f'target_minus_{arm}.png',dpi=150);fig.savefig(dest/f'target_minus_{arm}.pdf');plt.close(fig)
    with (dest/'paired_control_statistics.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verify_sources(sources)
    create_json(dest/'manifest.json',dict(sources_sha256=sources,population=args.population,inputs=[r['point'] for r in reports],
        interpretation='Statistical plots of fully verified controls; no new inference or method selection. ROI and low-rank controls have distinct limitations.'))
    print(dest,flush=True)


if __name__ == '__main__':main()
