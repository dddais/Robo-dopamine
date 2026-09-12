"""Export both positive and negative completed round27 input neighborhoods."""
import csv
import argparse
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .prepare import OUT
from .empirical_profile import sha
from mydata_bench.addbase_eval.prepare import create_json


ALL_SOURCES = [('qwen_image_text | last_frame', 'checkpoint_20260912_013043_validation_auto_1789147843246178029'), ('qwen_image_text | last_frame', 'checkpoint_20260912_013054_full_cohort_auto_1789147854329486339'), ('roboreward_image_text | all_frames', 'checkpoint_20260912_013155_validation_auto_1789147915239170139'), ('roboreward_image_text | all_frames', 'checkpoint_20260912_013206_full_cohort_auto_1789147926374328525'), ('qwen_video_text | last_frame', 'checkpoint_20260912_015840_validation_auto_1789149520010609971'), ('qwen_video_text | last_frame', 'checkpoint_20260912_015851_full_cohort_auto_1789149531345259189'), ('roboreward_text_image | all_frames', 'checkpoint_20260912_020123_validation_auto_1789149683882226615'), ('roboreward_text_image | all_frames', 'checkpoint_20260912_020135_full_cohort_auto_1789149695017300873'), ('qwen_text_video | all_frames', 'checkpoint_20260912_022739_validation_auto_1789151259225842798'), ('qwen_text_video | all_frames', 'checkpoint_20260912_022750_full_cohort_auto_1789151270069047756'), ('roboreward_interleaved | all_frames', 'checkpoint_20260912_025052_validation_auto_1789152652304583355'), ('roboreward_interleaved | all_frames', 'checkpoint_20260912_025103_full_cohort_auto_1789152663438076678')]
SOURCES = ALL_SOURCES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-frozen', action='store_true')
    args = parser.parse_args()
    destination = OUT/'analysis'/time.strftime('figures_factorized_kl_completed_%Y%m%d_%H%M%S')
    destination.mkdir(exist_ok=False)
    points = []; sources = {}
    for label,name in ALL_SOURCES if args.all_frozen else SOURCES:
        path = OUT/'analysis'/name/'points.json'; sources[str(path)] = sha(path)
        for row in json.loads(path.read_text()):
            if row['threshold'] != '0.125/0.875': continue
            if row['expected'] != row['valid'] or row['field'] != 'progress':
                raise ValueError('Require complete discrete native observations')
            points.append(dict(row, label=f"{label} | k{row['condition'].split('_')[-1]}"))
    if args.all_frozen and (len(points) != 36 or len({p['experiment'] for p in points}) != 6):
        raise ValueError('Require exactly six frozen three-k inputs in both populations')
    create_json(destination/'points.json',points)
    with (destination/'points.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(points[0])); writer.writeheader(); writer.writerows(points)
    metrics = [('delta_mae','MAE change (lower is better)',1),
        ('delta_accuracy_all','Overall accuracy change (pp)',100),
        ('delta_accuracy_suc','Success accuracy change (pp)',100),
        ('delta_accuracy_fail','Failure accuracy change (pp)',100)]
    for population in ['validation','full_cohort']:
        rows = [r for r in points if r['population']==population]
        if args.all_frozen: rows.sort(key=lambda r:(r['experiment'], int(r['condition'].split('_')[-1])))
        fig,axes = plt.subplots(1,4,figsize=(16.5,9.2) if args.all_frozen else (15.5,4.7),sharey=True,layout='constrained')
        for ax,(metric,title,scale) in zip(axes,metrics):
            for y,row in enumerate(rows):
                name = metric.replace('accuracy_','')
                est,lo,hi = [row[f]*scale for f in [name,metric+'_ci_low',metric+'_ci_high']]
                color = '#16826b' if row['meets_descriptive_gate'] else '#b34c4c'
                if args.all_frozen and row['meets_descriptive_gate'] and (row['delta_mae_ci_high'] >= 0 or
                        any(row[f'delta_accuracy_{c}_ci_low'] <= 0 for c in ['all','suc','fail'])):
                    color = '#c58a12'
                ax.errorbar(est,y,xerr=np.array([[est-lo],[hi-est]]),fmt='o',capsize=3,color=color)
            ax.axvline(0,color='.3',linewidth=.8)
            if metric=='delta_accuracy_all':ax.axvline(10,color='.5',linestyle='--',linewidth=.8)
            ax.grid(axis='x',alpha=.18);ax.set_xlabel(title)
            ax.set_yticks(range(len(rows)),[r['label'] for r in rows],fontsize=9)
        axes[0].invert_yaxis()
        legend = ('Green: descriptive gates + favorable CIs; amber: gates pass but a CI touches/crosses zero; red: gate fails'
                  if args.all_frozen else 'Paired video-cluster 95% CIs; green: four descriptive gates pass; red: at least one fails')
        fig.suptitle(f'Three-branch KL ≤ 0.8, α ≤ 2 minus matched native baseline | {population}\n'+legend,fontsize=11)
        fig.supxlabel('Adaptive exploration within this dataset; neighboring k values are dependent. Dashed line: +10 pp overall criterion.',fontsize=9)
        fig.savefig(destination/f'{population}.png',dpi=170)
        fig.savefig(destination/f'{population}.pdf');plt.close(fig)
    create_json(destination/'manifest.json',dict(sources_sha256=sources,all_six_frozen_inputs=args.all_frozen,
        interpretation='All selected complete neighborhoods, including failures and weak success evidence. '
            'Green is a descriptive per-condition gate, not independent confirmation or model-family completion.'))
    print(destination)


if __name__=='__main__':main()
