"""Publication-exportable display of completed same-head paired comparisons."""
import csv
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .prepare import OUT
from .empirical_profile import sha


SOURCES = [
    ('Qwen image→text | uniform α2', 'native_bias_comparison_qwen_image_text_uniform_binding_evidence_a2_20260911_221845.json', '#5265a2'),
    ('Qwen image→text | three branches', 'native_bias_comparison_qwen_image_text_uniform_factorized_evidence_a1_20260911_224630.json', '#218c75'),
    ('Qwen text→image | three branches', 'native_bias_comparison_qwen_text_image_uniform_factorized_evidence_a1_20260911_224512.json', '#218c75'),
    ('RR interleaved | proportional α1', 'native_bias_comparison_roboreward_interleaved_functional_evidence_a1_20260911_223750.json', '#a36437'),
    ('RR interleaved | uniform α2', 'native_bias_comparison_roboreward_interleaved_uniform_binding_evidence_a2_20260911_225224.json', '#5265a2'),
]


def main():
    destination = OUT/'analysis'/time.strftime('figures_native_bias_comparisons_%Y%m%d_%H%M%S')
    destination.mkdir(exist_ok=False)
    rows = []; sources = {}
    for label,name,color in SOURCES:
        path = OUT/'analysis'/name; sources[str(path)] = sha(path)
        data = json.loads(path.read_text())
        for key,value in data['results'].items():
            k,population = key.split('/')
            for metric,stat in value['contrast_minus_original_bias']['metrics'].items():
                rows.append(dict(label=label+' | '+k,population=population,metric=metric,
                    estimate=stat['estimate'],low=stat['ci95'][0],high=stat['ci95'][1],color=color,source=name))
    with (destination/'paired_changes.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics = [('delta_mae','MAE change (lower is better)',1),
        ('delta_accuracy_all','Overall accuracy change (pp)',100),
        ('delta_accuracy_suc','Success accuracy change (pp)',100),
        ('delta_accuracy_fail','Failure accuracy change (pp)',100)]
    for population in ['validation','full_cohort']:
        labels = list(dict.fromkeys(r['label'] for r in rows if r['population']==population))
        fig,axes = plt.subplots(1,4,figsize=(16,6.6),sharey=True,layout='constrained')
        for ax,(metric,title,scale) in zip(axes,metrics):
            for y,label in enumerate(labels):
                r = next(r for r in rows if r['population']==population and r['label']==label and r['metric']==metric)
                est = r['estimate']*scale
                ax.errorbar(est,y,xerr=np.array([[est-r['low']*scale],[r['high']*scale-est]]),
                    fmt='o',markersize=5,capsize=3,color=r['color'])
            ax.axvline(0,color='.3',linewidth=.8)
            ax.grid(axis='x',alpha=.18); ax.set_xlabel(title); ax.set_yticks(range(len(labels)),labels,fontsize=9)
        axes[0].invert_yaxis()
        fig.suptitle(f'Native contrast minus original ±6 steering | {population}\n'
            'Same inputs and heads; paired video-cluster 95% CIs; each method remains a separate family',fontsize=12)
        fig.savefig(destination/f'{population}.png',dpi=170)
        fig.savefig(destination/f'{population}.pdf')
        plt.close(fig)
    with (destination/'manifest.json').open('x') as f:
        json.dump(dict(sources_sha256=sources,
            interpretation='Descriptive after adaptive exploration. No pooling of methods or additional efficacy selection. '
                'ROI/head control counterexamples are retained in the research document.'),f,indent=2)
    print(destination)


if __name__ == '__main__':
    main()
