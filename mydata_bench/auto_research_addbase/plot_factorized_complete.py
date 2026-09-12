"""Six completed factorized candidates; original ranges and explicit extension."""
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT


SOURCES = [
    ('Qwen: image → text (last)', ['factorized_full_ablation_qwen_image_text_20260911_195649.json',
                                  'factorized_full_ablation_qwen_image_text_20260911_204400.json']),
    ('Qwen: text → image (last)', ['factorized_full_ablation_qwen_text_image_20260911_203137.json']),
    ('Qwen: text → video (all)', ['factorized_full_ablation_qwen_text_video_20260911_210334.json']),
    ('RoboReward: image → text (all)', ['factorized_full_ablation_roboreward_image_text_20260911_195649.json']),
    ('RoboReward: text → image (all)', ['factorized_full_ablation_roboreward_text_image_20260911_204114.json']),
    ('RoboReward: interleaved (all)', ['factorized_full_ablation_roboreward_interleaved_20260911_212713.json'])]


def main():
    destination = OUT / 'analysis' / time.strftime('figures_factorized_complete_%Y%m%d_%H%M%S')
    destination.mkdir(exist_ok=False)
    panels, hashes, plotted = [], {}, []
    for title, files in SOURCES:
        entries = {}
        for name in files:
            source = OUT / 'analysis' / name
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            for key, record in json.loads(source.read_text())['results'].items():
                if not key.endswith('/validation'):
                    continue
                k = int(key.split('/')[0].split('_')[-1])
                if k in entries:
                    raise ValueError('Duplicate condition')
                if any(v['expected'] != 660 or v['n'] != 660 for v in record['metrics'].values()):
                    raise ValueError('Require complete validation for all arms')
                entries[k] = record
                plotted.append(dict(panel=title, k=k, data=record))
        panels.append((title, sorted(entries), entries))
    styles = [('combined_minus_visual_only', 'Combined − visual-only', '#0072b2', -0.10),
              ('combined_minus_task_only', 'Combined − task-only', '#d55e00', 0.10)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for ax, (title, ks, entries) in zip(axes.flat, panels):
        for key, label, color, offset in styles:
            values = [entries[k]['comparisons'][key]['metrics']['delta_accuracy_suc'] for k in ks]
            y = [100*v['estimate'] for v in values]
            error = [[100*(v['estimate']-v['ci95'][0]) for v in values],
                     [100*(v['ci95'][1]-v['estimate']) for v in values]]
            ax.errorbar([i+offset for i in range(len(ks))], y, yerr=error, fmt='o', color=color,
                        capsize=3, markersize=4, label=label)
        ax.axhline(0, color='gray', lw=1)
        ax.set(title=title, xlabel='Tested head budget k', ylabel='Success accuracy change (pp)',
               xticks=range(len(ks)), xticklabels=ks, xlim=(-.55, len(ks)-.45))
        if len(ks) == 6:
            ax.axvspan(2.5, 5.5, color='gray', alpha=.12)
            ax.text(.98, .02, 'Shaded: registered adaptive extension', transform=ax.transAxes,
                    ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc='outside lower center', ncols=2)
    fig.suptitle('Task counterfactual benefits vary across inputs\n'
                 'Validation660; same stored branches; 95% video-cluster intervals, descriptive and unadjusted')
    for suffix in ['png', 'pdf']:
        fig.savefig(destination / f'success_branch_comparisons.{suffix}', dpi=170)
    plt.close(fig)
    arms = [('baseline', 'Baseline', '#777777', '--'), ('positive_only', 'Positive', '#e69f00', '-'),
            ('visual_only', 'Visual-only', '#0072b2', '-'), ('task_only', 'Task-only', '#d55e00', '-'),
            ('combined', 'Combined', '#009e73', '-')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for ax, (title, ks, entries) in zip(axes.flat, panels):
        for key, label, color, line in arms:
            ax.plot(range(len(ks)), [entries[k]['metrics'][key]['mae'] for k in ks],
                    linestyle=line, marker='o', markersize=3, color=color, label=label)
        ax.set(title=title, xlabel='Tested head budget k', ylabel='Native reward MAE (lower is better)',
               xticks=range(len(ks)), xticklabels=ks, xlim=(-.35, len(ks)-.65))
        if len(ks) == 6:
            ax.axvspan(2.5, 5.5, color='gray', alpha=.12)
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8, loc='best')
    fig.suptitle('Complete three-branch ablations: original ranges plus explicitly registered image→text extension\n'
                 'Validation660; five native reward classes; connecting lines do not certify untested k')
    for suffix in ['png', 'pdf']:
        fig.savefig(destination / f'mae_branch_comparisons.{suffix}', dpi=170)
    plt.close(fig)
    create_json(destination / 'manifest.json', dict(sources_sha256=hashes,
        source_code_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
        population='validation660', plotted_conditions=len(plotted), plotted_values=plotted,
        interpretation='All completed frozen ranges shown. Explicit adaptive extension shaded. '
            'No partial text_image extension is included. Bootstrap intervals do not correct adaptive selection. '
            'Arm comparisons do not establish the four-metric acceptance gate or two-model coverage.'))
    print(destination)


if __name__ == '__main__':
    main()
