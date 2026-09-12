"""Four complete round33 comparator plots from the final, verified synthesis."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .head_delta_worker import sha
from .summarize_head_delta_controls import ARMS, METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthesis', type=Path, required=True)
    parser.add_argument('--population', choices=['validation', 'full_cohort', 'old_holdout'], default='validation')
    args = parser.parse_args()
    source = (args.synthesis/'audit.json').resolve()
    if not source.is_relative_to(OUT.resolve()): raise ValueError('Only this session')
    digest = sha(source); data = json.loads(source.read_text())
    if data['status'] != 'pass': raise ValueError('Complete audited synthesis required')
    points = sorted(data['points'], key=lambda p: (p['model'], p['protocol']))
    dest = OUT/'analysis'/time.strftime(f'figures_head_delta_controls_{args.population}_%Y%m%d_%H%M%S')
    dest.mkdir()
    labels = ['MAE change', 'Total accuracy change (pp)', 'Success accuracy change (pp)', 'Failure accuracy change (pp)']
    for arm in ARMS:
        fig, axes = plt.subplots(len(points), 4, figsize=(17, 2.5*len(points)), squeeze=False, constrained_layout=True)
        for i, point in enumerate(points):
            key = f"{point['model']}/{point['protocol']}"
            cs = [c for c in data['comparisons'] if c['input'] == key and c['comparator'] == arm and c['population'] == args.population]
            cs.sort(key=lambda c: c['k']); ks = [c['k'] for c in cs]
            if ks != sorted(point['ks']): raise ValueError('Missing/extra frozen k')
            for j, metric in enumerate(METRICS):
                scale = 1 if j == 0 else 100; ax = axes[i, j]
                ys = [c['metrics'][metric]['estimate']*scale for c in cs]
                lo = [c['metrics'][metric]['ci95'][0]*scale for c in cs]
                hi = [c['metrics'][metric]['ci95'][1]*scale for c in cs]
                ax.plot(ks, ys, 'o-', color='#265a7f', lw=1.4, markersize=4)
                ax.vlines(ks, lo, hi, color='#265a7f', lw=1.2)
                ax.plot(ks, lo, '_', color='#265a7f'); ax.plot(ks, hi, '_', color='#265a7f')
                ax.axhline(0, color='gray', lw=.8); ax.grid(alpha=.2)
                ns = sorted(set(c['n'] for c in cs)); nlabel = str(ns[0]) if len(ns) == 1 else f'{ns[0]}-{ns[-1]}'
                ax.set(xticks=ks, xlabel='Measured top-k', ylabel=labels[j], title=f'{key}, n={nlabel}')
        fig.suptitle(f'Learned attention delta minus {arm}: {args.population}; video-cluster 95% intervals')
        note = ('Full population; same heads, input, native readout. '
            + ('B0 retains trained scalar gates; additional capacity and training are not isolated.' if arm == 'B0' else 'Extra supervision and training budget differ from original bias.')) if arm in ['B0','original_bias'] else (
                'Strict common input subset; exclusions fixed before labels. '
                + ('Equal-size disjoint ROI; no global ROI-specificity claim.' if arm == 'wrong_region' else 'Head identity and training exposure both change.'))
        fig.supxlabel(note+'\nAdaptive dataset-internal comparison; uncorrected intervals; lines connect measured k only.', fontsize=9)
        fig.savefig(dest/f'target_minus_{arm}.png', dpi=150)
        fig.savefig(dest/f'target_minus_{arm}.pdf'); plt.close(fig)
    if sha(source) != digest: raise ValueError('Source synthesis changed')
    create_json(dest/'manifest.json', dict(source_sha256={str(source): digest}, population=args.population,
        source_csv=str(args.synthesis/'paired_control_statistics.csv'), inputs=points,
        interpretation='Plotting verified statistics only, all frozen k retained.'))
    print(dest, flush=True)


if __name__ == '__main__':
    main()
