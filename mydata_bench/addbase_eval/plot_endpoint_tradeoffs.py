"""Standalone descriptive plot of the official-input endpoint tradeoffs."""
import argparse
import json
from mydata_bench.top_eval.versioning import load_analysis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .prepare import OUT, create_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-stem', default='official_endpoint_tradeoffs')
    args = parser.parse_args()
    folder = OUT / 'analysis_v1'
    names = ['meter_official', 'sole_official']
    thresholds = ['0.125/0.875', '0.2/0.8']
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for i, name in enumerate(names):
        data = load_analysis(folder / (name + '.json'))['conditions']
        base = data['baseline']['cohort']
        for j, threshold in enumerate(thresholds):
            ax = axes[i, j]
            for scope, color in [('last_frame', '#3369b0'), ('all_frames', '#c05b22')]:
                for k, marker in [(8, 'o'), (32, 's'), (64, '^')]:
                    condition = f'{scope}:target:{k}'
                    d = data[condition]['cohort']
                    delta = {s: 100 * (d['accuracy'][threshold][s]['rate_all_expected'] -
                                      base['accuracy'][threshold][s]['rate_all_expected'])
                             for s in ['suc', 'fail']}
                    assert d['expected'] == base['expected'] == 846
                    ax.scatter(delta['fail'], delta['suc'], color=color, marker=marker, s=60, zorder=3)
                    # Small deterministic text offsets only affect label placement.
                    offset = (6, 7) if k != 64 else (6, -13)
                    if name == 'sole_official' and threshold == '0.2/0.8' and scope == 'last_frame' and k == 64:
                        offset = (6, 10)
                    ax.annotate(('last' if scope == 'last_frame' else 'all') + f' {k}',
                                (delta['fail'], delta['suc']), textcoords='offset points',
                                xytext=offset, fontsize=8, color=color)
                    rows.append({'experiment': name, 'condition': condition, 'threshold': threshold,
                                 'population': 'cohort', 'expected': 846,
                                 'baseline_valid': base['n'], 'target_valid': d['n'],
                                 'suc_change_percentage_points': delta['suc'],
                                 'fail_change_percentage_points': delta['fail']})
            ax.axhline(0, color='#777777', linewidth=.8)
            ax.axvline(0, color='#777777', linewidth=.8)
            ax.margins(x=.2, y=.25)
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            ax.set_xlim(min(x0, -2), max(x1, 2)); ax.set_ylim(min(y0, -2), max(y1, 2))
            ax.grid(alpha=.15)
            ax.set_title(f'{name} | thresholds {threshold}')
            ax.set_xlabel('Failure accuracy change (percentage points)')
            ax.set_ylabel('Success accuracy change (percentage points)')
    fig.suptitle('Official inputs: endpoint accuracy changes relative to the same cohort baseline\n'
                 'Upper-right quadrant: both classes improve. Descriptive points; no accuracy significance tests.', fontsize=12)
    for extension in ['png', 'svg']:
        path = folder / f'{args.output_stem}.{extension}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)
    create_json(folder / f'{args.output_stem}_data.json', rows)
    print('Wrote official endpoint tradeoff PNG, SVG, and source point values.')


if __name__ == '__main__':
    main()
