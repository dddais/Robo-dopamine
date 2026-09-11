"""Descriptive interpretation digest; never selects or changes inference settings.

Run after score.py. This supplement checks all target conditions and provides
task heterogeneity and matched-control descriptions without adding hypothesis
tests or treating the best observed k as a confirmatory selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from .prepare import OUT, create_json


THRESHOLDS = ['0.125/0.875', '0.2/0.8']


def rate(data, threshold, split):
    return data.get('accuracy', {}).get(threshold, {}).get(split, {}).get('rate_all_expected')


def subtract(a, b):
    return None if a is None or b is None else a - b


def direction(value):
    return 'missing' if value is None else 'lower' if value < -1e-12 else 'higher' if value > 1e-12 else 'equal'


def small(data):
    result = {k: data.get(k) for k in ['expected', 'n', 'invalid', 'mae', 'mae_all_expected_bounds',
                                      'continuous_ordinal_mae', 'mean_progress_by_split', 'prediction_distributions', 'ordinal_prediction_distributions']}
    result['accuracy'] = {t: {s: rate(data, t, s) for s in ['all', 'suc', 'fail']} for t in THRESHOLDS}
    pair = data.get('pairwise', {})
    result['pairwise'] = {k: v for k, v in pair.items() if k != 'pairs'}
    return result


def pair_equal_video(pair):
    grouped = {}
    for row in pair.get('pairs', []):
        grouped.setdefault(row['suc_id'], []).append(row['continuous_delta'])
    return float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None


def digest_experiment(result, holm, name):
    output = {'full_baseline': small(result['conditions']['baseline']['full']), 'populations': {}}
    for population in ['cohort', 'holdout']:
        baseline = result['conditions']['baseline'][population]
        targets = {}
        for condition, records in result['conditions'].items():
            if ':target:' not in condition:
                continue
            data = records[population]
            change = data.get('paired_change', {})
            threshold_changes = {}
            for threshold in THRESHOLDS:
                delta = {s: subtract(rate(data, threshold, s), rate(baseline, threshold, s)) for s in ['all', 'suc', 'fail']}
                threshold_changes[threshold] = {'delta_fixed_denominator': delta,
                    'both_classes_strictly_higher': all(delta[s] is not None and delta[s] > 0 for s in ['suc', 'fail']),
                    'both_classes_no_lower': all(delta[s] is not None and delta[s] >= 0 for s in ['suc', 'fail'])}
            task_changes = {}
            for task, by_task in data.get('by_task', {}).items():
                reference = baseline['by_task'][task]
                task_changes[task] = {'expected': by_task['expected'], 'n': by_task['n'],
                    'baseline_n': reference['n'], 'complete': by_task['n'] == reference['n'] == by_task['expected'],
                    'mae_delta_valid_descriptive': subtract(by_task.get('mae'), reference.get('mae')),
                    'strict_acc_delta_fixed': {s: subtract(rate(by_task, THRESHOLDS[0], s), rate(reference, THRESHOLDS[0], s))
                                               for s in ['all', 'suc', 'fail']}}
            complete_tasks = [v for v in task_changes.values() if v['complete']]
            task_direction = dict(Counter(direction(v['mae_delta_valid_descriptive']) for v in complete_tasks))
            scope, _, k = condition.split(':')
            matched = result.get('matched_controls', {}).get(f'{scope}:{k}', {}).get(population, {})
            controls = matched.get('conditions', {})
            target_mae = controls.get(condition, {}).get('mae')
            contrast = {kind: subtract(target_mae, controls.get('baseline' if kind == 'baseline' else f'{scope}:{kind}:{k}', {}).get('mae'))
                        for kind in ['baseline', 'wrong_region', 'low_rank']}
            output_pair = data.get('pairwise', {})
            base_pair = baseline.get('pairwise', {})
            # Same pair IDs, not simply separately valid pair averages.
            left = {p['fail_id']: p for p in base_pair.get('pairs', [])}
            right = {p['fail_id']: p for p in output_pair.get('pairs', [])}
            common = sorted(set(left) & set(right))
            common_base = {'pairs': [left[e] for e in common]}
            common_target = {'pairs': [right[e] for e in common]}
            pair_change = {'common_pairs': len(common), 'baseline_pairs': base_pair.get('n', 0),
                'target_pairs': output_pair.get('n', 0),
                'pair_weighted_mean_delta_change': float(np.mean([right[e]['continuous_delta'] - left[e]['continuous_delta'] for e in common])) if common else None,
                'equal_video_mean_delta_change': subtract(pair_equal_video(common_target), pair_equal_video(common_base))}
            base_bounds = baseline.get('mae_all_expected_bounds')
            target_bounds = data.get('mae_all_expected_bounds')
            delta_bounds = ([target_bounds[0] - base_bounds[1], target_bounds[1] - base_bounds[0]]
                            if base_bounds is not None and target_bounds is not None else None)
            targets[condition] = {**small(data), 'paired_mae_change': change,
                'mae_delta_all_expected_bounds': delta_bounds,
                'lower_mae_for_all_missing_error_assignments': delta_bounds is not None and delta_bounds[1] < 0,
                'holm_p': holm[population]['adjusted_p'].get(f'{name}/{condition}'),
                'accuracy_changes': threshold_changes,
                'complete_task_mae_directions': task_direction,
                'task_count': len(task_changes), 'complete_task_count': len(complete_tasks), 'tasks': task_changes,
                'matched_controls': {'common_n': matched.get('common_n', 0),
                    'target_minus_control_mae': contrast,
                    'target_lower_than_both_controls': all(contrast[c] is not None and contrast[c] < 0 for c in ['wrong_region', 'low_rank'])},
                'common_pair_change': pair_change}
        output['populations'][population] = {'baseline': small(baseline), 'targets': targets}
    if 'secondary_success_head' in result:
        output['secondary_success_head_full_baseline'] = small(result['secondary_success_head']['baseline']['full'])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-name', default='analysis_v1')
    parser.add_argument('--output-name', default='interpretation_v1.json')
    args = parser.parse_args()
    folder = OUT / args.analysis_name
    index = json.loads((folder / 'index.json').read_text())
    holm = json.loads((folder / 'holm.json').read_text())
    report = {'purpose': 'Post-run descriptive digest; all conditions retained; no extra inferential tests.',
              'analysis_name': args.analysis_name, 'experiments': {}}
    for name in index['experiments']:
        result = json.loads((folder / (name + '.json')).read_text())
        report['experiments'][name] = digest_experiment(result, holm, name)
    aggregate = {}
    for population in ['cohort', 'holdout']:
        rows = [(name, condition, r) for name, e in report['experiments'].items()
                for condition, r in e['populations'][population]['targets'].items()]
        for model in ['meter', 'sole', 'all']:
            selected = [(n, c, r) for n, c, r in rows if model == 'all' or n.startswith(model + '_')]
            complete = [(n, c, r) for n, c, r in selected if r['paired_mae_change'].get('complete')]
            aggregate[f'{population}/{model}'] = {
                'target_conditions': len(selected), 'complete_target_comparisons': len(complete),
                'complete_mae_directions': dict(Counter(direction(r['paired_mae_change'].get('mae_delta')) for _, _, r in complete)),
                'complete_lower_mae_holm_lt_05': [f'{n}/{c}' for n, c, r in complete
                    if r['paired_mae_change']['mae_delta'] < 0 and r['holm_p'] is not None and r['holm_p'] < .05],
                'complete_lower_mae_and_both_classes_strictly_higher': {t: [f'{n}/{c}' for n, c, r in complete
                    if r['paired_mae_change']['mae_delta'] < 0 and r['accuracy_changes'][t]['both_classes_strictly_higher']] for t in THRESHOLDS},
                'descriptive_target_lower_than_both_controls': [f'{n}/{c}' for n, c, r in selected
                    if r['matched_controls']['target_lower_than_both_controls']],
                'lower_mae_for_all_missing_error_assignments': [f'{n}/{c}' for n, c, r in selected
                    if r['lower_mae_for_all_missing_error_assignments']]}
    report['aggregate'] = aggregate
    create_json(OUT / args.output_name, report)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
