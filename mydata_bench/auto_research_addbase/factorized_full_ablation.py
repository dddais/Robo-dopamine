"""Complete-population ablations of the three actually recorded native branches.

No new inference, selection, alpha search, or class-specific transformation.
The recorded factorized prediction is checked and used unchanged as the result.
"""
import argparse
from collections import Counter
import hashlib
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.score import summary
from .branch_ablation import native_branch
from .factorized_ablation import derive, pair_gain_ci, relative_roi
from .prepare import OUT
from .select_factorized import verified_rows
from .statistics import paired_statistics


def roi_strata(arms, scores, labels, scope):
    pairs = scores['baseline']['pairwise']['pairs']
    rows = arms['combined']
    result = []
    for identical in [True, False]:
        selected = [p for p in pairs if
                    (relative_roi(rows[p['suc_id']], scope) ==
                     relative_roi(rows[p['fail_id']], scope)) is identical]
        margins = {name: [value[p['suc_id']]['reward'] - value[p['fail_id']]['reward']
                          for p in selected] for name, value in arms.items()}
        entry = dict(identical_actual_relative_roi=identical, n_pairs=len(selected), arms={})
        for name, values in margins.items():
            hist = Counter('<0' if v < 0 else str(v) for v in values)
            counts = {key: hist[key] for key in ['<0', '0', '1', '2', '3', '4']}
            if sum(counts.values()) != len(selected):
                raise ValueError('Pair histogram count differs')
            entry['arms'][name] = dict(counts=counts,
                mean_reward_margin=float(np.mean(values)) if values else None)
        entry['combined_minus_visual_only_margin'] = pair_gain_ci(selected,
            [a-b for a, b in zip(margins['combined'], margins['visual_only'])], labels)
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    parser.add_argument('--protocol', choices=['image_text', 'text_image', 'interleaved', 'text_video', 'video_text'], required=True)
    parser.add_argument('--conditions', nargs='+', required=True)
    args = parser.parse_args()
    folder = OUT / 'experiments' / f'{args.model}_{args.protocol}_uniform_factorized_evidence_a1' / 'full_cohort'
    splits = json.loads((OUT / 'splits.json').read_text())
    ids = splits['full_cohort']
    sources = {}
    for name in ['runtime_config.json', 'requested_ids.json']:
        path = folder / name
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = json.loads((folder / 'runtime_config.json').read_text())
    expected = dict(contrast_weight=1, negative_strength=4, negative_task_strength=4,
                    contrast_negative_mode='visual_and_task', task_binding_fraction=.5,
                    task_binding_distribution='uniform')
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('Unexpected three-branch method')
    if len(ids) != 846 or set(json.loads((folder / 'requested_ids.json').read_text())) != set(ids):
        raise ValueError('Require complete full-cohort population')
    # Verify all requested rows and the actual formula before scoring labels.
    baseline = verified_rows(folder / 'predictions/baseline.jsonl', ids, sources, True)
    conditions = {condition: verified_rows(folder / 'binding_transport_s4/predictions' /
        f'{condition}.jsonl', ids, sources) for condition in args.conditions}
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {eid: all_labels[eid] for eid in ids}
    del all_labels
    results = {}
    for condition, rows in conditions.items():
        scope, _ = condition.split('_target_')
        arms = dict(baseline=baseline, combined=rows)
        for name in ['positive_only', 'visual_only', 'task_only']:
            arms[name] = {eid: derive(row, name) for eid, row in rows.items()}
        for name, branch in [('visual_negative', 'negative'), ('task_negative', 'negative_task')]:
            arms[name] = {eid: native_branch(row, branch, args.model) for eid, row in rows.items()}
        for population in ['validation', 'full_cohort', 'old_holdout']:
            requested = splits[population]
            scores = {name: summary(value, labels, requested) for name, value in arms.items()}
            comparisons = {f'combined_minus_{name}': paired_statistics(arms[name], rows, labels, requested)
                           for name in ['baseline', 'positive_only', 'visual_only', 'task_only']}
            results[f'{condition}/{population}'] = dict(metrics=scores, comparisons=comparisons,
                roi_strata=roi_strata(arms, scores, labels, scope))
    output = OUT / 'analysis' / time.strftime(f'factorized_full_ablation_{args.model}_{args.protocol}_%Y%m%d_%H%M%S.json')
    create_json(output, dict(arguments=vars(args), sources_sha256=sources, results=results,
        actual_forward_branches=3, derived_arms=['positive_only', 'visual_only', 'task_only'],
        actual_native_negative_arms=['visual_negative', 'task_negative'], selection_modified=False,
        interpretation='Same three stored forward passes; combined predictions are used unchanged after exact formula verification. '
            'Visual-only and task-only are algebraic ablations, not separately run candidates. '
            'Both threshold summaries are retained; discrete native classes make the thresholds equivalent. '
            'Video-cluster CIs are descriptive and unadjusted after adaptive exploration. ROI strata are observational.'))
    print(output)
    for key, value in results.items():
        if not key.endswith('/validation'):
            continue
        print(key)
        for name, score in value['metrics'].items():
            print(name, 'MAE', round(score['mae'], 3), 'accuracy',
                  [round(100*score['accuracy']['0.125/0.875'][c]['rate_all_expected'], 2)
                   for c in ['all', 'suc', 'fail']])


if __name__ == '__main__':
    main()
