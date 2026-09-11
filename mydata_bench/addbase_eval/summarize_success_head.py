"""Export descriptive success-head and task metrics from saved final predictions.

No inference or threshold tuning. Existing primary analyses remain unchanged.
Run from the repository root: python -m mydata_bench.addbase_eval.summarize_success_head
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PROTOCOLS = ('video_text', 'text_video', 'image_text', 'text_image', 'interleaved', 'official')
HEADS = {'progress': 'progress', 'success': 'success_probability'}
ORDINAL = (.125, .375, .625, .875)
THRESHOLDS = {'strict': (.125, .875), 'wide': (.2, .8)}


def summarize(rows, labels, requested, field):
    valid = {eid: rows[eid][field] for eid in requested if eid in rows
             and rows[eid]['status'] == 'ok' and rows[eid].get(field) is not None}
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in valid.values())
    ordinal = lambda p: 1 + sum(p >= t for t in ORDINAL)
    result = {'expected': len(requested), 'n': len(valid)}
    result['mae'] = (sum(abs(ordinal(p) - labels[eid]['reward']) for eid, p in valid.items())
                     / len(valid)) if valid else None
    result['accuracy'] = {}
    for rule in ('strict', 'wide', 'binary_0.5'):
        acc = {}
        for split in ('all', 'suc', 'fail'):
            ids = [eid for eid in requested if split == 'all' or labels[eid]['split'] == split]
            good = [eid for eid in ids if eid in valid]
            if rule == 'binary_0.5':
                correct = sum((valid[eid] > .5) == (labels[eid]['split'] == 'suc') for eid in good)
            else:
                low, high = THRESHOLDS[rule]
                correct = sum(valid[eid] >= high if labels[eid]['split'] == 'suc'
                              else valid[eid] <= low for eid in good)
            acc[split] = {'expected': len(ids), 'n': len(good), 'correct': correct,
                          'rate': correct / len(ids) if ids else None}
        acc['balanced_accuracy'] = ((acc['suc']['rate'] + acc['fail']['rate']) / 2
                                    if acc['suc']['rate'] is not None and acc['fail']['rate'] is not None
                                    else None)
        result['accuracy'][rule] = acc
    pairs = []
    for eid, p in valid.items():
        label = labels[eid]
        if label['split'] != 'fail' or label['source_suc_id'] not in valid:
            continue
        sid = label['source_suc_id']
        assert label['video_sha256'] == labels[sid]['video_sha256']
        pairs.append((sid, valid[sid] - p))
    result['pairwise'] = {
        'n': len(pairs), 'unique_suc_videos': len({sid for sid, _ in pairs}),
        'mean_difference': sum(d for _, d in pairs) / len(pairs) if pairs else None,
        'negative_rate': sum(d < -1e-9 for _, d in pairs) / len(pairs) if pairs else None,
    }
    return result


def check_existing(value, existing):
    assert value['expected'] == existing['expected'] and value['n'] == existing['n']
    assert math.isclose(value['mae'], existing['mae'], abs_tol=1e-12)
    for rule, key in [('strict', '0.125/0.875'), ('wide', '0.2/0.8')]:
        for split in ('all', 'suc', 'fail'):
            a, b = value['accuracy'][rule][split], existing['accuracy'][key][split]
            assert a['correct'] == b['correct'] and a['expected'] == b['expected']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/mydata_bench/experiments_v2_addbase'))
    parser.add_argument('--output-name', default='success_head_v1')
    args = parser.parse_args()
    root, output = args.root, args.root / args.output_name
    output.mkdir(exist_ok=False)
    labels = json.loads((root / 'labels_for_scoring_only.json').read_text())
    inputs = json.loads((root / 'inputs.json').read_text())
    populations = {pop: [r['example_id'] for r in inputs if pop == 'full' or
                        (r['cohort'] and (pop == 'cohort' or r['holdout']))]
                   for pop in ('full', 'cohort', 'holdout')}
    task_groups = {pop: {task: [eid for eid in ids if labels[eid]['subset'] == task]
                        for task in sorted({labels[eid]['subset'] for eid in ids})}
                   for pop, ids in populations.items()}
    inventory, csv_rows, matched_checks = {}, [], 0
    for protocol in PROTOCOLS:
        config = 'meter_' + protocol
        original_path = root / 'analysis_v1' / (config + '.json')
        original = json.loads(original_path.read_text())
        inventory[str(original_path)] = hashlib.sha256(original_path.read_bytes()).hexdigest()
        loaded, result = {}, {'config': config, 'conditions': {}, 'matched_controls': {}}
        for condition in original['conditions']:
            path = root / config / 'predictions' / (condition.replace(':', '_') + '.jsonl')
            digest, rows = hashlib.sha256(), {}
            with path.open('rb') as stream:
                for line in stream:
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    assert row['condition'] == condition
                    rows[row['example_id']] = {k: row.get(k) for k in ('status', *HEADS.values())}
            inventory[str(path)] = digest.hexdigest()
            loaded[condition] = rows
            record = {}
            for pop, ids in populations.items():
                if pop == 'full' and condition != 'baseline':
                    continue
                assert set(rows) == set(populations['full'] if condition == 'baseline' else populations['cohort'])
                record[pop] = {}
                for head, field in HEADS.items():
                    value = summarize(rows, labels, ids, field)
                    source = original['conditions'] if head == 'progress' else original['secondary_success_head']
                    check_existing(value, source[condition][pop])
                    matched_checks += 1
                    value['by_task'] = {task: summarize(rows, labels, tids, field)
                                        for task, tids in task_groups[pop].items()}
                    assert sum(v['n'] for v in value['by_task'].values()) == value['n']
                    for rule in ('strict', 'wide', 'binary_0.5'):
                        assert sum(v['accuracy'][rule]['all']['correct'] for v in value['by_task'].values()) == value['accuracy'][rule]['all']['correct']
                    record[pop][head] = value
                    for task, entry in [('ALL', value), *value['by_task'].items()]:
                        flat = {'config': config, 'condition': condition, 'population': pop, 'head': head,
                                'task': task, 'expected': entry['expected'], 'valid': entry['n'], 'ordinal_mae': entry['mae']}
                        for rule, acc in entry['accuracy'].items():
                            for split in ('all', 'suc', 'fail'):
                                for metric in ('correct', 'expected', 'rate'):
                                    flat[f'{rule}_{split}_{metric}'] = acc[split][metric]
                            flat[f'{rule}_balanced_accuracy'] = acc['balanced_accuracy']
                        csv_rows.append(flat)
            result['conditions'][condition] = record
        for scope in ('last_frame', 'all_frames'):
            for k in (8, 32, 64):
                conditions = ['baseline'] + [f'{scope}:{kind}:{k}' for kind in ('target', 'wrong_region', 'low_rank')]
                key = f'{scope}:{k}'
                result['matched_controls'][key] = {}
                for pop in ('cohort', 'holdout'):
                    common = [eid for eid in populations[pop] if all(loaded[c][eid]['status'] == 'ok'
                              and loaded[c][eid]['success_probability'] is not None for c in conditions)]
                    result['matched_controls'][key][pop] = {'n': len(common), 'conditions': {
                        c: summarize(loaded[c], labels, common, 'success_probability') for c in conditions}}
        (output / (config + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(config, 'complete', flush=True)
    for path in (root / 'inputs.json', root / 'labels_for_scoring_only.json', Path(__file__)):
        inventory[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (output / 'metrics_by_task.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest = {
        'purpose': 'Descriptive success-head supplement; no new inference or inferential significance tests',
        'verification_status': 'ANALYZED',
        'ordinal_thresholds': ORDINAL, 'endpoint_thresholds': THRESHOLDS,
        'binary_rule': 'score > 0.5 predicts suc; score <= 0.5 predicts fail; same rule for both heads',
        'accuracy_denominator': 'fixed expected; invalid outputs count as incorrect',
        'mae_denominator': 'valid outputs only',
        'record_selection': 'latest record by example_id within each final condition file; pilots excluded',
        'existing_population_head_summaries_matched': matched_checks,
        'csv_rows': len(csv_rows), 'source_sha256': inventory,
    }
    (output / 'index.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
