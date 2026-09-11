"""Check the completed reporting artifacts against their frozen denominators.

This is a numerical consistency audit of exports, not a model inference rerun.
The report candidate has links relative to its intended final destination.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from .prepare import OUT, ROOT, create_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='report_candidate_v1.md')
    parser.add_argument('--output-name', default='artifact_validation_v1.json')
    args = parser.parse_args()
    audit = json.loads((OUT / 'completion_audit_v1.json').read_text())
    state = json.loads((OUT / 'research/scheduler_completion.json').read_text())
    assert audit['complete'] and not audit['issues']
    assert all(v['complete'] for v in state['finished'].values())
    assert audit['evidence_checks']['predictions_observed'] == 197292
    folder = OUT / 'analysis_v1'
    index = json.loads((folder / 'index.json').read_text())
    assert len(index['experiments']) == 12 and index['rows'] == 468
    expected = {'full': 1213, 'cohort': 846, 'holdout': 730}
    checks = Counter()
    metrics = {}
    populations = {}
    for name in index['experiments']:
        data = json.loads((folder / (name + '.json')).read_text())
        assert len(data['conditions']) == 19
        assert (OUT / name / 'exp_record.md').stat().st_size > 0
        for condition, records in data['conditions'].items():
            assert set(records) == ({'full', 'cohort', 'holdout'} if condition == 'baseline' else {'cohort', 'holdout'})
            for population, d in records.items():
                populations[(name, condition, population)] = d
                assert d['expected'] == expected[population]
                assert d['n'] + d['invalid'] == d['expected']
                assert not d['missing_ids']
                assert sum(t['expected'] for t in d['by_task'].values()) == d['expected']
                assert sum(t['n'] for t in d['by_task'].values()) == d['n']
                if d['n']:
                    task_error_sum = sum(t['mae'] * t['n'] for t in d['by_task'].values() if t['n'])
                    assert math.isclose(task_error_sum / d['n'], d['mae'], abs_tol=1e-12)
                for task, t in d['by_task'].items():
                    metrics[(name, condition, population, task)] = t
                if not d['n']:
                    continue
                for threshold in ['0.125/0.875', '0.2/0.8']:
                    for split in ['all', 'suc', 'fail']:
                        a = d['accuracy'][threshold][split]
                        correct = sum(t.get('accuracy', {}).get(threshold, {}).get(split, {}).get('correct', 0)
                                      for t in d['by_task'].values())
                        assert correct == a['correct']
                        assert 0 <= a['correct'] <= a['valid'] <= a['expected']
                        assert math.isclose(a['rate_all_expected'], a['correct'] / a['expected'], abs_tol=1e-12)
                        checks['population_accuracy_checks'] += 1
                for field in ['prediction_distributions', 'ordinal_prediction_distributions']:
                    for split in ['all', 'suc', 'fail']:
                        dist = d[field][split]
                        assert sum(dist['counts'].values()) == dist['n']
                        for key, count in dist['counts'].items():
                            assert sum(t.get(field, {}).get(split, {}).get('counts', {}).get(key, 0)
                                       for t in d['by_task'].values()) == count
                        checks['population_distribution_checks'] += 1
                pairs = d['pairwise']
                assert pairs['n'] == len(pairs['pairs'])
                assert sum(pairs['ordinal_difference_counts'].values()) == pairs['n']
                assert sum(pairs['continuous_difference_counts'].values()) == pairs['n']
                checks['population_summary_checks'] += 1
    assert len(metrics) == 13176
    with (folder / 'task_metrics_all_populations.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 13176
    for row in rows:
        key = tuple(row[k] for k in ['experiment', 'condition', 'population', 'task'])
        d = metrics[key]
        assert [int(row[k]) for k in ['expected', 'valid', 'invalid']] == [d['expected'], d['n'], d['invalid']]
        for threshold in ['0.125/0.875', '0.2/0.8']:
            for split in ['all', 'suc', 'fail']:
                correct = d.get('accuracy', {}).get(threshold, {}).get(split, {}).get('correct', 0)
                assert int(row[f'correct_{threshold}_{split}']) == correct
    with (folder / 'task_distributions_all_populations.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 79056
    for row in rows:
        key = tuple(row[k] for k in ['experiment', 'condition', 'population', 'task'])
        field = {'ordinal_reward': 'ordinal_prediction_distributions', 'equal_width_progress': 'prediction_distributions'}[row['binning']]
        d = metrics[key].get(field, {}).get(row['split'], {})
        assert int(row['valid']) == d.get('n', 0)
        for k in range(1, 6):
            assert int(row[f'prediction_{k}']) == d.get('counts', {}).get(str(k), 0)
        assert int(row['valid']) + int(row['invalid']) == int(row['expected'])
    with (folder / 'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 468
    for row in rows:
        d = populations[tuple(row[k] for k in ['experiment', 'condition', 'population'])]
        for threshold in ['0.125/0.875', '0.2/0.8']:
            for split in ['all', 'suc', 'fail']:
                a = d.get('accuracy', {}).get(threshold, {}).get(split, {})
                for prefix, field in [('acc', 'rate_valid'), ('acc_valid', 'rate_valid'),
                                      ('acc_fixed', 'rate_all_expected')]:
                    value = a.get(field)
                    cell = row[f'{prefix}_{threshold}_{split}']
                    assert (cell == '') if value is None else math.isclose(float(cell), value, abs_tol=1e-12)
                checks['flat_accuracy_column_checks'] += 1
    overlap = json.loads((OUT / 'head_overlap_v1/overlap.json').read_text())
    rankings = json.loads((OUT / 'head_overlap_v1/rankings.json').read_text())
    assert len(overlap) == 2988 and len(rankings) == 54
    for row in overlap:
        k = row['k']
        a = set(map(tuple, rankings[row['first']]['pairs'][:k]))
        b = set(map(tuple, rankings[row['second']]['pairs'][:k]))
        assert len(a & b) == row['overlap_count']
    for filename in ['full_tables.md', 'paired_mae_changes.png', 'paired_mae_changes.svg',
                     'official_endpoint_tradeoffs_v2.png', 'official_endpoint_tradeoffs_v2.svg']:
        assert (folder / filename).stat().st_size > 0
    candidate = OUT / 'research' / args.report
    text = candidate.read_text()
    links = re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', text)
    for target in links:
        if target.startswith(('https://', 'http://', '#')):
            continue
        path = (ROOT / 'mydata_bench' / target.split('#')[0]).resolve()
        assert path.exists(), (target, path)
    result = {'status': 'passed', 'scope': 'Artifact consistency; no repeated model inference',
              'checks': dict(checks), 'task_metric_rows': 13176, 'task_distribution_rows': 79056,
              'overlap_rows': 2988, 'ranking_sets': 54, 'report_links_checked': len(links),
              'report_candidate': str(candidate), 'report_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()}
    create_json(OUT / 'research' / args.output_name, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
