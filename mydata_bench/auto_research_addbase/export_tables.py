"""Export reviewable tables and figures from an immutable scoring checkpoint."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

from .prepare import OUT


def write_table(path, rows):
    if not rows:
        return 0
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize_record(meta, record, metrics, distributions, pairwise):
    row = dict(meta)
    for name in ['expected', 'n', 'invalid', 'coverage', 'mae',
                 'continuous_ordinal_mae', 'mean_progress', 'ordinal_exact_accuracy']:
        row[name] = record.get(name)
    for threshold, values in record.get('accuracy', {}).items():
        for split in ['all', 'suc', 'fail']:
            for field in ['correct', 'valid', 'expected', 'rate_all_expected', 'rate_valid']:
                row[f'{threshold}_{split}_{field}'] = values[split][field]
    metrics.append(row)
    for definition in ['prediction_distributions', 'ordinal_prediction_distributions']:
        for split, values in record.get(definition, {}).items():
            for reward in range(1, 6):
                distributions.append(dict(meta, bin_definition=definition, split=split,
                    reward=reward, count=values['counts'][str(reward)],
                    rate=values['rates'][str(reward)], valid=values['n'],
                    expected=record.get('accuracy', {}).get('0.125/0.875', {}).get(split, {}).get('expected')))
    pairs = record.get('pairwise', {})
    for definition in ['ordinal_difference', 'continuous_difference']:
        for key, count in pairs.get(definition + '_counts', {}).items():
            pairwise.append(dict(meta, bin_definition=definition, difference_bin=key,
                count=count, rate=pairs[definition + '_rates'][key], n=pairs['n'],
                unique_suc_videos=pairs['unique_suc_videos'],
                continuous_mean_delta=pairs['continuous_mean_delta']))


def rank_tables(destination):
    ranking_sets = {}
    top8 = []
    for path in sorted((OUT / 'experiments').glob('*/ranking/ranking_*.json')):
        artifact = json.loads(path.read_text())
        if 'ranking' not in artifact:
            continue
        experiment, scope = path.parents[1].name, artifact['scope']
        ranking_sets[(experiment, scope)] = artifact['ranking']
        for number, head in enumerate(artifact['ranking'][:8], 1):
            top8.append(dict(experiment=experiment, scope=scope, rank=number,
                layer_zero_based=head['layer'], head_zero_based=head['head'], score=head['score'],
                query_kind=artifact['query_kind'], ranking_examples=artifact['n'],
                ranking_score=artifact['ranking_score'], task_score=head.get('task_score'),
                selection_source=head.get('selection_source')))
    overlaps = []
    for ((left, scope), a), ((right, other_scope), b) in itertools.combinations(ranking_sets.items(), 2):
        if scope != other_scope:
            continue
        for k in [8, 32, 64]:
            x = {(v['layer'], v['head']) for v in a[:k]}
            y = {(v['layer'], v['head']) for v in b[:k]}
            shared = x & y
            overlaps.append(dict(left=left, right=right, scope=scope, k=k,
                intersection=len(shared), overlap_fraction=len(shared)/k,
                jaccard=len(shared)/len(x | y),
                same_model=left.split('_')[0] == right.split('_')[0],
                common_indices=json.dumps(sorted(shared))))
    return {'top8_heads.csv': write_table(destination/'top8_heads.csv', top8),
            'head_overlaps.csv': write_table(destination/'head_overlaps.csv', overlaps)}


def frontier(points, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [('meter', 'progress'), ('meter', 'success_probability'),
              ('roboreward', 'progress'), ('qwen', 'progress'), ('sole', 'progress')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    handles = {}
    for ax, (model, field) in zip(axes.flat, groups):
        subset = [p for p in points if p['experiment'].split('_')[0] == model
                  and p['field'] == field and p['threshold'] == '0.125/0.875'
                  and p['valid'] == p['baseline_valid'] == p['expected']]
        families = sorted({p['method'].rsplit('_s', 1)[0] for p in subset})
        for family in families:
            items = [p for p in subset if p['method'].rsplit('_s', 1)[0] == family]
            artist = ax.scatter([100*p['delta_fail'] for p in items],
                                [100*p['delta_suc'] for p in items], s=18, alpha=.5, label=family)
            handles[family] = artist
        passed = [p for p in subset if p['meets_descriptive_gate']]
        if passed:
            ax.scatter([100*p['delta_fail'] for p in passed], [100*p['delta_suc'] for p in passed],
                       s=48, facecolors='none', edgecolors='black', linewidths=.8)
        ax.axvline(0, color='gray', linewidth=.8)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_title(f'{model}: {field}\n{len(subset)} complete points; {len(passed)} descriptive passes')
        ax.set_xlabel('Failure accuracy change (percentage points)')
        ax.set_ylabel('Success accuracy change (percentage points)')
        ax.grid(alpha=.15)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(handles.values(), handles.keys(), loc='center')
    axes.flat[-1].text(.05, .15, 'Each point: one protocol / method / dose / k.\n'
                      'Black circles meet all four descriptive gates.\n'
                      'Points are exploratory comparisons, not independent trials.\n'
                      'Only the 0.125 / 0.875 threshold is shown.', fontsize=9,
                      transform=axes.flat[-1].transAxes)
    population = points[0]['population'] if points else 'empty'
    fig.suptitle(f'Class tradeoffs on {population}; paired baseline within each protocol and readout')
    fig.savefig(destination/'class_tradeoffs.png', dpi=180)
    fig.savefig(destination/'class_tradeoffs.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--name')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_relative_to(OUT.resolve()):
        raise ValueError('Only this research session can be exported')
    destination = OUT/'analysis'/(args.name or time.strftime('exports_%Y%m%d_%H%M%S'))
    destination.mkdir(exist_ok=False)
    details = json.loads((checkpoint/'details.json').read_text())
    points = json.loads((checkpoint/'points.json').read_text())
    population = points[0]['population'] if points else 'empty'
    metrics, distributions, pairs = [], [], []
    seen_baseline = set()
    for key, record in details.items():
        experiment, method, condition, field = key.split('/')
        for branch, data_name in [('baseline', 'baseline'), ('intervention', 'intervention')]:
            baseline_id = (experiment, field)
            if branch == 'baseline' and baseline_id in seen_baseline:
                continue
            if branch == 'baseline':
                seen_baseline.add(baseline_id)
            meta = dict(experiment=experiment, field=field, population=population,
                        method='baseline' if branch == 'baseline' else method,
                        condition='baseline' if branch == 'baseline' else condition)
            summarize_record(dict(meta, task='ALL'), record[data_name], metrics, distributions, pairs)
            for task, values in record['baseline_by_task' if branch == 'baseline' else 'by_task'].items():
                summarize_record(dict(meta, task=task), values, metrics, distributions, pairs)
    counts = {name: write_table(destination/name, rows) for name, rows in [
        ('metrics_and_tasks.csv', metrics), ('prediction_distributions.csv', distributions),
        ('paired_difference_distributions.csv', pairs)]}
    counts.update(rank_tables(destination))
    frontier(points, destination)
    manifest = dict(source_checkpoint=str(checkpoint), population=population, rows=counts,
        sources={name: hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
                 for name in ['points.json', 'details.json']},
        interpretation='Repeated exploratory comparisons. Cross-model head overlap compares indices only, not functional equivalence. Invalid outputs never become reward 1.')
    with (destination/'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
    with (destination/'README.md').open('x') as handle:
        handle.write('# 实验检查点导出\n\n'
            f'来源：`{checkpoint.name}`；评分集合：`{population}`。\n\n'
            '`metrics_and_tasks.csv` 包含总体及每个任务的 MAE、两套阈值准确率、计数与覆盖率。'
            '`prediction_distributions.csv` 同时保留等宽五档和 Roboreward 序数五档。'
            '`paired_difference_distributions.csv` 列出同视频不同指令的负、零、1–4 差值分档，以及连续差值分档。\n\n'
            '`top8_heads.csv` 使用从 0 开始的 layer/head 索引；`head_overlaps.csv` 给出 top 8/32/64 重合。'
            '跨模型索引重合不等于功能相同，尤其不能把不同规模模型的同序号 head 视作同一权重。\n\n'
            '图中圆圈只表示达到描述性四指标门槛。探索集合的通过点不能视为验证成功；'
            '多协议、剂量、k 和读出形成相关比较，不能把点数称作独立实验数。'
            '所有比例都可结合 CSV 中的有效数与期望分母复核。\n')
    print(destination)
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
