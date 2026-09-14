"""Export a chosen endpoint threshold from an existing, immutable analysis."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from mydata_bench.basic_method.common import create_json, create_text, file_hash


def table(rows, threshold):
    text = (f'| 输入 | head | 条件 | 有效/总数 | MAE | 总准确率 ({threshold}) | suc | fail | pairwise Δ | SAS | 回退 |\n'
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for row in rows:
        values = ['—' if row[k] is None else f'{row[k]:.4f}'
                  for k in ('mae', 'accuracy', 'suc_accuracy', 'fail_accuracy', 'pairwise_delta')]
        text += (f'| {row["experiment"]} | {row["head"]} | {row["condition"]} | {row["valid"]}/{row["expected"]} | '
                 + ' | '.join(values) + f' | {row["sas_applied"]} | {row["baseline_fallback"]} |\n')
    return text


def export_view(source, destination, threshold):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if threshold not in ('0.125/0.875', '0.2/0.8', '0.3/0.7'):
        raise ValueError('Unsupported endpoint threshold')
    if destination.exists():
        raise FileExistsError('Choose a new output directory; existing reports are immutable')
    index = json.loads((source / 'index.json').read_text())
    results, rows, sources = {}, [], {str(source / 'index.json'): file_hash(source / 'index.json')}
    for name in index['experiments']:
        path = source / f'{name}.json'
        result = json.loads(path.read_text())
        results[name] = result
        sources[str(path)] = file_hash(path)
        model = result['config']['model']
        readouts = [('reward' if model in ('qwen', 'roboreward') else 'progress', result['conditions'])]
        if model == 'meter':
            if 'secondary_success_head' not in result:
                raise ValueError('Source analysis lacks the Robometer success head; run basic_method.reporting first')
            readouts.append(('success', result['secondary_success_head']))
        for head, records in readouts:
            for condition, record in records.items():
                full = record['full']
                accuracy = full.get('accuracy', {}).get(threshold)
                if accuracy is None and full['n']:
                    raise ValueError(f'Source lacks {threshold}: {name}/{head}/{condition}')
                accuracy = accuracy or {}
                rows.append({'experiment': name, 'model': model, 'head': head, 'condition': condition,
                             'endpoint_threshold': threshold, 'expected': full['expected'], 'valid': full['n'],
                             'mae': full.get('mae'),
                             **{field: accuracy.get(split, {}).get('rate_all_expected') for field, split in (
                                 ('accuracy', 'all'), ('suc_accuracy', 'suc'), ('fail_accuracy', 'fail'))},
                             'pairwise_delta': full.get('pairwise', {}).get('continuous_mean_delta'),
                             'sas_applied': full['sas_applied'], 'baseline_fallback': full['baseline_fallback'],
                             'invalid': full['invalid']})
    low, high = threshold.split('/')
    notes = (f'本版所有表格以 **{threshold}** 为端点准确率口径：输出 ≤{low} 判失败，≥{high} 判成功，'
             '中间区间不计端点正确。总／suc／fail 准确率分母均为对应全部应评测样本，无效输出不计正确。'
             'MAE 继续使用原 1–5 分档，pairwise Δ 和所有原始预测不变。'
             '最佳 SAS 按每个模型、每个输出 head 分别取预测全部有效且 MAE 最低的条件；'
             'MAE 并列时按本版准确率选择，仅作描述性比较。\n\n'
             'Robometer 的 progress 和 success 分别报告；两者来自同一次 forward 和同一份 ranking／干预。'
             '详细 JSON 保留源报告的全部阈值、holdout、task、分布和对照统计。\n\n')
    if threshold == '0.3/0.7':
        notes += '对归一化的离散评分，此宽松口径接受失败 1–2 分、成功 4–5 分。\n\n'
    best = []
    for model in sorted({r['model'] for r in rows}):
        text = f'# {model}：端点阈值 {threshold}\n\n' + notes
        for head in dict.fromkeys(r['head'] for r in rows if r['model'] == model):
            selected = [r for r in rows if r['model'] == model and r['head'] == head]
            candidates = [r for r in selected if ':target:' in r['condition']
                          and r['valid'] == r['expected'] and r['mae'] is not None]
            winner = min(candidates, key=lambda r: (r['mae'], -r['accuracy'], r['experiment'], r['condition'])) if candidates else None
            text += f'## {head} head\n\n### Baseline\n\n' + table([r for r in selected if r['condition'] == 'baseline'], threshold)
            text += '\n### 最佳完整 SAS（按 MAE）\n\n'
            if winner:
                best.append(winner)
                text += table([winner], threshold)
            else:
                text += '尚无预测全部有效的完整 SAS 条件。\n'
            text += '\n### 全部条件\n\n' + table(selected, threshold) + '\n'
        create_text(destination / f'{model}_summary.md', text)
    create_text(destination / 'summary.md', f'# 端点阈值 {threshold} 实验汇总\n\n' + notes
                + '## Baseline\n\n' + table([r for r in rows if r['condition'] == 'baseline'], threshold)
                + '\n## 各模型／head 最佳完整 SAS\n\n' + table(best, threshold))
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        create_text(destination / 'metrics.csv', buffer.getvalue())
    for name, result in results.items():
        create_json(destination / f'{name}.json', result)
    overlap = source / 'head_overlap.json'
    if overlap.exists():
        sources[str(overlap)] = file_hash(overlap)
        create_json(destination / overlap.name, json.loads(overlap.read_text()))
    create_json(destination / 'index.json', {
        **index, 'display_endpoint_threshold': threshold, 'source_analysis': str(source),
        'source_analysis_sha256': sources, 'view_version': 'endpoint_view_v1',
        'view_source_sha256': file_hash(__file__), 'metric_rows': len(rows),
    })
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', choices=['0.125/0.875', '0.2/0.8', '0.3/0.7'], default='0.2/0.8')
    args = parser.parse_args()
    print(export_view(args.source, args.output, args.threshold))


if __name__ == '__main__':
    main()
