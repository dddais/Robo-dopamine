"""Score basic-method predictions, including both Robometer output heads."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import itertools
import json
import math
from numbers import Real
from pathlib import Path

from mydata_bench.basic_method import score as primary_score
from mydata_bench.basic_method.common import (
    OUT, conditions, create_json, create_text, file_hash, fingerprint,
)
from mydata_bench.basic_method.run import cache_rows
from mydata_bench.basic_method.selection import add_selection_arguments, select_configs
from mydata_bench.addbase_eval.score import summary as original_summary, valid_prediction


def endpoint_accuracy(rows, labels, requested, low=.3, high=.7):
    ids = set(requested)
    good = {eid: row for eid, row in rows.items() if eid in ids and valid_prediction(row)}
    result = {}
    for split in ('all', 'suc', 'fail'):
        selected = ids if split == 'all' else {eid for eid in ids if labels[eid]['split'] == split}
        valid = selected & good.keys()
        correct = sum((1 if good[eid]['progress'] <= low else 5 if good[eid]['progress'] >= high else 0)
                      == labels[eid]['reward'] for eid in valid)
        result[split] = {'correct': correct, 'valid': len(valid), 'expected': len(selected),
                         'rate_valid': correct / len(valid) if valid else None,
                         'rate_all_expected': correct / len(selected) if selected else None}
    if all(result[s]['rate_valid'] is not None for s in ('suc', 'fail')):
        result['balanced_accuracy'] = (result['suc']['rate_valid'] + result['fail']['rate_valid']) / 2
    return result


def add_endpoint_accuracy(report, rows, labels, requested):
    report.setdefault('accuracy', {})['0.3/0.7'] = endpoint_accuracy(rows, labels, requested)
    return report


def summary(rows, labels, requested):
    # Extend only reporting. The frozen scorer and its MAE/distribution mapping
    # remain unchanged, including while other inference processes are running.
    return add_endpoint_accuracy(original_summary(rows, labels, requested), rows, labels, requested)


def success_readout(rows):
    """Missing/invalid success values never borrow the progress head prediction."""
    converted = {}
    for eid, row in rows.items():
        value = row.get('success_probability')
        valid = (isinstance(value, Real) and not isinstance(value, bool)
                 and math.isfinite(value) and 0 <= value <= 1)
        converted[eid] = {
            'progress': value if valid else None,
            'status': 'invalid_success_head' if row.get('status') == 'ok' and not valid else row.get('status'),
            'sas_applied': row.get('sas_applied', False),
            'baseline_fallback': row.get('baseline_fallback', False),
            'fallback_reason': row.get('fallback_reason'),
        }
    return converted


def population_report(rows, labels, requested):
    report = summary(rows, labels, requested)
    report['sas_applied'] = sum(rows.get(eid, {}).get('sas_applied', False) for eid in requested)
    report['baseline_fallback'] = sum(rows.get(eid, {}).get('baseline_fallback', False) for eid in requested)
    report['fallback_reasons'] = dict(Counter(rows[eid]['fallback_reason'] for eid in requested
                                             if rows.get(eid, {}).get('baseline_fallback')))
    report['status_counts'] = dict(Counter(rows.get(eid, {}).get('status', 'missing') for eid in requested))
    report['by_task'] = {task: summary(rows, labels, [eid for eid in requested if labels[eid]['subset'] == task])
                         for task in sorted({labels[eid]['subset'] for eid in requested})}
    return report


def score_experiment(cfg, *, allow_incomplete=False):
    result = primary_score.score_experiment(cfg, allow_incomplete=allow_incomplete)
    folder = Path(cfg['output_dir'])
    inputs = json.loads(Path(cfg['inputs']).read_text())
    labels = json.loads(Path(cfg['labels']).read_text())
    run_id = fingerprint(json.loads((folder / 'run_identity.json').read_text()))
    groups = {'full': [s['example_id'] for s in inputs],
              'holdout': [s['example_id'] for s in inputs if s['holdout']]}
    baseline = cache_rows(folder / 'predictions/baseline.jsonl', inputs, run_id, 'baseline')
    converted_baseline = success_readout(baseline)
    reports, loaded, primary_loaded = {}, {}, {}
    for condition in conditions(cfg):
        path = folder / 'predictions' / (condition.replace(':', '_') + '.jsonl')
        raw = cache_rows(path, inputs, run_id, condition)
        primary_rows = {eid: {'status': row.get('status'), 'progress': row.get('progress')}
                        for eid, row in raw.items()}
        primary_loaded[condition] = primary_rows
        primary_record = result['conditions'][condition]
        for name, requested in groups.items():
            add_endpoint_accuracy(primary_record[name], primary_rows, labels, requested)
            for task, task_report in primary_record[name]['by_task'].items():
                add_endpoint_accuracy(task_report, primary_rows, labels,
                                      [eid for eid in requested if labels[eid]['subset'] == task])
        if condition != 'baseline':
            applied = [eid for eid in groups['full'] if raw.get(eid, {}).get('sas_applied')]
            add_endpoint_accuracy(primary_record['applied_only'], primary_rows, labels, applied)
            add_endpoint_accuracy(primary_record['baseline_on_applied_only'], primary_loaded['baseline'], labels, applied)
        if cfg['model'] != 'meter':
            continue
        for eid, row in raw.items():
            if row.get('baseline_fallback') and row.get('success_probability') != baseline[eid].get('success_probability'):
                raise ValueError(f'Success-head fallback differs from its baseline: {condition}, {eid}')
        rows = success_readout(raw)
        record = {name: population_report(rows, labels, ids) for name, ids in groups.items()}
        if condition != 'baseline':
            applied = [eid for eid in groups['full'] if rows.get(eid, {}).get('sas_applied')]
            record['applied_only'] = summary(rows, labels, applied)
            record['baseline_on_applied_only'] = summary(converted_baseline, labels, applied)
        reports[condition] = record
        loaded[condition] = rows
    for matched_record in result['matched_controls'].values():
        for condition, report in matched_record['conditions'].items():
            add_endpoint_accuracy(report, primary_loaded[condition], labels, matched_record['example_ids'])
    if cfg['model'] != 'meter':
        return result
    matched = {}
    for scope in cfg['scopes']:
        for k in cfg['top_k']:
            names = ['baseline', *[f'{scope}:{kind}:{k}' for kind in ['target', *cfg['controls']]]]
            common = [eid for eid in groups['full']
                      if all(valid_prediction(loaded[name].get(eid, {})) for name in names)
                      and all(loaded[name][eid]['sas_applied'] for name in names[1:])]
            matched[f'{scope}:{k}'] = {
                'n': len(common), 'example_ids': common,
                'conditions': {name: summary(loaded[name], labels, common) for name in names}}
    result['secondary_success_head'] = reports
    result['secondary_success_head_matched_controls'] = matched
    result['success_head_definition'] = {
        'source': 'success_probability at the final progress token',
        'mae': 'same 1–5 ordinal mapping as the progress head: thresholds 0.125, 0.375, 0.625, 0.875',
        'endpoint_accuracy_thresholds': [[0.125, 0.875], [0.2, 0.8], [0.3, 0.7]],
        'pairwise_delta': 'success probability of the success instruction minus the paired failure instruction',
        'ranking': 'shared with the progress readout; no separate success-head ranking or inference',
    }
    return result


def flat_rows(name, cfg, result):
    primary_head = 'reward' if cfg['model'] in ('qwen', 'roboreward') else 'progress'
    readouts = [(primary_head, result['conditions'])]
    if cfg['model'] == 'meter':
        readouts.append(('success', result['secondary_success_head']))
    return [{'head': head, **primary_score.table_row(name, condition, record['full']),
             **{column: record['full'].get('accuracy', {}).get('0.3/0.7', {}).get(split, {}).get('rate_all_expected')
                for column, split in [('accuracy_03_07', 'all'), ('suc_accuracy_03_07', 'suc'), ('fail_accuracy_03_07', 'fail')]}}
            for head, records in readouts for condition, record in records.items()]


def markdown_table(rows):
    text = '| 输入 | head | 条件 | 有效/总数 | MAE | 总准确率 | suc | fail | 0.2/0.8准确率 | 0.3/0.7总准确率 | 0.3/0.7 suc | 0.3/0.7 fail | pairwise Δ | SAS | 回退 |\n'
    text += '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        text += (f'| {row["experiment"]} | {row["head"]} | {row["condition"]} | {row["valid"]}/{row["expected"]} | '
                 + ' | '.join(primary_score.format_number(row[key]) for key in (
                     'mae', 'accuracy', 'suc_accuracy', 'fail_accuracy', 'accuracy_02_08',
                     'accuracy_03_07', 'suc_accuracy_03_07', 'fail_accuracy_03_07', 'pairwise_delta'))
                 + f' | {row["sas_applied"]} | {row["baseline_fallback"]} |\n')
    return text


def best_row(rows):
    candidates = [r for r in rows if ':target:' in r['condition']
                  and r['valid'] == r['expected'] and r['mae'] is not None]
    return min(candidates, key=lambda r: (r['mae'], -r['accuracy'], r['experiment'], r['condition'])) if candidates else None


NOTES = ('MAE 沿用原汇总的 1–5 分档口径。主表总／suc／fail 准确率使用端点阈值 0.125/0.875，'
         '另列 0.2/0.8 总准确率及 0.3/0.7 的总／suc／fail 准确率；JSON 包含三套阈值的各分组准确率。'
         '0.3/0.7 按输出 ≤0.3 判失败、≥0.7 判成功，中间区间不计端点正确；'
         '对归一化的 1–5 离散评分，这一新增宽松口径接受失败 1–2 分／成功 4–5 分，原严格指标保留。'
         '准确率分母为全部应评测样本，'
         '无效输出不计正确；MAE 使用有效预测，并提供缺失值上下界。pairwise Δ 是归一化输出之差。'
         '主表包含 ranking 视频组，holdout 排除这些视频及其反事实指令。默认 bias 为 ±6。'
         '最佳 SAS 按各模型、各输出 head 分别选择全量预测有效且 MAE 最低的条件，仅作描述性比较。\n\n')

METER_NOTES = ('success head 读取已保存的最后一个进度 token 的 `success_probability`，与 progress head '
               '使用相同的 1–5 分档及端点阈值；这里的准确率不是 0.5 阈值二分类准确率。'
               '两个 head 来自同一次 forward、同一份 attention ranking 和干预。缺框回退复制对应 baseline '
               '的同一 head。两个 head 单独报告，不在两者之间挑选一个替代另一个。\n\n')


def write_reports(destination, configs, results, heads, selection):
    flat = [row for name, result in results.items() for row in flat_rows(name, result['config'], result)]
    best = []
    for model in sorted({cfg['model'] for cfg in configs}):
        rows = [r for r in flat if results[r['experiment']]['config']['model'] == model]
        text = f'# {model} basic method\n\n' + NOTES + (METER_NOTES if model == 'meter' else '')
        for head in dict.fromkeys(r['head'] for r in rows):
            selected = [r for r in rows if r['head'] == head]
            winner = best_row(selected)
            text += f'## {head} head\n\n### Baseline\n\n'
            text += markdown_table([r for r in selected if r['condition'] == 'baseline'])
            text += '\n### 最佳完整 SAS（按 MAE）\n\n'
            if winner:
                best.append(winner)
                text += markdown_table([winner])
            else:
                text += '尚无预测全部有效的完整 SAS 条件。\n'
            text += '\n### 全部条件\n\n' + markdown_table(selected) + '\n'
        create_text(destination / f'{model}_summary.md', text)
    create_text(destination / 'summary.md', '# Basic method 实验汇总\n\n' + NOTES
                + '\n## Baseline\n\n' + markdown_table([r for r in flat if r['condition'] == 'baseline'])
                + '\n## 各模型／head 最佳完整 SAS\n\n' + markdown_table(best))
    if flat:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
        create_text(destination / 'metrics.csv', buffer.getvalue())
    overlap = {}
    for a, b in itertools.combinations(sorted(heads), 2):
        overlap[f'{a} vs {b}'] = {
            'common_ranking_examples': len(set(heads[a]['example_ids']) & set(heads[b]['example_ids'])),
            'head_index_overlap': {str(k): len({(h['layer'], h['head']) for h in heads[a]['ranking'][:k]}
                                               & {(h['layer'], h['head']) for h in heads[b]['ranking'][:k]})
                                   for k in (8, 32, 64)}}
    create_json(destination / 'head_overlap.json', {'comparisons': overlap,
                'top8': {name: row['ranking'][:8] for name, row in heads.items()},
                'note': 'Index overlap across checkpoints does not establish functional head correspondence; ranking populations may differ.'})
    create_json(destination / 'index.json', {
        'experiments': list(results), 'expected_experiments': len(configs), 'selection': selection,
        'complete': len(results) == len(configs) and all(r['complete'] for r in results.values()),
        'all_predictions_valid': len(results) == len(configs) and all(r['valid'] == r['expected'] for r in flat),
        'reporting_version': 'basic_method_both_heads_v2',
        'endpoint_accuracy_thresholds': [[0.125, 0.875], [0.2, 0.8], [0.3, 0.7]],
        'reporting_sources': {str(p): file_hash(p) for p in (
            Path(__file__), Path(primary_score.__file__), Path(primary_score.summary.__code__.co_filename))},
        'completion_definition': 'complete counts saved rows; all_predictions_valid additionally checks readout validity',
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--output-name', default='analysis_both_heads_v1')
    parser.add_argument('--allow-incomplete', action='store_true')
    add_selection_arguments(parser)
    args = parser.parse_args()
    destination = args.root / args.output_name
    if destination.exists():
        parser.error('Analysis is append-only; choose a new --output-name')
    try:
        selected = select_configs(args.root, args.models, args.exclude_configs)
    except ValueError as exc:
        parser.error(str(exc))
    results, heads = {}, {}
    for _, cfg in selected:
        folder = Path(cfg['output_dir'])
        if not (folder / 'run_identity.json').exists() and args.allow_incomplete:
            continue
        result = score_experiment(cfg, allow_incomplete=args.allow_incomplete)
        results[folder.name] = result
        create_json(destination / f'{folder.name}.json', result)
        for scope in cfg['scopes']:
            path = folder / 'independent_ranking' / f'ranking_{scope}.json'
            if path.exists():
                ranking = json.loads(path.read_text())
                run_id = fingerprint(json.loads((folder / 'run_identity.json').read_text()))
                if ranking.get('run_id') != run_id:
                    raise ValueError('Ranking/report identity mismatch')
                heads[f'{folder.name}/{scope}'] = ranking
        print(f'Scored {folder.name}' + (' (progress + success)' if cfg['model'] == 'meter' else ''), flush=True)
    selection = {'models': args.models, 'exclude_configs': args.exclude_configs,
                 'configs': [Path(p).stem for p, _ in selected]}
    write_reports(destination, [cfg for _, cfg in selected], results, heads, selection)
    print(destination)


if __name__ == '__main__':
    main()
