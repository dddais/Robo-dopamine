"""Score the full population, with fallback provenance and video-group holdout."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import itertools
import json
from pathlib import Path

from mydata_bench.addbase_eval.score import summary, valid_prediction
from .common import OUT, create_json, create_text, fingerprint, conditions, validate_inputs
from .run import cache_rows
from .selection import add_selection_arguments, select_configs


def score_experiment(cfg, *, allow_incomplete=False):
    validate_inputs(cfg)
    folder = Path(cfg['output_dir'])
    inputs = json.loads(Path(cfg['inputs']).read_text())
    labels = json.loads(Path(cfg['labels']).read_text())
    identity = json.loads((folder / 'run_identity.json').read_text())
    if identity['config'] != cfg or json.loads((folder / 'run_config.json').read_text()) != cfg:
        raise ValueError('Scoring configuration differs from frozen run')
    all_ids = [s['example_id'] for s in inputs]
    if identity['evaluation_ids'] != all_ids:
        raise ValueError('Cannot report a limited run as a full-population experiment')
    run_id = fingerprint(identity)
    groups = {'full': all_ids, 'holdout': [s['example_id'] for s in inputs if s['holdout']]}
    baseline = cache_rows(folder / 'predictions/baseline.jsonl', inputs, run_id, 'baseline')
    result = {'config': cfg, 'conditions': {}, 'complete': True}
    loaded = {}
    for condition in conditions(cfg):
        path = folder / 'predictions' / (condition.replace(':', '_') + '.jsonl')
        rows = cache_rows(path, inputs, run_id, condition)
        if set(rows) != set(all_ids):
            result['complete'] = False
            if not allow_incomplete:
                raise ValueError(f'Incomplete condition: {path}; use --allow-incomplete for a progress report')
        for eid, row in rows.items():
            if row.get('baseline_fallback'):
                source = row.get('baseline_source', {})
                if source.get('row_sha256') != fingerprint(baseline.get(eid)):
                    raise ValueError(f'Fallback source changed: {condition}, {eid}')
                if (row.get('positive_bias') != 0 or row.get('negative_bias') != 0 or row.get('sas_applied')
                        or row.get('status') != baseline[eid]['status'] or row.get('progress') != baseline[eid].get('progress')):
                    raise ValueError(f'Invalid baseline fallback: {condition}, {eid}')
        record = {}
        for name, requested in groups.items():
            report = summary(rows, labels, requested)
            report['sas_applied'] = sum(rows.get(eid, {}).get('sas_applied', False) for eid in requested)
            report['baseline_fallback'] = sum(rows.get(eid, {}).get('baseline_fallback', False) for eid in requested)
            report['fallback_reasons'] = dict(Counter(rows[eid]['fallback_reason'] for eid in requested
                                                       if rows.get(eid, {}).get('baseline_fallback')))
            report['status_counts'] = dict(Counter(rows.get(eid, {}).get('status', 'missing') for eid in requested))
            report['by_task'] = {task: summary(rows, labels, [eid for eid in requested if labels[eid]['subset'] == task])
                                 for task in sorted({labels[eid]['subset'] for eid in requested})}
            record[name] = report
        if condition != 'baseline':
            applied = [eid for eid in all_ids if rows.get(eid, {}).get('sas_applied')]
            record['applied_only'] = summary(rows, labels, applied)
            record['baseline_on_applied_only'] = summary(baseline, labels, applied)
        result['conditions'][condition] = record
        loaded[condition] = rows
    result['matched_controls'] = {}
    for scope in cfg['scopes']:
        for k in cfg['top_k']:
            names = ['baseline', *[f'{scope}:{kind}:{k}' for kind in ['target', *cfg['controls']]]]
            common = [eid for eid in all_ids if all(valid_prediction(loaded[name].get(eid, {})) for name in names)
                      and all(loaded[name][eid]['sas_applied'] for name in names[1:])]
            result['matched_controls'][f'{scope}:{k}'] = {
                'n': len(common), 'example_ids': common,
                'conditions': {name: summary(loaded[name], labels, common) for name in names}}
    return result


def format_number(value):
    return '—' if value is None else f'{value:.4f}'


def table_row(experiment, condition, data):
    def accuracy(split, threshold='0.125/0.875'):
        return data.get('accuracy', {}).get(threshold, {}).get(split, {}).get('rate_all_expected')
    return {'experiment': experiment, 'condition': condition, 'expected': data['expected'], 'valid': data['n'],
            'mae': data.get('mae'), 'accuracy': accuracy('all'), 'suc_accuracy': accuracy('suc'),
            'fail_accuracy': accuracy('fail'), 'accuracy_02_08': accuracy('all', '0.2/0.8'),
            'pairwise_delta': data.get('pairwise', {}).get('continuous_mean_delta'),
            'sas_applied': data['sas_applied'], 'baseline_fallback': data['baseline_fallback'],
            'invalid': data['invalid']}


def markdown_table(rows):
    text = '| 输入 | 条件 | 有效/总数 | MAE | 总准确率 | suc | fail | 0.2/0.8准确率 | pairwise Δ | SAS | 回退 |\n'
    text += '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        text += (f'| {r["experiment"]} | {r["condition"]} | {r["valid"]}/{r["expected"]} | '
                 + ' | '.join(format_number(r[k]) for k in ('mae', 'accuracy', 'suc_accuracy', 'fail_accuracy',
                                                            'accuracy_02_08', 'pairwise_delta'))
                 + f' | {r["sas_applied"]} | {r["baseline_fallback"]} |\n')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--output-name', default='analysis_v1')
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
    configs = [cfg for _, cfg in selected]
    results, flat, heads = {}, [], {}
    for cfg in configs:
        folder = Path(cfg['output_dir'])
        if not (folder / 'run_identity.json').exists() and args.allow_incomplete:
            continue
        result = score_experiment(cfg, allow_incomplete=args.allow_incomplete)
        results[folder.name] = result
        create_json(destination / (folder.name + '.json'), result)
        for condition, record in result['conditions'].items():
            flat.append(table_row(folder.name, condition, record['full']))
        for scope in cfg['scopes']:
            path = folder / 'independent_ranking' / f'ranking_{scope}.json'
            if path.exists():
                ranking = json.loads(path.read_text())
                identity = json.loads((folder / 'run_identity.json').read_text())
                if ranking.get('run_id') != fingerprint(identity):
                    raise ValueError('Ranking/report identity mismatch')
                heads[f'{folder.name}/{scope}'] = ranking
        print(f'Scored {folder.name}', flush=True)
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
    if flat:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
        create_text(destination / 'metrics.csv', buffer.getvalue())
    notes = ('准确率使用固定总样本分母；无效输出不计为正确。MAE 使用有效预测并在 JSON 中给出缺失值上下界。'
             '主表包含 ranking 视频组；各配置 JSON 的 holdout 排除了这些视频及其反事实指令。'
             '“最佳”仅描述本次完整评测结果，不是独立调参结论。默认 bias 为 ±6。\n\n')
    best = []
    for model in sorted({cfg['model'] for cfg in configs}):
        rows = [r for r in flat if r['experiment'].startswith(model + '_')]
        candidates = [r for r in rows if ':target:' in r['condition'] and r['valid'] == r['expected'] and r['mae'] is not None]
        winner = min(candidates, key=lambda r: (r['mae'], -r['accuracy'], r['experiment'], r['condition'])) if candidates else None
        if winner:
            best.append(winner)
        text = f'# {model} basic method\n\n' + notes
        text += '完整配置中 MAE 最低的 SAS：\n\n' + markdown_table([winner]) + '\n' if winner else '尚无预测全部有效的完整 SAS 条件。\n\n'
        text += markdown_table(rows)
        create_text(destination / f'{model}_summary.md', text)
    create_text(destination / 'summary.md', '# Basic method 实验汇总\n\n' + notes
                + '## Baseline\n\n' + markdown_table([r for r in flat if r['condition'] == 'baseline'])
                + '\n## 各模型最佳完整 SAS\n\n' + markdown_table(best))
    create_json(destination / 'index.json', {'experiments': list(results), 'expected_experiments': len(configs),
                                            'selection': {'models': args.models, 'exclude_configs': args.exclude_configs,
                                                          'configs': [Path(p).stem for p, _ in selected]},
                                            'complete': len(results) == len(configs) and all(r['complete'] for r in results.values())})
    print(destination)


if __name__ == '__main__':
    main()
