"""Read immutable checkpoints to track exact method families across populations.

This is a descriptive inventory, not a success declaration or a new selection
policy. Different variants never contribute to one family's input count.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from .prepare import OUT
from mydata_bench.addbase_eval.prepare import create_json


PROTOCOLS = ['official', 'image_text', 'text_image', 'interleaved', 'text_video', 'video_text']
THRESHOLDS = ['0.125/0.875', '0.2/0.8']


def identity(experiment):
    model, rest = experiment.split('_', 1)
    for protocol in PROTOCOLS:
        if rest == protocol or rest.startswith(protocol + '_'):
            return model, protocol, rest[len(protocol):].lstrip('_') or 'unversioned'
    raise ValueError(experiment)


def main():
    latest = {}
    sources = {}
    for path in sorted((OUT / 'analysis').glob('checkpoint_*/points.json')):
        points = json.loads(path.read_text())
        used = False
        for p in points:
            if p.get('population') not in {'validation', 'full_cohort'}:
                continue
            if '_target_' not in p['condition'] or p.get('derived_only'):
                continue
            expected = 660 if p['population'] == 'validation' else 846
            if p.get('expected') != expected:
                continue
            key = tuple(p[k] for k in ['experiment', 'method', 'condition', 'field', 'threshold', 'population'])
            latest[key] = dict(p, checkpoint=str(path))
            used = True
        if used:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    by_condition = defaultdict(dict)
    for (experiment, method, condition, field, threshold, population), p in latest.items():
        by_condition[(experiment, method, condition, field)][(population, threshold)] = p
    by_range = defaultdict(list)
    points = []
    for (experiment, method, condition, field), records in sorted(by_condition.items()):
        model, protocol, variant = identity(experiment)
        scope, k = condition.split('_target_')
        required = [(population, threshold) for population in ['validation', 'full_cohort'] for threshold in THRESHOLDS]
        complete = all(key in records for key in required)
        passes = complete and all(records[key]['meets_descriptive_gate'] for key in required)
        row = dict(model=model, protocol=protocol, variant=variant, method=method, scope=scope,
                   k=int(k), field=field, both_populations_and_thresholds_present=complete,
                   passes_all_descriptive_gates=passes,
                   records=[dict(population=population, threshold=threshold, point=value)
                            for (population, threshold), value in sorted(records.items())])
        points.append(row)
        by_range[(model, protocol, variant, method, scope, field)].append(row)

    ranges = []
    for key, rows in sorted(by_range.items()):
        run = []
        for row in sorted(rows, key=lambda x: x['k']) + [None]:
            if row is not None and row['passes_all_descriptive_gates']:
                run.append(row)
                continue
            if len(run) >= 2:
                ranges.append(dict(zip(['model', 'protocol', 'variant', 'method', 'scope', 'field'], key),
                                   tested_ks=[p['k'] for p in run],
                                   interpretation='Adjacent observed passing checkpoints; untested integer k values are not certified'))
            run = []
    families = defaultdict(set)
    for row in ranges:
        families[(row['model'], row['variant'], row['method'].rsplit('_s', 1)[0], row['field'])].add(row['protocol'])
    counts = [dict(model=model, variant=variant, operator=operator, field=field, protocols=sorted(protocols), input_count=len(protocols))
              for (model, variant, operator, field), protocols in sorted(families.items())]
    primary_rows = []
    primary_path = OUT / 'selection_uniform_family_primary_v1.json'
    if primary_path.exists():
        sources[str(primary_path)] = hashlib.sha256(primary_path.read_bytes()).hexdigest()
        primary = json.loads(primary_path.read_text())
        indexed = {(p['model'],p['protocol'],p['variant'],p['method'],p['scope'],p['k'],p['field']):p for p in points}
        for candidate in primary['selected']:
            complete = []; passing = []; pending = []; uncertainty = []
            for k in candidate['ks']:
                point = indexed.get((candidate['model'],candidate['protocol'],candidate['variant'],
                                     candidate['method'],candidate['scope'],k,'progress'))
                if point is None or not point['both_populations_and_thresholds_present']:
                    pending.append(k)
                    continue
                complete.append(k)
                if point['passes_all_descriptive_gates']:passing.append(k)
                for record in point['records']:
                    p = record['point']
                    if p['threshold']=='0.125/0.875':
                        uncertainty.append(dict(k=k,population=p['population'],delta_suc=p['delta_suc'],
                            suc_ci_low=p.get('delta_accuracy_suc_ci_low'),suc_ci_high=p.get('delta_accuracy_suc_ci_high'),
                            checkpoint=p['checkpoint']))
            adjacent = [candidate['ks'][i:i+2] for i in range(len(candidate['ks'])-1)
                        if all(k in passing for k in candidate['ks'][i:i+2])]
            primary_rows.append(dict(model=candidate['model'],protocol=candidate['protocol'],alpha=candidate['alpha'],
                variant=candidate['variant'],scope=candidate['scope'],frozen_ks=candidate['ks'],completed_ks=complete,
                pending_ks=pending,passing_ks=passing,adjacent_passing_pairs=adjacent,
                descriptive_range_present=bool(adjacent),suc_uncertainty=uncertainty))
    primary_counts = {model:sum(p['model']==model and p['descriptive_range_present'] for p in primary_rows)
                      for model in ['qwen','roboreward']}
    factorized_rows = []
    factorized_path = OUT / 'selection_factorized_family_primary_v1.json'
    if factorized_path.exists():
        sources[str(factorized_path)] = hashlib.sha256(factorized_path.read_bytes()).hexdigest()
        indexed = {(p['model'],p['protocol'],p['variant'],p['method'],p['scope'],p['k'],p['field']):p for p in points}
        for candidate in json.loads(factorized_path.read_text())['selected']:
            complete, passing, pending = [], [], []
            for k in candidate['ks']:
                point = indexed.get((candidate['model'],candidate['protocol'],candidate['variant'],
                                     candidate['method'],candidate['scope'],k,'progress'))
                if point is None or not point['both_populations_and_thresholds_present']:
                    pending.append(k)
                else:
                    complete.append(k)
                    if point['passes_all_descriptive_gates']:
                        passing.append(k)
            adjacent = [candidate['ks'][i:i+2] for i in range(len(candidate['ks'])-1)
                        if all(k in passing for k in candidate['ks'][i:i+2])]
            factorized_rows.append(dict(model=candidate['model'],protocol=candidate['protocol'],
                variant=candidate['variant'],scope=candidate['scope'],frozen_ks=candidate['ks'],
                completed_ks=complete,passing_ks=passing,pending_ks=pending,
                adjacent_passing_pairs=adjacent,descriptive_range_present=bool(adjacent)))
    factorized_counts = {model:sum(p['model']==model and p['descriptive_range_present'] for p in factorized_rows)
                         for model in ['qwen','roboreward']}
    destination = OUT / 'analysis' / time.strftime('frontier_%Y%m%d_%H%M%S')
    create_json(destination / 'report.json', dict(
        created_at=time.time(), sources_sha256=sources, points=points, observed_ranges=ranges, family_input_counts=counts,
        frozen_primary_uniform_family=primary_rows,primary_descriptive_input_counts=primary_counts,
        primary_numeric_coverage_met=all(n>=3 for n in primary_counts.values()),
        frozen_primary_factorized_family=factorized_rows,factorized_primary_descriptive_input_counts=factorized_counts,
        factorized_primary_numeric_coverage_met=all(n>=3 for n in factorized_counts.values()),
        limitations=[
            'Exact variants remain separate; no pooling unrelated success points.',
            'Descriptions require all four gates in both 846 and 660 populations, and both original thresholds.',
            'This inventory does not assert statistical stability, superiority over original steering, or completed controls.',
            'Repeated validation inspection makes these dataset-internal adaptive exploration results.',
            'Threshold duplicates and multiple native heads are not separate models or inputs.'
        ]))
    lines = ['# 完整检查点方法范围清单', '',
             '仅汇总完整846与validation660、两套阈值均通过四项描述门槛的相邻已测点。不同variant分别计数；未测整数k、统计稳定性和对照充分性仍须另外检查。', '',
             '| 模型 | 方法variant | 输出头 | 输入 | 已测范围 |', '| --- | --- | --- | --- | --- |']
    for row in ranges:
        lines.append(f"| {row['model']} | {row['variant']} / {row['method']} | {row['field']} | {row['protocol']} | {row['scope']} k{','.join(map(str,row['tested_ks']))} |")
    lines += ['', '不同方法的输入数不能合并为同一方案的成功。详尽数值、检查点和SHA见report.json。', '']
    if primary_rows:
        lines += ['## 完整干预结果前冻结的uniform主候选', '',
                  '此表仅允许selection_uniform_family_primary_v1.json预先选定的α、scope与k范围；不从secondary结果中替换失败输入。数值范围出现仍不等于完成统计和控制审查。', '',
                  '| 模型 | 输入 | α / scope | 冻结k | 已完整k | 四门槛通过k | 待完成k |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for row in primary_rows:
            cells=[row['model'],row['protocol'],f"{row['alpha']} / {row['scope']}"]
            cells += [','.join(map(str,row[key])) or '—' for key in ['frozen_ks','completed_ks','passing_ks','pending_ks']]
            lines.append('| '+' | '.join(cells)+' |')
        lines += ['',f'主候选描述范围输入数：{primary_counts}。','']
    if factorized_rows:
        lines += ['## 冻结的三分支主候选', '',
                  '仅追踪selection_factorized_family_primary_v1.json的六个原选择；不合并uniform或其它variant。数值覆盖仍须另做控制与统计审查。', '',
                  '| 模型 | 输入 | scope | 冻结k | 已完整k | 四门槛通过k | 待完成k |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for row in factorized_rows:
            cells=[row['model'],row['protocol'],row['scope']]
            cells += [','.join(map(str,row[key])) or '—' for key in ['frozen_ks','completed_ks','passing_ks','pending_ks']]
            lines.append('| '+' | '.join(cells)+' |')
        lines += ['',f'三分支主候选描述范围输入数：{factorized_counts}。','']
    with (destination / 'report.md').open('x') as f:
        f.write('\n'.join(lines))
    print(destination)
    for row in counts:
        print(row)
    print('Frozen uniform primary descriptive ranges:',primary_counts)
    print('Frozen factorized primary descriptive ranges:',factorized_counts)


if __name__ == '__main__':
    main()
