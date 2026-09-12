"""Read completed round33 audits; preserve every frozen k and compare report revisions."""
import csv
import json
import math
from pathlib import Path
import time

from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .head_delta_worker import BASE, sha, verify_sources

METRICS = ['delta_mae', 'delta_accuracy_all', 'delta_accuracy_suc', 'delta_accuracy_fail']
ARMS = ['original_bias', 'B0', 'wrong_region', 'low_rank']
POPULATIONS = ['validation', 'full_cohort', 'old_holdout']


def point_key(row):
    return tuple(row[k] for k in ['experiment', 'method', 'condition', 'field', 'threshold'])


def main():
    sources = {}

    def read(path):
        path = Path(path).resolve()
        if not path.is_relative_to(OUT.resolve()):
            raise ValueError('Only the current research session')
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    complete = read(BASE/'complete.json')
    if complete['status'] != 'complete_full_target_and_controls':
        raise ValueError('Every frozen target and control must be complete')
    coverage = read(complete['coverage'])
    if sources[str(Path(complete['coverage']).resolve())] != complete['coverage_sha256']:
        raise ValueError('Frozen coverage changed')
    verify_sources(coverage['sources_sha256'])
    expected = {f"{e['point']['model']}/{e['point']['protocol']}": e['point'] for e in coverage['inputs']}
    if set(expected) != set(complete['controls']):
        raise ValueError('Missing/extra final input audit')
    rows = []; comparisons = []; exclusions = []; replay_sources = set(); revisions = []
    for key, report_path in sorted(complete['controls'].items()):
        report = read(report_path)
        if report['status'] != 'pass' or report['point'] != expected[key]:
            raise ValueError('Changed frozen selection or unverified control')
        verify_sources(report['sources_sha256'])
        point = report['point']
        replay_sources.update(p for p in report['sources_sha256']
            if ('_learned_head_gates_v1/' in p or '_learned_head_gates_k_coverage_v1/' in p)
            and '/predictions/' in p)
        want = {f'k{k}/{population}' for k in point['ks'] for population in POPULATIONS}
        if set(report['results']) != want:
            raise ValueError('Missing/extra frozen comparison')
        for k in point['ks']:
            for population in POPULATIONS:
                r = report['results'][f'k{k}/{population}']
                full = r['full_population_metrics']['baseline']
                common = r['strict_common_metrics']['baseline']
                excluded = {label: full['ordinal_prediction_distributions'][label]['n']
                    - common['ordinal_prediction_distributions'][label]['n'] for label in ['all', 'suc', 'fail']}
                if excluded['all'] != len(r['excluded_ids']) or excluded['all'] != excluded['suc']+excluded['fail']:
                    raise ValueError('Strict-subset exclusion counts differ')
                exclusions.append(dict(input=key, k=k, population=population, expected=r['expected'],
                    common_n=r['strict_common_n'], excluded_counts=excluded,
                    common_suc_n=common['ordinal_prediction_distributions']['suc']['n'],
                    common_fail_n=common['ordinal_prediction_distributions']['fail']['n']))
                for arm in ARMS:
                    is_full = arm in ['original_bias', 'B0']
                    st = (r[f'target_minus_{arm}_full_population'] if is_full
                          else r['strict_paired_changes'][f'target_minus_{arm}'])
                    n = r['expected'] if is_full else r['strict_common_n']
                    if st['status'] != 'complete' or st['n'] != n or set(st['metrics']) != set(METRICS):
                        raise ValueError('Incomplete paired comparison')
                    flags = {}; ci_flags = {}
                    for metric in METRICS:
                        v = st['metrics'][metric]; sign = -1 if metric == 'delta_mae' else 1
                        if not all(math.isfinite(z) for z in [v['estimate'], *v['ci95']]):
                            raise ValueError('Nonfinite comparison')
                        flags[metric] = sign*v['estimate'] > 0
                        ci_flags[metric] = v['ci95'][1] < 0 if sign == -1 else v['ci95'][0] > 0
                        rows.append(dict(model=point['model'], protocol=point['protocol'], scope=point['scope'],
                            k=k, center_k=point['center_k'], population=population, comparator=arm, n=n,
                            video_clusters=st['video_clusters'], metric=metric, estimate=v['estimate'],
                            ci_low=v['ci95'][0], ci_high=v['ci95'][1],
                            one_sided_cluster_signflip_p=v['one_sided_cluster_signflip_p'],
                            point_favorable=flags[metric], ci_favorable=ci_flags[metric]))
                    comparisons.append(dict(input=key, k=k, population=population, comparator=arm,
                        n=n, all_four_point_favorable=all(flags.values()), all_four_ci_favorable=all(ci_flags.values()),
                        metrics=st['metrics']))

    # Only Holm values can change when the same target rows are joined with controls.
    for population in POPULATIONS:
        before = read(Path(coverage['artifacts'][population]['checkpoint'])/'points.json')
        after = read(Path(complete['control_reports'][population]['checkpoint'])/'points.json')
        target_after = {point_key(r): r for r in after if '_target_' in r['condition']}
        if len(before) != len(target_after):
            raise ValueError('Target report coverage changed')
        changed_holm = 0
        for old in before:
            new = target_after[point_key(old)]
            a = {k: v for k, v in old.items() if not k.endswith('_holm_p')}
            b = {k: v for k, v in new.items() if not k.endswith('_holm_p')}
            if a != b:
                raise ValueError('Target effect, CI, raw p, or population changed in final statistics')
            changed_holm += any(old[m+'_holm_p'] != new[m+'_holm_p'] for m in METRICS)
        unique = [r for r in before if r['threshold'] == '0.125/0.875']
        unique_after = [r for r in target_after.values() if r['threshold'] == '0.125/0.875']
        revisions.append(dict(population=population, final_checkpoint_rows=len(after),
            final_holm_test_records_per_metric={m: sum(isinstance(r.get(m+'_p'), (int, float))
                and math.isfinite(r[m+'_p']) for r in after) for m in METRICS},
            distinct_target_configurations=len(unique), threshold_records=len(before),
            all_non_holm_fields_exactly_unchanged=True, threshold_rows_with_changed_holm=changed_holm,
            prior_all_four_holm_below_005=sum(all(r[m+'_holm_p'] < .05 for m in METRICS) for r in unique),
            final_all_four_holm_below_005=sum(all(r[m+'_holm_p'] < .05 for m in METRICS) for r in unique_after),
            final_target_holm_ranges={m: [min(r[m+'_holm_p'] for r in unique_after), max(r[m+'_holm_p'] for r in unique_after)] for m in METRICS}))

    verify_sources(sources)
    dest = OUT/'analysis'/time.strftime('head_delta_final_control_synthesis_%Y%m%d_%H%M%S')
    dest.mkdir()
    with (dest/'paired_control_statistics.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    create_json(dest/'audit.json', dict(status='pass', sources_sha256=sources,
        points=list(expected.values()), comparisons=comparisons, exclusions=exclusions,
        exact_B0_prior_gate_replay_prediction_sources=sorted(replay_sources), target_report_revisions=revisions,
        interpretation='All 61 frozen k retained. Full B0/original-bias and strict ROI/head populations differ. '
        'Uncorrected paired intervals, adaptive dataset-internal comparisons. No final scientific claim is automated.'))

    lines = ['# 第33轮完整对照：机械汇总与统计一致性', '',
        '仅汇总已完成逐例审核的全部冻结条件，不选择新参数。效应量/区间来自配对视频组统计；',
        '原±6与B0使用完整人口，wrong/low使用读标签前确定的严格共同子集。',
        'Δ均为target减对照，MAE负向/准确率正向有利；准确率单位为百分点。', '',
        '## 三人口完整k支持', '',
        '这里不要求相对对照提高10pp；“四CI有利”仅描述四项配对区间方向。', '',
        '| 人口 | 输入 | 对照 | 四CI有利的全部已测k | 四点估计有利数/全部k |',
        '| --- | --- | --- | --- | --- |']
    for population in POPULATIONS:
        for key in sorted(expected):
            for arm in ARMS:
                cs = [c for c in comparisons if c['population'] == population and c['input'] == key and c['comparator'] == arm]
                ks = ','.join(str(c['k']) for c in cs if c['all_four_ci_favorable']) or '无'
                lines.append(f"| {population} | {key} | {arm} | {ks} | {sum(c['all_four_point_favorable'] for c in cs)}/{len(cs)} |")
    lines += ['', '## 原discovery中心：validation完整效应与95%CI', '',
        '| 输入 | k | 对照 | n | ΔMAE [CI] | Δ总pp [CI] | Δsuc pp [CI] | Δfail pp [CI] |',
        '| --- | ---: | --- | ---: | --- | --- | --- | --- |']
    for key, point in sorted(expected.items()):
        for arm in ARMS:
            c, = [c for c in comparisons if c['input'] == key and c['k'] == point['center_k']
                and c['population'] == 'validation' and c['comparator'] == arm]
            cells = []
            for metric in METRICS:
                v = c['metrics'][metric]; scale = 1 if metric == 'delta_mae' else 100
                cells.append(f"{v['estimate']*scale:+.3f} [{v['ci95'][0]*scale:+.3f}, {v['ci95'][1]*scale:+.3f}]")
            lines.append(f"| {key} | {c['k']} | {arm} | {c['n']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 严格子集排除', '', '| 人口 | 输入 | 原n | 共同n（按k） | 排除suc/fail（按k） |',
        '| --- | --- | ---: | --- | --- |']
    for population in POPULATIONS:
        for key in sorted(expected):
            es = [e for e in exclusions if e['population'] == population and e['input'] == key]
            ns = ', '.join(f"{e['k']}:{e['common_n']}" for e in es)
            counts = ', '.join(f"{e['k']}:{e['excluded_counts']['suc']}/{e['excluded_counts']['fail']}" for e in es)
            lines.append(f"| {population} | {key} | {es[0]['expected']} | {ns} | {counts} |")
    lines += ['', '## 加入对照前后的主target报告', '',
        '| 人口 | 最终阈值记录 | 原/最终四Holm<.05配置数 | 主target其它字段 |',
        '| --- | ---: | --- | --- |']
    for r in revisions:
        lines.append(f"| {r['population']} | {r['final_checkpoint_rows']} | {r['prior_all_four_holm_below_005']}/{r['final_all_four_holm_below_005']} | 全部逐值一致 |")
    lines += ['', '两阈值在原生五档结果重复。Holm范围变化不能改写原效应量/CI；任何checkpoint校正都不覆盖整个自适应研究。',
        '最终checkpoint阈值记录总数不等于有效检验数；逐指标实际有限p数量另存audit.json。原±6/B0配对审计不在此checkpoint的Holm范围内。',
        '所有183个三人口k×4对照的精确效应、区间、单侧p、方向及来源见audit.json与CSV。',
        f"B0对已有第32/32b同scope/k预测的逐值回放来源共有{len(replay_sources)}个文件，清单在audit.json；没有已有来源的k不冒称旧文件复现。", '']
    (dest/'report.md').write_text('\n'.join(lines))
    print(dest, flush=True)


if __name__ == '__main__':
    main()
