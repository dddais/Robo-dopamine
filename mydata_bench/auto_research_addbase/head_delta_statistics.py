"""Round33 complete target statistics and frozen shared-coverage decision."""
import json
from pathlib import Path
import subprocess
import sys
import time
from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .head_delta_worker import BASE, POLICY, VARIANT, sha, verify_sources
from .head_delta_eval_tasks import folder
from .verify_head_delta_full import SELECTION

def reports(spec, controls=False):
    artifacts = {}
    experiments = [folder(p['model'],p['protocol'],'full_cohort').parent.name for p in spec['selected']]
    for population in ['validation', 'full_cohort', 'old_holdout']:
        name = time.strftime(f'checkpoint_head_delta_reft_{"controls" if controls else "target_union"}_{population}_%Y%m%d_%H%M%S')
        cmd = [sys.executable, '-B', '-m', 'mydata_bench.auto_research_addbase.analyze',
            '--population', 'full_cohort', '--name', name, '--uncertainty', '--experiments', *experiments]
        if not controls: cmd += ['--condition-kind', 'target']
        if population != 'full_cohort': cmd += ['--score-split', population]
        with (BASE/f'{name}.log').open('xb') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        checkpoint = OUT/'analysis'/name
        export_name = name.replace('checkpoint_', 'exports_', 1)
        with (BASE/f'{export_name}.log').open('xb') as log:
            subprocess.run([sys.executable, '-B', '-m', 'mydata_bench.auto_research_addbase.export_tables',
                '--checkpoint', str(checkpoint), '--name', export_name], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        artifacts[population] = dict(checkpoint=str(checkpoint), exports=str(OUT/'analysis'/export_name))
    return artifacts


def coverage(spec, audits, artifacts):
    sources = {str(POLICY): sha(POLICY), str(SELECTION): sha(SELECTION), **{str(p):sha(p) for p in audits.values()}}
    lookup = {}
    for population, report in artifacts.items():
        path = Path(report['checkpoint'])/'points.json'; sources[str(path)] = sha(path)
        for row in json.loads(path.read_text()):
            if row['threshold'] != '0.125/0.875': continue
            model, name = row['experiment'].split('_', 1)
            protocol = name.removesuffix('_'+VARIANT)
            key = (model, protocol, population, int(row['condition'].rsplit('_', 1)[1]))
            if key in lookup or row['field'] != 'progress' or '_target_' not in row['condition']:
                raise ValueError('Duplicate union k or changed primary field')
            lookup[key] = row
    if len(lookup) != 3*sum(len(p['ks']) for p in spec['selected']):
        raise ValueError('Require every frozen target condition in all three populations')
    entries = []
    for p in spec['selected']:
        support = {}
        for k in p['ks']:
            populations = {}
            for population in artifacts:
                r = lookup[(p['model'], p['protocol'], population, k)]
                ci = r['delta_mae_ci_high'] < 0 and all(r[f'delta_accuracy_{c}_ci_low'] > 0 for c in ['all', 'suc', 'fail'])
                populations[population] = dict(descriptive=r['meets_descriptive_gate'], four_ci_favorable=ci,
                    delta_mae=r['delta_mae'], delta_all=r['delta_all'], delta_suc=r['delta_suc'], delta_fail=r['delta_fail'])
            support[k] = dict(populations=populations, passes=all(v['descriptive'] and v['four_ci_favorable'] for v in populations.values()))
        pairs = [[a,b] for a,b in zip(p['ks'],p['ks'][1:]) if support[a]['passes'] and support[b]['passes']]
        entries.append(dict(point=p, support=support, adjacent_supported_pairs=pairs, input_passes=bool(pairs)))
    counts = {m:sum(e['input_passes'] for e in entries if e['point']['model'] == m) for m in ['qwen','roboreward']}
    verify_sources(sources)
    result = dict(shared_coverage=min(counts.values()) >= 3, input_counts=counts, inputs=entries,
        artifacts=artifacts, sources_sha256=sources,
        interpretation='Adaptive dataset-internal final rank4 method. Every frozen measured k is reported. Point gains must reach10pp; CI direction is checked, but CI lower bound need not exceed10pp. This is not independent validation or external generalization.')
    create_json(BASE/'coverage.json', result)
    return result

