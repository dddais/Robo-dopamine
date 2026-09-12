"""Round32b: audit all added k, assess the unchanged family, then conditional controls."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import append, latest
from .prepare import OUT
from .head_gate_worker import BASE as TRAINING, sha, verify_sources
from .head_gate_eval_tasks import command, verify_rows, NEIGHBORHOODS
from .head_gate_tasks import addition, QUEUE
from .compare_native_bias import same_input
from .head_gate_controls import verify_and_score


POLICY = OUT/'selection_learned_head_gates_k_coverage_v1.json'
BASE = OUT/'learned_head_gates_k_coverage_v1'
VARIANT = 'learned_head_gates_k_coverage_v1'


def specification():
    spec = json.loads(POLICY.read_text()); verify_sources(spec['sources_sha256'])
    source = TRAINING/'selection_complete_discovery_v1.json'
    old = json.loads(source.read_text()); verify_sources(old['sources_sha256'])
    expected = []
    for p in old['selected']:
        centers = [r['center_k'] for r in old['all_discovery_points'] if r['model'] == p['model']
            and r['protocol'] == p['protocol'] and r['scope'] == p['scope'] and r['passes']]
        ks = sorted({k for center in centers for k in NEIGHBORHOODS[center]})
        expected.append(dict(model=p['model'], protocol=p['protocol'], scope=p['scope'],
            original_ks=p['ks'], discovery_passing_centers=centers,
            extra_ks=sorted(set(ks)-set(p['ks'])), all_ks=ks))
    if spec['variant'] != VARIANT or spec['candidates'] != expected or sum(len(p['extra_ks']) for p in expected) != 22:
        raise ValueError('Require every remaining original passing-center neighborhood for all six inputs')
    return spec


def folder(p, original=False, bias=False):
    variant = ('learned_head_gates_k_coverage_matched_bias' if bias else
               'learned_head_gates_v1' if original else VARIANT)
    return OUT/'experiments'/f'{p["model"]}_{p["protocol"]}_{variant}'/'full_cohort'


def ready(p, controls=False):
    root = folder(p); ids = set(json.loads((OUT/'splits.json').read_text())['full_cohort'])
    files = [root/'predictions/baseline.jsonl']
    for k in p['extra_ks']:
        files += [root/'learned_head_gate_bias_s6/predictions'/f'{p["scope"]}_{arm}_{k}.jsonl'
                  for arm in (['target', 'wrong_region', 'low_rank'] if controls else ['target'])]
        if controls:
            files.append(folder(p, bias=True)/'bias_s6/predictions'/f'{p["scope"]}_target_{k}.jsonl')
    if controls: files.append(folder(p, bias=True)/'predictions/baseline.jsonl')
    return all(f.exists() and set(latest(f)) == ids for f in files)


def audit(p):
    spec = specification(); sources = {str(POLICY): sha(POLICY)}
    def read(path, rows=False):
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    root = folder(p); old = folder(p, original=True)
    ids = read(OUT/'splits.json')['full_cohort']
    if len(ids) != 846 or read(root/'requested_ids.json') != ids or read(old/'requested_ids.json') != ids:
        raise ValueError('Full original population and order must be preserved')
    cfg = read(root/'runtime_config.json'); before_cfg = read(old/'runtime_config.json')
    if {k:v for k,v in cfg.items() if k != 'output_dir'} != {k:v for k,v in before_cfg.items() if k != 'output_dir'}:
        raise ValueError('Only the additive output directory may differ')
    gate_file = TRAINING/'training'/p['model']/'final_gates.json'; gates = read(gate_file)
    baseline = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, gates, gate_file, baseline=True)
    old_base = verify_rows(read(old/'predictions/baseline.jsonl', True), ids, gates, gate_file, baseline=True)
    if any(not same_input(row, old_base[e]) or row['native_class_logits_positive'] != old_base[e]['native_class_logits_positive']
           for e, row in baseline.items()):
        raise ValueError('Actual same-batch unsteered baseline must replay exactly')
    ranks_file = OUT/'functional_selections'/f'stage8_{p["model"]}_v1'/f'{p["model"]}_{p["protocol"]}'/f'ranking_{p["scope"]}.json'
    ranks = read(ranks_file)
    if read(root.parent/'ranking'/ranks_file.name) != ranks or read(old.parent/'ranking'/ranks_file.name) != ranks:
        raise ValueError('Both variants must use the same entire head ranking')
    checks = []
    # Recheck the full union with identical row contracts, not just the newly measured k.
    for k in p['all_ks']:
        origin = root if k in p['extra_ks'] else old
        name = f'{p["scope"]}_target_{k}'
        selected = {(h['layer'], h['head']) for h in ranks['ranking'][:k]}
        rows = verify_rows(read(origin/'learned_head_gate_bias_s6/predictions'/f'{name}.jsonl', True), ids, gates, gate_file, selected)
        if len(selected) != k or any(not same_input(row, baseline[e]) or row['condition'] != name.replace('_target_', ':target:')
            for e, row in rows.items()):
            raise ValueError('Actual k, native input or condition changed')
        checks.append(dict(k=k, n=len(rows), newly_measured=k in p['extra_ks']))
    verify_sources(sources)
    dest = BASE/'audits'/f'{p["model"]}_{p["protocol"]}.json'
    create_json(dest, dict(status='pass', labels_read=False, point=p, sources_sha256=sources, checks=checks,
        interpretation='Complete old/new target union has identical final gates, ranking, native input and baseline. Same method identity, not independent repeated evidence.'))
    return dest


def reports(spec, controls=False):
    artifacts = {}
    experiments = [folder(p).parent.name for p in spec['candidates']]
    if not controls: experiments += [folder(p, original=True).parent.name for p in spec['candidates']]
    for population in ['validation', 'full_cohort', 'old_holdout']:
        name = time.strftime(f'checkpoint_head_gates_k_coverage_{"controls" if controls else "target_union"}_{population}_%Y%m%d_%H%M%S')
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
    sources = {str(POLICY): sha(POLICY), **{str(p):sha(p) for p in audits.values()}}
    lookup = {}
    for population, report in artifacts.items():
        path = Path(report['checkpoint'])/'points.json'; sources[str(path)] = sha(path)
        for row in json.loads(path.read_text()):
            if row['threshold'] != '0.125/0.875': continue
            model, name = row['experiment'].split('_', 1)
            protocol = name.removesuffix('_'+VARIANT).removesuffix('_learned_head_gates_v1')
            key = (model, protocol, population, int(row['condition'].rsplit('_', 1)[1]))
            if key in lookup or row['field'] != 'progress' or '_target_' not in row['condition']:
                raise ValueError('Duplicate union k or changed primary field')
            lookup[key] = row
    if len(lookup) != 3*sum(len(p['all_ks']) for p in spec['candidates']):
        raise ValueError('Require all40 conditions and all three populations before union selection')
    entries = []
    for p in spec['candidates']:
        support = {}
        for k in p['all_ks']:
            populations = {}
            for population in artifacts:
                r = lookup[(p['model'], p['protocol'], population, k)]
                ci = r['delta_mae_ci_high'] < 0 and all(r[f'delta_accuracy_{c}_ci_low'] > 0 for c in ['all', 'suc', 'fail'])
                populations[population] = dict(descriptive=r['meets_descriptive_gate'], four_ci_favorable=ci,
                    delta_mae=r['delta_mae'], delta_all=r['delta_all'], delta_suc=r['delta_suc'], delta_fail=r['delta_fail'])
            support[k] = dict(populations=populations, passes=all(v['descriptive'] and v['four_ci_favorable'] for v in populations.values()))
        pairs = [[a,b] for a,b in zip(p['all_ks'],p['all_ks'][1:]) if support[a]['passes'] and support[b]['passes']]
        entries.append(dict(point=p, support=support, adjacent_supported_pairs=pairs, input_passes=bool(pairs)))
    counts = {m:sum(e['input_passes'] for e in entries if e['point']['model'] == m) for m in ['qwen','roboreward']}
    verify_sources(sources)
    result = dict(shared_coverage=min(counts.values()) >= 3, input_counts=counts, inputs=entries,
        artifacts=artifacts, sources_sha256=sources,
        interpretation='Adaptive dataset-internal union of strictly identical final-gate methods. All40 measured k are reported. Point gains must reach10pp; CI direction is checked, but CI lower bound need not exceed10pp. This is not a retroactive pass of the original six neighborhoods.')
    create_json(BASE/'coverage.json', result)
    return result


def register_controls(spec):
    jobs = []
    for p in spec['candidates']:
        m = p['model']; protocol = p['protocol']; name = f'round32b_{m}_{protocol}_k_controls'
        jobs.append(dict(name=name, gpu=['qwen','roboreward'].index(m), min_free_mb=23000,
            depends_on=[f'round32b_{m}_{protocol}_k_coverage'],
            command=command(m,[protocol],'full_cohort',p['extra_ks'],[p['scope']])+['--variant', VARIANT,'--controls','wrong_region','low_rank']))
        bias_cmd = [sys.executable,'-B','-m','mydata_bench.auto_research_addbase.worker','--model',m,'--protocols',protocol,
            '--variant','learned_head_gates_k_coverage_matched_bias','--population','full_cohort','--methods','bias','--strengths','6',
            '--ks',*map(str,p['extra_ks']),'--scopes',p['scope'],'--batch-size','8','--ranking-prefix','ANSWER: ',
            '--contrast-weight','0','--frozen-ranking-root',str(OUT/'functional_selections'/f'stage8_{m}_v1')]
        jobs.append(dict(name=name+'_original_bias',gpu=['qwen','roboreward'].index(m),min_free_mb=23000,
            depends_on=[name],command=bias_cmd))
    return addition('stage32b_learned_head_gates_k_controls.json', jobs,
        {str(POLICY):sha(POLICY),str(BASE/'coverage.json'):sha(BASE/'coverage.json')})


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only completion chain')
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spec = specification(); start = time.time(); events=BASE/'events.jsonl'; audits={}; compared={}
        if events.exists():
            for line in events.read_text().splitlines():
                e=json.loads(line)
                if e['event'] in ['input_audited','controls_compared']:
                    if sha(e['artifact']) != e['sha256']:raise ValueError('Previous immutable artifact changed')
                    (audits if e['event']=='input_audited' else compared)[e['input']]=e['artifact']
        append(events,[dict(event='start',pid=os.getpid(),time=start)])
        while time.time()-start < 43200:
            for p in spec['candidates']:
                name=f'{p["model"]}/{p["protocol"]}'
                if name not in audits and ready(p):
                    artifact=audit(p);audits[name]=str(artifact)
                    append(events,[dict(event='input_audited',input=name,artifact=str(artifact),sha256=sha(artifact),time=time.time())])
            if len(audits)==6:
                decision_file=BASE/'coverage.json'
                if not decision_file.exists():
                    artifacts=reports(spec);decision=coverage(spec,audits,artifacts)
                    append(events,[dict(event='union_scored',shared_coverage=decision['shared_coverage'],input_counts=decision['input_counts'],time=time.time())])
                decision=json.loads(decision_file.read_text());verify_sources(decision['sources_sha256'])
                if not decision['shared_coverage']:
                    create_json(BASE/'complete.json',dict(status='complete_failed_shared_coverage',coverage=str(decision_file),coverage_sha256=sha(decision_file),new_controls_registered=False))
                    return
                jobs=QUEUE/'additions/stage32b_learned_head_gates_k_controls.json'
                if not jobs.with_suffix('.ready').exists():register_controls(spec)
                for p in spec['candidates']:
                    name=f'{p["model"]}/{p["protocol"]}'
                    if name not in compared and ready(p,controls=True):
                        artifact=verify_and_score(p['model'],p['protocol'],extension=True);compared[name]=str(artifact)
                        append(events,[dict(event='controls_compared',input=name,artifact=str(artifact),sha256=sha(artifact),time=time.time())])
                if len(compared)==6:
                    artifact=reports(spec,controls=True)
                    create_json(BASE/'complete.json',dict(status='complete_full_target_and_controls',coverage=str(decision_file),
                        coverage_sha256=sha(decision_file),controls=compared,control_reports=artifact,
                        interpretation='Final acceptance still requires original32 controls and all mechanism/statistical limitations to be synthesized.'))
                    return
            append(events,[dict(event='heartbeat',time=time.time(),audited=list(audits),controls_compared=list(compared))])
            time.sleep(30)
        append(events,[dict(event='timeout',time=time.time(),limit_seconds=43200)])


if __name__ == '__main__': main()
