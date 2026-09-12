"""Score/export round32 only after complete prediction audits, never partial selection."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .head_gate_worker import BASE, PROTOCOLS, sha, verify_sources


def run_reports(root, experiments, population, score_split=None):
    tag = score_split or population
    stamp = time.strftime('%Y%m%d_%H%M%S')
    name = f'checkpoint_head_gates_complete_{tag}_{stamp}'
    command = [sys.executable, '-B', '-m', 'mydata_bench.auto_research_addbase.analyze',
        '--population', population, '--name', name, '--uncertainty', '--experiments', *experiments]
    if score_split: command += ['--score-split', score_split]
    with (root/f'{name}.log').open('xb') as log:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], stdout=log,
            stderr=subprocess.STDOUT, check=True, timeout=3600)
    checkpoint = OUT/'analysis'/name
    export_name = f'exports_head_gates_complete_{tag}_{stamp}'
    with (root/f'{export_name}.log').open('xb') as log:
        subprocess.run([sys.executable, '-B', '-m', 'mydata_bench.auto_research_addbase.export_tables',
            '--checkpoint', str(checkpoint), '--name', export_name], cwd=Path(__file__).resolve().parents[2],
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    return dict(checkpoint=str(checkpoint), exports=str(OUT/'analysis'/export_name),
        points_sha256=sha(checkpoint/'points.json'), details_sha256=sha(checkpoint/'details.json'))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only statistics completion chain')
    root = BASE/'watch_statistics_20260912_1411'; root.mkdir(parents=True, exist_ok=True)
    with (root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.time(); events = root/'events.jsonl'; done = {}
        if events.exists():
            for line in events.read_text().splitlines():
                event = json.loads(line)
                if event['event'] == 'report_complete': done[event['population']] = event['artifacts']
        append(events, [dict(event='start', pid=os.getpid(), time=start)])
        selection_file = BASE/'selection_complete_discovery_v1.json'
        while time.time()-start < 43200:
            if selection_file.exists():
                selection = json.loads(selection_file.read_text()); verify_sources(selection['sources_sha256'])
                if 'discovery' not in done:
                    experiments = [f'{m}_{p}_learned_head_gates_v1' for m in ['qwen','roboreward'] for p in PROTOCOLS]
                    done['discovery'] = run_reports(root, experiments, 'discovery')
                    append(events, [dict(event='report_complete', population='discovery', artifacts=done['discovery'], time=time.time())])
                if not selection['selected']:
                    create_json(root/'complete_reports.json', dict(status='complete_discovery_no_full',
                        reports=done, selection=str(selection_file), selection_sha256=sha(selection_file)))
                    append(events, [dict(event='complete', time=time.time())]); return
                full_index = BASE/'watch_full_20260912_1355/complete_audit_index.json'
                if full_index.exists():
                    full = json.loads(full_index.read_text())
                    if full['status'] != 'pass' or full['selection_sha256'] != sha(selection_file):
                        raise ValueError('Full/control matrix must be completely audited before final scoring')
                    experiments = [f"{p['model']}_{p['protocol']}_learned_head_gates_v1" for p in selection['selected']]
                    for population in ['validation','full_cohort','old_holdout']:
                        if population in done: continue
                        done[population] = run_reports(root, experiments, 'full_cohort', None if population=='full_cohort' else population)
                        append(events, [dict(event='report_complete', population=population, artifacts=done[population], time=time.time())])
                    create_json(root/'complete_reports.json', dict(status='complete_all_requested_populations',
                        reports=done, full_audit_index=str(full_index), full_audit_sha256=sha(full_index),
                        interpretation='Reports include target and explicitly named controls. Only frozen target neighborhoods can count for primary family coverage.'))
                    append(events, [dict(event='complete', time=time.time())]); return
            append(events, [dict(event='heartbeat', time=time.time(), completed_reports=list(done))])
            time.sleep(30)
        append(events, [dict(event='watcher_timeout', time=time.time(), limit_seconds=43200)])


if __name__ == '__main__': main()
