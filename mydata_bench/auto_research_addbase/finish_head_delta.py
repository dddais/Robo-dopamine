"""CPU completion chain: fixed training, complete discovery, full targets and controls."""
import fcntl
import json
import os
import time
from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import append
from .head_delta_worker import BASE, sha, verify_sources
from .head_delta_eval_tasks import training_ready, verify_training, register_discovery, matrix_ready, select, register_full, QUEUE
from .verify_head_delta_full import SELECTION, selected, ready, verify
from .head_delta_statistics import reports, coverage
from .finish_robust_discovery import ensure_scheduler, queue_has_seen
from .prepare import OUT


def ensure_jobs(path, root):
    names = [job['name'] for job in json.loads(path.read_text())['jobs']]
    if names and not queue_has_seen(QUEUE, names): ensure_scheduler(QUEUE, root)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only round33 completion chain')
    root = BASE/'watch_20260912_1556'; root.mkdir(parents=True, exist_ok=True)
    with (root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = root/'events.jsonl'; start = time.time(); audits = {}; compared = {}
        if events.exists():
            for line in events.read_text().splitlines():
                event = json.loads(line)
                if event['event'] in ['input_audited', 'controls_compared']:
                    if sha(event['artifact']) != event['sha256']: raise ValueError('Previous immutable audit changed')
                    (audits if event['event'] == 'input_audited' else compared)[event['input']] = event['artifact']
        append(events, [dict(event='start', pid=os.getpid(), time=start)])
        ids = json.loads((OUT/'splits.json').read_text())['full_cohort']
        while time.time()-start < 43200:
            phase = 'waiting_for_both_fixed_final_checkpoints'
            if training_ready():
                audit = BASE/'fixed_training_audit_v1.json'
                if not audit.exists(): verify_training()
                discovery = QUEUE/'additions/stage33_head_delta_discovery.json'
                if not discovery.with_suffix('.ready').exists(): register_discovery(audit)
                ensure_jobs(discovery, root); phase = 'waiting_for_complete_60_condition_discovery'
                if matrix_ready():
                    if not SELECTION.exists(): select(SELECTION)
                    record = selected()
                    if not record['selected']:
                        create_json(BASE/'complete.json', dict(status='complete_failed_discovery_coverage',
                            selection=str(SELECTION), selection_sha256=sha(SELECTION), new_controls_registered=False))
                        return
                    full = QUEUE/'additions/stage33_head_delta_validation.json'
                    if not full.with_suffix('.ready').exists(): register_full(SELECTION)
                    ensure_jobs(full, root); phase = 'waiting_for_all_frozen_full_target_matrices'
                    for point in record['selected']:
                        key = f"{point['model']}/{point['protocol']}"
                        if key not in audits and ready(point, ids):
                            path = verify(point['model'], point['protocol']); audits[key] = str(path)
                            append(events, [dict(event='input_audited', input=key, artifact=str(path), sha256=sha(path), time=time.time())])
                    if len(audits) == len(record['selected']):
                        decision_file = BASE/'coverage.json'
                        if not decision_file.exists():
                            artifacts = reports(record); decision = coverage(record, audits, artifacts)
                            append(events, [dict(event='full_targets_scored', time=time.time(),
                                shared_coverage=decision['shared_coverage'], input_counts=decision['input_counts'])])
                        decision = json.loads(decision_file.read_text()); verify_sources(decision['sources_sha256'])
                        if not decision['shared_coverage']:
                            create_json(BASE/'complete.json', dict(status='complete_failed_shared_coverage',
                                coverage=str(decision_file), coverage_sha256=sha(decision_file), new_controls_registered=False))
                            return
                        from . import head_delta_controls as controls
                        jobs = QUEUE/'additions/stage33_head_delta_controls.json'
                        if not jobs.with_suffix('.ready').exists(): controls.register()
                        ensure_jobs(jobs, root); phase = 'waiting_for_all_frozen_conditional_controls'
                        for point in record['selected']:
                            key = f"{point['model']}/{point['protocol']}"
                            if key not in compared and controls.ready(point, ids):
                                path = controls.verify_and_score(point['model'], point['protocol']); compared[key] = str(path)
                                append(events, [dict(event='controls_compared', input=key, artifact=str(path), sha256=sha(path), time=time.time())])
                        if len(compared) == len(record['selected']):
                            artifact = reports(record, controls=True)
                            create_json(BASE/'complete.json', dict(status='complete_full_target_and_controls',
                                coverage=str(decision_file), coverage_sha256=sha(decision_file), controls=compared,
                                full_audits=audits, control_reports=artifact,
                                interpretation='All frozen experiments complete. Final scientific acceptance requires synthesis of effects, controls and limitations.'))
                            return
            append(events, [dict(event='heartbeat', time=time.time(), phase=phase, full_audited=list(audits), controls_compared=list(compared))])
            time.sleep(30)
        append(events, [dict(event='watcher_timeout', time=time.time(), limit_seconds=43200)])


if __name__ == '__main__': main()
