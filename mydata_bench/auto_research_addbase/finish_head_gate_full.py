"""Complete conditional round32 full audits and all prespecified controls."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .head_gate_worker import BASE, sha
from .head_gate_tasks import QUEUE
from .verify_head_gate_full import SELECTION, selected, ready, verify
from . import head_gate_controls as controls
from .finish_robust_discovery import ensure_scheduler, queue_has_seen


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only audit/control handoff')
    root = BASE/'watch_full_20260912_1355'; root.mkdir(parents=True, exist_ok=True)
    with (root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = root/'events.jsonl'; start = time.time(); completed = {}; compared = {}
        if events.exists():
            for line in events.read_text().splitlines():
                e = json.loads(line)
                if e['event'] in ['input_verified', 'controls_compared']:
                    if sha(e['artifact']) != e['sha256']: raise ValueError('Previous immutable audit changed')
                    (completed if e['event'] == 'input_verified' else compared)[e['input']] = e['artifact']
        append(events, [dict(event='start', pid=os.getpid(), time=start)])
        ids = json.loads((OUT/'splits.json').read_text())['full_cohort']
        record = None; selection_sha = None
        while time.time()-start < 43200:
            if record is None and SELECTION.exists():
                record = selected(); selection_sha = sha(SELECTION)
            if record is not None:
                if sha(SELECTION) != selection_sha: raise ValueError('Frozen selection changed')
                if not record['selected']:
                    create_json(root/'complete_audit_index.json', dict(status='not_expanded', full_inputs=0,
                        selection=str(SELECTION), selection_sha256=selection_sha,
                        interpretation='Failed shared discovery coverage; no full/control efficacy claim.'))
                    append(events, [dict(event='complete_no_full', time=time.time())]); return
                full = QUEUE/'additions/stage32_learned_head_gates_validation.json'
                control_jobs = QUEUE/'additions/stage32_learned_head_gates_controls.json'
                if full.with_suffix('.ready').exists():
                    if not control_jobs.with_suffix('.ready').exists(): controls.register()
                    names = [j['name'] for j in json.loads(control_jobs.read_text())['jobs']]
                    if not queue_has_seen(QUEUE, names): ensure_scheduler(QUEUE, root)
                for p in record['selected']:
                    name = f"{p['model']}/{p['protocol']}"
                    for event, done, is_ready, run in [
                        ('input_verified', completed, ready, verify),
                        ('controls_compared', compared, controls.ready, controls.verify_and_score)]:
                        if name in done or not is_ready(p, ids): continue
                        artifact = run(p['model'], p['protocol']); done[name] = str(artifact)
                        append(events, [dict(event=event, input=name, artifact=str(artifact), sha256=sha(artifact), time=time.time())])
                if len(completed) == len(compared) == len(record['selected']):
                    create_json(root/'complete_audit_index.json', dict(status='pass', selection=str(SELECTION),
                        selection_sha256=selection_sha, full_audits=completed, complete_controls=compared,
                        interpretation='Full frozen matrix and all prespecified controls complete. Family efficacy requires separate synthesis.'))
                    append(events, [dict(event='complete', time=time.time(), full_inputs=len(completed))]); return
            append(events, [dict(event='heartbeat', time=time.time(), full_audited=list(completed), controls_compared=list(compared))])
            time.sleep(30)
        append(events, [dict(event='watcher_timeout', time=time.time(), limit_seconds=43200)])


if __name__ == '__main__': main()
