"""CPU-only deterministic round22 actual-smoke and joint family gate chain."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .task_domain import QUEUE, smoke_ready, verify_smoke, register_discovery, matrix_ready, select, register_full
from .finish_robust_discovery import ensure_scheduler, queue_has_seen


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The task-domain completion trigger is CPU-only')
    folder=OUT / 'analysis_watch_task_domain_20260911_2210'; folder.mkdir(exist_ok=True)
    with (folder / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        append(folder / 'events.jsonl', [dict(event='start', time=time.time(), pid=os.getpid())])
        audit=OUT / 'audit/task_domain_actual_gate_v1.json'
        discovery=QUEUE / 'additions/stage22_task_domain_discovery.json'
        selection=OUT / 'selection_task_domain_family_full_v1.json'
        validation=QUEUE / 'additions/stage22_task_domain_validation.json'
        while True:
            phase='waiting for both models and both actual task-domain smoke inputs'
            if smoke_ready():
                if not audit.exists():
                    verify_smoke(audit)
                if not discovery.with_suffix('.ready').exists():
                    register_discovery(audit)
                names=[j['name'] for j in json.loads(discovery.read_text())['jobs']]
                if not queue_has_seen(QUEUE, names):
                    ensure_scheduler(QUEUE, folder)
                phase='waiting for both complete five-input task-domain discovery matrices'
                if matrix_ready():
                    if not selection.exists():
                        select(selection)
                    if not validation.with_suffix('.ready').exists():
                        register_full(selection, audit)
                    names=[j['name'] for j in json.loads(validation.read_text())['jobs']]
                    if queue_has_seen(QUEUE, names):
                        append(folder / 'events.jsonl', [dict(event='complete', time=time.time(),
                            selection=str(selection), full_jobs=len(names))])
                        return
                    ensure_scheduler(QUEUE, folder)
                    phase='waiting for frozen full tasks to be acknowledged'
            append(folder / 'events.jsonl', [dict(event='heartbeat', phase=phase, time=time.time())])
            time.sleep(30)


if __name__=='__main__':
    main()
