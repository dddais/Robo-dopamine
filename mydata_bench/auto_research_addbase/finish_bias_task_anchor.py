"""CPU-only completion handoff for the registered actual three-cell experiment."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .bias_task_tasks import QUEUE, smoke_ready, verify_smoke, register_discovery, matrix_ready, select, register_full
from .finish_robust_discovery import ensure_scheduler, queue_has_seen


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('The original-bias/task completion trigger is CPU-only')
    folder = OUT/'analysis_watch_bias_task_anchor_20260912_0003'; folder.mkdir(exist_ok=True)
    with (folder/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        append(folder/'events.jsonl',[dict(event='start',time=time.time(),pid=os.getpid())])
        audit = OUT/'audit/bias_task_anchor_actual_gate_v1.json'
        discovery = QUEUE/'additions/stage26_bias_task_anchor_discovery.json'
        selection = OUT/'selection_bias_task_anchor_family_full_v1.json'
        validation = QUEUE/'additions/stage26_bias_task_anchor_validation.json'
        started=time.time()
        while time.time()-started < 43200:
            phase = 'waiting for all actual three-cell smoke inputs'
            if smoke_ready():
                if not audit.exists(): verify_smoke(audit)
                if not discovery.with_suffix('.ready').exists(): register_discovery(audit)
                names = [j['name'] for j in json.loads(discovery.read_text())['jobs']]
                if not queue_has_seen(QUEUE,names): ensure_scheduler(QUEUE,folder)
                phase = 'waiting for both complete five-input three-cell matrices'
                if matrix_ready():
                    if not selection.exists(): select(selection)
                    if not validation.with_suffix('.ready').exists(): register_full(selection,audit)
                    names = [j['name'] for j in json.loads(validation.read_text())['jobs']]
                    if queue_has_seen(QUEUE,names):
                        append(folder/'events.jsonl',[dict(event='complete',time=time.time(),selection=str(selection),full_jobs=len(names))])
                        return
                    ensure_scheduler(QUEUE,folder)
                    phase = 'waiting for frozen full jobs to be acknowledged'
            append(folder/'events.jsonl',[dict(event='heartbeat',phase=phase,time=time.time())])
            time.sleep(30)
        append(folder/'events.jsonl',[dict(event='watcher_timeout',time=time.time(),limit_seconds=43200)])


if __name__ == '__main__':
    main()
