"""CPU-only, bounded completion chain for registered local head-output contrast."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .head_output_tasks import QUEUE, verify_smoke, register_discovery, matrix_ready, select, register_full
from .finish_robust_discovery import ensure_scheduler, queue_has_seen


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This completion chain is CPU-only')
    folder = OUT/'analysis_watch_head_output_20260912_1250'; folder.mkdir(exist_ok=True)
    with (folder/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        append(folder/'events.jsonl', [dict(event='start', time=time.time(), pid=os.getpid())])
        audit = OUT/'audit/head_output_actual_gate_v1.json'
        discovery = QUEUE/'additions/stage30_head_output_discovery.json'
        selection = OUT/'selection_head_output_family_full_v1.json'
        validation = QUEUE/'additions/stage30_head_output_validation.json'
        started = time.time()
        while time.time()-started < 43200:
            phase = 'waiting for both entire actual local head-output smoke matrices'
            if matrix_ready(smoke=True):
                if not audit.exists(): verify_smoke(audit)
                if not discovery.with_suffix('.ready').exists(): register_discovery(audit)
                names = [j['name'] for j in json.loads(discovery.read_text())['jobs']]
                if not queue_has_seen(QUEUE, names): ensure_scheduler(QUEUE, folder)
                phase = 'waiting for both complete five-input single-forward matrices'
                if matrix_ready():
                    if not selection.exists(): select(selection)
                    if not validation.with_suffix('.ready').exists(): register_full(selection, audit)
                    names = [j['name'] for j in json.loads(validation.read_text())['jobs']]
                    if queue_has_seen(QUEUE, names):
                        append(folder/'events.jsonl', [dict(event='complete', time=time.time(), selection=str(selection), full_jobs=len(names))])
                        return
                    ensure_scheduler(QUEUE, folder)
                    phase = 'waiting for frozen full jobs to be acknowledged'
            append(folder/'events.jsonl', [dict(event='heartbeat', phase=phase, time=time.time())])
            time.sleep(30)
        append(folder/'events.jsonl', [dict(event='watcher_timeout', time=time.time(), limit_seconds=43200)])


if __name__ == '__main__': main()
