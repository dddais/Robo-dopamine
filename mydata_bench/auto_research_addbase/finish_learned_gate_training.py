"""Complete fixed training, audit before resubstitution scoring, then freeze full jobs."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .learned_gate_worker import BASE
from .learned_gate_tasks import QUEUE
from .learned_gate_eval_tasks import training_ready, verify_training, register_discovery, matrix_ready, select, register_full
from .finish_robust_discovery import ensure_scheduler, queue_has_seen


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only completion watcher')
    root = BASE/'watch_training_20260912_1324'; root.mkdir(parents=True, exist_ok=True)
    with (root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        append(root/'events.jsonl', [dict(event='start', time=time.time(), pid=os.getpid())])
        start = time.time(); audit = BASE/'fixed_training_audit_v1.json'
        discovery = QUEUE/'additions/stage31_learned_gates_discovery.json'
        selection = BASE/'selection_complete_discovery_v1.json'
        validation = QUEUE/'additions/stage31_learned_gates_validation.json'
        while time.time()-start < 43200:
            phase = 'waiting for both fixed final gate checkpoints'
            if training_ready():
                if not audit.exists(): verify_training()
                if not discovery.with_suffix('.ready').exists(): register_discovery(audit)
                names = [j['name'] for j in json.loads(discovery.read_text())['jobs']]
                if not queue_has_seen(QUEUE, names): ensure_scheduler(QUEUE, root)
                phase = 'waiting for complete trained-gate discovery matrices'
                if matrix_ready():
                    if not selection.exists(): select(selection)
                    if not validation.with_suffix('.ready').exists(): register_full(selection)
                    names = [j['name'] for j in json.loads(validation.read_text())['jobs']]
                    if queue_has_seen(QUEUE, names):
                        append(root/'events.jsonl', [dict(event='complete', time=time.time(), selection=str(selection), full_jobs=len(names))])
                        return
                    ensure_scheduler(QUEUE, root)
                    phase = 'waiting for frozen full jobs to be acknowledged'
            append(root/'events.jsonl', [dict(event='heartbeat', time=time.time(), phase=phase)])
            time.sleep(30)
        append(root/'events.jsonl', [dict(event='watcher_timeout', time=time.time(), limit_seconds=43200)])


if __name__ == '__main__': main()
