"""Bounded CPU handoff: audit both real gradients before fixed gate training."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .learned_gate_worker import BASE
from .learned_gate_tasks import QUEUE, smoke_ready, verify_smoke, register_training
from .finish_robust_discovery import ensure_scheduler, queue_has_seen


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only gate watcher')
    root = BASE/'watch_20260912_1314'; root.mkdir(parents=True, exist_ok=True)
    with (root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        append(root/'events.jsonl', [dict(event='start', pid=os.getpid(), time=time.time())])
        started = time.time(); audit = BASE/'actual_gradient_gate_v1.json'
        training = QUEUE/'additions/stage31_learned_gates_training.json'
        while time.time()-started < 43200:
            if smoke_ready():
                if not audit.exists(): verify_smoke()
                if not training.with_suffix('.ready').exists(): register_training(audit)
                names = [j['name'] for j in json.loads(training.read_text())['jobs']]
                if queue_has_seen(QUEUE, names):
                    append(root/'events.jsonl', [dict(event='complete_training_registered', time=time.time(), audit=str(audit), jobs=names)])
                    return
                ensure_scheduler(QUEUE, root)
            append(root/'events.jsonl', [dict(event='heartbeat', time=time.time(), phase='waiting for both actual gradient smokes')])
            time.sleep(30)
        append(root/'events.jsonl', [dict(event='watcher_timeout', time=time.time(), limit_seconds=43200)])


if __name__ == '__main__': main()
