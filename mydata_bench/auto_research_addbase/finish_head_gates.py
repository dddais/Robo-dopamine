"""Bounded round32 CPU handoff: real smoke, fixed training, complete selection."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .head_gate_worker import BASE
from .head_gate_tasks import QUEUE, smoke_ready, verify_smoke, register_training
from .head_gate_eval_tasks import training_ready, verify_training, register_discovery, matrix_ready, select, register_full
from .finish_robust_discovery import ensure_scheduler, queue_has_seen


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only completion watcher')
    root = BASE/'watch_20260912_1350'; root.mkdir(parents=True, exist_ok=True)
    with (root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.time()
        append(root/'events.jsonl', [dict(event='start', time=start, pid=os.getpid())])
        while time.time()-start < 43200:
            phase = 'waiting for both real gradient smokes'
            if smoke_ready():
                gradient = BASE/'actual_gradient_gate_v1.json'
                if not gradient.exists(): verify_smoke()
                training = QUEUE/'additions/stage32_learned_head_gates_training.json'
                if not training.with_suffix('.ready').exists(): register_training(gradient)
                names = [j['name'] for j in json.loads(training.read_text())['jobs']]
                if not queue_has_seen(QUEUE, names): ensure_scheduler(QUEUE, root)
                phase = 'waiting for both fixed final checkpoints'
                if training_ready():
                    audit = BASE/'fixed_training_audit_v1.json'
                    if not audit.exists(): verify_training()
                    discovery = QUEUE/'additions/stage32_learned_head_gates_discovery.json'
                    if not discovery.with_suffix('.ready').exists(): register_discovery(audit)
                    names = [j['name'] for j in json.loads(discovery.read_text())['jobs']]
                    if not queue_has_seen(QUEUE, names): ensure_scheduler(QUEUE, root)
                    phase = 'waiting for complete five-input discovery matrices'
                    if matrix_ready():
                        selection = BASE/'selection_complete_discovery_v1.json'
                        if not selection.exists(): select(selection)
                        validation = QUEUE/'additions/stage32_learned_head_gates_validation.json'
                        if not validation.with_suffix('.ready').exists(): register_full(selection)
                        names = [j['name'] for j in json.loads(validation.read_text())['jobs']]
                        if queue_has_seen(QUEUE, names):
                            append(root/'events.jsonl', [dict(event='complete_selection', time=time.time(),
                                selection=str(selection), full_jobs=len(names))])
                            return
                        ensure_scheduler(QUEUE, root)
                        phase = 'waiting for frozen full jobs to be acknowledged'
            append(root/'events.jsonl', [dict(event='heartbeat', time=time.time(), phase=phase)])
            time.sleep(30)
        append(root/'events.jsonl', [dict(event='watcher_timeout', time=time.time(), limit_seconds=43200)])


if __name__ == '__main__': main()
