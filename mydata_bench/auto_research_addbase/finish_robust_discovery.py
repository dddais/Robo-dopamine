"""CPU-only completion trigger for the prespecified round18 full selection."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .select_robust_discovery import matrix_complete, register, select


def queue_has_seen(root, names):
    seen = set()
    for line in (root / 'events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('job'):
            seen.add(event['job'])
        if event['event'] == 'heartbeat':
            seen.update(event['states'])
    return set(names) <= seen


def ensure_scheduler(root, watcher_folder):
    """An idle queue can finish before a delayed CPU selector registers jobs."""
    if root.resolve() != (OUT / 'queue_gpu01_20260911_1700').resolve():
        raise ValueError('Wrong authorized queue')
    with (root / 'scheduler.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None  # The existing scheduler will consume the addition.
        fcntl.flock(lock, fcntl.LOCK_UN)
    history = []
    for line in (root / 'events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event['event'] == 'scheduler_start':
            history.append(event)
    if not history:
        raise ValueError('No existing authorized scheduler configuration')
    previous = history[-1]
    command = [sys.executable, '-B', '-m', 'mydata_bench.auto_research_addbase.durable_scheduler',
               '--plan', str(root / 'plan.json')]
    if previous.get('flexible_gpus'):
        command.append('--flexible-gpus')
    command += ['--priority-jobs', *previous.get('priority_jobs', [])]
    log_path = watcher_folder / f'scheduler_resume_{time.time_ns()}.log'
    with log_path.open('xb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), cwd=Path(__file__).resolve().parents[2])
    append(watcher_folder / 'events.jsonl', [dict(event='resume_idle_scheduler', time=time.time(),
        pid=child.pid, command=command, log=str(log_path), reason='New frozen jobs registered after scheduler released its lock')])
    return child.pid


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This CPU-only completion trigger requires empty GPU visibility')
    folder = OUT / 'analysis_watch_robust_selection_20260911_2003'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        append(folder / 'events.jsonl', [dict(event='start', pid=os.getpid(), time=time.time())])
        pending = ['qwen', 'roboreward']
        while pending:
            for model in list(pending):
                if not matrix_complete(model):
                    continue
                selection = OUT / f'selection_robustprofile_{model}_full_v1.json'
                addition = OUT / 'queue_gpu01_20260911_1700/additions' / f'stage18_{model}_robustprofile_validation.json'
                append(folder / 'events.jsonl', [dict(event='complete_matrix', model=model, time=time.time())])
                if not selection.exists():
                    select(model, selection)
                if not addition.with_suffix('.ready').exists():
                    register(model, selection)
                names = [job['name'] for job in json.loads(addition.read_text())['jobs']]
                queue = addition.parent.parent
                if not queue_has_seen(queue, names):
                    ensure_scheduler(queue, folder)
                    continue
                pending.remove(model)
                append(folder / 'events.jsonl', [dict(event='registered', model=model, time=time.time(),
                    selection=str(selection), inputs=len(json.loads(selection.read_text())['selected']))])
            append(folder / 'events.jsonl', [dict(event='heartbeat', pending=pending, time=time.time())])
            if pending:
                time.sleep(30)
        append(folder / 'events.jsonl', [dict(event='complete', time=time.time())])


if __name__ == '__main__':
    main()
