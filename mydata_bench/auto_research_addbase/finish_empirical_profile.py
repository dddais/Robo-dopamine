"""CPU-only handoff for the registered matched-profile empirical ablation."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .empirical_profile import QUEUE, profiles_ready, register_discovery, matrix_ready, select, register_full
from .finish_robust_discovery import ensure_scheduler, queue_has_seen


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This deterministic completion trigger is CPU-only')
    folder = OUT / 'analysis_watch_empirical_profile_20260911_2121'
    folder.mkdir(exist_ok=True)
    with (folder / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        append(folder / 'events.jsonl', [dict(event='start', pid=os.getpid(), time=time.time())])
        discovery = QUEUE / 'additions/stage21_empirical_profile_discovery.json'
        selection = OUT / 'selection_empiricalprofile_family_full_v1.json'
        validation = QUEUE / 'additions/stage21_empirical_profile_validation.json'
        while True:
            phase = 'waiting for both complete matched profiles and robust rankings'
            if profiles_ready():
                if not discovery.with_suffix('.ready').exists():
                    register_discovery()
                names = [j['name'] for j in json.loads(discovery.read_text())['jobs']]
                if not queue_has_seen(QUEUE, names):
                    ensure_scheduler(QUEUE, folder)
                phase = 'waiting for both complete empirical discovery matrices'
                if matrix_ready():
                    if not selection.exists():
                        select(selection)
                    if not validation.with_suffix('.ready').exists():
                        register_full(selection)
                    names = [j['name'] for j in json.loads(validation.read_text())['jobs']]
                    if queue_has_seen(QUEUE, names):
                        append(folder / 'events.jsonl', [dict(event='complete', time=time.time(),
                            selection=str(selection), full_jobs=len(names))])
                        return
                    ensure_scheduler(QUEUE, folder)
                    phase = 'waiting for frozen empirical full jobs to be acknowledged'
            append(folder / 'events.jsonl', [dict(event='heartbeat', phase=phase, time=time.time())])
            time.sleep(30)


if __name__ == '__main__':
    main()
