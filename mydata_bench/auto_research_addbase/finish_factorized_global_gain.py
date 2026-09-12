"""CPU-only actual-gate handoff for the already frozen shared gain family."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .factorized_gain_tasks import QUEUE,smoke_ready,verify_smoke,register_full
from .finish_robust_discovery import queue_has_seen,ensure_scheduler


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only gate chain')
    folder=OUT/'analysis_watch_factorized_global_gain_20260911_2315';folder.mkdir(exist_ok=True)
    with (folder/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        append(folder/'events.jsonl',[dict(event='start',pid=os.getpid(),time=time.time())])
        audit=OUT/'audit/factorized_global_gain_actual_gate_v1.json'
        addition=QUEUE/'additions/stage24_factorized_global_gain_validation.json'
        while True:
            phase='waiting for both complete selected-gain actual smoke matrices'
            if smoke_ready():
                if not audit.exists():verify_smoke(audit)
                if not addition.with_suffix('.ready').exists():register_full(audit)
                names=[j['name'] for j in json.loads(addition.read_text())['jobs']]
                if queue_has_seen(QUEUE,names):
                    append(folder/'events.jsonl',[dict(event='complete',time=time.time(),full_jobs=len(names))])
                    return
                ensure_scheduler(QUEUE,folder)
                phase='waiting for frozen full jobs to be acknowledged'
            append(folder/'events.jsonl',[dict(event='heartbeat',phase=phase,time=time.time())])
            time.sleep(30)


if __name__=='__main__':main()
