"""CPU-only actual-gate handoff for the already frozen shared KL-budget family."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .factorized_kl_tasks import QUEUE,smoke_ready,verify_smoke,register_full
from .finish_robust_discovery import queue_has_seen,ensure_scheduler


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only gate chain')
    folder=OUT/'analysis_watch_factorized_kl_20260912_0047';folder.mkdir(exist_ok=True)
    with (folder/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        append(folder/'events.jsonl',[dict(event='start',pid=os.getpid(),time=time.time())])
        audit=OUT/'audit/factorized_kl_actual_gate_v1.json'
        addition=QUEUE/'additions/stage27_factorized_kl_validation.json'
        started=time.time()
        while time.time()-started<43200:
            phase='waiting for both complete selected-budget actual smoke matrices'
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
        append(folder/'events.jsonl',[dict(event='watcher_timeout',time=time.time(),limit_seconds=43200)])


if __name__=='__main__':main()
