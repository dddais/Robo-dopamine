"""CPU-only actual-smoke and fixed native-primary-head completion chain."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .meter_factorized import QUEUE,smoke_ready,verify_smoke,register_discovery,matrix_ready,select,register_full
from .finish_robust_discovery import queue_has_seen,ensure_scheduler


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only meter gate chain')
    folder=OUT/'analysis_watch_meter_factorized_a2_20260911_2330';folder.mkdir(exist_ok=True)
    with (folder/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        append(folder/'events.jsonl',[dict(event='start',pid=os.getpid(),time=time.time())])
        audit=OUT/'audit/meter_factorized_a2_actual_gate_v1.json'
        discovery=QUEUE/'additions/stage25_meter_factorized_a2_discovery.json'
        selection=OUT/'selection_meter_factorized_a2_full_v1.json'
        validation=QUEUE/'additions/stage25_meter_factorized_a2_validation.json'
        while True:
            phase='waiting for both native meter actual smoke inputs'
            if smoke_ready():
                if not audit.exists():verify_smoke(audit)
                if not discovery.with_suffix('.ready').exists():register_discovery(audit)
                names=[j['name'] for j in json.loads(discovery.read_text())['jobs']]
                if not queue_has_seen(QUEUE,names):ensure_scheduler(QUEUE,folder)
                phase='waiting for all six complete native meter discovery inputs'
                if matrix_ready():
                    if not selection.exists():select(selection)
                    if not validation.with_suffix('.ready').exists():register_full(selection,audit)
                    names=[j['name'] for j in json.loads(validation.read_text())['jobs']]
                    if queue_has_seen(QUEUE,names):
                        append(folder/'events.jsonl',[dict(event='complete',time=time.time(),full_jobs=len(names))])
                        return
                    ensure_scheduler(QUEUE,folder)
                    phase='waiting for frozen meter full jobs to be acknowledged'
            append(folder/'events.jsonl',[dict(event='heartbeat',phase=phase,time=time.time())])
            time.sleep(30)


if __name__=='__main__':main()
