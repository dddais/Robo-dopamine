"""CPU-only deterministic handoff through the already registered round20 gates."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .verify_temporal import verify, register as register_discovery
from .select_temporal_discovery import completed, matrix_complete, select, register as register_full
from .finish_robust_discovery import queue_has_seen, ensure_scheduler


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Temporal completion trigger is CPU-only')
    folder=OUT/'analysis_watch_temporal_20260911_2045';folder.mkdir(exist_ok=True)
    queue=OUT/'queue_gpu01_20260911_1700'
    with (folder/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        append(folder/'events.jsonl',[dict(event='start',pid=os.getpid(),time=time.time())])
        audit=OUT/'audit/temporal_planes_actual_gate_v1.json'
        discovery=queue/'additions/stage20_temporal_planes_discovery.json'
        selection=OUT/'selection_temporal_family_full_v1.json'
        validation=queue/'additions/stage20_temporal_planes_validation.json'
        while True:
            smoke_ready=all(completed(OUT/'experiments'/f'{m}_{p}_uniform_temporal_evidence_a1'/'discovery_smoke')
                            for m in ['qwen','roboreward'] for p in ['image_text','text_video'])
            phase='waiting for both actual smoke matrices'
            if smoke_ready:
                if not audit.exists():
                    verify(audit)
                if not discovery.with_suffix('.ready').exists():
                    register_discovery(audit)
                names=[j['name'] for j in json.loads(discovery.read_text())['jobs']]
                if not queue_has_seen(queue,names):
                    ensure_scheduler(queue,folder)
                phase='waiting for both five-input discovery matrices'
                if matrix_complete():
                    if not selection.exists():
                        select(selection)
                    if not validation.with_suffix('.ready').exists():
                        register_full(selection,audit)
                    names=[j['name'] for j in json.loads(validation.read_text())['jobs']]
                    if queue_has_seen(queue,names):
                        append(folder/'events.jsonl',[dict(event='complete',time=time.time(),
                            selection=str(selection),full_jobs=len(names))])
                        return
                    ensure_scheduler(queue,folder)
                    phase='waiting for frozen full jobs to be acknowledged'
            append(folder/'events.jsonl',[dict(event='heartbeat',phase=phase,time=time.time())])
            time.sleep(30)


if __name__=='__main__':
    main()
