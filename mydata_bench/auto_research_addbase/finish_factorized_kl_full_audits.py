"""Bounded CPU audit handoff for all explicitly frozen round27 neighborhoods."""
import fcntl
import json
import os
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import append, latest
from .prepare import OUT
from .empirical_profile import sha
from .factorized_kl_tasks import SELECTION, selected, variant
from .verify_factorized_kl_full import verify


def ready(point, budget, ids):
    root = OUT/'experiments'/f"{point['model']}_{point['protocol']}_{variant(budget)}"/'full_cohort'
    files = [root/'predictions/baseline.jsonl'] + [root/'binding_transport_s4/predictions'/f"{point['scope']}_target_{k}.jsonl" for k in point['ks']]
    return all(p.exists() and set(latest(p)) == ids for p in files)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('This audit chain is CPU-only')
    folder = OUT/'analysis_watch_factorized_kl_full_audits_20260912_0128'; folder.mkdir(exist_ok=True)
    with (folder/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        record, budget = selected(); selection_sha = sha(SELECTION)
        ids = set(json.loads((OUT/'splits.json').read_text())['full_cohort'])
        if len(ids) != 846 or len(record['selected']) != 6: raise ValueError('Changed round27 full matrix')
        completed = {}; start = time.time()
        events = folder/'events.jsonl'
        if events.exists():
            for line in events.read_text().splitlines():
                e = json.loads(line)
                if e['event'] == 'input_verified':
                    if sha(e['audit']) != e['audit_sha256']: raise ValueError('Prior audit changed')
                    completed[e['input']] = e['audit']
        append(events, [dict(event='start', pid=os.getpid(), time=start, labels_read=False)])
        while time.time()-start < 43200:
            if sha(SELECTION) != selection_sha: raise ValueError('Frozen selection changed while waiting')
            for p in record['selected']:
                name = f"{p['model']}/{p['protocol']}"
                if name in completed or not ready(p, budget, ids): continue
                audit = verify(p['model'], p['protocol'])
                completed[name] = str(audit)
                append(events, [dict(event='input_verified', input=name, audit=str(audit), audit_sha256=sha(audit), time=time.time())])
            if len(completed) == 6:
                create_json(folder/'complete_audit_index.json', dict(status='pass', labels_read=False,
                    selection=str(SELECTION), selection_sha256=selection_sha, audits=completed,
                    interpretation='All six fixed full neighborhoods have verified actual branches, KL formula and same inputs; no efficacy or goal-completion claim'))
                append(events, [dict(event='complete', time=time.time(), inputs=6)])
                return
            append(events, [dict(event='heartbeat', time=time.time(), verified=list(completed))])
            time.sleep(30)
        append(events, [dict(event='watcher_timeout', time=time.time(), limit_seconds=43200)])


if __name__ == '__main__': main()
