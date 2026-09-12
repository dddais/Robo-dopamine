"""CPU-only append-only reports after full-cohort protocol completion events."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .prepare import OUT
from .durable_scheduler import include_additions
from mydata_bench.addbase_eval.run import append


def values(command, option):
    if option not in command:
        return []
    result = []
    for token in command[command.index(option)+1:]:
        if token.startswith('--'):
            break
        result.append(token)
    return result


def experiments(queue):
    jobs = include_additions(json.loads((queue/'plan.json').read_text()),queue)
    names = set()
    for job in jobs.values():
        command = job['command']
        pipeline = command[3].endswith('.functional_pipeline')
        if not pipeline and values(command,'--population') != ['full_cohort']:
            continue
        model = values(command,'--model')[0]
        variant = 'functional' if pipeline else ''.join(values(command,'--variant'))
        for protocol in values(command,'--protocols'):
            names.add(f'{model}_{protocol}' + ('_'+variant if variant else ''))
    return names


def revision(experiment):
    path = OUT/'experiments'/experiment/'full_cohort/worker_events.jsonl'
    if not path.exists():
        return None
    content = path.read_bytes()
    try:
        rows = [json.loads(line) for line in content.splitlines()]
    except json.JSONDecodeError:
        return None  # Do not interpret a concurrently appended marker as final.
    if not rows or rows[-1].get('event') != 'complete':
        return None
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-experiments',nargs='*',default=[])
    parser.add_argument('--queue',default='queue_gpu01_20260911_1700')
    parser.add_argument('--name',required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name or Path(args.queue).name != args.queue:
        raise ValueError('Invalid session-local name')
    queue = OUT/args.queue
    root = OUT/args.name
    root.mkdir(exist_ok=True)
    lock = (root/'watcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    events = root/'events.jsonl'
    known = {name:revision(name) for name in experiments(queue)}
    for name in args.initial_experiments:
        if name not in known:
            raise ValueError('Initial experiment not present in frozen queue')
        known[name] = None
    append(events,[dict(event='watcher_start',time=time.time(),pid=os.getpid(),initial_revisions=known,
                        intent='CPU reports only; does not select candidates, modify inference, or declare goal completion')])
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES='')
    started = time.time()
    while time.time()-started < 43200:
        for experiment in sorted(experiments(queue)):
            current = revision(experiment)
            if current is None or current == known.get(experiment):
                continue
            record = dict(experiment=experiment,worker_events_sha256=current,time=time.time())
            append(events,[dict(record,event='analysis_start')])
            checkpoints = []
            try:
                for population in ['validation','full_cohort','old_holdout']:
                    name = time.strftime('checkpoint_%Y%m%d_%H%M%S') + f'_{population}_auto_{time.time_ns()}'
                    command = [sys.executable,'-B','-m','mydata_bench.auto_research_addbase.analyze',
                               '--population','full_cohort','--uncertainty','--experiments',experiment,'--name',name]
                    if population != 'full_cohort':command += ['--score-split',population]
                    subprocess.run(command,check=True,env=environment,timeout=1200)
                    checkpoints.append(str(OUT/'analysis'/name))
                export_name = time.strftime('exports_%Y%m%d_%H%M%S') + f'_auto_{time.time_ns()}'
                subprocess.run([sys.executable,'-B','-m','mydata_bench.auto_research_addbase.export_tables',
                                '--checkpoint',checkpoints[1],'--name',export_name],check=True,env=environment,timeout=1200)
                append(events,[dict(record,event='analysis_complete',completed_at=time.time(),
                                    checkpoints=checkpoints,export_name=export_name)])
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
                append(events,[dict(record,event='analysis_failed',error=str(exc),checkpoints=checkpoints)])
            known[experiment] = current
        append(events,[dict(event='heartbeat',time=time.time(),known_revisions=known)])
        time.sleep(30)
    append(events,[dict(event='watcher_timeout',time=time.time(),limit_seconds=43200)])


if __name__ == '__main__':
    main()
