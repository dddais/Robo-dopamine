"""CPU-only post-control consistency audit and figures; never closes the research goal."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import append
from .head_delta_worker import BASE, sha


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This chain must run without GPU visibility')
    root = BASE/'control_analysis_watch_20260912_1715'; root.mkdir(parents=True, exist_ok=True)
    with (root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = root/'events.jsonl'; completed = {}; start = time.time()
        if events.exists():
            for line in events.read_text().splitlines():
                row = json.loads(line)
                if row['event'] == 'artifact_complete': completed[row['name']] = row
        append(events, [dict(event='start', pid=os.getpid(), time=start)])
        while time.time()-start < 43200:
            path = BASE/'complete.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result['status'] != 'complete_full_target_and_controls':
                    raise ValueError('Target/control chain ended without complete controls')
                for name in ['synthesis', 'validation_figures', 'full_figures']:
                    if name in completed: continue
                    module = 'summarize_head_delta_controls' if name == 'synthesis' else 'plot_head_delta_controls'
                    args = [] if name == 'synthesis' else ['--synthesis', completed['synthesis']['artifact'],
                        '--population', 'validation' if name == 'validation_figures' else 'full_cohort']
                    command = [sys.executable, '-B', '-m', 'mydata_bench.auto_research_addbase.'+module, *args]
                    log = root/f'{name}_{time.time_ns()}.log'
                    append(events, [dict(event='artifact_started', name=name, command=command, log=str(log), time=time.time())])
                    with log.open('xb') as handle:
                        subprocess.run(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                            check=True, timeout=3600)
                    artifacts = [line for line in log.read_text().splitlines()
                        if line.startswith(str(BASE.parent)) and Path(line).exists()]
                    if len(artifacts) != 1: raise ValueError('Exactly one newly completed artifact expected')
                    item = dict(event='artifact_complete', name=name, artifact=artifacts[0],
                        log=str(log), log_sha256=sha(log), time=time.time())
                    append(events, [item]); completed[name] = item
                create_json(root/'complete.json', dict(status='pass', input_complete=str(path),
                    input_complete_sha256=sha(path), artifacts=completed,
                    interpretation='Arithmetic comparisons and plots are complete. Visual review and final scientific synthesis remain manual.'))
                return
            append(events, [dict(event='heartbeat', phase='waiting_for_all_61k_controls_and_final_reports', time=time.time())])
            time.sleep(30)
        append(events, [dict(event='timeout', limit_seconds=43200, time=time.time())])


if __name__ == '__main__':
    main()
