"""CPU-only complete-target plots and statistical consistency checks after coverage."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import append
from .head_delta_worker import BASE, sha, verify_sources


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only post-target analysis chain')
    root=BASE/'analysis_watch_20260912_1638';root.mkdir(parents=True,exist_ok=True)
    with (root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();events=root/'events.jsonl';completed={}
        if events.exists():
            for line in events.read_text().splitlines():
                e=json.loads(line)
                if e['event']=='artifact_complete':completed[e['name']]=e
        append(events,[dict(event='start',pid=os.getpid(),time=start)])
        while time.time()-start<43200:
            path=BASE/'coverage.json'
            if path.exists():
                decision=json.loads(path.read_text());verify_sources(decision['sources_sha256'])
                checkpoints={k:v['checkpoint'] for k,v in decision['artifacts'].items()}
                commands={
                    'neighboring_k_validation':['plot_head_delta_checkpoint','--checkpoint',checkpoints['validation']],
                    'neighboring_k_full':['plot_head_delta_checkpoint','--checkpoint',checkpoints['full_cohort']],
                    'frozen_center_behavior':['plot_head_delta_behavior','--checkpoints',checkpoints['full_cohort']],
                    'paired_roi':['paired_head_delta_binding','--checkpoints',checkpoints['validation'],checkpoints['full_cohort']],
                    'statistical_structure':['audit_head_delta_statistics'],
                }
                for name,arguments in commands.items():
                    if name in completed:continue
                    log=root/f'{name}_{time.time_ns()}.log'
                    command=[sys.executable,'-B','-m','mydata_bench.auto_research_addbase.'+arguments[0],*arguments[1:]]
                    append(events,[dict(event='artifact_started',name=name,time=time.time(),command=command,log=str(log))])
                    with log.open('xb') as handle:
                        subprocess.run(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=3600)
                    output=log.read_text().strip().splitlines()
                    artifacts=[line for line in output if line.startswith(str(BASE.parent)) and Path(line).exists()]
                    if len(artifacts)!=1:raise ValueError('Exactly one new output artifact is required per analysis command')
                    entry=dict(event='artifact_complete',name=name,time=time.time(),artifact=artifacts[0],log=str(log),log_sha256=sha(log))
                    append(events,[entry]);completed[name]=entry
                create_json(root/'complete.json',dict(status='pass',coverage=str(path),coverage_sha256=sha(path),artifacts=completed,
                    interpretation='Complete frozen target visualizations and arithmetic checks. Figures still require visual inspection; scientific claims require synthesis and any conditional controls.'))
                return
            end=BASE/'complete.json'
            if end.exists() and json.loads(end.read_text())['status']=='complete_failed_discovery_coverage':
                create_json(root/'complete.json',dict(status='not_applicable_no_full_targets'));return
            append(events,[dict(event='heartbeat',time=time.time(),phase='waiting_for_complete_audited_target_coverage')])
            time.sleep(30)
        append(events,[dict(event='timeout',time=time.time(),limit_seconds=43200)])


if __name__=='__main__':main()
