"""Ordered profiling, CPU selection, and frozen evaluation; fail on any phase error."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .prepare import OUT
from mydata_bench.addbase_eval.run import append
from mydata_bench.addbase_eval.prepare import create_json


class _PipelineStop(BaseException):
    pass


def wait_for_child(process):
    """Unwind Popen.wait's lock before reaping a signalled child.

    Calling wait again inside its own signal handler can deadlock on Python's
    non-reentrant waitpid lock. Cleanup belongs outside that interrupted wait.
    """
    def stop(signum,frame):
        raise _PipelineStop(signum)
    previous={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        return process.wait()
    except _PipelineStop as interruption:
        for s in previous:signal.signal(s,signal.SIG_IGN)
        process.terminate()
        try:process.wait(timeout=50)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise SystemExit(128+interruption.args[0])
    finally:
        for s,handler in previous.items():signal.signal(s,handler)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['qwen','roboreward','meter'],required=True)
    p.add_argument('--protocols',nargs='+',required=True)
    p.add_argument('--selection-name',required=True)
    args=p.parse_args()
    if Path(args.selection_name).name!=args.selection_name:raise ValueError('Invalid selection name')
    record=OUT/'functional_selections'/f'{args.selection_name}_pipeline'
    intent={'arguments':vars(args),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'fixed_evaluation':'binding_transport, lambda4, task fraction .5, k8/16/32/64, all/last, all stated protocols',
        'supervision':'discovery labels select heads; no validation labels used for ranking; model weights frozen'}
    create_json(record/'intent.json',intent)
    append(record/'events.jsonl',[{'event':'attempt','time':time.time(),'pid':os.getpid()}])
    common=['--model',args.model,'--protocols',*args.protocols]
    def run(module,arguments):
        command=[sys.executable,'-B','-m','mydata_bench.auto_research_addbase.'+module,*arguments]
        append(record/'events.jsonl',[{'event':'start','time':time.time(),'command':command}])
        process=subprocess.Popen(command)
        try:
            code=wait_for_child(process)
            if code:raise subprocess.CalledProcessError(code,command)
        except SystemExit as exc:
            append(record/'events.jsonl',[{'event':'interrupted','time':time.time(),'command':command,'exit_code':exc.code}])
            raise
        append(record/'events.jsonl',[{'event':'complete','time':time.time(),'command':command}])
    run('profile_worker',common)
    run('select_profiles',common+['--name',args.selection_name])
    inference=common+['--variant','functional','--methods','binding_transport',
        '--task-binding-fraction','.5','--contrast-weight','0','--negative-strength','4',
        '--strengths','4','--ks','8','16','32','64',
        '--frozen-ranking-root',str(OUT/'functional_selections'/args.selection_name)]
    if args.model!='meter':inference+=['--ranking-prefix','ANSWER: ']
    run('worker',inference+['--population','discovery'])
    run('worker',inference+['--population','full_cohort'])


if __name__=='__main__':main()
