"""Run a frozen experiment after another owned worker exits; append-only log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def start_token(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--worker-module',choices=['worker','profile_worker','functional_pipeline'],default='worker')
    p.add_argument('worker_args',nargs=argparse.REMAINDER)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]/'results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/logs'
    command=[sys.executable,'-B','-m','mydata_bench.auto_research_addbase.'+args.worker_module]
    command+=args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
    record=root/f'{args.name}_queue.jsonl'
    with record.open('x') as f:
        f.write(json.dumps({'event':'queued','time':time.time(),'after_pid':args.after_pid,'gpu':args.gpu,'command':command})+'\n')
    token=start_token(args.after_pid)
    while token is not None and start_token(args.after_pid)==token:time.sleep(10)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu))
    with (root/f'{args.name}.log').open('x') as output:
        process=subprocess.Popen(command,env=environment,stdout=output,stderr=subprocess.STDOUT)
        with record.open('a') as f:f.write(json.dumps({'event':'started','pid':process.pid,'time':time.time()})+'\n')
        try:code=process.wait(timeout=43200)
        except subprocess.TimeoutExpired:
            print('Hard 12-hour experiment timeout; stopping owned worker',flush=True)
            process.terminate();code=process.wait(timeout=60)
    with record.open('a') as f:f.write(json.dumps({'event':'finished','returncode':code,'time':time.time()})+'\n')
    raise SystemExit(code)

if __name__=='__main__':main()
