"""Run the frozen matrix on GPUs 0/1/2, with process and timeout monitoring."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml
from mydata_bench.top_eval.versioning import validate_output_directory
from .prepare import OUT, ROOT, create_json


def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        return stat[2]!='Z'
    except FileNotFoundError:return False


def events(path):
    if not path.exists():return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adopt',action='append',default=[])
    args=parser.parse_args()
    queue=[]
    for p in json.loads((OUT/'matrix.json').read_text()):
        cfg=yaml.safe_load(Path(p).read_text())
        validate_output_directory(cfg, Path(cfg['output_dir']))
        queue.append((Path(p).stem,p,cfg))
    # Native official baselines get an early slot; no choosing jobs by their observed scores.
    queue.sort(key=lambda x:(x[0]!='sole_official',x[2]['model']!='meter',x[0]!='meter_official',x[0]))
    slots={};launched=set();finished={}
    for name,path,cfg in queue:
        if any(r['event']=='complete' for r in events(OUT/name/'events.jsonl')):
            launched.add(name);finished[name]={'exit_code':None,'complete':True,'adopted_completed':True}
    for item in args.adopt:
        gpu,name,pid=item.split(':');slots[int(gpu)]={'name':name,'pid':int(pid),'process':None,'start':time.monotonic()}
        launched.add(name)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for base in ['mydata_bench/addbase_eval','mydata_bench/meter_eval','mydata_bench/top_eval']
              for p in (ROOT/base).glob('*.py')}
    create_json(OUT/'research'/f'implementation_snapshot_{int(time.time())}.json',manifest)
    log=OUT/'research/scheduler_events.jsonl'
    with log.open('a') as f:f.write(json.dumps({'event':'scheduler_started','pid':os.getpid(),'time':time.time()})+'\n')
    while True:
        for gpu,job in list(slots.items()):
            if not alive(job['pid']):
                code=job['process'].poll() if job['process'] is not None else None
                es=events(OUT/job.get('output_name',job['name'])/'events.jsonl')
                complete=any(r['event']=='complete' for r in es)
                finished[job['name']]={'exit_code':code,'complete':complete}
                with log.open('a') as f:f.write(json.dumps({'event':'finished','gpu':gpu,'name':job['name'],'result':finished[job['name']],'time':time.time()})+'\n')
                if job.get('handle'):job['handle'].close()
                del slots[gpu]
            elif time.monotonic()-job['start']>72*3600:
                with log.open('a') as f:f.write(json.dumps({'event':'hard_timeout_72h','gpu':gpu,'name':job['name'],'pid':job['pid'],'time':time.time()})+'\n')
                os.kill(job['pid'],15)
        for gpu in [0,1,2]:
            if gpu in slots:continue
            candidate=next((q for q in queue if q[0] not in launched),None)
            helper=False
            if candidate is None:
                helper_name=f'sole_official_helper_gpu{gpu}'
                if helper_name in launched or not all((OUT/'sole_official'/f'ranking_{s}.json').exists() for s in ['last_frame','all_frames']):continue
                if not any(j.get('output_name',j['name'])=='sole_official' for j in slots.values()):continue
                _,path,cfg=next(q for q in queue if q[0]=='sole_official')
                name=helper_name;helper=True
            else:
                name,path,cfg=candidate
            count=4 if name=='meter_official' else 16 if cfg['model']=='meter' else 32
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
            command=[sys.executable,'-u','-m','mydata_bench.addbase_eval.run','--config',path,'--batch-size',str(count),'--retry-runtime-errors']
            if helper:command+=['--phase','steer']
            handle=(OUT/'research'/f'{name}_full.log').open('a')
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            slots[gpu]={'name':name,'output_name':Path(cfg['output_dir']).name,'pid':proc.pid,'process':proc,'start':time.monotonic(),'handle':handle}
            launched.add(name)
            with log.open('a') as f:f.write(json.dumps({'event':'started','gpu':gpu,'name':name,'pid':proc.pid,'command':command,'time':time.time()})+'\n')
        state={'time':time.time(),'running':{g:{'name':j['name'],'pid':j['pid'],'alive':alive(j['pid'])} for g,j in slots.items()},
               'finished':finished,'pending':[q[0] for q in queue if q[0] not in launched]}
        with (OUT/'research/scheduler_status.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        if not slots:break
        time.sleep(30)
    create_json(OUT/'research/scheduler_completion.json',state)


if __name__=='__main__':main()
