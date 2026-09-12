"""Small read-only status view; process release is never an efficacy claim."""
import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extra-pid',nargs='*',type=int,default=[])
    parser.add_argument('--queue',default='queue_gpu01_20260911_1700')
    args=parser.parse_args()
    root=OUT/args.queue
    if not root.resolve().is_relative_to(OUT.resolve()):raise ValueError('Wrong research session')
    plan=json.loads((root/'plan.json').read_text())
    jobs={j['name']:j for j in plan['jobs']}
    for marker in sorted((root/'additions').glob('*.ready')):
        for job in json.loads(marker.with_suffix('.json').read_text())['jobs']:
            jobs[job['name']]=job
    states={name:{'event':'pending'} for name in jobs}
    scheduler=None;heartbeat=None
    for line in (root/'events.jsonl').read_text().splitlines():
        try:event=json.loads(line)
        except json.JSONDecodeError:continue  # A concurrent writer may be finishing its last line.
        if event['event']=='scheduler_start':scheduler=event
        if event['event']=='heartbeat':heartbeat=event
        if event.get('job') in states and event['event'] in {'started','adopted','released','failed_to_start'}:
            states[event['job']]=event
    now=time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    if scheduler:
        alive=Path(f"/proc/{scheduler['pid']}/cmdline").exists()
        print('scheduler',scheduler['pid'],'present',alive,'heartbeat age seconds',round(now-heartbeat['time'],1) if heartbeat else None)
    for name,state in states.items():
        if state['event'] not in {'started','adopted'}:continue
        pid=state['pid'];base=Path(f'/proc/{pid}')
        try:
            command=base.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
            stat=base.joinpath('stat').read_text().rsplit(')',1)[1].split()
            alive=(stat[0] not in {'Z','X'} and stat[19]==state['token'] and command==jobs[name]['command'])
            children=base.joinpath('task',str(pid),'children').read_text().strip()
        except FileNotFoundError:alive=False;children=''
        last='';age=None;path=Path(state['log']) if state.get('log') else None
        if path and path.exists() and path.resolve().is_relative_to(OUT):
            age=round(now-path.stat().st_mtime,1)
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-2048))
                lines=f.read().decode(errors='replace').splitlines()
                if lines:last=lines[-1][-180:]
        print(name,'gpu',state.get('gpu',jobs[name]['gpu']),'pid',pid,'alive',alive,'children',children,'log_age_s',age,last)
        for child in children.split():
            try:child_command=Path(f'/proc/{child}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
            except (FileNotFoundError,ProcessLookupError):continue
            if 'mydata_bench.auto_research_addbase.matched_profile_worker' not in child_command:continue
            model=child_command[child_command.index('--model')+1]
            files=list((OUT/'experiments').glob(f'{model}_*_uniform_method_profile/discovery/binding_transport_s4_layer*/predictions/*.jsonl'))
            if files:
                latest_file=max(files,key=lambda p:p.stat().st_mtime)
                print('  current layer probe:',latest_file.parents[3].name,latest_file.parents[1].name,latest_file.stem)
    print('pending:',', '.join(name for name,state in states.items() if state['event']=='pending'))
    print('released:',sum(s['event']=='released' for s in states.values()),'failed_to_start:',sum(s['event']=='failed_to_start' for s in states.values()))
    for pid in args.extra_pid:
        base=Path(f'/proc/{pid}')
        try:
            command=base.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if base.joinpath('cwd').resolve()!=ROOT or 'mydata_bench.auto_research_addbase.worker' not in command:
                print('extra pid',pid,'does not match a worker in this target repository');continue
            path=base.joinpath('fd/1').resolve()
            last=''
            if path.is_relative_to(OUT) and path.is_file():
                with path.open('rb') as f:
                    f.seek(max(0,path.stat().st_size-2048))
                    lines=f.read().decode(errors='replace').splitlines()
                    if lines:last=lines[-1][-180:]
            print('legacy worker',pid,'model',command[command.index('--model')+1],last)
        except FileNotFoundError:print('extra pid',pid,'absent; review completion separately')
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in output.splitlines():
        if int(line.split(',')[0]) in [0,1]:print('GPU index, used_MB, free_MB, utilization_pct:',line)


if __name__=='__main__':main()
