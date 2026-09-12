"""Detached, append-only resource queue that can adopt surviving research workers.

Dependencies release GPU slots; a released process is NOT an efficacy or success
claim. Prediction completeness and worker/pipeline completion remain separate.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

# Scheduling and NVML queries need no CUDA context; children receive their
# explicitly authorized GPU in a separate environment below.
os.environ['CUDA_VISIBLE_DEVICES']=''

from .prepare import OUT
from .gpu_policy import ALLOWED_GPUS
from mydata_bench.addbase_eval.run import append


def process_info(pid):
    try:
        # Executable names may contain spaces; fields begin after the last ')'.
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        command=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
        return {'pid':pid,'state':stat[0],'token':stat[19],'command':command}
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def alive(pid,token):
    info=process_info(pid)
    return bool(info and info['state'] not in {'Z','X'} and info['token']==token)


def matching_process(command):
    matches=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        info=process_info(int(path.name))
        if info and info['state'] not in {'Z','X'} and info['command']==command:matches.append(info)
    if len(matches)>1:raise RuntimeError('Multiple existing workers for the same queued command')
    return matches[0] if matches else None


def validate_plan(plan):
    jobs={job['name']:job for job in plan['jobs']}
    if len(jobs)!=len(plan['jobs']):raise ValueError('Duplicate job identity')
    allowed={'mydata_bench.auto_research_addbase.worker','mydata_bench.auto_research_addbase.functional_pipeline',
             'mydata_bench.auto_research_addbase.matched_profile_worker',
             'mydata_bench.auto_research_addbase.matched_profile_pipeline',
             'mydata_bench.auto_research_addbase.learned_gate_worker',
             'mydata_bench.auto_research_addbase.head_gate_worker',
             'mydata_bench.auto_research_addbase.head_gate_inference',
             'mydata_bench.auto_research_addbase.head_delta_worker',
             'mydata_bench.auto_research_addbase.head_delta_inference'}
    for job in jobs.values():
        command=job['command']
        if command[:3]!=['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m'] or command[3] not in allowed:
            raise ValueError('Only already authorized research modules may run')
        if job['gpu'] not in ALLOWED_GPUS:raise ValueError('Forbidden GPU: user permits only physical devices 0 and 1')
        if not set(job.get('depends_on',[]))<=set(jobs):raise ValueError('Unknown dependency')
    pending=set(jobs);done=set()
    while pending:
        ready={name for name in pending if set(jobs[name].get('depends_on',[]))<=done}
        if not ready:raise ValueError('Dependency cycle')
        pending-=ready;done|=ready
    return jobs


def free_memory():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(line.split(',')[0]):int(line.split(',')[1]) for line in result.splitlines()}


def descendant_pids(pid):
    found=set();pending=[pid]
    while pending:
        current=pending.pop()
        if current in found:continue
        found.add(current)
        try:pending += [int(p) for p in Path(f'/proc/{current}/task/{current}/children').read_text().split()]
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return found


def gpu_process_memory():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'],text=True)
    usage={}
    for line in output.splitlines():
        pid,amount=[part.strip() for part in line.split(',',1)]
        if pid.isdigit() and amount.isdigit():usage[int(pid)]=usage.get(int(pid),0)+int(amount)
    return usage


def reserve_pipeline_memory(memory,jobs,states,usage,descendants=descendant_pids):
    """Keep a pipeline's slot through CPU selection and model reload gaps.

    The physical free-memory query alone sees a temporary release when a model
    child exits. Reserving only the unallocated part avoids counting its live
    model allocation twice, while preserving room for its next model child.
    """
    effective=dict(memory);holds={}
    for name,state in states.items():
        if state['status'] not in {'started','adopted'}:continue
        job=jobs[name]
        if not job['command'][3].endswith(('.functional_pipeline','.matched_profile_pipeline')):continue
        gpu=state['gpu']
        if gpu not in ALLOWED_GPUS:raise ValueError('Forbidden GPU in active pipeline reservation')
        allocated=sum(usage.get(pid,0) for pid in descendants(state['pid']))
        state['pipeline_peak_allocated_mb']=max(state.get('pipeline_peak_allocated_mb',0),allocated)
        target=state['pipeline_peak_allocated_mb'] or job['min_free_mb']
        held=max(0,target-allocated)
        effective[gpu]=max(0,effective.get(gpu,0)-held)
        holds[name]={'gpu':gpu,'allocated_mb':allocated,'reload_target_mb':target,'reserved_unallocated_mb':held}
    return effective,holds


def choose_gpu(job,memory,last_launch,now,flexible=False):
    candidates=ALLOWED_GPUS if flexible else [job['gpu']]
    eligible=[gpu for gpu in candidates if gpu in ALLOWED_GPUS and
              memory.get(gpu,0)>=job['min_free_mb'] and now-last_launch.get(gpu,0)>=90]
    return max(eligible,key=lambda gpu:(memory[gpu],gpu==job['gpu'],-gpu)) if eligible else None


def process_gpu(pid):
    environment=dict(item.split('=',1) for item in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in item)
    value=environment.get('CUDA_VISIBLE_DEVICES')
    if value not in {'0','1'}:raise ValueError('Existing research worker is not restricted to physical GPU0/1')
    return int(value)


def include_additions(plan,root):
    """Read only complete, immutable additions with a separate ready marker."""
    combined=dict(plan,jobs=list(plan['jobs']))
    for marker in sorted((root/'additions').glob('*.ready')):
        addition=json.loads(marker.with_suffix('.json').read_text())
        combined['jobs'].extend(addition['jobs'])
    return validate_plan(combined)


def restore_pipeline_peak(state,recorded):
    """Carry an observed reload reservation through a scheduler-only restart."""
    if recorded.get('gpu')==state['gpu'] and recorded.get('reload_target_mb',0)>0:
        state['pipeline_peak_allocated_mb']=recorded['reload_target_mb']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--priority-jobs',nargs='*',default=[],help='Scheduling order only; commands and dependencies stay frozen')
    p.add_argument('--flexible-gpus',action='store_true',help='Use either allowed physical GPU for pending jobs, recording actual assignment')
    args=p.parse_args()
    if not args.plan.resolve().is_relative_to(OUT.resolve()):raise ValueError('Wrong research session')
    plan=json.loads(args.plan.read_text())
    root=args.plan.parent
    jobs=include_additions(plan,root)
    if not set(args.priority_jobs)<=set(jobs):raise ValueError('Unknown priority job')
    priority={name:i for i,name in enumerate(args.priority_jobs)}
    root.mkdir(exist_ok=True)
    lock=(root/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    events=root/'events.jsonl'
    history=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    historical_holds={}
    for event in history:
        if event['event']=='heartbeat' and isinstance(event.get('pipeline_memory_holds'),dict):
            historical_holds=event['pipeline_memory_holds']
    states={name:{'status':'pending'} for name in jobs}
    for event in history:
        if event.get('job') in states and event['event'] in {'started','adopted','released','failed_to_start'}:
            states[event['job']]=dict(event,status=event['event'])
    processes={};outputs={};last_launch={};last_heartbeat=0.
    append(events,[{'event':'scheduler_start','pid':os.getpid(),'time':time.time(),'plan':str(args.plan.resolve()),
                   'priority_jobs':args.priority_jobs,'flexible_gpus':args.flexible_gpus,
                   'pipeline_memory_holds':True}])
    for name,job in jobs.items():
        if states[name]['status'] in {'released','failed_to_start'}:continue
        previous=states[name]
        info=matching_process(job['command'])
        if info:
            gpu=process_gpu(info['pid'])
            state={'event':'adopted','job':name,'pid':info['pid'],'token':info['token'],
                   'gpu':gpu,
                   'time':time.time(),'started_at':previous.get('started_at',job.get('original_started_at',time.time())),
                   'log':previous.get('log')}
            restore_pipeline_peak(state,historical_holds.get(name,{}))
            states[name]=dict(state,status='adopted');append(events,[state])
            last_launch[gpu]=max(last_launch.get(gpu,0),state['started_at'])
        elif previous['status'] in {'started','adopted'} or job.get('was_started'):
            state={'event':'released','job':name,'time':time.time(),'exit_code':None,
                   'reason':'Recorded process is absent; outputs require independent completeness review'}
            states[name]=dict(state,status='released');append(events,[state])
    while True:
        now=time.time()
        revised=include_additions(plan,root)
        for name,job in revised.items():
            if name not in jobs:
                jobs[name]=job;states[name]={'status':'pending'}
                append(events,[{'event':'registered_addition','job':name,'time':now}])
        for name,state in list(states.items()):
            if state['status'] not in {'started','adopted'}:continue
            if not alive(state['pid'],state['token']):
                code=processes[name].wait() if name in processes else None
                event={'event':'released','job':name,'time':now,'exit_code':code,
                       'reason':'Process terminated; success must also be checked in worker outputs'}
                states[name]=dict(event,status='released');append(events,[event])
                if name in outputs:outputs.pop(name).close()
            elif now-state['started_at']>43200 and not state.get('timeout_sent'):
                # PID identity and exact command were checked before adoption.
                try:os.kill(state['pid'],signal.SIGTERM)
                except ProcessLookupError:continue
                state['timeout_sent']=now
                append(events,[{'event':'timeout_term','job':name,'time':now,'pid':state['pid']}])
            elif state.get('timeout_sent') and now-state['timeout_sent']>55:
                try:os.kill(state['pid'],signal.SIGKILL)
                except ProcessLookupError:pass
        memory,pipeline_holds=reserve_pipeline_memory(free_memory(),jobs,states,gpu_process_memory())
        for name,job in sorted(jobs.items(),key=lambda item:priority.get(item[0],len(priority))):
            if states[name]['status']!='pending':continue
            if any(states[d]['status'] not in {'released','failed_to_start'} for d in job.get('depends_on',[])):continue
            if any(alive(x['pid'],x['token']) for x in job.get('external_waits',[])):continue
            gpu=choose_gpu(job,memory,last_launch,now,args.flexible_gpus)
            if gpu is None:continue
            existing=matching_process(job['command'])
            if existing:
                event={'event':'adopted','job':name,'pid':existing['pid'],'token':existing['token'],
                       'time':now,'started_at':now,'gpu':process_gpu(existing['pid'])}
                states[name]=dict(event,status='adopted');append(events,[event]);continue
            path=root/'logs'/f'{name}_{time.time_ns()}.log';path.parent.mkdir(exist_ok=True)
            output=path.open('x')
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            try:
                process=subprocess.Popen(job['command'],cwd=plan['cwd'],env=environment,
                    stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                info=process_info(process.pid)
                if not info:raise RuntimeError('Worker exited before its identity was recorded')
            except Exception as exc:
                output.close()
                event={'event':'failed_to_start','job':name,'time':now,'error':str(exc)}
                states[name]=dict(event,status='failed_to_start');append(events,[event]);continue
            processes[name]=process;outputs[name]=output;last_launch[gpu]=now
            memory[gpu]-=job['min_free_mb']
            event={'event':'started','job':name,'pid':process.pid,'token':info['token'],
                   'gpu':gpu,'preferred_gpu':job['gpu'],
                   'time':now,'started_at':now,'log':str(path)}
            states[name]=dict(event,status='started');append(events,[event])
        if now-last_heartbeat>60:
            append(events,[{'event':'heartbeat','time':now,
                           'states':{k:v['status'] for k,v in states.items()},'available_after_launch_reservations_mb':memory,
                           'pipeline_memory_holds':pipeline_holds}])
            last_heartbeat=now
        if all(state['status'] in {'released','failed_to_start'} for state in states.values()):
            append(events,[{'event':'scheduler_finished','time':now}]);return
        time.sleep(10)


if __name__=='__main__':main()
