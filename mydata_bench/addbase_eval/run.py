"""Resumable, append-only experiment execution; every condition is independent."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import yaml

from .prepare import create_json
from .runtime import Runtime
from mydata_bench.top_eval.versioning import (
    protocol_metadata, validate_records, validate_output_directory,
)


def append(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        f.flush()


def latest(path):
    if not path.exists(): return {}
    out = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r=json.loads(line);out[r['example_id']]=r
    return out


def batches(rows, count):
    for i in range(0,len(rows),count): yield rows[i:i+count]


def predict_condition(runtime, samples, condition, rankings, output):
    lock_path=output/'locks'/(condition.replace(':','_')+'.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            flags=fcntl.LOCK_EX if condition=='baseline' else fcntl.LOCK_EX|fcntl.LOCK_NB
            fcntl.flock(lock,flags)
        except BlockingIOError:
            print(f'{condition}: another GPU owns this independent condition',flush=True)
            return None
        try:
            return _predict_condition(runtime,samples,condition,rankings,output)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def _predict_condition(runtime, samples, condition, rankings, output):
    cfg=runtime.cfg
    if condition != 'baseline' and cfg['model']=='sole' and cfg['protocol']=='official':
        validate_records(cfg, rankings.values(), 'steering rankings')
    name=condition.replace(':','_')
    path=output/'predictions'/f'{name}.jsonl'
    done=latest(path)
    validate_records(cfg, done.values(), path)
    remaining=[s for s in samples if s['example_id'] not in done or
               (cfg.get('retry_runtime_errors',False) and done[s['example_id']]['status']=='runtime_error'
                and not str(done[s['example_id']].get('error','')).startswith('Wrong-region control unavailable'))]
    official=cfg['model']=='sole' and cfg['protocol']=='official'
    for batch in batches(remaining,cfg['batch_size']):
        active=batch[:]
        previous={s['example_id']:'0' for s in batch}
        traces={s['example_id']:[] for s in batch}
        final={}
        for step in range(1,8) if official else [None]:
            if not active: break
            try:
                rows=runtime.predict(active,condition,rankings,step,[previous[s['example_id']] for s in active] if official else None)
            except Exception as exc:
                # Record and isolate runtime failures; do not skip other conditions.
                if isinstance(exc, __import__('torch').OutOfMemoryError):
                    raise
                rows=[]
                for sample in active:
                    try:
                        rows += runtime.predict([sample],condition,rankings,step,[previous[sample['example_id']]] if official else None)
                    except Exception as inner:
                        rows.append({'example_id':sample['example_id'],'condition':condition,'status':'runtime_error',
                                     'progress':None,'step':step,'error':str(inner),'traceback':traceback.format_exc()})
            next_active=[]
            byid={s['example_id']:s for s in active}
            for row in rows:
                row.update(protocol_metadata(cfg))
                eid=row['example_id']
                if official:
                    traces[eid].append(row)
                    append(output/'steps'/f'{name}.jsonl',[row])
                if row['status']=='ok':
                    if official:
                        # SOLE predicts ABSOLUTE progress. Never add it as a GRM delta.
                        previous[eid]=row['percentage_text']
                    next_active.append(byid[eid])
                final[eid]=row
            active=next_active
        result=[]
        for sample in batch:
            row=dict(final[sample['example_id']])
            if official:
                row['step_count']=len(traces[sample['example_id']])
                row['progress_curve']=[0]+[r['progress'] for r in traces[sample['example_id']]]
                row['previous_values_are_model_predictions']=True
                if row['step_count'] != 7: row['status']='incomplete_rollout';row['progress']=None
            row['completed_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
            result.append(row)
        append(path,result)
        done.update({r['example_id']:r for r in result})
        print(f"{cfg['model']}/{cfg['protocol']} {condition} {len(done)}/{len(samples)} ok={sum(r['status']=='ok' for r in result)}/{len(result)}",flush=True)
    return done


def rank(runtime, samples, baseline, output):
    path=output/'ranking_observations.jsonl'
    cfg=runtime.cfg
    official=cfg['model']=='sole' and cfg['protocol']=='official'
    requested=samples
    if official:
        validate_records(cfg, baseline.values(), 'baseline used for ranking')
        samples=[s for s in requested if baseline.get(s['example_id'],{}).get('status')=='ok'
                 and baseline[s['example_id']].get('step_count')==7]
        create_json(output/'ranking_eligibility.json',{'requested':len(requested),'eligible':len(samples),
                    'excluded':{s['example_id']:baseline.get(s['example_id'],{}).get('status','missing')
                                for s in requested if s not in samples},
                    'reason':'Official terminal ranking requires a valid model-predicted predecessor at step 6'})
        if len(samples)<2:raise RuntimeError('Insufficient valid official ranking rollouts')
    done=latest(path)
    validate_records(cfg, done.values(), path)
    unexpected = set(done)-{s['example_id'] for s in samples}
    if unexpected:
        raise ValueError(f'Ranking cache contains examples outside the requested ranking set: {sorted(unexpected)}')
    remaining=[s for s in samples if s['example_id'] not in done]
    for batch in batches(remaining,cfg['batch_size']):
        prior=[baseline[s['example_id']]['previous_percentage_text'] for s in batch] if official else None
        rows=runtime.collect(batch,7 if official else None,prior)
        append(path,rows);done.update({r['example_id']:r for r in rows})
        print(f"RANK {cfg['model']}/{cfg['protocol']} {len(done)}/{len(samples)}",flush=True)
    valid=[r for r in done.values() if r['status']=='ok']
    if len(valid)!=len(samples): raise RuntimeError('Ranking observations incomplete')
    rankings={}
    for scope in cfg['scopes']:
        matrix=np.mean([r['raw_mass'][scope] for r in valid],axis=0)
        ordered=[{'layer':l,'head':h,'score':float(matrix[l,h])} for l in range(cfg['skip_early_layers'],36) for h in range(32)]
        ordered.sort(key=lambda r:(-r['score'],r['layer'],r['head']))
        artifact={'scope':scope,'num_layers':36,'num_heads':32,'skip_early_layers':cfg['skip_early_layers'],
                  'ranking_score':'mean_raw_mass','n':len(valid),'example_ids':[r['example_id'] for r in valid],
                  'query_kind':valid[0]['query_kind'],'ranking':ordered, **protocol_metadata(cfg)}
        create_json(output/f'ranking_{scope}.json',artifact)
        rankings[scope]=artifact
    return rankings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--phase',choices=['all','baseline','rank','steer'],default='all')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--output-suffix')
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--retry-runtime-errors',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if args.batch_size: cfg['batch_size']=args.batch_size
    output=Path(cfg['output_dir']+(args.output_suffix or ''))
    if args.limit and not args.output_suffix: raise ValueError('Pilot runs require a separate output suffix')
    validate_output_directory(cfg, output)
    samples=json.loads(Path(cfg['inputs']).read_text())
    if args.limit: samples=sorted(samples,key=lambda s:(not s['ranking'],s['example_id']))[:args.limit]
    output.mkdir(parents=True,exist_ok=True)
    create_json(output/'run_config.json',cfg)
    cfg['retry_runtime_errors']=args.retry_runtime_errors
    runtime=Runtime(cfg)
    create_json(output/'loading_audit.json',runtime.model.loading_audit)
    append(output/'events.jsonl',[{'example_id':str(time.time()),'event':'start','pid':os.getpid(),
                                   'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'phase':args.phase}])
    baseline_path=output/'predictions/baseline.jsonl'
    baseline=predict_condition(runtime,samples,'baseline',None,output) if args.phase in {'all','baseline'} else latest(baseline_path)
    if args.phase=='baseline': return
    ranking_samples=[s for s in samples if s['ranking']]
    rankings=rank(runtime,ranking_samples,baseline,output) if args.phase in {'all','rank'} else {
              scope:json.loads((output/f'ranking_{scope}.json').read_text()) for scope in cfg['scopes']}
    if args.phase=='rank': return
    cohort=[s for s in samples if s['cohort']]
    for scope in cfg['scopes']:
        for k in cfg['top_k']:
            for kind in ['target']+cfg['controls']:
                predict_condition(runtime,cohort,f'{scope}:{kind}:{k}',rankings,output)
    append(output/'events.jsonl',[{'example_id':str(time.time()),'event':'complete','phase':args.phase}])


if __name__=='__main__':main()
