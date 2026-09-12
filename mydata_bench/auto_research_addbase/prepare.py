"""Freeze group splits and configs before examining new predictions."""
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import yaml
from mydata_bench.addbase_eval.prepare import ROOT, OUT as OLD, create_json

OUT=ROOT/'results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911'
CONFIGS=ROOT/'mydata_bench/configs/v2_crossmodel/auto_20260911'

def main():
    samples=json.loads((OLD/'inputs.json').read_text())
    bytask=defaultdict(set)
    for s in samples:
        if s['cohort'] and s['holdout']:bytask[s['subset']].add(s['video_sha256'])
    # One video per task, chosen solely by a seeded content hash. Same-video
    # instruction variants always travel together, including cross-task variants.
    dev_groups={min(groups,key=lambda h:hashlib.sha256(('20260911:'+h).encode()).hexdigest())
                for groups in bytask.values()}
    split={name:[] for name in ['ranking','discovery','validation','full_cohort','old_holdout']}
    for s in samples:
        eid=s['example_id']
        if s['ranking']:split['ranking'].append(eid)
        if not s['cohort']:continue
        split['full_cohort'].append(eid)
        if s['holdout']:
            split['old_holdout'].append(eid)
            split['discovery' if s['video_sha256'] in dev_groups else 'validation'].append(eid)
    create_json(OUT/'splits.json',split)
    create_json(OUT/'split_audit.json',{'counts':{k:len(v) for k,v in split.items()},
         'discovery_tasks':dict(Counter(s['subset'] for s in samples if s['example_id'] in split['discovery'])),
         'rule':'one SHA256-selected holdout video group per task; original ranking groups excluded from validation',
         'input_sha256':hashlib.sha256((OLD/'inputs.json').read_bytes()).hexdigest()})
    for model in ['meter','sole','qwen','roboreward']:
        for protocol in ['video_text','text_video','image_text','text_image','interleaved']+(['official'] if model in {'meter','sole'} else []):
            source=OLD.parent.parent.parent/'unused'
            template=ROOT/f'mydata_bench/configs/v2_crossmodel_addbase/{"meter" if model=="meter" else "sole"}_{protocol}.yaml'
            cfg=yaml.safe_load(template.read_text())
            cfg.update(model=model,batch_size=4 if protocol=='official' else 8,
                       output_dir=str(OUT/'experiments'/f'{model}_{protocol}'),method='mass_transport',bias=4.0)
            if model in {'qwen','roboreward'}:
                cfg['model_path']=cfg['processor_path']=f'/home/dais/workspace/model/{"Qwen3-VL-8B-Instruct" if model=="qwen" else "RoboReward-8B"}'
                cfg['max_new_tokens']=128
            p=CONFIGS/f'{model}_{protocol}.json'
            create_json(p,cfg)
    print(json.dumps(json.loads((OUT/'split_audit.json').read_text()),indent=2))

if __name__=='__main__':main()
