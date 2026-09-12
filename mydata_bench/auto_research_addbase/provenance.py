"""Append-only source snapshots for subsequently started GPU workers."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch
import transformers
from .prepare import OUT
from mydata_bench.addbase_eval.prepare import ROOT, create_json


def snapshot(arguments):
    arguments=json.loads(json.dumps(arguments,default=str))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    from .gpu_policy import require_allowed_visibility
    require_allowed_visibility()
    name=f'{time.time_ns()}_pid{os.getpid()}'
    paths=list((ROOT/'mydata_bench/auto_research_addbase').glob('*.py'))
    paths+=list((ROOT/'mydata_bench/addbase_eval').glob('*.py'))
    paths+=[ROOT/f'mydata_bench/{p}' for p in ['top_eval/protocol.py','qwen_eval/protocols.py',
            'roboreward_eval/runner.py','attention_eval/masking.py','attention_eval/runtime.py']]
    sources={str(path.relative_to(ROOT)):{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'source':path.read_text()} for path in sorted(set(paths))}
    record={'run_id':name,'created_at':time.time(),'arguments':arguments,'pid':os.getpid(),
            'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
            'cuda_visible_devices':visible,'sources':sources,
            'scope':'Snapshot of repository source available at this worker startup; checkpoint tensors are separately audited.'}
    path=OUT/'code_snapshots'/f'{name}.json'
    create_json(path,record)
    return {'run_id':name,'source_snapshot':str(path),'source_snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'arguments':arguments,'pid':os.getpid(),'cuda_visible_devices':visible}
