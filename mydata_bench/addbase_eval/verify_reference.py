"""Independent reference-method parity using the cloned official source.

Only pure classes/functions needed for the readout are compiled, avoiding
imports of training, logging, server, or third-party API code.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
from torch import nn
import yaml

from .runtime import Runtime
from .prepare import OUT, ROOT, create_json


def selected(path,names):
    t=ast.parse(path.read_text())
    nodes=[]
    for n in t.body:
        if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names:nodes.append(n)
        elif isinstance(n,ast.ClassDef):
            nodes += [f for f in n.body if isinstance(f,ast.FunctionDef) and f.name in names]
    return compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec')


def main():
    cfg=yaml.safe_load((ROOT/'mydata_bench/configs/v2_crossmodel_addbase/meter_official.yaml').read_text())
    runtime=Runtime(cfg)
    samples=[s for s in json.loads((OUT/'inputs.json').read_text()) if s['ranking']][:2]
    inputs,maps,queries,texts=runtime.prepare(samples)
    with torch.inference_mode():
        hidden=runtime.model(**inputs,use_cache=False).last_hidden_state
        actual,success,_=runtime.model.read_progress(hidden,inputs['input_ids'],runtime.prog_id)
    refroot=ROOT/'mydata_bench/addbase_eval/references/robometer/robometer/models'
    namespace={'torch':torch,'nn':nn,'np':np,'Optional':Optional,'logger':SimpleNamespace(info=lambda *_:None)}
    exec(selected(refroot/'heads.py',{'PredictionHeadsMixin'}),namespace)
    methods={'squeeze_last_safe','_extract_hidden_state_from_token','_apply_heads_to_hidden_states','_process_token_extraction','convert_bins_to_continuous'}
    exec(selected(refroot/'rbm.py',methods),namespace)
    exec(selected(refroot/'utils.py',methods),namespace)
    refcls=type('OfficialPureReadout',(namespace['PredictionHeadsMixin'],),
                {name:namespace[name] for name in methods if name.startswith('_')})
    reference=refcls(hidden_dim=2560,model_config=SimpleNamespace(progress_loss_type='discrete',progress_discrete_bins=10)).to('cuda',dtype=torch.bfloat16).eval()
    reference.processor=runtime.processor;reference.tokenizer=runtime.processor.tokenizer
    reference.base_model_id='Qwen/Qwen3-VL-4B-Instruct'
    state={k:v for k,v in runtime.model.state_dict().items() if k.split('.')[0] in {'progress_head','success_head','preference_head'}}
    reference.load_state_dict(state,strict=True)
    with torch.inference_mode():
        progress_logits,success_logits,_=reference._process_token_extraction(hidden,inputs['input_ids'],'progress')
        official=namespace['convert_bins_to_continuous'](progress_logits['A'].float()).cpu()
        official_success=success_logits['A'].float().sigmoid().cpu()
    torch.testing.assert_close(torch.tensor(actual),official,rtol=0,atol=0)
    torch.testing.assert_close(torch.tensor(success),official_success,rtol=0,atol=0)
    # An observational ranking pass must not change subsequent ordinary inference.
    runtime.collect(samples)
    again=runtime.predict(samples,'baseline')
    differences=[abs(r['progress']-actual[i][-1]) for i,r in enumerate(again)]
    if max(differences)>0:raise AssertionError(f'Ranking pass changed baseline: {differences}')
    result={'status':'verified','sample_ids':[s['example_id'] for s in samples],
            'official_readout_max_abs_error':float((torch.tensor(actual)-official).abs().max()),
            'official_success_max_abs_error':float((torch.tensor(success)-official_success).abs().max()),
            'ranking_noninterference_max_abs_error':max(differences),'loading':runtime.model.loading_audit,
            'source_files':['models/rbm.py','models/heads.py','models/utils.py'],'official_progress':official.tolist()}
    create_json(OUT/'research/official_readout_parity.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
