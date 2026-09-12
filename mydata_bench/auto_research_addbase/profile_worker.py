"""Generate functional head probes. This GPU process never opens label files."""
import argparse
import json
from pathlib import Path
import time
from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import append, predict_condition
from .prepare import OUT, CONFIGS
from .dual_rank import dual_rank, alternating_union
from .evidence import EvidenceRuntime
from .provenance import snapshot


def layer_heads(ranking,layer):
    rows=[row for row in ranking if row['layer']==layer]
    visual=sorted(rows,key=lambda r:(-r['visual_score'],r['head']))
    instruction=sorted(rows,key=lambda r:(-r['task_score'],r['head']))
    return alternating_union(visual,instruction)[:8]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['qwen','roboreward','meter'],required=True)
    p.add_argument('--protocols',nargs='+',required=True)
    p.add_argument('--layers',nargs='+',type=int,default=list(range(8,36)))
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if not set(args.layers)<=set(range(8,36)):raise ValueError('Profile only layers 8 through 35')
    startup=snapshot(vars(args))
    split=json.loads((OUT/'splits.json').read_text())
    data=json.loads((OLD/'inputs.json').read_text())
    samples=[s for s in data if s['example_id'] in set(split['discovery'])]
    if args.limit:samples=samples[:args.limit]
    ranking_samples=[s for s in data if s['ranking']]
    runtime=None
    for protocol in args.protocols:
        cfg=json.loads((CONFIGS/f'{args.model}_{protocol}.json').read_text())
        cfg.update(output_dir=cfg['output_dir']+'_functional_profile',
            method='binding_transport',bias=4.,task_binding_fraction=.5,ranking_strategy='dual_mass',
            readout='evidence_contrast',contrast_weight=0.,negative_strength=4.,
            ranking_prefix='' if args.model=='meter' else 'ANSWER: ',
            profiling='single layer with 8 dual-selected heads; no labels in this process')
        create_json(CONFIGS/f'{args.model}_{protocol}_functional_profile.json',cfg)
        folder=Path(cfg['output_dir'])/('discovery_smoke' if args.limit else 'discovery')
        create_json(folder/'runtime_config.json',cfg)
        create_json(folder/'requested_ids.json',[s['example_id'] for s in samples])
        create_json(folder/'run_intents'/f'{startup["run_id"]}.json',startup)
        if runtime is None:runtime=EvidenceRuntime(cfg)
        else:runtime.cfg.clear();runtime.cfg.update(cfg)
        create_json(folder/'loading_audit.json',runtime.model.loading_audit)
        predict_condition(runtime,samples,'baseline',None,folder)
        rankings=dual_rank(runtime,ranking_samples,None,Path(cfg['output_dir'])/'ranking')
        for layer in args.layers:
            group={scope:dict(value,ranking=layer_heads(value['ranking'],layer))
                   for scope,value in rankings.items()}
            output=folder/f'binding_transport_s4_layer{layer}'
            create_json(output/'intervention_config.json',dict(runtime.cfg,profile_layer=layer,
                        profiled_heads={s:x['ranking'] for s,x in group.items()}))
            for scope in ['all_frames','last_frame']:
                predict_condition(runtime,samples,f'{scope}:target:8',group,output)
        append(folder/'worker_events.jsonl',[{'event':'complete','time':time.time(),'arguments':vars(args)}])


if __name__=='__main__':main()
