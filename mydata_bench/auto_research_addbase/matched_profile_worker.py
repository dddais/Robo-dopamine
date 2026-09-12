"""Round18 GPU probes of the exact uniform two-branch native readout.

No labels are read. Original per-layer head groups are copied without ranking
again, so profiling-batch padding cannot change the candidate pool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import append, predict_condition
from .prepare import OUT, CONFIGS
from .evidence import EvidenceRuntime
from .provenance import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    parser.add_argument('--protocols', nargs='+', required=True)
    parser.add_argument('--layers', nargs='+', type=int, default=list(range(8, 36)))
    parser.add_argument('--limit', type=int, choices=[8])
    args = parser.parse_args()
    if not set(args.layers) <= set(range(8, 36)) or len(set(args.layers)) != len(args.layers):
        raise ValueError('Unique candidate layers 8 through 35 required')
    if not set(args.protocols) <= {'image_text', 'text_image', 'interleaved', 'text_video', 'video_text'}:
        raise ValueError('Unknown input construction')
    startup = snapshot(vars(args))
    splits = json.loads((OUT / 'splits.json').read_text())
    data = json.loads((OLD / 'inputs.json').read_text())
    samples = [s for s in data if s['example_id'] in set(splits['discovery'])]
    if args.limit:
        samples = samples[:args.limit]
    runtime = None
    for protocol in args.protocols:
        cfg = json.loads((CONFIGS / f'{args.model}_{protocol}.json').read_text())
        cfg.update(output_dir=cfg['output_dir'] + '_uniform_method_profile', method='binding_transport', bias=4.,
                   task_binding_fraction=.5, task_binding_distribution='uniform',
                   readout='evidence_contrast', contrast_weight=1., negative_strength=4., ranking_prefix='ANSWER: ',
                   profiling='Round18 isolated fixed groups; true uniform positive and visual negative; no labels')
        create_json(CONFIGS / f'{args.model}_{protocol}_uniform_method_profile.json', cfg)
        root = Path(cfg['output_dir'])
        folder = root / ('discovery_smoke' if args.limit else 'discovery')
        create_json(folder / 'runtime_config.json', cfg)
        create_json(folder / 'requested_ids.json', [s['example_id'] for s in samples])
        create_json(folder / 'run_intents' / f'{startup["run_id"]}.json', startup)
        if runtime is None:
            runtime = EvidenceRuntime(cfg)
        else:
            runtime.cfg.clear(); runtime.cfg.update(cfg)
        create_json(folder / 'loading_audit.json', runtime.model.loading_audit)
        sources, rankings = {}, {}
        for scope in ['all_frames', 'last_frame']:
            path = OUT / 'functional_selections' / f'stage8_{args.model}_v1' / f'{args.model}_{protocol}' / f'ranking_{scope}.json'
            value = json.loads(path.read_text())
            if value['validation_ids_used'] or {r['layer'] for r in value['layer_profiles']} != set(range(8, 36)):
                raise ValueError('Original frozen candidate pool differs')
            for layer in value['layer_profiles']:
                if len(layer['heads']) != 8 or {h['layer'] for h in layer['heads']} != {layer['layer']}:
                    raise ValueError('Expected intact groups of eight within one layer')
            rankings[scope] = value
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            create_json(root / 'original_group_sources' / f'ranking_{scope}.json', value)
        predict_condition(runtime, samples, 'baseline', None, folder)
        for layer in args.layers:
            groups = {scope: dict(value, ranking=next(r['heads'] for r in value['layer_profiles'] if r['layer'] == layer))
                      for scope, value in rankings.items()}
            output = folder / f'binding_transport_s4_layer{layer}'
            create_json(output / 'intervention_config.json', dict(runtime.cfg, profile_layer=layer,
                profiled_heads={s: v['ranking'] for s, v in groups.items()}, original_group_sources_sha256=sources))
            for scope in ['all_frames', 'last_frame']:
                predict_condition(runtime, samples, f'{scope}:target:8', groups, output)
        append(folder / 'worker_events.jsonl', [dict(event='complete', time=time.time(), arguments=vars(args))])


if __name__ == '__main__':
    main()
