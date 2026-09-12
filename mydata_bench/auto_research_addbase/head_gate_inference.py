"""Round32 native inference: immutable final head gates and no label access."""
import argparse
import json

from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import append, predict_condition
from .prepare import OUT, CONFIGS
from .head_gate_worker import BASE, POLICY, PROTOCOLS, SCOPES, config, ranking, sha, policy
from .head_attention_gates import HeadGateRuntime
from .provenance import snapshot


def final_record(model):
    policy()
    path = BASE/'training'/model/'final_gates.json'
    r = json.loads(path.read_text())
    expected = dict(status='complete', model=model, variant='learned_head_gates_v1',
        forward_backward_steps=4200, optimizer_steps=525, trained_parameter_count=1792,
        backbone_requires_grad_count=0, selection_rule='Final fixed step only',
        validation_labels_used=[], policy_sha256=sha(POLICY))
    if any(r.get(k) != v for k, v in expected.items()):
        raise ValueError('Only the fixed final1792-parameter checkpoint is inference eligible')
    return path, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    p.add_argument('--protocols', choices=PROTOCOLS, nargs='+', required=True)
    p.add_argument('--population', choices=['discovery', 'full_cohort'], required=True)
    p.add_argument('--ks', nargs='+', type=int, required=True)
    p.add_argument('--scopes', choices=SCOPES, nargs='+', required=True)
    p.add_argument('--controls', choices=['target', 'wrong_region', 'low_rank'], nargs='+', default=['target'])
    p.add_argument('--variant', choices=['learned_head_gates_v1', 'learned_head_gates_k_coverage_v1'], default='learned_head_gates_v1')
    args = p.parse_args()
    if not set(args.ks) <= {8, 16, 24, 32, 40, 48, 64, 80}:
        raise ValueError('Only registered discovery and original neighborhood k are permitted')
    gate_file, record = final_record(args.model)
    if args.variant == 'learned_head_gates_k_coverage_v1':
        from .head_gate_worker import verify_sources
        extension = json.loads((OUT/'selection_learned_head_gates_k_coverage_v1.json').read_text())
        verify_sources(extension['sources_sha256'])
        if args.population != 'full_cohort':
            raise ValueError('Registered k coverage uses only full846 inference')
        for protocol in args.protocols:
            point, = [x for x in extension['candidates'] if x['model'] == args.model and x['protocol'] == protocol]
            if args.ks != point['extra_ks'] or args.scopes != [point['scope']]:
                raise ValueError('Require entire registered extra-k matrix in the original scope')
    start = snapshot(vars(args))
    split = json.loads((OUT/'splits.json').read_text())
    requested = set(split[args.population])
    samples = [s for s in json.loads((OLD/'inputs.json').read_text()) if s['example_id'] in requested]
    runtime = None
    for protocol in args.protocols:
        cfg = config(args.model, protocol)
        cfg.update(output_dir=str(OUT/'experiments'/f'{args.model}_{protocol}_{args.variant}'),
            learned_gate_file=str(gate_file), learned_gate_sha256=sha(gate_file))
        root = OUT/'experiments'/f'{args.model}_{protocol}_{args.variant}'
        folder = root/args.population
        create_json(CONFIGS/f'{args.model}_{protocol}_{args.variant}.json', cfg)
        create_json(folder/'runtime_config.json', cfg)
        create_json(folder/'requested_ids.json', [s['example_id'] for s in samples])
        create_json(folder/'run_intents'/f'{start["run_id"]}.json', start)
        if runtime is None:
            runtime = HeadGateRuntime(cfg)
            runtime.controller.fixed_values = record['gate_values']
            runtime.controller.values(runtime.model.device)
        else:
            runtime.cfg.clear(); runtime.cfg.update(cfg)
        create_json(folder/'loading_audit.json', runtime.model.loading_audit)
        ranks = ranking(args.model, protocol)
        for scope in SCOPES:
            create_json(root/'ranking'/f'ranking_{scope}.json', ranks[scope])
        predict_condition(runtime, samples, 'baseline', None, folder)
        intervention = folder/'learned_head_gate_bias_s6'
        create_json(intervention/'intervention_config.json', cfg)
        for scope in args.scopes:
            for k in args.ks:
                for kind in args.controls:
                    predict_condition(runtime, samples, f'{scope}:{kind}:{k}', ranks, intervention)
        append(folder/'worker_events.jsonl', [dict(event='complete', arguments=start['arguments'])])


if __name__ == '__main__': main()
