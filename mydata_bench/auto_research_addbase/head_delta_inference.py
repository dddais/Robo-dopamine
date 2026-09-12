"""Round33 frozen final rank4 inference; no scoring labels or CPU scheduler imports."""
import argparse
import json
import torch

from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import append, predict_condition
from .prepare import OUT, CONFIGS
from .head_delta_worker import BASE, POLICY, VARIANT, PROTOCOLS, SCOPES, config, ranking, sha, policy, fixed_gates, verify_sources
from .head_delta_reft import HeadDeltaRuntime
from .provenance import snapshot


METHOD = 'learned_head_delta_reft_s6'
B0_VARIANT = 'learned_head_delta_reft_B0_v1'


def final_record(model):
    policy()
    audit = json.loads((BASE/'fixed_training_audit_v1.json').read_text())
    if audit['status'] != 'pass' or audit['labels_read'] or len(audit['checks']) != 2:
        raise ValueError('Both fixed final training audits required before inference')
    verify_sources(audit['sources_sha256'])
    path = BASE/'training'/model/'final_adapter.json'; record = json.loads(path.read_text())
    gate_file, _ = fixed_gates(model)
    expected = dict(status='complete', model=model, variant=VARIANT, rank=4,
        forward_backward_steps=4200, optimizer_steps=525, adapter_parameter_count=28672,
        backbone_requires_grad_count=0, gate_trainable_parameter_count=0,
        selection_rule='Final fixed step only', validation_labels_used=[], policy_sha256=sha(POLICY),
        fixed_gate_file=str(gate_file), fixed_gate_sha256=sha(gate_file))
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Only fixed final rank4 checkpoint is inference eligible')
    return path, record


def inference_config(model, protocol, mode='learned'):
    path = BASE/'training'/model/'final_adapter.json'
    variant = VARIANT if mode == 'learned' else B0_VARIANT
    return dict(config(model, protocol), learned_adapter_file=str(path), learned_adapter_sha256=sha(path),
        adapter_mode=mode, output_dir=str(OUT/'experiments'/f'{model}_{protocol}_{variant}'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    parser.add_argument('--protocols', choices=PROTOCOLS, nargs='+', required=True)
    parser.add_argument('--population', choices=['discovery', 'full_cohort'], required=True)
    parser.add_argument('--ks', type=int, nargs='+', required=True)
    parser.add_argument('--scopes', choices=SCOPES, nargs='+', required=True)
    parser.add_argument('--controls', choices=['target', 'wrong_region', 'low_rank'], nargs='+', default=['target'])
    parser.add_argument('--adapter-mode', choices=['learned', 'B0'], default='learned')
    args = parser.parse_args(); path, record = final_record(args.model)
    if args.population == 'discovery':
        if args.ks != [8, 32, 64] or args.scopes != SCOPES or args.controls != ['target'] or args.adapter_mode != 'learned':
            raise ValueError('Require full registered discovery matrix')
    else:
        selection = json.loads((BASE/'selection_complete_discovery_v1.json').read_text())
        verify_sources(selection['sources_sha256'])
        if not selection['family_eligible_for_full']: raise ValueError('Shared discovery coverage required')
        for protocol in args.protocols:
            point, = [p for p in selection['selected'] if p['model'] == args.model and p['protocol'] == protocol]
            if args.ks != point['ks'] or args.scopes != [point['scope']]:
                raise ValueError('Require the entire frozen passing-center neighborhood union')
        if args.controls != ['target'] or args.adapter_mode == 'B0':
            decision = json.loads((BASE/'coverage.json').read_text()); verify_sources(decision['sources_sha256'])
            if not decision['shared_coverage']: raise ValueError('Conditional controls require shared full target coverage')
    start = snapshot(vars(args)); split = json.loads((OUT/'splits.json').read_text())
    requested = set(split[args.population])
    samples = [s for s in json.loads((OLD/'inputs.json').read_text()) if s['example_id'] in requested]
    runtime = None; gate_file, gates = fixed_gates(args.model)
    for protocol in args.protocols:
        cfg = inference_config(args.model, protocol, args.adapter_mode)
        from pathlib import Path
        root = Path(cfg['output_dir']); folder = root/args.population
        create_json(CONFIGS/f'{root.name}.json', cfg); create_json(folder/'runtime_config.json', cfg)
        create_json(folder/'requested_ids.json', [s['example_id'] for s in samples])
        create_json(folder/'run_intents'/f'{start["run_id"]}.json', start)
        if runtime is None:
            runtime = HeadDeltaRuntime(cfg); runtime.controller.fixed_values = gates['gate_values']
            runtime.controller.A = torch.tensor(record['A'], dtype=torch.float32, device=runtime.model.device)
            runtime.controller.B = torch.tensor(record['B'], dtype=torch.float32, device=runtime.model.device)
            if args.adapter_mode == 'B0': runtime.controller.B.zero_()
            runtime.controller.values(runtime.model.device); runtime.controller.matrices(runtime.model.device)
        else:
            runtime.cfg.clear(); runtime.cfg.update(cfg)
        create_json(folder/'loading_audit.json', runtime.model.loading_audit)
        ranks = ranking(args.model, protocol)
        for scope in SCOPES: create_json(root/'ranking'/f'ranking_{scope}.json', ranks[scope])
        predict_condition(runtime, samples, 'baseline', None, folder)
        intervention = folder/METHOD; create_json(intervention/'intervention_config.json', cfg)
        for scope in args.scopes:
            for k in args.ks:
                for kind in args.controls:
                    predict_condition(runtime, samples, f'{scope}:{kind}:{k}', ranks, intervention)
        append(folder/'worker_events.jsonl', [dict(event='complete', arguments=start['arguments'])])


if __name__ == '__main__': main()
