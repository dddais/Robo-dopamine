"""Dedicated round31 gradient smoke/training worker; inference never loads labels."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import append, latest
from .prepare import OUT, CONFIGS
from .compare_native_bias import same_input
from .learned_attention_gates import LearnedGateRuntime
from .provenance import snapshot


POLICY = OUT/'selection_learned_attention_gates_v1.json'
BASE = OUT/'learned_attention_gates_v1'
PROTOCOLS = ['image_text', 'text_image', 'interleaved', 'video_text', 'text_video']
SCOPES = ['all_frames', 'last_frame']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(sources):
    # Keep GPU workers independent of CPU scheduler imports and their visibility guard.
    for name, digest in sources.items():
        p = Path(name).resolve()
        if not p.is_relative_to(OUT.resolve()) or sha(p) != digest:
            raise ValueError('Frozen source changed or left this research session')


def policy():
    p = json.loads(POLICY.read_text()); t = p['training']
    if (p['variant'] != 'learned_attention_gates_v1' or p['protocols'] != PROTOCOLS or p['scopes'] != SCOPES
            or t['epochs'] != 3 or t['batch_size'] != 1 or t['gradient_accumulation'] != 8
            or t['training_ks'] != [32, 64] or t['seed'] != 20260912 or t['initial_theta'] != -2.
            or t['learning_rate'] != .05 or t['total_forward_backward_steps'] != 4200
            or t['total_optimizer_steps'] != 525 or t['trainable_parameters'] != 56):
        raise ValueError('Changed frozen training policy')
    verify_sources(p['sources_sha256'])
    return p


def config(model, protocol):
    cfg = json.loads((CONFIGS/f'{model}_{protocol}.json').read_text())
    cfg.update(model=model, protocol=protocol, method='learned_gate_bias', bias=6., require_task_positions=True,
        ranking_prefix='ANSWER: ', contrast_weight=0., contrast_negative_mode='learned_gates', negative_strength=4.,
        batch_size=8, readout='evidence_contrast', frozen_ranking_source=str(
            OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'))
    return cfg


def ranking(model, protocol):
    root = OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'
    return {scope: json.loads((root/f'ranking_{scope}.json').read_text()) for scope in SCOPES}


def population():
    splits = json.loads((OUT/'splits.json').read_text())
    ids = splits['discovery']
    if len(ids) != 70 or set(ids) & set(splits['validation']): raise ValueError('Invalid fixed discovery population')
    rows = [s for s in json.loads((OLD/'inputs.json').read_text()) if s['example_id'] in set(ids)]
    if len(rows) != 70: raise ValueError('Incomplete requested input population')
    return rows


def weight_probes(model):
    parameters = list(model.named_parameters())
    indexes = sorted({0, len(parameters)//4, len(parameters)//2, 3*len(parameters)//4, len(parameters)-1})
    result = {}
    for index in indexes:
        name, tensor = parameters[index]
        # Small fixed tensor slices, plus complete no-gradient and optimizer exclusion checks.
        sample = tensor.detach().reshape(-1)[:1024].contiguous().cpu().view(torch.uint8)
        result[name] = dict(shape=list(tensor.shape), first1024_sha256=hashlib.sha256(sample.numpy().tobytes()).hexdigest())
    return result


def smoke(model):
    spec = policy(); samples = population()[:8]; root = BASE/'smoke'/model
    if (root/'complete.json').exists(): raise ValueError('This immutable smoke already completed')
    runtime = LearnedGateRuntime(config(model, 'image_text'))
    runtime.controller.theta = torch.nn.Parameter(torch.full((28, 2), -2., device=runtime.model.device))
    initial = runtime.controller.theta.detach().clone()
    backbone = weight_probes(runtime.model); checks = []; sources = {str(POLICY): sha(POLICY)}
    for protocol in ['image_text', 'text_video']:
        cfg = config(model, protocol); runtime.cfg.clear(); runtime.cfg.update(cfg)
        ranks = ranking(model, protocol); folder = root/protocol
        create_json(folder/'runtime_config.json', cfg); create_json(folder/'requested_ids.json', [s['example_id'] for s in samples])
        baseline = runtime.predict(samples, 'baseline')
        append(folder/'baseline.jsonl', baseline)
        old_path = OUT/'experiments'/f'{model}_{protocol}_uniform_factorized_kl_b08'/'discovery_smoke/predictions/baseline.jsonl'
        old = latest(old_path); sources[str(old_path)] = sha(old_path)
        for row in baseline:
            if not same_input(row, old[row['example_id']]) or row['native_class_logits_positive'] != old[row['example_id']]['native_class_logits_positive']:
                raise ValueError('Actual same-batch unsteered baseline does not replay exactly')
        for scope in SCOPES:
            rank_file = Path(cfg['frozen_ranking_source'])/f'ranking_{scope}.json'
            sources[str(rank_file)] = sha(rank_file)
            for k in [8, 32]:
                condition = f'{scope}:target:{k}'
                runtime.controller.zero_probe = True
                try: zero = runtime.predict(samples, condition, ranks)
                finally: runtime.controller.zero_probe = False
                for row, base in zip(zero, baseline):
                    if not same_input(row, base) or row['native_class_logits_positive'] != base['native_class_logits_positive']:
                        raise ValueError('Explicit zero-gate probe must exactly reproduce original SDPA baseline')
                append(folder/f'{scope}_k{k}_zero.jsonl', zero)
                positive = runtime.predict(samples, condition, ranks)
                append(folder/f'{scope}_k{k}_initial.jsonl', positive)
                active = {h['layer']-8 for h in ranks[scope]['ranking'][:k]}
                gradients = []
                # Exactly the training batch size. Normal/zero replay above retains the original batch of eight.
                for sample in samples:
                    runtime.controller.theta.grad = None
                    logits, diagnostics, maps = runtime.differentiable_logits([sample], condition, ranks)
                    loss = (logits-logits.mean(-1, keepdim=True)).square().mean()
                    loss.backward()
                    grad = runtime.controller.theta.grad
                    inactive = [i for i in range(28) if i not in active]
                    if (grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0
                            or (inactive and bool((grad[inactive] != 0).any()))
                            or any(p.grad is not None or p.requires_grad for p in runtime.model.parameters())):
                        raise ValueError('Actual class-symmetric gradient probe failed frozen/active parameter contract')
                    gradients.append(dict(example_id=sample['example_id'], batch_size=1,
                        centered_all_class_square_loss=float(loss.detach()), gradient=grad.detach().cpu().tolist(),
                        active_layers=sorted(i+8 for i in active), token_audit=maps[0], attention_diagnostics=diagnostics))
                    del logits, loss
                create_json(folder/f'{scope}_k{k}_gradients.json', gradients)
                checks.append(dict(protocol=protocol, scope=scope, k=k, samples=8, gradient_probe_batch_size=1,
                    actual_zero_gate_logits_exact=True, actual_gradient_finite_nonzero=True,
                    inactive_layer_gradients_zero=True, backbone_gradients_absent=True,
                    actual_gradient_forwards=8, normal_inference_forwards=1, zero_probe_forwards=1))
                print(model, protocol, condition, 'gradient smoke passed', flush=True)
    if not torch.equal(initial, runtime.controller.theta) or backbone != weight_probes(runtime.model):
        raise ValueError('A parameter changed during a no-optimizer smoke')
    for path in root.rglob('*'):
        if path.is_file() and path.name != 'complete.json': sources[str(path)] = sha(path)
    create_json(root/'complete.json', dict(status='pass', labels_read=False, model=model, checks=checks,
        sources_sha256=sources, theta_unchanged=True, backbone_probe_hashes=backbone,
        backbone_requires_grad_count=0, optimizer_updates=0, trainable_parameters=56,
        interpretation='Eight actual conditions, 64 gradient observations, explicit batch8 zero-gate replay; training uses batch1. No efficacy selection.'))


def schedule(n, epoch):
    entries = [(i, p, s, k) for i in range(n) for p in PROTOCOLS for s in SCOPES for k in [32, 64]]
    random.Random(20260912+epoch).shuffle(entries)
    return entries


def train(model):
    spec = policy(); gate = BASE/'actual_gradient_gate_v1.json'
    record = json.loads(gate.read_text())
    if record['status'] != 'pass' or record['labels_read'] or record['models'] != ['qwen', 'roboreward']:
        raise ValueError('Both models must pass actual zero-gate and gradient smoke before any training')
    verify_sources(record['sources_sha256'])
    root = BASE/'training'/model
    if root.exists(): raise ValueError('Training output exists; inspect it rather than silently restarting')
    root.mkdir(parents=True)
    samples = population(); ids = [s['example_id'] for s in samples]
    # Training is the only model worker phase authorized to read these labels.
    labels_all = json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels = {e: labels_all[e] for e in ids}; del labels_all
    targets = {}
    for e, value in labels.items():
        # Label schema is checked explicitly before GPU training (see CPU/training gate).
        targets[e] = int(value['reward'])-1
    counts = {c: sum(t == c for t in targets.values()) for c in [0, 4]}
    if set(targets.values()) != {0, 4} or counts != {0: 45, 4: 25}: raise ValueError('Changed accepted training label population')
    create_json(root/'training_manifest.json', dict(policy=str(POLICY), policy_sha256=sha(POLICY),
        source_gradient_gate=str(gate), source_gradient_gate_sha256=sha(gate), training_ids=ids,
        labels_source_sha256=sha(OLD/'labels_for_scoring_only.json'), labels_used=ids,
        trainable_parameter_count=56, actual_backbone_trainable_parameter_count=0, class_counts=counts,
        independent_examples=70, independent_video_groups=28))
    runtime = LearnedGateRuntime(config(model, 'image_text'))
    runtime.controller.theta = torch.nn.Parameter(torch.full((28, 2), -2., device=runtime.model.device))
    theta = runtime.controller.theta
    optimizer = torch.optim.Adam([theta], lr=.05)
    if any(p is not theta for g in optimizer.param_groups for p in g['params']): raise ValueError('Only gates may be optimized')
    backbone = weight_probes(runtime.model)
    ranks = {p: ranking(model, p) for p in PROTOCOLS}
    for p in PROTOCOLS:
        for scope in SCOPES:
            create_json(root/'rankings'/p/f'ranking_{scope}.json', ranks[p][scope])
    steps = 0; updates = 0; started = time.time(); accumulated = 0.
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(3):
        entries = schedule(len(samples), epoch)
        create_json(root/f'schedule_epoch_{epoch+1}.json', entries)
        for index, protocol, scope, k in entries:
            cfg = config(model, protocol); runtime.cfg.clear(); runtime.cfg.update(cfg)
            sample = samples[index]; target = targets[sample['example_id']]
            logits, diagnostics, maps = runtime.differentiable_logits([sample], f'{scope}:target:{k}', ranks[protocol])
            ce = F.cross_entropy(logits, torch.tensor([target], device=logits.device))
            balanced = ce*(70/(2*counts[target]))
            penalty = .005*theta.sigmoid().mean()
            loss = balanced+penalty
            if not torch.isfinite(loss): raise ValueError('Non-finite actual training loss')
            (loss/8).backward(); accumulated += float(loss.detach()); steps += 1
            if steps % 8 == 0:
                if theta.grad is None or not torch.isfinite(theta.grad).all(): raise ValueError('Invalid learned gate gradient')
                if any(p.grad is not None or p.requires_grad for p in runtime.model.parameters()): raise ValueError('Backbone received training gradients')
                grad_norm = float(torch.nn.utils.clip_grad_norm_([theta], 1.))
                optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
                append(root/'optimizer_events.jsonl', [dict(event='optimizer_step', time=time.time(), epoch=epoch+1,
                    forward_backward_steps=steps, optimizer_steps=updates, mean_loss=accumulated/8,
                    gradient_norm_before_clip=grad_norm, gate_min=float(theta.sigmoid().min()), gate_max=float(theta.sigmoid().max()))])
                accumulated = 0.
                if updates % 5 == 0: print(model, 'training', steps, '/4200', 'updates', updates, '/525', flush=True)
            del logits, ce, balanced, penalty, loss
        create_json(root/f'epoch_{epoch+1}_gates_for_audit_only.json', dict(theta=theta.detach().cpu().tolist(),
            values=theta.sigmoid().detach().cpu().tolist(), inference_eligible=False, steps=steps))
    if steps != 4200 or updates != 525 or backbone != weight_probes(runtime.model):
        raise ValueError('Frozen duration or backbone integrity failed')
    create_json(root/'final_gates.json', dict(status='complete', model=model, variant='learned_attention_gates_v1',
        policy=str(POLICY), policy_sha256=sha(POLICY), training_ids=ids, gate_values=theta.sigmoid().detach().cpu().tolist(),
        theta=theta.detach().cpu().tolist(), forward_backward_steps=steps, optimizer_steps=updates,
        trained_parameter_count=56, backbone_requires_grad_count=0, backbone_probe_hashes=backbone,
        elapsed_seconds=time.time()-started, selection_rule='Final fixed step only', validation_labels_used=[]))
    print(root/'final_gates.json', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['qwen', 'roboreward'])
    parser.add_argument('--phase', required=True, choices=['smoke', 'train'])
    args = parser.parse_args(); policy()
    run = snapshot(vars(args))
    create_json(BASE/'worker_starts'/f'{run["run_id"]}.json', run)
    (smoke if args.phase == 'smoke' else train)(args.model)


if __name__ == '__main__': main()
