"""Actual gradient audit and training authorization within the fixed research plan."""
import json
from pathlib import Path

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .compare_native_bias import same_input, heads
from .durable_scheduler import include_additions, validate_plan
from .head_gate_worker import BASE, POLICY, policy, ranking


MODELS = ['qwen', 'roboreward']
QUEUE = OUT/'queue_gpu01_20260911_1700'


def addition(name, jobs, sources):
    current = include_additions(json.loads((QUEUE/'plan.json').read_text()), QUEUE)
    validate_plan(dict(jobs=list(current.values())+jobs))
    p = QUEUE/'additions'/name
    create_json(p, dict(jobs=jobs, sources_sha256=sources,
        interpretation='Round32 frozen-backbone learned attention gates. Physical GPU0/1 only.'))
    with p.with_suffix('.ready').open('x') as f: f.write('Frozen round32 prior gates satisfied; GPU0/1\n')
    return p


def command(model, phase):
    return ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
            'mydata_bench.auto_research_addbase.head_gate_worker', '--model', model, '--phase', phase]


def register_smoke(cpu):
    policy(); a = json.loads(cpu.read_text())
    if a['status'] != 'pass' or a['labels_read'] or a['head_gate_tests'] != 13 or a['tests_run'] < 24:
        raise ValueError('Gradient, schedule and prior-method CPU gates required')
    for name, digest in a['source_code_sha256'].items():
        p = Path(name).resolve()
        if not p.is_relative_to(Path(__file__).resolve().parent) or sha(p) != digest:
            raise ValueError('Tested research code changed')
    jobs = [dict(name=f'smoke_{m}_learned_head_gates', gpu=i, min_free_mb=60000, depends_on=[], command=command(m, 'smoke'))
            for i, m in enumerate(MODELS)]
    return addition('stage32_learned_head_gates_smoke.json', jobs, {str(cpu): sha(cpu), str(POLICY): sha(POLICY)})


def smoke_ready():
    return all((BASE/'smoke'/m/'complete.json').exists() for m in MODELS)


def verify_smoke():
    policy()
    if not smoke_ready(): raise ValueError('Wait for both actual complete gradient smokes')
    sources = {str(POLICY): sha(POLICY)}; checks = []
    def read(path, rows=False):
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    initial = float(1/(1+np.exp(np.float32(2.))))
    for model in MODELS:
        root = BASE/'smoke'/model; complete = read(root/'complete.json')
        verify_sources(complete['sources_sha256'])
        if (complete['status'] != 'pass' or complete['labels_read'] or len(complete['checks']) != 8
                or not complete['theta_unchanged'] or complete['optimizer_updates'] != 0
                or complete['backbone_requires_grad_count'] != 0 or complete['trainable_parameters'] != 1792):
            raise ValueError('Smoke did not satisfy actual frozen parameter/no-optimizer contract')
        for protocol in ['image_text', 'text_video']:
            folder = root/protocol; ids = read(folder/'requested_ids.json')
            if len(ids) != 8 or len(set(ids)) != 8: raise ValueError('Changed actual smoke population')
            base = read(folder/'baseline.jsonl', True)
            for scope in ['all_frames', 'last_frame']:
                ranks = ranking(model, protocol)[scope]['ranking']
                for k in [8, 32]:
                    expected_heads = {(h['layer'], h['head']) for h in ranks[:k]}
                    active = {(l-8,h) for l,h in expected_heads}
                    inactive = np.ones((28,32), dtype=bool)
                    for l,h in active: inactive[l,h] = False
                    zero = read(folder/f'{scope}_k{k}_zero.jsonl', True)
                    normal = read(folder/f'{scope}_k{k}_initial.jsonl', True)
                    gradients = read(folder/f'{scope}_k{k}_gradients.json')
                    if set(base) != set(ids) or set(zero) != set(ids) or set(normal) != set(ids) or [g['example_id'] for g in gradients] != ids:
                        raise ValueError('Incomplete actual same-sample normal/zero/gradient matrix')
                    for e in ids:
                        if (not same_input(zero[e], base[e]) or not same_input(normal[e], base[e])
                                or zero[e]['native_class_logits_positive'] != base[e]['native_class_logits_positive']):
                            raise ValueError('Actual input or zero-gate baseline replay differs')
                        for row, is_zero in [(zero[e], True), (normal[e], False)]:
                            if (row['status'] != 'ok' or row['contrast_weight'] != 0 or row['actual_forward_branches'] != 1
                                    or row['native_class_logits_negative'] is not None
                                    or heads(row, 'attention_diagnostics') != expected_heads):
                                raise ValueError('Actual single native forward/head set differs')
                            z = np.asarray(row['native_class_logits_positive'], dtype=float)
                            recorded = np.asarray(row['native_class_probabilities'], dtype=float)
                            if z.shape != (5,) or recorded.shape != (5,) or not np.isfinite([*z, *recorded, row['progress']]).all():
                                raise ValueError('Invalid actual five-class logits/probabilities')
                            p = np.exp(z-z.max()); p /= p.sum()
                            if (np.max(np.abs(p-recorded)) >= 1e-5 or row['reward'] != int(p.argmax())+1
                                    or row['progress'] != (row['reward']-1)/4):
                                raise ValueError('Actual native output does not reconstruct')
                            for d in row['attention_diagnostics'].values():
                                if (d['method'] != 'learned_head_gate_bias' or d['visual_cap'] != 6 or d['task_cap'] != 4
                                        or not d['causal_mask_preserved'] or not d['all_query_rows']
                                        or d['zero_gate_probe'] != is_zero or d['prefill_calls'] != 1
                                        or np.max(np.abs(np.asarray(d['gates'])-(0 if is_zero else initial))) > 1e-7):
                                    raise ValueError('Actual initial/zero gate values differ')
                    for g in gradients:
                        grad = np.asarray(g['gradient'], dtype=float)
                        if (grad.shape != (28, 32, 2) or not np.isfinite(grad).all() or np.abs(grad).sum() == 0
                                or np.any(grad[inactive] != 0) or g['batch_size'] != 1
                                or {tuple(x) for x in g['active_heads']} != {(l+8,h) for l,h in active}):
                            raise ValueError('Actual gradient support or shape differs')
                    checks.append(dict(model=model, protocol=protocol, scope=scope, k=k, n=8,
                        native_zero_baseline_exact=True, active_gate_gradient_verified=True))
    verify_sources(sources)
    dest = BASE/'actual_gradient_gate_v1.json'
    create_json(dest, dict(status='pass', labels_read=False, models=MODELS, checks=checks, sources_sha256=sources,
        normal_rows=128, zero_gate_rows=128, gradient_observations=128, optimizer_updates=0,
        interpretation='Both real checkpoints passed batch8 zero-gate baseline replay and batch1 class-symmetric gradient checks. No efficacy label or parameter update.'))
    return dest


def register_training(audit):
    a = json.loads(audit.read_text()); policy()
    if a['status'] != 'pass' or a['labels_read'] or a['models'] != MODELS or len(a['checks']) != 16:
        raise ValueError('Complete actual gradient audit required before fixed training')
    verify_sources(a['sources_sha256'])
    jobs = [dict(name=f'round32_{m}_learned_head_gates_train', gpu=i, min_free_mb=60000,
        depends_on=[f'smoke_{m}_learned_head_gates'], command=command(m, 'train')) for i, m in enumerate(MODELS)]
    return addition('stage32_learned_head_gates_training.json', jobs, {str(audit): sha(audit), str(POLICY): sha(POLICY)})
