"""Audit fixed-step training and evaluate one frozen gate family without labels in inference."""
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .compare_native_bias import same_input, heads
from .learned_gate_worker import BASE, POLICY, PROTOCOLS, SCOPES, policy, schedule, population
from .learned_gate_tasks import MODELS, addition


VARIANT = 'learned_attention_gates_v1'
NEIGHBORHOODS = {8: [8, 16, 24], 32: [24, 32, 40], 64: [48, 64, 80]}


def training_ready():
    return all((BASE/'training'/m/'final_gates.json').exists() for m in MODELS)


def verify_training():
    policy()
    if not training_ready(): raise ValueError('Wait for both fixed final checkpoints')
    ids = [s['example_id'] for s in population()]; sources = {str(POLICY): sha(POLICY)}; checks = []
    def read(path):
        sources[str(path)] = sha(path); return json.loads(path.read_text())
    gate = read(BASE/'actual_gradient_gate_v1.json'); verify_sources(gate['sources_sha256'])
    for model in MODELS:
        root = BASE/'training'/model; record = read(root/'final_gates.json'); manifest = read(root/'training_manifest.json')
        if (record['status'] != 'complete' or record['model'] != model or record['training_ids'] != ids
                or record['validation_labels_used'] or record['forward_backward_steps'] != 4200
                or record['optimizer_steps'] != 525 or record['trained_parameter_count'] != 56
                or record['backbone_requires_grad_count'] != 0 or record['policy_sha256'] != sha(POLICY)
                or record['selection_rule'] != 'Final fixed step only' or manifest['training_ids'] != ids
                or manifest['labels_used'] != ids or manifest['actual_backbone_trainable_parameter_count'] != 0
                or manifest['trainable_parameter_count'] != 56
                or manifest['labels_source_sha256'] != sha(OLD/'labels_for_scoring_only.json')):
            raise ValueError('Actual training duration, sample, parameter or checkpoint selection contract differs')
        values = np.asarray(record['gate_values'], dtype=float); theta = np.asarray(record['theta'], dtype=float)
        if (values.shape != (28, 2) or theta.shape != (28, 2) or not np.isfinite([*values.flat, *theta.flat]).all()
                or np.max(np.abs(values-1/(1+np.exp(-theta)))) > 1e-7
                or np.any((values < 0) | (values > 1)) or np.all(theta == -2.)):
            raise ValueError('Actual 56 trained sigmoid gates do not reconstruct')
        events_path = root/'optimizer_events.jsonl'; sources[str(events_path)] = sha(events_path)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        if len(events) != 525: raise ValueError('Incomplete fixed optimizer trajectory')
        for step, e in enumerate(events, 1):
            if (e['event'] != 'optimizer_step' or e['optimizer_steps'] != step or e['forward_backward_steps'] != 8*step
                    or e['epoch'] != (step-1)//175+1
                    or not np.isfinite([e['mean_loss'], e['gradient_norm_before_clip'], e['gate_min'], e['gate_max']]).all()
                    or not 0 <= e['gate_min'] <= e['gate_max'] <= 1):
                raise ValueError('Nonfinite or missing actual training steps')
        for epoch in range(3):
            if read(root/f'schedule_epoch_{epoch+1}.json') != [list(x) for x in schedule(70, epoch)]:
                raise ValueError('Actual fixed shuffled training schedule differs')
            checkpoint = read(root/f'epoch_{epoch+1}_gates_for_audit_only.json')
            if checkpoint['inference_eligible'] or checkpoint['steps'] != 1400*(epoch+1):
                raise ValueError('Intermediate checkpoint is incorrectly selected for inference')
            if epoch == 2 and (checkpoint['theta'] != record['theta'] or checkpoint['values'] != record['gate_values']):
                raise ValueError('Final inference coefficients differ from the fixed last training step')
        for protocol in PROTOCOLS:
            for scope in SCOPES:
                original = OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'/f'ranking_{scope}.json'
                if read(original) != read(root/'rankings'/protocol/f'ranking_{scope}.json'):
                    raise ValueError('Training used different head rankings')
        checks.append(dict(model=model, training_examples=70, actual_forward_backward_steps=4200, optimizer_updates=525,
            learned_parameters=56, min_gate=float(values.min()), max_gate=float(values.max()),
            final_checkpoint_only=True, schedule_and_source_heads_verified=True))
    verify_sources(sources)
    dest = BASE/'fixed_training_audit_v1.json'
    create_json(dest, dict(status='pass', labels_read=False, checks=checks, sources_sha256=sources,
        interpretation='Complete fixed-step optimization audit. Inference eligibility, not efficacy or independent generalization evidence.'))
    return dest


def command(model, protocols, population_name, ks, scopes=SCOPES, smoke=False):
    result = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
        'mydata_bench.auto_research_addbase.worker', '--model', model, '--protocols', *protocols,
        '--variant', VARIANT, '--methods', 'learned_gate_bias', '--strengths', '6', '--batch-size', '8',
        '--ranking-prefix', 'ANSWER: ', '--contrast-weight', '0', '--contrast-negative-mode', 'learned_gates',
        '--learned-gate-file', str(BASE/'training'/model/'final_gates.json'), '--ks', *map(str, ks),
        '--scopes', *scopes, '--population', population_name,
        '--frozen-ranking-root', str(OUT/'functional_selections'/f'stage8_{model}_v1')]
    return result+(['--limit', '8'] if smoke else [])


def register_discovery(audit):
    a = json.loads(audit.read_text())
    if a['status'] != 'pass' or a['labels_read'] or len(a['checks']) != 2: raise ValueError('Both final training audits required')
    verify_sources(a['sources_sha256'])
    cpu = OUT/'audit/learned_gates_inference_cpu_20260912_132508.json'
    tested = json.loads(cpu.read_text())
    if tested['status'] != 'pass' or tested['labels_read'] or tested['tests_run'] != 102 or tested['learned_gate_tests'] != 11:
        raise ValueError('Trained native inference integration must pass its CPU gate')
    from pathlib import Path
    for name in ['worker.py', 'learned_attention_gates.py', 'runtime.py', 'evidence.py', 'attention.py']:
        path = Path(__file__).resolve().parent/name
        if sha(path) != tested['source_code_sha256'][str(path)]:
            raise ValueError('Inference implementation changed after regression checks')
    jobs = [dict(name=f'round31_{m}_learned_gates_discovery', gpu=i, min_free_mb=23000,
        depends_on=[f'round31_{m}_learned_gates_train'], command=command(m, PROTOCOLS, 'discovery', [8, 32, 64]))
        for i, m in enumerate(MODELS)]
    return addition('stage31_learned_gates_discovery.json', jobs, {str(audit): sha(audit), str(POLICY): sha(POLICY), str(cpu): sha(cpu)})


def folder(model, protocol, population_name):
    return OUT/'experiments'/f'{model}_{protocol}_{VARIANT}'/population_name


def matrix_ready():
    for model in MODELS:
        for protocol in PROTOCOLS:
            root = folder(model, protocol, 'discovery'); request = root/'requested_ids.json'
            if not request.exists(): return False
            ids = set(json.loads(request.read_text()))
            if len(ids) != 70: return False
            files = [root/'predictions/baseline.jsonl']+[root/'learned_gate_bias_s6/predictions'/f'{s}_target_{k}.jsonl'
                     for s in SCOPES for k in [8, 32, 64]]
            if not all(path.exists() and set(latest(path)) == ids for path in files): return False
    return True


def verify_rows(rows, ids, gate_record, gate_file, expected_heads=None, baseline=False):
    if set(rows) != set(ids) or any(r['status'] != 'ok' for r in rows.values()): raise ValueError('Require all complete actual native rows')
    values = np.asarray(gate_record['gate_values'], dtype=float)
    for row in rows.values():
        if (row['readout'] != 'five_way_answer_likelihood_learned_attention_gates' or row['contrast_weight'] != 0
                or row['contrast_negative_mode'] != 'learned_gates' or row['actual_forward_branches'] != 1
                or row['native_class_logits_negative'] is not None or row['negative_attention_diagnostics']
                or row['learned_gate_file'] != str(gate_file) or row['learned_gate_sha256'] != sha(gate_file)
                or row['gate_parameter_count'] != 56 or not row['backbone_frozen'] or len(set(row['candidate_token_ids'])) != 5):
            raise ValueError('Actual fixed trained-gate/single-native-forward contract differs')
        z = np.asarray(row['native_class_logits_positive'], dtype=float)
        p = np.asarray(row['native_class_probabilities'], dtype=float)
        if z.shape != (5,) or p.shape != (5,) or not np.isfinite([*z, *p, row['progress']]).all(): raise ValueError('Nonfinite native output')
        expected = np.exp(z-z.max()); expected /= expected.sum()
        if np.max(np.abs(expected-p)) >= 1e-5 or row['reward'] != int(p.argmax())+1 or row['progress'] != (row['reward']-1)/4:
            raise ValueError('Actual uncontrasted five-class output does not reconstruct')
        if baseline:
            if row['condition'] != 'baseline' or row['attention_diagnostics']: raise ValueError('Require unsteered baseline')
        else:
            if heads(row, 'attention_diagnostics') != expected_heads: raise ValueError('Actual inference heads differ')
            for layer, d in row['attention_diagnostics'].items():
                if (d['method'] != 'learned_gate_bias' or not d['all_query_rows'] or not d['causal_mask_preserved']
                        or d['visual_cap'] != 6 or d['task_cap'] != 4 or d['zero_gate_probe']
                        or d['prefill_calls'] != 1 or np.max(np.abs(values[int(layer)-8]-d['gates'])) > 1e-7):
                    raise ValueError('Actual layer scalar values/mask differ from trained artifact')
    return rows


def select(destination):
    policy()
    if not matrix_ready(): raise ValueError('Wait for all ten actual learned-gate matrices')
    ids = [s['example_id'] for s in population()]; sources = {str(POLICY): sha(POLICY)}; matrices = {}
    def read(path, rows=False):
        sources[str(path)] = sha(path); return latest(path) if rows else json.loads(path.read_text())
    training = read(BASE/'fixed_training_audit_v1.json'); verify_sources(training['sources_sha256'])
    for model in MODELS:
        gate_file = BASE/'training'/model/'final_gates.json'; gate = read(gate_file)
        for protocol in PROTOCOLS:
            root = folder(model, protocol, 'discovery')
            if read(root/'requested_ids.json') != ids: raise ValueError('Changed inference discovery population')
            cfg = read(root/'runtime_config.json')
            if (cfg['model'] != model or cfg['learned_gate_sha256'] != sha(gate_file)
                    or cfg['contrast_weight'] != 0 or cfg['batch_size'] != 8 or cfg['max_pixels'] != 50176):
                raise ValueError('Actual evaluation configuration differs')
            base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, gate, gate_file, baseline=True)
            reference = read(OUT/'experiments'/f'{model}_{protocol}_head_output_contrast_norm_a1'/'discovery/predictions/baseline.jsonl', True)
            if any(not same_input(base[e], reference[e]) or base[e]['native_class_logits_positive'] != reference[e]['native_class_logits_positive'] for e in ids):
                raise ValueError('Actual same-batch baseline did not exactly replay')
            conditions = {}
            for scope in SCOPES:
                ranking_path = OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'/f'ranking_{scope}.json'
                rank = read(ranking_path)['ranking']
                if read(root.parent/'ranking'/f'ranking_{scope}.json')['ranking'] != rank: raise ValueError('Copied head ranking differs')
                for k in [8, 32, 64]:
                    chosen = {(h['layer'], h['head']) for h in rank[:k]}
                    if len(chosen) != k: raise ValueError('Duplicate heads')
                    rows = verify_rows(read(root/'learned_gate_bias_s6/predictions'/f'{scope}_target_{k}.jsonl', True), ids, gate, gate_file, chosen)
                    if any(not same_input(r, base[e]) or r['condition'] != f'{scope}:target:{k}' for e, r in rows.items()):
                        raise ValueError('Changed actual same-batch condition')
                    conditions[(scope, k)] = rows
            matrices[(model, protocol)] = base, conditions
    verify_sources(sources)
    all_labels = json.loads((OLD/'labels_for_scoring_only.json').read_text()); labels = {e: all_labels[e] for e in ids}; del all_labels
    points = []; proposed = []
    for (model, protocol), (baseline, conditions) in matrices.items():
        base = summary(baseline, labels, ids); passing = []
        for (scope, k), rows in conditions.items():
            score = summary(rows, labels, ids)
            delta = {c: score['accuracy']['0.125/0.875'][c]['rate_all_expected']-base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all', 'suc', 'fail']}
            gate = score['mae'] < base['mae'] and delta['all'] >= .1-1e-12 and min(delta['suc'], delta['fail']) > 0
            point = dict(model=model, protocol=protocol, scope=scope, center_k=k, variant=VARIANT,
                method='learned_gate_bias_s6', mae=score['mae'], baseline_mae=base['mae'], deltas=delta, passes=gate,
                accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'])
            points.append(point)
            if gate: passing.append(point)
        if passing:
            best = min(passing, key=lambda p: (p['mae'], -p['accuracy_all'], p['center_k'], p['scope']))
            proposed.append(dict(best, ks=NEIGHBORHOODS[best['center_k']]))
    counts = {m: sum(p['model'] == m for p in proposed) for m in MODELS}; eligible = min(counts.values()) >= 3
    create_json(destination, dict(created_at=time.time(), variant=VARIANT, all_discovery_points=points,
        proposed_input_candidates=proposed, input_counts=counts, family_eligible_for_full=eligible,
        selected=proposed if eligible else [], sources_sha256=sources, validation_labels_used=[],
        actual_intervention_rows_verified=4200, independent_conditions=60,
        interpretation='Discovery was used for 56-parameter supervised training and then selection. Resubstitution is not independent evidence; any full validation remains adaptive dataset-internal exploration.'))


def register_full(selection):
    record = json.loads(selection.read_text()); verify_sources(record['sources_sha256'])
    eligible = min(record['input_counts'].values()) >= 3
    if record['selected'] != (record['proposed_input_candidates'] if eligible else []): raise ValueError('Changed shared coverage rule')
    jobs = []; previous = {}
    for p in record['selected']:
        model = p['model']; name = f"validate_{model}_{p['protocol']}_learned_gates"
        jobs.append(dict(name=name, gpu=MODELS.index(model), min_free_mb=23000,
            depends_on=[previous.get(model, f'round31_{model}_learned_gates_discovery')],
            command=command(model, [p['protocol']], 'full_cohort', p['ks'], [p['scope']])))
        previous[model] = name
    return addition('stage31_learned_gates_validation.json', jobs, {str(selection): sha(selection), str(POLICY): sha(POLICY)})
