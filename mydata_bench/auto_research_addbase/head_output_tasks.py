"""Frozen gates and complete discovery selection for round30 head output contrast."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .compare_native_bias import same_input, heads
from .durable_scheduler import include_additions, validate_plan
from .head_output_contrast import compose, norm_error


MODELS = ['qwen', 'roboreward']
PROTOCOLS = ['image_text', 'text_image', 'interleaved', 'video_text', 'text_video']
SCOPES = ['all_frames', 'last_frame']
NEIGHBORHOODS = {8: [8, 16, 24], 32: [24, 32, 40], 64: [48, 64, 80]}
VARIANT = 'head_output_contrast_norm_a1'
POLICY = OUT/'selection_head_output_contrast_discovery_v1.json'
QUEUE = OUT/'queue_gpu01_20260911_1700'


def check_policy():
    spec = json.loads(POLICY.read_text())
    expected = dict(variant=VARIANT, models=MODELS, protocols=PROTOCOLS, scopes=SCOPES, ks=[8, 32, 64],
        strength=4, negative_strength=4, negative_task_strength=4, gain=1, contrast_weight=0,
        task_binding_fraction=.5, task_binding_distribution='uniform', actual_full_model_forwards=1,
        composition='o_candidate=o_original+(o_positive-(o_visual_negative+o_task_negative)/2); o_out=norm(o_original)*o_candidate/norm(o_candidate)')
    if any(spec.get(k) != v for k, v in expected.items()): raise ValueError('Changed round30 policy')
    for name, digest in spec['source_documents'].items():
        path = Path(name).resolve()
        if (not (path.is_relative_to(OUT.resolve()) or path == Path(__file__).resolve().parent/'PROGRESS_20260912_1240.md')
                or sha(path) != digest):
            raise ValueError('Frozen theory/evidence source changed or left the permitted paths')
    return spec


def folder(model, protocol, population):
    return OUT/'experiments'/f'{model}_{protocol}_{VARIANT}'/population


def rank_path(model, protocol, scope):
    return OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'/f'ranking_{scope}.json'


def verify_rows(rows, ids, expected_heads=None, baseline=False):
    if set(rows) != set(ids) or any(r['status'] != 'ok' for r in rows.values()):
        raise ValueError('Require complete successful actual head-output observations')
    # Diagnostics are shared across all rows of a batch. Audit each distinct payload once.
    checked = set()
    for row in rows.values():
        if (row.get('derived_only') or row['readout'] != 'five_way_answer_likelihood_head_output_contrast'
                or row['contrast_negative_mode'] != 'head_output' or row['contrast_weight'] != 0
                or row['actual_forward_branches'] != 1 or row['head_output_gain'] != 1
                or row['norm_relative_tolerance'] != .01 or len(set(row['candidate_token_ids'])) != 5
                or row['native_class_logits_negative'] is not None or row['negative_attention_diagnostics']):
            raise ValueError('Require one real native forward, with no final-logit contrast')
        z = np.asarray(row['native_class_logits_positive'], dtype=float)
        p = np.asarray(row['native_class_probabilities'], dtype=float)
        if z.shape != (5,) or p.shape != (5,) or not np.isfinite([*z, *p, row['progress']]).all():
            raise ValueError('Five finite native class outputs required')
        expected = np.exp(z-z.max()); expected /= expected.sum()
        if (np.max(np.abs(expected-p)) >= 1e-5 or row['reward'] != int(expected.argmax())+1
                or row['progress'] != (row['reward']-1)/4):
            raise ValueError('Native uncontrasted output does not reconstruct')
        diag = row['attention_diagnostics']
        if baseline:
            if diag or row['condition'] != 'baseline': raise ValueError('Actual baseline must be unsteered')
            continue
        observed = heads(row, 'attention_diagnostics')
        if not observed or (expected_heads is not None and observed != expected_heads):
            raise ValueError('Actual selected heads differ from frozen ranking')
        key = json.dumps(diag, sort_keys=True)
        if key in checked: continue
        checked.add(key)
        for layer, d in diag.items():
            if (d['method'] != 'head_output_contrast_norm' or d['strength'] != 4 or d['gain'] != 1
                    or not d['all_query_rows'] or not d['causal_mask_preserved'] or not d['local_QKV_shared']
                    or not d['unselected_head_outputs_exact'] or d['prefill_calls'] != 1 or d['decode_calls'] != 0
                    or d['local_attention_evaluations'] != 4 or not np.isfinite(d['max_relative_norm_error'])
                    or not 0 <= d['max_relative_norm_error'] <= .01):
                raise ValueError('Actual local computation or norm preservation differs')
            for name, method, strength in [('positive', 'binding_transport', 4),
                    ('visual_negative', 'mass_transport', -4), ('task_negative', 'binding_task_suppression', 4)]:
                b = d['local_branches'][name]
                if (b['method'] != method or b['strength'] != strength or b['heads'] != d['heads']
                        or not b['all_query_rows'] or not b['causal_mask_preserved'] or b['prefill_calls'] != 1):
                    raise ValueError('Actual local branch contract differs')
                if name == 'positive' and (b['task_binding_fraction'] != .5 or b['task_binding_distribution'] != 'uniform'):
                    raise ValueError('Wrong actual local task positive')
                if name == 'task_negative' and b['task_logit_strength'] != -4:
                    raise ValueError('Wrong actual local task negative')
            if len(d['probes']) != 1: raise ValueError('Require recorded actual single-prefill vectors')
            for probe in d['probes']:
                if probe['output_dtype'] not in {'torch.bfloat16', 'torch.float32'}:
                    raise ValueError('Unregistered output dtype')
                dtype = torch.bfloat16 if probe['output_dtype'] == 'torch.bfloat16' else torch.float32
                parts = [torch.tensor(probe['vectors'][name], dtype=dtype) for name in
                         ['original', 'positive', 'visual_negative', 'task_negative', 'output']]
                # CPU and GPU float32 norm reductions can differ at a dtype rounding boundary.
                # Compare under the predeclared 1% norm tolerance, without changing saved output.
                replay = compose(*parts[:4])
                torch.testing.assert_close(replay.float(), parts[4].float(), rtol=.01, atol=1e-6)
                if norm_error(parts[0], parts[4]) > .01:
                    raise ValueError('Actual recorded vector norm does not reconstruct')
    return rows


def complete(root, ks, scopes, n):
    request = root/'requested_ids.json'
    if not request.exists(): return False
    ids = json.loads(request.read_text())
    if len(ids) != n or len(set(ids)) != n: return False
    files = [root/'predictions/baseline.jsonl']+[root/'binding_transport_s4/predictions'/f'{s}_target_{k}.jsonl' for s in scopes for k in ks]
    return all(p.exists() and set(latest(p)) == set(ids) for p in files)


def command(model, protocols, population, ks, scopes=SCOPES, smoke=False):
    result = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
        'mydata_bench.auto_research_addbase.worker', '--model', model, '--protocols', *protocols,
        '--variant', VARIANT, '--methods', 'binding_transport', '--strengths', '4', '--batch-size', '8',
        '--ranking-prefix', 'ANSWER: ', '--contrast-weight', '0', '--contrast-negative-mode', 'head_output',
        '--task-binding-fraction', '.5', '--task-binding-distribution', 'uniform',
        '--ks', *map(str, ks), '--scopes', *scopes, '--population', population,
        '--frozen-ranking-root', str(rank_path(model, protocols[0], SCOPES[0]).parents[1])]
    return result+(['--limit', '8'] if smoke else [])


def write_addition(name, jobs, sources):
    existing = include_additions(json.loads((QUEUE/'plan.json').read_text()), QUEUE)
    validate_plan(dict(jobs=list(existing.values())+jobs))
    path = QUEUE/'additions'/name
    create_json(path, dict(jobs=jobs, sources_sha256=sources,
        interpretation='Round30 fixed sample-local head-output direction, one model forward; GPU0/1 only'))
    with path.with_suffix('.ready').open('x') as f: f.write('Prior gates passed; GPU0/1 only\n')
    return path


def register_smoke(cpu):
    check_policy(); a = json.loads(cpu.read_text())
    if a['status'] != 'pass' or a['labels_read'] or a['head_output_tests'] != 8 or a['tests_run'] < 85:
        raise ValueError('Require new CPU checks and prior-method regressions')
    for name, digest in a['source_code_sha256'].items():
        path = Path(name).resolve()
        if not path.is_relative_to(Path(__file__).resolve().parent) or sha(path) != digest:
            raise ValueError('Research implementation changed after CPU audit')
    jobs = [dict(name=f'smoke_{m}_head_output', gpu=i, min_free_mb=23000, depends_on=[],
        command=command(m, ['image_text', 'text_video'], 'discovery', [8, 32], smoke=True)) for i, m in enumerate(MODELS)]
    return write_addition('stage30_head_output_smoke.json', jobs, {str(cpu): sha(cpu), str(POLICY): sha(POLICY)})


def matrix_ready(smoke=False):
    return all(complete(folder(m, p, 'discovery_smoke' if smoke else 'discovery'),
                        [8, 32] if smoke else [8, 32, 64], SCOPES, 8 if smoke else 70)
               for m in MODELS for p in (['image_text', 'text_video'] if smoke else PROTOCOLS))


def audit_matrix(smoke=False):
    check_policy()
    if not matrix_ready(smoke): raise ValueError('Require complete explicitly requested matrix')
    sources = {str(POLICY): sha(POLICY)}; matrices = {}; checks = []
    def read(path, rows=False):
        sources[str(path)] = sha(path); return latest(path) if rows else json.loads(path.read_text())
    for m in MODELS:
        for p in (['image_text', 'text_video'] if smoke else PROTOCOLS):
            root = folder(m, p, 'discovery_smoke' if smoke else 'discovery')
            ids = read(root/'requested_ids.json')
            reference = OUT/'experiments'/f'{m}_{p}_uniform_factorized_kl_b08'/'discovery_smoke'
            if not smoke:
                reference = OUT/'experiments'/f'{m}_{p}_task_content_block_evidence_a1'/'discovery'
            if ids != read(reference/'requested_ids.json'): raise ValueError('Same-batch baseline population differs')
            cfg = read(root/'runtime_config.json')
            expected = dict(model=m, contrast_negative_mode='head_output', contrast_weight=0, batch_size=8,
                max_pixels=50176, ranking_prefix='ANSWER: ', task_binding_fraction=.5, task_binding_distribution='uniform',
                frozen_ranking_source=str(rank_path(m, p, SCOPES[0]).parent))
            if any(cfg.get(k) != v for k, v in expected.items()): raise ValueError('Actual configuration differs')
            base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, baseline=True)
            old_base = read(reference/'predictions/baseline.jsonl', True)
            if any(not same_input(base[e], old_base[e]) or base[e]['native_class_logits_positive'] != old_base[e]['native_class_logits_positive'] for e in ids):
                raise ValueError('Native same-batch baseline did not replay exactly')
            conditions = {}
            for s in SCOPES:
                rank = read(rank_path(m, p, s))['ranking']
                if read(root.parent/'ranking'/f'ranking_{s}.json')['ranking'] != rank: raise ValueError('Copied ranking differs')
                for k in ([8, 32] if smoke else [8, 32, 64]):
                    chosen = {(h['layer'], h['head']) for h in rank[:k]}
                    if len(chosen) != k: raise ValueError('Duplicate heads')
                    rows = verify_rows(read(root/'binding_transport_s4/predictions'/f'{s}_target_{k}.jsonl', True), ids, chosen)
                    if any(not same_input(r, base[e]) or r['condition'] != f'{s}:target:{k}' for e, r in rows.items()):
                        raise ValueError('Same actual input or condition differs')
                    conditions[(s, k)] = rows
                    checks.append(dict(model=m, protocol=p, scope=s, k=k, n=len(ids),
                        actual_native_formula_heads_norm_and_vectors_verified=True, same_input_baseline_exact=True))
            matrices[(m, p)] = (ids, base, conditions)
    verify_sources(sources)
    return matrices, sources, checks


def verify_smoke(destination):
    _, sources, checks = audit_matrix(True)
    create_json(destination, dict(status='pass', labels_read=False, checks=checks, sources_sha256=sources,
        actual_intervention_rows_verified=128, actual_model_forwards=128, actual_baseline_rows=32,
        interpretation='Actual one-forward local head-output formula and norm probes; no efficacy selection'))


def register_discovery(audit):
    a = json.loads(audit.read_text())
    if a['status'] != 'pass' or a['labels_read'] or len(a['checks']) != 16: raise ValueError('Actual smoke gate required')
    verify_sources(a['sources_sha256'])
    jobs = [dict(name=f'round30_{m}_head_output_discovery', gpu=i, min_free_mb=23000,
        depends_on=[f'smoke_{m}_head_output'], command=command(m, PROTOCOLS, 'discovery', [8, 32, 64])) for i, m in enumerate(MODELS)]
    return write_addition('stage30_head_output_discovery.json', jobs, {str(audit): sha(audit), str(POLICY): sha(POLICY)})


def select(destination):
    matrices, sources, checks = audit_matrix(False)
    split_ids = set(json.loads((OUT/'splits.json').read_text())['discovery'])
    if any(set(ids) != split_ids for ids, _, _ in matrices.values()): raise ValueError('Changed discovery population')
    labels = json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels = {e: labels[e] for e in split_ids}
    points = []; proposed = []
    for (m, p), (ids, baseline, conditions) in matrices.items():
        base = summary(baseline, labels, ids); passing = []
        for (s, k), rows in conditions.items():
            score = summary(rows, labels, ids)
            delta = {c: score['accuracy']['0.125/0.875'][c]['rate_all_expected']-base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all', 'suc', 'fail']}
            gate = score['mae'] < base['mae'] and delta['all'] >= .1-1e-12 and min(delta['suc'], delta['fail']) > 0
            point = dict(model=m, protocol=p, scope=s, center_k=k, variant=VARIANT, method='binding_transport_s4',
                mae=score['mae'], baseline_mae=base['mae'], deltas=delta, passes=gate,
                accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'])
            points.append(point)
            if gate: passing.append(point)
        if passing:
            best = min(passing, key=lambda x: (x['mae'], -x['accuracy_all'], x['center_k'], x['scope']))
            proposed.append(dict(best, ks=NEIGHBORHOODS[best['center_k']]))
    counts = {m: sum(p['model'] == m for p in proposed) for m in MODELS}; eligible = min(counts.values()) >= 3
    create_json(destination, dict(created_at=time.time(), variant=VARIANT, all_discovery_points=points,
        proposed_input_candidates=proposed, input_counts=counts, family_eligible_for_full=eligible,
        selected=proposed if eligible else [], sources_sha256=sources, validation_labels_used=[], checks=checks,
        actual_intervention_rows_verified=4200, independent_conditions=60, actual_model_forwards=4200,
        interpretation='One fixed head-output formula; dataset-internal adaptive exploration'))


def register_full(selection, audit):
    record = json.loads(selection.read_text()); a = json.loads(audit.read_text())
    eligible = min(record['input_counts'].values()) >= 3
    if record['selected'] != (record['proposed_input_candidates'] if eligible else []) or a['status'] != 'pass' or a['labels_read']:
        raise ValueError('Changed shared coverage or actual gate')
    verify_sources(dict(record['sources_sha256'], **a['sources_sha256']))
    jobs = []; previous = {}
    for p in record['selected']:
        m = p['model']; name = f"validate_{m}_{p['protocol']}_head_output"
        jobs.append(dict(name=name, gpu=MODELS.index(m), min_free_mb=23000,
            depends_on=[previous.get(m, f'round30_{m}_head_output_discovery')],
            command=command(m, [p['protocol']], 'full_cohort', p['ks'], [p['scope']])))
        previous[m] = name
    return write_addition('stage30_head_output_validation.json', jobs, {str(selection): sha(selection), str(audit): sha(audit)})
