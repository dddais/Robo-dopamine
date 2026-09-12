"""Complete-data gates for the fixed round28 two-forward task isolation method."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .compare_native_bias import same_input
from .durable_scheduler import include_additions, validate_plan


MODELS = ['qwen', 'roboreward']
PROTOCOLS = ['image_text', 'text_image', 'interleaved', 'video_text', 'text_video']
SCOPES = ['all_frames', 'last_frame']
NEIGHBORHOODS = {8: [8, 16, 24], 32: [24, 32, 40], 64: [48, 64, 80]}
VARIANT = 'task_content_block_evidence_a1'
POLICY = OUT/'selection_task_content_block_discovery_v1.json'
QUEUE = OUT/'queue_gpu01_20260911_1700'


def folder(model, protocol, population, probe=False):
    return OUT/'experiments'/f'{model}_{protocol}_{VARIANT}{"_probe" if probe else ""}'/population


def rank_path(model, protocol, scope):
    return OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'/f'ranking_{scope}.json'


def config_check(model, protocol, cfg, probe):
    expected = dict(model=model, contrast_negative_mode='task_content_block', contrast_weight=1,
        require_task_positions=True, content_block_probe=probe, method='bias', ranking_prefix='ANSWER: ',
        frozen_ranking_source=str(rank_path(model, protocol, SCOPES[0]).parent))
    if any(cfg.get(k) != v for k, v in expected.items()) or cfg.get('task_binding_fraction') is not None:
        raise ValueError('Actual task-content configuration differs from the frozen design')


def verify_rows(rows, ids, heads=None, baseline=False, probe=False):
    if set(rows) != set(ids) or any(r['status'] != 'ok' for r in rows.values()):
        raise ValueError('Require all and only complete valid requested observations')
    def vec(value):
        a = np.asarray(value, dtype=np.float32)
        if a.shape != (5,) or not np.isfinite(a).all(): raise ValueError('Invalid five-class vector')
        return a
    for r in rows.values():
        if (r.get('derived_only') or r['readout'] != 'five_way_answer_likelihood_task_content_block'
                or r['contrast_negative_mode'] != 'task_content_block' or len(set(r['candidate_token_ids'])) != 5):
            raise ValueError('Require actual native task-content readout')
        pos = vec(r['native_class_logits_positive'])
        if baseline:
            if (r['condition'] != 'baseline' or r['actual_forward_branches'] != 1 or r['contrast_weight'] != 0
                    or r['native_class_logits_blocked'] is not None or r['attention_diagnostics']
                    or r['blocked_attention_diagnostics'] or r['invariance_probe'] is not None
                    or r['actual_probe_forward_branches'] != 0):
                raise ValueError('Baseline must be an actual single unsteered forward')
            combined = pos
        else:
            if r['actual_forward_branches'] != 2 or r['contrast_weight'] != 1:
                raise ValueError('Changed actual branch count or weight')
            neg = vec(r['native_class_logits_blocked']); combined = pos+(pos-neg)
            actual = {(int(l), h) for l, d in r['attention_diagnostics'].items() for h in d['heads']}
            if not actual or (heads is not None and actual != heads): raise ValueError('Positive head selection differs')
            for d in r['attention_diagnostics'].values():
                if (d.get('bias') != 6 or not d.get('causal_mask_preserved') or not d.get('all_query_rows')
                        or d.get('generated_text_key_bias') != 0 or d['prefill_calls'] != 1):
                    raise ValueError('Actual original-bias positive differs')
            blocked = r['blocked_attention_diagnostics']
            if set(blocked) != {str(i) for i in range(36)}: raise ValueError('Not all actual language layers blocked')
            for d in blocked.values():
                if (d['method'] != 'task_key_block' or d['heads'] != list(range(32)) or d['roi_intervention']
                        or not d['causal_mask_preserved'] or not d['all_query_rows'] or d['prefill_calls'] != 1
                        or not d['fully_masked_rows_return_zero'] or min(d['task_token_counts']) <= 0):
                    raise ValueError('Actual negative must isolate T on all heads without ROI')
            if r['actual_probe_forward_branches'] != int(probe): raise ValueError('Unregistered probe cost')
            a = r['invariance_probe']
            if probe:
                if (not a or not a['negative_logits_exact'] or not a['original_masks_and_rotary_positions_exact']
                        or a['negative_head_diagnostics'] != blocked
                        or not np.array_equal(neg, vec(a['native_probe_logits'][a['batch_row']]))
                        or a['original_input_ids'] == a['perturbed_input_ids']
                        or a['actual_layout_observation'][0]['position_embeddings'] is None):
                    raise ValueError('Actual fixed-layout content-isolation probe did not pass')
            elif a is not None:
                raise ValueError('Probe must not silently enter the primary experiment')
        if not np.array_equal(combined, vec(r['native_class_logits_combined'])):
            raise ValueError('Combined logits do not reconstruct exactly')
        p = np.exp(combined.astype(float)-float(combined.max())); p /= p.sum()
        if (np.max(np.abs(p-vec(r['native_class_probabilities']))) >= 1e-5 or r['reward'] != int(p.argmax())+1
                or not np.isfinite(r['progress']) or r['progress'] != (r['reward']-1)/4):
            raise ValueError('Native probabilities and five-way output do not match the fixed formula')
    return rows


def complete(root, ks, scopes, n):
    request = root/'requested_ids.json'
    if not request.exists(): return False
    ids = json.loads(request.read_text())
    if len(ids) != n or len(set(ids)) != n: return False
    files = [root/'predictions/baseline.jsonl']+[root/'bias_s6/predictions'/f'{s}_target_{k}.jsonl' for s in scopes for k in ks]
    return all(p.exists() and set(latest(p)) == set(ids) for p in files)


def command(model, protocols, population, ks, scopes=SCOPES, probe=False):
    result = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
        'mydata_bench.auto_research_addbase.worker', '--model', model, '--protocols', *protocols,
        '--variant', VARIANT+('_probe' if probe else ''), '--methods', 'bias', '--strengths', '6',
        '--ranking-prefix', 'ANSWER: ', '--contrast-weight', '1', '--contrast-negative-mode', 'task_content_block',
        '--ks', *map(str, ks), '--scopes', *scopes, '--population', population,
        '--frozen-ranking-root', str(rank_path(model, protocols[0], SCOPES[0]).parents[1])]
    if probe: result += ['--content-block-probe', '--limit', '8']
    return result


def write_addition(name, jobs, sources):
    existing = include_additions(json.loads((QUEUE/'plan.json').read_text()), QUEUE)
    validate_plan(dict(jobs=list(existing.values())+jobs))
    path = QUEUE/'additions'/name
    create_json(path, dict(jobs=jobs, sources_sha256=sources,
        interpretation='Round28 fixed two-forward original-bias/task-content isolation; physical GPU0/1 only'))
    with path.with_suffix('.ready').open('x') as f: f.write('Registered complete prior gates; only GPU0/1\n')
    return path


def register_smoke(cpu_audit):
    a = json.loads(cpu_audit.read_text())
    if a['status'] != 'pass' or a['labels_read'] or a['content_block_tests'] != 10:
        raise ValueError('Require task isolation and previous-method CPU tests before actual probe')
    for path, digest in a['source_code_sha256'].items():
        p = Path(path).resolve()
        if not p.is_relative_to(Path(__file__).resolve().parent) or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Current task source differs from CPU audit')
    jobs = [dict(name=f'smoke_{m}_task_content_block', gpu=i, min_free_mb=23000, depends_on=[],
        command=command(m, ['image_text', 'text_video'], 'discovery', [8, 32], probe=True)) for i, m in enumerate(MODELS)]
    return write_addition('stage28_task_content_block_smoke.json', jobs, {str(cpu_audit): sha(cpu_audit), str(POLICY): sha(POLICY)})


def smoke_ready():
    return all(complete(folder(m, p, 'discovery_smoke', True), [8, 32], SCOPES, 8)
        for m in MODELS for p in ['image_text', 'text_video'])


def verify_smoke(destination):
    if not smoke_ready(): raise ValueError('Wait for the entire actual probe matrix')
    sources = {str(POLICY): sha(POLICY)}; checks = []
    def read(p, rows=False):
        sources[str(p)] = sha(p); return latest(p) if rows else json.loads(p.read_text())
    for m in MODELS:
        for p in ['image_text', 'text_video']:
            root = folder(m, p, 'discovery_smoke', True)
            reference = OUT/'experiments'/f'{m}_{p}_bias_task_anchor_evidence_a1'/'discovery'
            ids = read(root/'requested_ids.json')
            if ids != read(reference/'requested_ids.json')[:8]: raise ValueError('Changed smoke population/order')
            config_check(m, p, read(root/'runtime_config.json'), True)
            base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, baseline=True)
            old_base = read(reference/'predictions/baseline.jsonl', True)
            negative = None
            for e in ids:
                if not same_input(base[e], old_base[e]) or base[e]['native_class_logits_positive'] != old_base[e]['native_class_logits_positive']:
                    raise ValueError('Baseline does not reproduce the existing actual batch')
            for s in SCOPES:
                ranking = read(rank_path(m, p, s))['ranking']
                for k in [8, 32]:
                    rows = verify_rows(read(root/'bias_s6/predictions'/f'{s}_target_{k}.jsonl', True), ids,
                        {(h['layer'], h['head']) for h in ranking[:k]}, probe=True)
                    old = read(reference/'binding_transport_s4/predictions'/f'{s}_target_{k}.jsonl', True)
                    for e in ids:
                        if (not same_input(rows[e], base[e]) or not same_input(rows[e], old[e])
                                or rows[e]['native_class_logits_positive'] != old[e]['native_cell_logits']['bias']
                                or rows[e]['condition'] != f'{s}:target:{k}'):
                            raise ValueError('Actual positive bias or same input does not replay the matching existing branch')
                    n = {e: rows[e]['native_class_logits_blocked'] for e in ids}
                    if negative is not None and n != negative: raise ValueError('Negative unexpectedly depends on positive ROI/head selection')
                    negative = n
                    checks.append(dict(model=m, protocol=p, scope=s, k=k, n=8, baseline_and_bias_replays_exact=True,
                        content_replacement_probe_exact=True, all_negative_heads_and_formula_verified=True))
    create_json(destination, dict(status='pass', labels_read=False, checks=checks, sources_sha256=sources,
        actual_intervention_rows_verified=128, primary_forward_observations=256, extra_probe_forward_observations=128,
        interpretation='Actual fixed-layout instruction isolation; implementation/replay audit, no efficacy scoring'))


def register_discovery(audit):
    a = json.loads(audit.read_text())
    if a['status'] != 'pass' or a['labels_read'] or len(a['checks']) != 16: raise ValueError('Incomplete actual probe gate')
    verify_sources(a['sources_sha256'])
    jobs = [dict(name=f'round28_{m}_task_content_block_discovery', gpu=i, min_free_mb=23000,
        depends_on=[f'smoke_{m}_task_content_block'], command=command(m, PROTOCOLS, 'discovery', [8, 32, 64]))
        for i, m in enumerate(MODELS)]
    return write_addition('stage28_task_content_block_discovery.json', jobs, {str(audit): sha(audit), str(POLICY): sha(POLICY)})


def matrix_ready():
    return all(complete(folder(m, p, 'discovery'), [8, 32, 64], SCOPES, 70) for m in MODELS for p in PROTOCOLS)


def select(destination):
    if not matrix_ready(): raise ValueError('Require both entire five-input actual matrices')
    spec = json.loads(POLICY.read_text()); splits = json.loads((OUT/'splits.json').read_text()); ids = splits['discovery']
    if (len(ids) != 70 or set(ids) & set(splits['validation']) or spec['variant'] != VARIANT
            or spec['protocols'] != PROTOCOLS or spec['composition'] != '2*original_bias6-task_content_blocked'
            or spec['contrast_weight'] != 1): raise ValueError('Frozen policy or split differs')
    sources = {str(POLICY): sha(POLICY)}; matrices = {}
    def read(path, rows=False):
        sources[str(path)] = sha(path); return latest(path) if rows else json.loads(path.read_text())
    for m in MODELS:
        for p in PROTOCOLS:
            root = folder(m, p, 'discovery'); config_check(m, p, read(root/'runtime_config.json'), False)
            if set(read(root/'requested_ids.json')) != set(ids): raise ValueError('Changed discovery population')
            base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, baseline=True)
            conditions = {}; negative = None
            expected = {f'{s}_target_{k}.jsonl' for s in SCOPES for k in NEIGHBORHOODS}
            if {f.name for f in (root/'bias_s6/predictions').glob('*.jsonl')} != expected:
                raise ValueError('Require exactly the frozen six conditions per input')
            for s in SCOPES:
                ranking = read(rank_path(m, p, s))['ranking']
                if read(root.parent/'ranking'/f'ranking_{s}.json')['ranking'] != ranking:
                    raise ValueError('Actual copied ranking differs')
                for k in NEIGHBORHOODS:
                    heads = {(h['layer'], h['head']) for h in ranking[:k]}
                    if len(heads) != k: raise ValueError('Duplicate positive heads')
                    rows = verify_rows(read(root/'bias_s6/predictions'/f'{s}_target_{k}.jsonl', True), ids, heads)
                    if any(not same_input(r, base[e]) or r['condition'] != f'{s}:target:{k}' for e, r in rows.items()):
                        raise ValueError('Actual same-batch input or condition differs')
                    n = {e: r['native_class_logits_blocked'] for e, r in rows.items()}
                    if negative is not None and negative != n: raise ValueError('Negative depends on positive condition')
                    negative = n; conditions[(s, k)] = rows
            matrices[(m, p)] = base, conditions
    # Only after all actual rows, native formulas and configurations were verified.
    labels = json.loads((OLD/'labels_for_scoring_only.json').read_text()); labels = {e: labels[e] for e in ids}
    points = []; proposed = []
    for (m, p), (baseline, conditions) in matrices.items():
        base = summary(baseline, labels, ids); passing = []
        for (s, k), rows in conditions.items():
            score = summary(rows, labels, ids)
            delta = {c: score['accuracy']['0.125/0.875'][c]['rate_all_expected']-base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all', 'suc', 'fail']}
            gate = score['mae'] < base['mae'] and delta['all'] >= .1-1e-12 and min(delta['suc'], delta['fail']) > 0
            point = dict(model=m, protocol=p, scope=s, center_k=k, variant=VARIANT, method='bias_s6',
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
        selected=proposed if eligible else [], sources_sha256=sources, validation_labels_used=[],
        actual_intervention_rows_verified=4200, independent_conditions=60, actual_forward_observations=8400,
        interpretation='One fixed class-symmetric weight and current sample only; dataset-internal adaptive exploration'))


def register_full(selection, audit):
    record = json.loads(selection.read_text()); a = json.loads(audit.read_text())
    eligible = min(record['input_counts'].values()) >= 3
    if (record['family_eligible_for_full'] != eligible or record['selected'] != (record['proposed_input_candidates'] if eligible else [])
            or a['status'] != 'pass' or a['labels_read']): raise ValueError('Changed shared selection or actual gate')
    verify_sources(dict(record['sources_sha256'], **a['sources_sha256']))
    jobs = []; previous = {}
    for p in record['selected']:
        m = p['model']; name = f"validate_{m}_{p['protocol']}_task_content_block"
        jobs.append(dict(name=name, gpu=MODELS.index(m), min_free_mb=23000,
            depends_on=[previous.get(m, f'round28_{m}_task_content_block_discovery')],
            command=command(m, [p['protocol']], 'full_cohort', p['ks'], [p['scope']])))
        previous[m] = name
    return write_addition('stage28_task_content_block_validation.json', jobs, {str(selection): sha(selection), str(audit): sha(audit)})
