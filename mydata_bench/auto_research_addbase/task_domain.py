"""CPU verification and frozen selection for round22 pure instruction contrast."""
import json
import time

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .select_robust_discovery import PROTOCOLS, NEIGHBORHOODS, verify_native
from .select_temporal_discovery import completed
from .empirical_profile import sha, verify_sources
from .durable_scheduler import include_additions, validate_plan


MODELS = ['qwen', 'roboreward']
VARIANT = 'task_domain_evidence_a1'
POLICY = OUT / 'selection_task_domain_discovery_v1.json'
QUEUE = OUT / 'queue_gpu01_20260911_1700'


def write_addition(name, jobs, sources):
    existing=include_additions(json.loads((QUEUE / 'plan.json').read_text()), QUEUE)
    validate_plan(dict(jobs=list(existing.values())+jobs))
    path=QUEUE / 'additions' / name
    create_json(path, dict(jobs=jobs, sources_sha256=sources,
        interpretation='Registered round22 pure instruction-domain contrast; physical GPU0/1 only'))
    if not path.with_suffix('.ready').exists():
        with path.with_suffix('.ready').open('x') as f:
            f.write('Completed sources verified; registered round22 gates; physical GPU0/1 only\n')
    return path


def folder(model, protocol, population):
    return OUT / 'experiments' / f'{model}_{protocol}_{VARIANT}' / population


def rank_path(model, protocol):
    return OUT / 'functional_selections' / f'stage8_{model}_v1' / f'{model}_{protocol}' / 'ranking_all_frames.json'


def verify_rows(rows, ids, heads=None, baseline=False):
    verify_native(rows, ids, baseline, heads, negative_mode='task')
    for row in rows.values():
        if (row.get('visual_intervention') != 'bypassed' or row.get('positive_visual_strength') != 0
                or row.get('negative_visual_strength') != 0):
            raise ValueError('Actual output does not declare the pure-task visual bypass')
        if baseline:
            if row.get('actual_forward_branches') != 1:
                raise ValueError('Baseline must be one unsteered forward')
            continue
        if (row.get('actual_forward_branches') != 2 or row.get('negative_branch_kind') != 'task'
                or row.get('negative_strength_domain') != 'task' or row.get('negative_task_strength') != 4
                or not row.get('task_negative_is_actual') or row.get('contrast_weight') != 1
                or row['native_class_logits_negative_task'] != row['native_class_logits_negative']):
            raise ValueError('Require one actual task-negative branch and the fixed alpha1 formula')
        for field, method in [('attention_diagnostics', 'binding_transport'),
                              ('negative_attention_diagnostics', 'binding_task_suppression')]:
            for d in row[field].values():
                if (d.get('method') != method or d.get('strength') != 0
                        or d.get('visual_intervention') != 'bypassed'
                        or not d.get('visual_weights_unchanged_locally') or not d.get('text_domain_mass_preserved')):
                    raise ValueError('Actual controller did not execute the registered pure-task branch')
                if field == 'attention_diagnostics':
                    if d.get('task_binding_fraction') != .5 or d.get('task_binding_distribution') != 'uniform':
                        raise ValueError('Actual positive text operator differs')
                elif d.get('task_logit_strength') != -4 or d.get('task_binding_distribution') != 'exponential_suppression':
                    raise ValueError('Actual negative text operator differs')
    return rows


def same_input(a, b):
    return all(a[k] == b[k] for k in ['prompt', 'candidate_token_ids']) and (
        a['token_audit']['input_ids_sha256'] == b['token_audit']['input_ids_sha256'])


def config_check(model, protocol, cfg):
    expected = dict(contrast_negative_mode='task', contrast_weight=1, negative_task_strength=4,
        task_binding_fraction=.5, task_binding_distribution='uniform', ranking_prefix='ANSWER: ',
        frozen_ranking_source=str(rank_path(model, protocol).parent))
    if any(cfg.get(k) != v for k, v in expected.items()) or cfg.get('visual_mass_partition', 'global') != 'global':
        raise ValueError('Frozen pure instruction method differs')


def command(model, protocols, population, ks, controls=('target',), smoke=False):
    result = ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
        'mydata_bench.auto_research_addbase.worker', '--model', model, '--protocols', *protocols,
        '--variant', VARIANT, '--methods', 'binding_transport', '--task-binding-fraction', '.5',
        '--task-binding-distribution', 'uniform', '--ranking-prefix', 'ANSWER: ', '--contrast-weight', '1',
        '--contrast-negative-mode', 'task', '--negative-task-strength', '4', '--strengths', '0',
        '--ks', *map(str, ks), '--scopes', 'all_frames', '--controls', *controls, '--population', population,
        '--frozen-ranking-root', str(rank_path(model, protocols[0]).parents[1])]
    if smoke:
        result += ['--limit', '8']
    return result


def register_smoke(cpu_audit):
    audit = json.loads(cpu_audit.read_text())
    if audit['status'] != 'pass' or audit['labels_read'] or audit['regression_tests_run'] != 26:
        raise ValueError('CPU numerical and prior-method regression checks must pass')
    for path, digest in audit['source_code_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Task-domain implementation changed after its CPU audit')
    verify_sources({str(POLICY):sha(POLICY)})
    jobs = [dict(name=f'smoke_{m}_task_domain', gpu=i, min_free_mb=23000, depends_on=[],
        command=command(m, ['image_text', 'text_video'], 'discovery', [8, 32],
                        ['target', 'wrong_region'], smoke=True)) for i, m in enumerate(MODELS)]
    return write_addition('stage22_task_domain_smoke.json', jobs, {str(cpu_audit):sha(cpu_audit), str(POLICY):sha(POLICY)})


def smoke_ready():
    return all(completed(folder(m, p, 'discovery_smoke'))
               for m in MODELS for p in ['image_text', 'text_video'])


def verify_smoke(destination):
    if not smoke_ready():
        raise ValueError('Wait for both models and both complete pure-task smoke inputs')
    sources, checks = {str(POLICY):sha(POLICY)}, []
    def read(path, rows=False):
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    for model in MODELS:
        for protocol in ['image_text', 'text_video']:
            root = folder(model, protocol, 'discovery_smoke')
            reference = OUT / 'experiments' / f'{model}_{protocol}_uniform_binding_evidence_a1' / 'discovery'
            ids = read(root / 'requested_ids.json')
            if len(ids) != 8 or ids != read(reference / 'requested_ids.json')[:8]:
                raise ValueError('Require the original first discovery batch')
            config_check(model, protocol, read(root / 'runtime_config.json'))
            base = verify_rows(read(root / 'predictions/baseline.jsonl', True), ids, baseline=True)
            old = read(reference / 'predictions/baseline.jsonl', True)
            if any(not same_input(base[e], old[e]) or base[e]['native_class_logits_positive'] != old[e]['native_class_logits_positive'] for e in ids):
                raise ValueError('Unsteered actual baseline differs from the original batch')
            rank = read(rank_path(model, protocol))['ranking']
            paths = list(root.glob('binding_transport_s0/predictions/*.jsonl'))
            if {p.name for p in paths} != {f'all_frames_{c}_{k}.jsonl' for c in ['target', 'wrong_region'] for k in [8, 32]}:
                raise ValueError('Require exactly four smoke conditions')
            for k in [8, 32]:
                heads = {(h['layer'], h['head']) for h in rank[:k]}
                target = verify_rows(read(root / 'binding_transport_s0/predictions' / f'all_frames_target_{k}.jsonl', True), ids, heads)
                wrong = verify_rows(read(root / 'binding_transport_s0/predictions' / f'all_frames_wrong_region_{k}.jsonl', True), ids, heads)
                for eid in ids:
                    a, b = target[eid], wrong[eid]
                    if not same_input(a, b) or not same_input(a, base[eid]):
                        raise ValueError('Actual ROI-condition inputs differ')
                    for field in ['native_class_logits_positive', 'native_class_logits_negative', 'native_class_probabilities']:
                        if a[field] != b[field]:
                            raise ValueError('Explicitly bypassed visual ROI changes the actual task contrast')
                    for row, kind in [(a, 'target'), (b, 'wrong_region')]:
                        if row['condition'] != f'all_frames:{kind}:{k}':
                            raise ValueError('Recorded smoke condition differs')
                    mask=a['token_audit']; t=set(mask['target']['all_frames']); w=set(mask['wrong']['all_frames'])
                    if not t or len(t) != len(w) or t & w:
                        raise ValueError('Accepted control masks are not distinct equal-size sets')
                checks.append(dict(model=model, protocol=protocol, k=k, rows_per_control=8,
                    target_wrong_actual_branches_exact=True, baseline_exact=True,
                    actual_head_ids_native_formula_and_bypass_verified=True))
    create_json(destination, dict(status='pass', labels_read=False, checks=checks,
        actual_intervention_rows_verified=128, sources_sha256=sources,
        interpretation='Actual same-head ROI invariance, baseline replay and executed text operators verified. '
            'Local probability conservation was tested numerically on CPU; diagnostics do not measure every actual model query. No efficacy scoring.'))


def register_discovery(audit_path):
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'pass' or audit['labels_read'] or len(audit['checks']) != 8:
        raise ValueError('All actual pure-task smoke comparisons must pass')
    verify_sources(audit['sources_sha256'])
    jobs = [dict(name=f'round22_{m}_task_domain_discovery', gpu=i, min_free_mb=23000,
        depends_on=[f'smoke_{m}_task_domain'], command=command(m, PROTOCOLS, 'discovery', [8, 32, 64]))
        for i, m in enumerate(MODELS)]
    return write_addition('stage22_task_domain_discovery.json', jobs, {str(audit_path):sha(audit_path), str(POLICY):sha(POLICY)})


def matrix_ready():
    return all(completed(folder(m, p, 'discovery')) for m in MODELS for p in PROTOCOLS)


def select(destination):
    if not matrix_ready():
        raise ValueError('Wait for both complete five-input task-domain discovery matrices')
    splits = json.loads((OUT / 'splits.json').read_text()); ids = splits['discovery']
    if len(ids) != 70 or set(ids) & set(splits['validation']):
        raise ValueError('Discovery boundary differs')
    spec = json.loads(POLICY.read_text())
    if spec['variant'] != VARIANT or spec['discovery_ks'] != [8, 32, 64] or spec['scopes'] != ['all_frames']:
        raise ValueError('Frozen task-domain policy differs')
    sources, matrices = {str(POLICY):sha(POLICY)}, {}
    def read(path, predictions=False):
        sources[str(path)] = sha(path)
        return latest(path) if predictions else json.loads(path.read_text())
    for model in MODELS:
        for protocol in PROTOCOLS:
            root = folder(model, protocol, 'discovery')
            config_check(model, protocol, read(root / 'runtime_config.json'))
            if set(read(root / 'requested_ids.json')) != set(ids):
                raise ValueError('Actual pure-task discovery population differs')
            rank = read(rank_path(model, protocol))['ranking']
            base = verify_rows(read(root / 'predictions/baseline.jsonl', True), ids, baseline=True)
            paths = list(root.glob('binding_transport_s0/predictions/*.jsonl'))
            if {p.name for p in paths} != {f'all_frames_target_{k}.jsonl' for k in NEIGHBORHOODS}:
                raise ValueError('Require exactly three independent task-domain conditions per input')
            conditions = {}
            for k in NEIGHBORHOODS:
                heads = {(h['layer'], h['head']) for h in rank[:k]}
                if len(heads) != k:
                    raise ValueError('Duplicate fixed head')
                rows = verify_rows(read(root / 'binding_transport_s0/predictions' / f'all_frames_target_{k}.jsonl', True), ids, heads)
                if any(not same_input(rows[e], base[e]) for e in ids):
                    raise ValueError('Input differs from same-batch unsteered baseline')
                conditions[k] = rows
            matrices[(model, protocol)] = (base, conditions)
    # All ten inputs and all 2100 actual intervention rows precede scoring labels.
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {eid:all_labels[eid] for eid in ids}; del all_labels
    points, proposed = [], []
    for (model, protocol), (baseline, conditions) in matrices.items():
        base = summary(baseline, labels, ids); passing = []
        for k, rows in sorted(conditions.items()):
            score = summary(rows, labels, ids)
            delta = {c:score['accuracy']['0.125/0.875'][c]['rate_all_expected']-
                base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all', 'suc', 'fail']}
            gate = score['mae'] < base['mae'] and delta['all'] >= .1-1e-12 and min(delta['suc'], delta['fail']) > 0
            point = dict(model=model, protocol=protocol, variant=VARIANT, method='binding_transport_s0',
                scope='all_frames', center_k=k, mae=score['mae'], baseline_mae=base['mae'], deltas=delta,
                accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'], passes=gate)
            points.append(point)
            if gate:
                passing.append(point)
        if passing:
            best = min(passing, key=lambda p:(p['mae'], -p['accuracy_all'], p['center_k']))
            proposed.append(dict(best, ks=NEIGHBORHOODS[best['center_k']]))
    counts = {m:sum(p['model'] == m for p in proposed) for m in MODELS}; eligible = min(counts.values()) >= 3
    create_json(destination, dict(created_at=time.time(), variant=VARIANT, all_discovery_points=points,
        proposed_input_candidates=proposed, input_counts=counts, family_eligible_for_full=eligible,
        selected=proposed if eligible else [], sources_sha256=sources, validation_labels_used=[],
        independent_conditions=30, actual_intervention_rows_verified=2100,
        interpretation='One shared text operator and one fixed head source per input; no visual ROI redistribution. '
            'All-frames is only the legacy ranking-source name. Joint frozen family screen; no cross-method pooling.'))


def register_full(selection, audit_path):
    record=json.loads(selection.read_text()); audit=json.loads(audit_path.read_text())
    if (record['family_eligible_for_full'] != (min(record['input_counts'].values()) >= 3) or
        record['selected'] != (record['proposed_input_candidates'] if record['family_eligible_for_full'] else [])):
        raise ValueError('Joint pure-task family selection differs')
    if audit['status'] != 'pass' or audit['labels_read'] or len(audit['checks']) != 8:
        raise ValueError('Actual smoke has not passed')
    verify_sources(dict(record['sources_sha256'], **audit['sources_sha256']))
    jobs, previous = [], {}
    for p in record['selected']:
        m=p['model']; name=f"validate_{m}_{p['protocol']}_task_domain"
        jobs.append(dict(name=name, gpu=MODELS.index(m), min_free_mb=23000,
            depends_on=[previous.get(m, f'round22_{m}_task_domain_discovery')],
            command=command(m, [p['protocol']], 'full_cohort', p['ks'])))
        previous[m]=name
    return write_addition('stage22_task_domain_validation.json', jobs,
                          {str(selection):sha(selection), str(audit_path):sha(audit_path)})
