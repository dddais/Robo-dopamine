"""Registered round21 empirical-ranking ablation of the exact round18 profiles.

Only completed artifacts enter ranking and selection. Model inference stays in
the existing worker; no class-conditional prediction rule is introduced here.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .select_robust_discovery import PROTOCOLS, NEIGHBORHOODS, verify_native
from .select_temporal_discovery import completed
from .durable_scheduler import include_additions, validate_plan


MODELS = ['qwen', 'roboreward']
SCOPES = ['all_frames', 'last_frame']
VARIANT = 'uniform_evidence_empiricalprofile_a1'
POLICY = OUT / 'selection_method_matched_empirical_discovery_v1.json'
QUEUE = OUT / 'queue_gpu01_20260911_1700'
AUDIT = OUT / 'audit/matched_profile_actual_20260911_184717.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(sources):
    for path, digest in sources.items():
        if not Path(path).resolve().is_relative_to(OUT.resolve()) or sha(path) != digest:
            raise ValueError('Frozen source changed or is outside this session')


def rank_root(model):
    return OUT / 'functional_selections' / f'stage21_{model}_empirical_v1'


def profiles_ready():
    return all(completed(OUT / 'experiments' / f'{m}_{p}_uniform_method_profile' / 'discovery')
        and (OUT / 'functional_selections' / f'stage18_{m}_robust_v1' / f'{m}_{p}' /
             f'ranking_{s}.json').exists() for m in MODELS for p in PROTOCOLS for s in SCOPES)


def empirical_rank(value):
    """Change only the group order, with no filtering or within-group reranking."""
    profiles = value['layer_profiles']
    if (len(profiles) != 28 or {p['layer'] for p in profiles} != set(range(8, 36))
            or value['validation_ids_used'] or value['n'] != 70):
        raise ValueError('Require all 28 discovery-only profiles')
    for p in profiles:
        if (len(p['heads']) != 8 or {h['layer'] for h in p['heads']} != {p['layer']}
                or not np.isfinite([p['delta_nll_suc'], p['delta_nll_fail']]).all()):
            raise ValueError('Invalid finite empirical loss or fixed layer group')
    order = sorted(profiles, key=lambda p: (max(p['delta_nll_suc'], p['delta_nll_fail']),
        (p['delta_nll_suc'] + p['delta_nll_fail']) / 2, p['layer']))
    heads = [dict(h, score=-max(p['delta_nll_suc'], p['delta_nll_fail']),
        profile_layer=p['layer'], selection_source='discovery_method_matched_empirical_layer_profile')
        for p in order for h in p['heads']]
    if len({(h['layer'], h['head']) for h in heads}) != 224:
        raise ValueError('Require all 224 unique frozen heads')
    return order, heads


def build_rankings():
    if not profiles_ready():
        raise ValueError('Wait for both complete five-input matched profiles and frozen robust rankings')
    spec = json.loads(POLICY.read_text())
    if spec['variant'] != VARIANT or spec['protocols'] != PROTOCOLS or spec['discovery_ks'] != [8, 32, 64]:
        raise ValueError('Registered empirical policy differs')
    pending = []
    for model in MODELS:
        for protocol in PROTOCOLS:
            for scope in SCOPES:
                source = OUT / 'functional_selections' / f'stage18_{model}_robust_v1' / f'{model}_{protocol}' / f'ranking_{scope}.json'
                value = json.loads(source.read_text())
                verify_sources(value['sources_sha256'])
                order, heads = empirical_rank(value)
                record = dict(value, ranking=heads, layer_profiles=order,
                    ranking_score='negative max empirical class delta NLL; balanced empirical NLL then layer tie-break',
                    supervision='Same discovery70 native five-class NLL and exact matched profile forwards as round18',
                    selector_source_sha256=sha(__file__), policy=str(POLICY), policy_sha256=sha(POLICY),
                    robust_ranking_source=str(source), robust_ranking_sha256=sha(source),
                    sources_sha256=dict(value['sources_sha256'], **{str(source):sha(source), str(POLICY):sha(POLICY)}),
                    interpretation='Empirical ablation only; q90 and CI metadata are retained from the original profiles but do not determine this order')
                pending.append((rank_root(model) / f'{model}_{protocol}' / f'ranking_{scope}.json', record))
    # Validate every source before emitting any empirical ranking.
    for path, value in pending:
        create_json(path, value)
    return [path for path, _ in pending]


def command(model, protocols, population, ks, scopes):
    return ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
        'mydata_bench.auto_research_addbase.worker', '--model', model, '--protocols', *protocols,
        '--variant', VARIANT, '--methods', 'binding_transport', '--task-binding-fraction', '.5',
        '--task-binding-distribution', 'uniform', '--ranking-prefix', 'ANSWER: ', '--contrast-weight', '1',
        '--negative-strength', '4', '--strengths', '4', '--ks', *map(str, ks), '--scopes', *scopes,
        '--population', population, '--frozen-ranking-root', str(rank_root(model))]


def write_addition(name, jobs, sources):
    existing = include_additions(json.loads((QUEUE / 'plan.json').read_text()), QUEUE)
    validate_plan(dict(jobs=list(existing.values()) + jobs))
    path = QUEUE / 'additions' / name
    create_json(path, dict(jobs=jobs, sources_sha256=sources,
        interpretation='Registered empirical selector ablation; same actual method; physical GPU0/1 only'))
    if not path.with_suffix('.ready').exists():
        with path.with_suffix('.ready').open('x') as f:
            f.write('Completed sources verified; registered round21 gates; physical GPU0/1 only\n')
    return path


def register_discovery():
    paths = build_rankings()
    audit = json.loads(AUDIT.read_text())
    if audit['status'] != 'pass' or audit['labels_read'] or len(audit['checks']) != 4:
        raise ValueError('Require both models original exact-method actual smoke')
    verify_sources(audit['sources_sha256'])
    sources = {str(path):sha(path) for path in paths + [POLICY, AUDIT]}
    jobs = [dict(name=f'round21_{m}_empirical_profile_discovery', gpu=i, min_free_mb=23000,
        command=command(m, PROTOCOLS, 'discovery', [8, 32, 64], SCOPES), depends_on=[])
        for i, m in enumerate(MODELS)]
    return write_addition('stage21_empirical_profile_discovery.json', jobs, sources)


def matrix_ready():
    return all(completed(OUT / 'experiments' / f'{m}_{p}_{VARIANT}' / 'discovery')
               for m in MODELS for p in PROTOCOLS)


def same_actual(a, b):
    fields = ['prompt', 'candidate_token_ids', 'native_class_logits_positive', 'native_class_logits_negative']
    return all(a.get(k) == b.get(k) for k in fields) and (
        a['token_audit']['input_ids_sha256'] == b['token_audit']['input_ids_sha256'])


def select(destination):
    if not matrix_ready():
        raise ValueError('Wait for both complete five-input empirical discovery matrices')
    ids = json.loads((OUT / 'splits.json').read_text())['discovery']
    if len(ids) != 70:
        raise ValueError('Require original discovery70')
    sources = {str(POLICY):sha(POLICY)}
    def read(path):
        sources[str(path)] = sha(path)
        return latest(path)
    matrices = {}
    for model in MODELS:
        for protocol in PROTOCOLS:
            folder = OUT / 'experiments' / f'{model}_{protocol}_{VARIANT}' / 'discovery'
            profile = OUT / 'experiments' / f'{model}_{protocol}_uniform_method_profile' / 'discovery'
            cfg_path = folder / 'runtime_config.json'; sources[str(cfg_path)] = sha(cfg_path)
            cfg = json.loads(cfg_path.read_text())
            expected = dict(contrast_weight=1, negative_strength=4, bias=4, task_binding_fraction=.5,
                task_binding_distribution='uniform', ranking_prefix='ANSWER: ',
                frozen_ranking_source=str(rank_root(model) / f'{model}_{protocol}'))
            if any(cfg.get(k) != v for k, v in expected.items()) or cfg.get('visual_mass_partition', 'global') != 'global':
                raise ValueError('Frozen empirical method differs')
            for name in ['requested_ids.json', 'worker_events.jsonl']:
                sources[str(folder / name)] = sha(folder / name)
            if set(json.loads((folder / 'requested_ids.json').read_text())) != set(ids):
                raise ValueError('Empirical discovery population differs')
            baseline = verify_native(read(folder / 'predictions/baseline.jsonl'), ids, True)
            original_base = verify_native(read(profile / 'predictions/baseline.jsonl'), ids, True)
            if any(not same_actual(baseline[e], original_base[e]) for e in ids):
                raise ValueError('Unsteered baseline differs from same-batch actual profile')
            paths = list(folder.glob('binding_transport_s4/predictions/*.jsonl'))
            if {p.name for p in paths} != {f'{s}_target_{k}.jsonl' for s in SCOPES for k in NEIGHBORHOODS}:
                raise ValueError('Require exactly six complete empirical conditions')
            conditions = {}
            for scope in SCOPES:
                rp = rank_root(model) / f'{model}_{protocol}' / f'ranking_{scope}.json'
                sources[str(rp)] = sha(rp); rank = json.loads(rp.read_text())
                verify_sources(rank['sources_sha256'])
                if set(rank['example_ids']) != set(ids) or rank['validation_ids_used']:
                    raise ValueError('Ranking supervision boundary differs')
                _, reconstructed = empirical_rank(rank)
                if reconstructed != rank['ranking']:
                    raise ValueError('Frozen empirical group order differs')
                for k in NEIGHBORHOODS:
                    heads = {(h['layer'], h['head']) for h in rank['ranking'][:k]}
                    rows = verify_native(read(folder / 'binding_transport_s4/predictions' / f'{scope}_target_{k}.jsonl'), ids, expected_heads=heads)
                    if any(rows[e]['token_audit']['input_ids_sha256'] != baseline[e]['token_audit']['input_ids_sha256'] for e in ids):
                        raise ValueError('Input token batch differs from baseline')
                    if k == 8:
                        layer = rank['ranking'][0]['layer']
                        original = verify_native(read(profile / f'binding_transport_s4_layer{layer}' /
                            f'predictions/{scope}_target_8.jsonl'), ids, expected_heads=heads)
                        if any(not same_actual(rows[e], original[e]) for e in ids):
                            raise ValueError('Top8 actual branch replay differs from the matching profile')
                    conditions[(scope, k)] = rows
            matrices[(model, protocol)] = (baseline, conditions)
    # Both complete matrices and 1400 top8/700 baseline replay rows before labels.
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {e:all_labels[e] for e in ids}; del all_labels
    points, proposed = [], []
    for (model, protocol), (baseline, conditions) in matrices.items():
        base = summary(baseline, labels, ids); passing = []
        for (scope, k), rows in sorted(conditions.items()):
            score = summary(rows, labels, ids)
            delta = {c:score['accuracy']['0.125/0.875'][c]['rate_all_expected'] -
                base['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all', 'suc', 'fail']}
            gate = score['mae'] < base['mae'] and delta['all'] >= .1 - 1e-12 and min(delta['suc'], delta['fail']) > 0
            point = dict(model=model, protocol=protocol, variant=VARIANT, method='binding_transport_s4',
                scope=scope, center_k=k, mae=score['mae'], baseline_mae=base['mae'], deltas=delta,
                accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'], passes=gate)
            points.append(point)
            if gate:
                passing.append(point)
        if passing:
            best = min(passing, key=lambda p:(p['mae'], -p['accuracy_all'], p['center_k'], p['scope']))
            proposed.append(dict(best, ks=NEIGHBORHOODS[best['center_k']]))
    counts = {m:sum(p['model'] == m for p in proposed) for m in MODELS}
    eligible = min(counts.values()) >= 3
    create_json(destination, dict(created_at=time.time(), variant=VARIANT, all_discovery_points=points,
        proposed_input_candidates=proposed, input_counts=counts, family_eligible_for_full=eligible,
        selected=proposed if eligible else [], sources_sha256=sources, validation_labels_used=[],
        actual_intervention_rows_verified=4200, actual_top8_profile_replays_exact=1400,
        actual_baseline_profile_replays_exact=700,
        interpretation='Registered joint screen; empirical and robust families remain separate. Replays are implementation checks, not independent efficacy replications. Adaptive dataset-internal exploration.'))


def register_full(selection):
    record = json.loads(selection.read_text()); verify_sources(record['sources_sha256'])
    if (record['family_eligible_for_full'] != (min(record['input_counts'].values()) >= 3) or
            record['selected'] != (record['proposed_input_candidates'] if record['family_eligible_for_full'] else [])):
        raise ValueError('Joint empirical family gate differs')
    jobs, previous = [], {}
    for p in record['selected']:
        model = p['model']; name = f"validate_{model}_{p['protocol']}_empiricalprofile_evidence"
        jobs.append(dict(name=name, gpu=MODELS.index(model), min_free_mb=23000,
            command=command(model, [p['protocol']], 'full_cohort', p['ks'], [p['scope']]),
            depends_on=[previous.get(model, f'round21_{model}_empirical_profile_discovery')]))
        previous[model] = name
    return write_addition('stage21_empirical_profile_validation.json', jobs, {str(selection):sha(selection)})
