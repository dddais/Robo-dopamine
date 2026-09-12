"""Complete-data gates for the fixed round29 two-forward resolution method."""
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
VARIANT = 'resolution_evidence_a1'
POLICY = OUT/'selection_resolution_evidence_discovery_v1.json'
QUEUE = OUT/'queue_gpu01_20260911_1700'


def check_policy(spec=None):
    spec = json.loads(POLICY.read_text()) if spec is None else spec
    expected = dict(variant=VARIANT,models=MODELS,protocols=PROTOCOLS,scopes=SCOPES,ks=[8,32,64],
        composition='high_steered+(high_steered-low_steered)',contrast_weight=1,low_max_pixels=50176,
        high_max_pixels=200704,batch_size=8,classes=[1,2,3,4,5])
    if any(spec.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen resolution policy differs')
    return spec


def folder(model, protocol, population, probe=False):
    return OUT/'experiments'/f'{model}_{protocol}_{VARIANT}'/population


def rank_path(model, protocol, scope):
    return OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'/f'ranking_{scope}.json'


def config_check(model, protocol, cfg, probe=False):
    expected = dict(model=model, contrast_negative_mode='resolution', contrast_weight=1,
        max_pixels=50176, resolution_high_max_pixels=200704, batch_size=8, method='bias', ranking_prefix='ANSWER: ',
        frozen_ranking_source=str(rank_path(model, protocol, SCOPES[0]).parent))
    if any(cfg.get(k) != v for k, v in expected.items()) or cfg.get('task_binding_fraction') is not None:
        raise ValueError('Actual resolution configuration differs from frozen budgets and same heads')


def same_high(a, b):
    return same_input(dict(a, token_audit=a['high_token_audit'], prompt=a['high_prompt']),
                      dict(b, token_audit=b['high_token_audit'], prompt=b['high_prompt']))


def verify_rows(rows, ids, heads=None, baseline=False, probe=False):
    from .resolution_evidence import validate_views
    from mydata_bench.attention_eval.masking import ImageSpan, bbox_to_token_positions
    if set(rows) != set(ids) or any(r['status'] != 'ok' for r in rows.values()):
        raise ValueError('Require complete actual resolution observations')
    def vec(value):
        a = np.asarray(value, dtype=np.float32)
        if a.shape != (5,) or not np.isfinite(a).all(): raise ValueError('Invalid five-class vector')
        return a
    for r in rows.values():
        if (r.get('derived_only') or r['readout'] != 'five_way_answer_likelihood_resolution'
                or r['contrast_negative_mode'] != 'resolution' or len(set(r['candidate_token_ids'])) != 5
                or r['actual_forward_branches'] != 2 or r['low_max_pixels'] != 50176 or r['high_max_pixels'] != 200704):
            raise ValueError('Changed actual two-view budget/readout/cost')
        validate_views([r['token_audit']], [r['high_token_audit']])
        for field, pixels in [('token_audit',50176), ('high_token_audit',200704)]:
            mapping = r[field]
            if mapping['resolution_max_pixels'] != pixels: raise ValueError('Recorded actual view budget differs')
            for scope, records in mapping['alignment'].items():
                target = set(); domain = set()
                for a in records:
                    span = ImageSpan(a['span'], '', a['start'], a['end'], tuple(a['grid_thw']))
                    if a['mosaic']: raise ValueError('Resolution variant is full-frame only')
                    for box in a['boxes']: target.update(bbox_to_token_positions(span,box,(640,480)))
                    domain.update(range(a['start'],a['end']))
                if (set(mapping['target'][scope]) != target or set(mapping['negative'][scope]) != domain-target
                        or not target or not domain <= set(mapping['visual'])):
                    raise ValueError('Independent actual ROI-to-grid reconstruction differs')
        low,high = vec(r['native_class_logits_low']),vec(r['native_class_logits_high'])
        if baseline:
            if r['condition'] != 'baseline' or r['contrast_weight'] != 0 or r['attention_diagnostics'] or r['low_attention_diagnostics']:
                raise ValueError('Both baseline views must be actual unsteered forwards')
            z=low; positive=low
        else:
            if r['contrast_weight'] != 1: raise ValueError('Changed common resolution weight')
            z=high+(high-low); positive=high
            observed=[]
            for field in ['attention_diagnostics','low_attention_diagnostics']:
                d=r[field]; actual={(int(l),h) for l,v in d.items() for h in v['heads']}
                if not actual or (heads is not None and actual!=heads):raise ValueError('Actual view heads differ from fixed ranking')
                for v in d.values():
                    if (v.get('bias')!=6 or not v.get('causal_mask_preserved') or not v.get('all_query_rows')
                            or v.get('generated_text_key_bias')!=0 or v['prefill_calls']!=1):
                        raise ValueError('Each view must use the registered original bias6')
                observed.append(actual)
            if observed[0]!=observed[1]:raise ValueError('Different head sets between resolutions')
        if not np.array_equal(z,vec(r['native_class_logits_combined'])) or not np.array_equal(positive,vec(r['native_class_logits_positive'])):
            raise ValueError('Actual resolution arithmetic does not reconstruct')
        p=np.exp(z.astype(float)-float(z.max()));p/=p.sum()
        if (np.max(np.abs(p-vec(r['native_class_probabilities'])))>=1e-5 or r['reward']!=int(p.argmax())+1
                or not np.isfinite(r['progress']) or r['progress']!=(r['reward']-1)/4
                or r['high_resolution_progress']!=int(high.argmax())/4):
            raise ValueError('Native primary/secondary readouts differ from stored actual logits')
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
        '--variant', VARIANT, '--methods', 'bias', '--strengths', '6', '--batch-size', '8',
        '--ranking-prefix', 'ANSWER: ', '--contrast-weight', '1', '--contrast-negative-mode', 'resolution',
        '--resolution-high-max-pixels', '200704', '--ks', *map(str, ks), '--scopes', *scopes, '--population', population,
        '--frozen-ranking-root', str(rank_path(model, protocols[0], SCOPES[0]).parents[1])]
    if probe: result += ['--limit', '8']
    return result


def write_addition(name, jobs, sources):
    existing = include_additions(json.loads((QUEUE/'plan.json').read_text()), QUEUE)
    validate_plan(dict(jobs=list(existing.values())+jobs))
    path = QUEUE/'additions'/name
    create_json(path, dict(jobs=jobs, sources_sha256=sources,
        interpretation='Round29 fixed two-forward original-bias resolution contrast; physical GPU0/1 only'))
    with path.with_suffix('.ready').open('x') as f: f.write('Registered complete prior gates; only GPU0/1\n')
    return path


def register_smoke(cpu_audit):
    check_policy()
    a=json.loads(cpu_audit.read_text())
    if a['status']!='pass' or a['labels_read'] or a['resolution_tests']!=10:
        raise ValueError('Require resolution execution/geometry and prior method CPU regression gates')
    for path,digest in a['source_code_sha256'].items():
        p=Path(path).resolve()
        if not p.is_relative_to(Path(__file__).resolve().parent) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('Implementation changed after CPU audit')
    waits=[f'round28_{m}_task_content_block_discovery' for m in MODELS]
    jobs=[dict(name=f'smoke_{m}_resolution_evidence',gpu=i,min_free_mb=40000,depends_on=waits,
        command=command(m,['image_text','text_video'],'discovery',[8,32],probe=True)) for i,m in enumerate(MODELS)]
    return write_addition('stage29_resolution_evidence_smoke.json',jobs,{str(cpu_audit):sha(cpu_audit),str(POLICY):sha(POLICY)})


def smoke_ready():
    return all(complete(folder(m, p, 'discovery_smoke', True), [8, 32], SCOPES, 8)
        for m in MODELS for p in ['image_text', 'text_video'])


def verify_smoke(destination):
    if not smoke_ready():raise ValueError('Require all actual resolution smoke conditions')
    sources={str(POLICY):sha(POLICY)};checks=[]
    def read(p,rows=False):
        sources[str(p)]=sha(p);return latest(p) if rows else json.loads(p.read_text())
    for m in MODELS:
        for p in ['image_text','text_video']:
            root=folder(m,p,'discovery_smoke')
            reference=OUT/'experiments'/f'{m}_{p}_bias_task_anchor_evidence_a1'/'discovery'
            ids=read(root/'requested_ids.json')
            if ids!=read(reference/'requested_ids.json')[:8]:raise ValueError('Changed actual smoke batch')
            config_check(m,p,read(root/'runtime_config.json'))
            base=verify_rows(read(root/'predictions/baseline.jsonl',True),ids,baseline=True)
            old_base=read(reference/'predictions/baseline.jsonl',True)
            for e in ids:
                if not same_input(base[e],old_base[e]) or base[e]['native_class_logits_low']!=old_base[e]['native_class_logits_positive']:
                    raise ValueError('Low-resolution baseline does not exactly replay old matching batch')
            for scope in SCOPES:
                ranking=read(rank_path(m,p,scope))['ranking']
                if read(root.parent/'ranking'/f'ranking_{scope}.json')['ranking']!=ranking:raise ValueError('Copied ranking differs')
                for k in [8,32]:
                    rows=verify_rows(read(root/'bias_s6/predictions'/f'{scope}_target_{k}.jsonl',True),ids,
                        {(h['layer'],h['head']) for h in ranking[:k]})
                    old=read(reference/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl',True)
                    for e,r in rows.items():
                        if (not same_input(r,base[e]) or not same_high(r,base[e]) or not same_input(r,old[e])
                                or r['native_class_logits_low']!=old[e]['native_cell_logits']['bias']
                                or r['condition']!=f'{scope}:target:{k}'):
                            raise ValueError('Same-view input or actual low-bias replay differs')
                    checks.append(dict(model=m,protocol=p,scope=scope,k=k,n=8,
                        actual_low_baseline_and_bias_replay_exact=True,actual_high_geometry_fourfold=True,
                        same_nonvisual_content_and_full_frame_sampling=True,matching_high_baseline_input=True,
                        actual_same_heads_and_formula_verified=True))
    create_json(destination,dict(status='pass',labels_read=False,checks=checks,sources_sha256=sources,
        actual_intervention_rows_verified=128,actual_intervention_forward_observations=256,
        actual_baseline_rows=32,actual_baseline_forward_observations=64,
        interpretation='Actual two-resolution implementation and matching low-view source replay, no efficacy scoring'))


def register_discovery(audit):
    a = json.loads(audit.read_text())
    if a['status'] != 'pass' or a['labels_read'] or len(a['checks']) != 16: raise ValueError('Incomplete actual two-resolution smoke gate')
    verify_sources(a['sources_sha256'])
    jobs = [dict(name=f'round29_{m}_resolution_evidence_discovery', gpu=i, min_free_mb=40000,
        depends_on=[f'smoke_{m}_resolution_evidence'], command=command(m, PROTOCOLS, 'discovery', [8, 32, 64]))
        for i, m in enumerate(MODELS)]
    return write_addition('stage29_resolution_evidence_discovery.json', jobs, {str(audit): sha(audit), str(POLICY): sha(POLICY)})


def matrix_ready():
    return all(complete(folder(m, p, 'discovery'), [8, 32, 64], SCOPES, 70) for m in MODELS for p in PROTOCOLS)


def select(destination):
    if not matrix_ready(): raise ValueError('Require both entire five-input actual matrices')
    spec = check_policy(); splits = json.loads((OUT/'splits.json').read_text()); ids = splits['discovery']
    if (len(ids) != 70 or set(ids) & set(splits['validation']) or spec['variant'] != VARIANT
            or spec['protocols'] != PROTOCOLS or spec['composition'] != 'high_steered+(high_steered-low_steered)'
            or spec['contrast_weight'] != 1): raise ValueError('Frozen policy or split differs')
    sources = {str(POLICY): sha(POLICY)}; matrices = {}
    def read(path, rows=False):
        sources[str(path)] = sha(path); return latest(path) if rows else json.loads(path.read_text())
    for m in MODELS:
        for p in PROTOCOLS:
            root = folder(m, p, 'discovery'); config_check(m, p, read(root/'runtime_config.json'), False)
            if set(read(root/'requested_ids.json')) != set(ids): raise ValueError('Changed discovery population')
            base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, baseline=True)
            conditions = {}
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
                    if any(not same_high(r,base[e]) for e,r in rows.items()): raise ValueError('High-view intervention input differs from high-view baseline')
                    conditions[(s, k)] = rows
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
        m = p['model']; name = f"validate_{m}_{p['protocol']}_resolution_evidence"
        jobs.append(dict(name=name, gpu=MODELS.index(m), min_free_mb=40000,
            depends_on=[previous.get(m, f'round29_{m}_resolution_evidence_discovery')],
            command=command(m, [p['protocol']], 'full_cohort', p['ks'], [p['scope']])))
        previous[m] = name
    return write_addition('stage29_resolution_evidence_validation.json', jobs, {str(selection): sha(selection), str(audit): sha(audit)})
