"""Audit fixed-step training and evaluate one frozen gate family without labels in inference."""
import json
import time
from pathlib import Path

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .compare_native_bias import same_input, heads
from .head_delta_worker import BASE, POLICY, PROTOCOLS, SCOPES, policy, schedule, population
from .head_gate_tasks import MODELS, QUEUE
from .durable_scheduler import include_additions, validate_plan


VARIANT = 'learned_head_delta_reft_v1'
NEIGHBORHOODS = {8: [8, 16, 24], 32: [24, 32, 40], 64: [48, 64, 80]}


def addition(name, jobs, sources):
    current = include_additions(json.loads((QUEUE/'plan.json').read_text()), QUEUE)
    validate_plan(dict(jobs=list(current.values())+jobs))
    dest = QUEUE/'additions'/name
    create_json(dest, dict(jobs=jobs, sources_sha256=sources,
        interpretation='Round33 fixed-gate rank4 attention-increment adaptation. Physical GPU0/1 only.'))
    with dest.with_suffix('.ready').open('x') as handle: handle.write('Round33 prior audit gates satisfied; GPU0/1\n')
    return dest


def training_ready():
    return all((BASE/'training'/m/'final_adapter.json').exists() for m in MODELS)


def verify_training():
    from .head_delta_reft import initial_matrices
    from .head_delta_worker import fixed_gates
    policy()
    if not training_ready(): raise ValueError('Wait for both fixed final matrices')
    ids = [s['example_id'] for s in population()]; sources = {str(POLICY): sha(POLICY)}; checks = []; code_sources = {}
    def read(path):
        sources[str(path)] = sha(path); return json.loads(path.read_text())
    smoke = read(BASE/'actual_gradient_gate_v1.json'); verify_sources(smoke['sources_sha256'])
    expected_A, expected_B = initial_matrices()
    for model in MODELS:
        root = BASE/'training'/model; record = read(root/'final_adapter.json'); manifest = read(root/'training_manifest.json')
        starts = []
        for path in (BASE/'worker_starts').glob('*.json'):
            run = json.loads(path.read_text())
            if run['arguments'] == dict(model=model, phase='train'): starts.append((path, run))
        if len(starts) != 1: raise ValueError('Require exactly one registered fixed training trajectory per model')
        start_path, run = starts[0]; read(start_path)
        if run['cuda_visible_devices'] not in ['0', '1']:
            raise ValueError('Training must use only one permitted physical GPU')
        snap_path = Path(run['source_snapshot']); snap = read(snap_path)
        if sha(snap_path) != run['source_snapshot_sha256'] or snap['cuda_visible_devices'] != run['cuda_visible_devices']:
            raise ValueError('Training startup provenance changed')
        for name in ['head_delta_worker.py', 'head_delta_reft.py', 'head_attention_gates.py', 'head_gate_worker.py']:
            source = Path(__file__).parent/name
            key = f'mydata_bench/auto_research_addbase/{name}'
            if snap['sources'][key]['sha256'] != sha(source): raise ValueError('Executed training/operator source changed')
            code_sources[str(source)] = sha(source)
        original_probes = read(BASE/'smoke'/model/'complete.json')['backbone_probe_hashes']
        if record['backbone_probe_hashes'] != original_probes or len(original_probes) != 5:
            raise ValueError('Frozen backbone probe fingerprints differ from actual smoke')
        gate_file, gate = fixed_gates(model); sources[str(gate_file)] = sha(gate_file)
        expected = dict(status='complete', model=model, variant=VARIANT, rank=4, training_ids=ids,
            validation_labels_used=[], forward_backward_steps=4200, optimizer_steps=525, adapter_parameter_count=28672,
            backbone_requires_grad_count=0, gate_trainable_parameter_count=0, policy_sha256=sha(POLICY),
            selection_rule='Final fixed step only', fixed_gate_file=str(gate_file), fixed_gate_sha256=sha(gate_file))
        if any(record.get(k) != v for k,v in expected.items()): raise ValueError('Changed fixed final training contract')
        expected_manifest = dict(training_ids=ids, labels_used=ids, adapter_parameter_count=28672,
            backbone_trainable_parameter_count=0, gate_trainable_parameter_count=0, prior_gate_forward_backward_steps=4200,
            fixed_gate_file=str(gate_file), fixed_gate_sha256=sha(gate_file), independent_examples=70, independent_video_groups=28,
            labels_source_sha256=sha(OLD/'labels_for_scoring_only.json'), class_counts={'0':45,'4':25}, policy_sha256=sha(POLICY))
        if any(manifest.get(k) != v for k,v in expected_manifest.items()): raise ValueError('Changed actual training manifest')
        A=np.asarray(record['A'],dtype=np.float32); B=np.asarray(record['B'],dtype=np.float32)
        if A.shape!=(28,4,128) or B.shape!=(28,128,4) or not np.isfinite([*A.flat,*B.flat]).all() or not np.any(B):
            raise ValueError('Invalid trained rank4 matrices')
        initial=read(root/'initial_matrices.json')
        if not np.array_equal(initial['A'],expected_A.numpy()) or not np.array_equal(initial['B'],expected_B.numpy()):
            raise ValueError('Changed seeded initial matrices')
        if np.array_equal(A,expected_A.numpy()): raise ValueError('A never changed during training')
        events_path=root/'optimizer_events.jsonl';sources[str(events_path)]=sha(events_path)
        events=[json.loads(line) for line in events_path.read_text().splitlines()]
        if len(events)!=525:raise ValueError('Incomplete fixed optimizer trajectory')
        for step,e in enumerate(events,1):
            if (e['event']!='optimizer_step' or e['optimizer_steps']!=step or e['forward_backward_steps']!=8*step
                or e['epoch']!=(step-1)//175+1 or not np.isfinite([e[n] for n in ['mean_loss','gradient_norm_before_clip','A_l2','B_l2','operator_penalty']]).all()
                or min(e['A_l2'],e['B_l2'])<=0 or e['operator_penalty']<0):raise ValueError('Invalid actual training event')
        final_penalty=.01*np.square(B@A).sum(axis=(-2,-1)).mean()/128
        if not np.isclose(events[-1]['operator_penalty'],final_penalty,rtol=1e-5,atol=1e-8):
            raise ValueError('Final operator penalty does not reconstruct')
        for epoch in range(3):
            if read(root/f'schedule_epoch_{epoch+1}.json') != [list(x) for x in schedule(70,epoch)]:raise ValueError('Changed training schedule')
            checkpoint=read(root/f'epoch_{epoch+1}_matrices_for_audit_only.json')
            if checkpoint['inference_eligible'] or checkpoint['forward_backward_steps']!=1400*(epoch+1) or checkpoint['optimizer_steps']!=175*(epoch+1):
                raise ValueError('Invalid intermediate checkpoint contract')
            if epoch==2 and (checkpoint['A']!=record['A'] or checkpoint['B']!=record['B']):raise ValueError('Final checkpoint differs from last fixed step')
        union=set()
        for protocol in PROTOCOLS:
            for scope in SCOPES:
                original=OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'/f'ranking_{scope}.json'
                ranks=read(original)
                if ranks!=read(root/'rankings'/protocol/f'ranking_{scope}.json'):raise ValueError('Changed training head ranks')
                union.update(h['layer'] for h in ranks['ranking'][:64])
        inactive=[l-8 for l in range(8,36) if l not in union]
        if inactive and (not np.array_equal(A[inactive],expected_A.numpy()[inactive]) or np.any(B[inactive])):
            raise ValueError('Unexposed layers changed despite zero operator and no data gradient')
        checks.append(dict(model=model,training_examples=70,actual_forward_backward_steps=4200,optimizer_updates=525,
            learned_parameters=28672,fixed_gate_parameters=1792,prior_forward_backward_steps=4200,
            A_l2=float(np.linalg.norm(A)),B_l2=float(np.linalg.norm(B)),operator_penalty=float(final_penalty),
            active_layers=sorted(union),final_checkpoint_only=True,schedule_and_source_heads_verified=True))
    verify_sources(sources)
    for name, digest in code_sources.items():
        source = Path(name).resolve()
        if source.parent != Path(__file__).resolve().parent or sha(source) != digest:
            raise ValueError('Executed training code changed while auditing')
    dest=BASE/'fixed_training_audit_v1.json'
    create_json(dest,dict(status='pass',labels_read=False,checks=checks,sources_sha256=sources,source_code_sha256=code_sources,
        interpretation='Full fixed training and final matrix audit, not an efficacy or generalization claim.'))
    return dest


def command(model, protocols, population_name, ks, scopes=SCOPES):
    return ['/home/dais/miniconda3/envs/robo-dopamine/bin/python', '-B', '-m',
        'mydata_bench.auto_research_addbase.head_delta_inference', '--model', model, '--protocols', *protocols,
        '--ks', *map(str, ks), '--scopes', *scopes, '--population', population_name]


def register_discovery(audit):
    a = json.loads(audit.read_text())
    if a['status'] != 'pass' or a['labels_read'] or len(a['checks']) != 2: raise ValueError('Both final training audits required')
    verify_sources(a['sources_sha256'])
    jobs = [dict(name=f'round33_{m}_head_delta_discovery', gpu=i, min_free_mb=23000,
        depends_on=[f'round33_{m}_head_delta_train'], command=command(m, PROTOCOLS, 'discovery', [8, 32, 64]))
        for i, m in enumerate(MODELS)]
    return addition('stage33_head_delta_discovery.json', jobs, {str(audit): sha(audit), str(POLICY): sha(POLICY)})


def folder(model, protocol, population_name):
    return OUT/'experiments'/f'{model}_{protocol}_{VARIANT}'/population_name


def matrix_ready():
    for model in MODELS:
        for protocol in PROTOCOLS:
            root = folder(model, protocol, 'discovery'); request = root/'requested_ids.json'
            if not request.exists(): return False
            ids = set(json.loads(request.read_text()))
            if len(ids) != 70: return False
            files = [root/'predictions/baseline.jsonl']+[root/'learned_head_delta_reft_s6/predictions'/f'{s}_target_{k}.jsonl'
                     for s in SCOPES for k in [8, 32, 64]]
            if not all(path.exists() and set(latest(path)) == ids for path in files): return False
    return True


def verify_rows(rows, ids, adapter_record, adapter_file, expected_heads=None, baseline=False, mode='learned'):
    from .head_delta_tasks import native
    from .head_delta_worker import fixed_gates
    gate_file, gates = fixed_gates(adapter_record['model'])
    native(rows,ids,np.asarray(gates['gate_values']),None if baseline else expected_heads)
    A=np.asarray(adapter_record['A'],dtype=np.float32);B=np.asarray(adapter_record['B'],dtype=np.float32)
    if mode=='B0':B=np.zeros_like(B)
    for row in rows.values():
        expected=dict(learned_gate_file=str(gate_file),learned_gate_sha256=sha(gate_file),
            learned_adapter_file=str(adapter_file),learned_adapter_sha256=sha(adapter_file),adapter_mode=mode,
            contrast_negative_mode='learned_head_gates',gate_parameter_count=1792)
        if any(row.get(k)!=v for k,v in expected.items()) or len(set(row['candidate_token_ids']))!=5:
            raise ValueError('Actual final adapter/gate/native token identity changed')
        for layer,d in row['attention_diagnostics'].items():
            i=int(layer)-8
            if not np.isclose(d['A_l2'],np.linalg.norm(A[i]),rtol=1e-5,atol=1e-7) or not np.isclose(d['B_l2'],np.linalg.norm(B[i]),rtol=1e-5,atol=1e-7):
                raise ValueError('Actual layer matrices do not match final artifact norms')
    return rows


def select(destination):
    policy()
    if not matrix_ready(): raise ValueError('Wait for all ten actual learned-gate matrices')
    ids = [s['example_id'] for s in population()]; sources = {str(POLICY): sha(POLICY)}; matrices = {}
    def read(path, rows=False):
        sources[str(path)] = sha(path); return latest(path) if rows else json.loads(path.read_text())
    training = read(BASE/'fixed_training_audit_v1.json'); verify_sources(training['sources_sha256'])
    for model in MODELS:
        gate_file = BASE/'training'/model/'final_adapter.json'; gate = read(gate_file)
        for protocol in PROTOCOLS:
            root = folder(model, protocol, 'discovery')
            if read(root/'requested_ids.json') != ids: raise ValueError('Changed inference discovery population')
            cfg = read(root/'runtime_config.json')
            from .head_delta_inference import inference_config
            if cfg != inference_config(model, protocol):
                raise ValueError('Actual evaluation configuration differs')
            base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, gate, gate_file, baseline=True)
            reference = read(OUT/'experiments'/f'{model}_{protocol}_learned_head_gates_v1'/'discovery/predictions/baseline.jsonl', True)
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
                    rows = verify_rows(read(root/'learned_head_delta_reft_s6/predictions'/f'{scope}_target_{k}.jsonl', True), ids, gate, gate_file, chosen)
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
                method='learned_head_delta_reft_s6', mae=score['mae'], baseline_mae=base['mae'], deltas=delta, passes=gate,
                accuracy_all=score['accuracy']['0.125/0.875']['all']['rate_all_expected'])
            points.append(point)
            if gate: passing.append(point)
        if passing:
            best = min(passing, key=lambda p: (p['mae'], -p['accuracy_all'], p['center_k'], p['scope']))
            centers=sorted(p['center_k'] for p in passing if p['scope']==best['scope'])
            proposed.append(dict(best, discovery_passing_centers=centers, ks=sorted({k for c in centers for k in NEIGHBORHOODS[c]})))
    counts = {m: sum(p['model'] == m for p in proposed) for m in MODELS}; eligible = min(counts.values()) >= 3
    create_json(destination, dict(created_at=time.time(), variant=VARIANT, all_discovery_points=points,
        proposed_input_candidates=proposed, input_counts=counts, family_eligible_for_full=eligible,
        selected=proposed if eligible else [], sources_sha256=sources, validation_labels_used=[],
        actual_intervention_rows_verified=4200, distinct_conditions=60,
        interpretation='Discovery was used for prior1792-gate and new28672-matrix supervised training and then selection. Resubstitution is not independent evidence; any full validation remains adaptive dataset-internal exploration.'))


def register_full(selection):
    record = json.loads(selection.read_text()); verify_sources(record['sources_sha256'])
    eligible = min(record['input_counts'].values()) >= 3
    if record['selected'] != (record['proposed_input_candidates'] if eligible else []): raise ValueError('Changed shared coverage rule')
    jobs = []; previous = {}
    for p in record['selected']:
        model = p['model']; name = f"validate_{model}_{p['protocol']}_head_delta"
        jobs.append(dict(name=name, gpu=MODELS.index(model), min_free_mb=23000,
            depends_on=[previous.get(model, f'round33_{model}_head_delta_discovery')],
            command=command(model, [p['protocol']], 'full_cohort', p['ks'], [p['scope']])))
        previous[model] = name
    return addition('stage33_head_delta_validation.json', jobs, {str(selection): sha(selection), str(POLICY): sha(POLICY)})
