"""CPU verification gates for round33; GPU workers must never import this module."""
import json
from pathlib import Path

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .head_delta_worker import BASE,POLICY,VARIANT,policy,fixed_gates,sha,verify_sources,ranking
from .head_delta_reft import initial_matrices
from .head_gate_tasks import addition,QUEUE
from .compare_native_bias import same_input,heads
from .prepare import OUT


def smoke_ready():
    return all((BASE/'smoke'/m/'complete.json').exists() for m in ['qwen','roboreward'])


def native(rows,ids,gate_values,expected_heads=None,zero=False):
    if set(rows)!=set(ids) or any(r['status']!='ok' for r in rows.values()):raise ValueError('Complete native sample coverage required')
    for row in rows.values():
        if (row['readout']!='five_way_answer_likelihood_attention_increment_reft' or row['actual_forward_branches']!=1
            or row['contrast_weight']!=0 or row['native_class_logits_negative'] is not None or row['negative_attention_diagnostics']
            or row['adapter_rank']!=4 or row['adapter_parameter_count']!=28672 or row['fixed_gate_parameter_count']!=1792
            or not row['no_constant_adapter_term'] or not row['backbone_frozen']):raise ValueError('Changed native rank4 branch contract')
        z=np.asarray(row['native_class_logits_positive']);p=np.asarray(row['native_class_probabilities'])
        if z.shape!=(5,) or p.shape!=(5,) or not np.isfinite([*z,*p,row['progress']]).all():raise ValueError('Invalid native class vector')
        q=np.exp(z-z.max());q/=q.sum()
        if np.max(np.abs(p-q))>1e-5 or row['reward']!=int(p.argmax())+1 or row['progress']!=(row['reward']-1)/4:
            raise ValueError('Five-class native readout does not reconstruct')
        if expected_heads is None:
            if row['condition']!='baseline' or row['attention_diagnostics']:raise ValueError('Unsteered baseline required')
            continue
        if heads(row,'attention_diagnostics')!=expected_heads:raise ValueError('Actual selected heads differ')
        for layer,d in row['attention_diagnostics'].items():
            expected=np.zeros((len(d['heads']),2)) if zero else gate_values[int(layer)-8,d['heads']]
            if (d['method']!='learned_head_delta_reft' or not d['all_query_rows'] or not d['causal_mask_preserved']
                or not d['layer_shared_across_heads'] or not d['no_constant_term'] or d['adapter_rank']!=4
                or not d['separate_native_dtype_additions'] or d['zero_gate_probe']!=zero or d['prefill_calls']!=1
                or d['visual_cap']!=6 or d['task_cap']!=4 or np.max(np.abs(np.asarray(d['gates'])-expected))>1e-7):
                raise ValueError('Actual increment/gate/mask contract differs')
    return rows


def verify_smoke():
    policy()
    if not smoke_ready():raise ValueError('Both actual smokes required before audit')
    sources={str(POLICY):sha(POLICY)};checks=[]
    def read(path,rows=False):
        sources[str(path)]=sha(path);return latest(path) if rows else json.loads(path.read_text())
    expected_A,expected_B=initial_matrices()
    for model in ['qwen','roboreward']:
        root=BASE/'smoke'/model;c=read(root/'complete.json');verify_sources(c['sources_sha256'])
        if c['status']!='pass' or c['labels_read'] or c['optimizer_updates']!=0 or not c['parameters_restored'] or len(c['checks'])!=8:
            raise ValueError('Actual no-label/no-optimizer smoke contract differs')
        matrices=read(root/'initial_matrices.json');A=np.asarray(matrices['A'],dtype=np.float32);B=np.asarray(matrices['B'],dtype=np.float32)
        probe_B=np.asarray(matrices['nonzero_probe_B'],dtype=np.float32)
        if not np.array_equal(A,expected_A.numpy()) or not np.array_equal(B,expected_B.numpy()) or probe_B.shape!=(28,128,4):
            raise ValueError('Initial matrices changed')
        gate_file,gate=fixed_gates(model);sources[str(gate_file)]=sha(gate_file);gate_values=np.asarray(gate['gate_values'])
        for protocol in ['image_text','text_video']:
            dest=root/protocol;ids=read(dest/'requested_ids.json');base=native(read(dest/'baseline.jsonl',True),ids,gate_values)
            if len(ids)!=8:raise ValueError('Actual smoke must cover original first8 examples')
            reference=OUT/'experiments'/f'{model}_{protocol}_learned_head_gates_v1'/'discovery'
            old_base=read(reference/'predictions/baseline.jsonl',True)
            if any(not same_input(row,old_base[e]) or row['native_class_logits_positive']!=old_base[e]['native_class_logits_positive'] for e,row in base.items()):
                raise ValueError('Baseline reference does not replay')
            ranks=ranking(model,protocol)
            for scope in ['all_frames','last_frame']:
                for k in [8,32]:
                    prefix=f'{scope}_k{k}';selected={(h['layer'],h['head']) for h in ranks[scope]['ranking'][:k]}
                    initial=native(read(dest/f'{prefix}_B0.jsonl',True),ids,gate_values,selected)
                    zero=native(read(dest/f'{prefix}_zero_gate.jsonl',True),ids,gate_values,selected,zero=True)
                    old=read(reference/'learned_head_gate_bias_s6/predictions'/f'{scope}_target_{k}.jsonl',True)
                    for e in ids:
                        if (not same_input(initial[e],old[e]) or initial[e]['native_class_logits_positive']!=old[e]['native_class_logits_positive']
                            or not same_input(zero[e],base[e]) or zero[e]['native_class_logits_positive']!=base[e]['native_class_logits_positive']):
                            raise ValueError('Actual initial/zero logits do not exactly replay')
                        if any(d['B_l2']!=0 for d in initial[e]['attention_diagnostics'].values()):raise ValueError('Initial B was not zero')
                    path=dest/f'{prefix}_gradients.npz';sources[str(path)]=sha(path)
                    active=sorted({l-8 for l,h in selected});inactive=np.ones(28,dtype=bool);inactive[active]=False
                    with np.load(path,allow_pickle=False) as gradients:
                        for regime in ['B0','nonzero_B']:
                            for name,shape in [('A',(8,28,4,128)),('B',(8,28,128,4))]:
                                arr=gradients[f'{regime}_{name}']
                                if arr.shape!=shape or not np.isfinite(arr).all():raise ValueError('Actual gradient tensor shape/finite check failed')
                                norm=np.abs(arr).reshape(8,28,-1).sum(-1)
                                if np.any(norm[:,inactive]!=0):raise ValueError('Inactive-layer gradient is not zero')
                                if regime=='B0' and name=='A':
                                    if np.any(norm!=0):raise ValueError('A gradient must vanish at B0')
                                elif np.any(norm[:,active]<=0):raise ValueError('Active-layer gradient must be nonzero')
                    observations=read(dest/f'{prefix}_gradient_observations.json')
                    if len(observations)!=16 or [o['example_id'] for o in observations]!=ids+ids:
                        raise ValueError('Wrong actual gradient sample membership')
                    for observation in observations:
                        bm=B if observation['regime']=='B0' else probe_B
                        for layer,d in observation['attention_diagnostics'].items():
                            i=int(layer)-8;probe=d['increment_probe'];raw=np.asarray(probe['raw_delta'],dtype=np.float32)
                            expected=(raw@A[i].T)@bm[i].T
                            if not np.allclose(expected,probe['correction'],atol=1e-5,rtol=1e-4):raise ValueError('Captured B A d does not reconstruct')
                    checks.append(dict(model=model,protocol=protocol,scope=scope,k=k,native_rows=8,gradient_observations=16,
                        actual_B0_and_zero_replays=True,actual_gradient_tensors_verified=True))
    verify_sources(sources);path=BASE/'actual_gradient_gate_v1.json'
    create_json(path,dict(status='pass',labels_read=False,models=['qwen','roboreward'],checks=checks,sources_sha256=sources,
        normal_rows=128,zero_rows=128,gradient_observations=256,optimizer_updates=0,
        interpretation='Both real models passed initial/zero native replay and full matrix gradient tensor verification. No efficacy selection.'))
    return path


def register_training(audit):
    record=json.loads(audit.read_text());verify_sources(record['sources_sha256'])
    if record['status']!='pass' or record['labels_read'] or len(record['checks'])!=16:raise ValueError('Complete actual smoke gate required')
    jobs=[dict(name=f'round33_{m}_head_delta_train',gpu=i,min_free_mb=60000,depends_on=[f'round33_{m}_head_delta_smoke'],
        command=['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m','mydata_bench.auto_research_addbase.head_delta_worker',
                 '--model',m,'--phase','train']) for i,m in enumerate(['qwen','roboreward'])]
    return addition('stage33_head_delta_training.json',jobs,{str(POLICY):sha(POLICY),str(audit):sha(audit)})
