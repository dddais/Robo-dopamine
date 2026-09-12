"""Round33 label-free real smoke and fixed discovery-only matrix training."""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import append, latest
from .prepare import OUT
from .head_gate_worker import (BASE as GATE_BASE, PROTOCOLS, SCOPES, config as gate_config,
    ranking, population, schedule, weight_probes, sha, verify_sources)
from .head_delta_reft import HeadDeltaRuntime, initial_matrices, operator_penalty, increment_correction
from .compare_native_bias import same_input
from .provenance import snapshot


POLICY = OUT/'selection_learned_head_delta_reft_v1.json'
BASE = OUT/'learned_head_delta_reft_v1'
VARIANT = 'learned_head_delta_reft_v1'


def policy():
    spec=json.loads(POLICY.read_text());verify_sources(spec['sources_sha256'])
    t=spec['training'];p=spec['parameterization']
    if (spec['variant']!=VARIANT or spec['models']!=['qwen','roboreward'] or spec['protocols']!=PROTOCOLS
        or spec['scopes']!=SCOPES or p['rank']!=4 or p['head_dim']!=128 or p['trainable_parameters']!=28672
        or t['epochs']!=3 or t['learning_rate']!=.001 or t['training_ks']!=[32,64]
        or t['total_forward_backward_steps']!=4200 or t['total_optimizer_steps']!=525):
        raise ValueError('Frozen rank4 architecture or fixed training protocol changed')
    return spec


def fixed_gates(model):
    path=GATE_BASE/'training'/model/'final_gates.json';r=json.loads(path.read_text())
    if r['trained_parameter_count']!=1792 or r['validation_labels_used'] or r['optimizer_steps']!=525:
        raise ValueError('Require exactly the fixed final round32 gates')
    return path,r


def config(model,protocol):
    cfg=gate_config(model,protocol);path,_=fixed_gates(model)
    cfg.update(method='learned_head_delta_reft',adapter_rank=4,adapter_mode='learned',
        learned_gate_file=str(path),learned_gate_sha256=sha(path))
    return cfg


def setup(model,trainable=True):
    runtime=HeadDeltaRuntime(config(model,'image_text'));path,g=fixed_gates(model)
    runtime.controller.fixed_values=g['gate_values']
    A,B=initial_matrices(runtime.model.device)
    runtime.controller.A=torch.nn.Parameter(A) if trainable else A
    runtime.controller.B=torch.nn.Parameter(B) if trainable else B
    if any(p.requires_grad or p.grad is not None for p in runtime.model.parameters()):
        raise ValueError('Only external adapter matrices may carry gradients')
    return runtime


def probe_B(device):
    generator=torch.Generator(device='cpu').manual_seed(20260913)
    return (.001*torch.randn((28,128,4),generator=generator)).to(device)


def matrix_gradient_check(runtime,active,zero_B):
    result={};inactive=np.ones(28,dtype=bool);inactive[list(active)]=False
    for name,parameter in [('A',runtime.controller.A),('B',runtime.controller.B)]:
        gradient=parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():raise ValueError('Missing/nonfinite actual matrix gradient')
        array=gradient.detach().float().cpu().numpy().copy()
        norms=np.abs(array).reshape(28,-1).sum(1)
        if np.any(norms[inactive]!=0):raise ValueError('Inactive layer received a data gradient')
        if name=='A' and zero_B:
            if np.any(norms!=0):raise ValueError('At exact B0 all A gradients must vanish')
        elif np.any(norms[list(active)]<=0):raise ValueError('Every active layer must have nonzero actual matrix gradient')
        result[name]=array
    if runtime.controller.theta is not None or any(p.grad is not None or p.requires_grad for p in runtime.model.parameters()):
        raise ValueError('Fixed gates or backbone entered the gradient path as parameters')
    return result


def smoke(model):
    policy();root=BASE/'smoke'/model
    if root.exists():raise ValueError('Immutable smoke already has an output directory; inspect before any explicit recovery')
    root.mkdir(parents=True)
    runtime=setup(model);A=runtime.controller.A;B=runtime.controller.B
    initial_A=A.detach().clone();initial_B=B.detach().clone();nonzero=probe_B(B.device)
    runtime.controller.capture_probes=True
    create_json(root/'initial_matrices.json',dict(A=A.detach().cpu().tolist(),B=B.detach().cpu().tolist(),nonzero_probe_B=nonzero.cpu().tolist()))
    before=weight_probes(runtime.model);samples=population()[:8];checks=[];sources={str(POLICY):sha(POLICY)}
    for protocol in ['image_text','text_video']:
        cfg=config(model,protocol);runtime.cfg.clear();runtime.cfg.update(cfg);ranks=ranking(model,protocol)
        dest=root/protocol;create_json(dest/'runtime_config.json',cfg);create_json(dest/'requested_ids.json',[s['example_id'] for s in samples])
        baseline=runtime.predict(samples,'baseline');append(dest/'baseline.jsonl',baseline)
        reference_root=OUT/'experiments'/f'{model}_{protocol}_learned_head_gates_v1'/'discovery'
        baseline_path=reference_root/'predictions/baseline.jsonl';old_base=latest(baseline_path);sources[str(baseline_path)]=sha(baseline_path)
        for row in baseline:
            old=old_base[row['example_id']]
            if not same_input(row,old) or row['native_class_logits_positive']!=old['native_class_logits_positive']:
                raise ValueError('Actual same-batch baseline does not replay round32')
        for scope in SCOPES:
            rank_file=OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'/f'ranking_{scope}.json'
            sources[str(rank_file)]=sha(rank_file)
            for k in [8,32]:
                condition=f'{scope}:target:{k}';prefix=f'{scope}_k{k}'
                with torch.no_grad():B.copy_(initial_B)
                initial=runtime.predict(samples,condition,ranks);append(dest/f'{prefix}_B0.jsonl',initial)
                ref_path=reference_root/'learned_head_gate_bias_s6/predictions'/f'{scope}_target_{k}.jsonl'
                ref=latest(ref_path);sources[str(ref_path)]=sha(ref_path)
                if any(not same_input(row,ref[row['example_id']]) or row['native_class_logits_positive']!=ref[row['example_id']]['native_class_logits_positive'] for row in initial):
                    raise ValueError('B0 initial adapter must exactly replay actual final round32 gate logits')
                with torch.no_grad():B.copy_(nonzero)
                runtime.controller.zero_probe=True
                try:zero=runtime.predict(samples,condition,ranks)
                finally:runtime.controller.zero_probe=False
                append(dest/f'{prefix}_zero_gate.jsonl',zero)
                if any(not same_input(row,base) or row['native_class_logits_positive']!=base['native_class_logits_positive'] for row,base in zip(zero,baseline)):
                    raise ValueError('Zero attention increment with nonzero adapter must exactly replay original baseline')
                active=sorted({h['layer']-8 for h in ranks[scope]['ranking'][:k]});arrays={};observations=[]
                for regime,value in [('B0',initial_B),('nonzero_B',nonzero)]:
                    with torch.no_grad():B.copy_(value)
                    grads={'A':[],'B':[]}
                    for sample in samples:
                        A.grad=None;B.grad=None
                        logits,diagnostics,maps=runtime.differentiable_logits([sample],condition,ranks)
                        loss=(logits-logits.mean(-1,keepdim=True)).square().mean();loss.backward()
                        g=matrix_gradient_check(runtime,active,regime=='B0')
                        for name in grads:grads[name].append(g[name])
                        for layer,diag in diagnostics.items():
                            probe=diag['increment_probe'];index=int(layer)-8
                            raw=torch.tensor(probe['raw_delta'],device=A.device)
                            expected=increment_correction(raw,A[index].detach(),B[index].detach()).cpu().numpy()
                            if not np.allclose(expected,probe['correction'],atol=1e-5,rtol=1e-4):
                                raise ValueError('Actual captured attention correction does not reconstruct B A d')
                        observations.append(dict(example_id=sample['example_id'],regime=regime,batch_size=1,
                            symmetric_probe_loss=float(loss.detach()),active_layers=[x+8 for x in active],
                            attention_diagnostics=diagnostics,token_audit=maps[0]))
                        del logits,loss
                    for name in grads:arrays[f'{regime}_{name}']=np.stack(grads[name])
                with (dest/f'{prefix}_gradients.npz').open('xb') as f:np.savez_compressed(f,**arrays)
                create_json(dest/f'{prefix}_gradient_observations.json',observations)
                checks.append(dict(protocol=protocol,scope=scope,k=k,samples=8,B0_exact_round32=True,
                    zero_gate_exact_baseline=True,gradient_observations=16,active_layer_gradients_verified=True,
                    actual_correction_probes_reconstruct=True))
                print(model,protocol,condition,'rank4 smoke passed',flush=True)
    with torch.no_grad():A.copy_(initial_A);B.copy_(initial_B)
    A.grad=None;B.grad=None
    if not torch.equal(A,initial_A) or not torch.equal(B,initial_B) or before!=weight_probes(runtime.model):
        raise ValueError('No-optimizer smoke changed a parameter or frozen weight probe')
    for path in root.rglob('*'):
        if path.is_file():sources[str(path)]=sha(path)
    create_json(root/'complete.json',dict(status='pass',labels_read=False,model=model,checks=checks,
        sources_sha256=sources,optimizer_updates=0,parameters_restored=True,backbone_probe_hashes=before,
        backbone_requires_grad_count=0,adapter_parameter_count=28672,
        interpretation='Label-free real gradient and exact initial/zero replay checks. No efficacy claim.'))


def train(model):
    spec=policy();gate=BASE/'actual_gradient_gate_v1.json';g=json.loads(gate.read_text())
    if g['status']!='pass' or g['labels_read'] or g['models']!=['qwen','roboreward']:
        raise ValueError('Both actual smoke audits must pass before any matrix training')
    verify_sources(g['sources_sha256']);root=BASE/'training'/model
    if root.exists():raise ValueError('Immutable fixed matrix training already has outputs')
    root.mkdir(parents=True);samples=population();ids=[s['example_id'] for s in samples]
    all_labels=json.loads((OLD/'labels_for_scoring_only.json').read_text());labels={e:all_labels[e] for e in ids};del all_labels
    targets={e:int(v['reward'])-1 for e,v in labels.items()};counts={c:sum(t==c for t in targets.values()) for c in [0,4]}
    if counts!={0:45,4:25} or set(targets.values())!={0,4}:raise ValueError('Changed accepted discovery training labels')
    runtime=setup(model);A=runtime.controller.A;B=runtime.controller.B;params=[A,B]
    optimizer=torch.optim.Adam(params,lr=.001);before=weight_probes(runtime.model)
    if [id(p) for group in optimizer.param_groups for p in group['params']]!=[id(A),id(B)]:raise ValueError('Only A/B may enter Adam')
    gate_path,gate_record=fixed_gates(model);ranks={p:ranking(model,p) for p in PROTOCOLS}
    create_json(root/'training_manifest.json',dict(policy=str(POLICY),policy_sha256=sha(POLICY),training_ids=ids,labels_used=ids,
        labels_source_sha256=sha(OLD/'labels_for_scoring_only.json'),adapter_parameter_count=28672,
        fixed_gate_file=str(gate_path),fixed_gate_sha256=sha(gate_path),class_counts=counts,
        backbone_trainable_parameter_count=0,gate_trainable_parameter_count=0,prior_gate_forward_backward_steps=4200,
        independent_examples=70,independent_video_groups=28))
    for protocol in PROTOCOLS:
        for scope in SCOPES:create_json(root/'rankings'/protocol/f'ranking_{scope}.json',ranks[protocol][scope])
    create_json(root/'initial_matrices.json',dict(A=A.detach().cpu().tolist(),B=B.detach().cpu().tolist()))
    steps=0;updates=0;accumulated=0.;start=time.time();optimizer.zero_grad(set_to_none=True)
    for epoch in range(3):
        entries=schedule(70,epoch);create_json(root/f'schedule_epoch_{epoch+1}.json',entries)
        for index,protocol,scope,k in entries:
            runtime.cfg.clear();runtime.cfg.update(config(model,protocol));sample=samples[index];target=targets[sample['example_id']]
            logits,diagnostics,maps=runtime.differentiable_logits([sample],f'{scope}:target:{k}',ranks[protocol])
            ce=F.cross_entropy(logits,torch.tensor([target],device=logits.device))*(70/(2*counts[target]))
            penalty=operator_penalty(A,B);loss=ce+penalty
            if not torch.isfinite(loss):raise ValueError('Nonfinite matrix training loss')
            (loss/8).backward();accumulated+=float(loss.detach());steps+=1
            if steps%8==0:
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('Invalid matrix gradient')
                if runtime.controller.theta is not None or any(p.grad is not None or p.requires_grad for p in runtime.model.parameters()):
                    raise ValueError('A frozen model/gate parameter received gradients')
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.));optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                if any(not torch.isfinite(p).all() for p in params):raise ValueError('Nonfinite optimized matrix')
                append(root/'optimizer_events.jsonl',[dict(event='optimizer_step',time=time.time(),epoch=epoch+1,
                    forward_backward_steps=steps,optimizer_steps=updates,mean_loss=accumulated/8,gradient_norm_before_clip=norm,
                    A_l2=float(A.detach().norm()),B_l2=float(B.detach().norm()),operator_penalty=float(operator_penalty(A,B).detach()))])
                accumulated=0.
                if updates%5==0:print(model,'matrix training',steps,'/4200',updates,'/525',flush=True)
            del logits,ce,penalty,loss
        create_json(root/f'epoch_{epoch+1}_matrices_for_audit_only.json',dict(A=A.detach().cpu().tolist(),B=B.detach().cpu().tolist(),
            inference_eligible=False,forward_backward_steps=steps,optimizer_steps=updates))
    if steps!=4200 or updates!=525 or before!=weight_probes(runtime.model) or sha(gate_path)!=runtime.cfg['learned_gate_sha256']:
        raise ValueError('Changed duration, fixed gate artifact or frozen backbone probe')
    create_json(root/'final_adapter.json',dict(status='complete',model=model,variant=VARIANT,policy=str(POLICY),policy_sha256=sha(POLICY),
        A=A.detach().cpu().tolist(),B=B.detach().cpu().tolist(),rank=4,adapter_parameter_count=28672,
        training_ids=ids,validation_labels_used=[],forward_backward_steps=steps,optimizer_steps=updates,
        fixed_gate_file=str(gate_path),fixed_gate_sha256=sha(gate_path),backbone_requires_grad_count=0,gate_trainable_parameter_count=0,
        backbone_probe_hashes=before,selection_rule='Final fixed step only',elapsed_seconds=time.time()-start))
    print(root/'final_adapter.json',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','roboreward'],required=True)
    p.add_argument('--phase',choices=['smoke','train'],required=True);args=p.parse_args();policy()
    run=snapshot(vars(args));create_json(BASE/'worker_starts'/f'{run["run_id"]}.json',run)
    (smoke if args.phase=='smoke' else train)(args.model)


if __name__=='__main__':main()
