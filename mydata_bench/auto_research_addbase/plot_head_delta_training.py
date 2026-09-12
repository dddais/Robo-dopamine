"""Audited fixed training trajectories and label-free final operator diagnostics."""
import csv
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mydata_bench.addbase_eval.prepare import create_json
from .prepare import OUT
from .head_delta_worker import BASE, sha, verify_sources


def main():
    audit_path=BASE/'fixed_training_audit_v1.json';audit=json.loads(audit_path.read_text())
    if audit['status']!='pass' or audit['labels_read']:raise ValueError('Actual complete training audit required')
    verify_sources(audit['sources_sha256']);sources={str(audit_path):sha(audit_path)}
    events=[];operators=[];fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for i,model in enumerate(['qwen','roboreward']):
        root=BASE/'training'/model;p=root/'optimizer_events.jsonl';sources[str(p)]=sha(p)
        trajectory=[json.loads(line) for line in p.read_text().splitlines()]
        if len(trajectory)!=525:raise ValueError('All fixed optimizer updates required')
        events.extend(dict(model=model,**e) for e in trajectory)
        x=np.arange(1,526);y=np.array([e['mean_loss'] for e in trajectory])
        axes[i,0].plot(x,y,color='#999999',alpha=.45,lw=.7,label='Recorded 8-example update mean')
        axes[i,0].plot(x[24:],np.convolve(y,np.ones(25)/25,mode='valid'),color='#265a7f',label='25-update moving mean')
        for boundary in [175,350]:axes[i,0].axvline(boundary,color='gray',ls='--',lw=.8)
        axes[i,0].set(title=f'{model}: training objective',xlabel='Optimizer update',ylabel='Weighted five-class CE + regularizer')
        if i==0:axes[i,0].legend(fontsize=8)
        p=root/'final_adapter.json';sources[str(p)]=sha(p);record=json.loads(p.read_text())
        A=np.asarray(record['A'],dtype=np.float64);B=np.asarray(record['B'],dtype=np.float64)
        layer_rows=[]
        for j in range(28):
            matrix=B[j]@A[j];singular=np.linalg.svd(matrix,compute_uv=False)
            transformed=np.linalg.svd(np.eye(128)+matrix,compute_uv=False)
            entry=dict(model=model,layer=j+8,operator_frobenius=float(np.linalg.norm(matrix)),operator_spectral=float(singular[0]),
                operator_singular_1=float(singular[0]),operator_singular_2=float(singular[1]),operator_singular_3=float(singular[2]),operator_singular_4=float(singular[3]),
                identity_plus_operator_min_singular=float(transformed[-1]),identity_plus_operator_max_singular=float(transformed[0]))
            operators.append(entry);layer_rows.append(entry)
        layers=np.arange(8,36)
        axes[i,1].plot(layers,[r['operator_frobenius'] for r in layer_rows],'o-',label='Frobenius norm')
        axes[i,1].plot(layers,[r['operator_spectral'] for r in layer_rows],'s-',label='Spectral norm')
        axes[i,1].set(title=f'{model}: final B A',xlabel='Layer',ylabel='Operator norm');axes[i,1].legend(fontsize=8)
        axes[i,2].plot(layers,[r['identity_plus_operator_max_singular'] for r in layer_rows],'o-',label='Maximum singular value')
        axes[i,2].plot(layers,[r['identity_plus_operator_min_singular'] for r in layer_rows],'s-',label='Minimum singular value')
        axes[i,2].axhline(1,color='gray',lw=.8);axes[i,2].set(title=f'{model}: linear map I + B A',xlabel='Layer',ylabel='Singular value')
        axes[i,2].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('Round33: fixed discovery-only training and final rank4 increment operators')
    fig.supxlabel('70 examples / 28 video groups per model; prior gate training is additional. No validation efficacy is inferred from this plot.\nI + B A describes the float32 increment map; actual output uses two separate BF16 additions.',fontsize=9)
    dest=OUT/'analysis'/time.strftime('figures_head_delta_training_%Y%m%d_%H%M%S');dest.mkdir()
    fig.savefig(dest/'training_and_operators.png',dpi=150);fig.savefig(dest/'training_and_operators.pdf');plt.close(fig)
    for name,rows in [('optimizer_events',events),('final_operators',operators)]:
        with (dest/f'{name}.csv').open('x') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    verify_sources(sources);create_json(dest/'manifest.json',dict(sources_sha256=sources,labels_read=False,
        interpretation='Descriptive training trajectories and final operator diagnostics; no efficacy or generalization selection.'))
    print(dest,flush=True)


if __name__=='__main__':main()
