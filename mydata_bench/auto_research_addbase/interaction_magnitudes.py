"""Label-free descriptive sizes of the actually measured 2x2 logit effects."""
import csv
import json
import time

import numpy as np

from .prepare import OUT
from .joint_interaction import MODELS,PROTOCOLS,SCOPES,folder,matrix_ready,verify_rows
from .empirical_profile import sha
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.prepare import create_json


def softmax(z):
    p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)


def centered_norm(z):
    return np.linalg.norm(z-z.mean(-1,keepdims=True),axis=-1)


def main():
    if not matrix_ready():raise ValueError('Wait for both complete actual four-cell matrices')
    ids=json.loads((OUT/'splits.json').read_text())['discovery'];sources={};points=[]
    for m in MODELS:
        for p in PROTOCOLS:
            root=folder(m,p,'discovery')
            base_path=root/'predictions/baseline.jsonl';sources[str(base_path)]=sha(base_path)
            base=verify_rows(latest(base_path),ids,baseline=True)
            for scope in SCOPES:
                for k in [8,32,64]:
                    path=root/'binding_transport_s4/predictions'/f'{scope}_target_{k}.jsonl';sources[str(path)]=sha(path)
                    rows=verify_rows(latest(path),ids)
                    pp,mp,pm,mm=[np.asarray([rows[e]['native_cell_logits'][cell] for e in ids],dtype=np.float32).astype(float)
                        for cell in ['pp','mp','pm','mm']]
                    interaction=pp-mp-pm+mm
                    visual_main=(pp+pm-mp-mm)/2
                    task_main=(pp+mp-pm-mm)/2
                    magnitudes={name:centered_norm(z) for name,z in [('interaction',interaction),('visual_main',visual_main),('task_main',task_main)]}
                    # Class centering removes the arbitrary branch-wide logit offsets.
                    # Use the recorded float32 inference composition for output diagnostics.
                    actual=np.asarray([rows[e]['native_class_logits_combined'] for e in ids],dtype=float)
                    positive=softmax(pp);combined=softmax(actual)
                    kl=(combined*(np.log(combined)-np.log(positive))).sum(-1)
                    row=dict(model=m,protocol=p,scope=scope,k=k,n=len(ids),
                        mean_kl_combined_vs_positive=float(kl.mean()),
                        float64_recomposition_argmax_disagreements=int(np.sum((pp+interaction).argmax(-1)!=actual.argmax(-1))),
                        same_argmax_as_positive=float(np.mean(combined.argmax(-1)==positive.argmax(-1))),
                        same_argmax_as_baseline=float(np.mean(combined.argmax(-1)==np.asarray([base[e]['reward']-1 for e in ids]))))
                    for name,values in magnitudes.items():
                        row[f'{name}_centered_l2_mean']=float(values.mean())
                        row[f'{name}_centered_l2_median']=float(np.median(values))
                        row[f'{name}_centered_l2_q90']=float(np.quantile(values,.9))
                    points.append(row)
    destination=OUT/'analysis'/time.strftime('interaction_magnitudes_%Y%m%d_%H%M%S');destination.mkdir(exist_ok=False)
    create_json(destination/'points.json',points)
    with (destination/'points.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    create_json(destination/'manifest.json',dict(labels_read=False,sources_sha256=sources,
        interpretation='Descriptive within the already completed adaptive discovery experiment. '
            'Centered five-class logit main effects and interaction are algebraic internal perturbation responses, not identified semantic causes. '
            'KL and class agreements use recorded float32 combined logits; main-effect norms use float64 cell arithmetic. '
            'No new output rule, efficacy selection, or new model forwards. Repeated head conditions are not independent samples.'))
    print(destination)
    for model in MODELS:
        for protocol in PROTOCOLS:
            rows=[r for r in points if r['model']==model and r['protocol']==protocol]
            print(model,protocol,'range of same-class fractions vs positive',
                [round(f(r['same_argmax_as_positive'] for r in rows),3) for f in [min,max]],
                'mean KL range',[round(f(r['mean_kl_combined_vs_positive'] for r in rows),3) for f in [min,max]])


if __name__=='__main__':main()
