"""Label-free geometry of the recorded visual/task evidence vectors.

Subtracting each vector's class mean removes softmax-irrelevant offsets.
This diagnostic does not select weights, change predictions, or prove semantic
causality. It tests whether an equal average attenuates the visual update.
"""
import hashlib
import json
import time

import numpy as np

from .prepare import OUT
from mydata_bench.addbase_eval.prepare import create_json


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                quantiles_10_25_50_75_90=np.quantile(values,[.1,.25,.5,.75,.9]).tolist() if len(values) else None)


def main():
    frozen_path = OUT / 'selection_factorized_family_primary_v1.json'
    frozen = json.loads(frozen_path.read_text())
    selected = {(p['model'],p['protocol'],p['scope'],p['center_k']) for p in frozen['selected']}
    ids = set(json.loads((OUT/'splits.json').read_text())['discovery'])
    hashes = {str(frozen_path):hashlib.sha256(frozen_path.read_bytes()).hexdigest()}
    records = []
    for model in ['qwen','roboreward']:
        for protocol in ['image_text','text_image','interleaved','text_video','video_text']:
            folder = OUT/'experiments'/f'{model}_{protocol}_uniform_factorized_evidence_a1'/'discovery'
            events = [json.loads(l) for l in (folder/'worker_events.jsonl').read_text().splitlines()]
            if events[-1]['event'] != 'complete':
                raise ValueError('Wait for full discovery')
            for path in sorted(folder.glob('binding_transport_s4/predictions/*.jsonl')):
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows = {r['example_id']:r for r in map(json.loads,path.read_text().splitlines())}
                if set(rows)!=ids or any(r['status']!='ok' for r in rows.values()):
                    raise ValueError('Require exact valid discovery coverage')
                ordered = [rows[e] for e in sorted(ids)]
                p = np.asarray([r['native_class_logits_positive'] for r in ordered],dtype=float)
                v = np.asarray([r['native_class_logits_negative'] for r in ordered],dtype=float)
                t = np.asarray([r['native_class_logits_negative_task'] for r in ordered],dtype=float)
                if any(z.shape!=(70,5) or not np.isfinite(z).all() for z in [p,v,t]):
                    raise ValueError('Five finite native scores per branch required')
                dv,dt = p-v,p-t
                dv -= dv.mean(1,keepdims=True); dt -= dt.mean(1,keepdims=True)
                nv,nt = np.linalg.norm(dv,axis=1),np.linalg.norm(dt,axis=1)
                nm = np.linalg.norm(.5*(dv+dt),axis=1)
                valid = (nv>1e-12)&(nt>1e-12)
                cosine = np.sum(dv[valid]*dt[valid],axis=1)/(nv[valid]*nt[valid])
                scope,k = path.stem.split('_target_'); k=int(k)
                row = dict(model=model,protocol=protocol,scope=scope,k=k,n=70,
                    selected_primary=(model,protocol,scope,k) in selected,
                    visual_zero_count=int(np.sum(nv<=1e-12)),task_zero_count=int(np.sum(nt<=1e-12)),
                    visual_norm=describe(nv),task_norm=describe(nt),cosine=describe(cosine),
                    cosine_negative_fraction=float(np.mean(cosine<0)) if len(cosine) else None,
                    task_over_visual_norm=describe(nt[nv>1e-12]/nv[nv>1e-12]),
                    mixture_over_visual_norm=describe(nm[nv>1e-12]/nv[nv>1e-12]),
                    mixture_smaller_than_visual_fraction=float(np.mean(nm[nv>1e-12]<nv[nv>1e-12])) if np.any(nv>1e-12) else None,
                    source=str(path))
                records.append(row)
    if len(records)!=60:
        raise ValueError('Expected two complete thirty-condition matrices')
    output = OUT/'analysis'/time.strftime('factorized_vector_diagnostics_%Y%m%d_%H%M%S.json')
    create_json(output,dict(created_at=time.time(),labels_read=False,sources_sha256=hashes,records=records,
        definition='Class-centered d_visual=positive-visual_negative, d_task=positive-task_negative, d_mix=(d_visual+d_task)/2',
        interpretation='Descriptive geometry of native evidence updates, not correctness or causal semantics. '
                       'Small task norm or opposing directions can attenuate the equal-weight visual update. '
                       'Zero vectors are reported and excluded from undefined ratios/cosines. No coefficients selected.'))
    print(output)
    def median(record):
        values=record['quantiles_10_25_50_75_90']
        return round(values[2],3) if values else None
    for r in records:
        if r['selected_primary']:
            print(r['model'],r['protocol'],r['scope'],r['k'],
                  'median task/visual',median(r['task_over_visual_norm']),
                  'median cosine',median(r['cosine']),
                  'median mixture/visual',median(r['mixture_over_visual_norm']))


if __name__=='__main__':
    main()
