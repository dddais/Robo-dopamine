"""Discovery-only screening of baseline minus attention-suppressed evidence.

These are explicitly derived candidates, never claimed to be new GPU executions.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .analyze import for_field
from .branch_ablation import native_branch


def derived_row(baseline,negative,model):
    if baseline['status']!='ok' or negative['status']!='ok':raise ValueError('Incomplete native branch')
    b=np.asarray(baseline['native_class_logits_positive'],dtype=np.float32)
    n=np.asarray(negative['native_class_logits_negative'],dtype=np.float32)
    if b.shape!=n.shape:raise ValueError('Native branch shape mismatch')
    combined=2*b-n
    record={'status':'ok','example_id':baseline['example_id'],
            'native_class_logits_positive':combined.tolist()}
    if model=='meter':
        b=np.asarray([baseline['success_logit_positive']],dtype=np.float32)
        n=np.asarray([negative['success_logit_negative']],dtype=np.float32)
        record['success_logit_positive']=float((2*b-n)[0])
    result=native_branch(record,'positive',model)
    result['derived_from_recorded_branch']='2 * unsteered baseline - recorded target-suppressed negative'
    result['requires_actual_forward_verification']=True
    return result


def main():
    splits=json.loads((OUT/'splits.json').read_text())
    requested=splits['discovery']
    labels=json.loads((OLD/'labels_for_scoring_only.json').read_text())
    points=[];details={};sources={};agreement=[]
    destination=OUT/'derived_candidates'/time.strftime('zero_positive_%Y%m%d_%H%M%S')
    for folder in sorted((OUT/'experiments').glob('*_evidence_a1/discovery')):
        cfg=json.loads((folder/'runtime_config.json').read_text());model=cfg['model']
        if set(json.loads((folder/'requested_ids.json').read_text()))!=set(requested):raise ValueError('Discovery only')
        baseline_path=folder/'predictions/baseline.jsonl';baseline=latest(baseline_path)
        if set(baseline)!=set(requested):continue
        groups=defaultdict(list)
        for path in sorted(folder.glob('mass_transport_s*/predictions/*_target_*.jsonl')):
            rows=latest(path)
            if set(rows)==set(requested) and all(x['status']=='ok' for x in rows.values()):
                groups[path.stem].append((path,rows))
        sources[str(baseline_path)]=hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        for condition,observations in groups.items():
            path,negative=observations[0]
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            max_difference=0.
            for other_path,other in observations[1:]:
                for eid in requested:
                    a=np.asarray(negative[eid]['native_class_logits_negative'])
                    b=np.asarray(other[eid]['native_class_logits_negative'])
                    max_difference=max(max_difference,float(np.abs(a-b).max()))
                    if model=='meter':max_difference=max(max_difference,abs(negative[eid]['success_logit_negative']-other[eid]['success_logit_negative']))
            agreement.append({'experiment':folder.parent.name,'condition':condition,
                              'source_positive_strengths':len(observations),'max_negative_logit_difference':max_difference})
            if max_difference!=0:raise ValueError('Negative branch changed with positive dose; investigate first')
            rows={eid:derived_row(baseline[eid],negative[eid],model) for eid in requested}
            create_json(destination/'derived_rows'/folder.parent.name/f'{condition}.json',rows)
            for field in ['progress']+(['success_probability'] if model=='meter' else []):
                base=summary(for_field(baseline,field),labels,requested)
                result=summary(for_field(rows,field),labels,requested)
                for threshold in ['0.125/0.875','0.2/0.8']:
                    point={'experiment':folder.parent.name,'condition':condition,'field':field,
                        'method':'mass_transport_s0','threshold':threshold,'population':'discovery',
                        'derived_only':True,'expected':len(requested),'mae':result['mae'],'baseline_mae':base['mae'],
                        'delta_mae':result['mae']-base['mae']}
                    for split in ['all','suc','fail']:
                        point['delta_'+split]=result['accuracy'][threshold][split]['rate_all_expected']-base['accuracy'][threshold][split]['rate_all_expected']
                    point['meets_descriptive_gate']=(point['delta_mae']<0 and point['delta_all']>=.1-1e-12 and point['delta_suc']>0 and point['delta_fail']>0)
                    points.append(point)
                details[f'{folder.parent.name}/{condition}/{field}']={'baseline':base,'candidate':result}
    create_json(destination/'points.json',points)
    create_json(destination/'details.json',details)
    create_json(destination/'provenance.json',{'sources':sources,'negative_branch_agreement':agreement,
        'formula':'2 * baseline logits - negative logits','scope':'Frozen discovery only; requires actual forward verification',
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(destination)
    groups=defaultdict(list)
    for p in points:
        if p['threshold']=='0.125/0.875':groups[(p['experiment'],p['field'])].append(p)
    for name,items in groups.items():
        print(name)
        for p in sorted(items,key=lambda p:(not p['meets_descriptive_gate'],p['delta_mae']))[:2]:
            print(p['condition'],'MAE',round(p['baseline_mae'],3),'->',round(p['mae'],3),
                  'delta pp',*[round(p['delta_'+s]*100,2) for s in ['all','suc','fail']],
                  'derived PASS',p['meets_descriptive_gate'])


if __name__=='__main__':main()
