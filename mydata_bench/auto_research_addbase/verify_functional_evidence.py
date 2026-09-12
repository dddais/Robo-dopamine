"""Verify the fixed functional-head contrast against its positive-only run."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiments',nargs='+',required=True)
    args=p.parse_args()
    results=[];sources={}
    def source(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path)
    for experiment in args.experiments:
        folder=OUT/'experiments'/experiment/'discovery_smoke'
        if not (folder/'worker_events.jsonl').exists():raise ValueError('Wait for complete smoke')
        cfg=json.loads((folder/'runtime_config.json').read_text())
        ids=json.loads((folder/'requested_ids.json').read_text())
        if len(ids)!=8 or cfg['contrast_weight']!=1 or cfg['task_binding_fraction']!=.5:
            raise ValueError('Frozen smoke setting differs')
        reference_folder=OUT/'experiments'/f"{cfg['model']}_{cfg['protocol']}_functional"/'discovery'
        baseline=source(folder/'predictions/baseline.jsonl')
        old_baseline=source(reference_folder/'predictions/baseline.jsonl')
        for path in sorted(folder.glob('binding_transport_s4/predictions/*_target_*.jsonl')):
            rows=source(path)
            reference=source(reference_folder/'binding_transport_s4/predictions'/path.name)
            if set(rows)!=set(ids) or not set(ids)<=set(reference):raise ValueError('Missing current or reference predictions')
            scope,_,k=next(iter(rows.values()))['condition'].split(':');k=int(k)
            ranking=json.loads((folder.parent/'ranking'/f'ranking_{scope}.json').read_text())
            if ranking.get('validation_ids_used')!=[]:raise ValueError('Functional ranking must exclude validation labels')
            expected={(r['layer'],r['head']) for r in ranking['ranking'][:k]}
            error=0.
            for eid in ids:
                row=rows[eid];old=reference[eid]
                if row['status']!='ok' or old['status']!='ok':raise ValueError('Invalid native forward')
                for left,right in [(baseline[eid],old_baseline[eid]),(row,old)]:
                    if left['token_audit']['input_ids_sha256']!=right['token_audit']['input_ids_sha256']:
                        raise ValueError('Input or batch padding differs')
                    if left['native_class_logits_positive']!=right['native_class_logits_positive']:
                        raise ValueError('Positive branch is not exactly the frozen functional intervention')
                for field,method in [('attention_diagnostics','binding_transport'),('negative_attention_diagnostics','mass_transport')]:
                    actual={(int(layer),head) for layer,d in row[field].items() for head in d['heads']}
                    if actual!=expected:raise ValueError('Positive/negative head budget mismatch')
                    if not all(d['method']==method and d['all_query_rows'] and d['domain_mass_preserved'] for d in row[field].values()):
                        raise ValueError('Unexpected functional contrast hook')
                positive=np.asarray(row['native_class_logits_positive'],dtype=np.float32)
                negative=np.asarray(row['native_class_logits_negative'],dtype=np.float32)
                combined=(2*positive-negative).astype(np.float64)
                if combined.shape not in [(5,),(10,)]:raise ValueError('Native class set changed')
                probability=np.exp(combined-combined.max());probability/=probability.sum()
                error=max(error,float(np.max(np.abs(probability-np.asarray(row['native_class_probabilities'])))))
                if cfg['model']!='meter' and int(probability.argmax())+1!=row['reward']:
                    raise ValueError('Native five-way decision mismatch')
                if cfg['model']=='meter':
                    if row['success_logit_positive']!=old['success_logit_positive']:
                        raise ValueError('Positive native success head differs')
                    z=float(np.float32(2*np.float32(row['success_logit_positive'])-np.float32(row['success_logit_negative'])))
                    expected_success=np.exp(-np.logaddexp(0.,-z))
                    error=max(error,abs(row['success_probability']-expected_success),abs(row['progress']-probability@np.linspace(0,1,10)))
            if error>=1e-5:raise ValueError('Native probability composition mismatch')
            results.append({'experiment':experiment,'condition':path.stem,'n':8,'head_budget_per_branch':k,
                            'positive_logits_exactly_match_functional_alpha0':True,
                            'maximum_probability_error':error,'ranking_validation_ids_used':[]})
    if not results:raise ValueError('No completed comparison')
    destination=OUT/'audit'/time.strftime('functional_evidence_actual_%Y%m%d_%H%M%S.json')
    create_json(destination,{'status':'pass','checks':results,'sources':sources,'labels_read':False,
                             'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(destination)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
