"""Audit actual spatial-context smoke outputs, without scoring labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .context import context_profile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiments',nargs='+',required=True)
    args=p.parse_args()
    reports=[];sources={}
    def source(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path)
    for experiment in args.experiments:
        folder=OUT/'experiments'/experiment/'discovery_smoke'
        if not (folder/'worker_events.jsonl').exists():raise ValueError('Wait for complete smoke')
        cfg=json.loads((folder/'runtime_config.json').read_text())
        if cfg['context_radius']!=.15 or cfg['contrast_weight']!=1:raise ValueError('Frozen setting differs')
        ids=json.loads((folder/'requested_ids.json').read_text())
        if len(ids)!=8:raise ValueError('Matching-eight-sample smoke required')
        base=source(folder/'predictions/baseline.jsonl')
        old=OUT/'experiments'/f"{cfg['model']}_{cfg['protocol']}_evidence_a1"/'discovery'
        old_base=source(old/'predictions/baseline.jsonl')
        baseline_diff=0.
        for eid in ids:
            if base[eid]['status']!='ok':raise ValueError('Baseline failure')
            if base[eid]['token_audit']['input_ids_sha256']!=old_base[eid]['token_audit']['input_ids_sha256']:
                raise ValueError('Baseline batch or input mismatch')
            baseline_diff=max(baseline_diff,float(np.max(np.abs(np.asarray(base[eid]['native_class_logits_positive'])-old_base[eid]['native_class_logits_positive']))))
        if baseline_diff!=0:raise ValueError('Context option changed unsteered baseline')
        rankings={scope:json.loads((folder.parent/'ranking'/f'ranking_{scope}.json').read_text())
                  for scope in ['all_frames','last_frame']}
        paths=sorted(folder.glob('context_transport_s*/predictions/*.jsonl'))
        if len(paths)!=6:raise ValueError('Expected both scopes and target/wrong/low-rank at the frozen smoke dose')
        for path in paths:
            rows=source(path)
            if set(rows)!=set(ids):raise ValueError('Partial context condition')
            # File names use scope and arm names containing underscores; the
            # original condition is an unambiguous colon-delimited record.
            condition=next(iter(rows.values()))['condition']
            scope,kind,budget=condition.split(':');budget=int(budget)
            ordered=rankings[scope]['ranking']
            expected=ordered[-budget:] if kind=='low_rank' else ordered[:budget]
            expected={(r['layer'],r['head']) for r in expected}
            old_rows=source(old/'mass_transport_s4/predictions'/path.name) if kind=='target' else None
            valid=0;unavailable=[];probability_error=0.;negative_difference=0.;positive_change=0.
            for eid in ids:
                row=rows[eid]
                if row['status']!='ok':
                    if kind=='wrong_region' and 'insufficient disjoint cells' in json.dumps(row).lower():
                        unavailable.append(eid);continue
                    raise ValueError(f'Unexpected context smoke failure: {experiment} {condition} {eid}')
                valid+=1
                for field,method in [('attention_diagnostics','context_transport'),('negative_attention_diagnostics','mass_transport')]:
                    actual={(int(layer),head) for layer,diag in row[field].items() for head in diag['heads']}
                    if actual!=expected:raise ValueError('Actual hook head selection mismatch')
                    if not all(diag['method']==method and diag['domain_mass_preserved'] and diag['all_query_rows'] and diag['causal_mask_preserved'] for diag in row[field].values()):
                        raise ValueError('Actual hook method or scope mismatch')
                mapping=row['token_audit']
                region='wrong' if kind=='wrong_region' else 'target'
                _,_,audit=context_profile(mapping,scope,region,.15)
                if mapping['context_kernel_audit'][scope]!=audit:
                    raise ValueError('Saved spatial profile differs from accepted token geometry')
                ids_target,target,_=context_profile(mapping,scope,'target',.15)
                if mapping['wrong'][scope]:
                    ids_wrong,wrong,_=context_profile(mapping,scope,'wrong',.15)
                    if ids_target!=ids_wrong:raise ValueError('Control visual domains differ')
                    np.testing.assert_array_equal(np.sort(target),np.sort(wrong))
                positive=np.asarray(row['native_class_logits_positive'],dtype=np.float32)
                negative=np.asarray(row['native_class_logits_negative'],dtype=np.float32)
                combined=(2*positive-negative).astype(np.float64)
                if combined.shape!=(5,):raise ValueError('All five reward logits must remain available')
                probability=np.exp(combined-combined.max());probability/=probability.sum()
                probability_error=max(probability_error,float(np.max(np.abs(probability-np.asarray(row['native_class_probabilities'])))))
                if int(probability.argmax())+1!=row['reward']:raise ValueError('Native five-way readout mismatch')
                if old_rows is not None:
                    negative_difference=max(negative_difference,float(np.max(np.abs(negative-np.asarray(old_rows[eid]['native_class_logits_negative'])))))
                    positive_change=max(positive_change,float(np.max(np.abs(positive-np.asarray(old_rows[eid]['native_class_logits_positive'])))))
            if not valid or probability_error>=1e-5 or negative_difference!=0:
                raise ValueError(f'Context check failed: {experiment}/{condition}; valid={valid}; probability_error={probability_error}; negative_difference={negative_difference}')
            if kind=='target' and positive_change==0:raise ValueError('Positive context intervention was inert')
            reports.append({'experiment':experiment,'condition':condition,'valid':valid,'unavailable_controls':unavailable,
                            'baseline_max_logit_difference':baseline_diff,
                            'binary_reference_compared':old_rows is not None,
                            'negative_max_logit_difference':negative_difference if old_rows is not None else None,
                            'probability_reconstruction_max_error':probability_error,
                            'positive_max_change_from_binary':positive_change if old_rows is not None else None,
                            'actual_head_budget_per_branch':budget,'spatial_gain_multiset_matched':True})
    destination=OUT/'audit'/time.strftime('context_actual_forward_%Y%m%d_%H%M%S.json')
    create_json(destination,{'status':'pass','arguments':vars(args),'checks':reports,'sources':sources,
                             'comparison_scope':'The target arm has a prior binary-core reference. Controls are checked for geometry, head selection and probability reconstruction; no prior binary-control logit equivalence is claimed.',
                             'labels_read':False,'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(destination)
    print(json.dumps(reports,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
