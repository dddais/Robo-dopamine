"""Verify three real counterfactual forwards and their explicitly derived mean."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .prepare import OUT
from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=['qwen','roboreward'],required=True)
    args=parser.parse_args()
    checks=[];sources={}
    def read(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path)
    for model in args.models:
        folder=OUT/'experiments'/f'{model}_image_text_uniform_factorized_evidence_a1'/'discovery_smoke'
        reference=OUT/'experiments'/f'{model}_image_text_uniform_binding_evidence_a1'/'discovery_smoke'
        events=[json.loads(line) for line in (folder/'worker_events.jsonl').read_text().splitlines()]
        if not events or events[-1]['event']!='complete':raise ValueError('Wait for complete actual smoke')
        cfg=json.loads((folder/'runtime_config.json').read_text())
        if (cfg['contrast_negative_mode']!='visual_and_task' or cfg['contrast_weight']!=1 or
                cfg['negative_strength']!=4 or cfg['negative_task_strength']!=4 or cfg['task_binding_fraction']!=.5 or
                cfg['task_binding_distribution']!='uniform'):raise ValueError('Frozen factorized settings changed')
        ids=json.loads((folder/'requested_ids.json').read_text())
        if len(ids)!=8:raise ValueError('Eight matched smoke samples required')
        baseline,old_baseline=read(folder/'predictions/baseline.jsonl'),read(reference/'predictions/baseline.jsonl')
        paths=sorted(folder.glob('binding_transport_s4/predictions/*_target_*.jsonl'))
        if {p.stem for p in paths}!={f'{scope}_target_{k}' for scope in ['all_frames','last_frame'] for k in [8,32]}:
            raise ValueError('Four matched smoke conditions required')
        for path in paths:
            rows=read(path);previous=read(reference/'binding_transport_s4/predictions'/path.name)
            if any(set(r)!=set(ids) for r in [rows,previous,baseline,old_baseline]):raise ValueError('Matched coverage differs')
            scope,k=path.stem.split('_target_');k=int(k)
            ranking_path=folder.parent/'ranking'/f'ranking_{scope}.json'
            ranking=json.loads(ranking_path.read_text());sources[str(ranking_path)]=hashlib.sha256(ranking_path.read_bytes()).hexdigest()
            if ranking['validation_ids_used']:raise ValueError('Validation labels used for head selection')
            expected_heads={(h['layer'],h['head']) for h in ranking['ranking'][:k]}
            error=0.;task_change=0.;changed_probabilities=0
            for eid in ids:
                row,old=rows[eid],previous[eid]
                for a,b in [(row,old),(baseline[eid],old_baseline[eid])]:
                    if a['status']!='ok' or b['status']!='ok':raise ValueError('Failed native output')
                    if a['prompt']!=b['prompt'] or a['token_audit']['input_ids_sha256']!=b['token_audit']['input_ids_sha256']:
                        raise ValueError('Input or padding changed')
                    for field in ['native_class_logits_positive','native_class_logits_negative','candidate_token_ids']:
                        if a.get(field)!=b.get(field):raise ValueError('An original branch or native class set changed')
                if baseline[eid]['actual_forward_branches']!=1 or baseline[eid]['negative_mean_is_derived']:
                    raise ValueError('Baseline was not a single unmodified native forward')
                if row['actual_forward_branches']!=3 or not row['negative_mean_is_derived']:
                    raise ValueError('The three forwards and derived mean were not explicitly recorded')
                for field in ['attention_diagnostics','negative_attention_diagnostics','task_negative_attention_diagnostics']:
                    actual={(int(layer),head) for layer,item in row[field].items() for head in item['heads']}
                    if actual!=expected_heads or len(actual)!=k:raise ValueError('Counterfactual head budgets differ')
                for info in row['attention_diagnostics'].values():
                    if info['method']!='binding_transport' or info['task_binding_distribution']!='uniform':
                        raise ValueError('Positive operator changed')
                for info in row['negative_attention_diagnostics'].values():
                    if info['method']!='mass_transport' or info['strength']!=-4:raise ValueError('Visual counterfactual changed')
                for info in row['task_negative_attention_diagnostics'].values():
                    if not (info['method']=='binding_task_suppression' and info['strength']==4 and
                            info['task_logit_strength']==-4 and info['task_binding_distribution']=='exponential_suppression' and
                            info['text_domain_mass_preserved'] and info['domain_mass_preserved'] and info['all_query_rows']):
                        raise ValueError('New task counterfactual does not implement its frozen definition')
                positive=np.asarray(row['native_class_logits_positive'],dtype=np.float32)
                visual=np.asarray(row['native_class_logits_negative'],dtype=np.float32)
                task=np.asarray(row['native_class_logits_negative_task'],dtype=np.float32)
                if any(v.shape!=(5,) or not np.isfinite(v).all() for v in [positive,visual,task]):
                    raise ValueError('All five finite native logits required per actual branch')
                mean=.5*(visual+task)
                if not np.array_equal(mean,np.asarray(row['native_class_logits_negative_mean'],dtype=np.float32)):
                    raise ValueError('Derived mean differs from two recorded raw negative branches')
                combined=(2*positive-mean).astype(np.float64)
                probability=np.exp(combined-combined.max());probability/=probability.sum()
                error=max(error,float(np.max(np.abs(probability-row['native_class_probabilities']))))
                if row['reward']!=int(probability.argmax())+1 or row['progress']!=(row['reward']-1)/4:
                    raise ValueError('Final reward is not native five-way argmax')
                task_change=max(task_change,float(np.max(np.abs(task-visual))))
                changed_probabilities+=bool(np.max(np.abs(np.asarray(row['native_class_probabilities'])-old['native_class_probabilities']))>1e-6)
            if error>=1e-5 or task_change==0:raise ValueError('New branch or native composition failed actual verification')
            checks.append(dict(model=model,condition=path.stem,n=len(ids),baseline_positive_visual_negative_exact=True,
                three_actual_branches=True,head_budget_per_branch=k,negative_mean_explicitly_derived=True,
                max_probability_error=error,max_task_vs_visual_negative_logit_difference=task_change,
                changed_combined_probability_rows=changed_probabilities))
    destination=OUT/'audit'/time.strftime('factorized_actual_%Y%m%d_%H%M%S.json')
    create_json(destination,dict(status='pass',checks=checks,sources_sha256=sources,labels_read=False,
                                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(destination);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
