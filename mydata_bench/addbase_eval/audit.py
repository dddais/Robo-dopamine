"""Completion gate and evidence audit; failures remain visible, never imputed."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import argparse
import yaml
from .prepare import ROOT,OUT,create_json
from .run import latest


def audit_official_steps(folder, condition, outcomes):
    """Check every selected rollout step against the same-condition predecessor."""
    steps={}
    path=folder/'steps'/(condition.replace(':','_')+'.jsonl')
    with path.open() as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            assert row['condition']==condition and 1<=row['step']<=7
            diagnostic_ok=True
            if condition!='baseline' and row['status'] in {'ok','parse_error'}:
                diagnostics=row.get('attention_diagnostics',{})
                diagnostic_ok=bool(diagnostics) and sum(len(d['heads']) for d in diagnostics.values())==int(condition.split(':')[-1])
                diagnostic_ok=diagnostic_ok and all(int(layer)>=8 and d['prefill_calls']>0 and d['all_query_rows']
                    and d['causal_mask_preserved'] and d['generated_text_key_bias']==0 for layer,d in diagnostics.items())
            steps[(row['example_id'],row['step'])]={
                'progress':row.get('progress'),'previous':row.get('previous_percentage'),'status':row['status'],
                'prompt_hash':row.get('token_audit',{}).get('prompt_sha256'),'diagnostic_ok':diagnostic_ok}
    checks=Counter()
    for eid,outcome in outcomes.items():
        count=outcome['step_count'];curve=outcome['progress_curve']
        assert 1<=count<=7 and len(curve)==count+1 and curve[0]==0
        for step in range(1,count+1):
            row=steps[(eid,step)]
            assert row['progress']==curve[step],(condition,eid,step,'curve mismatch')
            assert row['diagnostic_ok'],(condition,eid,step,'inactive step hooks')
            if row['previous'] is not None:
                assert curve[step-1] is not None
                assert abs(row['previous']/100-curve[step-1])<1e-9,(condition,eid,step,'predecessor mismatch')
                checks['official_model_step_predecessors_checked']+=1
            if step<count:assert row['status']=='ok'
            if condition!='baseline' and row['status'] in {'ok','parse_error'}:
                checks['official_steps_with_active_hooks_checked']+=1
            checks['official_step_records_checked']+=1
        assert steps[(eid,count)]['prompt_hash']==outcome.get('token_audit',{}).get('prompt_sha256')
        if outcome['status']=='ok':assert count==7 and steps[(eid,count)]['status']=='ok'
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='completion_audit_v1.json');args=parser.parse_args()
    inputs=json.loads((OUT/'inputs.json').read_text())
    allids={s['example_id'] for s in inputs};cohort={s['example_id'] for s in inputs if s['cohort']}
    report={'experiments':{},'expected_experiments':12,'issues':[],'evidence_checks':Counter()}
    for path in json.loads((OUT/'matrix.json').read_text()):
        cfg=yaml.safe_load(Path(path).read_text());folder=Path(cfg['output_dir'])
        conditions=['baseline']+[f'{s}:{kind}:{k}' for s in cfg['scopes'] for k in cfg['top_k'] for kind in ['target']+cfg['controls']]
        experiment={}
        for condition in conditions:
            p=folder/'predictions'/(condition.replace(':','_')+'.jsonl')
            rows=latest(p);expected=allids if condition=='baseline' else cohort
            extra=set(rows)-expected;missing=expected-set(rows)
            statuses=Counter(r['status'] for r in rows.values())
            infeasible=sum(str(r.get('error','')).startswith('Wrong-region control unavailable') for r in rows.values())
            format_failures=sum(bool(r.get('parse_error')) for r in rows.values())
            experiment[condition]={'expected':len(expected),'observed':len(rows),'statuses':dict(statuses),
                                  'missing':sorted(missing),'extra':sorted(extra),
                                  'infeasible_control_records':infeasible,'format_failure_records':format_failures,
                                  'other_invalid_records':len(rows)-statuses['ok']-infeasible-format_failures,
                                  'negative_progress':sum(r.get('progress') is not None and r['progress']<0 for r in rows.values()),
                                  'percentage_clipped':sum(bool(r.get('percentage_clipped')) for r in rows.values()),
                                  'generation_at_token_cap':sum(r.get('generated_tokens',0)>=cfg.get('max_new_tokens',512) for r in rows.values())}
            if missing or extra:report['issues'].append(f'{folder.name}/{condition}: missing={len(missing)}, extra={len(extra)}')
            if experiment[condition]['other_invalid_records']:
                report['issues'].append(f"{folder.name}/{condition}: unclassified invalid records={experiment[condition]['other_invalid_records']}")
            for eid,r in rows.items():
                assert r['example_id']==eid and r['condition']==condition
                assert r['status'] in {'ok','parse_error','runtime_error','incomplete_rollout'}
                if r['status']!='ok':
                    error=r.get('error','')
                    expected_infeasible=('wrong_region' in condition and str(error).startswith('Wrong-region control unavailable'))
                    if error and not expected_infeasible:
                        report['issues'].append(f'{folder.name}/{condition}/{eid}: unexpected runtime error: {error}')
                    continue
                assert math.isfinite(r['progress']) and -1<=r['progress']<=1
                audit=r['token_audit']
                if audit.get('target'):
                    for scope in cfg['scopes']:
                        target=set(audit['target'][scope]);negative=set(audit['negative'][scope]);wrong=set(audit['wrong'][scope])
                        assert target and not target&negative and (target|negative)<=set(audit['visual'])
                        if wrong:assert not target&wrong and len(target)==len(wrong)
                        report['evidence_checks']['token_domain_checks']+=1
                if condition!='baseline':
                    diagnostics=r['attention_diagnostics']
                    assert diagnostics, (folder.name,condition,eid,'empty diagnostics')
                    num_heads=sum(len(d['heads']) for d in diagnostics.values())
                    assert num_heads==int(condition.split(':')[-1])
                    for layer,d in diagnostics.items():
                        assert int(layer)>=8 and d['prefill_calls']>0 and d['all_query_rows'] and d['causal_mask_preserved']
                        assert d['generated_text_key_bias']==0
                    report['evidence_checks']['steered_predictions_with_active_hooks']+=1
                if cfg['model']=='sole' and cfg['protocol']=='official':
                    assert r['step_count']==7 and len(r['progress_curve'])==8 and r['progress_curve'][0]==0
                    assert abs(r['previous_percentage']/100-r['progress_curve'][-2])<1e-9
                    assert r['progress']==r['progress_curve'][-1]
                    report['evidence_checks']['official_terminal_recurrence_checks']+=1
            if cfg['model']=='sole' and cfg['protocol']=='official':
                report['evidence_checks'].update(audit_official_steps(folder,condition,rows))
            report['evidence_checks']['predictions_observed']+=len(rows)
        for scope in cfg['scopes']:
            p=folder/f'ranking_{scope}.json'
            if not p.exists():report['issues'].append(f'{folder.name}/{scope}: ranking missing');continue
            rank=json.loads(p.read_text());assert len(rank['ranking'])==28*32
            assert all(r['layer']>=8 for r in rank['ranking'])
        report['experiments'][folder.name]=experiment
    report['complete']=not report['issues']
    report['definition']='All 1213 baseline / 846 per intervention records attempted; parse failures and infeasible controls explicitly counted, not called valid outputs.'
    report['evidence_checks']=dict(report['evidence_checks'])
    report['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for directory in ['addbase_eval','meter_eval','top_eval']
                             for p in (ROOT/'mydata_bench'/directory).glob('*.py')}
    create_json(OUT/args.output_name,report)
    print(json.dumps({'complete':report['complete'],'issues':report['issues'],'evidence_checks':report['evidence_checks']},indent=2))
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
