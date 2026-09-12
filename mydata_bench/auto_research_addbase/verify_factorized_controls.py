"""Verify every requested actual three-branch control before paired scoring."""
import argparse
import json
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .compare_native_bias import native, heads, same_input
from .empirical_profile import sha


def verify(model, protocol, scope, ks, gain=1):
    if gain not in [1,2]: raise ValueError('Only explicitly registered factorized gains are supported')
    root = OUT/'experiments'/f'{model}_{protocol}_uniform_factorized_evidence_a{gain}'/'full_cohort'
    ids = json.loads((OUT/'splits.json').read_text())['full_cohort']
    sources = {}; checks = []; failures = []
    def read(path, rows=False):
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    if set(read(root/'requested_ids.json')) != set(ids):
        raise ValueError('Require full-cohort requested population')
    cfg = read(root/'runtime_config.json')
    expected = dict(contrast_negative_mode='visual_and_task',contrast_weight=gain,negative_strength=4,
        negative_task_strength=4,task_binding_fraction=.5,task_binding_distribution='uniform')
    if any(cfg.get(k) != v for k,v in expected.items()):
        raise ValueError('Frozen three-branch configuration differs')
    def actual(rows, requested, baseline=False):
        if gain==1: return native(rows,requested,True)
        from .factorized_gain_tasks import verify_native
        return verify_native(rows,requested,2.,baseline=baseline)
    baseline = actual(read(root/'predictions/baseline.jsonl',True),ids,baseline=True)
    ranking = read(OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'/f'ranking_{scope}.json')['ranking']
    for k in ks:
        for kind in ['target','wrong_region','low_rank']:
            path = root/'binding_transport_s4/predictions'/f'{scope}_{kind}_{k}.jsonl'
            rows = read(path,True)
            if set(rows) != set(ids):
                raise ValueError('Wait for each explicitly requested complete control file')
            valid = {e:r for e,r in rows.items() if r['status']=='ok'}
            for e,r in rows.items():
                if r['status'] != 'ok':
                    if kind != 'wrong_region' or 'Wrong-region control unavailable: insufficient disjoint cells' not in str(r):
                        raise ValueError('Unexpected control error')
                    failures.append(dict(k=k,control=kind,example_id=e,error=r))
            actual(valid,list(valid))
            selected = ranking[-k:] if kind == 'low_rank' else ranking[:k]
            expected_heads = {(h['layer'],h['head']) for h in selected}
            if len(expected_heads) != k: raise ValueError('Duplicate heads')
            for e,row in valid.items():
                if not same_input(row,baseline[e]) or row['condition'] != f'{scope}:{kind}:{k}':
                    raise ValueError('Actual input or condition differs')
                for field,method,strength in [('attention_diagnostics','binding_transport',4),
                        ('negative_attention_diagnostics','mass_transport',-4),
                        ('task_negative_attention_diagnostics','binding_task_suppression',4)]:
                    if heads(row,field) != expected_heads: raise ValueError('Actual three-branch heads differ')
                    for d in row[field].values():
                        if d['method'] != method or d['strength'] != strength:
                            raise ValueError('Actual control operator differs')
                        if method == 'binding_transport' and (d['task_binding_fraction'] != .5 or d['task_binding_distribution'] != 'uniform'):
                            raise ValueError('Actual instruction positive differs')
                        if method == 'binding_task_suppression' and d['task_logit_strength'] != -4:
                            raise ValueError('Actual task negative differs')
                if kind == 'wrong_region':
                    audit = row['token_audit']; target = set(audit['target'][scope]); wrong = set(audit['wrong'][scope])
                    if not target or target & wrong or len(target) != len(wrong):
                        raise ValueError('Wrong ROI is not disjoint and equal-size')
            checks.append(dict(k=k,control=kind,expected=len(ids),valid=len(valid),
                actual_formula_heads_input_and_roi_verified=True))
    prefix = 'factorized_controls' if gain==1 else 'factorized_a2_controls'
    path = OUT/'audit'/time.strftime(f'{prefix}_{model}_{protocol}_%Y%m%d_%H%M%S.json')
    create_json(path,dict(status='pass',labels_read=False,sources_sha256=sources,checks=checks,
        permitted_unavailable_controls=failures,verified_intervention_rows=sum(c['valid'] for c in checks),
        interpretation='All requested observations accounted for before scoring; invalid wrong-region rows are retained, never imputed.'))
    print(path)
    print(checks)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['qwen','roboreward'])
    parser.add_argument('--protocol',required=True,choices=['image_text','text_image','interleaved','text_video','video_text'])
    parser.add_argument('--scope',required=True,choices=['all_frames','last_frame'])
    parser.add_argument('--ks',required=True,nargs='+',type=int)
    parser.add_argument('--gain',type=int,choices=[1,2],default=1)
    args = parser.parse_args(); verify(args.model,args.protocol,args.scope,args.ks,args.gain)


if __name__ == '__main__':
    main()
