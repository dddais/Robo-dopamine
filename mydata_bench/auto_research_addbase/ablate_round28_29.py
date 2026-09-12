"""Descriptive readouts of already executed branches; no new inferred branch."""
import json
import time
import numpy as np

from .prepare import OUT
from .empirical_profile import verify_sources, sha
from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary


def readout(rows, field):
    result = {}
    for e, r in rows.items():
        z = np.asarray(r[field], dtype=float)
        if z.shape != (5,) or not np.isfinite(z).all(): raise ValueError('Require a real recorded five-class branch')
        reward = int(z.argmax())+1
        result[e] = dict(r, reward=reward, progress=(reward-1)/4,
            ablation_source_field=field, actual_additional_model_forwards=0)
    return result


def main():
    matrices = {}; sources = {}
    for variant, policy_name in [('task_content_block_evidence_a1','selection_task_content_block_family_full_v1.json'),
                                 ('resolution_evidence_a1','selection_resolution_evidence_family_full_v1.json')]:
        policy = OUT/policy_name; record = json.loads(policy.read_text()); verify_sources(record['sources_sha256'])
        sources[str(policy)] = sha(policy)
        for model in ['qwen','roboreward']:
            for protocol in ['image_text','text_image','interleaved','video_text','text_video']:
                root = OUT/'experiments'/f'{model}_{protocol}_{variant}'/'discovery'
                ids = json.loads((root/'requested_ids.json').read_text())
                basepath = root/'predictions/baseline.jsonl'; sources[str(basepath)] = sha(basepath)
                base = latest(basepath); conditions = {}
                for path in sorted((root/'bias_s6/predictions').glob('*.jsonl')):
                    sources[str(path)] = sha(path); rows = latest(path)
                    if set(rows) != set(ids) or any(r['status']!='ok' for r in rows.values()): raise ValueError('Incomplete actual branch')
                    conditions[path.stem] = rows
                if len(ids)!=70 or len(conditions)!=6: raise ValueError('Incomplete branch matrix')
                matrices[(variant,model,protocol)] = ids,base,conditions
    labels = json.loads((OLD/'labels_for_scoring_only.json').read_text())
    points = []; details = {}; coverage = {}
    for (variant,model,protocol),(ids,base,conditions) in matrices.items():
        baseline = summary(base,labels,ids)
        arms = {'original_bias6':'native_class_logits_positive','task_content_blocked':'native_class_logits_blocked'} if variant.startswith('task_content') else {
            'low_resolution_bias6':'native_class_logits_low','high_resolution_bias6':'native_class_logits_high'}
        if variant.startswith('resolution'):
            conditions = dict(conditions,unsteered_high=base)
        for condition, rows in conditions.items():
            current = {'high_resolution_unsteered':'native_class_logits_high'} if condition=='unsteered_high' else dict(arms,composed='native_class_logits_combined')
            for name, field in current.items():
                derived = readout(rows,field); score = summary(derived,labels,ids)
                delta = {c:score['accuracy']['0.125/0.875'][c]['rate_all_expected']-baseline['accuracy']['0.125/0.875'][c]['rate_all_expected'] for c in ['all','suc','fail']}
                passed = score['mae'] < baseline['mae'] and delta['all']>=.1-1e-12 and min(delta['suc'],delta['fail'])>0
                point = dict(variant=variant,model=model,protocol=protocol,condition=condition,arm=name,source_field=field,
                    n=70,mae=score['mae'],baseline_mae=baseline['mae'],deltas=delta,descriptive_gate=passed)
                points.append(point); details['/'.join([variant,model,protocol,condition,name])] = score
                if passed: coverage.setdefault((variant,model,name),set()).add(protocol)
    dest = OUT/'analysis'/time.strftime('actual_branch_ablation_round28_29_%Y%m%d_%H%M%S')
    create_json(dest/'points.json',points);create_json(dest/'details.json',details)
    create_json(dest/'manifest.json',dict(sources_sha256=sources,additional_model_forwards=0,
        coverage={ '/'.join(k):sorted(v) for k,v in coverage.items()},
        interpretation='Readouts of actual stored branches only. Descriptive discovery analysis after complete source audit; not independent replication or a new family selection. Negative instruction-blocked branch repeats across k and is not independent evidence. High unsteered is reported once per input; larger resolution is a separate compute/input factor.'))
    print(dest)
    for k,v in sorted(coverage.items()): print(k,len(v),sorted(v))
    for p in points:
        if p['arm']=='high_resolution_unsteered':print(p)


if __name__=='__main__':main()
