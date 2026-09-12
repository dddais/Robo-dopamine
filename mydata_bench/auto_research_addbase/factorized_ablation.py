"""Read-only, explicitly derived branch ablations for completed round17 discovery.

These ablations are not additional selected candidates or new GPU forwards.
ROI strata are observational and never enter inference.
"""
import argparse
from collections import Counter
import hashlib
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT


def relative_roi(row, scope):
    audit = row['token_audit']
    lookup = {p: i for i, p in enumerate(audit['visual'])}
    return len(lookup), tuple(sorted(lookup[p] for p in audit['target'][scope]))


def derive(row, branch):
    positive = np.asarray(row['native_class_logits_positive'], dtype=np.float32)
    visual = np.asarray(row['native_class_logits_negative'], dtype=np.float32)
    task = np.asarray(row['native_class_logits_negative_task'], dtype=np.float32)
    z = dict(positive_only=positive, visual_only=2*positive-visual,
             task_only=2*positive-task, factorized=2*positive-.5*(visual+task))[branch].astype(float)
    p = np.exp(z-z.max()); p /= p.sum()
    reward = int(p.argmax()) + 1
    if branch == 'factorized' and (reward != row['reward'] or np.max(np.abs(p-row['native_class_probabilities'])) >= 1e-5):
        raise ValueError('Actual three-branch composition does not reconstruct')
    return dict(row, progress=(reward-1)/4, reward=reward, derived_only=branch!='factorized')


def pair_gain_ci(pairs, values, labels):
    if not pairs:
        return None
    groups = sorted({labels[p['suc_id']]['video_sha256'] for p in pairs})
    totals = np.zeros(len(groups)); counts = np.zeros(len(groups))
    for pair, value in zip(pairs, values):
        g = groups.index(labels[pair['suc_id']]['video_sha256'])
        totals[g] += value; counts[g] += 1
    index = np.random.default_rng(20260911).integers(0, len(groups), size=(5000,len(groups)))
    boot = totals[index].sum(1)/counts[index].sum(1)
    return dict(video_groups=len(groups), mean=float(totals.sum()/counts.sum()), ci95=np.quantile(boot,[.025,.975]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen','roboreward'], required=True)
    args = parser.parse_args()
    selection_path = OUT / f'selection_factorized_{args.model}_full_v1.json'
    selection = json.loads(selection_path.read_text())  # Selection must already be frozen.
    selected = {(p['protocol'],p['scope'],p['center_k']) for p in selection['selected']}
    ids = json.loads((OUT/'splits.json').read_text())['discovery']
    all_labels = json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels = {e:all_labels[e] for e in ids}; del all_labels
    sources = {str(selection_path):hashlib.sha256(selection_path.read_bytes()).hexdigest()}
    records, strata = [], []
    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = latest(path)
        if set(rows)!=set(ids) or any(r['status']!='ok' for r in rows.values()):
            raise ValueError('Require complete discovery')
        return rows
    for protocol in ['image_text','text_image','interleaved','text_video','video_text']:
        folder = OUT/'experiments'/f'{args.model}_{protocol}_uniform_factorized_evidence_a1'/'discovery'
        old_folder = OUT/'experiments'/f'{args.model}_{protocol}_uniform_binding_evidence_a1'/'discovery'
        baseline = read(folder/'predictions/baseline.jsonl')
        for path in sorted(folder.glob('binding_transport_s4/predictions/*.jsonl')):
            rows = read(path)
            old = read(old_folder/'binding_transport_s4/predictions'/path.name)
            scope,k = path.stem.split('_target_'); k = int(k)
            exact = sum(all(rows[e][f]==old[e][f] for f in ['native_class_logits_positive','native_class_logits_negative','prompt'])
                        and rows[e]['token_audit']['input_ids_sha256']==old[e]['token_audit']['input_ids_sha256'] for e in ids)
            arms = dict(baseline=baseline)
            arms.update({arm:{e:derive(row,arm) for e,row in rows.items()} for arm in ['positive_only','visual_only','task_only','factorized']})
            scores = {arm:summary(value,labels,ids) for arm,value in arms.items()}
            records.append(dict(protocol=protocol,scope=scope,k=k,n=len(ids),original_uniform_positive_visual_negative_input_exact_rows=exact,
                metrics={arm:dict(mae=s['mae'],accuracy=s['accuracy'],prediction_distributions=s['prediction_distributions'],pairwise=s['pairwise']) for arm,s in scores.items()}))
            if (protocol,scope,k) not in selected:
                continue
            pairs = scores['baseline']['pairwise']['pairs']
            for identical in [True,False]:
                group = [p for p in pairs if (relative_roi(rows[p['suc_id']],scope)==relative_roi(rows[p['fail_id']],scope)) is identical]
                margins = {arm:[arms[arm][p['suc_id']]['reward']-arms[arm][p['fail_id']]['reward'] for p in group] for arm in arms}
                entry = dict(protocol=protocol,scope=scope,k=k,identical_actual_relative_roi=identical,n_pairs=len(group),arms={})
                for arm,values in margins.items():
                    hist = Counter('<0' if v<0 else str(v) for v in values)
                    counts = {c:hist[c] for c in ['<0','0','1','2','3','4']}
                    if sum(counts.values())!=len(group):
                        raise ValueError('Pair histogram does not sum to actual pair count')
                    entry['arms'][arm] = dict(mean_reward_margin=float(np.mean(values)) if values else None,counts=counts)
                entry['factorized_minus_visual_only_margin'] = pair_gain_ci(group,
                    [a-b for a,b in zip(margins['factorized'],margins['visual_only'])],labels)
                strata.append(entry)
    if len(records)!=30:
        raise ValueError('The complete 30-condition matrix is required')
    output = OUT/'analysis'/time.strftime(f'factorized_ablation_{args.model}_%Y%m%d_%H%M%S.json')
    create_json(output,dict(created_at=time.time(),model=args.model,sources_sha256=sources,records=records,selected_pair_strata=strata,
        derived_arms=['positive_only','visual_only','task_only'],candidate_selection_modified=False,
        interpretation='Ablations derived from the same three actual native branches, no additional GPU forwards. '
                       'Original uniform branch exact-match counts are reported; matched derived visual-only is the comparator. '
                       'Selected-input ROI strata are observational, bootstrap intervals unadjusted, discovery repeatedly explored.'))
    print(output)
    print('Original positive/visual-negative/input exact rows:',sum(r['original_uniform_positive_visual_negative_input_exact_rows'] for r in records),'/ 2100')
    for row in strata:
        print(row['protocol'],row['scope'],row['k'],'identical_roi',row['identical_actual_relative_roi'],row['n_pairs'],row['factorized_minus_visual_only_margin'])


if __name__=='__main__':
    main()
