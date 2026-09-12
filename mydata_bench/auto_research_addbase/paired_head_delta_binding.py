"""Post-inference same-video separation, stratified by accepted ROI equality."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .paired_mask_analysis import aggregate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    args = p.parse_args()
    sources = {}
    geometry_file = OUT/'analysis/roi_quantization_v1.json'
    geometry = json.loads(geometry_file.read_text()); sources[str(geometry_file)] = sha(geometry_file)
    inputs_file = OLD/'inputs.json'; input_sha = sha(inputs_file)
    video = {r['example_id']: r['video_sha256'] for r in json.loads(inputs_file.read_text())}
    results = {}
    for checkpoint in args.checkpoints:
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_relative_to(OUT.resolve()): raise ValueError('Only this research session')
        detail = checkpoint/'details.json'; sources[str(detail)] = sha(detail)
        for key, record in json.loads(detail.read_text()).items():
            experiment, method, condition, field = key.split('/')
            if not experiment.endswith('learned_head_delta_reft_v1') or '_target_' not in condition or field != 'progress':
                raise ValueError('Only audited round33 primary target checkpoints')
            scope = 'last_frame' if condition.startswith('last_frame') else 'all_frames'
            protocol = 'video_text' if any(s in experiment for s in ['text_video','video_text']) else 'image_text'
            identical = {(r['suc_id'],r['fail_id']):r['identical'] for r in geometry[protocol]['scopes'][scope]['pair_details']}
            before = {(r['suc_id'],r['fail_id']):r for r in record['baseline']['pairwise']['pairs']}
            after = {(r['suc_id'],r['fail_id']):r for r in record['intervention']['pairwise']['pairs']}
            if set(before) != set(after) or not set(before) <= set(identical):
                raise ValueError('Require identical complete pair membership and accepted geometry coverage')
            entry = {}
            for name, equality in [('identical_roi', True),('different_roi',False)]:
                members = [pair for pair in before if identical[pair] is equality]
                if not members: continue
                grouped = defaultdict(list)
                for pair in members:
                    a,b = pair
                    if video[a] != video[b]: raise ValueError('Pair must use the same video')
                    grouped[video[a]].append(after[pair]['continuous_delta']-before[pair]['continuous_delta'])
                totals = np.asarray([sum(values) for values in grouped.values()]); n = np.asarray([len(values) for values in grouped.values()])
                rng = np.random.default_rng(20260912)
                index = rng.integers(0,len(totals),size=(5000,len(totals)))
                boot = totals[index].sum(1)/n[index].sum(1)
                a = aggregate([before[pair] for pair in members]); b = aggregate([after[pair] for pair in members])
                estimate = float(totals.sum()/n.sum())
                if abs(estimate-(b['continuous_mean_delta']-a['continuous_mean_delta'])) > 1e-12:
                    raise ValueError('Paired change does not reconstruct')
                entry[name] = dict(n_pairs=len(members), video_clusters=len(grouped), baseline=a, intervention=b,
                    delta_mean_separation=estimate, delta_mean_separation_ci95=np.quantile(boot,[.025,.975]).tolist(),
                    delta_strict_positive_fraction=b['strict_positive_fraction']-a['strict_positive_fraction'])
            results[f'{checkpoint.name}/{key}'] = entry
    verify_sources(sources)
    if sha(inputs_file) != input_sha: raise ValueError('Input/video metadata source changed')
    dest = OUT/'analysis'/time.strftime('head_delta_paired_roi_%Y%m%d_%H%M%S.json')
    create_json(dest,dict(sources_sha256=sources,input_metadata_source=str(inputs_file),input_metadata_sha256=input_sha,
        results=results,draws=5000,seed=20260912,resampling_unit='video_sha256 with all its available instruction pairs',
        interpretation='Descriptive post-inference stratification. ROI equality never changes predictions. '
            'Equal ROI does not imply equal activations; task composition/difficulty can differ. '
            'Uncorrected adaptive intervals, not independent causal proof of grounding or instruction binding.'))
    print(dest,flush=True)
    print('Complete paired ROI conditions:',len(results),flush=True)


if __name__ == '__main__': main()
