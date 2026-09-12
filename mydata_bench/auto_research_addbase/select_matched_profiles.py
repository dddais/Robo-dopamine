"""Round18 fixed robust native-NLL selector; uses discovery labels only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import create_json, OUT as OLD
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT


def native_loss(rows, ids, labels, baseline=False):
    if set(rows) != set(ids) or any(rows[e]['status'] != 'ok' for e in ids):
        raise ValueError('Require exact, complete discovery coverage')
    positive = np.asarray([rows[e]['native_class_logits_positive'] for e in ids], dtype=np.float32)
    if positive.shape != (len(ids), 5) or not np.isfinite(positive).all():
        raise ValueError('Five finite positive logits required')
    if baseline:
        combined = positive
    else:
        negative = np.asarray([rows[e]['native_class_logits_negative'] for e in ids], dtype=np.float32)
        if negative.shape != positive.shape or not np.isfinite(negative).all():
            raise ValueError('Five finite actual negative logits required')
        combined = 2 * positive - negative
    z = combined.astype(float)
    shifted = z - z.max(1, keepdims=True)
    p = np.exp(shifted); p /= p.sum(1, keepdims=True)
    if np.max(np.abs(p - [rows[e]['native_class_probabilities'] for e in ids])) >= 1e-5:
        raise ValueError('Profile is not the specified actual uniform contrast')
    targets = np.asarray([labels[e]['reward'] - 1 for e in ids])
    return np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(ids)), targets]


def robust_scores(deltas, cluster, category, n_groups):
    sums = np.zeros((n_groups, 28, 2)); counts = np.zeros((n_groups, 2))
    for i, (g, c) in enumerate(zip(cluster, category)):
        sums[g, :, c] += deltas[i]
        counts[g, c] += 1
    weights = np.random.default_rng(20260911).multinomial(n_groups, np.ones(n_groups) / n_groups, size=5000)
    weights = weights[(weights @ counts > 0).all(1)]
    boot = np.einsum('bg,glc->blc', weights, sums) / (weights @ counts)[:, None, :]
    means = sums.sum(0) / counts.sum(0)
    upper = np.quantile(boot.max(-1), .9, axis=0)
    return means, upper, boot, len(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    args = parser.parse_args()
    splits = json.loads((OUT / 'splits.json').read_text())
    ids = splits['discovery']
    if len(ids) != 70 or set(ids) & set(splits['validation']):
        raise ValueError('Frozen discovery boundary differs')
    protocols = ['image_text', 'text_image', 'interleaved', 'text_video', 'video_text']
    for protocol in protocols:
        events = OUT / 'experiments' / f'{args.model}_{protocol}_uniform_method_profile' / 'discovery/worker_events.jsonl'
        rows = [json.loads(l) for l in events.read_text().splitlines()]
        if not rows or rows[-1]['event'] != 'complete' or rows[-1]['arguments']['layers'] != list(range(8, 36)):
            raise ValueError('Wait for all five complete 28-layer profiles')
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {e: all_labels[e] for e in ids}; del all_labels
    groups = sorted({labels[e]['video_sha256'] for e in ids})
    cluster = np.asarray([groups.index(labels[e]['video_sha256']) for e in ids])
    category = np.asarray([0 if labels[e]['split'] == 'suc' else 1 for e in ids])
    destination = OUT / 'functional_selections' / f'stage18_{args.model}_robust_v1'
    for protocol in protocols:
        folder = OUT / 'experiments' / f'{args.model}_{protocol}_uniform_method_profile' / 'discovery'
        sources = {}
        def read(path):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            return latest(path)
        baseline = native_loss(read(folder / 'predictions/baseline.jsonl'), ids, labels, True)
        for scope in ['all_frames', 'last_frame']:
            deltas = np.stack([native_loss(read(folder / f'binding_transport_s4_layer{layer}' /
                f'predictions/{scope}_target_8.jsonl'), ids, labels) - baseline for layer in range(8, 36)], axis=1)
            means, upper, boot, draws = robust_scores(deltas, cluster, category, len(groups))
            order = np.lexsort((np.arange(28), means.mean(1), upper))
            profiles, ranking = [], []
            for i in order:
                layer = int(i) + 8
                source = folder / f'binding_transport_s4_layer{layer}' / 'intervention_config.json'
                cfg = json.loads(source.read_text())
                sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                heads = cfg['profiled_heads'][scope]
                profiles.append(dict(layer=layer, robust_worst_class_nll_q90=float(upper[i]),
                    delta_nll_suc=float(means[i, 0]), delta_nll_fail=float(means[i, 1]), heads=heads,
                    ci95_suc=np.quantile(boot[:, i, 0], [.025, .975]).tolist(),
                    ci95_fail=np.quantile(boot[:, i, 1], [.025, .975]).tolist()))
                ranking += [dict(h, score=-float(upper[i]), profile_layer=layer,
                    selection_source='discovery_method_matched_robust_layer_profile') for h in heads]
            if len(ranking) != 224 or len({(h['layer'], h['head']) for h in ranking}) != 224:
                raise ValueError('Frozen candidate budget differs')
            create_json(destination / f'{args.model}_{protocol}' / f'ranking_{scope}.json', dict(
                scope=scope, num_layers=36, num_heads=32, skip_early_layers=8,
                ranking=ranking, layer_profiles=profiles, n=len(ids), example_ids=ids, video_clusters=len(groups),
                draws=draws, seed=20260911, quantile=.9, sources_sha256=dict(sources),
                ranking_score='negative q90 of bootstrap max class delta NLL; empirical balanced NLL then layer tie-break',
                query_kind='answer_format_prefix', validation_ids_used=[],
                supervision='discovery70 native five-class NLL; frozen weights; class-balanced robust selection',
                loss_target='2*uniform_positive-visual_negative, five native reward classes',
                selector_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                interpretation='Robustness objective within discovery; unadjusted bootstrap intervals, no independent validation guarantee'))
        print(destination / f'{args.model}_{protocol}')


if __name__ == '__main__':
    main()
