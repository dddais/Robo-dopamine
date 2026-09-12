"""CPU diagnostic of the existing few-shot layer selection; never changes heads.

Video cluster resampling measures sensitivity within discovery. It is not an
independent validation of selected heads or a correction for adaptive search.
"""
import hashlib
import json
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT


def losses(rows, ids, labels):
    if set(rows) != set(ids) or any(rows[e]['status'] != 'ok' for e in ids):
        raise ValueError('Require all and only discovery observations')
    z = np.asarray([rows[e]['native_class_logits_positive'] for e in ids], dtype=float)
    if z.shape != (len(ids), 5) or not np.isfinite(z).all():
        raise ValueError('Require all five finite native logits')
    target = np.asarray([labels[e]['reward'] - 1 for e in ids])
    return np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1) - z[np.arange(len(ids)), target]


def rank(means):
    worst = means.max(-1)
    balanced = means.mean(-1)
    layer = np.broadcast_to(np.arange(28), worst.shape)
    return np.lexsort((layer, balanced, worst), axis=-1)


def main():
    splits = json.loads((OUT / 'splits.json').read_text())
    ids = splits['discovery']
    if set(ids) & set(splits['validation']):
        raise ValueError('Overlapping splits')
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {e: all_labels[e] for e in ids}
    del all_labels
    groups = sorted({labels[e]['video_sha256'] for e in ids})
    cluster = np.asarray([groups.index(labels[e]['video_sha256']) for e in ids])
    category = np.asarray([0 if labels[e]['split'] == 'suc' else 1 for e in ids])
    counts = np.zeros((len(groups), 2))
    for g, c in zip(cluster, category):
        counts[g, c] += 1
    rng = np.random.default_rng(20260911)
    weights = rng.multinomial(len(groups), np.ones(len(groups)) / len(groups), size=5000)
    weights = weights[(weights @ counts > 0).all(1)]
    loo = np.ones((len(groups), len(groups))) - np.eye(len(groups))
    if not (loo @ counts > 0).all():
        raise ValueError('A leave-one-video-out set has no class coverage')
    records, hashes = [], {}

    def read(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path)

    for model in ['qwen', 'roboreward']:
        for protocol in ['image_text', 'text_image', 'interleaved', 'text_video', 'video_text']:
            root = OUT / 'experiments' / f'{model}_{protocol}_functional_profile' / 'discovery'
            base = losses(read(root / 'predictions/baseline.jsonl'), ids, labels)
            for scope in ['all_frames', 'last_frame']:
                deltas = np.stack([losses(read(root / f'binding_transport_s4_layer{layer}' /
                    f'predictions/{scope}_target_8.jsonl'), ids, labels) - base for layer in range(8, 36)], axis=1)
                sums = np.zeros((len(groups), 28, 2))
                for i, (g, c) in enumerate(zip(cluster, category)):
                    sums[g, :, c] += deltas[i]
                means = sums.sum(0) / counts.sum(0)
                order = rank(means)
                frozen = OUT / 'functional_selections' / f'stage8_{model}_v1' / f'{model}_{protocol}' / f'ranking_{scope}.json'
                hashes[str(frozen)] = hashlib.sha256(frozen.read_bytes()).hexdigest()
                selected = json.loads(frozen.read_text())
                if [int(i) + 8 for i in order] != [r['layer'] for r in selected['layer_profiles']]:
                    raise ValueError('Independent loss/ranking reconstruction disagrees with frozen selector')
                boot = np.einsum('bg,glc->blc', weights, sums) / (weights @ counts)[:, None, :]
                leave = np.einsum('bg,glc->blc', loo, sums) / (loo @ counts)[:, None, :]
                boot_order, loo_order = rank(boot), rank(leave)
                budgets = {}
                for n in [1, 4, 8]:
                    fixed = set(order[:n])
                    overlaps = np.asarray([len(fixed & set(r[:n])) for r in boot_order])
                    loo_overlaps = np.asarray([len(fixed & set(r[:n])) for r in loo_order])
                    budgets[str(n * 8)] = dict(
                        frozen_layers=[int(i) + 8 for i in order[:n]],
                        bootstrap_jaccard_mean=float(np.mean(overlaps / (2 * n - overlaps))),
                        bootstrap_exact_set_rate=float(np.mean(overlaps == n)),
                        leave_one_video_out_exact_set_rate=float(np.mean(loo_overlaps == n)),
                        bootstrap_layer_inclusion={str(i + 8):float(np.mean(np.any(boot_order[:, :n] == i, axis=1))) for i in range(28)})
                layer_rows = []
                for i in order:
                    layer_rows.append(dict(layer=int(i) + 8, delta_nll_suc=float(means[i, 0]),
                        delta_nll_fail=float(means[i, 1]), ci95_suc=np.quantile(boot[:, i, 0], [.025, .975]).tolist(),
                        ci95_fail=np.quantile(boot[:, i, 1], [.025, .975]).tolist(),
                        both_class_improvement_resample_fraction=float(np.mean((boot[:, i, :] < 0).all(1)))))
                records.append(dict(model=model, protocol=protocol, scope=scope, budgets=budgets, layers=layer_rows))
                print(model, protocol, scope, 'exact top8 bootstrap', round(budgets['8']['bootstrap_exact_set_rate'], 3),
                      'top32 Jaccard', round(budgets['32']['bootstrap_jaccard_mean'], 3), flush=True)
    destination = OUT / 'analysis' / time.strftime('profile_stability_%Y%m%d_%H%M%S.json')
    create_json(destination, dict(created_at=time.time(), n=len(ids), video_clusters=len(groups),
        draws=len(weights), seed=20260911, selector_reconstruction='exact for all frozen layer orders',
        sources_sha256=hashes, records=records, validation_labels_used=[],
        interpretation='Sensitivity of single-layer discovery selection, not inference or independent validation. '
                       'Class means keep sample weighting; all same-video instructions resample together. '
                       'Intervals are descriptive and unadjusted; no ranking or candidate modified.'))
    print(destination)


if __name__ == '__main__':
    main()
