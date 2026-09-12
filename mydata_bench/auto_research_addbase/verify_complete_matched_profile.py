"""Audit completed true-method profiles and export selected-head comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .select_matched_profiles import native_loss, robust_scores
from .select_robust_discovery import PROTOCOLS, verify_native
from .select_temporal_discovery import completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    model = parser.parse_args().model
    if not all(completed(OUT / 'experiments' / f'{model}_{p}_uniform_method_profile' / 'discovery')
               for p in PROTOCOLS):
        raise ValueError('All five actual method-matched profiles must complete first')
    ids = json.loads((OUT / 'splits.json').read_text())['discovery']
    if len(ids) != 70:
        raise ValueError('Require original discovery70')
    sources, stored, checks = {}, {}, []
    def read(path, predictions=False):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return latest(path) if predictions else json.loads(path.read_text())
    for protocol in PROTOCOLS:
        folder = OUT / 'experiments' / f'{model}_{protocol}_uniform_method_profile' / 'discovery'
        if set(read(folder / 'requested_ids.json')) != set(ids):
            raise ValueError('Profile population differs')
        baseline = verify_native(read(folder / 'predictions/baseline.jsonl', True), ids, True)
        for scope in ['all_frames', 'last_frame']:
            rank = read(OUT / 'functional_selections' / f'stage18_{model}_robust_v1' /
                        f'{model}_{protocol}' / f'ranking_{scope}.json')
            old = read(OUT / 'functional_selections' / f'stage8_{model}_v1' /
                       f'{model}_{protocol}' / f'ranking_{scope}.json')
            rows_by_layer = {}
            for layer in range(8, 36):
                cfg = read(folder / f'binding_transport_s4_layer{layer}' / 'intervention_config.json')
                expected = {(h['layer'], h['head']) for h in cfg['profiled_heads'][scope]}
                original = next(p for p in old['layer_profiles'] if p['layer'] == layer)['heads']
                if cfg['profiled_heads'][scope] != original or len(expected) != 8:
                    raise ValueError('Actual probe group differs from original frozen pool')
                rows = verify_native(read(folder / f'binding_transport_s4_layer{layer}' /
                    f'predictions/{scope}_target_8.jsonl', True), ids, expected_heads=expected)
                for eid, row in rows.items():
                    if (any(row[k] != baseline[eid][k] for k in ['prompt', 'candidate_token_ids']) or
                        row['token_audit']['input_ids_sha256'] != baseline[eid]['token_audit']['input_ids_sha256']):
                        raise ValueError('Probe input differs from original baseline batch')
                    for field, method, strength in [('attention_diagnostics', 'binding_transport', 4),
                            ('negative_attention_diagnostics', 'mass_transport', -4)]:
                        diag = row[field][str(layer)]
                        if diag['method'] != method or diag['strength'] != strength:
                            raise ValueError('Actual probe does not match registered branch')
                        if field == 'attention_diagnostics' and (diag.get('task_binding_fraction') != .5 or
                                diag.get('task_binding_distribution') != 'uniform'):
                            raise ValueError('Actual task-binding distribution differs')
                rows_by_layer[layer] = rows
            stored[(protocol, scope)] = (baseline, rows_by_layer, rank, old)
    all_labels = json.loads((OLD / 'labels_for_scoring_only.json').read_text())
    labels = {e:all_labels[e] for e in ids}; del all_labels
    groups = sorted({labels[e]['video_sha256'] for e in ids})
    cluster = np.asarray([groups.index(labels[e]['video_sha256']) for e in ids])
    category = np.asarray([0 if labels[e]['split'] == 'suc' else 1 for e in ids])
    for (protocol, scope), (baseline, rows_by_layer, rank, old) in stored.items():
        base = native_loss(baseline, ids, labels, True)
        delta = np.stack([native_loss(rows_by_layer[layer], ids, labels)-base for layer in range(8, 36)], axis=1)
        means, upper, _, draws = robust_scores(delta, cluster, category, len(groups))
        order = np.lexsort((np.arange(28), means.mean(1), upper))
        if [int(i)+8 for i in order] != [p['layer'] for p in rank['layer_profiles']]:
            raise ValueError('Complete actual logits do not reconstruct the frozen robust order')
        for p in rank['layer_profiles']:
            i = p['layer']-8
            if (p['delta_nll_suc'] != means[i, 0] or p['delta_nll_fail'] != means[i, 1] or
                    p['robust_worst_class_nll_q90'] != upper[i]):
                raise ValueError('Stored profile statistics differ from actual logits')
        overlap = {}
        for k in [8, 32, 64]:
            a = {(h['layer'], h['head']) for h in rank['ranking'][:k]}
            b = {(h['layer'], h['head']) for h in old['ranking'][:k]}
            overlap[str(k)] = dict(shared=len(a & b), fraction=len(a & b)/k,
                                  jaccard=len(a & b)/len(a | b))
        checks.append(dict(protocol=protocol, scope=scope, actual_intervention_rows=1960,
            both_branches_and_frozen_heads_verified=True, robust_ranking_reconstructed=True,
            top8=rank['ranking'][:8], top8_layer=rank['layer_profiles'][0]['layer'],
            old_top8_layer=old['layer_profiles'][0]['layer'], top8_profile=rank['layer_profiles'][0],
            original_stage8_head_overlap=overlap, bootstrap_draws=draws))
    path = OUT / 'audit' / time.strftime(f'complete_matched_profile_{model}_%Y%m%d_%H%M%S.json')
    create_json(path, dict(status='pass', model=model, checks=checks, sources_sha256=sources,
        discovery_labels_used=ids, validation_labels_used=[], actual_intervention_rows_verified=19600,
        actual_unsteered_baseline_rows_verified=350, video_clusters=len(groups),
        interpretation='Complete actual positive and negative probes, fixed groups and inputs verified; '
            'frozen ranking statistics reconstructed. Head-index overlap is descriptive; no rankings or inference changed. '
            'This audit does not guarantee efficacy of multiple layer groups composed together.'))
    print(path)
    for c in checks:
        print(c['protocol'], c['scope'], 'layer', c['old_top8_layer'], '→', c['top8_layer'],
              'top32 overlap', c['original_stage8_head_overlap']['32']['fraction'])


if __name__ == '__main__':
    main()
