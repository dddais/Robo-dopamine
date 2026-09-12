"""Audit round27 actual dynamic-KL controls before strict paired statistics."""
import argparse
import json
import time

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .empirical_profile import sha, verify_sources
from .factorized_kl_tasks import verify_rows, variant
from .compare_native_bias import native, heads, same_input
from .statistics import paired_statistics


POLICY = OUT/'selection_factorized_kl_controls_v1.json'


def verify_and_score(model, protocol):
    sources = {}
    def read(path, rows=False):
        if not path.resolve().is_relative_to(OUT.resolve()):
            raise ValueError('Only this research session is permitted')
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    policy = read(POLICY)
    verify_sources({policy['source']: policy['source_sha256']})
    point, = [p for p in policy['points'] if p['model'] == model and p['protocol'] == protocol]
    budget = policy['budget']; scope = point['scope']; ks = point['ks']
    if budget != .8 or policy['original_bias_variant'] != 'factorized_kl_matched_bias':
        raise ValueError('Changed frozen KL control family')
    splits = read(OUT/'splits.json'); ids = splits['full_cohort']
    if len(ids) != 846 or len(set(ids)) != 846:
        raise ValueError('Require full846')
    root = OUT/'experiments'/f'{model}_{protocol}_{variant(budget)}'/'full_cohort'
    bias_root = OUT/'experiments'/f'{model}_{protocol}_{policy["original_bias_variant"]}'/'full_cohort'
    rank_root = OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'
    ranking = read(rank_root/f'ranking_{scope}.json')['ranking']
    for r, contrast in [(root, True), (bias_root, False)]:
        if read(r/'requested_ids.json') != ids:
            raise ValueError('Require identical complete requested population')
        cfg = read(r/'runtime_config.json')
        expected = dict(model=model, ranking_prefix='ANSWER: ', frozen_ranking_source=str(rank_root),
            contrast_weight=2 if contrast else 0, task_binding_fraction=.5, task_binding_distribution='uniform')
        if contrast:
            expected.update(contrast_kl_budget=budget, contrast_negative_mode='visual_and_task',
                            negative_strength=4, negative_task_strength=4)
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError('Runtime differs from frozen matched control configuration')
        if read(r.parent/'ranking'/f'ranking_{scope}.json')['ranking'] != ranking:
            raise ValueError('Copied ranking differs from frozen stage8')
    base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, budget, baseline=True)
    bias_base = native(read(bias_root/'predictions/baseline.jsonl', True), ids)
    for e in ids:
        if (not same_input(base[e], bias_base[e]) or base[e]['native_class_logits_positive'] != bias_base[e]['native_class_logits_positive']
                or base[e]['attention_diagnostics'] or base[e]['negative_attention_diagnostics']
                or base[e]['task_negative_attention_diagnostics'] or bias_base[e]['attention_diagnostics']
                or bias_base[e]['contrast_weight'] != 0):
            raise ValueError('Actual unsteered same-batch baselines do not match exactly')
    matrices = {}; audit = []; exclusions = {}
    for k in ks:
        arms = {'baseline': base}; unavailable = {}; mismatches = {}
        for kind in ['target', 'wrong_region', 'low_rank', 'original_bias']:
            is_bias = kind == 'original_bias'
            path = (bias_root/'bias_s6/predictions'/f'{scope}_target_{k}.jsonl' if is_bias else
                    root/'binding_transport_s4/predictions'/f'{scope}_{kind}_{k}.jsonl')
            rows = read(path, True)
            if set(rows) != set(ids):
                raise ValueError('Wait for every explicitly requested observation')
            valid = {e: r for e, r in rows.items() if r['status'] == 'ok'}
            unavailable[kind] = sorted(set(ids)-set(valid))
            for e in unavailable[kind]:
                if kind != 'wrong_region' or 'Wrong-region control unavailable: insufficient disjoint cells' not in str(rows[e]):
                    raise ValueError('Unexpected inference failure')
            chosen = ranking[-k:] if kind == 'low_rank' else ranking[:k]
            expected_heads = {(h['layer'], h['head']) for h in chosen}
            if len(expected_heads) != k:
                raise ValueError('Missing or duplicate heads')
            if is_bias:
                native(valid, list(valid))
            else:
                verify_rows(valid, list(valid), budget, expected_heads)
            mismatches[kind] = []
            for e, row in valid.items():
                if row['condition'] != f'{scope}:{"target" if is_bias else kind}:{k}':
                    raise ValueError('Wrong actual condition')
                if not same_input(row, base[e]):
                    mismatches[kind].append(e)
                if is_bias:
                    if row['contrast_weight'] != 0 or heads(row, 'attention_diagnostics') != expected_heads:
                        raise ValueError('Original bias must be one uncontrasted same-head forward')
                    if any(d.get('bias') != 6 or d.get('generated_text_key_bias') != 0 for d in row['attention_diagnostics'].values()):
                        raise ValueError('Reference is not original +/-6')
                else:
                    for field in ['attention_diagnostics', 'negative_attention_diagnostics', 'task_negative_attention_diagnostics']:
                        if any(not d['causal_mask_preserved'] or not d['all_query_rows'] or d['prefill_calls'] < 1 for d in row[field].values()):
                            raise ValueError('Actual mask/query contract differs')
                if kind == 'wrong_region':
                    a = row['token_audit']; target = set(a['target'][scope]); wrong = set(a['wrong'][scope])
                    if not target or target & wrong or len(target) != len(wrong):
                        raise ValueError('Wrong-region must be nonempty, disjoint and equal-size')
            if kind != 'wrong_region' and (unavailable[kind] or mismatches[kind]):
                raise ValueError('Only unavailable-region retry may differ from the original input batch')
            arms[kind] = rows
            audit.append(dict(k=k, control=kind, expected=len(ids), valid=len(valid),
                              input_mismatch_n=len(mismatches[kind]), actual_formula_heads_and_roi_verified=True))
        matrices[k] = arms
        exclusions[k] = dict(unavailable=unavailable, input_mismatch=mismatches)
    verify_sources(sources)
    # Freeze all validity/exact-input exclusions before accessing scoring labels.
    labels_all = json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels = {e: labels_all[e] for e in ids}; del labels_all
    results = {}
    for k, arms in matrices.items():
        omitted = {e for groups in exclusions[k].values() for members in groups.values() for e in members}
        for population in ['validation', 'full_cohort']:
            requested = splits[population]; common = [e for e in requested if e not in omitted]
            results[f'k{k}/{population}'] = dict(expected=len(requested), common_n=len(common),
                excluded_ids=[e for e in requested if e in omitted],
                full_population_metrics={name: summary(arms[name], labels, requested) for name in ['baseline', 'target', 'original_bias']},
                target_minus_original_bias_full_population=paired_statistics(arms['original_bias'], arms['target'], labels, requested),
                strict_common_metrics={name: summary(rows, labels, common) for name, rows in arms.items()},
                strict_paired_changes={f'target_minus_{name}': paired_statistics(rows, arms['target'], labels, common)
                    for name, rows in arms.items() if name != 'target'})
    dest = OUT/'analysis'/time.strftime(f'factorized_kl_controls_{model}_{protocol}_%Y%m%d_%H%M%S.json')
    create_json(dest, dict(status='pass', policy=str(POLICY), point=point, sources_sha256=sources,
        actual_audit=audit, exclusions_before_labels=exclusions, results=results,
        interpretation='Current dynamic KL=.8, actual three branches and same stage8 heads. Original bias comparison uses full population. '
            'ROI/head sensitivity uses only complete exact-padded-input common observations; excluded originals retained. '
            'Adaptive descriptive uncertainty; local controls do not rescue failed family coverage.'))
    print(dest, flush=True)
    for key, value in results.items():
        print(key, 'common', value['common_n'], {m: dict(estimate=x['estimate'], ci95=x['ci95'])
            for m, x in value['target_minus_original_bias_full_population']['metrics'].items()}, flush=True)
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    verify_and_score(args.model, args.protocol)
