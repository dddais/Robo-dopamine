"""Round33 controls fixed for every selected input and neighboring k."""
import json
import time

from mydata_bench.addbase_eval.prepare import OUT as OLD, create_json
from mydata_bench.addbase_eval.run import latest
from mydata_bench.addbase_eval.score import summary
from .prepare import OUT
from .head_delta_worker import BASE, sha, verify_sources
from .head_delta_eval_tasks import addition, MODELS
from .head_delta_eval_tasks import command, folder, verify_rows
from .verify_head_delta_full import SELECTION, selected
from .compare_native_bias import native, heads, same_input
from .statistics import paired_statistics


POLICY = OUT/'selection_learned_head_delta_reft_v1.json'
BIAS_VARIANT = 'learned_head_delta_reft_matched_bias'
B0_VARIANT = 'learned_head_delta_reft_B0_v1'


def bias_folder(model, protocol):
    return OUT/'experiments'/f'{model}_{protocol}_{BIAS_VARIANT}'/'full_cohort'


def b0_folder(model, protocol):
    return OUT/'experiments'/f'{model}_{protocol}_{B0_VARIANT}'/'full_cohort'


def register():
    record=selected();spec=json.loads(POLICY.read_text());verify_sources(spec['sources_sha256'])
    decision_path=BASE/'coverage.json';decision=json.loads(decision_path.read_text());verify_sources(decision['sources_sha256'])
    if not decision['shared_coverage']:raise ValueError('Conditional controls require all full targets to pass shared coverage')
    jobs=[]
    for p in record['selected']:
        m=p['model'];protocol=p['protocol'];name=f'controls_{m}_{protocol}_head_delta'
        jobs.append(dict(name=name,gpu=MODELS.index(m),min_free_mb=23000,
            depends_on=[f'validate_{m}_{protocol}_head_delta'],
            command=command(m,[protocol],'full_cohort',p['ks'],[p['scope']])+['--controls','wrong_region','low_rank']))
        jobs.append(dict(name=name+'_B0',gpu=MODELS.index(m),min_free_mb=23000,depends_on=[name],
            command=command(m,[protocol],'full_cohort',p['ks'],[p['scope']])+['--adapter-mode','B0']))
        original=['/home/dais/miniconda3/envs/robo-dopamine/bin/python','-B','-m',
            'mydata_bench.auto_research_addbase.worker','--model',m,'--protocols',protocol,
            '--variant',BIAS_VARIANT,'--population','full_cohort','--methods','bias','--strengths','6',
            '--ks',*map(str,p['ks']),'--scopes',p['scope'],'--batch-size','8','--ranking-prefix','ANSWER: ',
            '--contrast-weight','0','--frozen-ranking-root',str(OUT/'functional_selections'/f'stage8_{m}_v1')]
        jobs.append(dict(name=name+'_original_bias',gpu=MODELS.index(m),min_free_mb=23000,depends_on=[name+'_B0'],command=original))
    return addition('stage33_head_delta_controls.json',jobs,
        {str(POLICY):sha(POLICY),str(SELECTION):sha(SELECTION),str(decision_path):sha(decision_path)})


def ready(p, ids):
    root = folder(p['model'], p['protocol'], 'full_cohort'); bias = bias_folder(p['model'], p['protocol'])
    b0 = b0_folder(p['model'],p['protocol'])
    paths = [bias/'predictions/baseline.jsonl',b0/'predictions/baseline.jsonl']
    for k in p['ks']:
        paths += [root/'learned_head_delta_reft_s6/predictions'/f"{p['scope']}_{arm}_{k}.jsonl" for arm in ['target', 'wrong_region', 'low_rank']]
        paths += [bias/'bias_s6/predictions'/f"{p['scope']}_target_{k}.jsonl",b0/'learned_head_delta_reft_s6/predictions'/f"{p['scope']}_target_{k}.jsonl"]
    return all(path.exists() and set(latest(path)) == set(ids) for path in paths)


def verify_and_score(model, protocol):
    record = selected(); point, = [p for p in record['selected'] if p['model'] == model and p['protocol'] == protocol]
    sources = {str(POLICY): sha(POLICY), str(SELECTION): sha(SELECTION)}
    def read(path, rows=False):
        if not path.resolve().is_relative_to(OUT.resolve()): raise ValueError('Wrong research session')
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    splits = read(OUT/'splits.json'); ids = splits['full_cohort']; scope = point['scope']
    root = folder(model, protocol, 'full_cohort'); bias = bias_folder(model, protocol)
    b0 = b0_folder(model,protocol)
    gate_file = BASE/'training'/model/'final_adapter.json'; gates = read(gate_file)
    rank_root = OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'
    ranking = read(rank_root/f'ranking_{scope}.json')['ranking']
    for path in [root, bias, b0]:
        if read(path/'requested_ids.json') != ids: raise ValueError('Changed complete population')
        cfg = read(path/'runtime_config.json')
        expected = dict(model=model, batch_size=8, max_pixels=50176, contrast_weight=0,
            ranking_prefix='ANSWER: ', frozen_ranking_source=str(rank_root))
        if any(cfg.get(k) != v for k, v in expected.items()): raise ValueError('Changed matched configuration')
        if read(path.parent/'ranking'/f'ranking_{scope}.json')['ranking'] != ranking:
            raise ValueError('Changed original stage8 heads')
    from .head_delta_inference import inference_config
    if read(root/'runtime_config.json')!=inference_config(model,protocol) or read(b0/'runtime_config.json')!=inference_config(model,protocol,'B0'):
        raise ValueError('Actual same-adapter/B0 configuration changed')
    base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, gates, gate_file, baseline=True)
    b0_base = verify_rows(read(b0/'predictions/baseline.jsonl',True),ids,gates,gate_file,baseline=True,mode='B0')
    bias_base = native(read(bias/'predictions/baseline.jsonl', True), ids)
    if any(not same_input(base[e], bias_base[e]) or base[e]['native_class_logits_positive'] != bias_base[e]['native_class_logits_positive']
           or bias_base[e]['attention_diagnostics'] for e in ids):
        raise ValueError('Original-bias and learned baseline must replay exactly')
    if any(not same_input(base[e],b0_base[e]) or base[e]['native_class_logits_positive']!=b0_base[e]['native_class_logits_positive'] for e in ids):
        raise ValueError('B0 baseline does not exactly replay')
    matrices = {}; exclusions = {}; audit = []
    for k in point['ks']:
        arms = {'baseline': base}; invalid = {}; mismatch = {}
        for arm in ['target', 'wrong_region', 'low_rank', 'original_bias', 'B0']:
            is_bias = arm == 'original_bias'
            path = bias/'bias_s6/predictions'/f'{scope}_target_{k}.jsonl' if is_bias else root/'learned_head_delta_reft_s6/predictions'/f'{scope}_{arm}_{k}.jsonl'
            if arm=='B0': path=b0/'learned_head_delta_reft_s6/predictions'/f'{scope}_target_{k}.jsonl'
            rows = read(path, True)
            if set(rows) != set(ids): raise ValueError('Incomplete full control matrix')
            valid = {e: r for e, r in rows.items() if r['status'] == 'ok'}
            invalid[arm] = sorted(set(ids)-set(valid))
            if any(arm != 'wrong_region' or 'Wrong-region control unavailable: insufficient disjoint cells' not in str(rows[e]) for e in invalid[arm]):
                raise ValueError('Unexpected invalid control')
            expected = {(h['layer'], h['head']) for h in (ranking[-k:] if arm == 'low_rank' else ranking[:k])}
            if len(expected) != k: raise ValueError('Duplicate/missing control heads')
            if is_bias: native(valid, list(valid))
            else: verify_rows(valid, list(valid), gates, gate_file, expected,mode='B0' if arm=='B0' else 'learned')
            mismatch[arm] = []
            for e, row in valid.items():
                if row['condition'] != f'{scope}:{"target" if is_bias or arm=="B0" else arm}:{k}': raise ValueError('Changed control condition')
                if not same_input(row, base[e]): mismatch[arm].append(e)
                if is_bias and (row['contrast_weight'] != 0 or heads(row, 'attention_diagnostics') != expected
                        or any(d.get('bias') != 6 or d.get('generated_text_key_bias') != 0 for d in row['attention_diagnostics'].values())):
                    raise ValueError('Require same-head original uncontrasted +/-6')
                if arm == 'wrong_region':
                    a = row['token_audit']; target = set(a['target'][scope]); wrong = set(a['wrong'][scope])
                    if not target or target & wrong or len(target) != len(wrong):
                        raise ValueError('Wrong-region is not disjoint and equal-size')
            if arm != 'wrong_region' and (invalid[arm] or mismatch[arm]):
                raise ValueError('Only unavailable wrong-region retry can change input padding')
            arms[arm] = rows
            audit.append(dict(k=k, arm=arm, valid=len(valid), input_mismatch_n=len(mismatch[arm])))
        for prior_variant in ['learned_head_gates_v1','learned_head_gates_k_coverage_v1']:
            reference=OUT/'experiments'/f'{model}_{protocol}_{prior_variant}'/'full_cohort/learned_head_gate_bias_s6/predictions'/f'{scope}_target_{k}.jsonl'
            if reference.exists():
                old=read(reference,True)
                if set(old)!=set(ids) or any(not same_input(old[e],arms['B0'][e]) or old[e]['native_class_logits_positive']!=arms['B0'][e]['native_class_logits_positive'] for e in ids):
                    raise ValueError('Actual B0 reference does not exactly replay existing round32 fixed gates')
        matrices[k] = arms; exclusions[k] = dict(invalid=invalid, mismatch=mismatch)
    verify_sources(sources)
    # All exact matching/exclusion choices above are determined without labels.
    all_labels = json.loads((OLD/'labels_for_scoring_only.json').read_text())
    labels = {e: all_labels[e] for e in ids}; del all_labels
    results = {}
    for k, arms in matrices.items():
        omitted = {e for groups in exclusions[k].values() for members in groups.values() for e in members}
        for population in ['validation', 'full_cohort', 'old_holdout']:
            requested = splits[population]; common = [e for e in requested if e not in omitted]
            results[f'k{k}/{population}'] = dict(expected=len(requested), strict_common_n=len(common),
                excluded_ids=[e for e in requested if e in omitted],
                full_population_metrics={arm: summary(arms[arm], labels, requested) for arm in ['baseline', 'target', 'original_bias', 'B0']},
                target_minus_original_bias_full_population=paired_statistics(arms['original_bias'], arms['target'], labels, requested),
                target_minus_B0_full_population=paired_statistics(arms['B0'],arms['target'],labels,requested),
                strict_common_metrics={arm: summary(rows, labels, common) for arm, rows in arms.items()},
                strict_paired_changes={f'target_minus_{arm}': paired_statistics(rows, arms['target'], labels, common)
                    for arm, rows in arms.items() if arm != 'target'})
    prefix = 'head_delta_controls'
    dest = OUT/'analysis'/time.strftime(f'{prefix}_{model}_{protocol}_%Y%m%d_%H%M%S.json')
    create_json(dest, dict(status='pass', point=point, sources_sha256=sources, actual_audit=audit,
        exclusions_before_labels=exclusions, results=results,
        interpretation='Adaptive dataset-internal comparison. Low-rank changes both head identity and training exposure; '
            'it is not an isolated causal estimate of ranking. Full original-bias/B0 comparisons and strict common ROI comparisons are distinct. B0 retains prior trained gates and removes the new matrix increment; extra optimization and capacity are not isolated.'))
    print(dest, flush=True)
    return dest
