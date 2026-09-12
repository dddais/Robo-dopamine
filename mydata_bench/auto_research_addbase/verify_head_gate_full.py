"""Verify every frozen round32 full-cohort row before any efficacy scoring."""
import argparse
import json
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import latest
from .prepare import OUT
from .head_gate_worker import BASE, POLICY, sha, verify_sources
from .head_gate_eval_tasks import VARIANT, NEIGHBORHOODS, verify_rows, folder
from .compare_native_bias import same_input


SELECTION = BASE/'selection_complete_discovery_v1.json'


def selected():
    r = json.loads(SELECTION.read_text())
    verify_sources(r['sources_sha256'])
    eligible = min(r['input_counts'].values()) >= 3
    if r['family_eligible_for_full'] != eligible or r['selected'] != (r['proposed_input_candidates'] if eligible else []):
        raise ValueError('Frozen shared input coverage selection changed')
    for p in r['selected']:
        if p['ks'] != NEIGHBORHOODS[p['center_k']] or p['variant'] != VARIANT:
            raise ValueError('Frozen original neighborhood differs')
    return r


def ready(p, ids):
    root = folder(p['model'], p['protocol'], 'full_cohort')
    files = [root/'predictions/baseline.jsonl'] + [root/'learned_head_gate_bias_s6/predictions'/f"{p['scope']}_target_{k}.jsonl" for k in p['ks']]
    return all(path.exists() and set(latest(path)) == set(ids) for path in files)


def verify(model, protocol):
    selection = selected()
    p, = [p for p in selection['selected'] if p['model'] == model and p['protocol'] == protocol]
    sources = {str(SELECTION): sha(SELECTION), str(POLICY): sha(POLICY)}
    def read(path, rows=False):
        sources[str(path)] = sha(path)
        return latest(path) if rows else json.loads(path.read_text())
    ids = read(OUT/'splits.json')['full_cohort']
    root = folder(model, protocol, 'full_cohort')
    if len(ids) != 846 or len(set(ids)) != 846 or read(root/'requested_ids.json') != ids:
        raise ValueError('Require complete original full846 population')
    gate_file = BASE/'training'/model/'final_gates.json'; gate = read(gate_file)
    training = read(BASE/'fixed_training_audit_v1.json')
    if training['status'] != 'pass' or training['labels_read']:
        raise ValueError('Fixed training audit must pass first')
    verify_sources(training['sources_sha256'])
    cfg = read(root/'runtime_config.json')
    rank_root = OUT/'functional_selections'/f'stage8_{model}_v1'/f'{model}_{protocol}'
    expected = dict(model=model, protocol=protocol, learned_gate_file=str(gate_file),
        learned_gate_sha256=sha(gate_file), contrast_weight=0, contrast_negative_mode='learned_head_gates',
        batch_size=8, max_pixels=50176, method='learned_head_gate_bias', bias=6,
        ranking_prefix='ANSWER: ', frozen_ranking_source=str(rank_root))
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual full inference configuration differs')
    ranks = read(rank_root/f"ranking_{p['scope']}.json")['ranking']
    if read(root.parent/'ranking'/f"ranking_{p['scope']}.json")['ranking'] != ranks:
        raise ValueError('Actual head source changed')
    base = verify_rows(read(root/'predictions/baseline.jsonl', True), ids, gate, gate_file, baseline=True)
    checks = []
    for k in p['ks']:
        h = {(r['layer'], r['head']) for r in ranks[:k]}
        if len(h) != k: raise ValueError('Duplicate/missing heads')
        rows = verify_rows(read(root/'learned_head_gate_bias_s6/predictions'/f"{p['scope']}_target_{k}.jsonl", True), ids, gate, gate_file, h)
        if any(not same_input(row, base[e]) or row['condition'] != f"{p['scope']}:target:{k}" for e, row in rows.items()):
            raise ValueError('Actual full same-batch input/condition differs')
        checks.append(dict(k=k, scope=p['scope'], n=len(rows), actual_fixed_gates_heads_inputs_and_native_output_verified=True))
    verify_sources(sources)
    path = OUT/'audit'/time.strftime(f'head_gates_full_{model}_{protocol}_%Y%m%d_%H%M%S.json')
    create_json(path, dict(status='pass', labels_read=False, model=model, protocol=protocol,
        frozen_point=p, checks=checks, sources_sha256=sources,
        interpretation='Complete fixed-neighborhood full846 implementation audit. No efficacy claim.'))
    print(path, flush=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['qwen', 'roboreward'])
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args(); verify(args.model, args.protocol)
