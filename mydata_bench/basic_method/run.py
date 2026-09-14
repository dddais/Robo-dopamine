"""Append-only execution with whole-example baseline fallback before steering."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import yaml

from mydata_bench.io import artifact_fingerprint
from mydata_bench.top_eval.versioning import is_official_sole, protocol_metadata
from .common import (append, latest, create_json, fingerprint, conditions, validate_config,
                     validate_inputs, validate_media, implementation_identity)
from .grounding import eligibility, sample_identity, ControlUnavailable


def cache_rows(path, samples, run_id, condition=None):
    rows = latest(path)
    requested = {s['example_id']: s for s in samples}
    if set(rows) - set(requested):
        raise ValueError(f'Cache contains IDs outside requested population: {path}')
    for eid, row in rows.items():
        if (row.get('run_id') != run_id or row.get('sample_id') != sample_identity(requested[eid])
                or condition is not None and row.get('condition') != condition):
            raise ValueError(f'Incompatible cache identity/condition: {path}, {eid}')
    return rows


def fallback_row(sample, condition, baseline, baseline_path, check, run_id):
    if baseline.get('example_id') != sample['example_id'] or baseline.get('condition') != 'baseline':
        raise ValueError('Fallback requires the matching baseline example')
    if baseline.get('sample_id') != sample_identity(sample) or baseline.get('run_id') != run_id:
        raise ValueError('Fallback baseline belongs to different input/configuration')
    row = copy.deepcopy(baseline)
    row.update(condition=condition, sas_applied=False, baseline_fallback=True,
               fallback_reason=check['reason'], grounding_check=check,
               baseline_source={'path': str(baseline_path), 'example_id': sample['example_id'],
                                'row_sha256': fingerprint(baseline), 'run_id': run_id},
               positive_bias=0.0, negative_bias=0.0)
    # A parse error in the source baseline stays a parse error in the effective output.
    return row


def check_sample(runtime, sample, scope, baseline=None):
    native = runtime.cfg['model'] in ('qwen', 'roboreward') and runtime.cfg['protocol'] == 'official'
    indices = None
    if native:
        if baseline is not None:
            indices = baseline['token_audit']['native_indices']
        else:
            _, mapping, _, _ = runtime.prepare(sample)
            indices = mapping['native_indices']
    check = eligibility(sample, runtime.cfg, scope, indices)
    if check['eligible'] and is_official_sole(runtime.cfg) and scope == 'last_frame':
        check.update(runtime.geometry_preflight(sample, scope))
    return check


def rollout(runtime, sample, condition, ranking, output):
    official = is_official_sole(runtime.cfg)
    previous, trace = '0', []
    for step in range(1, 8) if official else [None]:
        row = runtime.predict(sample, condition, ranking, step, previous)
        if official:
            append(output / 'steps' / (condition.replace(':', '_') + '.jsonl'), [row])
            trace.append({'step': step, 'status': row['status'], 'progress': row.get('progress'),
                          'previous_percentage_text': previous,
                          'prompt_sha256': row.get('token_audit', {}).get('prompt_sha256')})
        if row['status'] != 'ok':
            break
        if official:
            previous = row['percentage_text']
    if official:
        row.update(step_count=len(trace), rollout=trace,
                   progress_curve=[0] + [r['progress'] for r in trace],
                   previous_values_are_model_predictions=True)
        if len(trace) != 7:
            row.update(progress=None, incomplete_rollout=True)
    return row


def predict_condition(runtime, samples, condition, rankings, output, run_id):
    output = Path(output)
    path = output / 'predictions' / (condition.replace(':', '_') + '.jsonl')
    done = cache_rows(path, samples, run_id, condition)
    baseline_path = output / 'predictions/baseline.jsonl'
    base = cache_rows(baseline_path, samples, run_id, 'baseline') if condition != 'baseline' else {}
    if condition != 'baseline' and set(base) != {s['example_id'] for s in samples}:
        raise ValueError('Full matching baseline must finish before steering/fallback')
    for sample in samples:
        eid = sample['example_id']
        if eid in done:
            continue
        check, ranking = None, None
        if condition != 'baseline':
            scope, kind, _ = condition.split(':')
            check = check_sample(runtime, sample, scope, base[eid])
            if not check['eligible']:
                row = fallback_row(sample, condition, base[eid], baseline_path, check, run_id)
                append(path, [row])
                done[eid] = row
                continue
            ranking = rankings[scope]
            if kind == 'wrong_region':
                try:
                    # Check all recursive steps before any steered prediction occurs.
                    runtime.control_preflight(sample, scope)
                except ControlUnavailable as exc:
                    row = {'example_id': eid, 'condition': condition, 'status': 'control_unavailable',
                           'progress': None, 'error': str(exc), 'sas_applied': False,
                           'baseline_fallback': False, 'grounding_check': check,
                           'run_id': run_id, 'sample_id': sample_identity(sample), **protocol_metadata(runtime.cfg)}
                    append(path, [row])
                    done[eid] = row
                    continue
        # Model/runtime/file errors stop this job. They are never disguised as missing grounding.
        row = rollout(runtime, sample, condition, ranking, output)
        if condition != 'baseline' and 'native_indices' in row['token_audit']:
            if row['token_audit']['native_indices'] != base[eid]['token_audit']['native_indices']:
                raise ValueError('Processor sampling changed between baseline and SAS')
        row.update(run_id=run_id, sample_id=sample_identity(sample),
                   sas_applied=condition != 'baseline', baseline_fallback=False,
                   fallback_reason=None, grounding_check=check,
                   positive_bias=runtime.cfg['bias'] if condition != 'baseline' else 0.0,
                   negative_bias=-runtime.cfg['bias'] if condition != 'baseline' else 0.0,
                   completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        append(path, [row])
        done[eid] = row
        print(f'{runtime.cfg["model"]}/{runtime.cfg["protocol"]} {condition} {len(done)}/{len(samples)} {row["status"]}', flush=True)
    return done


def rank(runtime, samples, baseline, output, run_id):
    rankings = {}
    official = is_official_sole(runtime.cfg)
    for scope in runtime.cfg['scopes']:
        checks = {s['example_id']: check_sample(runtime, s, scope, baseline.get(s['example_id'])) for s in samples}
        eligible = [s for s in samples if checks[s['example_id']]['eligible']]
        if official:
            for sample in eligible[:]:
                b = baseline.get(sample['example_id'], {})
                if b.get('status') != 'ok' or b.get('step_count') != 7:
                    checks[sample['example_id']] = {**checks[sample['example_id']], 'eligible': False,
                                                   'reason': 'invalid_ranking_baseline_rollout'}
                    eligible.remove(sample)
        create_json(output / f'ranking_eligibility_{scope}.json', checks)
        if len(eligible) < 2:
            raise ValueError(f'Insufficient eligible independent ranking examples for {scope}: {len(eligible)}')
        path = output / f'ranking_observations_{scope}.jsonl'
        observations = cache_rows(path, eligible, run_id)
        if any(r.get('scope') != scope or r.get('status') != 'ok' for r in observations.values()):
            raise ValueError('Ranking cache scope/status mismatch')
        for sample in eligible:
            eid = sample['example_id']
            if eid in observations:
                continue
            if official:
                history = baseline[eid]['rollout']
                collected = [runtime.collect(sample, scope, step['step'], step['previous_percentage_text'])
                             for step in history]
            else:
                collected = [runtime.collect(sample, scope)]
            row = {'example_id': eid, 'sample_id': sample_identity(sample), 'run_id': run_id, 'status': 'ok',
                   'raw_mass': np.mean([c['raw_mass'] for c in collected], axis=0).tolist(),
                   'scope': scope, 'query_kind': collected[0]['query_kind'],
                   'aggregation': 'mean_over_seven_unsteered_steps' if official else 'last_prompt_or_prog_token',
                   'token_audits': [c['token_audit'] for c in collected], **protocol_metadata(runtime.cfg)}
            append(path, [row])
            observations[eid] = row
            print(f'RANK {scope} {len(observations)}/{len(eligible)}', flush=True)
        matrix = np.mean([observations[s['example_id']]['raw_mass'] for s in eligible], axis=0)
        if not np.isfinite(matrix).all() or matrix.shape != (runtime.num_layers, runtime.num_heads):
            raise ValueError('Invalid ranking observation shape/values')
        heads = [{'layer': l, 'head': h, 'score': float(matrix[l, h])}
                 for l in range(runtime.cfg['skip_early_layers'], runtime.num_layers) for h in range(runtime.num_heads)]
        heads.sort(key=lambda h: (-h['score'], h['layer'], h['head']))
        if len(heads) < max(runtime.cfg['top_k']):
            raise ValueError('Requested top_k exceeds available ranked heads')
        artifact = {'run_id': run_id, 'scope': scope, 'ranking': heads,
                    'example_ids': [s['example_id'] for s in eligible], 'n': len(eligible),
                    'num_layers': runtime.num_layers, 'num_heads': runtime.num_heads,
                    'ranking_score': 'mean_raw_mass', 'checks_sha256': fingerprint(checks),
                    **protocol_metadata(runtime.cfg)}
        create_json(output / f'ranking_{scope}.json', artifact)
        rankings[scope] = artifact
    return rankings


def preflight(runtime, samples, ranking_samples, output):
    report = {}
    for name, population in [('evaluation', samples), ('ranking', ranking_samples)]:
        report[name] = {}
        for index, sample in enumerate(population):
            checks = {}
            for scope in runtime.cfg['scopes']:
                check = check_sample(runtime, sample, scope)
                if check['eligible']:
                    try:
                        runtime.control_preflight(sample, scope)
                        check['wrong_region_available'] = True
                    except ControlUnavailable:
                        check['wrong_region_available'] = False
                checks[scope] = check
            report[name][sample['example_id']] = checks
            if (index + 1) % 25 == 0:
                print(f'PREFLIGHT {name} {index + 1}/{len(population)}', flush=True)
    create_json(output / 'preflight.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--phase', choices=['all', 'preflight', 'baseline', 'rank', 'steer'], default='all')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ranking-limit', type=int)
    parser.add_argument('--output-suffix', default='')
    args = parser.parse_args()
    if (args.limit is not None or args.ranking_limit is not None) and not args.output_suffix:
        parser.error('Limited validation requires a separate --output-suffix')
    if any(n is not None and n <= 0 for n in (args.limit, args.ranking_limit)):
        parser.error('Limits must be positive')
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    validate_inputs(cfg)
    samples = json.loads(Path(cfg['inputs']).read_text())
    ranking_samples = json.loads(Path(cfg['ranking_inputs']).read_text())
    if args.limit:
        samples = samples[:args.limit]
    if args.ranking_limit:
        ranking_samples = ranking_samples[:args.ranking_limit]
    output = Path(cfg['output_dir'] + args.output_suffix)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'config': cfg, 'implementation': implementation_identity(),
                    'input_media': validate_media(cfg, samples + ranking_samples),
                    'evaluation_ids': [s['example_id'] for s in samples],
                    'ranking_ids': [s['example_id'] for s in ranking_samples],
                    'model': artifact_fingerprint(cfg['model_path']),
                    'processor': artifact_fingerprint(cfg['processor_path']),
                    'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'qwen-vl-utils')}}
        create_json(output / 'run_config.json', cfg)
        create_json(output / 'run_identity.json', identity)
        run_id = fingerprint(identity)
        if args.phase == 'all' and (output / 'completion.json').exists():
            completion = json.loads((output / 'completion.json').read_text())
            if completion['run_id'] != run_id or completion['conditions'] != conditions(cfg):
                raise ValueError('Completion manifest identity mismatch')
            for condition in conditions(cfg):
                path = output / 'predictions' / (condition.replace(':', '_') + '.jsonl')
                if len(cache_rows(path, samples, run_id, condition)) != len(samples):
                    raise ValueError(f'Completed run has missing predictions: {path}')
            print(f'Already complete: {output}', flush=True)
            return
        from .runtime import Runtime
        runtime = Runtime(cfg, processor_only=args.phase == 'preflight')
        if args.phase == 'preflight':
            preflight(runtime, samples, ranking_samples, output)
            print(f'Preflight complete: {output / "preflight.json"}')
            return
        create_json(output / 'loading_audit.json', runtime.model.loading_audit)
        if args.phase in ('all', 'baseline'):
            predict_condition(runtime, samples, 'baseline', None, output, run_id)
        if args.phase == 'baseline':
            return
        ranking_output = output / 'independent_ranking'
        ranking_output.mkdir(exist_ok=True)
        if args.phase in ('all', 'rank'):
            # Native metadata and SOLE history must come from this same processor/config.
            need_base = is_official_sole(cfg) or cfg['protocol'] == 'official' and cfg['model'] in ('qwen', 'roboreward')
            baseline = predict_condition(runtime, ranking_samples, 'baseline', None, ranking_output, run_id) if need_base else {}
            rankings = rank(runtime, ranking_samples, baseline, ranking_output, run_id)
        else:
            rankings = {s: json.loads((ranking_output / f'ranking_{s}.json').read_text()) for s in cfg['scopes']}
            if any(r['run_id'] != run_id or r['scope'] != s for s, r in rankings.items()):
                raise ValueError('Ranking identity mismatch')
        if args.phase == 'rank':
            return
        for condition in conditions(cfg)[1:]:
            predict_condition(runtime, samples, condition, rankings, output, run_id)
        create_json(output / 'completion.json', {'run_id': run_id, 'conditions': conditions(cfg),
                                               'examples_per_condition': len(samples)})


if __name__ == '__main__':
    main()
