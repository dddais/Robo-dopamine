"""Replay saved predictions on another GPU without touching experiment outputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mydata_bench.basic_method.common import append, create_json
from mydata_bench.basic_method.run import predict_condition
from .core import condition_path, load_run, strict_rows


def stable(value):
    if isinstance(value, list):
        return [stable(v) for v in value]
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in ('duration_seconds', 'completed_at')}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', default='last_frame:target:8')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Verification requires a fresh output directory')
    cfg, samples, run_id, rankings = load_run(args.config)
    saved = strict_rows(condition_path(cfg, args.condition), samples, run_id, args.condition)
    applied = [s for s in samples if saved.get(s['example_id'], {}).get('sas_applied')
               and saved[s['example_id']]['status'] == 'ok']
    if len(applied) < 2:
        raise ValueError('Two existing actually-steered samples are required')
    # Prefer both label groups when available; otherwise span the saved prefix.
    first = applied[0]
    second = next((s for s in applied if s['example_id'].split('/')[0] != first['example_id'].split('/')[0]), applied[-1])
    chosen = [first, second]
    baseline = strict_rows(Path(cfg['output_dir']) / 'predictions/baseline.jsonl', samples, run_id, 'baseline')
    args.output.mkdir(parents=True)
    append(args.output / 'predictions/baseline.jsonl', [copy.deepcopy(baseline[s['example_id']]) for s in chosen])
    from mydata_bench.basic_method.runtime import Runtime
    runtime = Runtime(cfg)
    result = predict_condition(runtime, chosen, args.condition, rankings, args.output, run_id)
    comparisons = {}
    for sample in chosen:
        eid = sample['example_id']
        expected, actual = stable(saved[eid]), stable(result[eid])
        differences = [k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k)]
        comparisons[eid] = {'exact_except_timing': not differences, 'different_fields': differences}
    # A second invocation must perform no inference for completed examples.
    def unexpected_inference(*unused, **unused_kwargs):
        raise AssertionError('Completed cache unexpectedly reran inference')
    runtime.predict = unexpected_inference
    resumed = predict_condition(runtime, chosen, args.condition, rankings, args.output, run_id)
    if stable(resumed) != stable(result):
        raise AssertionError('Resume changed saved predictions')
    report = {'config': str(args.config), 'condition': args.condition, 'comparisons': comparisons,
              'resume_without_inference': True, 'run_id': run_id,
              'passed': all(c['exact_except_timing'] for c in comparisons.values())}
    create_json(args.output / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
