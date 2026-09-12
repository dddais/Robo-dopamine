"""Round18 verified smoke -> full profiling -> frozen robust heads -> discovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mydata_bench.addbase_eval.prepare import create_json
from mydata_bench.addbase_eval.run import append
from .prepare import OUT
from .functional_pipeline import wait_for_child
from .gpu_policy import require_allowed_visibility


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'roboreward'], required=True)
    parser.add_argument('--protocols', nargs='+', required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    require_allowed_visibility()
    protocols = ['image_text', 'text_image', 'interleaved', 'text_video', 'video_text']
    if args.protocols != protocols or not args.audit.resolve().is_relative_to((OUT / 'audit').resolve()):
        raise ValueError('Require the five frozen protocols and a local smoke audit')
    audit = json.loads(args.audit.read_text())
    checks = [c for c in audit['checks'] if c['model'] == args.model]
    if (audit['status'] != 'pass' or audit['labels_read'] or len(checks) != 2 or
            {c['scope'] for c in checks} != {'all_frames', 'last_frame'} or
            any(not c['baseline_positive_negative_exact'] or not c['actual_heads_exact'] or c['n'] != 8 for c in checks)):
        raise ValueError('Both same-group actual smoke comparisons must pass')
    for path, sha in audit['sources_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('Actual smoke changed after verification')
    folder = OUT / 'functional_selections' / f'stage18_{args.model}_pipeline'
    create_json(folder / 'intent.json', dict(model=args.model, protocols=protocols, audit=str(args.audit),
        audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        phases=['true method profiling', 'CPU discovery-label robust ranking', 'five-input discovery'],
        no_full_validation_before_new_selection=True))
    append(folder / 'events.jsonl', [dict(event='attempt', time=time.time(), pid=os.getpid(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])])
    def run(module, arguments, cpu=False):
        command = [sys.executable, '-B', '-m', 'mydata_bench.auto_research_addbase.' + module, *arguments]
        append(folder / 'events.jsonl', [dict(event='start', time=time.time(), command=command)])
        process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='') if cpu else None)
        try:
            code = wait_for_child(process)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except SystemExit as exc:
            append(folder / 'events.jsonl', [dict(event='interrupted', time=time.time(), command=command, exit_code=exc.code)])
            raise
        append(folder / 'events.jsonl', [dict(event='complete', time=time.time(), command=command)])
    common = ['--model', args.model, '--protocols', *protocols]
    run('matched_profile_worker', common)
    run('select_matched_profiles', ['--model', args.model], cpu=True)
    run('worker', common + ['--variant', 'uniform_evidence_robustprofile_a1', '--methods', 'binding_transport',
        '--task-binding-fraction', '.5', '--task-binding-distribution', 'uniform', '--contrast-weight', '1',
        '--negative-strength', '4', '--strengths', '4', '--ks', '8', '32', '64', '--ranking-prefix', 'ANSWER: ',
        '--frozen-ranking-root', str(OUT / 'functional_selections' / f'stage18_{args.model}_robust_v1'),
        '--population', 'discovery'])


if __name__ == '__main__':
    main()
