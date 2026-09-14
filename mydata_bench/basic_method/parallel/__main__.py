"""Use idle GPUs for disjoint conditions of an existing SOLE experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from mydata_bench.basic_method.common import ROOT, conditions, create_json
from mydata_bench.basic_method.schedule import gpu_snapshot, GPUProbeError, unavailable_reasons
from .core import (
    available_conditions, condition_lock, condition_path, execution_manifest,
    finish_run, load_run, lock_file, log_event, strict_rows, verify_manifest,
)


def worker(args):
    if args.lease_fd is None:
        raise ValueError('Workers must inherit the coordinator experiment lock')
    # The inherited descriptor keeps the experiment lock alive even if the
    # coordinator unexpectedly exits while a worker is still writing.
    expected = Path(yaml.safe_load(args.config.read_text())['output_dir']) / '.run.lock'
    inherited = os.fstat(args.lease_fd)
    locked = expected.stat()
    if (inherited.st_dev, inherited.st_ino) != (locked.st_dev, locked.st_ino):
        raise ValueError('Wrong inherited experiment lock')
    manifest = verify_manifest(args.execution_dir)
    cfg, samples, run_id, rankings = load_run(args.config, check_media=False)
    if manifest['run_id'] != run_id or manifest['config_path'] != str(args.config.resolve()):
        raise ValueError('Worker execution identity mismatch')
    from mydata_bench.basic_method.runtime import Runtime
    from mydata_bench.basic_method.run import predict_condition
    runtime = Runtime(cfg)
    log_event(args.execution_dir, 'worker_ready', gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for condition in conditions(cfg)[1:]:
        with lock_file(condition_lock(args.execution_dir, condition)) as handle:
            if handle is None:
                continue
            existing = strict_rows(condition_path(cfg, condition), samples, run_id, condition)
            if len(existing) == len(samples):
                continue
            log_event(args.execution_dir, 'condition_started', condition=condition, existing=len(existing))
            predict_condition(runtime, samples, condition, rankings, Path(cfg['output_dir']), run_id)
            if len(strict_rows(condition_path(cfg, condition), samples, run_id, condition)) != len(samples):
                raise ValueError('Incomplete condition returned from frozen predictor')
            log_event(args.execution_dir, 'condition_finished', condition=condition, examples=len(samples))
    log_event(args.execution_dir, 'worker_finished')


def coordinate(args, cfg, samples, run_id, lease):
    directory = args.execution_dir
    directory.mkdir(parents=True, exist_ok=True)
    manifest = execution_manifest(args.config, cfg, run_id, args.gpus)
    create_json(directory / 'manifest.json', manifest)
    jobs, attempts, idle_counts = {}, {}, {g: 0 for g in args.gpus}
    completed_cache = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    log_event(directory, 'coordinator_started', gpus=args.gpus, run_id=run_id)
    last_status = 0
    try:
        while not stopping:
            for gpu, job in list(jobs.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                log_event(directory, 'worker_exited', gpu=gpu, worker_pid=job['process'].pid, exit_code=code)
                del jobs[gpu]
                idle_counts[gpu] = 0
                if code:
                    attempts[gpu] = attempts.get(gpu, 0) + 1
                    if attempts[gpu] >= 3:
                        raise RuntimeError(f'GPU {gpu} worker failed three times; see worker logs')
            pending, claimed = available_conditions(directory, cfg, samples, run_id, completed_cache)
            if not pending and not claimed and not jobs:
                finish_run(cfg, samples, run_id)
                log_event(directory, 'experiment_complete', run_id=run_id)
                print('All conditions complete: ' + cfg['output_dir'], flush=True)
                return
            if pending:
                try:
                    snapshot = gpu_snapshot()
                except GPUProbeError as exc:
                    log_event(directory, 'gpu_probe_failed', error=str(exc))
                    snapshot = {}
                for gpu in args.gpus:
                    if gpu in jobs or gpu not in snapshot:
                        continue
                    reasons = unavailable_reasons(snapshot[gpu], 24 * 1024, 10)
                    idle_counts[gpu] = 0 if reasons else idle_counts[gpu] + 1
                    if idle_counts[gpu] < 2:
                        continue
                    stamp = str(time.time_ns())
                    handle = (directory / f'gpu{gpu}_{stamp}.log').open('x')
                    command = [sys.executable, '-u', '-m', 'mydata_bench.basic_method.parallel',
                               '--config', str(args.config.resolve()), '--execution-dir', str(directory.resolve()),
                               '--worker', '--lease-fd', str(lease.fileno())]
                    try:
                        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(lease.fileno(),), env={**os.environ,
                                'CUDA_VISIBLE_DEVICES': snapshot[gpu].uuid, 'TOKENIZERS_PARALLELISM': 'false'})
                    except BaseException:
                        handle.close()
                        raise
                    jobs[gpu] = {'process': process, 'log': handle}
                    log_event(directory, 'worker_spawned', gpu=gpu, gpu_uuid=snapshot[gpu].uuid,
                              worker_pid=process.pid, log=handle.name)
                    print(f'GPU {gpu}: worker {process.pid}', flush=True)
            if time.monotonic() - last_status >= 60:
                log_event(directory, 'status', pending=pending, claimed=claimed,
                          workers={g: j['process'].pid for g, j in jobs.items()})
                print(f'Conditions pending={len(pending)}, active={len(claimed)}; GPUs={list(jobs)}', flush=True)
                last_status = time.monotonic()
            time.sleep(5)
        raise KeyboardInterrupt('Coordinator stopping')
    finally:
        for job in jobs.values():
            if job['process'].poll() is None:
                try:
                    os.killpg(job['process'].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for job in jobs.values():
            try:
                job['process'].wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(job['process'].pid, signal.SIGKILL)
                job['process'].wait()
            finally:
                job['log'].close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--execution-dir', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('Distinct nonnegative GPU IDs required')
    if args.worker:
        worker(args)
        return
    cfg = yaml.safe_load(args.config.read_text())
    if args.check_only:
        cfg, samples, run_id, _ = load_run(args.config)
        print(json.dumps({'identity_verified': True, 'run_id': run_id, 'samples': len(samples),
                          'steering_conditions': len(conditions(cfg)) - 1}, indent=2))
        return
    with lock_file(Path(cfg['output_dir']) / '.run.lock') as lease:
        if lease is None:
            raise RuntimeError('Original runner or another coordinator still owns this experiment')
        cfg, samples, run_id, _ = load_run(args.config)
        snapshot = gpu_snapshot()
        if set(args.gpus) - snapshot.keys():
            raise ValueError('Requested physical GPU IDs do not exist')
        coordinate(args, cfg, samples, run_id, lease)


if __name__ == '__main__':
    main()
