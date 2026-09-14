"""Queue whole configurations with optional two-worker GPU sharing and memory reservations."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import ROOT, OUT, append, conditions
from .selection import add_selection_arguments, select_configs


class GPUProbeError(RuntimeError):
    """Unknown GPU status must never be interpreted as an idle device."""


@dataclass
class GPUState:
    index: int
    uuid: str
    free_mib: int
    total_mib: int
    utilization: int
    compute_pids: list[int]


def gpu_snapshot():
    def query(fields):
        result = subprocess.run(['nvidia-smi', fields, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=10)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))

    try:
        states = {}
        for index, uuid, free, total, utilization in query('--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu'):
            state = GPUState(int(index), uuid.strip(), int(free), int(total), int(utilization), [])
            if (state.index in states or state.index < 0 or not state.uuid.startswith('GPU-')
                    or not 0 <= state.free_mib <= state.total_mib or state.total_mib <= 0
                    or not 0 <= state.utilization <= 100):
                raise ValueError(f'Invalid GPU status: {state}')
            states[state.index] = state
        by_uuid = {s.uuid: s for s in states.values()}
        if not states or len(by_uuid) != len(states):
            raise ValueError('Missing or duplicate GPU identities')
        for uuid, pid in query('--query-compute-apps=gpu_uuid,pid'):
            by_uuid[uuid.strip()].compute_pids.append(int(pid))
        return states
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        raise GPUProbeError(f'Cannot establish GPU availability: {exc}') from exc


def unavailable_reasons(state, min_free_mib, max_utilization):
    reasons = []
    if state.compute_pids:
        reasons.append(f'compute processes {state.compute_pids}')
    if state.free_mib < min_free_mib:
        reasons.append(f'free memory {state.free_mib} < {min_free_mib} MiB')
    if state.utilization > max_utilization:
        reasons.append(f'utilization {state.utilization}% > {max_utilization}%')
    return reasons


def exclusive_config(cfg):
    # Concurrency throughput and peak memory have only been profiled for the
    # current 15-config subset. Keep unprofiled long/native workloads exclusive.
    return cfg['model'] == 'sole' or (cfg['model'] == 'qwen' and cfg['protocol'] == 'official')


def admission_reasons(state, jobs, cfg, args):
    if len(jobs) >= args.jobs_per_gpu:
        return ['worker slots full']
    if not jobs:
        return unavailable_reasons(state, max(args.min_free_memory_mib,
            args.job_memory_mib + args.memory_headroom_mib if args.jobs_per_gpu > 1 else 0), args.max_utilization)
    if exclusive_config(cfg) or any(job['exclusive'] for job in jobs):
        return ['configuration requires an exclusive GPU']
    groups = {job['process'].pid for job in jobs}
    external = []
    for pid in state.compute_pids:
        if pid in groups:
            continue
        try:
            belongs = os.getpgid(pid) in groups
        except (ProcessLookupError, PermissionError):
            belongs = False
        if not belongs:
            external.append(pid)
    reasons = [f'external compute processes {external}'] if external else []
    # Reserve the FULL budget of each existing worker in addition to measured
    # allocations. This is deliberately conservative and covers model-loading
    # gaps even before nvidia-smi can see a new worker's CUDA context.
    required = args.job_memory_mib + args.memory_headroom_mib + sum(j['reserved_mib'] for j in jobs)
    if state.free_mib < required:
        reasons.append(f'free memory {state.free_mib} < {required} MiB including worker reservations')
    # A busy device is expected once our first worker starts. The utilization
    # threshold still protects empty slots from external activity, but is not
    # a reason to reject a second explicitly requested worker of this scheduler.
    return reasons


def prioritize(pending):
    # SOLE official has seven recursive calls per example. Starting its long
    # configurations early reduces the tail without changing any predictions.
    return sorted(pending, key=lambda item: (item[1]['model'] != 'sole',
                  item[1]['protocol'] != 'official' if item[1]['model'] == 'sole' else False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    add_selection_arguments(parser)
    parser.add_argument('--jobs-per-gpu', type=int, choices=[1, 2], default=1,
                        help='Maximum workers per physical GPU (default: 1; SOLE / Qwen official stay exclusive)')
    parser.add_argument('--job-memory-mib', type=int, default=24 * 1024,
                        help='Extra reservation per worker when sharing a GPU (default: 24 GiB)')
    parser.add_argument('--memory-headroom-mib', type=int, default=8 * 1024,
                        help='Free-memory margin when sharing a GPU (default: 8 GiB)')
    parser.add_argument('--min-free-memory-mib', type=int, default=60 * 1024,
                        help='Required free memory before the first worker on a GPU (default: 60 GiB)')
    parser.add_argument('--max-utilization', type=int, default=10,
                        help='Utilization threshold before the first worker; own active workers may share')
    parser.add_argument('--poll-seconds', type=int, default=10)
    parser.add_argument('--idle-checks', type=int, default=2,
                        help='Consecutive admissible resource checks required before launch')
    parser.add_argument('--dry-run', action='store_true', help='Read GPU status and selected queue; launch nothing, write nothing')
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('GPU IDs must be distinct and nonnegative')
    if (min(args.min_free_memory_mib, args.job_memory_mib) <= 0 or args.memory_headroom_mib < 0
            or not 0 <= args.max_utilization <= 100
            or not 1 <= args.poll_seconds <= 60 or args.idle_checks < 1):
        parser.error('Invalid thresholds, idle checks, or poll interval (1–60 seconds)')
    try:
        pending = prioritize(select_configs(args.root, args.models, args.exclude_configs))
    except ValueError as exc:
        parser.error(str(exc))
    selection = {'models': args.models, 'exclude_configs': args.exclude_configs,
                 'configs': [Path(path).stem for path, _ in pending],
                 'jobs_per_gpu': args.jobs_per_gpu,
                 'exclusive_configs': [Path(path).stem for path, cfg in pending if exclusive_config(cfg)],
                 'job_memory_mib': args.job_memory_mib, 'memory_headroom_mib': args.memory_headroom_mib,
                 'baseline_conditions': len(pending),
                 'steering_conditions': sum(len(conditions(cfg)) - 1 for _, cfg in pending)}

    def inspect():
        states = gpu_snapshot()
        missing = set(args.gpus) - states.keys()
        if missing:
            raise ValueError(f'GPU IDs absent from nvidia-smi: {sorted(missing)}')
        if any(states[g].total_mib < args.min_free_memory_mib for g in args.gpus):
            raise ValueError('Requested free-memory threshold exceeds a selected GPU capacity')
        if args.jobs_per_gpu > 1 and any(states[g].total_mib < args.job_memory_mib + args.memory_headroom_mib for g in args.gpus):
            raise ValueError('Worker memory budget exceeds a selected GPU capacity')
        return {g: {'state': asdict(states[g]), 'unavailable_reasons': unavailable_reasons(
            states[g], max(args.min_free_memory_mib,
                args.job_memory_mib + args.memory_headroom_mib if args.jobs_per_gpu > 1 else 0),
            args.max_utilization)} for g in args.gpus}

    if args.dry_run:
        try:
            selection['gpus'] = inspect()
        except GPUProbeError as exc:
            selection['gpu_probe_error'] = str(exc)
        print(json.dumps({**selection, 'idle_checks_required': args.idle_checks,
                          'poll_seconds': args.poll_seconds}, ensure_ascii=False, indent=2))
        return
    log_dir = args.root / 'scheduler'
    log_dir.mkdir(parents=True, exist_ok=True)
    session = str(time.time_ns())
    running, failures = {}, []
    idle_counts = dict.fromkeys(args.gpus, 0)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    def record(event, **values):
        append(log_dir / f'{session}.jsonl', [{'event': event, 'time': time.time(), **values}])

    def gpu_jobs(gpu):
        return [job for job in running.values() if job['gpu'] == gpu]

    with (log_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        last_status = float('-inf')
        record('selection', **selection)
        record('scheduler_started', pid=os.getpid(), gpus=args.gpus,
               min_free_memory_mib=args.min_free_memory_mib, max_utilization=args.max_utilization,
               idle_checks=args.idle_checks, poll_seconds=args.poll_seconds)
        try:
            while (pending or running) and not stopping:
                for pid, job in list(running.items()):
                    gpu = job['gpu']
                    code = job['process'].poll()
                    if code is not None:
                        job['handle'].close()
                        record('finished', gpu=gpu, pid=pid, config=job['config'], exit_code=code,
                               elapsed_seconds=time.monotonic() - job['started'])
                        print(f'GPU {gpu}: {Path(job["config"]).stem} finished, exit_code={code}', flush=True)
                        if code:
                            failures.append(job['config'])
                        del running[pid]
                        idle_counts[gpu] = 0
                status, probe_error = {}, None
                if pending and any(len(gpu_jobs(g)) < args.jobs_per_gpu for g in args.gpus):
                    try:
                        status = inspect()
                    except GPUProbeError as exc:
                        probe_error = str(exc)
                    # Spread workers across cards before adding a second worker.
                    for gpu in sorted(args.gpus, key=lambda g: len(gpu_jobs(g))):
                        if gpu not in status or not pending:
                            idle_counts[gpu] = 0
                            continue
                        jobs = gpu_jobs(gpu)
                        state = GPUState(**status[gpu]['state'])
                        candidates = [(i, admission_reasons(state, jobs, cfg, args))
                                      for i, (_, cfg) in enumerate(pending)]
                        selected = next((i for i, reasons in candidates if not reasons), None)
                        status[gpu]['unavailable_reasons'] = candidates[0][1] if selected is None else []
                        status[gpu]['active_configs'] = [j['config'] for j in jobs]
                        if selected is None:
                            idle_counts[gpu] = 0
                            continue
                        idle_counts[gpu] += 1
                        if idle_counts[gpu] < args.idle_checks or not pending or stopping:
                            continue
                        path, cfg = pending.pop(selected)
                        handle = (log_dir / f'{session}_{Path(path).stem}.log').open('x')
                        command = [sys.executable, '-u', '-m', 'mydata_bench.basic_method.run', '--config', path]
                        # UUIDs avoid CUDA / nvidia-smi enumeration disagreements.
                        uuid = status[gpu]['state']['uuid']
                        try:
                            proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': uuid, 'TOKENIZERS_PARALLELISM': 'false'})
                        except OSError as exc:
                            handle.close()
                            failures.append(path)
                            record('failed_to_start', gpu=gpu, config=path, error=str(exc))
                            print(f'GPU {gpu}: {Path(path).stem} failed to start: {exc}', flush=True)
                            idle_counts[gpu] = 0
                            continue
                        running[proc.pid] = {'process': proc, 'handle': handle, 'config': path, 'gpu': gpu,
                                             'exclusive': exclusive_config(cfg), 'reserved_mib': args.job_memory_mib,
                                             'started': time.monotonic()}
                        idle_counts[gpu] = 0
                        record('started', gpu=gpu, gpu_uuid=uuid, pid=proc.pid, config=path, gpu_status=status[gpu])
                        print(f'GPU {gpu}: {Path(path).stem}, pid={proc.pid}', flush=True)
                if time.monotonic() - last_status >= 60 and (pending or running):
                    record('status', pending=len(pending), running={g: [j['config'] for j in gpu_jobs(g)] for g in args.gpus},
                           gpu_status=status, probe_error=probe_error, idle_counts=idle_counts)
                    print(f'Pending {len(pending)}, running {len(running)}, failed {len(failures)}'
                          + (f'; waiting: {probe_error}' if probe_error else ''), flush=True)
                    if pending and not running and status:
                        print('Waiting for idle GPUs: ' + json.dumps(status, ensure_ascii=False), flush=True)
                    last_status = time.monotonic()
                # Also sleep when external jobs occupy every GPU or the query
                # fails: an empty worker list must not cause a busy loop.
                if (pending or running) and not stopping:
                    time.sleep(args.poll_seconds)
        finally:
            # Hold the scheduler lock until our private worker groups stop.
            # External GPU processes are never signalled.
            for job in running.values():
                if job['process'].poll() is None:
                    try:
                        os.killpg(job['process'].pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for pid, job in list(running.items()):
                try:
                    job['process'].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(job['process'].pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    job['process'].wait()
                finally:
                    job['handle'].close()
                record('interrupted', gpu=job['gpu'], pid=pid, config=job['config'], exit_code=job['process'].returncode)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        record('scheduler_finished', failed=failures, interrupted=stopping, pending=[p for p, _ in pending])
        print(json.dumps({'failed': failures, 'interrupted': stopping, 'log_dir': str(log_dir)}, ensure_ascii=False), flush=True)
    if stopping:
        raise SystemExit(130)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
