from __future__ import annotations

from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time

import yaml

from mydata_bench.io import artifact_fingerprint
from mydata_bench.basic_method.common import (
    append, conditions, create_json, file_hash, fingerprint,
    implementation_identity, validate_config, validate_inputs, validate_media,
)
from mydata_bench.basic_method.run import cache_rows


@contextmanager
def lock_file(path):
    """Yield None on contention; never unlink a lock inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield handle
    finally:
        handle.close()


def log_event(directory, event, **values):
    directory = Path(directory)
    with (directory / '.events.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        append(directory / 'events.jsonl', [dict(time=time.time(), pid=os.getpid(), event=event, **values)])


def strict_rows(path, samples, run_id, condition):
    rows = cache_rows(path, samples, run_id, condition)
    if Path(path).exists():
        with Path(path).open() as handle:
            count = sum(bool(line.strip()) for line in handle)
        if count != len(rows):
            raise ValueError(f'Duplicate prediction IDs in {path}')
    return rows


def load_run(config_path, *, check_media=True):
    cfg = yaml.safe_load(Path(config_path).read_text())
    validate_config(cfg)
    if cfg['model'] != 'sole':
        raise ValueError('This continuation entry point is limited to SOLE')
    validate_inputs(cfg)
    output = Path(cfg['output_dir'])
    samples = json.loads(Path(cfg['inputs']).read_text())
    rank_samples = json.loads(Path(cfg['ranking_inputs']).read_text())
    saved = json.loads((output / 'run_identity.json').read_text())
    current = {
        'config': cfg, 'implementation': implementation_identity(),
        'input_media': validate_media(cfg, samples + rank_samples) if check_media else saved['input_media'],
        'evaluation_ids': [s['example_id'] for s in samples],
        'ranking_ids': [s['example_id'] for s in rank_samples],
        'model': artifact_fingerprint(cfg['model_path']),
        'processor': artifact_fingerprint(cfg['processor_path']),
        'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'qwen-vl-utils')},
    }
    if current != saved or json.loads((output / 'run_config.json').read_text()) != cfg:
        raise ValueError('Frozen experiment identity changed; refusing mixed continuation')
    run_id = fingerprint(saved)
    baseline = strict_rows(output / 'predictions/baseline.jsonl', samples, run_id, 'baseline')
    if len(baseline) != len(samples):
        raise ValueError('A complete matching baseline is required before parallel steering')
    rankings = {}
    for scope in cfg['scopes']:
        value = json.loads((output / 'independent_ranking' / f'ranking_{scope}.json').read_text())
        if value['run_id'] != run_id or value['scope'] != scope:
            raise ValueError('Frozen ranking identity mismatch')
        rankings[scope] = value
    return cfg, samples, run_id, rankings


def execution_manifest(config_path, cfg, run_id, gpus):
    output = Path(cfg['output_dir'])
    artifacts = [output / 'run_identity.json', output / 'run_config.json',
                 output / 'predictions/baseline.jsonl']
    artifacts += [output / 'independent_ranking' / f'ranking_{s}.json' for s in cfg['scopes']]
    return {
        'version': 'sole_condition_parallel_v1', 'config_path': str(Path(config_path).resolve()),
        'run_id': run_id, 'gpus': list(gpus), 'conditions': conditions(cfg)[1:],
        'frozen_artifacts': {str(p): file_hash(p) for p in artifacts},
        'orchestration_sources': {str(p): file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'inference': 'unchanged basic_method.run.predict_condition; batch=1; original recursive history',
    }


def verify_manifest(directory):
    manifest = json.loads((Path(directory) / 'manifest.json').read_text())
    for field in ('frozen_artifacts', 'orchestration_sources'):
        for path, expected in manifest[field].items():
            if file_hash(path) != expected:
                raise ValueError(f'Continuation artifact changed: {path}')
    return manifest


def condition_path(cfg, condition):
    return Path(cfg['output_dir']) / 'predictions' / (condition.replace(':', '_') + '.jsonl')


def condition_lock(directory, condition):
    return Path(directory) / 'locks' / (condition.replace(':', '_') + '.lock')


def available_conditions(directory, cfg, samples, run_id, completed_cache=None):
    pending, claimed = [], []
    completed_cache = {} if completed_cache is None else completed_cache
    for condition in conditions(cfg)[1:]:
        with lock_file(condition_lock(directory, condition)) as handle:
            if handle is None:
                claimed.append(condition)
            else:
                path = condition_path(cfg, condition)
                stat = path.stat() if path.exists() else None
                stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns) if stat else None
                if stamp is not None and completed_cache.get(condition) == stamp:
                    continue
                if len(strict_rows(path, samples, run_id, condition)) != len(samples):
                    pending.append(condition)
                else:
                    completed_cache[condition] = stamp
    return pending, claimed


def finish_run(cfg, samples, run_id):
    for condition in conditions(cfg):
        if len(strict_rows(condition_path(cfg, condition), samples, run_id, condition)) != len(samples):
            raise ValueError(f'Cannot mark incomplete condition finished: {condition}')
    create_json(Path(cfg['output_dir']) / 'completion.json', {
        'run_id': run_id, 'conditions': conditions(cfg), 'examples_per_condition': len(samples),
    })
