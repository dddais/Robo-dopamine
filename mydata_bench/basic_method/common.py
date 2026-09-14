from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/mydata_bench/experiments_v2_basic_method'
CONFIGS = ROOT / 'mydata_bench/configs/v2_basic_method'
VERSION = 'basic_method_exact_v1'
MODELS = ('grm', 'roboreward', 'qwen', 'meter', 'sole')
PROTOCOLS = ('official', 'text_image', 'image_text', 'interleaved')
SCOPES = ('last_frame', 'all_frames')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def create_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f'Refusing to replace artifact: {path}; choose a new output directory')
        return
    with path.open('x') as handle:
        handle.write(text)


def create_json(path, value):
    create_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + '\n')


def append(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


def latest(path):
    if not Path(path).exists():
        return {}
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row['example_id']] = row
    return rows


def conditions(cfg):
    return ['baseline'] + [f'{scope}:{kind}:{k}' for scope in cfg['scopes']
                          for k in cfg['top_k'] for kind in ['target', *cfg['controls']]]


def validate_config(cfg):
    from mydata_bench.top_eval.versioning import validate_protocol_config
    validate_protocol_config(cfg)
    if cfg.get('basic_method_version') != VERSION:
        raise ValueError('Not a versioned basic_method config')
    if cfg['model'] not in MODELS or cfg['protocol'] not in PROTOCOLS:
        raise ValueError('Unknown model or input protocol')
    if (set(cfg['scopes']) - set(SCOPES) or not cfg['scopes']
            or set(cfg['controls']) - {'wrong_region', 'low_rank'}):
        raise ValueError('Unknown scope/control; wrong_region is not a tracked wrong object')
    if cfg['bias'] <= 0 or not cfg['top_k'] or any(k <= 0 for k in cfg['top_k']):
        raise ValueError('Positive bias and top_k required')
    if cfg.get('query_scope') != 'all' or cfg.get('decoding') != 'greedy':
        raise ValueError('This implementation requires all-query bias and greedy decoding')
    if cfg.get('num_frames') != 8:
        raise ValueError('Independent-image and SOLE protocols require eight frames')
    if cfg['model'] == 'grm' and (cfg.get('eval_mode') != 'forward' or cfg.get('grm_all_frame_slots') != [
            'reference_start', 'before_cam_high', 'after_cam_high']):
        raise ValueError('GRM requires forward mode and the three declared front-camera slots')


def validate_inputs(cfg):
    for field in ('inputs', 'ranking_inputs', 'labels'):
        if file_hash(cfg[field]) != cfg[field + '_sha256']:
            raise ValueError(f'Frozen {field} changed; prepare a new experiment directory')


def media_hashes(paths):
    """Hash shared physical files once, retaining every referenced path."""
    cache, result = {}, {}
    for path in sorted(set(paths)):
        stat = Path(path).stat()
        key = stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
        if key not in cache:
            cache[key] = file_hash(path)
        result[path] = cache[key]
    return result


def validate_media(cfg, samples):
    path = Path(cfg['inputs']).parent / 'input_artifacts.json'
    frozen = json.loads(path.read_text())
    paths = []
    for sample in samples:
        if cfg['model'] in ('qwen', 'roboreward') and cfg['protocol'] == 'official':
            paths.append(sample['video_path'])
        else:
            paths.extend(sample['grm_image_paths'] if cfg['model'] == 'grm' else sample['image_paths'])
    actual = media_hashes(paths)
    if any(frozen.get(p) != digest for p, digest in actual.items()):
        raise ValueError('Frozen image/video input changed; prepare a new experiment directory')
    return fingerprint(actual)


def implementation_identity():
    # No Git calls. Changes to implementation or shared protocol code invalidate resume.
    sources = [p for base in ('basic_method', 'addbase_eval', 'top_eval', 'meter_eval',
                             'qwen_eval', 'roboreward_eval', 'attention_eval')
               for p in (ROOT / 'mydata_bench' / base).glob('*.py')]
    sources += [ROOT / 'mydata_bench' / p for p in ('protocol.py', 'video.py', 'data.py')]
    # SOLE imports live prompt/composite definitions from this vendored source.
    sources.append(ROOT / 'mydata_bench/addbase_eval/references/rewardgen/rewardgen/sole.py')
    return fingerprint({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(sources)})
