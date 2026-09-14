"""Create a fresh matrix; reuse image pixels read-only, never old bbox caches."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

from mydata_bench.data import load_episodes
from mydata_bench.io import read_jsonl
from mydata_bench.video import extract_frame_at, extract_uniform_image_sequence
from mydata_bench.top_eval.versioning import SOLE_PROTOCOL_VERSION
from .common import (ROOT, OUT, CONFIGS, MODELS, PROTOCOLS, VERSION, create_json,
                     create_text, file_hash, fingerprint, media_hashes)
from .grounding import eligibility, track_boxes, valid_box

DATA = Path('/home/dais/workspace/data/mydata_v2/new')
RELEASE = ROOT / 'results/mydata_bench/grounding_v2_release'
RANK_RELEASE = ROOT / 'results/mydata_bench/ranking_grounding_v2_release'
COHORT = ROOT / 'results/mydata_bench/cohorts/auto_grounded_v2_release/example_ids.json'


def release_rows(folder):
    rows = {}
    for row in read_jsonl(folder / 'sam3/grounding.jsonl'):
        key = row['example_id'], row['frame']
        if key in rows:
            raise ValueError(f'Duplicate release row: {key}')
        rows[key] = row
    targets = {}
    for row in read_jsonl(folder / 'targets.jsonl'):
        if row['example_id'] in targets:
            raise ValueError('Duplicate target ID')
        targets[row['example_id']] = row
    return rows, targets


def attach_grounding(sample, rows, targets, allow_ids, source):
    eid = sample['example_id']
    endpoints = [rows.get((eid, frame)) for frame in ('first', 'last')]
    eligible = eid in allow_ids
    descriptor = {'eligible': eligible, 'release': str(source),
                  'target_sha256': fingerprint(targets[eid])}
    for row in endpoints:
        if row is None or row['video_sha256'] != sample['video_sha256']:
            raise ValueError(f'Release/input identity mismatch: {eid}')
        if row['provenance']['task'] != sample['task']:
            raise ValueError(f'Release/input instruction mismatch: {eid}')
    descriptor['endpoint_sha256'] = fingerprint(endpoints)
    if eligible:
        for row in endpoints:
            if row['status'] != 'ok' or not valid_box(row['bbox']):
                raise ValueError(f'Eligible release ID has invalid endpoint: {eid}')
            for path in (row['provenance']['image_path'], row.get('mask_path')):
                if path and not Path(path).is_file():
                    raise FileNotFoundError(path)
        path = endpoints[-1]['provenance']['tracking_path']
        descriptor.update(tracking_path=path, tracking_sha256=file_hash(path))
    sample = {**sample, 'cohort': eligible, 'grounding': descriptor}
    if eligible:
        track_boxes(sample)
    return sample


def build_configs(out, configs):
    matrix = []
    for model in MODELS:
        model_path = (ROOT / 'pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview' if model == 'grm'
                      else Path('/home/dais/workspace/model') / {
                          'roboreward': 'RoboReward-8B', 'qwen': 'Qwen3-VL-8B-Instruct',
                          'meter': 'Robometer-4B', 'sole': 'SOLE-R1-8B'}[model])
        for protocol in PROTOCOLS:
            cfg = {'basic_method_version': VERSION, 'model': model, 'protocol': protocol,
                   'num_frames': 8, 'min_pixels': 1024, 'max_pixels': 50176,
                   'max_new_tokens': {'grm': 32, 'roboreward': 5000, 'qwen': 32,
                                      'meter': 1, 'sole': 512}[model],
                   'skip_early_layers': 8, 'bias': 6.0, 'query_scope': 'all',
                   'scopes': ['last_frame', 'all_frames'], 'top_k': [8, 32, 64],
                   'controls': ['wrong_region', 'low_rank'], 'seed': 20260913,
                   'decoding': 'greedy', 'model_path': str(model_path),
                   'processor_path': str(Path('/home/dais/workspace/model/Qwen3-VL-4B-Instruct')
                                         if model == 'meter' else model_path),
                   'output_dir': str(out / f'{model}_{protocol}')}
            if model == 'grm':
                cfg.update(min_pixels=12544, max_pixels=76800, eval_mode='forward',
                           grm_all_frame_slots=['reference_start', 'before_cam_high', 'after_cam_high'])
            if protocol == 'official':
                if model == 'meter':
                    cfg['max_pixels'] = 16777216
                if model == 'sole':
                    cfg.update(max_pixels=12845056, sole_protocol_version=SOLE_PROTOCOL_VERSION)
                if model in ('qwen', 'roboreward'):
                    cfg['sampling'] = 'checkpoint_native_mp4'
            for field, name in [('inputs', 'inputs.json'), ('ranking_inputs', 'ranking_inputs.json'),
                                ('labels', 'labels_for_scoring_only.json')]:
                cfg[field] = str(out / name)
                cfg[field + '_sha256'] = file_hash(out / name)
            path = configs / f'{model}_{protocol}.yaml'
            create_text(path, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            matrix.append(str(path))
    return matrix


def prepare(out=OUT, configs=CONFIGS, data=DATA):
    out, configs, data = Path(out).resolve(), Path(configs).resolve(), Path(data).resolve()
    rows, targets = release_rows(RELEASE)
    rank_rows, rank_targets = release_rows(RANK_RELEASE)
    cohort_list = json.loads(COHORT.read_text())
    if len(cohort_list) != len(set(cohort_list)):
        raise ValueError('Duplicate cohort ID')
    cohort = set(cohort_list)
    episodes = list(load_episodes(data, 'all'))
    ids = {e.example_id for e in episodes}
    if len(ids) != len(episodes) or not cohort <= ids:
        raise ValueError('Evaluation/cohort ID mismatch')
    ranking = list(load_episodes(data, 'suc', metadata_file=data / 'ranking_data.jsonl'))
    ranking_ids = {e.example_id for e in ranking}
    if not ranking_ids <= ids:
        raise ValueError('Ranking metadata must reference the same full dataset')
    ranking_hashes = {e.video_sha256 for e in ranking}
    rank_eligible = {eid for eid in ranking_ids if all(
        rank_rows.get((eid, frame), {}).get('status') == 'ok' for frame in ('first', 'last'))}
    old_path = ROOT / 'results/mydata_bench/experiments_v2_addbase/inputs.json'
    old = {r['example_id']: r for r in json.loads(old_path.read_text())} if old_path.exists() else {}
    # Pixel-only reuse is checked against this episode's actual video digest.
    image_cache, records, labels = {}, [], {}
    for index, episode in enumerate(episodes):
        eid, digest = episode.example_id, episode.video_sha256
        if digest not in image_cache:
            cached = old.get(eid, {})
            if (cached.get('video_sha256') == digest and len(cached.get('image_paths', [])) == 8
                    and all(Path(p).is_file() for p in cached['image_paths'])):
                paths, sampling = cached['image_paths'], dict(cached['sampling'])
            else:
                paths, sampling = extract_uniform_image_sequence(episode.video_path, out / 'frames' / digest, count=8)
            image_cache[digest] = paths, sampling
        paths, sampling = image_cache[digest]
        sampling = {**sampling, 'source_video_path': episode.video_path}
        views = dict(episode.view_paths)
        terminal = sampling['terminal_source_index']
        view_images = {'front': {'first': paths[0], 'last': paths[-1]}}
        for view in ('left_wrist', 'right_wrist'):
            view_images[view] = {}
            for frame, source_index in [('first', 0), ('last', terminal)]:
                endpoint = rows[(eid, frame)]['provenance'].get('view_endpoint_paths', {}).get(view, {})
                path = endpoint.get(frame)
                if not path or not Path(path).is_file() or endpoint.get(frame + '_index') != source_index:
                    _, path = extract_frame_at(views[view], out / 'frames' / digest / view / f'{source_index}.png', source_index)
                view_images[view][frame] = path
        sample = {'example_id': eid, 'task': episode.task, 'subset': episode.subset,
                  'video_sha256': digest, 'video_path': episode.video_path, 'view_paths': views,
                  'image_paths': paths, 'sampling': sampling, 'holdout': digest not in ranking_hashes,
                  'grm_image_paths': [paths[0], str(ROOT / 'examples/blank_goal.png'),
                                      *[view_images[v]['first'] for v in ('front', 'left_wrist', 'right_wrist')],
                                      *[view_images[v]['last'] for v in ('front', 'left_wrist', 'right_wrist')]]}
        records.append(attach_grounding(sample, rows, targets, cohort, RELEASE))
        labels[eid] = {'reward': episode.reward, 'split': episode.split, 'subset': episode.subset,
                       'source_suc_id': episode.source_suc_id, 'video_sha256': digest}
        if (index + 1) % 100 == 0:
            print(f'Prepared {index + 1}/{len(episodes)}', flush=True)
    by_id = {r['example_id']: r for r in records}
    ranking_records = []
    for e in ranking:
        sample = by_id[e.example_id]
        if sample['task'] != e.task or sample['video_sha256'] != e.video_sha256:
            raise ValueError('Ranking/evaluation input mismatch')
        ranking_records.append(attach_grounding(sample, rank_rows, rank_targets, rank_eligible, RANK_RELEASE))
    create_json(out / 'inputs.json', records)
    create_json(out / 'ranking_inputs.json', ranking_records)
    create_json(out / 'labels_for_scoring_only.json', labels)
    print('Freezing image/video content hashes', flush=True)
    create_json(out / 'input_artifacts.json', media_hashes([
        path for sample in records for path in [sample['video_path'], *sample['image_paths'], *sample['grm_image_paths']]]))
    matrix = build_configs(out, configs)
    create_json(out / 'matrix.json', matrix)
    coverage = {}
    for cfg_path in matrix:
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        populations = {}
        native = cfg['model'] in ('roboreward', 'qwen') and cfg['protocol'] == 'official'
        for name, samples in [('evaluation', records), ('ranking', ranking_records)]:
            populations[name] = {}
            for scope in cfg['scopes']:
                if native:
                    populations[name][scope] = {'pending': 'actual processor sampling checked before SAS'}
                    continue
                checks = [eligibility(s, cfg, scope) for s in samples]
                populations[name][scope] = {'total': len(checks), 'eligible': sum(c['eligible'] for c in checks),
                                            'reasons': dict(Counter(c['reason'] for c in checks if not c['eligible']))}
        coverage[Path(cfg_path).stem] = populations
    inventory = {'total': len(records), 'release_eligible': len(cohort), 'ranking_total': len(ranking_records),
                 'ranking_release_eligible': len(rank_eligible), 'unique_videos': len(image_cache),
                 'ranking_video_groups_in_full': sorted(ranking_hashes),
                 'holdout_total': sum(s['holdout'] for s in records), 'baseline_conditions': len(matrix),
                 'steering_conditions': len(matrix) * 2 * 3 * 3, 'bias': 6.0,
                 'grounding_sources': {str(p): file_hash(p) for p in
                     [COHORT, RELEASE / 'sam3/grounding.jsonl', RELEASE / 'targets.jsonl',
                      RANK_RELEASE / 'sam3/grounding.jsonl', RANK_RELEASE / 'targets.jsonl',
                      data / 'metadata.jsonl', data / 'ranking_data.jsonl']}}
    create_json(out / 'grounding_coverage.json', coverage)
    create_json(out / 'inventory.json', inventory)
    print(json.dumps({k: v for k, v in inventory.items() if k not in ('grounding_sources', 'ranking_video_groups_in_full')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--config-dir', type=Path, default=CONFIGS)
    parser.add_argument('--dataset-root', type=Path, default=DATA)
    args = parser.parse_args()
    prepare(args.output_dir, args.config_dir, args.dataset_root)


if __name__ == '__main__':
    main()
