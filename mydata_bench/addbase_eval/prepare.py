"""Freeze inputs and experiment matrix without changing existing artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from mydata_bench.data import load_episodes
from mydata_bench.io import read_jsonl
from mydata_bench.video import extract_uniform_image_sequence

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/mydata_bench/experiments_v2_addbase'
DATA = Path('/home/dais/workspace/data/mydata_v2/new')
CONFIGS = ROOT / 'mydata_bench/configs/v2_crossmodel_addbase'


def create_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f'Refusing to replace existing artifact: {path}')
        return
    with path.open('x') as f:
        f.write(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    cohorts = set(json.loads((ROOT / 'results/mydata_bench/cohorts/auto_grounded_v2/example_ids.json').read_text()))
    grounds = {}
    for row in read_jsonl(ROOT / 'results/mydata_bench/grounding_v2/sam3/grounding.jsonl'):
        if row.get('status') == 'ok':
            grounds.setdefault(row['example_id'], {})[row['frame']] = row
    ranking_ids = {e.example_id for e in load_episodes(DATA, 'suc', metadata_file=DATA/'ranking_data.jsonl', compute_hash=False)}
    episodes = list(load_episodes(DATA, 'all'))
    ranking_hashes = {e.video_sha256 for e in episodes if e.example_id in ranking_ids}
    records, labels, frame_cache = [], {}, {}
    for i, e in enumerate(episodes):
        if e.video_sha256 not in frame_cache:
            # Reuse existing images read-only when an exact terminal-inclusive cache exists.
            cached = ROOT / 'results/mydata_bench/experiments_v2_corssmodel/attention_13_roboreward_interleaved_all_frames/image_sequences/cohort' / e.video_sha256
            if len(list(cached.glob('image_*_source_*.png'))) != 8:
                cached = OUT / 'frames' / e.video_sha256
            frame_cache[e.video_sha256] = extract_uniform_image_sequence(e.video_path, cached, count=8)
        paths, sampling = frame_cache[e.video_sha256]
        row = {'example_id': e.example_id, 'video_path': e.video_path, 'video_sha256': e.video_sha256,
               'task': e.task, 'subset': e.subset, 'image_paths': paths, 'sampling': sampling,
               'cohort': e.example_id in cohorts, 'ranking': e.example_id in ranking_ids and e.example_id in cohorts,
               'holdout': e.video_sha256 not in ranking_hashes}
        if row['cohort']:
            g = grounds[e.example_id]
            row.update(first_bbox=g['first']['bbox'], last_bbox=g['last']['bbox'],
                       tracking_path=g['last']['provenance']['tracking_path'])
        records.append(row)
        labels[e.example_id] = {'reward': e.reward, 'split': e.split, 'subset': e.subset,
                               'source_suc_id': e.source_suc_id, 'video_sha256': e.video_sha256}
        if (i+1) % 100 == 0:
            print(f'Prepared {i+1}/{len(episodes)}', flush=True)
    create_json(OUT/'inputs.json', records)
    create_json(OUT/'labels_for_scoring_only.json', labels)
    create_json(OUT/'inventory.json', {'total': len(records), 'cohort': sum(r['cohort'] for r in records),
                'ranking': sum(r['ranking'] for r in records), 'unique_videos': len(frame_cache),
                'holdout_cohort': sum(r['cohort'] and r['holdout'] for r in records),
                'ranking_group_exclusion': sorted(ranking_hashes), 'seed': 20260909})
    matrix = []
    for model in ['meter', 'sole']:
        for protocol in ['video_text', 'text_video', 'image_text', 'text_image', 'interleaved', 'official']:
            cfg = {'model': model, 'protocol': protocol, 'num_frames': 8, 'min_pixels': 1024,
                   'max_pixels': 50176, 'max_new_tokens': 512, 'batch_size': 8 if model == 'sole' else 4,
                   'skip_early_layers': 8, 'ranking_score': 'raw_mass', 'bias': 6.0,
                   'query_scope': 'all', 'top_k': [8, 32, 64], 'scopes': ['last_frame', 'all_frames'],
                   'controls': ['wrong_region', 'low_rank'], 'seed': 20260909,
                   'inputs': str(OUT/'inputs.json'), 'output_dir': str(OUT/f'{model}_{protocol}'),
                   'model_path': f'/home/dais/workspace/model/{"Robometer-4B" if model == "meter" else "SOLE-R1-8B"}',
                   'processor_path': f'/home/dais/workspace/model/{"Qwen3-VL-4B-Instruct" if model == "meter" else "SOLE-R1-8B"}',
                   'decoding': 'greedy', 'negative_scope': 'selected_temporal_frames'}
            # The official Robometer contract includes native image sizing and per-frame tokens.
            if protocol == 'official':
                cfg['max_pixels'] = 16777216 if model == 'meter' else 12845056
                cfg['batch_size'] = 1 if model == 'meter' else 8
            path = CONFIGS / f'{model}_{protocol}.yaml'
            serial = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            if path.exists():
                if path.read_text() != serial:
                    raise FileExistsError(path)
            else:
                with path.open('x') as f: f.write(serial)
            matrix.append(str(path))
    create_json(OUT/'matrix.json', matrix)
    print(json.dumps(json.loads((OUT/'inventory.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    main()
