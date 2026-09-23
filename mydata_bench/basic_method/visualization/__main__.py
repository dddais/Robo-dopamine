"""Visualize frozen basic_method GRM baseline versus SAS on front-camera slots.

Run ``python -m mydata_bench.basic_method.visualization --help`` for examples.
Existing experiments are only read; all diagnostic files go to a new directory.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np
import yaml

from mydata_bench.basic_method.common import (
    CONFIGS, create_json, file_hash, fingerprint, implementation_identity, latest,
    validate_config, validate_inputs, validate_media,
)
from mydata_bench.basic_method.grounding import (
    MissingGrounding, eligibility, exact_boxes, sample_identity,
)
from mydata_bench.io import artifact_fingerprint

FRONT_SLOTS = {'reference_start': 0, 'before_cam_high': 2, 'after_cam_high': 5}


def select_samples(samples, args):
    directory = args.sample_dir
    if args.start_id is not None or args.end_id is not None:
        if directory is None:
            raise ValueError('--start-id/--end-id require --sample-dir')
        if any(v is not None and v < 0 for v in (args.start_id, args.end_id)):
            raise ValueError('Sample IDs must be nonnegative integers')
        if args.start_id is not None and args.end_id is not None and args.start_id > args.end_id:
            raise ValueError('--start-id must be <= --end-id')
    if directory is not None:
        if args.example_id:
            raise ValueError('--sample-dir cannot be combined with --example-id')
        directory = directory.strip().removeprefix('./').rstrip('/')
        if not directory or directory.startswith('/') or any(p in ('', '.', '..') for p in directory.split('/')):
            raise ValueError('--sample-dir must be a dataset-relative directory, e.g. fail/ljx_lfz_task_1_3/')
        numbered = []
        for sample in samples:
            parent, _, number = sample['example_id'].rpartition('/')
            # Exact directory boundary and direct numeric children only.
            if parent != directory or re.fullmatch(r'[0-9]+', number) is None:
                continue
            value = int(number)
            if (args.start_id is None or value >= args.start_id) and (args.end_id is None or value <= args.end_id):
                numbered.append((value, sample['example_id'], sample))
        samples = [s for _, _, s in sorted(numbered, key=lambda row: (row[0], row[1]))]
    by_id = {s['example_id']: s for s in samples}
    if args.example_id:
        unknown = set(args.example_id) - set(by_id)
        if unknown:
            raise ValueError(f'Unknown example IDs: {sorted(unknown)}')
        samples = [by_id[eid] for eid in dict.fromkeys(args.example_id)]
    selected = [s for s in samples if (args.subset is None or s['subset'] == args.subset)
                and (args.split is None or s['example_id'].startswith(args.split + '/'))
                and (not args.holdout_only or s['holdout'])]
    limit = args.limit if args.limit is not None else (0 if args.example_id or directory is not None else 4)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError('No samples match the filters')
    return selected


def choose_heads(ranking, top_k, spec):
    steered = ranking['ranking'][:top_k]
    if len(steered) != top_k:
        raise ValueError('Not enough independently ranked heads')
    if spec is None:
        observed = steered
    else:
        observed = []
        for token in spec.split(','):
            match = re.fullmatch(r'L(\d+)H(\d+)', token.strip(), flags=re.IGNORECASE)
            if not match:
                raise ValueError('--heads must look like L19H23,L20H4 (zero-based)')
            layer, head = map(int, match.groups())
            observed.append({'layer': layer, 'head': head})
    for group in (steered, observed):
        pairs = [(int(h['layer']), int(h['head'])) for h in group]
        if len(set(pairs)) != len(pairs) or any(
                not 0 <= l < ranking['num_layers'] or not 0 <= h < ranking['num_heads'] for l, h in pairs):
            raise ValueError('Duplicate or out-of-range heads')
    return steered, observed


def load_experiment(args):
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    if cfg['model'] != 'grm':
        raise ValueError('This visualization entry point supports GRM configs only')
    if args.scope not in cfg['scopes'] or args.top_k not in cfg['top_k']:
        raise ValueError('Requested scope/top-k is not in the frozen experiment config')
    validate_inputs(cfg)
    root = Path(cfg['output_dir'])
    identity = json.loads((root / 'run_identity.json').read_text())
    if identity['config'] != cfg or json.loads((root / 'run_config.json').read_text()) != cfg:
        raise ValueError('Config differs from the source experiment')
    if identity['implementation'] != implementation_identity():
        raise ValueError('Inference implementation differs from the source experiment')
    run_id = fingerprint(identity)
    ranking_path = root / 'independent_ranking' / f'ranking_{args.scope}.json'
    ranking = json.loads(ranking_path.read_text())
    if ranking['run_id'] != run_id or ranking['scope'] != args.scope:
        raise ValueError('Independent ranking belongs to a different experiment/scope')
    samples = select_samples(json.loads(Path(cfg['inputs']).read_text()), args)
    if set(s['example_id'] for s in samples) - set(identity['evaluation_ids']):
        raise ValueError('Selected samples are outside the source run')
    steered, observed = choose_heads(ranking, args.top_k, args.heads)
    return cfg, identity, run_id, ranking_path, ranking, samples, steered, observed


def context(runtime, sample, scope, focus):
    check = eligibility(sample, runtime.cfg, scope)
    batch, mapping, query, _ = runtime.prepare(sample, scope if check['eligible'] else None)
    slots = {}
    for label in focus:
        index = FRONT_SLOTS[label]
        record = mapping['records'][index]
        span = record['span']
        if len(record['sources']) != 1 or span.grid_thw[0] != 1:
            raise ValueError('Expected a single exact front-camera source frame per GRM slot')
        source = record['sources'][0]
        box, box_reason = None, None
        try:
            box = exact_boxes(sample, [source])[source]
        except MissingGrounding as exc:
            box_reason = exc.reason
        slots[label] = {'path': sample['grm_image_paths'][index], 'source_frame': source,
                        'start': span.start, 'end': span.end, 'grid_thw': list(span.grid_thw),
                        'bbox': box, 'bbox_unavailable_reason': box_reason, 'size': record['size']}
    if query != batch['input_ids'].shape[1] - 1:
        raise ValueError('GRM visualization expects the last prompt query')
    return check, mapping, slots


def prediction_match(row, saved, sample, run_id, condition):
    if saved is None:
        return {'available': False, 'matches': None}
    if (saved.get('run_id') != run_id or saved.get('sample_id') != sample_identity(sample)
            or saved.get('condition') != condition):
        raise ValueError('Saved prediction identity mismatch')
    for field in ('prompt_sha256', 'input_ids_sha256'):
        if saved['token_audit'][field] != row['token_audit'][field]:
            raise ValueError(f'Visualization input differs from cached prediction: {field}')
    fields = ('status', 'raw_output', 'progress')
    return {'available': True, 'matches': all(row.get(k) == saved.get(k) for k in fields),
            'saved': {k: saved.get(k) for k in fields}, 'saved_row_sha256': fingerprint(saved)}


def capture_pair(runtime, sample, args, ranking, observed):
    """Run one forward-protocol input pair without rendering or touching caches."""
    from .capture import LastPromptCapture

    check, mapping, slots = context(runtime, sample, args.scope, args.focus_images)
    condition = f'{args.scope}:target:{args.top_k}'
    rows, weights = {}, {}
    for label, actual in [('baseline', 'baseline'), ('sas', condition)]:
        if label == 'sas' and not check['eligible']:
            rows[label], weights[label] = dict(rows['baseline']), weights['baseline'].copy()
            rows[label].update(condition=condition, baseline_fallback=True, sas_applied=False,
                               fallback_reason=check['reason'], positive_bias=0., negative_bias=0.)
            continue
        with LastPromptCapture(runtime.controller, observed, mapping['sequence_length']) as capture:
            rows[label] = runtime.predict(sample, actual, ranking)
        weights[label] = capture.result()
        for field in ('prompt_sha256', 'input_ids_sha256'):
            if rows[label]['token_audit'][field] != mapping[field]:
                raise ValueError(f'Captured input drifted: {field}')
    return check, mapping, slots, rows, weights


def visualize_sample(runtime, sample, args, ranking, observed, cached, run_id, output):
    from mydata_bench.attention_eval.masking import ImageSpan, bbox_to_token_positions
    from .render import grid_metrics, save_comparison, vector_to_grid

    check, mapping, slots, rows, weights = capture_pair(runtime, sample, args, ranking, observed)
    condition = f'{args.scope}:target:{args.top_k}'
    comparisons = {name: prediction_match(rows[name], cached[name].get(sample['example_id']), sample,
                                         run_id, 'baseline' if name == 'baseline' else condition)
                   for name in rows}
    safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', sample['example_id']).strip('_')[:100]
    folder = output / (safe_id + '_' + fingerprint(sample['example_id'])[:8])
    folder.mkdir()
    arrays = {'heads': np.array([[h['layer'], h['head']] for h in observed], dtype=np.int64),
              'baseline_rows': weights['baseline'], 'sas_rows': weights['sas']}
    metrics, figures = {}, []
    actual_bias = runtime.cfg['bias'] if check['eligible'] else 0.
    status = f'SAS {args.scope}, top-{args.top_k}, bias ±{actual_bias:g}' if check['eligible'] else (
        'Baseline fallback: ' + check['reason'])
    for slot, info in slots.items():
        grids = {name: np.stack([vector_to_grid(w[info['start']:info['end']], tuple(info['grid_thw']), 2)
                                for w in weights[name]]) for name in rows}
        for name, grid in grids.items():
            arrays[f'{name}_{slot}'] = grid
        target = None
        if info['bbox'] is not None:
            span = ImageSpan(slot, info['path'], info['start'], info['end'], tuple(info['grid_thw']))
            target = np.zeros(info['end'] - info['start'], dtype=bool)
            positions = bbox_to_token_positions(span, info['bbox'], tuple(info['size']))
            target[np.array(positions, dtype=int) - info['start']] = True
            target = target.reshape(grids['baseline'].shape[1:])
            arrays[f'target_{slot}'] = target
        choices = [('mean', f'mean of {len(observed)} observed heads', None)] + [
            (f'L{h["layer"]}H{h["head"]}', f'L{h["layer"]}H{h["head"]}', i)
            for i, h in enumerate(observed[:args.per_head])]
        metrics[slot] = {'per_head': {
            f'L{h["layer"]}H{h["head"]}': {name: grid_metrics(grids[name][i], target) for name in rows}
            for i, h in enumerate(observed)}}
        for name, description, i in choices:
            filename = f'{slot}_{name}.png'
            paired = [grids[c].mean(axis=0) if i is None else grids[c][i] for c in ('baseline', 'sas')]
            subtitles = [f'GRM baseline | p={rows["baseline"].get("progress")}',
                         f'GRM + SAS | p={rows["sas"].get("progress")}' if check['eligible'] else 'Baseline fallback']
            metrics[slot][name] = save_comparison(
                folder / filename, info['path'], *paired, info['bbox'], target,
                f'{sample["example_id"]} | {sample["task"]}\n{slot} / source {info["source_frame"]} | '
                f'{description} | {status} | last prompt query', subtitles, args.normalization, args.alpha)
            figures.append(str((folder / filename).relative_to(output)))
    np.savez_compressed(folder / 'attention.npz', **arrays)
    metadata = {'example_id': sample['example_id'], 'sample_id': sample_identity(sample), 'task': sample['task'],
                'condition': condition, 'query_kind': 'last_prompt', 'query_position': mapping['query'],
                'sas_applied': check['eligible'], 'baseline_fallback': not check['eligible'],
                'positive_bias': actual_bias, 'negative_bias': -actual_bias, 'grounding_check': check,
                'observed_heads': observed, 'slots': slots, 'metrics': metrics,
                'predictions': rows, 'saved_prediction_comparison': comparisons,
                'probability_method': 'float32 QK softmax after actual SAS bias and causal mask',
                'token_audit': runtime.audit(mapping)}
    create_json(folder / 'metadata.json', metadata)
    mismatch = any(v['matches'] is False for v in comparisons.values())
    return {'example_id': sample['example_id'], 'task': sample['task'], 'status': status + (
                ' | WARNING: regenerated score differs from saved experiment' if mismatch else ''),
            'saved_predictions_match': {k: v['matches'] for k, v in comparisons.items()},
            'figures': figures, 'metadata': str((folder / 'metadata.json').relative_to(output))}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=CONFIGS / 'grm_official.yaml')
    ap.add_argument('--scope', choices=['last_frame', 'all_frames'], default='last_frame')
    ap.add_argument('--top-k', type=int, default=8, help='SAS intervention heads from the existing ranking')
    ap.add_argument('--heads', help='Optional observed heads, e.g. L19H23,L20H4; does not change SAS heads')
    ap.add_argument('--focus-images', nargs='+', choices=list(FRONT_SLOTS), default=['after_cam_high'])
    ap.add_argument('--per-head', type=int, default=2, help='Also render this many individual observed heads; 0 = mean only')
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument('--example-id', action='append', help='Exact example ID; repeat to select several')
    selection.add_argument('--sample-dir', help='Dataset-relative directory, e.g. fail/ljx_lfz_task_1_3/; select all matching numeric IDs')
    ap.add_argument('--start-id', type=int, help='Inclusive starting sample number within --sample-dir')
    ap.add_argument('--end-id', type=int, help='Inclusive ending sample number within --sample-dir')
    ap.add_argument('--subset', help='Task subset, e.g. task1_1')
    ap.add_argument('--split', choices=['suc', 'fail'])
    ap.add_argument('--holdout-only', action='store_true')
    ap.add_argument('--limit', type=int, help='Default: all matches with sample-dir/explicit IDs, otherwise first 4; 0 = all matches')
    ap.add_argument('--normalization', choices=['raw', 'image_fraction'], default='raw')
    ap.add_argument('--alpha', type=float, default=.5)
    ap.add_argument('--output-dir', type=Path, help='Must be a new directory')
    ap.add_argument('--list-samples', action='store_true', help='Print matching IDs and exact-frame eligibility without loading the processor/model')
    ap.add_argument('--preflight', action='store_true', help='Validate inputs and token geometry with the processor only; no GPU')
    ap.add_argument('--video', action='store_true', help='Sample each entire three-camera video and export smooth attention MP4s')
    ap.add_argument('--progress-only', action='store_true',
                    help='Output original front video and GRM/SAS progress curves; skip attention capture/videos (implies --video)')
    sampling = ap.add_mutually_exclusive_group()
    sampling.add_argument('--num-samples', '--video-num-samples', type=int,
                          help='Uniformly sample this many video frames including both endpoints; default 30')
    sampling.add_argument('--frame-interval', type=int, help='Sample every N source frames, also include the last frame')
    ap.add_argument('--fps', type=float, default=5., help='Output playback FPS; 0 preserves approximate source duration')
    ap.add_argument('--video-head-index', type=int, default=0,
                    help='Index in observed heads to render in videos; -1 = mean; default 0, like the reference script')
    ap.add_argument('--video-scale', choices=['reference', 'shared'], default='reference',
                    help='reference: independently min/max each map for spatial detail; shared: one scale for both conditions over the clip')
    ap.add_argument('--blur-sigma', type=float, default=3., help='Spatial Gaussian blur in source-image pixels after bicubic resizing; 0 disables')
    ap.add_argument('--render-only', type=Path, metavar='VIDEO_OUTPUT',
                    help='Re-render saved video predictions/attention; no model/GPU, requires a new --output-dir')
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.progress_only:
        args.video = True
    if args.per_head < 0 or args.limit is not None and args.limit < 0 or not 0 <= args.alpha <= 1:
        ap.error('per-head/limit must be nonnegative and alpha must be in [0, 1]')
    if (not np.isfinite(args.fps) or args.fps < 0 or not np.isfinite(args.blur_sigma) or args.blur_sigma < 0
            or args.num_samples is not None and args.num_samples < 2
            or args.frame_interval is not None and args.frame_interval < 1):
        ap.error('fps/blur-sigma must be finite and nonnegative; num-samples >= 2; frame-interval >= 1')
    if args.render_only is not None:
        if args.output_dir is None or args.preflight or args.list_samples:
            ap.error('--render-only requires --output-dir and cannot be combined with preflight/list-samples')
        if args.sample_dir is not None or args.start_id is not None or args.end_id is not None:
            ap.error('--sample-dir/--start-id/--end-id select inference inputs and cannot be used with --render-only')
        from .video import rerender
        rerender(args)
        return
    cfg, identity, run_id, ranking_path, ranking, samples, steered, observed = load_experiment(args)
    if args.list_samples:
        for sample in samples:
            check = eligibility(sample, cfg, args.scope)
            print(f'{sample["example_id"]}\t{sample["subset"]}\t'
                  f'{"eligible" if check["eligible"] else "fallback:" + check["reason"]}\t{sample["task"]}')
        return
    if args.output_dir is None:
        ap.error('--output-dir is required unless --list-samples is used')
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new visualization directory: {output}')
    media = validate_media(cfg, samples)
    for name in ('model', 'processor'):
        if artifact_fingerprint(cfg[name + '_path']) != identity[name]:
            raise ValueError(f'{name} artifacts differ from the source experiment')
    versions = {p: importlib.metadata.version(p) for p in identity['versions']}
    if versions != identity['versions']:
        raise ValueError(f'Use the original experiment environment: {identity["versions"]}; found {versions}')
    if args.video:
        from .video import run_videos
        run_videos(args, cfg, identity, run_id, ranking_path, ranking, samples, steered, observed)
        return
    from .render import write_gallery
    from mydata_bench.basic_method.runtime import Runtime

    output.mkdir(parents=True, exist_ok=False)
    manifest = {'config': cfg, 'source_run_id': run_id, 'ranking_path': str(ranking_path),
                'ranking_sha256': file_hash(ranking_path), 'steered_heads': steered, 'observed_heads': observed,
                'example_ids': [s['example_id'] for s in samples], 'input_media_sha256': media,
                'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'visualization_sources': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')},
                'reference_probe_sha256': file_hash(Path(__file__).resolve().parents[3] / 'scan_localization_heads_best.py'),
                'versions': versions}
    create_json(output / 'manifest.json', manifest)
    print('Loading GRM processor' + ('' if args.preflight else ' and model'), flush=True)
    runtime = Runtime(cfg, processor_only=args.preflight)
    try:
        if args.preflight:
            audits = {}
            for sample in samples:
                check, mapping, slots = context(runtime, sample, args.scope, args.focus_images)
                audits[sample['example_id']] = {'grounding_check': check, 'slots': slots, 'token_audit': runtime.audit(mapping)}
            create_json(output / 'preflight.json', audits)
            print(f'Preflight passed for {len(samples)} samples: {output}', flush=True)
            return
        if runtime.num_heads != ranking['num_heads'] or runtime.num_layers != ranking['num_layers']:
            raise ValueError('Model architecture differs from the independent ranking')
        pred_dir = Path(cfg['output_dir']) / 'predictions'
        cached = {'baseline': latest(pred_dir / 'baseline.jsonl'),
                  'sas': latest(pred_dir / f'{args.scope}_target_{args.top_k}.jsonl')}
        items = []
        for i, sample in enumerate(samples, 1):
            item = visualize_sample(runtime, sample, args, ranking, observed, cached, run_id, output)
            items.append(item)
            write_gallery(output, items, f'GRM vs SAS · {cfg["protocol"]} · {args.scope} · top-{args.top_k}')
            print(f'[{i}/{len(samples)}] {sample["example_id"]}: {item["status"]}', flush=True)
        create_json(output / 'summary.json', {'samples': items, 'complete': True})
        print(f'Visualization complete: {output / "index.html"}', flush=True)
    finally:
        if runtime.controller is not None:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            runtime.controller.clear()
            ALL_ATTENTION_FUNCTIONS.register('sdpa', runtime.controller.original)


if __name__ == '__main__':
    main()
