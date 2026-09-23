"""Whole-trajectory GRM/SAS attention using the frozen experiment's protocol.

Each sampled time gets a fresh forward input: fixed reference/BEFORE at the
start, synchronized three-view AFTER at that time. The original track terminal
identity is retained: sampling a prefix must not relabel the full-video track.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image

from mydata_bench.basic_method.common import create_json, file_hash, fingerprint, latest
from mydata_bench.io import object_fingerprint
from .render import grid_metrics, vector_to_grid
from .video_render import VideoWriter, comparison_frame, overlay, panel, write_video_gallery

VIEWS = ('front', 'left_wrist', 'right_wrist')


def sample_indices(frame_count, count=30, interval=None):
    if frame_count < 1:
        raise ValueError('Video has no frames')
    if interval is not None:
        if interval < 1:
            raise ValueError('Frame interval must be positive')
        return sorted(set(range(0, frame_count, interval)) | {frame_count - 1})
    if count < 2:
        raise ValueError('Request at least two samples to include both video endpoints')
    return np.unique(np.rint(np.linspace(0, frame_count - 1, min(count, frame_count))).astype(int)).tolist()


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f'Cannot open camera video: {path}')
        info = {'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': float(cap.get(cv2.CAP_PROP_FPS)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        if info['frame_count'] < 1 or not np.isfinite(info['fps']) or info['fps'] <= 0:
            raise ValueError(f'Invalid video timing: {path}: {info}')
        return info
    finally:
        cap.release()


def extract_frames(path, indices, folder):
    """Decode sequentially to avoid approximate codec seeking between samples."""
    folder.mkdir(parents=True, exist_ok=True)
    selected = set(indices)
    cap = cv2.VideoCapture(str(path))
    paths = {}
    try:
        for index in range(max(indices) + 1):
            if not cap.grab():
                raise RuntimeError(f'Camera video ended before requested frame {index}: {path}')
            if index not in selected:
                continue
            ok, pixels = cap.retrieve()
            if not ok or pixels is None:
                raise RuntimeError(f'Cannot decode exact source frame {index}: {path}')
            destination = folder / f'frame_{index:06d}.png'
            if not cv2.imwrite(str(destination), pixels):
                raise OSError(f'Could not save {destination}')
            paths[index] = str(destination.resolve())
    finally:
        cap.release()
    return paths


def trajectory_sample(parent, index, extracted):
    sample = copy.deepcopy(parent)
    if index == parent['sampling']['terminal_source_index']:
        # Keep exactly the cached endpoint pixels for a direct score/input check.
        after = parent['grm_image_paths'][5:8]
    elif index == 0:
        after = [parent['grm_image_paths'][i] for i in (0, 3, 4)]
    else:
        after = [extracted[view][index] for view in VIEWS]
    sample['grm_image_paths'] = parent['grm_image_paths'][:5] + list(after)
    sample['sampling']['selected_source_indices'] = [0, index]
    sample['visualization_source_frame'] = index
    sample['visualization_parent_sample_id'] = fingerprint(parent)
    return sample


def prepare_trajectory(parent, args, folder):
    paths = parent['view_paths']
    info = {view: inspect_video(paths[view]) for view in VIEWS}
    front = info['front']
    terminal = parent['sampling']['terminal_source_index']
    if front['frame_count'] - 1 != terminal:
        raise ValueError('Source video duration differs from the frozen experiment')
    if not np.isclose(front['fps'], parent['sampling']['source_fps'], rtol=.001):
        raise ValueError('Source FPS differs from the frozen experiment')
    for view in VIEWS[1:]:
        if info[view]['frame_count'] <= terminal or not np.isclose(info[view]['fps'], front['fps'], rtol=.001):
            raise ValueError(f'Camera {view} is not synchronized by source index with the front camera')
    hashes = {view: file_hash(paths[view]) for view in VIEWS}
    # This dataset identifies an episode by the combined hashes of all three
    # camera files (see data.load_episodes), not by the front MP4 hash alone.
    if object_fingerprint(hashes) != parent['video_sha256']:
        raise ValueError('Source camera videos differ from the grounding/experiment identity')
    indices = sample_indices(front['frame_count'], args.num_samples or 30, args.frame_interval)
    extracted = {view: extract_frames(paths[view], indices, folder / 'source_frames' / view) for view in VIEWS}
    # Verify source-camera endpoints against frozen inputs, including wrist videos
    # whose complete files were not part of the original input-artifacts manifest.
    for view, first_slot, last_slot in zip(VIEWS, (0, 3, 4), (5, 6, 7)):
        for index, slot in ((0, first_slot), (terminal, last_slot)):
            with Image.open(extracted[view][index]) as decoded, Image.open(parent['grm_image_paths'][slot]) as frozen:
                if not np.array_equal(np.asarray(decoded.convert('RGB')), np.asarray(frozen.convert('RGB'))):
                    raise ValueError(f'Source {view} endpoint pixels differ from frozen input at frame {index}')
    inputs = [trajectory_sample(parent, index, extracted) for index in indices]
    create_json(folder / 'trajectory_inputs.json', inputs)
    return inputs, {'source_videos': {v: {'path': paths[v], 'sha256': hashes[v], **info[v]} for v in VIEWS},
                    'source_indices': indices, 'source_fps': front['fps'], 'source_frame_count': front['frame_count'],
                    'source_duration_seconds': front['frame_count'] / front['fps'],
                    'sampling_policy': 'source_index; fixed BEFORE at 0; synchronized AFTER; endpoints included',
                    'parent_sample_id': fingerprint(parent)}


def token_mask(info):
    from mydata_bench.attention_eval.masking import ImageSpan, bbox_to_token_positions
    if info['bbox'] is None:
        return None
    span = ImageSpan('after_cam_high', info['path'], info['start'], info['end'], tuple(info['grid_thw']))
    mask = np.zeros(info['end'] - info['start'], dtype=bool)
    positions = bbox_to_token_positions(span, info['bbox'], tuple(info['size']))
    mask[np.asarray(positions, dtype=int) - info['start']] = True
    return mask.reshape(info['grid_thw'][1] // 2, info['grid_thw'][2] // 2)


def capture_frame(runtime, sample, args, ranking, observed, folder):
    from .__main__ import capture_pair
    check, mapping, slots, predictions, weights = capture_pair(runtime, sample, args, ranking, observed)
    info = slots['after_cam_high']
    grids = {name: np.stack([vector_to_grid(w[info['start']:info['end']], tuple(info['grid_thw']), 2)
                            for w in weights[name]]) for name in predictions}
    target = token_mask(info)
    index = sample['visualization_source_frame']
    artifact = folder / 'attention' / f'frame_{index:06d}.npz'
    artifact.parent.mkdir(exist_ok=True)
    arrays = {**grids, 'heads': np.asarray([[h['layer'], h['head']] for h in observed]),
              'baseline_rows': weights['baseline'], 'sas_rows': weights['sas']}
    if target is not None:
        arrays['target'] = target
    np.savez_compressed(artifact, **arrays)
    record = {'source_index': index, 'source_time_seconds': index / sample['sampling']['source_fps'],
              'image_path': info['path'], 'image_sha256': file_hash(info['path']), 'bbox': info['bbox'],
              'grounding_check': check, 'sas_applied': check['eligible'], 'baseline_fallback': not check['eligible'],
              'positive_bias': runtime.cfg['bias'] if check['eligible'] else 0.,
              'negative_bias': -runtime.cfg['bias'] if check['eligible'] else 0.,
              'sample_id': fingerprint(sample), 'token_audit': runtime.audit(mapping),
              'predictions': predictions, 'npz': str(artifact.relative_to(folder)), 'npz_sha256': file_hash(artifact),
              'per_head_metrics': {name: [grid_metrics(g, target) for g in grids[name]] for name in predictions}}
    create_json(artifact.with_suffix('.json'), record)
    return record


def rendering_options(args):
    return {key: getattr(args, key) for key in ('fps', 'video_head_index', 'video_scale', 'normalization', 'alpha', 'blur_sigma', 'progress_only')}


def render_clip(source, manifest, output, args):
    """Re-render a clip from raw arrays; never run model inference here."""
    if args.progress_only:
        from .progress import render_progress_clip
        return render_progress_clip(source, manifest, output, args)
    output.mkdir(parents=True, exist_ok=True)
    head_index = args.video_head_index
    observed = manifest['observed_heads']
    if head_index < -1 or head_index >= len(observed):
        raise ValueError('video-head-index is outside the captured heads')
    head = observed[head_index] if head_index >= 0 else None
    head_label = f'L{head["layer"]}H{head["head"]}' if head else f'mean of {len(observed)} heads'
    frames = manifest['frames']
    all_grids, all_metrics = [], []
    for frame in frames:
        path = source / frame['npz']
        if file_hash(path) != frame['npz_sha256'] or file_hash(frame['image_path']) != frame['image_sha256']:
            raise ValueError('Saved video arrays/source frame changed')
        with np.load(path) as saved:
            if not np.array_equal(saved['heads'], [[h['layer'], h['head']] for h in observed]):
                raise ValueError('Saved attention head order differs from manifest')
            target = saved['target'] if 'target' in saved else None
            grids = {c: saved[c][head_index] if head_index >= 0 else saved[c].mean(axis=0) for c in ('baseline', 'sas')}
            all_metrics.append({c: grid_metrics(g, target) for c, g in grids.items()})
            if args.normalization == 'image_fraction':
                grids = {c: g / g.sum() if g.sum() else np.zeros_like(g) for c, g in grids.items()}
            all_grids.append(grids)
    maximum = max(1e-12, max(float(g.max()) for pair in all_grids for g in pair.values()))
    fps = args.fps or len(frames) / manifest['trajectory']['source_duration_seconds']
    scale_label = ('Relative spatial intensity: each map independently normalized; raw mass shown above'
                   if args.video_scale == 'reference' else
                   f'Shared clip scale: 0 - {maximum:.5g} ({args.normalization}); raw mass shown above')
    paths = {c: output / f'{c}.mp4' for c in ('baseline', 'sas', 'comparison')}
    preview_indices = set(sample_indices(len(frames), 4))
    previews, writers = [], {}
    with ExitStack() as stack:
        for i, (frame, grids, metrics) in enumerate(zip(frames, all_grids, all_metrics)):
            with Image.open(frame['image_path']) as im:
                pixels = np.asarray(im.convert('RGB'))
            cards = [panel(pixels, 'Front camera', f'Source frame {frame["source_index"]}  |  {frame["source_time_seconds"]:.2f} s',
                           accent=(160, 174, 196))]
            for condition in ('baseline', 'sas'):
                rendered = overlay(pixels, grids[condition], frame['bbox'], scale=args.video_scale,
                                   maximum=maximum, alpha=args.alpha, blur_sigma=args.blur_sigma)
                progress = frame['predictions'][condition].get('progress')
                progress_text = f'{progress:.3f}' if progress is not None else 'invalid'
                title = ('GRM' if condition == 'baseline' else 'GRM + SAS') + f'  |  progress {progress_text}'
                if condition == 'sas' and frame['baseline_fallback']:
                    title = 'SAS unavailable: baseline fallback'
                m = metrics[condition]
                fraction = f'{m["target_fraction"]:.1%}' if m['target_fraction'] is not None else 'n/a'
                detail = f'Image mass {m["image_mass"]:.4f}  |  target / image {fraction}'
                cards.append(panel(rendered, title, detail,
                                   accent=(90, 173, 250) if condition == 'baseline' else (235, 158, 86)))
            subtitle = (f'{manifest["example_id"]}  |  {head_label}  |  {manifest["scope"]} top-{manifest["top_k"]}'
                        f'  |  sampled {i + 1}/{len(frames)}  |  t={frame["source_time_seconds"]:.2f}s')
            composed = comparison_frame(cards, manifest['task'], subtitle,
                                         frame['source_index'] / max(1, manifest['trajectory']['source_frame_count'] - 1), scale_label)
            outputs = {'baseline': cards[1], 'sas': cards[2], 'comparison': composed}
            for name, picture in outputs.items():
                h, w = picture.shape[:2]
                picture = np.pad(picture, ((0, h % 2), (0, w % 2), (0, 0)), mode='edge')
                if name not in writers:
                    writers[name] = VideoWriter(paths[name], (picture.shape[1], picture.shape[0]), fps)
                    stack.callback(writers[name].close)
                writers[name].write(picture)
            if i in preview_indices:
                previews.append(composed)
            if i == len(frames) // 2:
                Image.fromarray(composed).save(output / 'preview.png')
    Image.fromarray(np.concatenate(previews, axis=0)).save(output / 'contact_sheet.jpg', quality=93)
    result = {'videos': {c: str(p.name) for c, p in paths.items()}, 'num_samples': len(frames),
              'fallback_frames': sum(f['baseline_fallback'] for f in frames),
              'fps': fps, 'duration_seconds': len(frames) / fps, 'observed_head': head_label,
              'options': rendering_options(args), 'shared_maximum': maximum if args.video_scale == 'shared' else None,
              'display_note': scale_label, 'source_manifest': str((source / 'attention_video_manifest.json').resolve())}
    create_json(output / 'render.json', result)
    return result


def gallery_item(output, folder, manifest, result, metadata):
    return {'example_id': manifest['example_id'], 'task': manifest['task'],
            **{name: str((folder / filename).relative_to(output)) for name, filename in result['videos'].items()},
            **{name: str((folder / filename).relative_to(output)) for name, filename in result.get('artifacts', {}).items()},
            'preview': str((folder / 'preview.png').relative_to(output)),
            'metadata': str(metadata.relative_to(output)), 'num_samples': result['num_samples'],
            'fallback_frames': result['fallback_frames']}


def run_videos(args, cfg, identity, run_id, ranking_path, ranking, samples, steered, observed):
    from .__main__ import context, prediction_match
    from mydata_bench.basic_method.runtime import Runtime
    if args.focus_images != ['after_cam_high']:
        raise ValueError('Video mode visualizes after_cam_high; use the default --focus-images')
    if not args.progress_only and not -1 <= args.video_head_index < len(observed):
        raise ValueError('video-head-index is outside the observed heads')
    if not args.preflight and shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is required for MP4 output')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    create_json(output / 'manifest.json', {
        'mode': 'progress_only' if args.progress_only else 'sampled_trajectory', 'source_run_id': run_id, 'config': cfg,
        'source_model_identity': identity['model'], 'ranking_sha256': file_hash(ranking_path),
        'steered_heads': steered, 'observed_heads': observed,
        'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'visualization_sources': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')},
    })
    prepared = []
    for sample in samples:
        slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', sample['example_id']).strip('_')[:100]
        folder = output / (slug + '_' + fingerprint(sample['example_id'])[:8])
        folder.mkdir()
        print(f'Sampling full video: {sample["example_id"]}', flush=True)
        inputs, trajectory = prepare_trajectory(sample, args, folder)
        prepared.append((sample, folder, inputs, trajectory))
    print('Loading GRM processor' + ('' if args.preflight else ' and model'), flush=True)
    runtime = Runtime(cfg, processor_only=args.preflight)
    try:
        if args.preflight:
            report = {}
            for sample, folder, inputs, trajectory in prepared:
                checks = []
                for item in inputs:
                    check, mapping, slots = context(runtime, item, args.scope, args.focus_images)
                    checks.append({'source_index': item['visualization_source_frame'], 'grounding': check,
                                   'sequence_length': mapping['sequence_length'], 'slot': slots['after_cam_high']})
                report[sample['example_id']] = {'trajectory': trajectory, 'frames': checks}
            create_json(output / 'preflight.json', report)
            print(f'Video preflight passed: {output}', flush=True)
            return
        if runtime.num_heads != ranking['num_heads'] or runtime.num_layers != ranking['num_layers']:
            raise ValueError('Model architecture differs from source ranking')
        condition = f'{args.scope}:target:{args.top_k}'
        pred_dir = Path(cfg['output_dir']) / 'predictions'
        cached = {'baseline': latest(pred_dir / 'baseline.jsonl'),
                  'sas': latest(pred_dir / f'{args.scope}_target_{args.top_k}.jsonl')}
        gallery = []
        for parent, folder, inputs, trajectory in prepared:
            records = []
            for i, item in enumerate(inputs, 1):
                if args.progress_only:
                    from .progress import predict_progress_frame
                    record = predict_progress_frame(runtime, item, args, ranking, folder)
                else:
                    record = capture_frame(runtime, item, args, ranking, observed, folder)
                records.append(record)
                label = 'SAS applied' if record['sas_applied'] else 'baseline fallback: ' + record['grounding_check']['reason']
                print(f'{parent["example_id"]} [{i}/{len(inputs)}] frame {record["source_index"]}: {label}', flush=True)
            terminal_comparison = {name: prediction_match(records[-1]['predictions'][name], cached[name].get(parent['example_id']),
                                      parent, run_id, 'baseline' if name == 'baseline' else condition)
                                   for name in ('baseline', 'sas')}
            manifest = {'example_id': parent['example_id'], 'task': parent['task'], 'source_run_id': run_id,
                        'protocol': cfg['protocol'], 'scope': args.scope, 'top_k': args.top_k, 'bias': cfg['bias'],
                        'query_kind': 'last_prompt', 'observed_heads': observed, 'steered_heads': steered,
                        'trajectory': trajectory, 'frames': records, 'terminal_prediction_comparison': terminal_comparison,
                        'interior_frames_are_new_forward_inference': True,
                        'attention_captured': not args.progress_only}
            path = folder / ('progress_video_manifest.json' if args.progress_only else 'attention_video_manifest.json')
            create_json(path, manifest)
            if any(v['matches'] is False for v in terminal_comparison.values()):
                print(f'WARNING: terminal regenerated score differs from source experiment: {parent["example_id"]}', flush=True)
            result = render_clip(folder, manifest, folder, args)
            gallery.append(gallery_item(output, folder, manifest, result, path))
            write_video_gallery(output, gallery)
        create_json(output / 'summary.json', {'complete': True, 'samples': gallery})
        print(f'Video visualization complete: {output / "index.html"}', flush=True)
    finally:
        if runtime.controller is not None:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            runtime.controller.clear()
            ALL_ATTENTION_FUNCTIONS.register('sdpa', runtime.controller.original)


def rerender(args):
    source = args.render_only.resolve()
    names = ('attention_video_manifest.json', 'progress_video_manifest.json')
    manifests = [source / name for name in names if (source / name).is_file()]
    if not manifests:
        manifests = sorted(p for name in names for p in source.glob('*/' + name))
    if not manifests:
        raise FileNotFoundError(f'No saved video inference in {source}')
    if not args.progress_only and any(p.name == 'progress_video_manifest.json' for p in manifests):
        raise ValueError('Saved progress-only runs contain no attention; use --progress-only when re-rendering them')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    gallery = []
    for path in manifests:
        manifest = json.loads(path.read_text())
        folder = output / path.parent.name
        result = render_clip(path.parent, manifest, folder, args)
        # Keep the original manifest beside the re-render, with an explicit link
        # to its unchanged raw arrays instead of claiming those arrays were copied.
        metadata = folder / 'source.json'
        create_json(metadata, {'capture_manifest': str(path), 'capture_sha256': file_hash(path), 'render': result})
        gallery.append(gallery_item(output, folder, manifest, result, metadata))
    write_video_gallery(output, gallery)
    create_json(output / 'summary.json', {'complete': True, 'render_only': True, 'source': str(source), 'samples': gallery})
    print(f'Re-render complete (no model inference): {output / "index.html"}', flush=True)
