"""Model-specific prompts, precise frame/token geometry and official inputs."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import json

import numpy as np
from PIL import Image
from qwen_vl_utils import smart_resize

from mydata_bench.attention_eval.masking import ImageSpan, bbox_to_token_positions
from mydata_bench.attention_eval.runtime import find_contiguous_spans
from mydata_bench.top_eval.protocol import SYSTEM_PROMPT, OFFICIAL_QUESTION, create_composite_frame, format_percentage
from mydata_bench.top_eval.versioning import is_official_sole, validate_protocol_config

METER_PROMPT = "The task for the robot is '{task}'. Given the trajectory video, predict the task progress at each frame, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."
PROG = '<|prog_token|>'
SPECIAL_TOKENS = ['<|split_token|>', '<|reward_token|>', '<|pref_token|>', '<|sim_token|>', PROG]


@lru_cache(maxsize=128)
def read_frames(paths):
    return tuple(np.asarray(Image.open(p).convert('RGB')) for p in paths)


@lru_cache(maxsize=2048)
def read_track(path):
    with open(path) as f: track = json.load(f)
    return {int(r['frame_index']): r['bbox'] for r in track['frames'] if isinstance(r.get('bbox'), list)}


def resize_image(a, cfg):
    h, w = a.shape[:2]
    rh, rw = smart_resize(h, w, factor=32, min_pixels=cfg['min_pixels'], max_pixels=cfg['max_pixels'])
    return Image.fromarray(a).resize((rw, rh))


def prompt_payload(sample, cfg, step=None, previous=0):
    frames = read_frames(tuple(sample['image_paths']))
    source_indices = sample['sampling']['selected_source_indices']
    model, protocol = cfg['model'], cfg['protocol']
    images, videos, video_metadata, span_specs = [], [], [], []
    content = []
    if is_official_sole(cfg):
        validate_protocol_config(cfg)
        assert step is not None and 1 <= step < len(frames)
        previous = format_percentage(previous)
        selected = [0, step-1, step]
        composite = create_composite_frame(None, frames[0], None, frames[step-1], None, frames[step], view_type='external')
        # Official RewardGen applies factor-28 resizing before Qwen3 processing.
        ch, cw = composite.shape[:2]
        rh, rw = smart_resize(ch, cw, factor=28, min_pixels=3136, max_pixels=12845056)
        intermediate = Image.fromarray(composite).resize((rw, rh))
        # The second resize belongs to the checkpoint processor. A PIL resize
        # to factor 32 followed by do_resize=False is not pixel-equivalent.
        images = [intermediate]
        span_specs = [{'sources': [source_indices[i] for i in selected], 'size': [cw, ch],
                       'source_size': [frames[0].shape[1], frames[0].shape[0]], 'mosaic': True}]
        content = [{'type': 'image'}, {'type': 'text', 'text': OFFICIAL_QUESTION.format(task_description=sample['task'], prev_progress=previous)}]
    else:
        prompt = (METER_PROMPT.format(task=sample['task']) if model == 'meter' else
                  f"The task description is: {sample['task']}. The observations are in chronological order. The task progress at the first timestep is 0%. Predict the task progress at the final timestep.")
        is_video = protocol in {'text_video', 'video_text'}
        if is_video:
            resized = [np.asarray(resize_image(a, cfg)) for a in frames]
            import torch
            videos = [torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2)]
            video_metadata = [{'total_num_frames': sample['sampling']['decoded_frame_count'],
                               'fps': sample['sampling']['source_fps'], 'frames_indices': source_indices}]
            visual = [{'type': 'video'}]
            for j in range(0, len(frames), 2):
                span_specs.append({'sources': source_indices[j:j+2], 'size': [frames[j].shape[1], frames[j].shape[0]], 'mosaic': False})
        else:
            images = [resize_image(a, cfg) for a in frames]
            visual = []
            for j, a in enumerate(frames):
                visual.append({'type': 'image'})
                if protocol in {'interleaved', 'official'}:
                    visual.append({'type': 'text', 'text': PROG if model == 'meter' else f' Frame {j+1} of {len(frames)}. '})
                span_specs.append({'sources': [source_indices[j]], 'size': [a.shape[1], a.shape[0]], 'mosaic': False})
        text = [{'type': 'text', 'text': prompt}]
        content = visual + text if protocol in {'image_text', 'video_text'} else text + visual
        if model == 'meter' and protocol not in {'interleaved', 'official'}:
            content.append({'type': 'text', 'text': PROG})
    messages = [{'role': 'user', 'content': content}]
    if model == 'sole': messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    return {'messages': messages, 'images': images, 'videos': videos, 'video_metadata': video_metadata,
            'span_specs': span_specs, 'step': step, 'previous': previous,
            'processor_kwargs': {'do_resize': True, 'add_special_tokens': False} if is_official_sole(cfg) else {}}


def align_input(ids, grids, payload, sample, cfg):
    video = bool(payload['videos'])
    spans = find_contiguous_spans(ids, 151656 if video else 151655)
    if video:
        if len(grids) != 1: raise ValueError('Expected one video grid')
        t, h, w = map(int, grids[0])
        expanded = [(1, h, w)]*t
    else:
        expanded = [tuple(map(int, g)) for g in grids]
    specs = payload['span_specs']
    if not len(spans) == len(expanded) == len(specs):
        raise ValueError(f'Token/frame alignment mismatch: {len(spans)}, {len(expanded)}, {len(specs)}')
    records = []
    for i, ((start, end), grid, spec) in enumerate(zip(spans, expanded, specs)):
        span = ImageSpan(f'span_{i}', '', start, end, grid)
        if span.token_count != int(np.prod(grid))//4: raise ValueError('Visual grid/token mismatch')
        records.append({'span': span, **spec})
    result = {'records': records, 'visual': [p for a,b in spans for p in range(a,b)],
              'sequence_length': len(ids), 'target': {}, 'negative': {}, 'wrong': {}, 'alignment': {}}
    if not sample.get('cohort'): return result
    track = read_track(sample['tracking_path'])
    for scope in ['last_frame', 'all_frames']:
        selected, negative_domain, detail = [], [], []
        chosen = records[-1:] if scope == 'last_frame' else records
        for record in chosen:
            sources = record['sources']
            span, size = record['span'], record['size']
            source_boxes, used = [], []
            for source in sources:
                index = min(track, key=lambda n: (abs(n-source), n))
                source_boxes.append(track[index]); used.append(index)
            if record['mosaic']:
                # Select cells intersecting the requested source-image content.
                # Boundary cells may also contain padding or an adjacent tile.
                sw, sh = record['source_size']
                scale = 384/max(sw, sh)
                rw, rh = int(sw*scale), int(sh*scale)
                ox, oy = (384-rw)//2, (384-rh)//2
                tiles = [2] if scope == 'last_frame' else [0,1,2]
                boxes = []
                for tile in tiles:
                    box = source_boxes[tile]
                    offset = tile*389
                    transformed = [offset+ox+box[0]*rw/sw, oy+box[1]*rh/sh,
                                   offset+ox+box[2]*rw/sw, oy+box[3]*rh/sh]
                    boxes.append(transformed)
                    selected += bbox_to_token_positions(span, transformed, tuple(size))
                    negative_domain += bbox_to_token_positions(span, [offset+ox, oy, offset+ox+rw, oy+rh], tuple(size))
            else:
                # Native temporal tubelets jointly encode BOTH sampled source frames.
                # Use their union; do not pretend the merged key belongs only to the last frame.
                boxes = [[min(b[0] for b in source_boxes), min(b[1] for b in source_boxes),
                          max(b[2] for b in source_boxes), max(b[3] for b in source_boxes)]]
                selected += bbox_to_token_positions(span, boxes[0], tuple(size))
                negative_domain += list(range(span.start, span.end))
            detail.append({'span': span.label, 'start': span.start, 'end': span.end,
                           'grid_thw': list(span.grid_thw), 'source_frames': sources,
                           'tracking_frames': used, 'boxes': boxes, 'mosaic': record['mosaic']})
        target = sorted(set(selected))
        domain = set(negative_domain)
        if not target or not set(target) <= domain: raise ValueError('Invalid target key set')
        # Equal-cardinality disjoint farthest-grid controls, within each selected temporal domain.
        wrong = []
        wrong_reason = None
        for record in records:
            span = record['span']; width = span.grid_thw[2]//2
            inside = [p for p in target if span.start <= p < span.end]
            if not inside: continue
            candidates = [p for p in domain-set(target) if span.start <= p < span.end]
            coords = np.array([((p-span.start)//width, (p-span.start)%width) for p in inside])
            center = coords.mean(0)
            candidates.sort(key=lambda p: (-(((p-span.start)//width-center[0])**2+((p-span.start)%width-center[1])**2),p))
            if len(candidates) < len(inside):
                wrong_reason = 'Insufficient non-target cells for an equal-sized disjoint control in the same temporal span'
                break
            wrong += candidates[:len(inside)]
        result['target'][scope] = target
        result['negative'][scope] = sorted(domain-set(target))
        result['wrong'][scope] = sorted(wrong) if wrong_reason is None else []
        if wrong_reason is not None:
            result.setdefault('control_unavailable',{})[scope] = wrong_reason
        result['alignment'][scope] = detail
    return result
