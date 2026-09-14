"""Model inputs and scope-specific token masks, with grounding-free baselines."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mydata_bench.addbase_eval.attention import AttentionController
from mydata_bench.addbase_eval.protocols import (PROG, SPECIAL_TOKENS, prompt_payload,
                                               align_input, resize_image)
from mydata_bench.attention_eval.masking import bbox_to_token_positions
from mydata_bench.meter_eval.model import RobometerModel
from mydata_bench.protocol import IMAGE_LABELS, chat_messages, parse_score, progress
from mydata_bench.qwen_eval.protocols import interleaved_image_sequence_messages
from mydata_bench.roboreward_eval.runner import native_prompt, parse_native_score, _native_video_metadata
from mydata_bench.top_eval.protocol import parse_progress, format_percentage
from mydata_bench.top_eval.versioning import is_official_sole, protocol_metadata
from .common import fingerprint, validate_config
from .grounding import exact_boxes, ControlUnavailable, UnusableGeometry, sample_identity


def align_exact(ids, grids, payload, sample, cfg, scope=None):
    # Reuse the tested grid/span checks only. The old nearest-frame lookup is bypassed.
    mapping = align_input(ids, grids, payload, {'cohort': False}, cfg)
    mapping['source_indices'] = [i for spec in payload['span_specs'] for i in spec['sources']]
    if 'native_indices' in payload:
        mapping['native_indices'] = payload['native_indices']
    if scope is None:
        return mapping
    records = mapping['records']
    if cfg['model'] == 'grm':
        chosen = [records[5]] if scope == 'last_frame' else [records[i] for i in (0, 2, 5)]
    else:
        chosen = records[-1:] if scope == 'last_frame' else records
    selected, domain, details = set(), set(), []
    for record in chosen:
        span, size, sources = record['span'], tuple(record['size']), record['sources']
        tiles = ([2] if scope == 'last_frame' else [0, 1, 2]) if record['mosaic'] else None
        needed = [sources[i] for i in tiles] if tiles is not None else sources
        boxes = exact_boxes(sample, needed)
        mapped_boxes = []
        if tiles is not None:
            sw, sh = record['source_size']
            scale = 384 / max(sw, sh)
            rw, rh = int(sw * scale), int(sh * scale)
            ox, oy = (384 - rw) // 2, (384 - rh) // 2
            for tile in tiles:
                b, offset = boxes[sources[tile]], tile * 389
                transformed = [offset + ox + b[0] * rw / sw, oy + b[1] * rh / sh,
                               offset + ox + b[2] * rw / sw, oy + b[3] * rh / sh]
                selected.update(bbox_to_token_positions(span, transformed, size))
                domain.update(bbox_to_token_positions(span, [offset + ox, oy, offset + ox + rw, oy + rh], size))
                mapped_boxes.append(transformed)
        else:
            # A temporal tubelet is a union of the individual source masks in token space.
            # Do not fill the rectangular gap between disjoint source boxes.
            for source in sources:
                box = boxes[source]
                selected.update(bbox_to_token_positions(span, box, size))
                mapped_boxes.append(box)
            domain.update(range(span.start, span.end))
        details.append({'source_frames': needed, 'tracking_frames': needed,
                        'span': span.label, 'grid_thw': list(span.grid_thw), 'boxes': mapped_boxes})
    if not selected or not selected <= domain:
        raise ValueError('Invalid target token geometry')
    if scope == 'last_frame' and chosen[-1]['mosaic']:
        # Qwen's spatial merge can straddle the 5px separator. Intersecting
        # the current tile is insufficient: a key can also cover the previous
        # tile. Select only cells fully inside the current temporal tile.
        record = chosen[-1]
        span, width = record['span'], record['size'][0]
        cols = span.grid_thw[2] // 2
        left, right = 2 * 389, 2 * 389 + 384
        isolated = {p for p in domain if (p - span.start) % cols * width / cols >= left
                    and ((p - span.start) % cols + 1) * width / cols <= right}
        mapping['excluded_mosaic_boundary_keys'] = sorted(domain - isolated)
        mapping['mosaic_boundary_policy'] = 'fully_inside_current_tile'
        domain = isolated
        selected &= domain
        if not selected:
            raise UnusableGeometry(needed)
    wrong, reason = set(), None
    for record in chosen:
        span = record['span']
        width = span.grid_thw[2] // 2
        inside = [p for p in selected if span.start <= p < span.end]
        if not inside:
            continue
        candidates = [p for p in domain - selected if span.start <= p < span.end]
        center = np.mean([((p - span.start) // width, (p - span.start) % width) for p in inside], axis=0)
        candidates.sort(key=lambda p: (-(((p - span.start) // width - center[0]) ** 2
                                         + ((p - span.start) % width - center[1]) ** 2), p))
        if len(candidates) < len(inside):
            reason = 'insufficient_disjoint_tokens'
            break
        wrong.update(candidates[:len(inside)])
    mapping['target'][scope] = sorted(selected)
    mapping['negative'][scope] = sorted(domain - selected)
    mapping['wrong'][scope] = sorted(wrong) if reason is None else []
    mapping['control_unavailable'] = reason
    mapping['alignment'][scope] = details
    return mapping


class Runtime:
    def __init__(self, cfg, *, processor_only=False):
        validate_config(cfg)
        self.cfg = cfg
        self._control_checks = {}
        self._geometry_checks = {}
        torch.manual_seed(cfg['seed'])
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(cfg['processor_path'], local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        if cfg['model'] == 'meter':
            for token in SPECIAL_TOKENS:
                if token not in self.processor.tokenizer.get_vocab():
                    self.processor.tokenizer.add_special_tokens({'additional_special_tokens': [token]})
            self.prog_id = self.processor.tokenizer.convert_tokens_to_ids(PROG)
        self.model = self.controller = None
        if processor_only:
            return
        if cfg['model'] == 'meter':
            self.model = RobometerModel.load(cfg['model_path'])
            if len(self.processor.tokenizer) != self.model.config.text_config.vocab_size:
                raise ValueError('Checkpoint/tokenizer vocab mismatch')
        else:
            self.model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
                cfg['model_path'], dtype=torch.bfloat16, device_map='cuda:0',
                attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
            if any(loading[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')):
                raise ValueError(f'Non-exact checkpoint loading: {loading}')
            self.model.loading_audit = loading
        self.model.eval()
        self.layers = self.model.model.language_model.layers
        self.num_layers = len(self.layers)
        self.num_heads = self.model.config.text_config.num_attention_heads
        vision = self.model.config.vision_config
        if (vision.spatial_merge_size != 2 or vision.temporal_patch_size != 2
                or self.model.config.image_token_id != 151655 or self.model.config.video_token_id != 151656):
            raise ValueError('Unsupported vision token geometry for these grounding maps')
        self.controller = AttentionController(self.layers)

    def payload(self, sample, step=None, previous=0):
        cfg = self.cfg
        if cfg['model'] in ('meter', 'sole'):
            return prompt_payload(sample, cfg, step, previous)
        if cfg['model'] == 'grm':
            paths = sample['grm_image_paths']
            messages = chat_messages(sample['task'], prompt_mode='official')
            if cfg['protocol'] in ('text_image', 'image_text'):
                parts = messages[0]['content']
                text = [{'type': 'text', 'text': ''.join(p['text'] for p in parts if p['type'] == 'text')
                         + '\nImage order: ' + ', '.join(IMAGE_LABELS)}]
                visual = [p for p in parts if p['type'] == 'image']
                messages[0]['content'] = text + visual if cfg['protocol'] == 'text_image' else visual + text
            source = sample['sampling']['selected_source_indices']
            indices = [source[0], None, source[0], None, None, source[-1], None, None]
        else:
            paths = sample['image_paths']
            indices = sample['sampling']['selected_source_indices']
            if cfg['protocol'] == 'interleaved':
                messages = interleaved_image_sequence_messages(sample['task'], paths)
            else:
                text = [{'type': 'text', 'text': native_prompt(sample['task'])}]
                visual = [{'type': 'image'} for _ in paths]
                messages = [{'role': 'user', 'content': text + visual if cfg['protocol'] == 'text_image' else visual + text}]
        arrays = []
        for path in paths:
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert('RGB')))
        grm = cfg['model'] == 'grm'
        return {'messages': messages, 'images': [Image.fromarray(a) if grm else resize_image(a, cfg) for a in arrays],
                'videos': [], 'span_specs': [
                    {'sources': [index] if index is not None else [], 'size': [a.shape[1], a.shape[0]], 'mosaic': False}
                    for index, a in zip(indices, arrays)],
                'processor_kwargs': {'do_resize': True, 'min_pixels': cfg['min_pixels'],
                                     'max_pixels': cfg['max_pixels']} if grm else {}}

    def prepare(self, sample, scope=None, step=None, previous=0):
        cfg = self.cfg
        native = cfg['model'] in ('qwen', 'roboreward') and cfg['protocol'] == 'official'
        if native:
            messages = [{'role': 'user', 'content': [
                {'type': 'text', 'text': native_prompt(sample['task'])},
                {'type': 'video', 'video': sample['video_path']}]}]
            batch = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_metadata=True)
            metadata = _native_video_metadata(batch.pop('video_metadata'))
            batch = batch.convert_to_tensors('pt')
            indices = list(map(int, metadata['frames_indices']))
            padded = indices + indices[-1:] if len(indices) % 2 else indices
            grids = batch['video_grid_thw'].tolist()
            payload = {'videos': [True], 'native_indices': indices, 'span_specs': [
                {'sources': padded[i:i + 2], 'size': [sample['sampling']['width'], sample['sampling']['height']],
                 'mosaic': False} for i in range(0, len(padded), 2)]}
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            payload = self.payload(sample, step, previous)
            text = self.processor.apply_chat_template(payload['messages'], tokenize=False,
                      add_generation_prompt=cfg['model'] != 'meter', add_vision_id=cfg['model'] == 'meter',
                      enable_thinking=False)
            kwargs = {'text': [text], 'images': payload['images'], 'padding': True,
                      'return_tensors': 'pt', 'do_resize': False, **payload['processor_kwargs']}
            batch = self.processor(**kwargs)
            grids = batch['image_grid_thw'].tolist()
        ids = batch['input_ids'][0].tolist()
        mapping = align_exact(ids, grids, payload, sample, cfg, scope)
        query = max(j for j, v in enumerate(ids) if v == self.prog_id) if cfg['model'] == 'meter' else len(ids) - 1
        mapping.update(query=query, prompt_sha256=fingerprint(text), input_ids_sha256=fingerprint(ids))
        return batch, mapping, query, text

    def _move(self, batch):
        return {k: v.to(device=self.model.device, dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                for k, v in batch.items() if torch.is_tensor(v)}

    @staticmethod
    def audit(mapping):
        return {k: v for k, v in mapping.items() if k != 'records'}

    def control_preflight(self, sample, scope):
        key = sample_identity(sample), scope
        if key in self._control_checks:
            reason = self._control_checks[key]
            if reason:
                raise ControlUnavailable(reason)
            return
        for step in range(1, 8) if is_official_sole(self.cfg) else [None]:
            _, mapping, _, _ = self.prepare(sample, scope, step)
            if mapping['control_unavailable']:
                self._control_checks[key] = mapping['control_unavailable']
                raise ControlUnavailable(mapping['control_unavailable'])
        self._control_checks[key] = None

    def geometry_preflight(self, sample, scope):
        """Resolve all SOLE current-tile geometries before any recursive intervention."""
        key = sample_identity(sample), scope
        if key not in self._geometry_checks:
            bad_frames, excluded, control_reason = [], 0, None
            for step in range(1, 8):
                try:
                    _, mapping, _, _ = self.prepare(sample, scope, step)
                    excluded += len(mapping.get('excluded_mosaic_boundary_keys', []))
                    control_reason = control_reason or mapping['control_unavailable']
                except UnusableGeometry as exc:
                    bad_frames += exc.frames
            self._geometry_checks[key] = {
                'eligible': not bad_frames, 'reason': 'no_isolated_target_tokens' if bad_frames else None,
                'unusable_frames': sorted(set(bad_frames)), 'excluded_boundary_tokens_over_steps': excluded,
                'mosaic_boundary_policy': 'fully_inside_current_tile'}
            if not bad_frames:
                self._control_checks[key] = control_reason
        return dict(self._geometry_checks[key])

    def collect(self, sample, scope, step=None, previous=0):
        batch, mapping, query, text = self.prepare(sample, scope, step, previous)
        state = self.controller.rank([mapping], [query], self.num_layers, self.num_heads, scopes=[scope])
        try:
            with torch.inference_mode():
                extra = {} if self.cfg['model'] == 'meter' else {'logits_to_keep': 1}
                self.model(**self._move(batch), use_cache=False, **extra)
            if len(state['seen']) != self.num_layers:
                raise RuntimeError('Incomplete head observations')
            return {'raw_mass': state['raw'][scope][0].tolist(), 'token_audit': self.audit(mapping),
                    'query_kind': 'final_prog_token' if self.cfg['model'] == 'meter' else 'last_prompt'}
        finally:
            self.controller.clear()

    def predict(self, sample, condition, ranking=None, step=None, previous=0):
        cfg = self.cfg
        scope = None if condition == 'baseline' else condition.split(':')[0]
        batch, mapping, _, text = self.prepare(sample, scope, step, previous)
        state = None
        if scope is not None:
            _, kind, k = condition.split(':')
            if kind == 'wrong_region' and mapping['control_unavailable']:
                raise ControlUnavailable(mapping['control_unavailable'])
            heads = ranking['ranking'][-int(k):] if kind == 'low_rank' else ranking['ranking'][:int(k)]
            if len(heads) != int(k):
                raise ValueError('Insufficient ranked heads')
            # A dense all-zero mask can select a different BF16 SDPA kernel
            # from the ordinary implicit causal path. Zero bias must use the
            # exact baseline path, including for numerical audit calls.
            if cfg['bias'] != 0:
                state = self.controller.steer([mapping], heads, cfg['bias'], scope,
                                              'wrong' if kind == 'wrong_region' else 'target')
            else:
                self.controller.clear()
        started = time.monotonic()
        try:
            inputs = self._move(batch)
            with torch.inference_mode():
                if cfg['model'] == 'meter':
                    output = self.model(**inputs, use_cache=False)
                    values, successes, positions = self.model.read_progress(output.last_hidden_state, inputs['input_ids'], self.prog_id)
                    row = {'status': 'ok', 'progress': values[0][-1], 'per_frame_progress': values[0],
                           'success_probability': successes[0][-1], 'per_frame_success': successes[0],
                           'progress_token_positions': positions[0]}
                else:
                    output = self.model.generate(**inputs, do_sample=False, max_new_tokens=cfg['max_new_tokens'],
                            temperature=None, top_p=None, top_k=None, use_cache=True, logits_to_keep=1,
                            pad_token_id=self.processor.tokenizer.pad_token_id)
                    tokens = output[:, inputs['input_ids'].shape[1]:]
                    raw = self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
                    row = {'raw_output': raw, 'generated_tokens': int(tokens.shape[1])}
                    try:
                        if cfg['model'] == 'sole':
                            row.update(parse_progress(raw))
                        elif cfg['model'] == 'grm':
                            signed = parse_score(raw)
                            row.update(status='ok', progress=progress(signed), signed_score=signed)
                        else:
                            score = parse_native_score(raw)
                            row.update(status='ok', progress=(score - 1) / 4, predicted_reward=score)
                    except ValueError as exc:
                        row.update(status='parse_error', progress=None, error=str(exc))
            row.update(example_id=sample['example_id'], condition=condition, step=step,
                       token_audit=self.audit(mapping), prompt=text,
                       attention_diagnostics=state['diagnostics'] if state else {},
                       duration_seconds=time.monotonic() - started, **protocol_metadata(cfg))
            if state is not None:
                expected_layers = {str(h['layer']) for h in heads}
                if (set(state['diagnostics']) != expected_layers
                        or any(d['prefill_calls'] < 1 for d in state['diagnostics'].values())):
                    raise RuntimeError('Requested SAS heads did not execute through the attention controller')
            if is_official_sole(cfg):
                row['previous_percentage_text'] = format_percentage(previous)
            return row
        finally:
            self.controller.clear()
