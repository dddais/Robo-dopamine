"""Optional visual category proposals for unresolved SAM3 detections."""
from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict
from typing import Any

from PIL import Image

from ..io import sha256_file
from .base import _box_iou, _deduplicate_candidates


def parse_visual_boxes(response: str) -> list[list[float]]:
    """Read normalized boxes; reject malformed/out-of-range model coordinates."""
    text = response.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != '```':
            raise ValueError('Unclosed visual grounding JSON fence')
        text = '\n'.join(lines[1:-1])
    value = json.loads(text)
    if isinstance(value, dict):
        value = value.get('boxes')
    elif isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        value = [row.get('boxes') for row in value]
    if not isinstance(value, list):
        raise ValueError('Visual grounding must return a boxes array')
    if len(value) == 4 and all(type(x) in (float, int) for x in value):
        value = [value]
    if len(value) > 40:
        raise ValueError('Too many visual grounding proposals')
    boxes = []
    for box in value:
        if not isinstance(box, list) or len(box) != 4 or not all(
            type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1000 for x in box
        ) or not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError('Visual grounding requires legal xyxy coordinates in 0..1000')
        boxes.append([float(x) for x in box])
    return boxes


class VisualCategoryResolver:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        retry_size = config.get('empty_retry_long_side', 0)
        if type(retry_size) is not int or not 0 <= retry_size <= 4096:
            raise ValueError('visual_resolver.empty_retry_long_side must be an integer in 0..4096')
        descriptions = config.get('category_descriptions', {})
        if not isinstance(descriptions, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in descriptions.items()):
            raise ValueError('visual_resolver.category_descriptions must map query strings to nonempty descriptions')
        self._model = None
        self._cache: OrderedDict = OrderedDict()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.config['model_path'])
        self._device = self.config.get('device', 'cuda')
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config['model_path'], torch_dtype=torch.bfloat16,
            device_map=self._device, attn_implementation='sdpa',
        ).eval()

    def _generate_response(self, image, query: str) -> str:
        self._load()
        description = self.config.get('category_descriptions', {}).get(query)
        prompt_query = f'{query} ({description})' if description else query
        prompt = (
            f'Locate all objects matching the category "{prompt_query}" in this image. '
            'Return a JSON object with a "boxes" array containing one [x1,y1,x2,y2] bounding box '
            'for each matching physical object, using coordinates normalized to 0..1000. '
            'Distinguish object subtypes and colors. If none match, return {"boxes": []}. '
            'Enclose each complete object separately.'
        )
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt',
        ).to(self._device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=int(self.config.get('max_new_tokens', 384)), do_sample=False,
            )
        return self._processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )[0]

    def _propose(self, image_path: str, query: str) -> tuple[list[list[float]], dict]:
        key = (sha256_file(image_path), query)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        response = self._generate_response(image_path, query)
        diagnostics = {'query': query, 'response': response, 'model_path': self.config['model_path']}
        if query in self.config.get('category_descriptions', {}):
            diagnostics['category_description'] = self.config['category_descriptions'][query]
        try:
            boxes = parse_visual_boxes(response)
        except (ValueError, TypeError) as exc:
            boxes = []
            diagnostics['error'] = str(exc)
        # Small objects can disappear during the visual encoder's patching. A
        # valid empty answer gets at most one full-image upscale retry; retain
        # the entire scene so competing instances cannot be cropped away.
        long_side = self.config.get('empty_retry_long_side', 0)
        if not boxes and 'error' not in diagnostics and long_side:
            with Image.open(image_path) as original:
                width, height = original.size
                if max(width, height) < long_side:
                    scale = long_side / max(width, height)
                    size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    enlarged = original.convert('RGB').resize(size, Image.Resampling.LANCZOS)
                    try:
                        retry = self._generate_response(enlarged, query)
                    finally:
                        enlarged.close()
                    diagnostics['scale_retry'] = {
                        'trigger': 'valid_empty_response', 'original_response': response,
                        'source_size': [width, height], 'input_size': list(size),
                    }
                    diagnostics['response'] = retry
                    try:
                        boxes = parse_visual_boxes(retry)
                    except (ValueError, TypeError) as exc:
                        boxes = []
                        diagnostics['error'] = str(exc)
        result = (boxes, diagnostics)
        self._cache[key] = result
        while len(self._cache) > int(self.config.get('cache_size', 32)):
            self._cache.popitem(last=False)
        return result

    def resolve(self, image_path: str, queries: list[str], sam3_candidates: list[dict]) -> tuple[list[dict], dict]:
        with Image.open(image_path) as image:
            width, height = image.size
        rows, diagnostics = [], []
        for priority, query in enumerate(queries):
            boxes, diagnostic = self._propose(image_path, query)
            diagnostic = copy.deepcopy(diagnostic)
            diagnostics.append(diagnostic)
            query_rows, support = [], []
            used = set()
            for instance_index, normalized in enumerate(boxes):
                box = [normalized[0] * width / 1000, normalized[1] * height / 1000,
                       normalized[2] * width / 1000, normalized[3] * height / 1000]
                eligible = [(i, candidate) for i, candidate in enumerate(sam3_candidates)
                            if i not in used and candidate.get('semantic_query', candidate.get('query')) == query]
                eligible.sort(key=lambda pair: _box_iou(pair[1]['bbox'], box), reverse=True)
                match = eligible[0] if eligible else None
                separated = len(eligible) < 2 or (
                    _box_iou(eligible[0][1]['bbox'], box) - _box_iou(eligible[1][1]['bbox'], box)
                    >= float(self.config.get('sam3_match_margin', 0.1))
                )
                row: dict[str, Any] = {'bbox': box}
                matched = False
                if match is not None and separated and _box_iou(match[1]['bbox'], box) >= float(self.config.get('sam3_match_iou', 0.5)):
                    matched = True
                    used.add(match[0])
                    row = dict(match[1])
                    row['sam3_detector_score'] = row.get('score')
                # A generated box has no calibrated detection probability. Equal
                # scores preserve real instance ambiguity rather than ranking by text order.
                row.update(score=0.0, score_kind='unscored_visual_proposal', query=query,
                           semantic_query=query, query_priority=priority, label=query,
                           visual_proposal_bbox=box, proposal_origin='qwen3_vl',
                           visual_instance_group=sha256_file(image_path) + ':' + query,
                           visual_instance_index=instance_index)
                query_rows.append(row)
                support.append({'matched': matched, 'best_iou': _box_iou(match[1]['bbox'], box) if match else 0.0})
            if diagnostic.get('scale_retry') and self.config.get('empty_retry_requires_sam_support', False):
                accepted = bool(query_rows) and all(item['matched'] for item in support)
                diagnostic['scale_retry'].update(sam3_support=support, accepted=accepted)
                if not accepted:
                    # Reject the whole proposal set: dropping unsupported rivals
                    # could manufacture a unique instance from a real ambiguity.
                    diagnostic['scale_retry']['rejection_reason'] = 'sam3_support_not_unique'
                    query_rows = []
            rows.extend(query_rows)
        return _deduplicate_candidates(rows), {'queries': diagnostics}
