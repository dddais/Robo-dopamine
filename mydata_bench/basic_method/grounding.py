"""Exact source-frame eligibility. Missing files and bad identities are errors."""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

from .common import file_hash, fingerprint


class MissingGrounding(ValueError):
    def __init__(self, reason, frames=()):
        self.reason, self.frames = reason, sorted(set(frames))
        super().__init__(f'{reason}: {self.frames}')


class ControlUnavailable(ValueError):
    pass


class UnusableGeometry(ValueError):
    """A valid box occupies only tokens shared with an unselected source frame."""
    def __init__(self, frames):
        self.frames = sorted(set(frames))
        super().__init__(f'no_isolated_target_tokens: {self.frames}')


def valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in box)
            and box[2] > box[0] and box[3] > box[1])


@lru_cache(maxsize=2048)
def _load_track(path, digest, size, mtime_ns):
    if file_hash(path) != digest:
        raise ValueError(f'Frozen track changed: {path}')
    track = json.loads(Path(path).read_text())
    boxes = {}
    for row in track['frames']:
        index = int(row['frame_index'])
        if index in boxes:
            raise ValueError(f'Duplicate tracking frame {index}: {path}')
        if row.get('bbox') is not None:
            if not valid_box(row['bbox']):
                raise ValueError(f'Malformed bbox at {index}: {path}')
            boxes[index] = row['bbox']
    return track, boxes


def track_boxes(sample):
    ground = sample['grounding']
    if not ground['eligible']:
        raise MissingGrounding('not_in_release')
    path = Path(ground['tracking_path'])
    stat = path.stat()  # Missing artifact is NEVER a baseline fallback reason.
    track, boxes = _load_track(str(path), ground['tracking_sha256'], stat.st_size, stat.st_mtime_ns)
    for field in ('example_id', 'video_sha256'):
        if track.get(field) != sample[field]:
            raise ValueError(f'Track {field} mismatch for {sample["example_id"]}')
    if int(track['terminal_frame_index']) != sample['sampling']['terminal_source_index']:
        raise ValueError(f'Track/input terminal mismatch: {sample["example_id"]}')
    return boxes


def exact_boxes(sample, indices):
    boxes = track_boxes(sample)
    missing = sorted(set(indices) - set(boxes))
    if missing:
        raise MissingGrounding('missing_exact_frames', missing)
    return {i: boxes[i] for i in indices}


def required_frames(sample, cfg, scope, native_indices=None):
    if scope not in ('last_frame', 'all_frames'):
        raise ValueError(scope)
    indices = sample['sampling']['selected_source_indices']
    if cfg['model'] == 'grm':
        return [indices[-1]] if scope == 'last_frame' else sorted({indices[0], indices[-1]})
    if cfg['model'] == 'sole' and cfg['protocol'] == 'official':
        # All seven current frames will be steered during the recursive rollout.
        return sorted(set(indices[1:] if scope == 'last_frame' else indices))
    if cfg['model'] in ('qwen', 'roboreward') and cfg['protocol'] == 'official':
        if native_indices is None:
            raise ValueError('Native video eligibility requires actual processor frame indices')
        indices = list(native_indices)
        if not indices:
            raise ValueError('Empty native video input')
        # Odd-length native videos pad their last temporal unit by repeating the final frame.
        if len(indices) % 2:
            indices += indices[-1:]
        return sorted(set(indices[-2:] if scope == 'last_frame' else indices))
    return sorted(set(indices[-1:] if scope == 'last_frame' else indices))


def eligibility(sample, cfg, scope, native_indices=None):
    required = required_frames(sample, cfg, scope, native_indices)
    try:
        exact_boxes(sample, required)
    except MissingGrounding as exc:
        return {'eligible': False, 'reason': exc.reason, 'missing_frames': exc.frames,
                'required_frames': required}
    return {'eligible': True, 'reason': None, 'missing_frames': [], 'required_frames': required}


def sample_identity(sample):
    return fingerprint(sample)
