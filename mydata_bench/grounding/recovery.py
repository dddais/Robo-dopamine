"""Offline terminal recovery, accepted only after a unique reverse identity check."""
from __future__ import annotations

from .base import _box_iou


def recover_terminal_track(grounder, video_path, first_selection, last_image_path, queries,
                           anchor_index, terminal_index, forward_tracks):
    forward_diagnostics = dict(grounder.last_tracking_diagnostics)
    config = grounder.config.get('terminal_recovery', {})
    candidates = grounder.candidates(last_image_path, queries)
    maximum = int(config.get('max_candidates', 6))
    if maximum < 1:
        raise ValueError('terminal_recovery.max_candidates must be positive')
    recovery = {'method': 'reverse_cycle', 'attempts': []}
    visual_config = grounder.config.get('visual_resolver', {})
    force_visual = any(query in visual_config.get('always_verify_queries', []) for query in queries)
    if (not candidates or len(candidates) > maximum or force_visual) and visual_config.get('enabled'):
        candidates, visual = grounder.visual_candidates(last_image_path, queries, candidates)
        recovery['visual_proposals'] = visual
    recovery['candidate_count'] = len(candidates)
    accepted = []
    if len(candidates) > maximum:
        # Truncating would leave untested competing identities while claiming uniqueness.
        recovery['status'] = 'too_many_terminal_candidates'
    elif not candidates:
        recovery['status'] = 'terminal_target_not_detected'
    else:
        minimum = float(config.get('anchor_iou', 0.5))
        if not 0 < minimum <= 1:
            raise ValueError('terminal_recovery.anchor_iou must be in (0, 1]')
        for candidate in candidates:
            reverse = grounder.track(
                video_path, candidate['bbox'], terminal_index,
                anchor_mask=candidate.get('_mask'), terminal_index=anchor_index,
                propagation_direction='backward',
            )
            by_frame = {row['frame_index']: row for row in reverse}
            first, last = by_frame.get(anchor_index), by_frame.get(terminal_index)
            overlap = _box_iou(first['bbox'], first_selection['bbox']) if first else 0.0
            valid = first is not None and last is not None and overlap >= minimum
            # A full reverse pass must keep one ID. No per-frame combination of attempts.
            valid = valid and len({row['obj_id'] for row in reverse}) == 1
            diagnostics = dict(grounder.last_tracking_diagnostics)
            recovery['attempts'].append({
                'terminal_bbox': candidate['bbox'], 'anchor_iou': overlap,
                'accepted_identity': valid, 'frame_coverage': diagnostics.get('frame_coverage'),
            })
            if valid:
                accepted.append((reverse, diagnostics, overlap))
        recovery['status'] = 'accepted' if len(accepted) == 1 else (
            'ambiguous_reverse_identity' if accepted else 'reverse_identity_not_verified'
        )
    if len(accepted) != 1:
        grounder.last_tracking_diagnostics = {**forward_diagnostics, 'terminal_recovery': recovery}
        return forward_tracks
    tracks, diagnostics, overlap = accepted[0]
    present = {row['frame_index'] for row in tracks}
    missing = [i for i in range(anchor_index, terminal_index + 1) if i not in present]
    longest = current = 0
    for i in range(anchor_index, terminal_index + 1):
        current = 0 if i in present else current + 1
        longest = max(longest, current)
    grounder.last_tracking_diagnostics = {
        **diagnostics, 'terminal_present': terminal_index in present,
        'tracked_frame_count': len(present), 'missing_frame_count': len(missing),
        'frame_coverage': 1 - len(missing) / (terminal_index - anchor_index + 1),
        'longest_missing_run': longest, 'prompt_frame_index': terminal_index,
        'cycle_anchor_iou': overlap, 'terminal_recovery': recovery,
        'forward_diagnostics': forward_diagnostics,
    }
    return tracks
