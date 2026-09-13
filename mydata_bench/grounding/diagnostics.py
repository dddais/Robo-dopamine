"""Inspect automatic grounding coverage without confusing it with accuracy."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..io import read_jsonl, write_json


def _latest(run: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(run / 'grounding.jsonl'):
        if row.get('frame') in {'first', 'last'}:
            grouped.setdefault(row['example_id'], {})[row['frame']] = row
    return grouped


def diagnose(run: Path, baseline_run: Path | None = None) -> dict[str, Any]:
    grouped = _latest(run)
    status_pairs = Counter()
    first_failure_reasons = Counter()
    tracking = []
    for example_id, frames in sorted(grouped.items()):
        first, last = frames.get('first', {}), frames.get('last', {})
        status_pairs[f"{first.get('status', 'missing')}/{last.get('status', 'missing')}"] += 1
        if first.get('status') != 'ok':
            first_failure_reasons[first.get('selection_reason') or first.get('error_type') or 'unknown'] += 1
        diagnostics = last.get('provenance', {}).get('tracking_diagnostics')
        if diagnostics:
            tracking.append({'example_id': example_id, **diagnostics})
    result = {
        'population': len(grouped),
        'dual_endpoint_count': status_pairs['ok/ok'],
        'endpoint_status_pairs': dict(status_pairs),
        'first_failure_reasons': dict(first_failure_reasons),
        'tracks_with_diagnostics': len(tracking),
        'tracks_covering_every_frame': sum(row.get('frame_coverage') == 1 for row in tracking),
        'tracking_diagnostics': tracking,
        'interpretation': 'Automatic coverage only; neither endpoint status nor a stable ID proves correct physical-instance tracking.',
    }
    if baseline_run is not None:
        baseline = _latest(baseline_run)
        transitions = Counter()
        for example_id in sorted(set(grouped) & set(baseline)):
            old_ok = all(baseline[example_id].get(f, {}).get('status') == 'ok' for f in ('first', 'last'))
            new_ok = all(grouped[example_id].get(f, {}).get('status') == 'ok' for f in ('first', 'last'))
            transitions[f"{'ok' if old_ok else 'unavailable'} -> {'ok' if new_ok else 'unavailable'}"] += 1
        result['paired_comparison'] = {
            'common_population': sum(transitions.values()),
            'dual_endpoint_transitions': dict(transitions),
            'baseline_run': str(baseline_run.resolve()),
        }
    write_json(run / 'grounding_diagnostics.json', result)
    return result
