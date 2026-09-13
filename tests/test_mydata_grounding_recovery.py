import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mydata_bench.grounding.base import select_unambiguous_candidate
from mydata_bench.grounding.recovery import recover_terminal_track
from mydata_bench.grounding.sam3 import SAM3Grounder
from mydata_bench.grounding.visual import VisualCategoryResolver, parse_visual_boxes


@pytest.mark.parametrize('response', [
    '{"boxes": [[100, 200, 300, 400]]}',
    '```json\n[{"boxes": [100, 200, 300, 400], "label": "salmon sushi"}]\n```',
])
def test_visual_coordinate_formats(response):
    assert parse_visual_boxes(response) == [[100., 200., 300., 400.]]


@pytest.mark.parametrize('box', [
    [200, 200, 100, 300], [-1, 0, 100, 100], [0, 0, 1001, 100],
    [True, 0, 100, 100], [0, 0, float('nan'), 100], [0, 0, 0, 100],
])
def test_visual_boxes_reject_bad_coordinates(box):
    with pytest.raises(ValueError):
        parse_visual_boxes(json.dumps({'boxes': [box]}))


def test_visual_category_filter_reuses_masks_without_inventing_confidence(tmp_path, monkeypatch):
    image = tmp_path / 'frame.png'
    Image.new('RGB', (200, 100)).save(image)
    resolver = VisualCategoryResolver({})
    monkeypatch.setattr(resolver, '_propose', lambda *args: ([[100, 200, 300, 400]], {}))
    mask = np.ones((100, 200), np.uint8)
    candidates = [
        {'bbox': [20, 20, 60, 40], 'score': .8, 'semantic_query': 'salmon sushi', '_mask': mask},
        {'bbox': [100, 30, 120, 60], 'score': .98, 'semantic_query': 'salmon sushi'},
    ]
    rows, _ = resolver.resolve(str(image), ['salmon sushi'], candidates)
    assert len(rows) == 1 and rows[0]['_mask'] is mask
    assert rows[0]['sam3_detector_score'] == .8
    assert rows[0]['score_kind'] == 'unscored_visual_proposal'
    assert rows[0]['bbox'] == [20, 20, 60, 40]
    # Two real instances remain ambiguous regardless of their model output order.
    monkeypatch.setattr(resolver, '_propose', lambda *args: (
        [[100, 200, 300, 400], [500, 300, 600, 600]], {}))
    rows, _ = resolver.resolve(str(image), ['salmon sushi'], candidates)
    assert select_unambiguous_candidate(rows)[0] is None


def test_backward_tracking_uses_reverse_request_and_chronological_coverage(monkeypatch):
    import cv2

    output = {'out_obj_ids': [3], 'out_probs': [.9], 'out_boxes_xywh': [[.1, .2, .2, .2]],
              'out_binary_masks': np.ones((1, 100, 100), np.uint8)}
    requests = []

    class Predictor:
        def handle_request(self, request):
            requests.append(request)
            return {'session_id': 'x'} if request['type'] == 'start_session' else {'outputs': output}

        def handle_stream_request(self, request):
            requests.append(request)
            for i in [2, 1, 0]:
                yield {'frame_index': i, 'outputs': output}

    capture = SimpleNamespace(isOpened=lambda: True, release=lambda: None,
                              get=lambda key: 3 if key == cv2.CAP_PROP_FRAME_COUNT else 100)
    monkeypatch.setattr('mydata_bench.grounding.sam3.cv2.VideoCapture', lambda _: capture)
    grounder = SAM3Grounder({})
    grounder._video_predictor = Predictor()
    tracks = grounder.track('video', [10, 20, 30, 40], 2, terminal_index=0,
                            propagation_direction='backward')
    assert [row['frame_index'] for row in tracks] == [0, 1, 2]
    propagation = next(r for r in requests if r['type'] == 'propagate_in_video')
    assert propagation['propagation_direction'] == 'backward'
    assert propagation['start_frame_index'] == 2
    assert grounder.last_tracking_diagnostics['frame_coverage'] == 1
    assert '_mask' in tracks[0] and '_mask' not in tracks[1] and '_mask' in tracks[2]
    assert requests[-1]['type'] == 'close_session'


def test_overlapping_visual_instances_are_not_collapsed_after_sam3_snapping(tmp_path, monkeypatch):
    image = tmp_path / 'carrots.png'
    Image.new('RGB', (640, 480)).save(image)
    resolver = VisualCategoryResolver({})
    monkeypatch.setattr(resolver, '_propose', lambda *args: (
        [[852, 700, 968, 845], [845, 723, 958, 856]], {}))
    candidates = [
        {'bbox': [545.85, 347.95, 608.84, 407.97], 'score': .968, 'semantic_query': 'carrot'},
        {'bbox': [563.24, 334.21, 617.65, 399.74], 'score': .968, 'semantic_query': 'carrot'},
    ]
    rows, _ = resolver.resolve(str(image), ['carrot'], candidates)
    assert len(rows) == 2
    assert select_unambiguous_candidate(rows)[0] is None


class ReverseGrounder:
    def __init__(self, matches):
        self.config = {'terminal_recovery': {'max_candidates': 3, 'anchor_iou': .5}}
        self.last_tracking_diagnostics = {'obj_id': 7, 'terminal_present': False, 'frame_coverage': 1/3}
        self.matches = matches
        self.calls = []

    def candidates(self, *args):
        return [{'bbox': [10 + 30*i, 10, 20 + 30*i, 20], 'score': .9} for i in range(len(self.matches))]

    def track(self, video_path, box, index, **kwargs):
        self.calls.append((index, kwargs))
        i = int((box[0] - 10) // 30)
        first = [10, 10, 20, 20] if self.matches[i] else [70, 10, 80, 20]
        self.last_tracking_diagnostics = {'obj_id': i, 'frame_coverage': 2/3, 'propagation_direction': 'backward'}
        return [{'frame_index': 0, 'bbox': first, 'obj_id': i}, {'frame_index': 2, 'bbox': box, 'obj_id': i}]


def test_terminal_recovery_requires_unique_reverse_identity_and_keeps_gaps():
    grounder = ReverseGrounder([False, True])
    forward = [{'frame_index': 0, 'bbox': [10, 10, 20, 20], 'obj_id': 7}]
    result = recover_terminal_track(grounder, 'video', forward[0], 'last.png', ['cup'], 0, 2, forward)
    assert [r['obj_id'] for r in result] == [1, 1]
    assert [r['frame_index'] for r in result] == [0, 2]
    assert grounder.last_tracking_diagnostics['missing_frame_count'] == 1
    assert grounder.last_tracking_diagnostics['terminal_present']
    assert grounder.last_tracking_diagnostics['terminal_recovery']['status'] == 'accepted'
    assert all(index == 2 and kwargs['terminal_index'] == 0 for index, kwargs in grounder.calls)


@pytest.mark.parametrize('matches,status', [
    ([False], 'reverse_identity_not_verified'),
    ([True, True], 'ambiguous_reverse_identity'),
    ([True] * 4, 'too_many_terminal_candidates'),
])
def test_terminal_recovery_cannot_choose_arbitrary_or_unchecked_candidate(matches, status):
    grounder = ReverseGrounder(matches)
    forward = [{'frame_index': 0, 'bbox': [10, 10, 20, 20], 'obj_id': 7}]
    result = recover_terminal_track(grounder, 'video', forward[0], 'last.png', ['cup'], 0, 2, forward)
    assert result is forward
    assert grounder.last_tracking_diagnostics['obj_id'] == 7
    assert grounder.last_tracking_diagnostics['terminal_recovery']['status'] == status
    if len(matches) > 3:
        assert not grounder.calls


@pytest.mark.parametrize('verified', [True, False])
def test_tracking_failure_rechecks_semantics_before_recovery(tmp_path, monkeypatch, verified):
    from mydata_bench.grounding import pipeline
    from mydata_bench.grounding.parser import heuristic_parse
    from mydata_bench.io import read_jsonl, write_jsonl
    from mydata_bench.schemas import EpisodeRecord

    frame = tmp_path / 'frame.png'
    Image.new('RGB', (100, 100)).save(frame)
    episode = EpisodeRecord('x', 'video', 'Pick up the cup and place it in the plate.', 1, 'test', 'a'*64)
    config = {'grounding': {'output_dir': str(tmp_path / 'run')}, 'sam3': {
        'tracking': True, 'tracking_preview': False,
        'visual_resolver': {'enabled': True, 'verify_on_tracking_failure': True},
    }}
    write_jsonl(tmp_path / 'run/targets.jsonl', [heuristic_parse(episode.task, 'x').to_dict()])
    monkeypatch.setattr(pipeline, 'load_configured_episodes', lambda _: ([episode], None))
    monkeypatch.setattr(pipeline, 'extract_endpoints', lambda *args: SimpleNamespace(
        first_path=str(frame), last_path=str(frame), first_index=0, last_index=2))

    class Grounder:
        fingerprint = 'fake'

        def __init__(self, config):
            self.last_tracking_diagnostics = {}

        def candidates(self, *args):
            return [{'bbox': [10, 10, 20, 20], 'score': .9}]

        def visual_candidates(self, *args):
            return ([{'bbox': [50, 10, 60, 20], 'score': 0.0}] if verified else []), {'queries': []}

        def track(self, video, bbox, *args, **kwargs):
            indices = [0, 2] if bbox[0] == 50 else [0]
            self.last_tracking_diagnostics = {'frame_coverage': len(indices)/3}
            return [{'bbox': bbox, 'frame_index': i, 'obj_id': 1, 'score': .9} for i in indices]

    monkeypatch.setattr(pipeline, 'SAM3Grounder', Grounder)
    rows = list(read_jsonl(pipeline.run_grounding(config, 'sam3')))
    assert [r['status'] for r in rows] == (['ok', 'ok'] if verified else ['no_detection', 'no_detection'])
    assert rows[0]['provenance']['visual_grounding']['trigger'] == 'forward_tracking_missing_terminal'
    if verified:
        assert rows[0]['bbox'] == rows[1]['bbox'] == [50, 10, 60, 20]
