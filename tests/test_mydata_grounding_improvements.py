from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from mydata_bench.grounding import pipeline
from mydata_bench.grounding.base import select_relational_candidate, select_unambiguous_candidate
from mydata_bench.grounding.parser import build_queries, heuristic_parse, normalize_target
from mydata_bench.grounding.sam3 import SAM3Grounder, _interior_point
from mydata_bench.io import read_jsonl, write_jsonl
from mydata_bench.schemas import EpisodeRecord


@pytest.mark.parametrize('task,subject,relation,rank', [
    ('Pick up the second shrimp sushi from the left and put it in the plate.', 'shrimp sushi', 'nth_from_left', 2),
    ('Grasp the 3rd red cup from the right.', 'red cup', 'nth_from_right', 3),
    ('Lift the leftmost pen.', 'pen', 'nth_from_left', 1),
    ('Pick up the left pen and place it in the box.', 'pen', 'nth_from_left', 1),
    ('Touch the pen on the right.', 'pen', 'nth_from_right', 1),
    ('Pick up the salmon sushi to the left of the red block and put it in the plate.', 'salmon sushi', 'left_of', None),
    ('Grasp the blue cup nearest to the teapot.', 'blue cup', 'closest_to', None),
])
def test_spatial_parser_preserves_semantic_subject(task, subject, relation, rank):
    target = heuristic_parse(task)
    assert target.relation == relation
    assert target.ordinal_index == rank
    assert build_queries(target) == [subject]
    assert not target.multi_target


def test_queries_do_not_drop_color_subtype_or_object_part():
    for phrase, head in [('red cup', 'cup'), ('salmon sushi', 'sushi')]:
        target = normalize_target('x', {'target_phrase': phrase, 'head_noun': head}, parser='test', parser_fingerprint='x')
        assert build_queries(target) == [phrase]
    target = heuristic_parse("Open the teapot's lid.")
    assert 'teapot' not in build_queries(target)
    target = normalize_target('x', {'target_phrase': 'red cup', 'head_noun': 'cup', 'subject_phrase': 'cup'},
                              parser='test', parser_fingerprint='x')
    assert build_queries(target) == ['red cup']
    assert heuristic_parse('Pick the red cup then lift the blue cup.').multi_target


def test_legacy_ordinal_parse_gets_geometry_without_rerunning_llm():
    target = pipeline._target_from_row({'example_id': 'x', 'target_phrase': 'fourth cup from the left',
                                        'head_noun': 'cup', 'relation': None})
    assert target.relation == 'nth_from_left'
    assert target.ordinal_index == 4
    assert build_queries(target) == ['cup']


def candidate(x, score=0.9, **kwargs):
    return {'bbox': [x, 10, x + 10, 25], 'score': score, 'query': 'cup', **kwargs}


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / 'frame.png'
    Image.new('RGB', (200, 100), 'white').save(path)
    return str(path)


def test_ordinal_counts_instances_not_duplicate_queries(frame):
    rows = [candidate(10, .99), candidate(10.1, .98), candidate(60, .7), candidate(100, .95)]
    selected, _, _ = select_relational_candidate(frame, rows, [], 'nth_from_left', ordinal_index=2)
    assert selected['bbox'][0] == 60
    assert selected['instance_count'] == 3
    assert select_relational_candidate(frame, rows, [], 'nth_from_left', ordinal_index=4)[0] is None
    assert select_relational_candidate(frame, rows, [], 'nth_from_right', ordinal_index=1)[0]['bbox'][0] == 100


def test_ordinal_and_reference_ambiguity_are_explicit(frame):
    rows = [candidate(10), {'bbox': [10.5, 45, 20.5, 60], 'score': .95}]
    assert select_relational_candidate(frame, rows, [], 'nth_from_left', ordinal_index=1)[2] == 'ambiguous_ordinal_geometry'
    refs = [candidate(60, .9), candidate(100, .89)]
    assert select_relational_candidate(frame, [candidate(10)], refs, 'left_of')[2] == 'ambiguous_reference_candidates'


def test_query_specificity_and_dedup_before_ambiguity():
    rows = [candidate(10, .7, query_priority=0), candidate(10, .99, query_priority=1), candidate(80, .98, query_priority=1)]
    assert select_unambiguous_candidate(rows)[0]['bbox'][0] == 10
    assert select_unambiguous_candidate([candidate(10), candidate(80, .89)])[0] is None
    assert select_unambiguous_candidate([candidate(10), candidate(10.1, .89)])[0] is not None


def test_reference_ambiguity_can_preserve_a_unique_target(frame):
    targets = [candidate(10), candidate(160)]
    references = [candidate(60), candidate(80, .89)]
    selected, reference, reason = select_relational_candidate(
        frame, targets, references, 'closest_to', reference_consensus=True)
    assert selected['bbox'] == targets[0]['bbox']
    assert len(selected['reference_consensus_boxes']) == 2
    assert 'reference_bbox' not in selected
    assert reference is None
    assert reason.endswith('reference_consensus')
    # Moving the second reference changes which target is nearest: still reject.
    references[1] = candidate(130, .89)
    assert select_relational_candidate(
        frame, targets, references, 'closest_to', reference_consensus=True)[0] is None
    # One reference makes a directional relation impossible: do not ignore it.
    assert select_relational_candidate(
        frame, [candidate(90)], references, 'left_of', reference_consensus=True)[0] is None


def test_interior_point_is_inside_nonconvex_mask():
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (50, 50), 30, 1, 5)
    x, y = _interior_point(mask, [18, 18, 82, 82], 100, 100)
    assert mask[int(y * 100), int(x * 100)] == 1
    assert mask[50, 50] == 0  # The previous centroid prompt would hit background.


def track_output(ids=(7,), boxes=None):
    return {'out_obj_ids': np.array(ids), 'out_probs': np.array([.9] * len(ids)),
            'out_boxes_xywh': np.array(boxes if boxes is not None else [[.1, .2, .2, .2]] * len(ids)),
            'out_binary_masks': np.ones((len(ids), 10, 10), dtype=bool)}


def test_missing_id_never_switches_to_higher_confidence_object():
    assert SAM3Grounder._one_track_output(track_output((9,)), 3, 100, 100, 7) is None
    assert SAM3Grounder._one_track_output(track_output((7, 9)), 3, 100, 100, 7)['obj_id'] == 7
    output = track_output()
    output['out_binary_masks'][:] = 0
    assert SAM3Grounder._one_track_output(output, 3, 100, 100, 7) is None
    output = track_output()
    output['out_boxes_xywh'][0, 0] = np.nan
    assert SAM3Grounder._one_track_output(output, 3, 100, 100, 7) is None


class Capture:
    def isOpened(self):
        return True

    def get(self, key):
        return 100 if key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) else 3

    def release(self):
        pass


def test_refinement_requires_cache_replaces_old_pass_and_closes(monkeypatch):
    class Predictor:
        def __init__(self):
            self.passes = 0
            self.requests = []

        def handle_request(self, request):
            self.requests.append(request)
            if request['type'] == 'start_session':
                return {'session_id': 'test'}
            if request['type'] == 'add_prompt':
                if 'points' in request:
                    assert self.passes == 1
                    assert request['obj_id'] == 7
                return {'outputs': track_output()}
            return {}

        def handle_stream_request(self, request):
            self.passes += 1
            # First pass lacks terminal; correction loses frame 1 but recovers terminal.
            frames = [0, 1] if self.passes == 1 else [0, 2]
            for index in frames:
                yield {'frame_index': index, 'outputs': track_output()}

    predictor = Predictor()
    grounder = SAM3Grounder({})
    grounder._video_predictor = predictor
    monkeypatch.setattr('mydata_bench.grounding.sam3.cv2.VideoCapture', lambda _: Capture())
    rows = grounder.track('video', [10, 20, 30, 40])
    assert [r['frame_index'] for r in rows] == [0, 2]
    assert grounder.last_tracking_diagnostics['missing_frame_count'] == 1
    assert grounder.last_tracking_diagnostics['longest_missing_run'] == 1
    assert predictor.requests[-1]['type'] == 'close_session'


def test_visual_anchor_binds_overlap_instead_of_top_score(monkeypatch):
    class Predictor:
        def handle_request(self, request):
            if request['type'] == 'start_session':
                return {'session_id': 'test'}
            output = track_output((9, 7), [[.7, .2, .2, .2], [.1, .2, .2, .2]])
            output['out_probs'] = np.array([.99, .7])
            return {'outputs': output}

        def handle_stream_request(self, request):
            for i in range(3):
                yield {'frame_index': i, 'outputs': track_output((7,))}

    grounder = SAM3Grounder({})
    grounder._video_predictor = Predictor()
    monkeypatch.setattr('mydata_bench.grounding.sam3.cv2.VideoCapture', lambda _: Capture())
    assert {r['obj_id'] for r in grounder.track('video', [10, 20, 30, 40])} == {7}


def test_tiles_restore_coordinates_and_masks_and_cache_is_copy_safe(frame, monkeypatch):
    grounder = SAM3Grounder({'detection_tile_grid': 2, 'candidate_cache_size': 2})
    monkeypatch.setattr(grounder, '_load', lambda: None)
    calls = []

    def detect(image, query):
        calls.append(image.size)
        if image.size == (200, 100):
            return []
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[10:25, 10:20] = 1
        return [{'bbox': [10, 10, 20, 25], 'score': .9, '_mask': mask}]

    monkeypatch.setattr(grounder, '_image_candidates', detect)
    rows = grounder.candidates(frame, ['cup'])
    assert len(rows) == 4
    assert max(r['bbox'][0] for r in rows) > 90
    for row in rows:
        assert row['_mask'].shape == (100, 200)
        x, y = int(row['bbox'][0]), int(row['bbox'][1])
        assert row['_mask'][y, x]
    rows[0].pop('_mask')
    assert all('_mask' in r for r in grounder.candidates(frame, ['cup']))
    assert len(calls) == 5


def test_aliases_keep_modifiers_and_do_not_match_partial_words():
    grounder = SAM3Grounder({'query_aliases': {'cup': ['small bowl'], 'shrimp sushi': ['shrimp']}})
    assert grounder._query_variants('red cup') == ['red cup', 'red small bowl']
    assert grounder._query_variants('cupboard') == ['cupboard']
    assert grounder._query_variants('shrimp sushi') == ['shrimp sushi', 'shrimp']


def test_resume_replaces_both_endpoints_and_preserves_old_track(tmp_path, frame, monkeypatch):
    episode = EpisodeRecord('x', 'video', 'Pick the cup.', 1, 'test', 'a' * 64)
    config = {'grounding': {'output_dir': str(tmp_path / 'run')},
              'sam3': {'tracking': True, 'tracking_preview': False}}
    target = heuristic_parse(episode.task, episode.example_id)
    write_jsonl(tmp_path / 'run/targets.jsonl', [target.to_dict()])
    monkeypatch.setattr(pipeline, 'load_configured_episodes', lambda _: ([episode], None))
    monkeypatch.setattr(pipeline, 'extract_endpoints', lambda *args: SimpleNamespace(
        first_path=frame, last_path=frame, first_index=0, last_index=2))

    class FakeGrounder:
        attempts = 0
        fingerprint = 'backend-v2'
        last_tracking_diagnostics = {'frame_coverage': 1.0}

        def __init__(self, config):
            pass

        def candidates(self, *args):
            return [candidate(10 + 20 * self.attempts)]

        def track(self, video, bbox, anchor, **kwargs):
            type(self).attempts += 1
            return [] if self.attempts == 1 else [{'frame_index': 2, 'bbox': bbox, 'score': .9, 'obj_id': 1}]

    monkeypatch.setattr(pipeline, 'SAM3Grounder', FakeGrounder)
    path = pipeline.run_grounding(config, 'sam3')
    before = list(read_jsonl(path))
    assert [r['status'] for r in before] == ['ok', 'no_detection']
    old_track = Path(before[0]['provenance']['tracking_path'])
    old_bytes = old_track.read_bytes()
    pipeline.run_grounding(config, 'sam3')
    assert FakeGrounder.attempts == 1  # Failure retries are explicit.
    pipeline.run_grounding(config, 'sam3', retry_failed=True)
    after = list(read_jsonl(path))[-2:]
    assert [r['status'] for r in after] == ['ok', 'ok']
    assert after[0]['bbox'] != before[0]['bbox']
    assert after[0]['provenance']['tracking_path'] == after[1]['provenance']['tracking_path']
    assert after[0]['provenance']['tracking_path'] != str(old_track)
    assert old_track.read_bytes() == old_bytes
    pipeline.run_grounding(config, 'sam3', retry_failed=True)
    assert FakeGrounder.attempts == 2
    FakeGrounder.fingerprint = 'backend-v3'
    pipeline.run_grounding(config, 'sam3')
    assert FakeGrounder.attempts == 3
    FakeGrounder.fingerprint = 'backend-v4'

    def fail_tracking(*args, **kwargs):
        raise RuntimeError('video backend failure')

    monkeypatch.setattr(FakeGrounder, 'track', fail_tracking)
    pipeline.run_grounding(config, 'sam3')
    failed = list(read_jsonl(path))[-2:]
    assert [r['status'] for r in failed] == ['ok', 'invalid']
    assert failed[-1]['provenance']['tracking_error']['error'] == 'video backend failure'


def test_exact_frame_tracks_never_borrow_a_future_bbox(tmp_path, monkeypatch):
    from mydata_bench.attention_eval import runtime
    path = tmp_path / 'track.json'
    payload = {'terminal_frame_index': 40, 'bbox_frame_policy': 'exact',
               'frames': [{'frame_index': i, 'bbox': [10, 10, 20, 20]} for i in (0, 21, 40)]}
    path.write_text(json.dumps(payload))
    sample = {'video_sha256': 'a' * 64, 'last': {'frame_index': 40, 'provenance': {
        'tracking_path': str(path),
        'view_endpoint_paths': {v: {'first': 'first.png', 'last': 'last.png'} for v in ('front', 'left_wrist', 'right_wrist')},
        'view_paths': {v: 'video.mp4' for v in ('front', 'left_wrist', 'right_wrist')},
    }}}
    monkeypatch.setattr(runtime, 'extract_frame_at', lambda _, p, i: (i, str(p)))
    with pytest.raises(ValueError, match='source frame 20'):
        runtime.incremental_steps(sample, {'output_dir': str(tmp_path), 'frame_interval': 20})
    payload.pop('bbox_frame_policy')
    path.write_text(json.dumps(payload))
    # Historical artifact semantics remain reproducible.
    steps = runtime.incremental_steps(sample, {'output_dir': str(tmp_path), 'frame_interval': 20})
    assert steps[0]['after_bbox_frame_index'] == 21


def test_cohort_optional_coverage_and_attempt_consistency(tmp_path, monkeypatch):
    from mydata_bench import cohorts
    episodes = [EpisodeRecord(k, 'video', 'Pick cup.', 1, 'test', 'a' * 64) for k in ('full', 'gaps', 'mixed', 'legacy')]
    monkeypatch.setattr(cohorts, 'load_episodes', lambda *args: episodes)
    rows = []
    for episode in episodes:
        for frame_name in ('first', 'last'):
            provenance = {'input_fingerprint': 'one', 'tracking_path': 'one',
                          'tracking_diagnostics': {'frame_coverage': .6 if episode.example_id == 'gaps' else 1.0}}
            if episode.example_id == 'mixed' and frame_name == 'last':
                provenance['tracking_path'] = 'two'
            if episode.example_id == 'legacy':
                provenance = {}
            rows.append({'example_id': episode.example_id, 'frame': frame_name,
                         'status': 'ok', 'provenance': provenance})
    write_jsonl(tmp_path / 'run/grounding.jsonl', rows)
    result = cohorts.freeze_auto_grounded_cohort(tmp_path, tmp_path / 'run', tmp_path / 'all')
    assert result['selected_count'] == 3
    result = cohorts.freeze_auto_grounded_cohort(tmp_path, tmp_path / 'run', tmp_path / 'full', min_tracking_coverage=1.0)
    assert json.loads(Path(result['example_ids_file']).read_text()) == ['full']


def test_diagnostics_uses_latest_failure_and_paired_population(tmp_path):
    from mydata_bench.grounding.diagnostics import diagnose
    rows = [{'example_id': 'x', 'frame': f, 'status': 'ok'} for f in ('first', 'last')]
    write_jsonl(tmp_path / 'old/grounding.jsonl', rows)
    write_jsonl(tmp_path / 'new/grounding.jsonl', rows + [{'example_id': 'x', 'frame': 'last', 'status': 'no_detection'}])
    result = diagnose(tmp_path / 'new', tmp_path / 'old')
    assert result['dual_endpoint_count'] == 0
    assert result['paired_comparison']['dual_endpoint_transitions'] == {'ok -> unavailable': 1}
