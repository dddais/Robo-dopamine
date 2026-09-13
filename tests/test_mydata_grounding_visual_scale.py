import json

import pytest
from PIL import Image

from mydata_bench.grounding.visual import VisualCategoryResolver


def resolver_with_responses(tmp_path, monkeypatch, responses, *, image_size=(640, 480), config=None):
    path = tmp_path / 'frame.png'
    Image.new('RGB', image_size).save(path)
    resolver = VisualCategoryResolver({'model_path': 'fake', **(config or {})})
    calls = []
    def generate(image, query):
        calls.append((image.size if isinstance(image, Image.Image) else None, query))
        return responses[len(calls) - 1]
    monkeypatch.setattr(resolver, '_generate_response', generate)
    return resolver, str(path), calls


def test_empty_retry_retains_scene_and_maps_normalized_boxes_to_original_pixels(tmp_path, monkeypatch):
    resolver, path, calls = resolver_with_responses(tmp_path, monkeypatch, [
        '{"boxes": []}', '{"boxes": [[100, 200, 300, 400], [500, 200, 700, 400]]}',
    ], config={'empty_retry_long_side': 1280})
    rows, diagnostics = resolver.resolve(path, ['shrimp sushi'], [])
    assert calls == [(None, 'shrimp sushi'), ((1280, 960), 'shrimp sushi')]
    assert rows[0]['bbox'] == [64, 96, 192, 192]
    assert rows[1]['bbox'] == [320, 96, 448, 192]
    assert all(row['score_kind'] == 'unscored_visual_proposal' for row in rows)
    assert len({row['visual_instance_index'] for row in rows}) == 2
    diagnostic = diagnostics['queries'][0]
    assert diagnostic['scale_retry']['original_response'] == '{"boxes": []}'
    assert diagnostic['scale_retry']['source_size'] == [640, 480]
    assert len(json.loads(diagnostic['response'])['boxes']) == 2
    resolver.resolve(path, ['shrimp sushi'], [])
    assert len(calls) == 2  # Successful retry is cached with the original image identity.


@pytest.mark.parametrize('responses,config,image_size', [
    (['{"boxes": []}'], {}, (640, 480)),
    (['{"boxes": []}'], {'empty_retry_long_side': 1280}, (1280, 960)),
    (['broken JSON'], {'empty_retry_long_side': 1280}, (640, 480)),
    (['{"boxes": [[100, 100, 200, 200], [300, 100, 400, 200]]}'],
        {'empty_retry_long_side': 1280}, (640, 480)),
])
def test_retry_is_bounded_and_does_not_overwrite_existing_instances(
        tmp_path, monkeypatch, responses, config, image_size):
    resolver, path, calls = resolver_with_responses(
        tmp_path, monkeypatch, responses, config=config, image_size=image_size)
    resolver._propose(path, 'carrot')
    assert len(calls) == 1


@pytest.mark.parametrize('retry', ['{"boxes": []}', 'broken JSON', '{"boxes": [[-1,0,20,20]]}'])
def test_empty_or_invalid_retry_does_not_manufacture_candidates(tmp_path, monkeypatch, retry):
    resolver, path, calls = resolver_with_responses(tmp_path, monkeypatch,
        ['{"boxes": []}', retry], config={'empty_retry_long_side': 1280})
    boxes, diagnostics = resolver._propose(path, 'cup')
    assert not boxes and len(calls) == 2
    assert diagnostics['response'] == retry
    if retry != '{"boxes": []}':
        assert 'error' in diagnostics


@pytest.mark.parametrize('size', [-1, 5000, True, 1280.0, '1280'])
def test_reject_invalid_retry_size(size):
    with pytest.raises(ValueError, match='empty_retry_long_side'):
        VisualCategoryResolver({'empty_retry_long_side': size})


@pytest.mark.parametrize('sam_count,expected_count', [(0, 0), (1, 0), (2, 2)])
def test_scaled_proposals_require_unique_sam_support_for_the_whole_set(tmp_path, monkeypatch, sam_count, expected_count):
    resolver, path, _ = resolver_with_responses(tmp_path, monkeypatch, [
        '{"boxes": []}', '{"boxes": [[100, 200, 300, 400], [500, 200, 700, 400]]}',
    ], config={'empty_retry_long_side': 1280, 'empty_retry_requires_sam_support': True})
    sam = [{'bbox': box, 'score': .9, 'semantic_query': 'mushroom'} for box in
           [[64, 96, 192, 192], [320, 96, 448, 192]][:sam_count]]
    rows, diagnostic = resolver.resolve(path, ['mushroom'], sam)
    assert len(rows) == expected_count
    assert diagnostic['queries'][0]['scale_retry']['accepted'] == bool(expected_count)
    # Diagnostics remain specific to this candidate set even when generated boxes are cached.
    rows, diagnostic = resolver.resolve(path, ['mushroom'], [])
    assert not rows and not diagnostic['queries'][0]['scale_retry']['accepted']


def test_description_keeps_the_original_semantic_query_for_sam_matching(tmp_path, monkeypatch):
    resolver, path, calls = resolver_with_responses(tmp_path, monkeypatch,
        ['{"boxes": [[100, 200, 300, 400]]}'], config={'category_descriptions': {'surf clam sushi': 'red-tipped clam topping'}})
    rows, diagnostic = resolver.resolve(path, ['surf clam sushi'], [
        {'bbox': [64, 96, 192, 192], 'score': .9, 'semantic_query': 'surf clam sushi'}])
    assert rows[0]['sam3_detector_score'] == .9
    assert rows[0]['semantic_query'] == diagnostic['queries'][0]['query'] == 'surf clam sushi'
    assert diagnostic['queries'][0]['category_description'] == 'red-tipped clam topping'


@pytest.mark.parametrize('descriptions', [[], {'cup': ''}, {'cup': 1}, {1: 'cup'}])
def test_invalid_category_descriptions_are_rejected(descriptions):
    with pytest.raises(ValueError, match='category_descriptions'):
        VisualCategoryResolver({'category_descriptions': descriptions})


@pytest.mark.parametrize('has_visual_target', [True, False])
def test_configured_category_is_rechecked_even_when_sam_selection_and_track_would_succeed(
        tmp_path, monkeypatch, has_visual_target):
    from types import SimpleNamespace
    from mydata_bench.grounding.parser import heuristic_parse
    from mydata_bench.io import read_jsonl, write_jsonl
    from mydata_bench.schemas import EpisodeRecord
    from mydata_bench.grounding import pipeline
    frame = tmp_path / 'frame.png'
    Image.new('RGB', (100, 100)).save(frame)
    episode = EpisodeRecord('x', 'video', 'Pick up the surf clam sushi and place it in the plate.', 1, 'test', 'a'*64)
    config = {'grounding': {'output_dir': str(tmp_path / 'run')}, 'sam3': {
        'tracking': True, 'tracking_preview': False,
        'visual_resolver': {'enabled': True, 'always_verify_queries': ['surf clam sushi']},
    }}
    write_jsonl(tmp_path / 'run/targets.jsonl', [heuristic_parse(episode.task, 'x').to_dict()])
    monkeypatch.setattr(pipeline, 'load_configured_episodes', lambda _: ([episode], None))
    monkeypatch.setattr(pipeline, 'extract_endpoints', lambda *args: SimpleNamespace(
        first_path=str(frame), last_path=str(frame), first_index=0, last_index=2))
    tracked = []
    class Grounder:
        fingerprint = 'fake'
        def __init__(self, config):self.last_tracking_diagnostics = {}
        def candidates(self, *args):return [{'bbox': [10, 10, 20, 20], 'score': .9}]
        def visual_candidates(self, *args):
            return ([{'bbox': [50, 10, 60, 20], 'score': 0.0}] if has_visual_target else []), {'queries': []}
        def track(self, video, bbox, *args, **kwargs):
            tracked.append(bbox);self.last_tracking_diagnostics = {'frame_coverage': 2/3}
            return [{'bbox': bbox, 'frame_index': i, 'obj_id': 1, 'score': .9} for i in [0, 2]]
    monkeypatch.setattr(pipeline, 'SAM3Grounder', Grounder)
    rows = list(read_jsonl(pipeline.run_grounding(config, 'sam3')))
    assert rows[0]['provenance']['visual_grounding']['trigger'] == 'configured_category_semantic_verification'
    if has_visual_target:
        assert tracked == [[50, 10, 60, 20]]
        assert all(r['bbox'] == [50, 10, 60, 20] for r in rows)
    else:
        assert not tracked
        assert all(r['status'] == 'no_detection' for r in rows)


def test_terminal_candidates_for_configured_category_are_visually_revalidated():
    from types import SimpleNamespace
    from mydata_bench.grounding import recovery
    seen = []
    class Grounder:
        config = {'terminal_recovery': {'max_candidates': 3, 'anchor_iou': .5},
                  'visual_resolver': {'enabled': True, 'always_verify_queries': ['surf clam sushi']}}
        last_tracking_diagnostics = {'terminal_present': False}
        def candidates(self, *args):return [{'bbox': [10, 10, 20, 20]}]
        def visual_candidates(self, *args):
            seen.append('visual')
            return [{'bbox': [50, 10, 60, 20]}], {'queries': []}
        def track(self, video, bbox, *args, **kwargs):
            seen.append(bbox);self.last_tracking_diagnostics = {'frame_coverage': 2/3}
            return [{'bbox': [1, 1, 5, 5], 'frame_index': 0, 'obj_id': 1},
                    {'bbox': bbox, 'frame_index': 2, 'obj_id': 1}]
    grounder = Grounder()
    result = recovery.recover_terminal_track(grounder, 'video', {'bbox': [1, 1, 5, 5]},
        'last.png', ['surf clam sushi'], 0, 2, [])
    assert seen == ['visual', [50, 10, 60, 20]]
    assert result[-1]['bbox'] == [50, 10, 60, 20]
