from types import SimpleNamespace

import numpy as np
import pytest

from mydata_bench.grounding import sam3 as module
SAM3Grounder = module.SAM3Grounder
_interior_point = module._interior_point
_interior_points = module._interior_points


def test_distributed_prompts_cover_cap_and_stem_without_entering_background():
    mask = np.zeros((80, 100), np.uint8)
    mask[10:30, 20:70] = 1
    mask[25:65, 35:60] = 1
    prompts = _interior_points(mask, [20, 10, 70, 65], 100, 80, 3)
    assert len(prompts) == 3
    assert prompts[0] == _interior_point(mask, [20, 10, 70, 65], 100, 80)
    pixels = [(int(x*100), int(y*80)) for x, y in prompts]
    assert all(mask[y, x] for x, y in pixels)
    assert min(y for x, y in pixels) < 30 and max(y for x, y in pixels) > 40
    assert len(set(pixels)) == 3


def test_ring_prompts_stay_outside_the_hole():
    mask = np.zeros((50, 50), np.uint8)
    mask[5:45, 5:45] = 1
    mask[15:35, 15:35] = 0
    prompts = _interior_points(mask, [5, 5, 45, 45], 50, 50, 3)
    assert all(mask[int(y*50), int(x*50)] for x, y in prompts)


@pytest.mark.parametrize('mask', [None, np.zeros((20, 20), np.uint8)])
def test_no_mask_does_not_invent_additional_foreground_points(mask):
    assert _interior_points(mask, [2, 4, 10, 12], 20, 20, 3) == [[.3, .4]]


def test_tiny_mask_avoids_duplicate_points():
    mask = np.zeros((10, 10), np.uint8);mask[5, 5] = 1
    assert _interior_points(mask, [5, 5, 6, 6], 10, 10, 3) == [[.55, .55]]


@pytest.mark.parametrize('count', [0, 9, True, 1.5, '3'])
def test_invalid_point_count(count):
    with pytest.raises(ValueError, match='tracking_positive_points'):
        _interior_points(None, [0, 0, 2, 2], 10, 10, count)


def test_tracker_sends_multiple_positive_points_and_keeps_identity(monkeypatch):
    import cv2
    mask = np.zeros((100, 100), np.uint8);mask[20:70, 10:40] = 1
    output = {'out_obj_ids': [7], 'out_probs': [.9], 'out_boxes_xywh': [[.1, .2, .3, .5]],
              'out_binary_masks': mask[None]}
    requests = []
    class Predictor:
        def handle_request(self, request):
            requests.append(request)
            return {'session_id': 'x'} if request['type'] == 'start_session' else {'outputs': output}
        def handle_stream_request(self, request):
            for i in [0, 1, 2]:yield {'frame_index': i, 'outputs': output}
    capture = SimpleNamespace(isOpened=lambda: True, release=lambda: None,
        get=lambda key: 3 if key == cv2.CAP_PROP_FRAME_COUNT else 100)
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda _: capture)
    grounder = SAM3Grounder({'tracking_prompt': 'mask_refine', 'tracking_positive_points': 3})
    grounder._video_predictor = Predictor()
    rows = grounder.track('video', [10, 20, 40, 70], 0, anchor_mask=mask, terminal_index=2)
    prompt = next(r for r in requests if r.get('points'))
    assert len(prompt['points']) == 3 and prompt['point_labels'] == [1, 1, 1]
    assert prompt['obj_id'] == 7 and {r['obj_id'] for r in rows} == {7}
    assert grounder.last_tracking_diagnostics['prompt_point_count'] == 3
    assert grounder.last_tracking_diagnostics['frame_coverage'] == 1
    assert requests[-1]['type'] == 'close_session'
