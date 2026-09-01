from mydata_bench.attention_eval.masking import Head
import pytest

from mydata_bench.gaze_debug import (
    find_subsequence,
    matched_random_heads,
    select_condition_specs,
)


def test_find_subsequence_can_choose_first_or_last():
    sequence = [9, 1, 2, 8, 1, 2, 7]
    assert find_subsequence(sequence, [1, 2]) == [1, 2]
    assert find_subsequence(sequence, [1, 2], last=True) == [4, 5]
    assert find_subsequence(sequence, [3]) == []


def test_random_heads_match_layer_distribution_and_exclude_gaze():
    gaze = [Head(19, 1), Head(19, 2), Head(23, 4), Head(27, 7)]
    first = matched_random_heads(gaze, num_heads=8, seed=123)
    second = matched_random_heads(gaze, num_heads=8, seed=123)
    assert first == second
    assert [item.layer for item in first] == [19, 19, 23, 27]
    assert not set(first) & set(gaze)


def test_select_condition_specs_preserves_requested_order_and_rejects_unknown():
    conditions = [
        ("baseline", [], 0.0, "all", "none"),
        ("gaze_decode", [], 6.0, "decode", "all_visual"),
    ]
    assert [row[0] for row in select_condition_specs(conditions, ["gaze_decode"])] == [
        "gaze_decode"
    ]
    assert select_condition_specs(conditions, None) == conditions
    with pytest.raises(ValueError, match="Unknown gaze-debug conditions"):
        select_condition_specs(conditions, ["typo"])
