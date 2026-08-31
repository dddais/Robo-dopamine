import numpy as np

from mydata_bench.qwen_eval.attention_experiment import _adaptive_router_decision


def test_adaptive_router_uses_frozen_standardization_and_strict_threshold():
    router = {
        "feature_center": [1.0, 2.0],
        "feature_scale": [2.0, 4.0],
        "coefficients_with_intercept": [0.25, 2.0, -1.0],
        "fail_branch_if_score_below": 0.0,
    }
    score, branch, standardized = _adaptive_router_decision([3.0, 6.0], router)
    assert np.allclose(standardized, [1.0, 1.0])
    assert score == 1.25
    assert branch == "success"

    score, branch, _ = _adaptive_router_decision([-1.0, 6.0], router)
    assert score == -2.75
    assert branch == "fail"
