import pytest

torch = pytest.importorskip("torch")

from mydata_bench.attention_eval.masking import make_attention_mask_hook


def test_per_head_signed_attention_biases_are_additive_and_opposite():
    diagnostics = {}
    hook = make_attention_mask_hook(
        [1, 2],
        [1],
        [3],
        4,
        6.0,
        diagnostics,
        query_scope="all",
        head_biases={1: 6.0, 2: -6.0},
    )
    mask = torch.zeros((1, 4, 2, 6), dtype=torch.float32)
    _args, kwargs = hook(None, (), {"attention_mask": mask})
    steered = kwargs["attention_mask"]

    assert torch.all(steered[0, 1, :, 1] == 6.0)
    assert torch.all(steered[0, 1, :, 3] == -6.0)
    assert torch.all(steered[0, 2, :, 1] == -6.0)
    assert torch.all(steered[0, 2, :, 3] == 6.0)
    assert diagnostics["head_biases"] == {"1": 6.0, "2": -6.0}
