from mydata_bench.qwen_eval.runner import _model_load_kwargs as qwen_model_load_kwargs
from mydata_bench.roboreward_eval.runner import (
    _model_load_kwargs as roboreward_model_load_kwargs,
)


def test_baseline_model_load_kwargs_preserve_historical_backend_default():
    dtype = object()
    for builder in (qwen_model_load_kwargs, roboreward_model_load_kwargs):
        kwargs = builder({"device_map": "cpu"}, dtype)
        assert kwargs == {"torch_dtype": dtype, "device_map": "cpu"}
        assert "attn_implementation" not in kwargs


def test_baseline_model_load_kwargs_accept_explicit_backend_override():
    dtype = object()
    for builder in (qwen_model_load_kwargs, roboreward_model_load_kwargs):
        kwargs = builder({"attn_implementation": "eager"}, dtype)
        assert kwargs["attn_implementation"] == "eager"
        assert kwargs["device_map"] == "auto"
