"""Isolated evaluation support for locally collected counterfactual datasets."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "audit_prepared": (".data", "audit_prepared"),
    "load_labels": (".data", "load_labels"),
    "load_model_inputs": (".data", "load_model_inputs"),
    "prepare_dataset": (".data", "prepare_dataset"),
    "run_baseline": (".runner", "run_baseline"),
    "score_run": (".metrics", "score_run"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public helpers on demand without importing model runtimes eagerly."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
