"""RoboReward-8B attention entry point.

RoboReward-8B is a Qwen3-VL checkpoint, so it uses the common Qwen attention
engine while retaining a model-specific configuration, output directory and
command under ``roboreward_eval``.  Native video and the explicit sampled-image
ablation are allowed; the Robo-Dopamine progress prompt is not.
"""

from __future__ import annotations

from ..config import load_config
from ..qwen_eval.attention_cli import main as _qwen_main


def validate_roboreward_config(path: str) -> None:
    config = load_config(path)
    attention = config.get("attention_steer", {})
    allowed = {
        "roborewardbench_native",
        "roborewardbench_image_sequence",
        "roborewardbench_interleaved_image_sequence",
    }
    if attention.get("protocol") not in allowed:
        raise ValueError(
            "RoboReward attention supports only the discrete native-video or "
            "sampled-image protocols"
        )


def main(argv: list[str] | None = None) -> None:
    # Keep parser behavior identical to Qwen's runner, after enforcing the
    # native-only contract in a lightweight pre-parse pass.
    import sys

    values = list(sys.argv[1:] if argv is None else argv)
    if "--config" in values:
        index = values.index("--config")
        if index + 1 < len(values):
            validate_roboreward_config(values[index + 1])
    _qwen_main(values)
