"""Clean-room RoboRewardBench evaluation package.

The package intentionally keeps labels out of model-facing payloads and shares
versioned records across raw evaluation, grounding, and causal attention work.
"""

from .schemas import (
    SCHEMA_VERSION,
    AttentionRecord,
    EpisodeRecord,
    FrameRecord,
    GroundingRecord,
    TargetSpec,
)

__all__ = [
    "SCHEMA_VERSION",
    "EpisodeRecord",
    "FrameRecord",
    "TargetSpec",
    "GroundingRecord",
    "AttentionRecord",
]

