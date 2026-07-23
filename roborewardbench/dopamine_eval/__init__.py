"""Instruction-conditioned object grounding for Robo-Dopamine evaluation.

The package deliberately keeps language parsing and visual grounding separate:

1. local text-only LLMs extract the directly manipulated entity;
2. GroundingDINO localizes that entity in the benchmark endpoint frames;
3. deterministic aggregation records agreement, confidence, and temporal
   consistency without using reward labels or ``gpt5_mini_check`` text.
"""

from .instruction_parser import (
    TARGET_TYPES,
    build_grounding_queries,
    compare_parses,
    heuristic_parse,
    normalize_parse_payload,
)

__all__ = [
    "TARGET_TYPES",
    "build_grounding_queries",
    "compare_parses",
    "heuristic_parse",
    "normalize_parse_payload",
]
