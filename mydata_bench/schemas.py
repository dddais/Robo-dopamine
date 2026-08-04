from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

SCHEMA_VERSION = "1.0.0"


class Record:
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = self.schema_version
        return value


@dataclass(frozen=True)
class EpisodeRecord(Record):
    example_id: str
    video_path: str
    task: str
    reward: int
    subset: str
    video_sha256: str
    split: str = "test"
    gpt5_mini_check: str | None = None

    def __post_init__(self) -> None:
        if not self.example_id or not self.task or not self.subset:
            raise ValueError("Episode identity, task, and subset must be non-empty")
        if self.reward not in {1, 2, 3, 4, 5}:
            raise ValueError(f"reward must be 1..5, got {self.reward}")
        if self.video_sha256 and len(self.video_sha256) != 64:
            raise ValueError("video_sha256 must be empty or a SHA-256 hex digest")

    def model_payload(self) -> dict[str, Any]:
        """Return the only episode fields permitted before final metric joins."""
        return {
            "example_id": self.example_id,
            "video_path": self.video_path,
            "task": self.task,
            "subset": self.subset,
            "video_sha256": self.video_sha256,
        }


@dataclass(frozen=True)
class FrameRecord(Record):
    example_id: str
    video_sha256: str
    first_index: int
    last_index: int
    first_path: str
    last_path: str
    width: int
    height: int
    first_sha256: str
    last_sha256: str
    reported_frame_count: int
    last_decode_fallback: int = 0


@dataclass(frozen=True)
class TargetSpec(Record):
    example_id: str
    target_phrase: str
    head_noun: str
    attributes: tuple[str, ...] = ()
    entity_type: str = "object"
    parent_object: str | None = None
    reference_object: str | None = None
    relation: str | None = None
    targets: tuple[str, ...] = ()
    multi_target: bool = False
    ambiguous: bool = False
    parser: str = "heuristic"
    parser_fingerprint: str = ""
    raw_output: str | None = None

    @property
    def formal_scope(self) -> bool:
        return (
            self.entity_type in {"object", "object_part"}
            and not self.multi_target
            and not self.ambiguous
        )

    def model_payload(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class GroundingRecord(Record):
    example_id: str
    video_sha256: str
    backend: str
    query: tuple[str, ...]
    frame: str
    frame_index: int
    bbox: tuple[float, float, float, float] | None
    mask_path: str | None
    score: float | None
    candidates: tuple[dict[str, Any], ...] = ()
    selection_reason: str = ""
    audit_status: str = "pending"
    provenance: dict[str, Any] = field(default_factory=dict)
    grounding_fingerprint: str = ""
    status: str = "ok"


@dataclass(frozen=True)
class AttentionRecord(Record):
    example_id: str
    video_sha256: str
    ranking_source: str
    heads: tuple[tuple[int, int], ...]
    bias: float
    condition: str
    raw_output: str
    signed_score: float | None
    bbox_positions: tuple[int, ...] = ()
    image_positions: tuple[int, ...] = ()
    query_positions: tuple[int, ...] = ()
    hook_diagnostics: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"


def validate_bbox(
    bbox: list[float] | tuple[float, ...] | None, width: int, height: int
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("bbox must contain x1,y1,x2,y2")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox {bbox} is outside image {width}x{height}")
    return x1, y1, x2, y2

