"""Domain dataclasses and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class LivenessMode(StrEnum):
    OFF = "off"
    PASSIVE = "passive"
    ACTIVE = "active"
    BOTH = "both"


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class DetectedFace:
    bbox: BBox
    landmarks: object  # np.ndarray (5,2) — typed loosely to avoid numpy import
    det_score: float
    crop: object  # np.ndarray aligned 112x112
    embedding: object | None = None  # np.ndarray (512,) L2-normed


@dataclass
class MatchResult:
    person_id: str | None
    name: str | None
    distance: float
    is_unknown: bool


@dataclass
class LivenessResult:
    score: float
    is_live: bool
    blink_detected: bool
    texture_score: float


@dataclass
class ProcessedFace:
    bbox: BBox
    person_id: str | None
    name: str | None
    distance: float
    is_unknown: bool
    liveness: LivenessResult | None
    logged: bool = False
    alert_triggered: bool = False


@dataclass
class FrameResult:
    faces: list[ProcessedFace] = field(default_factory=list)
    fps: float = 0.0
    camera_ok: bool = True
