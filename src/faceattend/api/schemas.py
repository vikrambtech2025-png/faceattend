"""Pydantic schemas for API."""

from __future__ import annotations

from pydantic import BaseModel


class FaceBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class LivenessInfo(BaseModel):
    score: float
    is_live: bool
    blink_detected: bool
    texture_score: float


class ProcessedFaceOut(BaseModel):
    bbox: FaceBox
    person_id: str | None = None
    name: str | None = None
    distance: float
    is_unknown: bool
    liveness: LivenessInfo | None = None
    logged: bool = False
    alert_triggered: bool = False


class FrameResultOut(BaseModel):
    faces: list[ProcessedFaceOut]
    fps: float
    camera_ok: bool


class EnrollRequest(BaseModel):
    name: str
    employee_id: str | None = None
    consent: bool = False
    pose_tag: str | None = None


class PersonOut(BaseModel):
    id: str
    name: str
    employee_id: str | None = None
    consent_given: bool


class AttendanceOut(BaseModel):
    id: str
    person_id: str | None = None
    is_unknown: bool
    camera_id: str
    timestamp: str
    confidence: float
    liveness_score: float | None = None
    snapshot_path: str | None = None
    erased: bool


class HealthOut(BaseModel):
    status: str
    ready: bool
    version: str


class AlertOut(BaseModel):
    id: str
    attendance_id: str | None = None
    timestamp: str
    snapshot_path: str | None = None
    webhook_status: str | None = None
    resolved: bool
