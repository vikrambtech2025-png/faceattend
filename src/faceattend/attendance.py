"""Attendance logging with dedup window."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from .config import get_settings
from .storage import Attendance


class AttendanceService:
    def __init__(self, session_factory):
        self._sf = session_factory
        self._last: dict = {}

    def should_log(self, person_id: str | None, camera_id: str, is_unknown: bool) -> bool:
        # Unknowns are logged per alert cooldown handled separately; still dedup by camera
        minutes = get_settings().dedup_minutes
        now = datetime.now(UTC)
        key = (person_id, camera_id, is_unknown)
        last = self._last.get(key)
        if last is not None and (now - last).total_seconds() < minutes * 60:
            return False
        cutoff = now - timedelta(minutes=minutes)
        with self._sf() as s:
            q = s.query(Attendance).filter(
                Attendance.camera_id == camera_id,
                Attendance.timestamp >= cutoff,
            )
            if person_id is None:
                q = q.filter(Attendance.is_unknown.is_(True))
            else:
                q = q.filter(Attendance.person_id == person_id)
            ok = int(q.count()) == 0
        if ok:
            self._last[key] = now
        return ok

    def log(
        self,
        person_id: str | None,
        camera_id: str,
        confidence: float,
        is_unknown: bool,
        liveness_score: float | None = None,
        snapshot_path: str | None = None,
    ) -> str:
        aid = str(uuid.uuid4())
        with self._sf() as s:
            row = Attendance(
                id=aid,
                person_id=person_id,
                is_unknown=is_unknown,
                camera_id=camera_id,
                timestamp=datetime.now(UTC),
                confidence=confidence,
                liveness_score=liveness_score,
                snapshot_path=snapshot_path,
            )
            s.add(row)
            s.commit()
        return aid

    def recent(self, limit: int = 100, camera_id: str | None = None) -> list[dict]:
        with self._sf() as s:
            q = s.query(Attendance).order_by(Attendance.timestamp.desc())
            if camera_id:
                q = q.filter(Attendance.camera_id == camera_id)
            rows = q.limit(limit).all()
            return [
                {
                    "id": r.id,
                    "person_id": r.person_id,
                    "is_unknown": r.is_unknown,
                    "camera_id": r.camera_id,
                    "timestamp": r.timestamp.isoformat(),
                    "confidence": r.confidence,
                    "liveness_score": r.liveness_score,
                    "snapshot_path": r.snapshot_path,
                    "erased": r.erased,
                }
                for r in rows
            ]

    def export_csv(self, path: str) -> int:
        import csv

        rows = self.recent(limit=100000)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return len(rows)
