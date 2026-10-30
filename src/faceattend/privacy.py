"""Privacy: retention, anonymization, erasure, audit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import get_settings
from .storage import Attendance, AuditLog, Person


class PrivacyService:
    def __init__(self, session_factory, snapshot_dir: str = "./data/snapshots"):
        self._sf = session_factory
        self._snapshot_dir = Path(snapshot_dir)

    def erase_person(self, person_id: str, actor: str = "system") -> bool:
        with self._sf() as s:
            p = s.get(Person, person_id)
            if p is None:
                return False
            # anonymize attendance
            s.query(Attendance).filter(Attendance.person_id == person_id).update(
                {"person_id": None, "erased": True}
            )
            # delete embeddings
            from .storage import Embedding as EmbRow

            s.query(EmbRow).filter(EmbRow.person_id == person_id).delete()
            p.deleted_at = datetime.now(UTC)
            p.name = "erased"
            p.notes = "erased"
            # audit
            s.add(
                AuditLog(
                    action="erase",
                    actor=actor,
                    target_id=person_id,
                    details='{"reason":"user_request"}',
                )
            )
            s.commit()
            return True

    def export_person(self, person_id: str) -> dict | None:
        with self._sf() as s:
            p = s.get(Person, person_id)
            if p is None or p.deleted_at is not None:
                return None
            att = s.query(Attendance).filter(Attendance.person_id == person_id).all()
            return {
                "id": p.id,
                "name": p.name,
                "employee_id": p.employee_id,
                "consent_given": p.consent_given,
                "created_at": p.created_at.isoformat(),
                "attendance": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "camera_id": a.camera_id,
                        "confidence": a.confidence,
                    }
                    for a in att
                ],
            }

    def purge_retention(self) -> dict:
        settings = get_settings()
        now = datetime.now(UTC)
        snap_cutoff = now - timedelta(days=settings.retention_days)
        anon_cutoff = now - timedelta(days=settings.anonymize_days)
        removed_snaps = 0
        # remove old snapshots
        if self._snapshot_dir.exists():
            for f in self._snapshot_dir.glob("*.jpg"):
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
                    if mtime < snap_cutoff:
                        f.unlink()
                        removed_snaps += 1
                except OSError:
                    pass
        anonymized = 0
        with self._sf() as s:
            rows = (
                s.query(Attendance)
                .filter(Attendance.timestamp < anon_cutoff, Attendance.person_id.isnot(None))
                .all()
            )
            for r in rows:
                r.person_id = None
                r.erased = True
                anonymized += 1
            s.add(
                AuditLog(
                    action="retention_purge",
                    actor="system",
                    details=f'{{"snapshots":{removed_snaps},"anonymized":{anonymized}}}',
                )
            )
            s.commit()
        return {"snapshots_removed": removed_snaps, "attendance_anonymized": anonymized}

    def audit(
        self, action: str, actor: str | None, target_id: str | None, details: dict | None = None
    ) -> None:
        import json

        with self._sf() as s:
            s.add(
                AuditLog(
                    action=action,
                    actor=actor,
                    target_id=target_id,
                    details=json.dumps(details or {}),
                )
            )
            s.commit()
