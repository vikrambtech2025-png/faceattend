"""Unknown-person alerting with cooldown."""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import httpx

from .config import get_settings
from .storage import Alert


class AlertManager:
    def __init__(self, session_factory, snapshot_dir: str = "./data/snapshots"):
        self._sf = session_factory
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_alert: dict[str, float] = {}
        self._lock = threading.Lock()

    def maybe_alert(self, camera_id: str, frame, attendance_id: str | None = None) -> str | None:
        settings = get_settings()
        now = time.monotonic()
        with self._lock:
            last = self._last_alert.get(camera_id, 0.0)
            if now - last < settings.alert_cooldown_sec:
                return None
            self._last_alert[camera_id] = now

        # save snapshot
        snapshot_path: Path | None = self._snapshot_dir / f"unknown_{int(time.time())}.jpg"
        try:
            import cv2

            cv2.imwrite(str(snapshot_path), frame)
        except Exception:
            snapshot_path = None

        # record alert
        alert_id = None
        with self._sf() as s:
            from uuid import uuid4

            alert = Alert(
                id=str(uuid4()),
                attendance_id=attendance_id,
                timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                snapshot_path=str(snapshot_path) if snapshot_path else None,
                webhook_status="skipped",
            )
            s.add(alert)
            s.commit()
            alert_id = alert.id

        # fire webhook (best effort, timeout)
        url = settings.webhook_url.strip()
        if url and snapshot_path:
            status = "failed"
            try:
                with open(snapshot_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                payload = {
                    "camera_id": camera_id,
                    "timestamp": time.time(),
                    "snapshot_b64": b64,
                    "attendance_id": attendance_id,
                }
                with httpx.Client(timeout=5.0) as c:
                    r = c.post(url, json=payload)
                    r.raise_for_status()
                status = "sent"
            except Exception:
                status = "failed"
            with self._sf() as s:
                a = s.get(Alert, alert_id)
                if a is not None:
                    a.webhook_status = status
                    s.commit()
        return alert_id
