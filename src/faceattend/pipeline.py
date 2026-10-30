"""Per-frame processing pipeline."""

from __future__ import annotations

import threading
import time

import numpy as np

from .attendance import AttendanceService
from .config import get_settings
from .detection import DetectorConfig, FaceDetector
from .domain import FrameResult, ProcessedFace
from .liveness import LivenessChecker
from .recognition import Gallery


class Pipeline:
    def __init__(self, session_factory, camera_id: str = "default", detector=None):
        settings = get_settings()
        self.settings = settings
        self.camera_id = camera_id
        self._sf = session_factory
        self._sf = session_factory
        if detector is None:
            self.detector = FaceDetector(DetectorConfig())
            self.detector.load()
        else:
            self.detector = detector
        self.gallery = Gallery(session_factory)
        self.attendance = AttendanceService(session_factory)
        self.liveness = LivenessChecker(mode=settings.liveness_mode)
        self._alert_manager = None  # set externally if needed
        self._lock = threading.Lock()
        self._fps = 0.0
        self._frame_count = 0
        self._fps_t0 = time.monotonic()

    def set_alert_manager(self, am) -> None:
        self._alert_manager = am

    def reload_gallery(self) -> None:
        self.gallery.reload()

    def process(self, frame: np.ndarray) -> FrameResult:
        if frame is None:
            return FrameResult(faces=[], fps=self._fps, camera_ok=False)

        faces = self.detector.detect(frame)
        processed: list[ProcessedFace] = []

        for f in faces:
            import numpy as np

            emb = f.embedding
            match = (
                self.gallery.match(np.asarray(emb, dtype=np.float32))
                if emb is not None
                else None
            )
            is_unknown = True
            person_id = None
            name = None
            distance = 1.0
            if match is not None:
                is_unknown = match.is_unknown
                person_id = match.person_id
                name = match.name
                distance = match.distance

            # Auto-register unknown faces (no manual enrollment needed)
            if is_unknown and emb is not None:
                try:
                    new_pid, new_name = self._auto_enroll(emb)
                    person_id = new_pid
                    name = new_name
                    is_unknown = False
                    distance = 0.0
                    self.gallery.reload()
                except Exception:  # noqa: S110
                    pass

            # liveness
            liv = None
            if self.settings.liveness_mode != "off":
                try:
                    liv = self.liveness.update(frame, f)
                except Exception:
                    liv = None

            # decide logging
            logged = False
            alert_triggered = False
            live_ok = True

            if is_unknown:
                # alert path
                if self._alert_manager is not None and live_ok:
                    # cooldown handled inside
                    aid = (
                        self.attendance.log(
                            None,
                            self.camera_id,
                            distance,
                            True,
                            liveness_score=(liv.score if liv else None),
                        )
                        if self.attendance.should_log(None, self.camera_id, True)
                        else None
                    )
                    if aid:
                        logged = True
                    # alert with cooldown
                    if self._alert_manager.maybe_alert(self.camera_id, frame) is not None:
                        alert_triggered = True
            else:
                if live_ok and self.attendance.should_log(person_id, self.camera_id, False):
                    self.attendance.log(
                        person_id,
                        self.camera_id,
                        1.0 - distance,
                        False,
                        liveness_score=(liv.score if liv else None),
                    )
                    logged = True

            processed.append(
                ProcessedFace(
                    bbox=f.bbox,
                    person_id=person_id,
                    name=name,
                    distance=distance,
                    is_unknown=is_unknown,
                    liveness=liv,
                    logged=logged,
                    alert_triggered=alert_triggered,
                )
            )

        # fps
        self._frame_count += 1
        now = time.monotonic()
        dt = now - self._fps_t0
        if dt >= 1.0:
            self._fps = self._frame_count / dt
            self._frame_count = 0
            self._fps_t0 = now

        return FrameResult(faces=processed, fps=self._fps, camera_ok=True)

    def _auto_enroll(self, embedding):
        import datetime as _dt

        import numpy as np

        from .storage import Embedding as EmbRow
        from .storage import Person, bytes_to_embedding, embedding_to_bytes

        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n

        merge_threshold = 0.65
        with self._sf() as sess:
            persons = sess.query(Person).filter(Person.deleted_at.is_(None)).all()
            pmap = {p.id: p.name for p in persons}
            from .storage import Embedding as ERow

            best_pid = None
            best_dist = 1.0
            for row in sess.query(ERow).all():
                if row.person_id not in pmap:
                    continue
                v = bytes_to_embedding(row.vector, row.dim)
                vn = np.linalg.norm(v)
                if vn > 0:
                    v = v / vn
                dist = float(1.0 - np.dot(v, vec))
                if dist < best_dist:
                    best_dist = dist
                    best_pid = row.person_id

            if best_pid is not None and best_dist <= merge_threshold:
                emb = EmbRow(
                    person_id=best_pid,
                    vector=embedding_to_bytes(vec),
                    dim=int(vec.shape[0]),
                    model_version="facenet-vggface2",
                    pose_tag="auto",
                )
                sess.add(emb)
                sess.commit()
                return best_pid, pmap[best_pid]

            count = sess.query(Person).filter(Person.deleted_at.is_(None)).count()
            person = Person(
                name=f"Student_{count + 1}",
                employee_id=None,
                consent_given=True,
                consent_at=_dt.datetime.now(_dt.UTC),
                consent_version="1.0",
            )
            sess.add(person)
            sess.flush()
            emb = EmbRow(
                person_id=person.id,
                vector=embedding_to_bytes(vec),
                dim=int(vec.shape[0]),
                model_version="facenet-vggface2",
                pose_tag="auto",
            )
            sess.add(emb)
            sess.commit()
            return person.id, person.name

    def close(self) -> None:
        self.liveness.close()
        self.detector = None  # type: ignore
