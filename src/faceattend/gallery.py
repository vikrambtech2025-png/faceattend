"""Enrollment and gallery management."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .config import get_settings
from .detection import DetectorConfig, FaceDetector
from .storage import (
    Embedding,
    Person,
    embedding_to_bytes,
)


class GalleryManager:
    def __init__(self, session_factory, detector=None):
        self._sf = session_factory
        if detector is None:
            self._detector = FaceDetector(DetectorConfig())
            self._detector.load()
        else:
            self._detector = detector

    def enroll_from_image(
        self,
        name: str,
        image_path: str | Path,
        employee_id: str | None = None,
        consent: bool = False,
        pose_tag: str | None = None,
    ) -> str:
        if not consent:
            raise ValueError("consent required for enrollment")
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cannot read image: {image_path}")
        faces = self._detector.detect(img)
        if not faces:
            raise ValueError("no face detected")
        if len(faces) > 1:
            raise ValueError("multiple faces detected; provide a single-face image")
        face = faces[0]
        if face.embedding is None:
            raise ValueError("embedding extraction failed")
        return self._store(name, employee_id, consent, face.embedding, pose_tag)

    def enroll_from_frame(
        self,
        name: str,
        frame: np.ndarray,
        employee_id: str | None = None,
        consent: bool = False,
        pose_tag: str | None = None,
    ) -> str:
        if not consent:
            raise ValueError("consent required for enrollment")
        faces = self._detector.detect(frame)
        if not faces:
            raise ValueError("no face detected")
        if len(faces) > 1:
            raise ValueError("multiple faces detected")
        face = faces[0]
        if face.embedding is None:
            raise ValueError("embedding extraction failed")
        return self._store(name, employee_id, consent, face.embedding, pose_tag)

    def _store(self, name, employee_id, consent, embedding, pose_tag) -> str:
        with self._sf() as s:
            # upsert by employee_id
            person = None
            if employee_id:
                person = s.query(Person).filter(Person.employee_id == employee_id).first()
            if person is None:
                person = Person(
                    name=name,
                    employee_id=employee_id,
                    consent_given=consent,
                    consent_at=datetime.now(UTC) if consent else None,
                    consent_version="1.0" if consent else None,
                )
                s.add(person)
                s.flush()
            else:
                person.name = name
                person.consent_given = consent
                if consent and person.consent_at is None:
                    person.consent_at = datetime.now(UTC)
            emb = Embedding(
                person_id=person.id,
                vector=embedding_to_bytes(embedding),
                dim=int(np.asarray(embedding).shape[0]),
                model_version=get_settings().model_name,
                pose_tag=pose_tag,
            )
            s.add(emb)
            s.commit()
            return person.id

    def delete_person(self, person_id: str) -> bool:
        with self._sf() as s:
            p = s.get(Person, person_id)
            if p is None:
                return False
            p.deleted_at = datetime.now(UTC)
            # remove embeddings
            from .storage import Embedding as EmbRow

            s.query(EmbRow).filter(EmbRow.person_id == person_id).delete()
            s.commit()
            return True

    def list_persons(self) -> list[dict]:
        with self._sf() as s:
            rows = s.query(Person).filter(Person.deleted_at.is_(None)).all()
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "employee_id": p.employee_id,
                    "consent_given": p.consent_given,
                }
                for p in rows
            ]
