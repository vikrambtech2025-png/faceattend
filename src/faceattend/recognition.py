"""Embedding matching against a known gallery."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .config import get_settings
from .domain import MatchResult
from .storage import (
    Person,
    bytes_to_embedding,
)


@dataclass
class GalleryEntry:
    person_id: str
    name: str
    embedding: np.ndarray  # (512,) L2-normed


class Gallery:
    """In-memory gallery of enrolled embeddings, backed by SQLite."""

    def __init__(self, session_factory):
        self._sf = session_factory
        self._lock = threading.RLock()
        self._entries: list[GalleryEntry] = []
        self.reload()

    def reload(self) -> None:
        with self._lock:
            entries: list[GalleryEntry] = []
            with self._sf() as s:
                persons = s.query(Person).filter(Person.deleted_at.is_(None)).all()
                pmap = {p.id: p.name for p in persons}
                from .storage import Embedding as EmbRow

                for row in s.query(EmbRow).all():
                    if row.person_id not in pmap:
                        continue
                    vec = bytes_to_embedding(row.vector, row.dim)
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / n
                    entries.append(GalleryEntry(row.person_id, pmap[row.person_id], vec))
            self._entries = entries

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def match(self, embedding: np.ndarray) -> MatchResult:
        settings = get_settings()
        threshold = settings.match_threshold
        with self._lock:
            if not self._entries or embedding is None:
                return MatchResult(None, None, 1.0, True)
            q = np.asarray(embedding, dtype=np.float32).reshape(-1)
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n
            matrix = np.stack([e.embedding for e in self._entries])  # (N, D)
            sims = matrix @ q
            idx = int(np.argmax(sims))
            best = self._entries[idx]
            distance = float(1.0 - sims[idx])
            is_unknown = distance > threshold
            if is_unknown:
                return MatchResult(None, None, distance, True)
            return MatchResult(best.person_id, best.name, distance, False)
