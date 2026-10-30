"""Thread-safe camera capture abstraction."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray | None
    timestamp: float
    ok: bool


class CameraSource:
    """Wraps a cv2.VideoCapture with a background reader thread (latest-frame buffer)."""

    def __init__(self, source=0, width: int | None = None, height: int | None = None):
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_t0 = time.monotonic()

    def start(self) -> CameraSource:
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            now = time.monotonic()
            with self._lock:
                self._latest = FramePacket(frame=frame if ok else None, timestamp=now, ok=ok)
            if ok:
                self._frame_count += 1
                dt = now - self._fps_t0
                if dt >= 1.0:
                    self._fps = self._frame_count / dt
                    self._frame_count = 0
                    self._fps_t0 = now
            else:
                time.sleep(0.05)

    def read(self) -> FramePacket:
        with self._lock:
            if self._latest is None:
                return FramePacket(None, time.monotonic(), False)
            return self._latest

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_opened(self) -> bool:
        return bool(self._cap.isOpened())

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()
