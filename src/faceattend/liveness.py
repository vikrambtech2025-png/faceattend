"""Anti-spoofing: active (blink/EAR) + passive (texture)."""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from .domain import LivenessResult

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(landmarks, idxs) -> float:
    pts = landmarks[idxs]
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    if c == 0:
        return 0.0
    return float((a + b) / (2.0 * c))


class LivenessChecker:
    def __init__(self, mode: str = "both"):
        self.mode = mode
        self._mesh = None
        self._lock = threading.Lock()
        self._ear_history: deque = deque(maxlen=30)
        self._blink_times: deque = deque(maxlen=10)
        self._last_reset = time.monotonic()

    def _ensure_mesh(self):
        if self._mesh is not None:
            return
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset(self) -> None:
        with self._lock:
            self._ear_history.clear()
            self._blink_times.clear()
            self._last_reset = time.monotonic()

    def update(self, frame, face) -> LivenessResult:
        active_score = 0.0
        blink_detected = False
        texture_score = 0.0

        if self.mode in ("passive", "both"):
            texture_score = self._texture_score(face.crop if hasattr(face, "crop") else None)

        if self.mode in ("active", "both"):
            blink_detected = self._update_blink(frame)
            active_score = 1.0 if blink_detected else 0.0
            # also accept recent blink within last 3s
            now = time.monotonic()
            recent = any(now - t < 3.0 for t in self._blink_times)
            if recent:
                active_score = 1.0
                blink_detected = True

        if self.mode == "off":
            score = 1.0
        elif self.mode == "passive":
            score = texture_score
        elif self.mode == "active":
            score = active_score
        else:
            # both: require active blink OR strong texture
            score = max(active_score, texture_score * 0.6)

        is_live = score >= 0.5
        return LivenessResult(
            score=score, is_live=is_live, blink_detected=blink_detected, texture_score=texture_score
        )

    def _update_blink(self, frame) -> bool:
        self._ensure_mesh()
        assert self._mesh is not None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return False
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        left_ear = _ear(pts, LEFT_EYE)
        right_ear = _ear(pts, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        with self._lock:
            self._ear_history.append(ear)
            # detect blink: EAR dipped below 0.2 then recovered
            if len(self._ear_history) >= 3:
                prev = list(self._ear_history)[-3:]
                if prev[0] > 0.25 and prev[1] < 0.2 and prev[2] > 0.25:
                    self._blink_times.append(time.monotonic())
                    return True
        return False

    @staticmethod
    def _texture_score(crop) -> float:
        if crop is None:
            return 0.5
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            # LBP-like variance: high texture => live; flat/print => low
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # normalize roughly: live faces often >100
            score = min(1.0, lap_var / 200.0)
            # color diversity
            if crop.ndim == 3:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                sat_std = float(np.std(hsv[:, :, 1]))
                score = (score + min(1.0, sat_std / 40.0)) / 2.0
            return float(score)
        except Exception:
            return 0.5

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None
