"""Face detection + embedding via PyTorch (facenet-pytorch MTCNN + InceptionResnetV1)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .domain import BBox, DetectedFace


@dataclass
class DetectorConfig:
    device: str = "cpu"
    image_size: int = 160


class FaceDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._mtcnn = None
        self._resnet = None

    def load(self) -> None:
        if self._mtcnn is not None:
            return
        from facenet_pytorch import MTCNN, InceptionResnetV1

        device = self.config.device
        self._mtcnn = MTCNN(
            image_size=self.config.image_size,
            margin=0,
            keep_all=True,
            device=device,
            post_process=True,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.8,
        )
        self._resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self._device = device

    def _ensure(self) -> None:
        if self._mtcnn is None:
            self.load()

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        self._ensure()
        assert self._mtcnn is not None and self._resnet is not None
        if frame is None or frame.size == 0:
            return []
        h0, w0 = frame.shape[:2]
        scale = 1.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if w0 > 320:
            scale = 320.0 / w0
            small = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
            srgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            boxes, probs = self._mtcnn.detect(srgb, landmarks=False)
            if boxes is not None:
                boxes = boxes / scale
        else:
            boxes, probs = self._mtcnn.detect(rgb, landmarks=False)
        if boxes is None:
            return []
        out_boxes = []
        out_probs = []
        for b, p in zip(boxes, probs, strict=False):
            out_boxes.append(b)
            out_probs.append(p)
        # batched crop + embed
        crops = []
        valid = []
        for idx, box in enumerate(out_boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                valid.append(None)
                crops.append(None)
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                valid.append(None)
                crops.append(None)
                continue
            crop = cv2.resize(crop, (160, 160))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            t = (t - 0.5) / 0.5
            crops.append(t)
            valid.append((x1, y1, x2, y2, out_probs[idx]))
        # batch
        batch_tensors = [c for c in crops if c is not None]
        embs = None
        if batch_tensors:
            try:
                with torch.no_grad():
                    bt = torch.stack(batch_tensors).to(self._device)
                    embs = self._resnet(bt).cpu().numpy()
            except Exception:
                embs = None
        # assemble
        out: list[DetectedFace] = []
        ei = 0
        for _i, v in enumerate(valid):
            if v is None:
                continue
            x1, y1, x2, y2, prob = v
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                crop = np.zeros((112, 112, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (112, 112))
            emb = None
            if embs is not None and ei < len(embs):
                e = embs[ei]
                n = np.linalg.norm(e)
                if n > 0:
                    emb = (e / n).astype(np.float32)
            ei += 1
            out.append(
                DetectedFace(
                    bbox=BBox(x1, y1, x2, y2),
                    landmarks=None,
                    det_score=float(prob if prob is not None else 0.0),
                    crop=crop,
                    embedding=emb,
                )
            )
        return out