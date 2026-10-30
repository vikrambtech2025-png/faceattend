"""FastAPI routes."""

from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import get_settings
from .schemas import (
    AlertOut,
    AttendanceOut,
    FaceBox,
    FrameResultOut,
    HealthOut,
    LivenessInfo,
    PersonOut,
    ProcessedFaceOut,
)

limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


def _state(request: Request):
    return request.app.state


def _decode_image(data: bytes) -> np.ndarray:
    settings = get_settings()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="image too large")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=415, detail="invalid image")
    # cap resolution
    h, w = img.shape[:2]
    max_side = 1280
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@router.get("/metrics")
async def metrics(request: Request):
    st = _state(request)
    sf = st.session_factory
    from ..storage import Attendance, Person

    with sf() as s:
        persons = s.query(Person).filter(Person.deleted_at.is_(None)).count()
        attendance = s.query(Attendance).count()
    return {
        "persons": persons,
        "attendance_rows": attendance,
        "ready": getattr(st, "ready", False),
    }


@router.get("/health", response_model=HealthOut)
async def health(request: Request):
    st = _state(request)
    ready = getattr(st, "ready", False)
    return HealthOut(status="ok", ready=ready, version="0.1.0")


@router.get("/ready")
async def ready(request: Request):
    st = _state(request)
    if not getattr(st, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready"}


@router.post("/detect", response_model=FrameResultOut)
@limiter.limit("60/minute")
async def detect(request: Request, file: UploadFile = File(...)):
    st = _state(request)
    if st.detector is None:
        raise HTTPException(status_code=503, detail="not ready")
    data = await file.read()
    img = _decode_image(data)
    faces = st.detector.detect(img)
    out_faces = []
    for f in faces:
        out_faces.append(
            ProcessedFaceOut(
                bbox=FaceBox(x1=f.bbox.x1, y1=f.bbox.y1, x2=f.bbox.x2, y2=f.bbox.y2),
                distance=0.0,
                is_unknown=False,
            )
        )
    return FrameResultOut(faces=out_faces, fps=0.0, camera_ok=True)


@router.post("/recognize", response_model=FrameResultOut)
@limiter.limit("30/minute")
async def recognize(
    request: Request, file: UploadFile = File(...), camera_id: str = Form("default")
):
    import asyncio

    st = _state(request)
    if st.pipeline is None:
        raise HTTPException(status_code=503, detail="not ready")
    data = await file.read()
    img = _decode_image(data)

    sem = st.inference_sem
    async with sem:
        result = await asyncio.to_thread(st.pipeline.process, img)

    # override camera_id? pipeline fixed; log uses its own. For simplicity use default.
    out_faces = []
    for pf in result.faces:
        liv = None
        if pf.liveness is not None:
            liv = LivenessInfo(
                score=pf.liveness.score,
                is_live=pf.liveness.is_live,
                blink_detected=pf.liveness.blink_detected,
                texture_score=pf.liveness.texture_score,
            )
        out_faces.append(
            ProcessedFaceOut(
                bbox=FaceBox(x1=pf.bbox.x1, y1=pf.bbox.y1, x2=pf.bbox.x2, y2=pf.bbox.y2),
                person_id=pf.person_id,
                name=pf.name,
                distance=pf.distance,
                is_unknown=pf.is_unknown,
                liveness=liv,
                logged=pf.logged,
                alert_triggered=pf.alert_triggered,
            )
        )
    return FrameResultOut(faces=out_faces, fps=result.fps, camera_ok=result.camera_ok)


@router.post("/enroll")
@limiter.limit("10/minute")
async def enroll(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    employee_id: str | None = Form(None),
    consent: bool = Form(False),
    pose_tag: str | None = Form(None),
):
    st = _state(request)
    if st.gallery_manager is None:
        raise HTTPException(status_code=503, detail="not ready")
    data = await file.read()
    img = _decode_image(data)
    if not consent:
        raise HTTPException(status_code=400, detail="consent required")
    try:
        pid = st.gallery_manager.enroll_from_frame(
            name=name, frame=img, employee_id=employee_id, consent=consent, pose_tag=pose_tag
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    st.pipeline.reload_gallery()
    st.privacy.audit("enroll", "api", pid, {"name": name})
    return {"person_id": pid, "name": name}


@router.get("/persons", response_model=list[PersonOut])
async def list_persons(request: Request):
    st = _state(request)
    return [PersonOut(**p) for p in st.gallery_manager.list_persons()]


@router.delete("/persons/{person_id}")
async def delete_person(request: Request, person_id: str):
    st = _state(request)
    ok = st.privacy.erase_person(person_id, actor="api")
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    st.pipeline.reload_gallery()
    return {"deleted": True}


@router.get("/persons/{person_id}/export")
async def export_person(request: Request, person_id: str):
    st = _state(request)
    data = st.privacy.export_person(person_id)
    if data is None:
        raise HTTPException(status_code=404, detail="not found")
    return data


@router.get("/attendance", response_model=list[AttendanceOut])
async def list_attendance(request: Request, limit: int = 100, camera_id: str | None = None):
    st = _state(request)
    rows = st.attendance_service.recent(limit=limit, camera_id=camera_id)
    return [AttendanceOut(**r) for r in rows]


@router.get("/attendance/export.csv")
async def export_csv(request: Request):
    st = _state(request)
    path = "data/attendance_export.csv"
    import os

    os.makedirs("data", exist_ok=True)
    n = st.attendance_service.export_csv(path)
    st.privacy.audit("export", "api", None, {"rows": n})
    return FileResponse(path, media_type="text/csv", filename="attendance.csv")


@router.get("/alerts", response_model=list[AlertOut])
async def list_alerts(request: Request, limit: int = 50):
    from ..storage import Alert

    st = _state(request)
    with st.session_factory() as s:
        rows = s.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()
        return [
            AlertOut(
                id=r.id,
                attendance_id=r.attendance_id,
                timestamp=r.timestamp.isoformat(),
                snapshot_path=r.snapshot_path,
                webhook_status=r.webhook_status,
                resolved=r.resolved,
            )
            for r in rows
        ]


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(request: Request, alert_id: str):
    from ..storage import Alert

    st = _state(request)
    with st.session_factory() as s:
        a = s.get(Alert, alert_id)
        if a is None:
            raise HTTPException(status_code=404, detail="not found")
        a.resolved = True
        a.resolved_at = datetime.now(UTC)
        s.commit()
    return {"resolved": True}


@router.post("/admin/purge")
async def purge(request: Request):
    st = _state(request)
    res = st.privacy.purge_retention()
    return res


@router.post("/admin/reload-gallery")
async def reload_gallery(request: Request):
    st = _state(request)
    st.pipeline.reload_gallery()
    return {"size": st.pipeline.gallery.size}
