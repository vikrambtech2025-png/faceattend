"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import get_settings
from ..gallery import GalleryManager
from ..logging import configure_logging, get_logger
from ..pipeline import Pipeline
from ..privacy import PrivacyService
from ..storage import make_session_factory
from .routes import router

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log = get_logger("startup")
    settings = get_settings()
    # db
    sf = make_session_factory(str(settings.db_path))
    app.state.session_factory = sf
    app.state.settings = settings
    from ..detection import DetectorConfig, FaceDetector

    detector = FaceDetector(DetectorConfig())
    app.state.detector = detector
    app.state.gallery_manager = None
    app.state.attendance_service = None
    app.state.privacy = None
    app.state.pipeline = None
    app.state.alert_manager = None
    # concurrency
    app.state.inference_sem = asyncio.Semaphore(settings.inference_concurrency)
    app.state.ready = False

    def _load_model() -> None:
        try:
            detector.load()
            from ..alerts import AlertManager
            from ..attendance import AttendanceService

            gm = GalleryManager(sf, detector=detector)
            app.state.gallery_manager = gm
            app.state.attendance_service = AttendanceService(sf)
            app.state.privacy = PrivacyService(sf, snapshot_dir="./data/snapshots")
            pipe = Pipeline(sf, camera_id="default", detector=detector)
            app.state.pipeline = pipe
            am = AlertManager(sf, snapshot_dir="./data/snapshots")
            pipe.set_alert_manager(am)
            app.state.alert_manager = am
            app.state.ready = True
            log.info("startup complete", model="pytorch-facenet-vggface2")
        except Exception as exc:
            log.error("model load failed", error=str(exc))
            app.state.ready = False

    import threading

    threading.Thread(target=_load_model, daemon=True).start()
    try:
        yield
    finally:
        app.state.ready = False
        pipe = getattr(app.state, "pipeline", None)
        if pipe is not None:
            pipe.close()
        log.info("shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="FaceAttend", version="0.1.0", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security headers + optional API key gate
    from starlette.middleware.base import BaseHTTPMiddleware

    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            return response

    app.add_middleware(SecurityHeadersMiddleware)

    api_key = os.environ.get("FACEATTEND_API_KEY", "").strip()
    if api_key:

        class APIKeyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.url.path in ("/health", "/ready", "/docs", "/openapi.json"):
                    return await call_next(request)
                if request.headers.get("x-api-key") != api_key:
                    return JSONResponse(status_code=401, content={"detail": "invalid api key"})
                return await call_next(request)

        app.add_middleware(APIKeyMiddleware)

    # rate limiter
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rl(request: Request, exc: RateLimitExceeded):
        return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})

    # static + templates
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None
    app.state.templates = templates

    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        if templates is None:
            return HTMLResponse("<h1>FaceAttend</h1><p>UI not installed.</p>")
        return templates.TemplateResponse(request, "index.html")

    @app.websocket("/ws/recognize")
    async def ws_recognize(ws: WebSocket):
        await ws.accept()
        st = ws.app.state
        try:
            while True:
                data = await ws.receive_bytes()
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    await ws.send_json({"error": "invalid frame"})
                    continue
                sem = st.inference_sem
                async with sem:
                    result = await asyncio.to_thread(st.pipeline.process, img)
                faces = []
                for pf in result.faces:
                    liv = None
                    if pf.liveness is not None:
                        liv = {
                            "score": pf.liveness.score,
                            "is_live": pf.liveness.is_live,
                            "blink_detected": pf.liveness.blink_detected,
                            "texture_score": pf.liveness.texture_score,
                        }
                    faces.append(
                        {
                            "bbox": {
                                "x1": pf.bbox.x1,
                                "y1": pf.bbox.y1,
                                "x2": pf.bbox.x2,
                                "y2": pf.bbox.y2,
                            },
                            "person_id": pf.person_id,
                            "name": pf.name,
                            "distance": pf.distance,
                            "is_unknown": pf.is_unknown,
                            "liveness": liv,
                            "logged": pf.logged,
                            "alert_triggered": pf.alert_triggered,
                        }
                    )
                await ws.send_json(
                    {"faces": faces, "fps": result.fps, "camera_ok": result.camera_ok}
                )
        except WebSocketDisconnect:
            pass
        except Exception as e:
            import contextlib

            with contextlib.suppress(Exception):
                await ws.send_json({"error": str(e)})

    return app


app = create_app()
