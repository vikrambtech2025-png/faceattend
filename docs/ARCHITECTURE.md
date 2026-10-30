# Architecture

## Layers
- `api/` transport (FastAPI, schemas, websocket)
- `services` (pipeline, gallery, attendance, alerts, privacy)
- `domain/` pure dataclasses (no framework deps)
- `infra` (storage/SQLAlchemy, camera, detection/insightface, liveness/mediapipe)

## Data flow (per frame)
1. CameraSource (threaded) -> latest frame
2. Pipeline.process(frame):
   - FaceDetector.detect -> list[DetectedFace] (bbox, landmarks, embedding)
   - Gallery.match(embedding) -> MatchResult
   - LivenessChecker.update -> LivenessResult
   - AttendanceService.should_log + log
   - AlertManager.maybe_alert (unknown only)
3. Return FrameResult -> API -> client overlay

## Concurrency
- Single uvicorn worker (model not fork-safe).
- Inference guarded by asyncio.Semaphore (configurable).
- CPU-bound work via asyncio.to_thread.
- SQLite WAL mode for concurrent readers.

## Storage
- SQLite (WAL) via SQLAlchemy 2.x.
- Embeddings stored as float32 bytes + dim + model_version.