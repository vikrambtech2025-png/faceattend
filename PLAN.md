# FaceAttend — Face Recognition Attendance System (Portfolio Project)

> Detect faces → encode with ArcFace → match against a known gallery → log attendance with timestamps → anti-spoofing → unknown-person alerts → privacy-first storage.

**Target:** A production-flavored, end-to-end CV/ML portfolio project demonstrating detection, embedding, matching, storage, API, UI, deployment, evaluation, and responsible-AI practices.

---

## 1. Project Identity

- **Name:** FaceAttend
- **Tagline:** "Privacy-first, real-time face-recognition attendance with liveness detection."
- **Primary audience:** AI/ML engineer recruiters, hiring managers, and technical interviewers.
- **Success criteria (portfolio):**
  - Live demo (webcam) runs at ≥15 FPS on CPU.
  - Top-1 recognition accuracy ≥95% on a held-out test set of ≥20 identities.
  - Spoof detection HTER ≤10% on a small print/replay set.
  - End-to-end reproducible via `docker compose up`.
  - README with demo GIF, metrics table, architecture diagram, privacy section.

---

## 2. Tech Stack (decisions + rationale)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Broad library support, type hints |
| Package mgmt | `uv` + `pyproject.toml` | Fast, reproducible, modern |
| Detection + Recognition | `insightface` (buffalo_l: RetinaFace + ArcFace) | One package, strong accuracy, ONNX backend |
| Inference runtime | `onnxruntime` (CPU) / `onnxruntime-gpu` | Portable, fast |
| CV I/O | `opencv-python` | Camera capture, drawing, preprocessing |
| Numerics | `numpy`, `scipy` | Embedding math, similarity |
| Storage | `sqlite3` (stdlib) + `SQLAlchemy` 2.x | Zero-dep default, ORM for clarity |
| Liveness | MediaPipe FaceMesh (EAR blink) + texture heuristics (LBP) | Free, runs on CPU, two complementary signals |
| API | `FastAPI` + `uvicorn` | Async, typed, OpenAPI docs |
| UI | `Streamlit` | Fastest path to a polished demo |
| Config | `pydantic-settings` | Typed env/CLI config |
| Alerts | `httpx` (webhook) + optional SMTP | Simple, testable |
| Testing | `pytest`, `pytest-cov` | Standard |
| Lint/format | `ruff`, `black`, `mypy` | Modern Python quality bar |
| CI | GitHub Actions | Lint + typecheck + tests + docker build |
| Deploy | Docker + docker-compose; HF Spaces (Streamlit) | Free tiers, reproducible |
| Observability | `structlog` | Structured logs |

**Alternative considered:** `face_recognition` (dlib). Rejected: weaker accuracy, harder install on Windows, no anti-spoof. `insightface` wins on accuracy + packaging.

---

## 3. Repository Layout

```
faceattend/
├── pyproject.toml
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── PLAN.md
├── src/faceattend/
│   ├── __init__.py
│   ├── config.py                 # pydantic settings
│   ├── logging.py                # structlog setup
│   ├── camera.py                 # thread-safe capture abstraction
│   ├── detection.py              # RetinaFace wrapper
│   ├── recognition.py            # ArcFace embedding + matching
│   ├── liveness.py               # blink (EAR) + texture score
│   ├── gallery.py                # enroll / update / delete identities
│   ├── attendance.py             # logging service + dedup window
│   ├── storage.py                # SQLAlchemy models + repo
│   ├── alerts.py                 # unknown-person notifier
│   ├── privacy.py                # retention, erasure, audit
│   ├── pipeline.py               # orchestrates per-frame processing
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── ui/
│       └── app.py                # Streamlit
├── scripts/
│   ├── enroll.py                 # CLI enrollment
│   ├── run_camera.py             # headless camera loop
│   ├── export_csv.py
│   └── benchmark.py
├── tests/
│   ├── test_detection.py
│   ├── test_recognition.py
│   ├── test_matching.py
│   ├── test_liveness.py
│   ├── test_attendance_dedup.py
│   ├── test_storage.py
│   ├── test_privacy.py
│   └── fixtures/                 # sample images/videos (small)
├── data/                         # gitignored except .gitkeep
│   ├── gallery/
│   └── attendance.db
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PRIVACY.md
│   ├── EVALUATION.md
│   └── adr/
│       ├── 0001-insightface.md
│       └── 0002-streamlit-vs-fastapi.md
└── .github/workflows/ci.yml
```

---

## 4. Data Model (SQLite)

```sql
-- persons: enrolled identities
CREATE TABLE persons (
  id            TEXT PRIMARY KEY,          -- uuid4
  name          TEXT NOT NULL,
  employee_id   TEXT UNIQUE,
  consent_given INTEGER NOT NULL DEFAULT 0, -- 0/1
  consent_at    TEXT,
  created_at    TEXT NOT NULL,
  deleted_at    TEXT,                      -- soft delete for audit
  notes         TEXT
);

-- embeddings: one or many per person (multi-pose enrollment)
CREATE TABLE embeddings (
  id            TEXT PRIMARY KEY,
  person_id     TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  vector        BLOB NOT NULL,             -- float32 array serialized
  model_version TEXT NOT NULL,             -- e.g. buffalo_l/arcface_r100
  pose_tag      TEXT,                      -- frontal/left/right
  created_at    TEXT NOT NULL
);

-- attendance: one row per recognized appearance (after dedup window)
CREATE TABLE attendance (
  id            TEXT PRIMARY KEY,
  person_id     TEXT REFERENCES persons(id),  -- NULL for unknown
  is_unknown    INTEGER NOT NULL DEFAULT 0,
  camera_id     TEXT NOT NULL DEFAULT 'default',
  timestamp     TEXT NOT NULL,
  confidence    REAL NOT NULL,
  liveness_score REAL,
  snapshot_path TEXT,                      -- only for unknown alerts by default
  created_at    TEXT NOT NULL
);

-- alerts: unknown-person events
CREATE TABLE alerts (
  id            TEXT PRIMARY KEY,
  attendance_id TEXT REFERENCES attendance(id),
  timestamp     TEXT NOT NULL,
  snapshot_path TEXT,
  webhook_status TEXT,                      -- sent/failed/skipped
  resolved      INTEGER NOT NULL DEFAULT 0,
  resolved_at   TEXT
);

-- audit_log: privacy-relevant actions
CREATE TABLE audit_log (
  id            TEXT PRIMARY KEY,
  action        TEXT NOT NULL,             -- enroll/erase/export/retention_purge
  actor         TEXT,                      -- user/system
  target_id     TEXT,
  timestamp     TEXT NOT NULL,
  details       TEXT                        -- JSON
);
```

Indexes: `attendance(timestamp)`, `attendance(person_id)`, `embeddings(person_id)`.

---

## 5. Core Algorithms

### 5.1 Detection
- `insightface.app.FaceAnalysis(name="buffalo_l")` → RetinaFace.
- Input: BGR frame; resize longest side to 640 for speed; scale boxes back.
- Output: list of `Face` objects with `bbox`, `kps` (5 landmarks), `det_score`.

### 5.2 Recognition
- `app.get(img)` returns face with `.embedding` (512-d ArcFace).
- Normalize embedding (L2) → cosine similarity = dot product.
- Gallery: stack all embeddings into matrix `G` (N×512), plus `person_ids` array.
- Match: `sims = G @ q`; best idx; if `1 - sims[best] <= threshold_distance` (default 0.5) → known else unknown.
- Multi-face: process each detected face independently.

### 5.3 Dedup window
- For each recognized person, only log if no attendance row within last `DEDUP_MINUTES` (default 5) for that person+camera. Prevents spamming.

### 5.4 Liveness (anti-spoofing)
Two signals combined:
1. **Active (blink challenge):** MediaPipe FaceMesh → compute EAR (Eye Aspect Ratio) per eye; if EAR drops below 0.2 for ≥2 consecutive frames within a 3s window → blink detected → live.
2. **Passive (texture):** Crop face; compute LBP variance + color saturation std; spoof (print/screen) tends to have lower texture variance and moiré. Heuristic threshold tuned on a small set.
- Combined score `liveness = 0.6*active + 0.4*passive` (or either alone configurable). If `liveness < 0.5` → mark attendance as "unverified" and skip logging (or log with flag).

**Honest limitation note in README:** Heuristic liveness is not production-grade; recommend dedicated model (MiniFASNet/Silent-Face) for real deployments. Provide an interface to swap.

### 5.5 Unknown-person alert
- If best match is unknown AND `alert_cooldown` (default 60s) since last alert for that camera → save snapshot, insert alert, fire webhook (HTTP POST with snapshot base64 + timestamp) + optional email. Cooldown prevents flood.

### 5.6 Performance
- Run full pipeline every frame but allow `--detect-every N` (default 1) with IoU tracker for boxes between. For portfolio, keep simple (every frame) but optimize resize + batched embedding.
- Target: 640px detection, 112×112 recognition crop, ONNX CPU.

---

## 6. Phased Delivery Plan (15 working days)

### Phase 0 — Setup (Day 1)
- [ ] `git init`, `pyproject.toml`, `uv` env, pre-commit (ruff/black/mypy), CI skeleton.
- [ ] `.env.example`, `.gitignore`, LICENSE (MIT).
- **Deliverable:** `uv run python -c "import insightface; print('ok')"`.

### Phase 1 — Camera + Detection (Days 2–3)
- [ ] `camera.py`: `CameraSource` (webcam index / video file / RTSP url), thread-safe latest-frame buffer, FPS meter.
- [ ] `detection.py`: `FaceDetector.detect(frame) -> list[DetectedFace]`.
- [ ] `scripts/run_camera.py`: show window with boxes + FPS.
- [ ] Tests: detection on 2 fixture images (assert ≥1 face, bbox in bounds).
- **Deliverable:** live webcam demo with bounding boxes.

### Phase 2 — Recognition + Gallery (Days 4–5)
- [ ] `recognition.py`: `Embedder.embed(face_crop)`, `Matcher.match(embedding, gallery)`.
- [ ] `gallery.py`: enroll from image folder or webcam capture; store embedding + metadata; list/delete.
- [ ] `scripts/enroll.py --name "Alice" --images ./photos/alice/`.
- [ ] Unit tests for cosine matching with synthetic embeddings.
- **Deliverable:** enroll 5 people, recognize them live.

### Phase 3 — Attendance Logging (Day 6)
- [ ] `storage.py`: SQLAlchemy models + migrations (alembic optional; use `create_all` for MVP).
- [ ] `attendance.py`: `log_if_not_duplicate(person_id, camera_id, confidence, ...)`.
- [ ] `scripts/export_csv.py`.
- [ ] Tests: dedup window logic with frozen time.
- **Deliverable:** SQLite populated; CSV export works.

### Phase 4 — Anti-Spoofing (Days 7–8)
- [ ] `liveness.py`: EAR blink detector + texture score.
- [ ] Integrate into pipeline; add `--liveness-mode {off,passive,active,both}`.
- [ ] Collect small spoof set (printed photo, phone screen) for tuning.
- [ ] Tests: EAR computation on synthetic landmark sequences.
- **Deliverable:** live vs. photo discrimination demo.

### Phase 5 — Unknown-Person Alerts (Day 9)
- [ ] `alerts.py`: snapshot save, webhook POST, cooldown.
- [ ] Configurable webhook URL; graceful failure.
- [ ] Tests: mock httpx, assert cooldown respected.
- **Deliverable:** unknown face triggers alert + snapshot.

### Phase 6 — Streamlit UI (Days 10–11)
- [ ] Pages: Live Monitor, Enrollment, Attendance Log, Analytics (counts per person/day), Settings (thresholds).
- [ ] Sidebar config; real-time feed via `st.image` + placeholder.
- [ ] Download CSV button.
- **Deliverable:** polished web UI.

### Phase 7 — FastAPI (Day 12)
- [ ] Endpoints: `GET /health`, `POST /detect`, `POST /recognize`, `POST /enroll`, `GET /attendance`, `DELETE /persons/{id}` (erasure), `POST /export`.
- [ ] Pydantic schemas; OpenAPI docs.
- [ ] Tests with `TestClient`.
- **Deliverable:** documented REST API.

### Phase 8 — Privacy & Compliance (Day 13)
- [ ] Consent flag required for enrollment (CLI `--consent`).
- [ ] Retention job: purge snapshots older than `RETENTION_DAYS` (default 30); anonymize attendance older than `ANONYMIZE_DAYS` (default 90).
- [ ] Erasure endpoint removes embeddings + person; attendance rows anonymized (person_id=NULL, is_unknown=0 → marked erased).
- [ ] Audit log writes for enroll/erase/export/purge.
- [ ] `docs/PRIVACY.md`: data inventory, lawful basis, retention, rights, security controls.
- **Deliverable:** privacy section + working erasure.

### Phase 9 — Evaluation (Day 14)
- [ ] `scripts/benchmark.py`: latency p50/p95, FPS.
- [ ] Evaluation on own test set + LFW (subset): accuracy, confusion matrix, ROC, TAR@FAR.
- [ ] Spoof eval: HTER/EER on custom set.
- [ ] `docs/EVALUATION.md` with tables + plots.
- **Deliverable:** metrics in README.

### Phase 10 — Deployment + Polish (Day 15)
- [ ] `Dockerfile` (python:3.11-slim, model cache layer), `docker-compose.yml`.
- [ ] GitHub Actions: lint, typecheck, tests, docker build.
- [ ] README with demo GIF, badges, quickstart, architecture, privacy, roadmap.
- [ ] Record 60s demo video.
- **Deliverable:** public repo + deployed demo.

---

## 7. Configuration (`.env`)

```
FACEATTEND_DB_PATH=./data/attendance.db
FACEATTEND_MODEL_NAME=buffalo_l
FACEATTEND_DET_SIZE=640
FACEATTEND_MATCH_THRESHOLD=0.5          # cosine distance
FACEATTEND_DEDUP_MINUTES=5
FACEATTEND_ALERT_COOLDOWN_SEC=60
FACEATTEND_WEBHOOK_URL=
FACEATTEND_RETENTION_DAYS=30
FACEATTEND_ANONYMIZE_DAYS=90
FACEATTEND_LIVENESS_MODE=both
FACEATTEND_CAMERA_INDEX=0
FACEATTEND_LOG_LEVEL=INFO
```

---

## 8. Key Interfaces (sketches)

```python
# detection.py
@dataclass
class DetectedFace:
    bbox: tuple[int,int,int,int]   # x1,y1,x2,y2
    landmarks: np.ndarray          # 5x2
    det_score: float
    crop: np.ndarray               # 112x112 aligned

class FaceDetector:
    def detect(self, frame: np.ndarray) -> list[DetectedFace]: ...

# recognition.py
class Embedder:
    def embed(self, face: DetectedFace) -> np.ndarray: ...  # (512,) L2-normed

@dataclass
class MatchResult:
    person_id: str | None
    name: str | None
    distance: float
    is_unknown: bool

class Matcher:
    def match(self, embedding: np.ndarray) -> MatchResult: ...
    def reload_gallery(self) -> None: ...

# liveness.py
@dataclass
class LivenessResult:
    score: float           # 0..1
    is_live: bool
    blink_detected: bool
    texture_score: float

class LivenessChecker:
    def update(self, frame, face) -> LivenessResult: ...
    def reset(self) -> None: ...

# pipeline.py
@dataclass
class FrameResult:
    faces: list[ProcessedFace]   # bbox, name, distance, liveness, logged?
    fps: float

class Pipeline:
    def process(self, frame: np.ndarray) -> FrameResult: ...
```

---

## 9. Matching Threshold Tuning

- InsightFace ArcFace cosine similarity: same person typically ≥0.5 sim (distance ≤0.5). 
- Plan: collect 5+ images per enrolled person (varied pose/lighting), split train/test, sweep threshold, plot FAR/FRR, choose threshold at FAR=1e-3 or best F1.
- Document chosen threshold + FAR/FRR in EVALUATION.md.
- Per-deployment recalibration note (domain shift).

---

## 10. Testing Strategy

- **Unit:** matching math, dedup window, EAR, config validation, storage CRUD, privacy erasure.
- **Integration:** enrollment → recognition → attendance row; alert cooldown.
- **Property-ish:** threshold monotonicity.
- **Fixtures:** 2–3 small images + a 5s video clip (git LFS or download script).
- **CI:** run on ubuntu-latest with CPU onnxruntime; model downloaded & cached in CI.
- **Coverage target:** ≥70% on `src/faceattend` (excluding UI).

---

## 11. Privacy & GDPR-ish Design (docs/PRIVACY.md outline)

1. **Data inventory:** embeddings (biometric), names, timestamps, optional snapshots.
2. **Lawful basis:** explicit consent captured at enrollment; documented.
3. **Minimization:** store embeddings, not raw images, by default; snapshots only for unknown alerts.
4. **Retention:** snapshots 30d, attendance anonymized after 90d, audit log 1y.
5. **Rights:** access (export), erasure (delete), rectification (re-enroll).
6. **Security:** DB file permissions, optional SQLCipher, no secrets in logs, PII redaction in logs.
7. **Third parties:** none by default; webhook opt-in.
8. **Bias/fairness:** note known performance variation across demographics; recommend diverse enrollment; no demographic inference stored.
9. **Transparency:** on-screen indicator when camera is recording; signage guidance.

---

## 12. Evaluation Plan (docs/EVALUATION.md)

- **Dataset:** own 20-identity set (5 images each, varied) + LFW subset (100 pairs).
- **Metrics:** Top-1/Top-5 accuracy, TAR@FAR (1e-2, 1e-3), confusion matrix, per-identity F1.
- **Latency:** p50/p95 detection+embedding per frame, end-to-end FPS on CPU.
- **Spoof:** HTER, EER on custom print/replay set (≥20 attempts).
- **Failure analysis:** gallery of false accepts/rejects with notes.
- **Ablations:** threshold sweep; liveness on/off; det_size 320 vs 640.

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| insightface model download (~300MB) slows CI/Docker | Cache in Docker layer; CI cache action; document manual download |
| Windows webcam in Docker | Run natively for dev; Docker for API-only mode; document |
| Threshold doesn't generalize | Per-deployment calibration script + documented procedure |
| Heuristic liveness weak | Interface to swap in MiniFASNet; honest limitations in README |
| Lighting/pose failures | Multi-pose enrollment; augment with flips/brightness in gallery builder |
| PII in logs | Redact names in structlog; log IDs only |
| Model licensing | buffalo_l is non-commercial research; note in README; offer commercial alternative (ArcFace ONNX) |

---

## 14. README Outline (for final polish)

1. Title + badges (CI, license, python)
2. Demo GIF (60s)
3. Features (bullets)
4. Quickstart (3 commands)
5. Architecture diagram (mermaid)
6. Configuration table
7. Evaluation results (table + link)
8. Privacy & responsible use
9. Project structure
10. Roadmap
11. Contributing / License

---

## 15. Portfolio Talking Points (for interviews)

- Why ArcFace over FaceNet (angular margin, better inter-class separation).
- Why cosine distance + threshold calibration + FAR/FRR trade-off.
- Dedup window to avoid log spam; idempotency.
- Liveness: active (challenge) vs passive (texture); why combine; limitations.
- Privacy: embeddings are biometric data; consent, retention, erasure, minimization.
- Performance: resize, ONNX, every-N-frames, IoU tracking.
- Failure modes: occlusion, lighting, demographic bias, spoof attacks.

---

## 16. Stretch Goals (post-MVP)

- [ ] Multi-camera support with person re-ID.
- [ ] Mobile enrollment via QR + web form.
- [ ] Replace heuristic liveness with MiniFASNet (ONNX).
- [ ] Export to Prometheus metrics; Grafana dashboard.
- [ ] Role-based access (admin vs. viewer) in API.
- [ ] Edge deployment on Raspberry Pi (optimize to 320 det size).
- [ ] Federated: on-device embeddings only, no raw images ever leave device.

---

## 17. First 3 Commands to Start

```bash
cd D:\AI-Projects\agentic-ai\opencv
uv init --name faceattend --python 3.11
uv add insightface onnxruntime opencv-python numpy scipy sqlalchemy pydantic-settings fastapi uvicorn streamlit structlog httpx mediapipe
uv add --dev pytest pytest-cov ruff black mypy pre-commit
```

Then create `src/faceattend/__init__.py` and begin Phase 0.

---

*Plan version 1.0 — generated for portfolio project FaceAttend.*