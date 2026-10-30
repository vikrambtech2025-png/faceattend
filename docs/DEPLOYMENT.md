# Deployment

See plan file for full details. Summary:
- `docker compose up` -> app on :8000
- HF Spaces (Docker) for free public demo
- Render/Railway for always-on
- VPS + Caddy for production/RTSP

Health: GET /health, /ready. Metrics: GET /metrics (planned).