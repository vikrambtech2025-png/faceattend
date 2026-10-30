from fastapi.testclient import TestClient

from faceattend.api.main import create_app


def test_health():
    app = create_app()
    # lifespan won't run automatically with TestClient? It does on context.
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
