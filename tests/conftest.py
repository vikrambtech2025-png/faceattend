import pytest

from faceattend.config import reset_settings


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("FACEATTEND_DB_PATH", str(db))
    reset_settings()
    yield
    reset_settings()
