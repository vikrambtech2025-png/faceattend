"""Application configuration (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACEATTEND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_path: Path = Field(default=Path("./data/attendance.db"))
    model_name: str = Field(default="buffalo_l")
    det_size: int = Field(default=640)
    match_threshold: float = Field(default=0.5)
    dedup_minutes: int = Field(default=5)
    alert_cooldown_sec: int = Field(default=60)
    webhook_url: str = Field(default="")
    retention_days: int = Field(default=30)
    anonymize_days: int = Field(default=90)
    liveness_mode: str = Field(default="off")
    camera_index: int = Field(default=0)
    log_level: str = Field(default="INFO")
    allowed_origins: str = Field(default="http://localhost:8000")
    max_upload_mb: int = Field(default=5)
    inference_concurrency: int = Field(default=2)

    @field_validator("det_size")
    @classmethod
    def _det_size(cls, v: int) -> int:
        if v not in (320, 640, 1280):
            raise ValueError("det_size must be 320, 640, or 1280")
        return v

    @field_validator("match_threshold")
    @classmethod
    def _threshold(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("match_threshold must be in (0,1)")
        return v

    @field_validator("liveness_mode")
    @classmethod
    def _liveness(cls, v: str) -> str:
        if v not in ("off", "passive", "active", "both"):
            raise ValueError("liveness_mode must be off/passive/active/both")
        return v

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """For tests."""
    global _settings
    _settings = None
