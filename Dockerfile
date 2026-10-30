# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv sync --no-dev 2>/dev/null || pip install --no-cache-dir -e .

FROM base AS runtime
COPY --from=deps /app/.venv /app/.venv 2>/dev/null || true
ENV PATH="/app/.venv/bin:/usr/local/bin:$PATH"
COPY src/ scripts/ /app/
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["python", "-m", "uvicorn", "faceattend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]