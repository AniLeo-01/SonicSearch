# SonicSearch API: CPU-only, models downloaded on first start into /home/app/.cache (a volume in docker-compose.yml)
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /bin/uv

# ffmpeg decodes audio for speaker labelling and converts uploads
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app && mkdir /home/app/.cache && chown app:app /home/app/.cache

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# dependencies first, so code changes don't reinstall them
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project --no-dev
COPY app ./app
COPY scripts ./scripts

ENV PATH=/opt/venv/bin:$PATH HOME=/home/app HF_HOME=/home/app/.cache/huggingface DATA_DIR=/app/data PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
