# Runs on AWS Fargate Graviton (linux/arm64). OpenCV 5 headless wheels ship for aarch64.

# The browser code is shipped minified (one unreadable line per file, no comments, no source maps).
FROM node:22-alpine AS web
WORKDIR /w
COPY web ./web
RUN npx --yes esbuild@0.24.0 web/app.js web/theatre.js web/viewer.js --minify --format=esm --target=es2022 \
    --legal-comments=none --outdir=/w/min

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY vision ./vision
COPY agents ./agents
COPY server ./server
COPY web ./web
COPY --from=web /w/min/ ./web/
RUN uv sync --frozen --no-dev && uv run python -m vision.models.fetch \
    && find /app -name "*.py" -path "*/tests/*" -delete

ENV PATH="/app/.venv/bin:$PATH" DATA_DIR=/tmp/data
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
