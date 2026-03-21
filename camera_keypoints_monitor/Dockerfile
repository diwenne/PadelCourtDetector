# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS app

# ---- Build-time args (optional secrets)
ARG APP_SUPABASE_KEY
ARG APP_SUPABASE_URL
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

# ---- Environment setup
ENV APP_SUPABASE_KEY=${APP_SUPABASE_KEY} \
    APP_SUPABASE_URL=${APP_SUPABASE_URL} \
    AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_CREATE=false \ 
    CAMERA_KEYPOINTS_CHECK_IN_DOCKERFILE=true
    

# ---- Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      unzip \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# tools for curl + tar.xz
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates xz-utils && rm -rf /var/lib/apt/lists/*

# FFmpeg: n7.1 release branch, linux x86_64, GPL static build
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz
RUN set -eux; \
    curl -fL "$FFMPEG_URL" -o /tmp/ffmpeg.tar.xz; \
    mkdir -p /opt/ffmpeg; \
    tar -xf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1; \
    ln -sf /opt/ffmpeg/bin/ffmpeg  /usr/local/bin/ffmpeg; \
    ln -sf /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe; \
    rm /tmp/ffmpeg.tar.xz; \
    ffmpeg -version | head -n1


WORKDIR /app

# ---- Dependency layer
COPY pyproject.toml poetry.lock* ./
RUN python -m pip install --upgrade pip \
 && pip install "poetry==${POETRY_VERSION}" \
 && poetry install --no-root --no-interaction

# ---- App layer
COPY src ./src
COPY README.md ./
RUN poetry install --no-interaction

# ---- Security: run as non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ---- Default command
CMD ["python3", "src/camera_keypoints.py"]
