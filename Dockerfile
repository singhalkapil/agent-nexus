# syntax=docker/dockerfile:1
# Multi-stage production build for Agent Nexus Gateway

# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

# Install pip-tools or build wheels directly
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Stage 2: Minimal Distroless / Hardened Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Security Hardening: create an unprivileged non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -m appuser

# Copy installed site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup pyproject.toml README.md /app/

# Install the application in editable/package mode
RUN pip install --no-deps -e /app

# Ensure transient uploads directory exists with non-root ownership
RUN mkdir -p /app/uploads && chown -R appuser:appgroup /app/uploads

# Switch to unprivileged runtime user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "agent_nexus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
