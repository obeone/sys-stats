# syntax=docker/dockerfile:1

# --- Build stage: install the package into an isolated venv ---
# Same base as the runtime stage, so the venv copied across is built against the
# very interpreter that will run it. Slim is enough because every dependency has
# a manylinux wheel for the two platforms this image targets — psutil's abi3
# wheels cover x86_64 and aarch64 — so nothing compiles from source. Adding a
# platform without wheels means going back to the full image for its toolchain.
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy only what the build needs first, for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# uv as a static binary from its own image, pinned so builds stay reproducible.
# That image publishes linux/amd64 and linux/arm64, exactly the platform list in
# .github/workflows/build-and-publish.yaml.
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /bin/uv

# The cache mount is a different filesystem from /opt/venv, so let uv copy
# instead of trying to hardlink and falling back with a warning.
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python .

# --- Runtime stage: slim image with just the venv ---
FROM python:3.12-slim

# Only the labels that never change live here. LABEL cannot read pyproject.toml
# (its values expand ARG/ENV, never the output of a RUN), so duplicating the
# version would just be one more place to forget on a bump. The workflow reads
# pyproject.toml and injects org.opencontainers.image.version and .revision at
# build time instead; a local build legitimately has neither.
LABEL org.opencontainers.image.title="sys-stats" \
      org.opencontainers.image.description="Real-time system, GPU and Ollama monitoring dashboard (terminal + web)." \
      org.opencontainers.image.source="https://github.com/obeone/sys-stats" \
      org.opencontainers.image.url="https://github.com/obeone/sys-stats" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="obeone <obeone@obeone.org>"

# High UID/GID to stay clear of host users mapped into the container.
RUN groupadd -r -g 10001 stats && \
    useradd -r -u 10001 -g stats -d /app -m stats

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER stats
WORKDIR /app

# Flask metrics server listens here (see sys_stats.server).
EXPOSE 5000

# Probe the web UI rather than /stats: the latter shells out to nvidia-smi and
# polls Ollama on every call, which is far too heavy to run every 30 seconds.
# The port is read from the environment because PORT overrides the default.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '5000') + '/', timeout=4).status == 200 else 1)"]

# Console-script entry point installed by the package.
CMD ["sys-stats-server"]
