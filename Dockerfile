# syntax=docker/dockerfile:1

# --- Build stage: resolve and install the package into an isolated venv ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy only what the build needs first, for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install .

# --- Runtime stage: slim image with just the venv ---
FROM python:3.12-slim

RUN useradd -d /app -m stats

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER stats
WORKDIR /app

# Flask metrics server listens here (see sys_stats.server).
EXPOSE 5000

# Console-script entry point installed by the package.
CMD ["sys-stats-server"]
