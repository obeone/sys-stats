# syntax=docker/dockerfile:1

# --- Build stage: install the package into an isolated venv ---
# Use the full python image (ships a C toolchain) so native deps such as
# psutil build from source on the exotic arches the workflow targets
# (i386, arm/v7), which the slim/uv images do not all publish.
FROM python:3.12 AS builder

WORKDIR /app

# Copy only what the build needs first, for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install .

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
