#!/usr/bin/env bash
# Build a standalone binary of the terminal dashboard with Nuitka.

nuitka \
    --output-dir=build \
    --output-filename=sys-stats \
    --low-memory \
    ./src/sys_stats/cli.py
