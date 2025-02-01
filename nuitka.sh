#!/usr/bin/env bash

nuitka \
    --output-dir=build \
    --output-filename=sys-stats \
    --low-memory \
    ./cli.py
