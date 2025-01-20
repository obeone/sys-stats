#!/usr/bin/env bash

nuitka \
    --output-dir=build \
    --include-module=prettytable \
    --output-filename=sys-stats \
    --low-memory \
    ./cli.py
