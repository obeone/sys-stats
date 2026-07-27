





# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sys-stats` is a real-time system / GPU / Ollama monitoring dashboard, shipped as a
single installable Python package (`src/sys_stats`, hatchling, src-layout) exposing
two console scripts:

| Script             | Module              | Role                                          |
| ------------------ | ------------------- | --------------------------------------------- |
| `sys-stats-server` | `sys_stats.server`  | Flask API + web UI, serves `/` and `/stats`    |
| `sys-stats`        | `sys_stats.cli`     | Rich terminal dashboard, HTTP **client** of `/stats` |

`python -m sys_stats` is an alias for the CLI.

## Architecture

Everything hinges on one contract: the JSON payload returned by `GET /stats`.

- **Server side** ([server.py](src/sys_stats/server.py)) is the only place that touches the
  machine. It gathers metrics from three unrelated sources and merges them into one
  response:
  - `psutil` for CPU / RAM / top processes,
  - `GPUtil` for GPU enumeration, plus **direct `nvidia-smi` subprocess calls**
    (`get_gpu_fan_and_power`, `get_gpu_processes`) for fan speed, power draw and
    per-process VRAM, which GPUtil does not expose. These parse CSV
    (`--format=csv,noheader,nounits`) and must degrade to `[]`/`{}` when the binary
    is missing or fails.
  - Ollama's `/api/ps`, only when `OLLAMA_API_URL` is set (otherwise it returns an
    empty `{"models": []}` — never an error).
- **Two independent consumers** of that payload: the inline JS in
  [templates/index.html](src/sys_stats/templates/index.html) (no build step, no bundler,
  no external JS deps) and the Rich CLI. **Changing a key in the `/stats` response
  means updating both**, plus the CLI's `build_*_panel` functions.
- Units are deliberately normalised server-side: memory is converted to **bytes**
  before leaving `/stats` (GPUtil hands out MiB), `load` is a percentage, `powerDraw`
  is watts. The CLI's `human_readable_size` assumes bytes.

### CLI concurrency model

[cli.py](src/sys_stats/cli.py) runs a `readchar` keyboard listener in a daemon thread
alongside the main `rich.live.Live` render loop. Shared state (`is_paused`,
`show_help_flag`, `refresh_interval`, `latest_stats`) lives in module-level globals
guarded by `state_lock` / `stats_lock`, with `exit_event` and `rebuild_layout_event`
for signalling. The sleep between refreshes is sliced into 100 ms chunks so key
presses feel instant. Any new interactive feature goes through the same
event + lock pattern, not through busy-waiting or a second render loop.

## Commands

```bash
# Dev install (editable, with pytest + ruff)
uv venv && source .venv/bin/activate && uv pip install -e '.[dev]'

# Tests and lint
pytest                                 # whole suite
pytest tests/test_stats_endpoint.py    # one file
pytest -k gpu_processes                # one test / group
ruff check .                           # add --fix to autofix

# Run the two halves
sys-stats-server                       # HOST / PORT / FLASK_DEBUG env vars
sys-stats --url http://localhost:5000/stats --interval 5

# Docker
docker compose up -d                                    # no GPU
docker compose -f compose.yaml -f compose.gpu.yaml up -d # NVIDIA

# Standalone binary of the CLI
./nuitka.sh                            # outputs build/sys-stats
```

On macOS port 5000 is taken by AirPlay Receiver — run the server with `PORT=5051`
when smoke-testing locally.

### Testing conventions

No test touches the real machine: `psutil`, `GPUtil`, `subprocess.run` and
`requests.get` are all monkeypatched at the `sys_stats.server` module boundary.
[tests/test_stats_endpoint.py](tests/test_stats_endpoint.py) is the contract test for
the `/stats` payload — if you add or rename a key there, that file is the one that
must change first. Two classes of failure matter most and are already covered:
unit conversions (MiB → bytes, fraction → percent) and graceful degradation when a
collector returns nothing.

Beware `psutil.process_iter(attrs)`: it does **not** raise on permission errors, it
fills the unreadable attributes with `None`. Any new field read from it needs a
`None` guard, otherwise sorting or arithmetic turns the whole endpoint into a 500 on
unprivileged hosts.

## Environment variables

| Variable             | Consumed by | Effect                                          |
| -------------------- | ----------- | ----------------------------------------------- |
| `OLLAMA_API_URL`     | server      | Enables the Ollama panel; unset ⇒ panel empty   |
| `HOST` / `PORT`      | server      | Bind address, default `0.0.0.0:5000`            |
| `FLASK_DEBUG`        | server      | `true` enables Flask debug mode                 |
| `SYS_STATS_API_URL`  | CLI         | Default `--url`, default `http://localhost:5000/stats` |

## Deployment constraints worth knowing

- The container needs `pid: host` and `privileged: true` (see [compose.yaml](compose.yaml))
  to see the host's processes — process listings are meaningless without it.
- The Dockerfile intentionally builds on the **full** `python:3.12` image (not slim/uv):
  the CI matrix targets `linux/amd64,arm64,i386,arm/v7` and `psutil` must compile from
  source on the exotic arches. Don't "optimise" that back to slim without checking the
  workflow's platform list.
- [.github/workflows/build-and-publish.yaml](.github/workflows/build-and-publish.yaml)
  pushes and cosign-signs to both `ghcr.io/obeone/sys-stats` and
  `docker.io/obeoneorg/sys-stats` on `main` only; PRs build without pushing.
- [compose/](compose/) is a Kompose-generated Helm chart (chart name `compose`), not
  hand-written. Treat it as generated output.
