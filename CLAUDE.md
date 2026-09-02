





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

# Distribution (uv)
uv build                               # wheel + sdist in dist/
uvx --from . sys-stats                 # run the CLI without installing it
uv tool install .                      # install both scripts on PATH
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

## Versioning

**Any user-visible change bumps the version, and the bump touches every file that
spells it out.** Three files do:

| File               | Reference                             |
| ------------------ | ------------------------------------- |
| `pyproject.toml`   | `version = "X.Y.Z"` — source of truth |
| `compose.yaml`     | `image: obeoneorg/sys-stats:X.Y.Z`    |
| `chart/Chart.yaml` | `appVersion: "X.Y.Z"`                 |

Nothing reconciles the three, and bumping `pyproject.toml` alone leaves the other two
pointing at a tag nobody published. `chart/Chart.yaml` also carries its own `version:`,
which is the chart's semver and moves independently of the application's.

Everything else derives the version instead of repeating it, and must stay that way:
`src/sys_stats/__init__.py` reads it from the installed package metadata, and
[the workflow](.github/workflows/build-and-publish.yaml) parses `pyproject.toml` to
produce both the `:X.Y.Z` image tag and the `org.opencontainers.image.version` label.
The Dockerfile deliberately carries no version: `LABEL` can only expand `ARG`/`ENV`,
never the output of a `RUN`, so it cannot read `pyproject.toml` and hardcoding the
value there would only add a fourth place to forget. The chart's `image.tag` is
`"{{ .Chart.AppVersion }}"`, rendered through `tpl`, so `values.yaml` is not a fourth
place either.

Semver applies to the package as a whole: the `/stats` payload is a public contract
(see [Architecture](#architecture)), so renaming or removing a key there is a major
bump, not a patch.

The bump is only half a release. Publishing `:X.Y.Z` takes a matching git tag:
`git tag v1.4.0 && git push origin v1.4.0`, once the bump commit is on `main`. The
workflow refuses to build when the tag and `pyproject.toml` disagree, so `v1.4.0` on
a `1.3.0` tree fails instead of publishing an image whose tag lies.

## Deployment constraints worth knowing

- The container needs `pid: host` and `privileged: true` (see [compose.yaml](compose.yaml))
  to see the host's processes — process listings are meaningless without it.
- The CI matrix targets `linux/amd64,linux/arm64` only. It used to also carry `i386`
  and `arm/v7`; those were dropped in 1.2.0, and `1.1.0` is the last tag that carries
  all four — it stays published as-is. Adding a platform back means re-checking
  that both uv and the native deps have wheels for it, or that the build stage can
  compile them.
- Both Dockerfile stages are `python:3.12-slim`, and the build stage carries no C
  toolchain: on amd64 and arm64 every dependency resolves to a manylinux wheel, psutil
  included (its `abi3` wheels cover `x86_64` and `aarch64`). Restoring a platform
  without wheels means restoring the full `python:3.12` builder along with it. The two
  stages share a base on purpose, so the venv copied across matches the interpreter
  that runs it.
- uv arrives via `COPY --from=ghcr.io/astral-sh/uv:<version>`, pinned. That image
  publishes only `linux/amd64` and `linux/arm64` — fine today, and one more thing to
  revisit before widening the platform list.
- [.github/workflows/build-and-publish.yaml](.github/workflows/build-and-publish.yaml)
  pushes and cosign-signs to both `ghcr.io/obeone/sys-stats` and
  `docker.io/obeoneorg/sys-stats`, and what it tags depends on the ref: a `v*` tag
  publishes `:X.Y.Z` and moves `:latest`, while a commit on `main` publishes `:edge`
  plus an immutable `:sha-<short>`. Version tags therefore stay pinned to one build.
  PRs and `workflow_dispatch` runs build without pushing or authenticating.
- [chart/](chart/) is the Helm chart, hand-written on top of the
  [bjw-s common library](https://github.com/bjw-s-labs/helm-charts) — `templates/`
  holds nothing but the library loader and `NOTES.txt`, so every deployment knob is a
  values key, and the library's own values reference is the authority on what is
  accepted. It replaced a Kompose dump that used to live in `compose/`, and it is
  versioned on its own line, not with the package. The two host-level requirements above
  reappear there as `defaultPodOptions.hostPID` and the container's
  `securityContext.privileged`, both on by default; `chart/charts/` is a fetched
  dependency and is gitignored, `Chart.lock` is committed.
