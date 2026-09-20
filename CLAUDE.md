





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

- **Collector side** ([collectors.py](src/sys_stats/collectors.py)) is the only place that
  touches the machine; `server.py` is a thin Flask layer with no direct calls into
  `psutil`, `GPUtil` or `nvidia-smi`. It gathers metrics from three unrelated sources
  and merges them into one response:
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
`requests.get` are all monkeypatched at the `sys_stats.collectors` module boundary.
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

| Variable                      | Consumed by | Effect                                          |
| ----------------------------- | ----------- | ------------------------------------------------ |
| `OLLAMA_API_URL`              | server      | Enables the Ollama panel; unset ⇒ panel empty   |
| `HOST` / `PORT`               | server      | Bind address, default `0.0.0.0:5000`            |
| `FLASK_DEBUG`                 | server      | `true` enables Flask debug mode                 |
| `SYS_STATS_AUTOSTART`         | server      | Set to `0`/`false`/`no` to skip the module-scope sampler autostart, default on |
| `SYS_STATS_SAMPLE_INTERVAL`   | sampler     | Seconds between background samples, default `2.0` |
| `SYS_STATS_TOP_PROCESSES_MAX` | sampler     | Per-process ranking cap the sampler collects, default `50`; a `?limit=` above this returns only what was sampled |
| `SYS_STATS_PANEL_MAX_TEMPS`   | server      | Caps the `/panel` `temps` list; unset (default) means no cap |
| `SYS_STATS_PANEL_MAX_FANS`    | server      | Caps the `/panel` `fans` list; unset (default) means no cap |
| `SYS_STATS_PANEL_MAX_GPUS`    | server      | Caps the `/panel` `gpu` list; unset (default) means no cap |
| `SYS_STATS_PANEL_ONLY`        | server      | `1`/`true`/`yes` registers ONLY `/panel`: `/`, `/stats` and `/favicon.png` are never registered, not just guarded; unset (default) registers every route |
| `SYS_STATS_INSTANCE_LABEL`    | server      | Optional label shown in the web UI's title and body, telling apart two co-located instances reporting on different views of the same box (e.g. a Kubernetes pod vs. the underlying hypervisor); unset (default) leaves the page exactly as before this variable existed |
| `SYS_STATS_HOSTNAME`          | server      | Overrides `/panel`'s `host` field, otherwise `socket.gethostname()` read fresh per request; unset (default) reports the real hostname. `/panel` only, never `/stats`, and never merged with `SYS_STATS_INSTANCE_LABEL` (that one is a rewritable display label, this one is a machine identity a consumer string-compares) |
| `SYS_STATS_DCGM_URL`          | sampler     | A dcgm-exporter `/metrics` URL; when set, `/panel`'s `gpu[]` is scraped from it instead of GPUtil/`nvidia-smi`, for hosts with no NVIDIA driver of their own (GPUs passed through to a VM). `/stats` never reads this variable and stays on the GPUtil/`nvidia-smi` path either way. Unset (default) leaves `/panel` on that same path too |
| `SYS_STATS_IPMI_INTERVAL`     | sampler     | Seconds between polls of the two `ipmi/`-prefixed `/panel` collectors (`temps`, `fans`), independent of `SYS_STATS_SAMPLE_INTERVAL`; default `30.0`. Between polls, `temps`/`fans` keep serving the last IPMI reading instead of dropping it; the first pass after startup always polls immediately. hwmon sensors keep the normal per-pass cadence |
| `SYS_STATS_API_URL`           | CLI         | Default `--url`, default `http://localhost:5000/stats` |

## Versioning and releases

**No file in the tree decides the version any more: git tags do.** Pushing a
`v*` tag is the entire release ceremony, and there is nothing to bump before it.

```bash
git tag v1.6.0 && git push origin v1.6.0
```

That single push runs the tests, publishes `:1.6.0` and moves `:latest` on both
registries, opens a GitHub release whose body is the changelog GitHub generates
from the commits and merged PRs since the previous tag, and lands a commit on
`main` pointing the deployment manifests at the new version.

`pyproject.toml` declares `dynamic = ["version"]`, and hatch-vcs derives it from
`git describe`: a build on the tag is `1.6.0`, three commits later it is
`1.6.1.dev3+g<sha>`. A tag therefore cannot disagree with what it publishes. Two
consequences that are easy to trip over:

- **Every job that installs or builds the package needs the tags**, which is why
  the checkouts pass `fetch-depth: 0`. A shallow clone leaves hatch-vcs with
  nothing to derive from, and the install fails outright rather than quietly
  mislabelling itself.
- **The image build has no repository to read.** `.dockerignore` keeps `.git` out
  of the build context on purpose, so the Dockerfile takes a `SYS_STATS_VERSION`
  build argument and feeds it to setuptools-scm through
  `SETUPTOOLS_SCM_PRETEND_VERSION`. The workflow always passes it; a bare local
  `docker build` gets `0.0.0.dev0+local`, which is meant to look wrong.

Three files still spell a version out, because they name an image tag a human
pulls and so cannot derive anything at deploy time: `compose.yaml`, the README
quickstart, and `chart/Chart.yaml` (`appVersion`, plus a patch bump of the chart's
own `version`, which otherwise moves independently). Do not edit them by hand:
[scripts/sync-versions.sh](scripts/sync-versions.sh) rewrites all three, and the
release workflow runs it on `main` once the images are out. The trade-off is
deliberate and worth knowing: the *tagged* tree still shows the previous version
in those three files, `main` does not. If that push is ever refused, run
`./scripts/sync-versions.sh 1.6.0` and commit it yourself.

Everything else derives the version and must stay that way:
`src/sys_stats/__init__.py` reads it from the installed package metadata, the
workflow injects `org.opencontainers.image.version` at build time, and the chart's
`image.tag` is `"{{ .Chart.AppVersion }}"`, rendered through `tpl`. The Dockerfile
carries no literal version either: `LABEL` can only expand `ARG`/`ENV`, never the
output of a `RUN`.

Semver applies to the package as a whole: the `/stats` payload is a public contract
(see [Architecture](#architecture)), so renaming or removing a key there is a major
bump, not a patch.

## Deployment constraints worth knowing

- `pid: host` (see [compose.yaml](compose.yaml)) is necessary and sufficient for the
  process tables: /proc is world-readable, so the image's unprivileged UID 10001
  sees every process complete, name, command line and memory alike. The same holds
  for hwmon temperature and fan sensors (/sys is mounted read-only into every
  container by default) and for the GPU panels (the NVIDIA container runtime
  injects nvidia-smi and the driver libraries against the `compute`/`utility`
  capabilities reservation, independent of any privilege bit). None of that needs
  `privileged: true`.
- IPMI is the one exception. The two `ipmi/`-prefixed collectors reach the BMC
  through `/dev/ipmi0`, a root:root 0600 character device a normal container has
  no entry for, so they need the device mapped in plus root, or the device's
  group, to open it. `compose.ipmi.yaml` is the opt-in overlay for that
  (`docker compose -f compose.yaml -f compose.ipmi.yaml up -d`). Kubernetes has no
  per-device request, so `chart/values.yaml`'s commented IPMI block falls back to
  `privileged: true` plus `runAsUser: 0` plus a hostPath CharDevice volume instead.
- `ipmitool` is now an unconditional Dockerfile dependency (see
  [Dockerfile](Dockerfile)): the sampler calls the two IPMI collectors on its
  first pass regardless of hardware, so without the binary they raise
  `FileNotFoundError` and degrade to an empty list on every machine, not only
  ones without a BMC.
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
