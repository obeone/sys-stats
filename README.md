# 📊 Sys-Stats Dashboard ✨

Welcome to the **Sys-Stats Dashboard**! This project is designed to monitor and visualize system performance in real-time through a sleek and interactive dashboard. Whether you're interested in CPU, RAM, or GPU usage, this tool provides comprehensive insights at your fingertips. 

## 🎢 Features

- **Real-time Monitoring**: Get instant updates on system metrics such as CPU load, memory usage, GPU stats, and essential processes.
  
- **Interactive Dashboards**: Use the web-based or command-line interface for intuitive dashboards, allowing detailed insights into system behavior.
  
- **Cross-Platform Support**: Deployed using Docker, ensuring consistency and easy setup across different operating environments.
  
- **Customizable Refresh Rates**: Adapt the monitoring frequency to suit your needs, enabling faster updates or conserving resources when needed.

- **Ollama API Integration**: Out-of-the-box compatibility with the Ollama API for additional system-specific metrics and insights.

## 🎆 Screenshots

![Web Dashboard](https://raw.githubusercontent.com/obeone/sys-stats/main/docs/web.png)
![CLI Dashboard](https://raw.githubusercontent.com/obeone/sys-stats/main/docs/cli.png)

## 🚀 Getting Started

Ready to get your Sys-Stats Dashboard up and running? Follow these steps:

### 🐳 Running with Docker

#### Requirements

Ensure you have the following installed on your system:

- Docker 🐳
- NVIDIA drivers (if you want GPU monitoring) 🎮

#### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/obeone/sys-stats.git
   cd sys-stats
   `````

1. **Running the System:**

    - **With nvidia GPU Support:**

    ```bash
    docker compose -f compose.yaml -f compose.gpu.yaml up -d
    ```

    - **Without nvidia GPU Support:**
  
    ```bash
    docker compose up -d
    ```

   The service will be running at `http://localhost:5000`.

### ☸️ Running on Kubernetes

[`chart/`](chart/) is a Helm chart built on the
[bjw-s common library](https://github.com/bjw-s-labs/helm-charts):

```bash
helm dependency update chart/
helm upgrade --install sys-stats ./chart --namespace monitoring --create-namespace
```

The dashboard reports on the node the pod lands on, so the chart defaults to the
host PID namespace and a privileged container — the process tables are empty
without them. Pin the pod to the machine you actually want to watch, claim a GPU
if you want the NVIDIA panels, and point `OLLAMA_API_URL` somewhere for the
Ollama one. [`chart/README.md`](chart/README.md) has the values for all three.

### 📦 Installing as a CLI

Sys-Stats is a proper Python package, so any standard Python installer puts the
commands on your `PATH`. Pick the one you already use.

Whichever method you choose, you get two console scripts:

| Command            | Purpose                                            |
| ------------------ | -------------------------------------------------- |
| `sys-stats`        | The Rich terminal dashboard (client).              |
| `sys-stats-server` | The Flask metrics API + web UI (serves `/stats` and `/panel`). |

#### With uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is the fastest option and isolates the tool
in its own environment:

```bash
# Install straight from the repository...
uv tool install git+https://github.com/obeone/sys-stats.git

# ...or from a local checkout
git clone https://github.com/obeone/sys-stats.git
uv tool install ./sys-stats
```

Upgrade later with `uv tool upgrade sys-stats`, remove with
`uv tool uninstall sys-stats`.

To run the dashboard once without installing anything permanent:

```bash
uvx --from git+https://github.com/obeone/sys-stats.git sys-stats
```

Building the distributable artifacts from a checkout is `uv build`, which drops
a wheel and an sdist in `dist/`.

#### With pipx

[`pipx`](https://pipx.pypa.io/) also installs the CLI in a dedicated virtual
environment, keeping it isolated from your system Python:

```bash
# From the repository...
pipx install git+https://github.com/obeone/sys-stats.git

# ...or from a local checkout
pipx install ./sys-stats
```

Upgrade with `pipx upgrade sys-stats`, remove with `pipx uninstall sys-stats`.

#### With pip

Plain `pip` works too — ideally inside a virtual environment so it doesn't
pollute your system packages:

```bash
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# From the repository...
pip install git+https://github.com/obeone/sys-stats.git

# ...or from a local checkout
pip install ./sys-stats
```

With this method the `sys-stats` and `sys-stats-server` commands are available
whenever the virtual environment is activated.

### ⚙️ Running Without Docker

#### Prerequisites

- **Python 3.10+**
- **NVIDIA drivers**: for GPU monitoring (optional).

#### Setup Instructions

1. **Install the package** (see [Installing as a CLI](#-installing-as-a-cli))
   or, for development, in an editable virtual environment:

   ```bash
   git clone https://github.com/obeone/sys-stats.git
   cd sys-stats
   uv venv && source .venv/bin/activate
   uv pip install -e '.[dev]'
   ```

   The `[dev]` extra adds `pytest` and `ruff`. Run the checks with:

   ```bash
   pytest
   ruff check .
   ```

   Note for macOS: port 5000 is used by AirPlay Receiver, so start the server
   with `PORT=5051 sys-stats-server` if you don't want to disable it.

2. **Configure environment variables (optional):**

   To enable Ollama metrics, point the server at your Ollama instance:

   ```bash
   export OLLAMA_API_URL="http://localhost:11434"
   ```

3. **Start the server:**

   ```bash
   sys-stats-server
   ```

   The application starts on `http://localhost:5000`. It honours the `HOST`,
   `PORT` and `FLASK_DEBUG` environment variables.

   A background sampler thread, not each request, collects the metrics; it
   starts as soon as the server module is imported, so any WSGI entry point
   works, not just `sys-stats-server`. Set `SYS_STATS_AUTOSTART` to `0`,
   `false` or `no` to skip that autostart (default on — this is for
   embedding the module without a live sampler, not something a normal
   deployment needs to touch).
   `SYS_STATS_SAMPLE_INTERVAL` sets how many seconds it waits between
   samples (default `2.0`), and `SYS_STATS_TOP_PROCESSES_MAX` caps how many
   entries it collects per per-process ranking (default `50`) — a `?limit=`
   above that cap only returns what was already sampled.

4. **`/panel` (optional):** a compact, frozen-schema endpoint built for a
   small embedded display (an ESP32-S3 wall panel, in particular) polling
   every few seconds — no process lists, no Ollama data, just CPU/RAM/swap/
   GPU/sensor numbers plus an `age` in seconds telling the display how stale
   the sample is. It shares the same background sampling pass as `/stats`,
   so enabling it costs nothing extra in `nvidia-smi` calls. Three optional
   caps truncate its lists for a display with limited room:
   `SYS_STATS_PANEL_MAX_TEMPS`, `SYS_STATS_PANEL_MAX_FANS` and
   `SYS_STATS_PANEL_MAX_GPUS` — each unset by default, meaning no cap.

5. **`SYS_STATS_PANEL_ONLY` (optional, security-relevant):** set to `1`,
   `true` or `yes` to register only the `/panel` route — `/`, `/stats` and
   `/favicon.png` are never registered at all, so a request to them gets
   Flask's own 404 rather than a guarded rejection. `/stats` exposes the
   full host process table, complete command lines included, with no
   authentication; on a host you do not fully trust the network of (a
   hypervisor on a LAN whose guest WiFi shares a VLAN with the main
   network, say), and where the wall-display consumer only ever needs
   `/panel`, removing the route beats guarding it. Default is off: unset
   registers every route exactly as before this flag existed. Note that the
   Helm chart's default liveness/readiness/startup probes hit `/`, so
   enabling this in the chart means repointing them at `/panel` too (see
   `chart/values.yaml`).

## 📺 Using the CLI

To use the terminal dashboard for live monitoring, run:

```bash
sys-stats [--url http://localhost:5000/stats] [--interval 5]
```

This launches the dashboard with a 5-second refresh interval. The API URL can
also be set via the `SYS_STATS_API_URL` environment variable.

Every panel carries a number in its title, and a one line key helper is pinned
at the bottom of the screen. Press `h` for the full list.

| Key | Action |
| --- | --- |
| `q` / `r` | Quit, force a refresh |
| `p` / `-` / `+` | Pause, slow down, speed up |
| `1`..`9` | Focus the panel carrying that number |
| `Tab` / `Shift+Tab` | Focus the next or previous panel |
| `Enter` / `Esc` | Zoom the focused panel full screen, and come back |
| arrows, `PgUp`/`PgDn`, `Home`/`End` | Scroll the focused panel |

A panel that cannot show every row says how many it is hiding, so a busy host
never drops rows silently. A zoomed panel keeps refreshing and keeps scrolling.

The layout follows the terminal: a wide one gets a row of columns, a narrower one
falls back to a grid, and panels stop where their content stops instead of
framing empty space. On a machine with several GPUs the summary keeps cumulated
figures, each card gets its own detail table (or one row per card when they no
longer fit side by side), and the GPU processes list tells you which card each
process is holding VRAM on. The Ollama panel shows each model's context window
next to its VRAM footprint.

You're all set! Enjoy the Sys-Stats Dashboard.

## 🧑‍💻 Contributing

We welcome contributions from the community! Feel free to open issues, suggest features, or submit pull requests. Let's build a better Sys-Stats Dashboard together!

## 📝 Notes

This repo is clearly messy, but it was supposed to be only for my own use!
