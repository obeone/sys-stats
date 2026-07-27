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

### 📦 Installing as a CLI

Sys-Stats is a proper Python package, so any standard Python installer puts the
commands on your `PATH`. Pick the one you already use.

Whichever method you choose, you get two console scripts:

| Command            | Purpose                                            |
| ------------------ | -------------------------------------------------- |
| `sys-stats`        | The Rich terminal dashboard (client).              |
| `sys-stats-server` | The Flask metrics API + web UI (serves `/stats`).  |

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

## 📺 Using the CLI

To use the terminal dashboard for live monitoring, run:

```bash
sys-stats [--url http://localhost:5000/stats] [--interval 5]
```

This launches the dashboard with a 5-second refresh interval (adjustable at
runtime with `+` / `-`; press `h` for the full keyboard help). The API URL can
also be set via the `SYS_STATS_API_URL` environment variable.

You're all set! Enjoy the Sys-Stats Dashboard.

## 🧑‍💻 Contributing

We welcome contributions from the community! Feel free to open issues, suggest features, or submit pull requests. Let's build a better Sys-Stats Dashboard together!

## 📝 Notes

This repo is clearly messy, but it was supposed to be only for my own use!
