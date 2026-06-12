"""Sys-Stats: a real-time system, GPU and Ollama monitoring dashboard.

Exposes two entry points:

- ``sys-stats`` -- the Rich terminal dashboard (:mod:`sys_stats.cli`).
- ``sys-stats-server`` -- the Flask metrics API/web UI (:mod:`sys_stats.server`).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sys-stats")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
