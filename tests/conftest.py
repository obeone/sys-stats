"""Shared pytest configuration for the whole test suite.

Importing :mod:`sys_stats.server` now starts the background sampler at
module scope (so any WSGI entry point that only imports ``app`` gets a live
cache, not just the ``sys-stats-server`` console script). pytest imports
``tests/test_stats_endpoint.py`` -- which imports ``sys_stats.server`` -- at
collection time, before any test-level ``monkeypatch`` can run. Left alone,
that import would spawn a real sampler thread calling real ``psutil`` /
``GPUtil`` / ``nvidia-smi``, touching the actual machine this suite is meant
to never touch.

Setting ``SYS_STATS_AUTOSTART=0`` here, before that import happens, trips
the dedicated opt-out ``sys_stats.server`` exposes for exactly this case, so
no real background thread spawns during collection. This is intentionally a
different flag than ``FLASK_DEBUG``: that one is the production debug-mode
toggle and must stay off so the suite runs under the same conditions as
production.
"""

import os

os.environ.setdefault("SYS_STATS_AUTOSTART", "0")
