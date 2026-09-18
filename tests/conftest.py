"""Shared pytest configuration for the whole test suite.

Importing :mod:`sys_stats.server` now starts the background sampler at
module scope (so any WSGI entry point that only imports ``app`` gets a live
cache, not just the ``sys-stats-server`` console script). pytest imports
``tests/test_stats_endpoint.py`` -- which imports ``sys_stats.server`` -- at
collection time, before any test-level ``monkeypatch`` can run. Left alone,
that import would spawn a real sampler thread calling real ``psutil`` /
``GPUtil`` / ``nvidia-smi``, touching the actual machine this suite is meant
to never touch.

Setting ``FLASK_DEBUG=true`` here, before that import happens, trips the
same guard ``sys_stats.server`` uses to skip the sampler in Flask's debug
reloader monitor process (no ``WERKZEUG_RUN_MAIN`` set), so no real
background thread spawns during collection.
"""

import os

os.environ.setdefault("FLASK_DEBUG", "true")
