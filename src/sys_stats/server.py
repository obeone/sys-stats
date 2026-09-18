#!/usr/bin/env python3

import copy
import logging
import os
from typing import Any

import coloredlogs
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from . import sampler

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Start the background sampler as soon as this module is imported, not just
# when server.main() runs the ``sys-stats-server`` console script. A WSGI
# entry point that imports ``app`` directly (``gunicorn sys_stats.server:app``,
# ``flask --app sys_stats.server run``) never calls main(), and without this
# the cache would never fill: every request would burn the full
# wait_for_first_snapshot() timeout and still find nothing. start() is
# lock-guarded and idempotent, so a later call from main() is harmless.
#
# In Flask's debug reloader the module is imported twice: once by the
# lightweight monitor process, once by the actual worker (marked by
# WERKZEUG_RUN_MAIN). Starting the sampler in both would spawn two threads
# calling psutil.cpu_percent independently, corrupting each other's baseline
# exactly like the two-caller problem the sampler exists to prevent, so it is
# skipped in the monitor process.
_in_debug_reloader_monitor = (
    os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    and os.environ.get('WERKZEUG_RUN_MAIN') != 'true'
)

# SYS_STATS_AUTOSTART is an unrelated concern: an explicit opt-out for
# callers -- namely this project's own test suite -- that need to import
# this module without spawning a thread that touches the real machine. It
# defaults to on; production entry points never need to set it.
_autostart_disabled = os.getenv('SYS_STATS_AUTOSTART', '1').strip().lower() in ('0', 'false', 'no')

if not _in_debug_reloader_monitor and not _autostart_disabled:
    sampler.start()

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.png')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'favicon.png', mimetype='image/png')


#: Floor on how long ``/stats`` waits for the sampler's first snapshot on a
#: cold start, in seconds, regardless of how short the configured sampling
#: interval is.
_MIN_FIRST_SNAPSHOT_TIMEOUT = 5.0


def _first_snapshot_timeout() -> float:
    """Compute how long ``/stats`` waits for the sampler's first snapshot.

    ``sampler._run`` waits one full sampling interval before it takes its
    first sample (see :mod:`sys_stats.sampler`). A wait shorter than that
    would time out on every cold start whenever
    ``SYS_STATS_SAMPLE_INTERVAL`` is configured above the floor, so the
    timeout is derived from the same interval instead of a value hardcoded
    independently of it.

    Returns
    -------
    float
        Seconds to wait: twice the configured interval plus a one second
        margin, never less than :data:`_MIN_FIRST_SNAPSHOT_TIMEOUT`.
    """
    interval = sampler._get_sample_interval()
    return max(_MIN_FIRST_SNAPSHOT_TIMEOUT, interval * 2 + 1.0)

# Per-process ranking keys carried by the sampler payload. ``/stats`` slices
# each of these down to the requested ``limit``; every other key is returned
# as sampled.
_RANKING_KEYS = ("top_cpu", "top_memory", "top_gpu_processes")


def _rank_gpu_processes(processes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Re-select the top ``limit`` GPU processes by VRAM, then reapply display order.

    The sampler's cached ``top_gpu_processes`` is stored in *display* order
    (grouped by ``gpu_index``, heaviest first within each card; see
    :func:`sys_stats.collectors.get_gpu_processes`), not in descending
    ``memory_used`` order. Slicing that list directly would keep whichever
    GPU happens to sort first rather than the biggest VRAM consumers across
    every card. This mirrors the collector's own two-phase sort exactly:
    select by memory first, then reorder the surviving slice for display.

    Parameters
    ----------
    processes : list of dict
        The cached ``top_gpu_processes`` entries, in display order.
    limit : int
        Maximum number of entries to keep.

    Returns
    -------
    list of dict
        The ``limit`` heaviest entries, ordered by ``gpu_index`` (unresolved
        entries last) then by descending ``memory_used``.
    """
    by_memory = sorted(processes, key=lambda p: -p["memory_used"])
    top = by_memory[:limit]
    top.sort(key=lambda p: (p["gpu_index"] is None, p["gpu_index"] or 0, -p["memory_used"]))
    return top


def _slice_to_limit(stats: dict[str, Any], limit: int) -> dict[str, Any]:
    """Return a copy of a cached payload with its rankings capped at ``limit``.

    The sampler always collects each ranking up to its own cap
    (``SYS_STATS_TOP_PROCESSES_MAX``), independent of what any individual
    request asks for. This deep-copies the payload before slicing it down to
    what the caller actually requested, so mutating the returned dict, down
    to a value nested inside a ranking entry or inside another top-level key
    such as ``ram``, can never corrupt the shared cache.

    Parameters
    ----------
    stats : dict
        The cached payload, as returned by
        :func:`sys_stats.collectors.collect_stats`.
    limit : int
        Maximum number of entries to keep in each ranking.

    Returns
    -------
    dict
        A deep copy of ``stats`` with ``top_cpu``, ``top_memory`` and
        ``top_gpu_processes`` replaced by rankings capped at ``limit``, same
        key order. ``top_gpu_processes`` is re-selected by descending VRAM
        usage before truncation, then redisplayed in the collector's order;
        the other two rankings are already sorted by their metric, so a
        plain slice keeps the heaviest entries.
    """
    response = copy.deepcopy(stats)
    warned = False
    for key in _RANKING_KEYS:
        sampled = response[key]
        if limit > len(sampled) and not warned:
            # The sampler already returned everything it collected; asking
            # for more than that cannot be satisfied without re-collecting
            # inline, which is exactly what the sampler exists to avoid.
            logger.warning(
                f"Requested limit={limit} exceeds the {len(sampled)} entries "
                f"sampled; returning what is available instead of "
                f"re-collecting"
            )
            warned = True
        if key == "top_gpu_processes":
            response[key] = _rank_gpu_processes(sampled, limit)
        else:
            response[key] = sampled[:limit]
    return response


@app.route('/stats', methods=['GET'])
def get_stats():
    """Serve the ``/stats`` payload consumed by the web UI and the Rich CLI.

    Reads the latest snapshot the background sampler (:mod:`sys_stats.sampler`)
    has already collected instead of collecting inline, then slices the
    per-process rankings down to the requested ``limit``. On a cold start,
    with no snapshot yet, it waits briefly for the sampler's first sample
    rather than returning an error.

    Returns
    -------
    flask.Response
        JSON response whose top-level keys are the public contract of the
        project; adding keys is a minor bump, renaming one is a major bump.
    """
    limit_str = request.args.get("limit", "5")
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 5

    stats, _wall_ts, _monotonic_ts = sampler.get_snapshot()
    if stats is None:
        stats, _wall_ts, _monotonic_ts = sampler.wait_for_first_snapshot(
            timeout=_first_snapshot_timeout()
        )

    if stats is None:
        # Every sampler iteration failed within the wait window (the sampler
        # itself is crash-proof and logs each failure), or the sampler never
        # started at all. A 200 + {} here would be a lie: both consumers (the
        # inline JS dashboard and the Rich CLI) expect every key in the
        # contract to be present and would fail on a missing key instead of
        # seeing the real error. 503 tells the caller to retry rather than
        # silently rendering nothing. With the sampler now started at module
        # import time, this path is only reachable as a genuine failure, not
        # during normal startup.
        logger.error("No sampler snapshot available after waiting; returning 503")
        return jsonify({}), 503

    return jsonify(_slice_to_limit(stats, limit))

def main() -> None:
    """Console-script entry point: run the Flask metrics server.

    Honours the ``FLASK_DEBUG``, ``HOST`` and ``PORT`` environment variables.
    """
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))

    # The sampler is already started at module import time, above; nothing
    # left to do here but run the app.
    app.run(host=host, port=port, debug=debug_mode)


if __name__ == '__main__':
    main()
