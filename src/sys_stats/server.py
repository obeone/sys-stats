#!/usr/bin/env python3

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


#: How long ``/stats`` waits for the sampler's first snapshot on a cold
#: start, in seconds, before giving up rather than blocking the request
#: forever.
_FIRST_SNAPSHOT_TIMEOUT = 5.0

# Per-process ranking keys carried by the sampler payload. ``/stats`` slices
# each of these down to the requested ``limit``; every other key is returned
# as sampled.
_RANKING_KEYS = ("top_cpu", "top_memory", "top_gpu_processes")


def _slice_to_limit(stats: dict[str, Any], limit: int) -> dict[str, Any]:
    """Return a copy of a cached payload with its rankings capped at ``limit``.

    The sampler always collects each ranking up to its own cap
    (``SYS_STATS_TOP_PROCESSES_MAX``), independent of what any individual
    request asks for. This slices a *copy* down to what the caller actually
    requested, so mutating the returned dict (as Flask's JSON encoder does
    not, but a future caller might) can never corrupt the shared cache.

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
        A shallow copy of ``stats`` with ``top_cpu``, ``top_memory`` and
        ``top_gpu_processes`` replaced by sliced list copies, same key order.
    """
    response = dict(stats)
    warned = False
    for key in _RANKING_KEYS:
        sampled = stats[key]
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
        response[key] = list(sampled[:limit])
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
            timeout=_FIRST_SNAPSHOT_TIMEOUT
        )

    if stats is None:
        # Every sampler iteration failed within the wait window (the sampler
        # itself is crash-proof and logs each failure). There is nothing
        # real to serve; returning an empty-shell payload keeps existing
        # consumers, which expect 200 + JSON, working rather than handing
        # them a 500.
        logger.error("No sampler snapshot available after waiting; returning an empty payload")
        return jsonify({})

    return jsonify(_slice_to_limit(stats, limit))

def main() -> None:
    """Console-script entry point: run the Flask metrics server.

    Honours the ``FLASK_DEBUG``, ``HOST`` and ``PORT`` environment variables.
    """
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))

    # In debug mode Flask's reloader re-execs this module in a worker
    # process (marked by WERKZEUG_RUN_MAIN) after an initial import in a
    # lightweight monitor process. Starting the sampler in both would spawn
    # two threads calling psutil.cpu_percent independently, corrupting each
    # other's baseline exactly like the two-caller problem the sampler
    # exists to prevent, so skip it in the monitor process.
    if not debug_mode or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        sampler.start()

    app.run(host=host, port=port, debug=debug_mode)


if __name__ == '__main__':
    main()
