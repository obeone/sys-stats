#!/usr/bin/env python3

import copy
import logging
import os
import socket
import time
from typing import Any

import coloredlogs
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

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

#: Env var gating whether the app registers the general-purpose routes
#: (``/``, ``/stats``, ``/favicon.png``) at all, versus only ``/panel`` and
#: its command-line-free companion ``/panel/procs``.
_PANEL_ONLY_ENV = "SYS_STATS_PANEL_ONLY"


def _panel_only_enabled() -> bool:
    """Read whether only the ``/panel`` routes should be registered.

    This is a security control, not a convenience toggle. ``/stats`` exposes
    the full host process table -- complete command lines -- unauthenticated,
    and a token passed as a CLI argument is readable to anyone on the same
    host or LAN segment. A route that was never registered cannot be reached
    by a path-traversal trick, a proxy quirk, or a future middleware bug the
    way a route guarded by a ``before_request`` check still could -- hence
    this gates route *registration* itself, not a request-time check.

    Returns
    -------
    bool
        ``True`` when :data:`_PANEL_ONLY_ENV` is set to ``"1"``, ``"true"``
        or ``"yes"`` (case-insensitive). ``False`` otherwise, including when
        it is unset -- the default registers every route exactly as before
        this flag existed.
    """
    return os.getenv(_PANEL_ONLY_ENV, "").strip().lower() in ("1", "true", "yes")


# Read once at import time: routes are registered exactly once, when this
# module is imported, so there is nothing to gain from re-reading the env
# var per request the way the sampler's own env readers do.
_panel_only = _panel_only_enabled()

#: Env var naming which physical instance the web UI is looking at, shown in
#: the page title and body so two co-located deployments (e.g. a Kubernetes
#: pod seeing a Talos VM, and a second instance on the Proxmox hypervisor
#: underneath it, both reporting on the same box under the same app name)
#: are told apart at a glance instead of looking like one is lying.
_INSTANCE_LABEL_ENV = "SYS_STATS_INSTANCE_LABEL"


def _instance_label() -> str | None:
    """Read the optional instance label displayed in the web UI.

    Read per-request rather than cached at import time, like the ``/panel``
    truncation caps (:func:`_get_panel_cap`), so relabeling an instance takes
    effect without a restart.

    This is human-facing prose for the web UI title/body, meant to be
    rewritten for readability (e.g. "Talos VM (bart-worker)"). It must never
    be merged with or used to derive :func:`_panel_hostname`'s ``host``
    value: that one is a machine identity a consumer does an exact string
    comparison against, and relabeling this for looks must never change it.

    Returns
    -------
    str or None
        The env var's value with surrounding whitespace stripped, or
        ``None`` when it is unset or blank -- either case renders the page
        exactly as it looked before this variable existed: no empty
        element, no dangling separator.
    """
    raw = os.getenv(_INSTANCE_LABEL_ENV)
    if raw is None:
        return None
    label = raw.strip()
    return label or None


#: Env var overriding the ``host`` value ``/panel`` reports, for hosts where
#: ``socket.gethostname()`` is meaningless (e.g. a container reporting its
#: pod name rather than the physical machine it runs on).
_HOSTNAME_ENV = "SYS_STATS_HOSTNAME"


def _panel_hostname() -> str:
    """Read the machine identity reported in the ``/panel`` payload's ``host`` key.

    Read per request, not cached at import time: ``hostnamectl
    set-hostname`` changes the value at runtime, and caching it at import
    would keep announcing the old name indefinitely for the cost of one
    cheap syscall per request.

    ``/panel``'s other guards (``ready``, ``age``, ``err``) protect against
    data that is missing or stale. None of them protects against data that
    is fresh, correct, and describing a different machine than the consumer
    thinks it is talking to -- which a duplicated IP, a repointed DNS
    record, a migrated DHCP lease, or a service moved while keeping its name
    can all produce without raising any error. ``host`` lets the consumer
    detect that case with a plain string comparison.

    This is deliberately independent from :func:`_instance_label`: that one
    is prose meant to be rewritten for the web UI, this one is a machine
    identity a consumer compares byte-for-byte (see its docstring for why
    merging the two was rejected). Never derive one from the other.

    Returns
    -------
    str
        :data:`_HOSTNAME_ENV`'s value verbatim when set, otherwise
        ``socket.gethostname()`` verbatim -- neither is shortened or
        normalised, since a consumer comparing this against an FQDN
        constant would break if it were.
    """
    override = os.getenv(_HOSTNAME_ENV)
    if override:
        return override
    return socket.gethostname()


@app.errorhandler(Exception)
def handle_exception(e):
    """Turn an uncaught exception into a JSON 500, without masking routing errors.

    ``Exception`` also matches Werkzeug's ``HTTPException`` (404, 405, ...),
    since it is a subclass. Without the check below, a request to a route
    that was never registered -- exactly what ``SYS_STATS_PANEL_ONLY`` relies
    on for `/`, `/stats` and `/favicon.png` -- would be coerced into a 500
    "Unhandled exception" instead of Flask's own 404, which defeats the
    whole point of not registering the route: the response would look like a
    crash rather than like the endpoint never existed.
    """
    if isinstance(e, HTTPException):
        return e
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": str(e)}), 500

if not _panel_only:
    @app.route('/')
    def index():
        return render_template('index.html', instance_label=_instance_label())


    @app.route('/favicon.png')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'templates'), 'favicon.png', mimetype='image/png')


#: Floor on how long ``/stats`` waits for the sampler's first snapshot on a
#: cold start, in seconds, regardless of how short the configured sampling
#: interval is.
_MIN_FIRST_SNAPSHOT_TIMEOUT = 5.0

#: Headroom added on top of the one interval the sampler sleeps before its
#: first sample, covering the duration of the collection itself: a psutil
#: process sweep, up to two ``nvidia-smi`` subprocess round trips and, when
#: ``OLLAMA_API_URL`` is set, one HTTP call. Nobody timed that pass, so this
#: margin is a judgement about how much slack a first collection deserves,
#: not a multiple of a measured duration.
_FIRST_SNAPSHOT_COLLECTION_MARGIN = 3.0


def _first_snapshot_timeout() -> float:
    """Compute how long ``/stats`` waits for the sampler's first snapshot.

    ``sampler._run`` primes the ``cpu_percent`` baseline as soon as it
    starts, then sleeps exactly one sampling interval before taking its
    first sample (see :mod:`sys_stats.sampler`). A cold start therefore
    costs one interval plus one collection, and the wait is derived from
    that rather than hardcoded independently of the interval.

    This used to double the interval, which was not a derivation at all: it
    compensated for a sampler loop that waited once *before* priming and
    once more before sampling. With that fixed, doubling would only make a
    genuinely dead sampler hold the request thread twice as long as needed.

    Returns
    -------
    float
        Seconds to wait: the configured interval plus
        :data:`_FIRST_SNAPSHOT_COLLECTION_MARGIN`, never less than
        :data:`_MIN_FIRST_SNAPSHOT_TIMEOUT`.
    """
    interval = sampler._get_sample_interval()
    return max(_MIN_FIRST_SNAPSHOT_TIMEOUT, interval + _FIRST_SNAPSHOT_COLLECTION_MARGIN)

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

    # The warning is about the sampler's CAP truncating data, so it is the
    # cap -- not the length of any one ranking -- that ``limit`` is compared
    # against. A ranking shorter than ``limit`` while the cap was never
    # reached simply means the host has nothing more to report: on any
    # machine without an NVIDIA GPU ``top_gpu_processes`` is permanently
    # empty, and comparing against its length made every single request log
    # "exceeds the 0 entries sampled", a standing false alarm that made a
    # perfectly healthy 200 look like a 503 that failed to fire.
    cap = sampler._get_top_processes_cap()
    if limit > cap:
        # The sampler only ever collects up to its own cap; asking for more
        # than that cannot be satisfied without re-collecting inline, which
        # is exactly what the sampler exists to avoid.
        logger.warning(
            f"Requested limit={limit} exceeds the sampler's cap of {cap} "
            f"entries (SYS_STATS_TOP_PROCESSES_MAX); returning what was "
            f"sampled instead of re-collecting"
        )

    for key in _RANKING_KEYS:
        sampled = response[key]
        if key == "top_gpu_processes":
            response[key] = _rank_gpu_processes(sampled, limit)
        else:
            response[key] = sampled[:limit]
    return response


if not _panel_only:
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
            200 with the JSON payload whose top-level keys are the public
            contract of the project (adding keys is a minor bump, renaming one
            is a major bump), or 503 with ``{}`` when no snapshot landed within
            the cold-start wait, or when the one that did is not a complete
            payload.
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

        # "A snapshot exists" and "the snapshot is usable" are two different
        # questions, and only the first one was ever asked here. A non-None but
        # incomplete cache entry -- an empty dict, or anything not produced by
        # collect_stats -- used to reach _slice_to_limit and surface as a 500
        # KeyError naming a contract key, which tells a caller nothing about
        # what to do next. Absence of usable data is a 503 and a retry, exactly
        # like the branch above and like /panel's own contract.
        missing = [key for key in _RANKING_KEYS if key not in stats]
        if missing:
            logger.error(
                f"Sampler snapshot is missing {', '.join(missing)}; it is not a "
                f"complete /stats payload, returning 503"
            )
            return jsonify({}), 503

        return jsonify(_slice_to_limit(stats, limit))


# Env vars capping each /panel list, each unset by default meaning no cap.
# Read per-request (not cached) so a config change takes effect without a
# restart, matching the sampler's own env readers.
_PANEL_MAX_TEMPS_ENV = "SYS_STATS_PANEL_MAX_TEMPS"
_PANEL_MAX_FANS_ENV = "SYS_STATS_PANEL_MAX_FANS"
_PANEL_MAX_GPUS_ENV = "SYS_STATS_PANEL_MAX_GPUS"


def _get_panel_cap(env_var: str) -> int | None:
    """Read an optional truncation cap for the ``/panel`` route.

    Unlike the sampler's env readers (:func:`sys_stats.sampler._get_sample_interval`,
    :func:`sys_stats.sampler._get_top_processes_cap`), there is no numeric
    default here: an unset, non-numeric or non-positive value means "do not
    truncate" rather than falling back to some fixed cap.

    Parameters
    ----------
    env_var : str
        Name of the environment variable to read.

    Returns
    -------
    int or None
        The configured cap, or ``None`` for "no cap".
    """
    raw = os.getenv(env_var)
    if raw is None:
        return None

    try:
        cap = int(raw)
    except ValueError:
        logger.warning(f"Ignoring non-numeric {env_var}={raw!r}; no cap applied")
        return None

    if cap <= 0:
        logger.warning(f"Ignoring non-positive {env_var}={raw!r}; no cap applied")
        return None

    return cap


def _cap_list(items: list[Any], cap: int | None) -> list[Any]:
    """Slice ``items`` to at most ``cap`` entries, without re-sorting.

    Parameters
    ----------
    items : list
        Entries already sorted by the collector that produced them (by name
        for temperatures and fans; by GPU index for the GPU list).
    cap : int or None
        Maximum number of entries to keep, or ``None`` for no cap.

    Returns
    -------
    list
        ``items`` truncated to ``cap`` entries. Slicing after the incoming
        sort keeps truncation deterministic -- never re-sort here.
    """
    return items if cap is None else items[:cap]


def _round1(value: float | None) -> float:
    """Round a metric to 1 decimal place, treating ``None`` as ``0.0``.

    Some GPUtil-reported fields (temperature in particular) can surface as
    ``None`` depending on the driver, and the ``/panel`` contract has no
    nullable fields -- the firmware's ArduinoJson filter reads a fixed POD
    struct with no room for a missing value.

    Parameters
    ----------
    value : float or None
        The raw metric.

    Returns
    -------
    float
        ``value`` rounded to 1 decimal, or ``0.0`` when ``value`` is ``None``.
    """
    return round(value, 1) if value is not None else 0.0


def _build_panel_gpu_entry(gpu: dict[str, Any]) -> dict[str, Any]:
    """Convert one ``/stats``-shaped GPU entry into its ``/panel`` shape.

    ``/stats`` deliberately keeps ``memoryTotal`` in MiB next to
    ``memoryUsed`` in bytes (see CLAUDE.md, issue #16, a known and
    deliberately preserved inconsistency). ``/panel`` must not inherit it:
    ``mem_total`` is converted to bytes here so every memory field in the
    panel payload shares the same unit.

    Parameters
    ----------
    gpu : dict
        One entry of the cached ``stats["gpu"]`` list, as produced by
        :func:`sys_stats.collectors.collect_stats`.

    Returns
    -------
    dict
        The ``/panel`` contract's per-GPU shape: ``i``, ``n``, ``load``,
        ``mem_used``, ``mem_total``, ``mem_pct``, ``temp``, ``fan``, ``power``.
    """
    return {
        "i": gpu["id"],
        "n": gpu["name"],
        "load": _round1(gpu.get("load")),
        "mem_used": gpu.get("memoryUsed") or 0,
        "mem_total": int((gpu.get("memoryTotal") or 0) * 1024 * 1024),
        "mem_pct": _round1(gpu.get("memoryPercent")),
        "temp": _round1(gpu.get("temperature")),
        "fan": int(round(gpu.get("fanSpeed") or 0.0)),
        "power": _round1(gpu.get("powerDraw")),
    }


def _build_panel_payload(
    stats: dict[str, Any],
    extras: dict[str, Any],
    wall_ts: float,
    monotonic_ts: float,
) -> dict[str, Any]:
    """Assemble the ``/panel`` response from the shared snapshot and its extras.

    Parameters
    ----------
    stats : dict
        The same cached payload ``/stats`` serves, as returned by
        :func:`sys_stats.collectors.collect_stats`. Its ``gpu``, ``cpu`` and
        ``ram`` sections are reused here rather than re-collected, which is
        what guarantees ``/panel`` never triggers a second round of
        ``nvidia-smi`` subprocess calls.
    extras : dict
        The ``/panel``-only extras collected in the same sampling pass, see
        :func:`sys_stats.sampler._collect_panel_extras`.
    wall_ts : float
        ``time.time()`` recorded when the sample was stored -- the frozen
        contract's ``ts``, for a human running ``curl``; the firmware does
        not read it.
    monotonic_ts : float
        ``time.monotonic()`` recorded when the sample was stored, used below
        to compute ``age`` at response time.

    Returns
    -------
    dict
        The full ``/panel`` v1 payload, ready for ``jsonify``.
    """
    # Computed at RESPONSE time from the monotonic stamp, never from ts: an
    # NTP step or a manual clock adjustment between the sample and this
    # request would make wall-clock arithmetic silently negative or absurd.
    # Deliberately not cached alongside the snapshot either, or it would be
    # frozen at whatever it was when the sample landed.
    age = max(0, int(time.monotonic() - monotonic_ts))

    all_temps = extras["temps"]
    all_fans = extras["fans"]

    # extras only carries "dcgm_gpu" when SYS_STATS_DCGM_URL is configured
    # (see sampler._get_dcgm_url / sampler._collect_panel_extras). Its
    # absence is exactly "build gpu[] from stats["gpu"] the way this always
    # has", not "collected and empty" -- current behaviour stays byte-for-
    # byte unchanged when the variable is unset. get_dcgm_gpus already
    # returns entries in this exact /panel shape (see
    # sys_stats.collectors.get_dcgm_gpus), so no further reshaping through
    # _build_panel_gpu_entry is needed on that path.
    dcgm_gpus = extras.get("dcgm_gpu")
    if dcgm_gpus is not None:
        gpu_source: list[dict[str, Any]] = dcgm_gpus
        gpu_entries = _cap_list(gpu_source, _get_panel_cap(_PANEL_MAX_GPUS_ENV))
    else:
        gpu_source = stats["gpu"]
        gpu_entries = [
            _build_panel_gpu_entry(g)
            for g in _cap_list(gpu_source, _get_panel_cap(_PANEL_MAX_GPUS_ENV))
        ]

    return {
        "v": 1,
        "ts": int(wall_ts),
        "age": age,
        "ready": True,
        # See _panel_hostname: machine identity for a consumer's strcmp,
        # read fresh every response, never merged with _instance_label's
        # display prose. /stats does not carry this key -- see get_stats.
        "host": _panel_hostname(),
        "cpu": {
            "pct": _round1(stats["cpu"]),
            "n": stats["summary"]["cpu"]["cores"],
            "mhz": extras["mhz"],
            # This "load" is the os.getloadavg() 1/5/15 minute triple, NOT a
            # percentage. /stats uses the same key "load" (per-GPU and in
            # its "summary" block) for a load *percentage* instead -- that
            # collision is part of the frozen firmware contract and is kept
            # intentionally, not "fixed" to match /stats.
            "load": [float(v) for v in extras["load"]],
            "per": extras["per_core"],
        },
        "mem": {
            "used": stats["ram"]["used"],
            "total": stats["ram"]["total"],
            "pct": _round1(stats["ram"]["percent"]),
        },
        "swap": extras["swap"],
        "gpu": gpu_entries,
        "gpu_n": len(gpu_source),
        "temps": _cap_list(all_temps, _get_panel_cap(_PANEL_MAX_TEMPS_ENV)),
        "temps_n": len(all_temps),
        "fans": _cap_list(all_fans, _get_panel_cap(_PANEL_MAX_FANS_ENV)),
        "fans_n": len(all_fans),
        "err": extras["err"],
    }


@app.route('/panel', methods=['GET'])
def get_panel():
    """Serve the compact ``/panel`` payload for the ESP32-S3 wall display.

    Frozen contract, schema version 1: fixed keys, fixed nesting depth, no
    process lists, no Ollama data, no cmdlines. Reads the same cached sample
    ``/stats`` serves, plus the ``/panel``-only extras collected in the same
    sampling pass (see :mod:`sys_stats.sampler`), so this route never
    triggers a second round of ``nvidia-smi`` subprocess calls on top of
    what ``/stats`` already causes.

    Returns
    -------
    flask.Response
        200 with the full payload when a snapshot exists. 503 with exactly
        ``{"v": 1, "ready": false}`` otherwise -- a payload of zeroes would
        render as a dead machine on the wall display, which is why absence
        is a 503, never a 200 full of zeroes.
    """
    stats, extras, wall_ts, monotonic_ts = sampler.get_panel_snapshot()
    if stats is None or extras is None:
        return jsonify({"v": 1, "ready": False}), 503

    payload = _build_panel_payload(stats, extras, wall_ts, monotonic_ts)
    # Deep-copied for the same reason /stats' _slice_to_limit is: several
    # values above (extras["swap"], the capped temps/fans entries) are the
    # exact same dict objects the sampler's cache holds, not copies, so a
    # caller mutating the returned payload must never be able to poison it.
    return jsonify(copy.deepcopy(payload))


#: The only fields ``/panel/procs`` copies out of each ranking entry. This is
#: an allowlist on purpose: the collectors attach ``cmdline`` (and may attach
#: more tomorrow) to every process, and a denylist would leak whatever new
#: field nobody remembered to strip. Anything not named here never leaves.
_PROCS_ALLOWED_FIELDS: dict[str, tuple[str, ...]] = {
    "top_cpu": ("pid", "name", "cpu_percent"),
    "top_memory": ("pid", "name", "memory_usage"),
    "top_gpu_processes": ("pid", "name", "memory_used", "gpu_index"),
}

#: Fields kept from each ``ollama_processes["models"]`` entry. That block is
#: Ollama's own ``/api/ps`` answer passed through verbatim, so its shape is
#: not ours; both spellings the wall panel falls back between are kept.
_PROCS_ALLOWED_OLLAMA_FIELDS = ("name", "model", "size_vram", "size")

#: Rankings that get a derived ``args`` field on top of their allowlisted
#: fields. Deliberately just these two: ``top_gpu_processes`` and Ollama's
#: model list are left exactly as documented, unexpanded.
_PROCS_ARGS_RANKING_KEYS = ("top_cpu", "top_memory")

#: Hard cap, in characters, on the ``args`` field ``/panel/procs`` sends. The
#: client is an ESP32-S3 with a small heap that stores these in fixed-size
#: buffers; a `kvm` command line can run to several kilobytes, so this is a
#: plain slice (no ellipsis -- the client appends its own) rather than a
#: soft formatting choice.
_PROCS_ARGS_MAX_LEN = 60


def _extract_args(entry: Any) -> str:
    """Derive the wall panel's ``args`` field from one ranking entry's argv.

    Parameters
    ----------
    entry : Any
        One cached ``top_cpu``/``top_memory`` entry, as built by
        :func:`sys_stats.collectors.get_top_processes_by_cpu` or
        :func:`sys_stats.collectors.get_top_processes_by_memory`. Anything
        that is not a dict, or whose ``argv`` is missing, not a list, or
        holds fewer than two elements (no arguments beyond ``argv[0]``,
        ``AccessDenied``, a zombie, or the collector's own ``"N/A"``
        fallback for an empty ``cmdline``) yields no arguments.

    Returns
    -------
    str
        ``argv[1:]`` joined by single spaces, sliced to at most
        :data:`_PROCS_ARGS_MAX_LEN` characters. ``""`` when there is
        nothing to show.
    """
    if not isinstance(entry, dict):
        return ""
    argv = entry.get("argv")
    if not isinstance(argv, list) or len(argv) < 2:
        return ""
    return " ".join(str(part) for part in argv[1:])[:_PROCS_ARGS_MAX_LEN]


def _pick_fields(entry: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """Copy only the allowlisted ``fields`` out of one process or model entry.

    Parameters
    ----------
    entry : Any
        One entry of a cached ranking or of Ollama's model list. Anything
        that is not a dict (a malformed upstream answer) yields ``{}``.
    fields : tuple of str
        The allowlisted keys to copy.

    Returns
    -------
    dict
        A new dict holding the allowlisted keys present in ``entry``, in the
        order of ``fields``. Absent keys are left out rather than nulled, so
        a consumer's own fallback (e.g. ``name`` then ``model``) still works.
    """
    if not isinstance(entry, dict):
        return {}
    return {key: entry[key] for key in fields if key in entry}


def _build_procs_payload(stats: dict[str, Any], limit: int) -> dict[str, Any]:
    """Reduce a cached ``/stats`` payload to the ``/panel/procs`` allowlist.

    Parameters
    ----------
    stats : dict
        The cached sampler payload, as returned by
        :func:`sys_stats.collectors.collect_stats`.
    limit : int
        Maximum number of entries per ranking, applied exactly as ``/stats``
        applies it (see :func:`_slice_to_limit`).

    Returns
    -------
    dict
        ``top_cpu``, ``top_memory``, ``top_gpu_processes`` and
        ``ollama_processes`` with the same nesting as ``/stats``, but every
        entry rebuilt from :data:`_PROCS_ALLOWED_FIELDS` and
        :data:`_PROCS_ALLOWED_OLLAMA_FIELDS` only, plus an ``args`` string
        added to each ``top_cpu``/``top_memory`` entry (see
        :func:`_extract_args`). Nothing else is copied.
    """
    # Slicing first reuses /stats' exact ranking semantics (including the
    # cross-GPU VRAM reselection), so a given ?limit= means the same rows on
    # both routes. _slice_to_limit deep-copies, so the cache stays untouched.
    sliced = _slice_to_limit(stats, limit)

    payload: dict[str, Any] = {
        key: [_pick_fields(entry, fields) for entry in sliced[key]]
        for key, fields in _PROCS_ALLOWED_FIELDS.items()
    }

    # top_cpu/top_memory get one derived field beyond the allowlist: args,
    # computed from the cached entry's own argv (never itself exposed) so
    # the wall panel can tell apart processes that otherwise all render as
    # the same name (e.g. every VM shows up as "kvm").
    for key in _PROCS_ARGS_RANKING_KEYS:
        for out_entry, src_entry in zip(payload[key], sliced[key], strict=True):
            out_entry["args"] = _extract_args(src_entry)

    # Ollama's answer is external data: a failed or odd /api/ps reply may not
    # be a dict at all, and its model list is not guaranteed to be a list.
    ollama = sliced.get("ollama_processes")
    models = ollama.get("models") if isinstance(ollama, dict) else None
    payload["ollama_processes"] = {
        "models": [
            _pick_fields(model, _PROCS_ALLOWED_OLLAMA_FIELDS)
            for model in (models if isinstance(models, list) else [])
        ]
    }
    return payload


# Registered in BOTH modes, unlike /stats. It exists precisely so a panel-only
# deployment (reachable from an untrusted network) can still show process
# names and figures without ever exposing the command lines /stats carries.
@app.route('/panel/procs', methods=['GET'])
def get_panel_procs():
    """Serve the process and Ollama lists, reduced to names, figures and args.

    Same keys, nesting and ``?limit=`` handling as the matching parts of
    ``/stats``, so a client decoding ``/stats`` needs only a URL change, but
    rebuilt from an explicit allowlist: no full ``cmdline``, environment,
    working directory or user ever appears. ``top_cpu`` and ``top_memory``
    entries do carry ``args`` -- ``argv[1:]`` joined by spaces and truncated
    server side to at most :data:`_PROCS_ARGS_MAX_LEN` characters, ``""``
    when there are none or the underlying ``cmdline`` was unavailable --
    since the wall panel needs some way to tell processes that share a name
    apart (every VM otherwise renders as the same "kvm" row). Reads the same
    cached snapshot ``/stats`` reads, so it adds no process enumeration of
    its own.

    Returns
    -------
    flask.Response
        200 with the reduced payload, or 503 with ``{}`` under exactly the
        conditions ``/stats`` returns one: no snapshot within the cold-start
        wait, or a snapshot missing a ranking key.
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

    # Same reasoning as get_stats: absence of usable data is a retryable 503,
    # never a 200 with empty lists that would read as an idle machine.
    if stats is None or any(key not in stats for key in _RANKING_KEYS):
        logger.error("No complete sampler snapshot for /panel/procs; returning 503")
        return jsonify({}), 503

    return jsonify(_build_procs_payload(stats, limit))


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
