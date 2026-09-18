"""Background sampler that owns all metric collection for the server.

:mod:`sys_stats.collectors` knows how to gather one snapshot of the host;
this module decides *when*. A single daemon thread calls
:func:`sys_stats.collectors.collect_stats` on a fixed interval and stores the
result in a module-level cache guarded by a lock. Flask routes never collect
metrics themselves anymore: they read the latest cached snapshot, which keeps
a request from blocking on ``nvidia-smi`` subprocesses or a full CPU sampling
window.

The single most important rule enforced here is the ``psutil.cpu_percent``
baseline: that call only means something relative to the *previous* call
(``interval=None`` reports the load since the last invocation), so this
module must be the only place in the process that ever calls it. A second
caller with a different interval would silently corrupt the baseline for
both.
"""

import logging
import os
import threading
import time
from typing import Any

import coloredlogs
import psutil

from . import collectors

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

#: Fallback sampling interval, in seconds, used when ``SYS_STATS_SAMPLE_INTERVAL``
#: is unset or invalid.
DEFAULT_SAMPLE_INTERVAL = 2.0

#: Fallback cap on how many per-process entries the sampler collects, used
#: when ``SYS_STATS_TOP_PROCESSES_MAX`` is unset or invalid. Requests for a
#: larger ``?limit=`` than this cannot be satisfied without re-collecting.
DEFAULT_TOP_PROCESSES_MAX = 50

# Cache of the latest sample. Every access (read or write) must go through
# ``_lock``; the two timestamps travel together with the payload so a caller
# can tell how stale a snapshot is without racing the writer.
_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_wall_ts: float | None = None
_monotonic_ts: float | None = None

# Set once the first sample has landed, so ``wait_for_first_snapshot`` can
# block on it instead of polling.
_first_snapshot_event = threading.Event()

# Guards start()/stop so two threads racing to start the sampler cannot spawn
# two loops that fight over the same cpu_percent baseline.
_thread_lock = threading.Lock()
_thread: threading.Thread | None = None

# Signals the loop to exit its interval wait immediately. Also doubles as an
# interruptible sleep: ``_stop_event.wait(timeout=interval)`` sleeps for the
# interval unless told to stop, with no separate polling loop needed.
_stop_event = threading.Event()


def _get_sample_interval() -> float:
    """Read the sampling interval from ``SYS_STATS_SAMPLE_INTERVAL``.

    Returns
    -------
    float
        The configured interval in seconds, or :data:`DEFAULT_SAMPLE_INTERVAL`
        when the environment variable is unset, not a number, or not
        strictly positive.
    """
    raw = os.getenv("SYS_STATS_SAMPLE_INTERVAL")
    if raw is None:
        return DEFAULT_SAMPLE_INTERVAL

    try:
        interval = float(raw)
    except ValueError:
        logger.warning(
            f"Ignoring non-numeric SYS_STATS_SAMPLE_INTERVAL={raw!r}; "
            f"using default {DEFAULT_SAMPLE_INTERVAL}s"
        )
        return DEFAULT_SAMPLE_INTERVAL

    if interval <= 0:
        logger.warning(
            f"Ignoring non-positive SYS_STATS_SAMPLE_INTERVAL={raw!r}; "
            f"using default {DEFAULT_SAMPLE_INTERVAL}s"
        )
        return DEFAULT_SAMPLE_INTERVAL

    return interval


def _get_top_processes_cap() -> int:
    """Read the per-process sampling cap from ``SYS_STATS_TOP_PROCESSES_MAX``.

    Returns
    -------
    int
        The configured cap, or :data:`DEFAULT_TOP_PROCESSES_MAX` when the
        environment variable is unset, not an integer, or not strictly
        positive.
    """
    raw = os.getenv("SYS_STATS_TOP_PROCESSES_MAX")
    if raw is None:
        return DEFAULT_TOP_PROCESSES_MAX

    try:
        cap = int(raw)
    except ValueError:
        logger.warning(
            f"Ignoring non-numeric SYS_STATS_TOP_PROCESSES_MAX={raw!r}; "
            f"using default {DEFAULT_TOP_PROCESSES_MAX}"
        )
        return DEFAULT_TOP_PROCESSES_MAX

    if cap <= 0:
        logger.warning(
            f"Ignoring non-positive SYS_STATS_TOP_PROCESSES_MAX={raw!r}; "
            f"using default {DEFAULT_TOP_PROCESSES_MAX}"
        )
        return DEFAULT_TOP_PROCESSES_MAX

    return cap


def _store_snapshot(stats: dict[str, Any]) -> None:
    """Store a freshly collected sample as the current cache entry.

    Parameters
    ----------
    stats : dict
        The payload returned by :func:`sys_stats.collectors.collect_stats`.
    """
    global _cache, _wall_ts, _monotonic_ts
    with _lock:
        _cache = stats
        _wall_ts = time.time()
        _monotonic_ts = time.monotonic()
    # Only ever needs setting once; subsequent samples leave it set so a late
    # waiter that arrives after the first sample returns immediately.
    _first_snapshot_event.set()


def _sample_once(limit: int) -> None:
    """Collect exactly one sample and store it.

    Exists as its own step so both the loop and tests can trigger a single
    collection deterministically, without going through the loop's timer.
    Any exception raised here is the caller's responsibility to handle: the
    loop wraps this call so a bad collection never kills the thread.

    Parameters
    ----------
    limit : int
        Forwarded to :func:`sys_stats.collectors.collect_stats` as the cap
        on each per-process ranking.
    """
    stats = collectors.collect_stats(limit=limit)
    _store_snapshot(stats)


def _run(interval: float, limit: int) -> None:
    """Sampling loop body, run on the background thread.

    Parameters
    ----------
    interval : float
        Seconds to wait between samples.
    limit : int
        Forwarded to each :func:`_sample_once` call.
    """
    # psutil.cpu_percent(interval=None) reports usage since the *previous*
    # call. The very first call in a process has no previous call to compare
    # against, so it returns a meaningless 0.0 (or an arbitrary bootstrap
    # value depending on platform). Calling it once here, before the loop
    # proper, "primes the pump": this reading is thrown away, and every
    # subsequent call (made inside collect_stats, one interval later) is
    # meaningful relative to it.
    psutil.cpu_percent(interval=None)

    # _stop_event.wait(timeout=interval) sleeps for `interval` seconds unless
    # stop() sets the event first, in which case it returns True immediately
    # and the loop exits. This gives an interruptible sleep without a manual
    # polling loop.
    while not _stop_event.wait(timeout=interval):
        try:
            _sample_once(limit)
        except Exception:
            # A single bad collection (e.g. nvidia-smi wedged, Ollama
            # unreachable in a way collectors.py did not already guard
            # against) must never take the whole sampler down: the cache
            # would freeze forever and /stats would serve stale data with no
            # way to recover short of restarting the process.
            logger.exception(
                "Sampler iteration failed; keeping the previous snapshot and "
                "retrying next interval"
            )


def start() -> None:
    """Start the background sampler thread if it is not already running.

    Idempotent: calling this more than once (e.g. from multiple request
    handlers racing at startup) only ever starts one thread.
    """
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop_event.clear()
        interval = _get_sample_interval()
        limit = _get_top_processes_cap()
        _thread = threading.Thread(
            target=_run, args=(interval, limit), name="sys-stats-sampler", daemon=True
        )
        _thread.start()


def get_snapshot() -> tuple[dict[str, Any] | None, float | None, float | None]:
    """Return the latest cached sample.

    Returns
    -------
    tuple
        ``(stats, wall_ts, monotonic_ts)``. ``stats`` is the dict last
        returned by :func:`sys_stats.collectors.collect_stats`, or ``None``
        if no sample has landed yet. ``wall_ts`` is the ``time.time()`` and
        ``monotonic_ts`` the ``time.monotonic()`` value recorded when that
        sample was stored, both ``None`` alongside a ``None`` ``stats``.
    """
    with _lock:
        return _cache, _wall_ts, _monotonic_ts


def wait_for_first_snapshot(
    timeout: float,
) -> tuple[dict[str, Any] | None, float | None, float | None]:
    """Block until a sample exists, or ``timeout`` seconds have elapsed.

    Used by the ``/stats`` route on a cold start, so the very first request
    still gets real data instead of an error while the sampler is taking its
    first sample.

    Parameters
    ----------
    timeout : float
        Maximum number of seconds to wait.

    Returns
    -------
    tuple
        Same shape as :func:`get_snapshot`. Still ``(None, None, None)`` if
        the timeout elapsed before any sample landed.
    """
    _first_snapshot_event.wait(timeout=timeout)
    return get_snapshot()


def _reset_for_tests() -> None:
    """Stop the sampler thread and clear the cache.

    Not part of the public API. A background thread left running between
    tests would keep sampling on its own schedule after ``monkeypatch``
    reverts the stubs a test installed, touching the real machine and
    corrupting the next test's cache with unrelated data. Tests call this
    before and after using the sampler to start from, and leave, a clean
    slate.
    """
    global _thread, _cache, _wall_ts, _monotonic_ts
    _stop_event.set()
    with _thread_lock:
        if _thread is not None:
            _thread.join(timeout=5)
            _thread = None
    _stop_event.clear()
    with _lock:
        _cache = None
        _wall_ts = None
        _monotonic_ts = None
    _first_snapshot_event.clear()
