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

# Extra collectors gathered in the same sampling pass as ``_cache`` but only
# ever consumed by the ``/panel`` route (temperatures, fans, swap, per-core
# CPU, load average, CPU frequency; see ``_collect_panel_extras``). Kept in a
# separate cache slot -- rather than folded into ``_cache`` -- so ``/stats``,
# which reads ``_cache`` through ``get_snapshot``, is provably unaffected by
# anything collected here for ``/panel``.
_panel_extras: dict[str, Any] | None = None

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


def _store_snapshot(stats: dict[str, Any], panel_extras: dict[str, Any] | None = None) -> None:
    """Store a freshly collected sample as the current cache entry.

    Parameters
    ----------
    stats : dict
        The payload returned by :func:`sys_stats.collectors.collect_stats`.
    panel_extras : dict, optional
        The ``/panel``-only extras collected in the same pass (see
        :func:`_collect_panel_extras`). Defaults to ``None`` so existing
        direct callers that only care about the ``/stats`` cache (tests, in
        particular) do not need to pass one.
    """
    global _cache, _panel_extras, _wall_ts, _monotonic_ts
    with _lock:
        _cache = stats
        _panel_extras = panel_extras
        _wall_ts = time.time()
        _monotonic_ts = time.monotonic()
    # Only ever needs setting once; subsequent samples leave it set so a late
    # waiter that arrives after the first sample returns immediately.
    _first_snapshot_event.set()


def _collect_panel_extras() -> dict[str, Any]:
    """Collect the metrics unique to the ``/panel`` payload, on top of ``/stats``.

    Runs in the same sampling pass as :func:`sys_stats.collectors.collect_stats`
    (see :func:`_sample_once`) so ``/panel`` never triggers a second round of
    ``nvidia-smi`` subprocess calls: its GPU section is built by the route
    from the very same ``collect_stats()`` output ``/stats`` serves, not
    collected again here.

    Each collector is wrapped individually so one failing sensor degrades
    only its own field, with a safe default substituted and a short tag
    appended to ``err``, instead of losing every panel field the way an
    uncaught exception in :func:`_sample_once` would drop the whole
    iteration. This applies in particular to the two independent fan
    sources below: hwmon (:func:`sys_stats.collectors.get_fans`) and IPMI
    (:func:`sys_stats.collectors.get_ipmi_fans`) are collected and guarded
    separately, so one raising never costs the other its data.

    ``fans`` is the union of both sources: concatenated, THEN sorted by
    ``"n"``. The two are disjoint in practice (hwmon names look like
    ``nct6775/fan1``, IPMI names look like ``FAN1``), so no deduplication
    or priority rule is needed -- a Proxmox hypervisor host with zero hwmon
    fans and five IPMI ones, and a desktop with the reverse, both fall out
    of the same code path.

    Returns
    -------
    dict
        Keys ``temps``, ``fans``, ``swap``, ``per_core``, ``load``, ``mhz``
        and ``err``. ``err`` lists the short tags of whichever collectors
        raised, in call order; empty when every collector succeeded.
    """
    err: list[str] = []

    try:
        temps = collectors.get_temperatures()
    except Exception:
        logger.exception("Panel: failed to collect temperatures")
        temps = []
        err.append("temps")

    try:
        hwmon_fans = collectors.get_fans()
    except Exception:
        logger.exception("Panel: failed to collect hwmon fan speeds")
        hwmon_fans = []
        err.append("fans_hwmon")

    try:
        ipmi_fans = collectors.get_ipmi_fans()
    except Exception:
        logger.exception("Panel: failed to collect IPMI fan speeds")
        ipmi_fans = []
        err.append("fans_ipmi")

    # Concatenate first, sort second -- never the reverse. Positional
    # stability on the wall display is part of the frozen /panel contract
    # (see get_temperatures/get_fans/get_ipmi_fans), and sorting only the
    # individual sources before concatenating would not guarantee the
    # union itself comes out ordered.
    fans = sorted(hwmon_fans + ipmi_fans, key=lambda e: e["n"])

    try:
        swap = collectors.get_swap()
    except Exception:
        logger.exception("Panel: failed to collect swap usage")
        swap = {"used": 0, "total": 0, "pct": 0.0}
        err.append("swap")

    try:
        per_core = collectors.get_per_core_cpu()
    except Exception:
        logger.exception("Panel: failed to collect per-core CPU usage")
        per_core = []
        err.append("per_core")

    try:
        load = collectors.get_load_average()
    except Exception:
        logger.exception("Panel: failed to collect load average")
        load = [0.0, 0.0, 0.0]
        err.append("loadavg")

    try:
        mhz = collectors.get_cpu_frequency_mhz()
    except Exception:
        logger.exception("Panel: failed to collect CPU frequency")
        mhz = 0
        err.append("cpu_freq")

    return {
        "temps": temps,
        "fans": fans,
        "swap": swap,
        "per_core": per_core,
        "load": load,
        "mhz": mhz,
        "err": err,
    }


def _sample_once(limit: int) -> None:
    """Collect exactly one sample and store it.

    Exists as its own step so both the loop and tests can trigger a single
    collection deterministically, without going through the loop's timer.
    Any exception raised by :func:`sys_stats.collectors.collect_stats` here
    is the caller's responsibility to handle: the loop wraps this call so a
    bad collection never kills the thread. Also gathers the ``/panel``-only
    extras (:func:`_collect_panel_extras`) in the same pass, which never
    raises on its own -- each of its collectors is individually guarded --
    so it never turns a healthy ``/stats`` collection into a dropped
    iteration.

    Parameters
    ----------
    limit : int
        Forwarded to :func:`sys_stats.collectors.collect_stats` as the cap
        on each per-process ranking.
    """
    stats = collectors.collect_stats(limit=limit)
    panel_extras = _collect_panel_extras()
    _store_snapshot(stats, panel_extras)


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
    # value depending on platform). The loop's first iteration "primes the
    # pump" by making (and discarding) that first call instead of collecting
    # a real sample; every call after that (made inside collect_stats, one
    # interval later) is meaningful relative to it. The priming call runs
    # inside the same try/except as a real sample below, so a failure there
    # is logged and retried on the next interval instead of silently killing
    # the thread with no retry and an empty cache forever.
    #
    # psutil keeps a SEPARATE internal baseline for the percpu variant
    # (``cpu_percent(percpu=True)``, used by collectors.get_per_core_cpu)
    # from the one it keeps for the plain call above; priming one does
    # nothing for the other. Both calls are primed together here, in the
    # same guarded step, so neither baseline is ever read before it has a
    # previous call to compare against.
    #
    # THIRD, and separate again: collectors.get_top_processes_by_cpu() reads
    # ``cpu_percent`` off ``psutil.process_iter([..., "cpu_percent", ...])``,
    # which keeps its baseline per-Process object (keyed by pid), entirely
    # independent of the two module-level baselines above -- priming those
    # two does nothing for this one, and this one is not redundant with
    # them. Left unprimed, every process reports ``cpu_percent: 0.0`` on the
    # first real sample, so ``top_cpu`` sorts an all-zero column into an
    # arbitrary order that looks plausible and self-corrects one interval
    # later, which is exactly why it is easy to miss. Priming sweeps
    # ``process_iter`` once here and discards the result -- establishing the
    # baseline is the only goal, not collecting data -- tolerating processes
    # that vanish mid-sweep the same way the collector itself does.
    primed = False

    # The interval wait sits at the BOTTOM of the loop, so priming runs
    # immediately on entry and the first real sample lands exactly one
    # interval later. With the wait at the top instead, the sequence was
    # wait-prime-wait-sample and every cold start cost TWO intervals, which
    # at SYS_STATS_SAMPLE_INTERVAL=12 meant 24 seconds of 503s from /panel
    # and a blocked /stats. The wait BETWEEN priming and the first sample is
    # not dead time and must stay: that elapsed interval is precisely what
    # gives the cpu_percent baseline something to measure against.
    #
    # _stop_event.wait(timeout=interval) sleeps for `interval` seconds unless
    # stop() sets the event first, in which case it returns True immediately
    # and the loop exits. This gives an interruptible sleep without a manual
    # polling loop.
    while not _stop_event.is_set():
        try:
            if not primed:
                psutil.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None, percpu=True)
                for proc in psutil.process_iter(["cpu_percent"]):
                    try:
                        proc.info["cpu_percent"]
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                primed = True
            else:
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

        # Interruptible sleep, at the bottom of the loop on purpose (see
        # above). A failed priming attempt leaves ``primed`` False, so the
        # next iteration retries it rather than sampling against a baseline
        # that was never established.
        if _stop_event.wait(timeout=interval):
            break


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


def get_panel_snapshot() -> tuple[
    dict[str, Any] | None, dict[str, Any] | None, float | None, float | None
]:
    """Return the latest cached sample together with its ``/panel``-only extras.

    Reads the exact same cache entry :func:`get_snapshot` reads, plus the
    additional collectors (temperatures, fans, swap, per-core CPU, load
    average, CPU frequency) gathered in the same sampling pass exclusively
    for the ``/panel`` route. Kept as a separate accessor, rather than
    folding the extras into :func:`get_snapshot`'s return value, so
    ``/stats`` -- which calls :func:`get_snapshot` -- is provably unaffected
    by anything this module now collects for ``/panel``.

    Returns
    -------
    tuple
        ``(stats, panel_extras, wall_ts, monotonic_ts)``, all ``None`` when
        no sample has landed yet.
    """
    with _lock:
        return _cache, _panel_extras, _wall_ts, _monotonic_ts


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
    global _thread, _cache, _panel_extras, _wall_ts, _monotonic_ts
    _stop_event.set()
    with _thread_lock:
        if _thread is not None:
            _thread.join(timeout=5)
            _thread = None
    _stop_event.clear()
    with _lock:
        _cache = None
        _panel_extras = None
        _wall_ts = None
        _monotonic_ts = None
    _first_snapshot_event.clear()
