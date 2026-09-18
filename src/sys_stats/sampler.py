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

#: Fallback polling interval, in seconds, for the two ``ipmi/``-prefixed
#: collectors (``get_ipmi_temperatures``, ``get_ipmi_fans``), used when
#: ``SYS_STATS_IPMI_INTERVAL`` is unset or invalid. Deliberately much slower
#: than :data:`DEFAULT_SAMPLE_INTERVAL`: fan speed and chassis temperature
#: move on timescales of tens of seconds, not the 2s default sampling
#: cadence, and each poll is an `ipmitool` round trip to the host's BMC.
DEFAULT_IPMI_INTERVAL = 30.0

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

# Last IPMI poll's raw results, reused by ``_collect_panel_extras`` between
# polls so ``temps``/``fans`` keep their IPMI half populated even on a
# sampling pass that does not talk to the BMC (see
# ``_ipmi_last_poll_monotonic`` below). Only ever touched by the sampler
# thread's single-threaded loop (or a test calling ``_collect_panel_extras``
# directly), the same way the rest of this module's collection state is --
# no lock needed for the same reason ``_run``'s psutil-priming state needs
# none.
_ipmi_temps_cache: list[dict[str, Any]] = []
_ipmi_fans_cache: list[dict[str, Any]] = []

# time.monotonic() of the last IPMI poll, or None before the first one. None
# is also how the very first pass is told apart from every later one: it is
# the only case where ``_collect_panel_extras`` polls regardless of
# ``SYS_STATS_IPMI_INTERVAL`` having elapsed.
_ipmi_last_poll_monotonic: float | None = None

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


def _get_ipmi_interval() -> float:
    """Read the IPMI polling interval from ``SYS_STATS_IPMI_INTERVAL``.

    Same validation shape as :func:`_get_sample_interval`: non-numeric and
    non-positive values both fall back to the default with a logged
    warning, rather than being silently coerced or left to raise later.

    Returns
    -------
    float
        The configured interval in seconds, or :data:`DEFAULT_IPMI_INTERVAL`
        when the environment variable is unset, not a number, or not
        strictly positive.
    """
    raw = os.getenv("SYS_STATS_IPMI_INTERVAL")
    if raw is None:
        return DEFAULT_IPMI_INTERVAL

    try:
        interval = float(raw)
    except ValueError:
        logger.warning(
            f"Ignoring non-numeric SYS_STATS_IPMI_INTERVAL={raw!r}; "
            f"using default {DEFAULT_IPMI_INTERVAL}s"
        )
        return DEFAULT_IPMI_INTERVAL

    if interval <= 0:
        logger.warning(
            f"Ignoring non-positive SYS_STATS_IPMI_INTERVAL={raw!r}; "
            f"using default {DEFAULT_IPMI_INTERVAL}s"
        )
        return DEFAULT_IPMI_INTERVAL

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


def _get_dcgm_url() -> str | None:
    """Read the configured dcgm-exporter endpoint from ``SYS_STATS_DCGM_URL``.

    Unset (default): ``/panel``'s ``gpu[]`` keeps being built from the same
    GPUtil/nvidia-smi path ``/stats`` uses (see
    :func:`sys_stats.collectors.collect_stats`) -- current behaviour,
    unchanged. Set: ``/panel``'s ``gpu[]`` is built from
    :func:`sys_stats.collectors.get_dcgm_gpus` instead (see
    :func:`_collect_panel_extras`). ``/stats`` itself never reads this
    variable and is unaffected either way.

    Read fresh on every call rather than cached at import time, like
    :func:`sys_stats.server._instance_label`, so pointing this at a
    different endpoint takes effect on the next sampling pass without a
    restart.

    Returns
    -------
    str or None
        The configured URL with surrounding whitespace stripped, or
        ``None`` when the variable is unset or blank.
    """
    raw = os.getenv("SYS_STATS_DCGM_URL")
    if raw is None:
        return None
    url = raw.strip()
    return url or None


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
    iteration. This applies in particular to the independent hwmon/IPMI
    source pairs below: temperatures
    (:func:`sys_stats.collectors.get_temperatures` /
    :func:`sys_stats.collectors.get_ipmi_temperatures`) and fans
    (:func:`sys_stats.collectors.get_fans` /
    :func:`sys_stats.collectors.get_ipmi_fans`) are each collected and
    guarded separately, so one source raising never costs its sibling its
    data.

    The hwmon collectors run every pass, on the same
    ``SYS_STATS_SAMPLE_INTERVAL`` cadence as everything else. The two
    ``ipmi/``-prefixed collectors instead run on their own, slower
    ``SYS_STATS_IPMI_INTERVAL`` cadence (see :func:`_get_ipmi_interval`,
    default :data:`DEFAULT_IPMI_INTERVAL`): each is an ``ipmitool`` round
    trip to the host's BMC, and fan speed / chassis temperature move on a
    timescale of tens of seconds, not ``SYS_STATS_SAMPLE_INTERVAL``'s
    default 2s. Between IPMI polls, this function reuses the last polled
    result (``_ipmi_temps_cache`` / ``_ipmi_fans_cache``) instead of
    dropping it, so ``temps``/``fans`` never lose their IPMI half on an
    intermediate pass -- the wall display renders both lists positionally,
    so entries disappearing and reappearing would shift every row below
    them. The very first pass always polls immediately, rather than waiting
    a full ``SYS_STATS_IPMI_INTERVAL`` before IPMI data appears at all.

    ``temps`` and ``fans`` are each the union of their two sources:
    concatenated, THEN sorted by ``"n"``. The two are disjoint in practice
    (hwmon names look like ``nct6775/fan1`` / ``k10temp/Tctl``, IPMI names
    are prefixed ``ipmi/`` by :func:`sys_stats.collectors.get_ipmi_fans` /
    :func:`sys_stats.collectors.get_ipmi_temperatures`, e.g. ``ipmi/FAN1``
    / ``ipmi/CPU1 Temp``), so no deduplication or priority rule is needed
    -- a Proxmox hypervisor host with zero hwmon sensors and several IPMI
    ones, and a desktop with the reverse, both fall out of the same code
    path. Some BMCs report GPU temperatures among their sensors; those
    stay in ``temps`` under their prefixed BMC name rather than being
    folded into ``gpu[]``, where a temperature-only entry would read as an
    idle card instead of missing data.

    When ``SYS_STATS_DCGM_URL`` is configured (see :func:`_get_dcgm_url`),
    also scrapes a dcgm-exporter Prometheus endpoint in this same pass via
    :func:`sys_stats.collectors.get_dcgm_gpus` -- no second thread, no
    second pass -- and stores the result under the ``dcgm_gpu`` key, for
    ``/panel``'s route to use in place of the ``stats["gpu"]``/nvidia-smi
    path (see :func:`sys_stats.server._build_panel_payload`). Unlike every
    other field here, this key is only present in the returned dict at all
    when the URL is configured: its absence is exactly "unchanged from
    before this feature existed", not "collected and empty". Unlike this
    function's other collectors, :func:`sys_stats.collectors.get_dcgm_gpus`
    is documented to never raise -- it degrades to ``[]`` internally, the
    same contract as the nvidia-smi collectors -- so an empty result while
    configured is treated as the failure signal itself (this deployment's
    dcgm-exporter always monitors its host's passed-through GPUs, so a
    successful scrape reporting zero of them is not a case this code
    distinguishes from a failed one) and tags ``"gpu"`` into ``err``.

    Returns
    -------
    dict
        Keys ``temps``, ``fans``, ``swap``, ``per_core``, ``load``, ``mhz``,
        ``err`` and, only when ``SYS_STATS_DCGM_URL`` is configured,
        ``dcgm_gpu``. ``err`` lists the short tags of whichever collectors
        raised (or, for ``"gpu"``, failed to scrape), in call order; empty
        when every collector succeeded.
    """
    err: list[str] = []

    global _ipmi_temps_cache, _ipmi_fans_cache, _ipmi_last_poll_monotonic

    try:
        hwmon_temps = collectors.get_temperatures()
    except Exception:
        logger.exception("Panel: failed to collect hwmon temperatures")
        hwmon_temps = []
        err.append("temps_hwmon")

    # ipmi_due governs both IPMI collectors below (temps and fans), so the
    # two are always polled together on the SYS_STATS_IPMI_INTERVAL cadence,
    # separate from the hwmon collectors above and below, which run every
    # pass at SYS_STATS_SAMPLE_INTERVAL like before this feature existed.
    # None means "never polled yet"; that -- not a fresh but short-lived
    # interval -- is what makes the very first pass poll immediately instead
    # of waiting a full interval.
    now = time.monotonic()
    ipmi_interval = _get_ipmi_interval()
    ipmi_due = (
        _ipmi_last_poll_monotonic is None or (now - _ipmi_last_poll_monotonic) >= ipmi_interval
    )

    if ipmi_due:
        try:
            ipmi_temps = collectors.get_ipmi_temperatures()
        except Exception:
            logger.exception("Panel: failed to collect IPMI temperatures")
            err.append("temps_ipmi")
            # Keep serving the last good value rather than clearing the
            # cache: a wall panel losing its IPMI temperature readings
            # entirely is worse than showing readings up to
            # SYS_STATS_IPMI_INTERVAL seconds old, and the "temps_ipmi" tag
            # above is what signals the failure to a consumer, not an empty
            # list. Design judgement, not a measured outcome.
            ipmi_temps = _ipmi_temps_cache
        else:
            _ipmi_temps_cache = ipmi_temps
    else:
        ipmi_temps = _ipmi_temps_cache

    # Concatenate first, sort second -- never the reverse. Positional
    # stability on the wall display is part of the frozen /panel contract
    # (see get_temperatures/get_ipmi_temperatures/get_fans/get_ipmi_fans),
    # and sorting only the individual sources before concatenating would
    # not guarantee the union itself comes out ordered.
    temps = sorted(hwmon_temps + ipmi_temps, key=lambda e: e["n"])

    try:
        hwmon_fans = collectors.get_fans()
    except Exception:
        logger.exception("Panel: failed to collect hwmon fan speeds")
        hwmon_fans = []
        err.append("fans_hwmon")

    if ipmi_due:
        try:
            ipmi_fans = collectors.get_ipmi_fans()
        except Exception:
            logger.exception("Panel: failed to collect IPMI fan speeds")
            err.append("fans_ipmi")
            # Same cache-preserving choice as the IPMI temperatures branch
            # above: keep the last good reading, let "fans_ipmi" carry the
            # failure signal.
            ipmi_fans = _ipmi_fans_cache
        else:
            _ipmi_fans_cache = ipmi_fans
        # Advanced on both success and failure, once per pass that actually
        # polled. A BMC that is failing or timing out is, if anything, the
        # case to back off from hardest -- retrying it every sampling pass
        # instead of waiting a full interval would defeat the point of
        # throttling these two collectors. Design judgement, not a measured
        # outcome.
        _ipmi_last_poll_monotonic = now
    else:
        ipmi_fans = _ipmi_fans_cache

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

    extras: dict[str, Any] = {
        "temps": temps,
        "fans": fans,
        "swap": swap,
        "per_core": per_core,
        "load": load,
        "mhz": mhz,
        "err": err,
    }

    dcgm_url = _get_dcgm_url()
    if dcgm_url:
        try:
            dcgm_gpus = collectors.get_dcgm_gpus(dcgm_url)
        except Exception:
            # collectors.get_dcgm_gpus is documented to never raise (same
            # contract as the nvidia-smi collectors); this is a defensive
            # backstop only, matching every other collector call above.
            logger.exception("Panel: failed to collect DCGM GPU metrics")
            dcgm_gpus = []

        if not dcgm_gpus:
            err.append("gpu")
        extras["dcgm_gpu"] = dcgm_gpus

    return extras


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
    # against, so its return value is meaningless -- psutil documents this
    # and tests/test_sampler.py pins the priming order. (An earlier version
    # of this comment added "or an arbitrary bootstrap value depending on
    # platform"; that was never observed here, so it is gone.) The loop's
    # first iteration "primes the
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

    Also clears the IPMI poll cache and its timestamp (see
    :func:`_collect_panel_extras`): leaving ``_ipmi_last_poll_monotonic``
    set between tests would make the next test's first call to
    ``_collect_panel_extras`` see the interval as already having elapsed
    (or not), instead of the "never polled yet" state each test expects to
    start from.
    """
    global _thread, _cache, _panel_extras, _wall_ts, _monotonic_ts
    global _ipmi_temps_cache, _ipmi_fans_cache, _ipmi_last_poll_monotonic
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
    _ipmi_temps_cache = []
    _ipmi_fans_cache = []
    _ipmi_last_poll_monotonic = None
    _first_snapshot_event.clear()
