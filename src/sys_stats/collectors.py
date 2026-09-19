"""Metric-collection logic shared by the ``/stats`` route and any other caller.

Every function here talks directly to the host machine (psutil, GPUtil,
``nvidia-smi`` via subprocess, or the Ollama HTTP API) and returns plain
Python data structures, with no dependency on a Flask application or request
context. :func:`collect_stats` is the single entry point: it merges all of
them into the exact dict the ``/stats`` endpoint has always returned.
"""

import datetime
import logging
import os
import re
import subprocess
import time
from typing import Any
from urllib.parse import urljoin

import coloredlogs
import GPUtil
import psutil
import requests

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

def get_top_processes_by_cpu(limit: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve the top processes by CPU usage.
    """
    processes = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "cmdline"]):
        try:
            # ``process_iter`` does not raise for attributes it cannot read: it
            # fills them with None. That is the common case for root-owned
            # processes when the server cannot see them at all (macOS, or a
            # container started without `pid: host`), and sorting None against
            # a float would blow up the whole endpoint. Note that privilege is
            # not the missing ingredient in the container case: /proc is
            # world-readable, so the host PID namespace alone is enough.
            processes.append({
                "pid": p.info["pid"],
                "name": p.info["name"],
                "cpu_percent": p.info["cpu_percent"] or 0.0,
                "cmdline": " ".join(p.info["cmdline"]) if p.info["cmdline"] else "N/A"
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    processes.sort(key=lambda x: x["cpu_percent"], reverse=True)
    logger.debug(f"Top CPU processes: {processes[:limit]}")
    return processes[:limit]

def get_top_processes_by_memory(limit: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve the top processes by memory usage.
    """
    processes = []
    for p in psutil.process_iter(["pid", "name", "memory_percent", "memory_info", "cmdline"]):
        try:
            # See get_top_processes_by_cpu: unreadable attributes come back as
            # None, including the whole memory_info namedtuple.
            memory_info = p.info["memory_info"]
            processes.append({
                "pid": p.info["pid"],
                "name": p.info["name"],
                "memory_usage": memory_info.rss if memory_info else 0,
                "memory_percent": p.info["memory_percent"] or 0.0,
                "cmdline": " ".join(p.info["cmdline"]) if p.info["cmdline"] else "N/A"
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    processes.sort(key=lambda x: x["memory_percent"], reverse=True)
    logger.debug(f"Top Memory processes: {processes[:limit]}")
    return processes[:limit]

# The GPU each compute app runs on is only reachable through ``gpu_uuid``:
# nvidia-smi has no ``--query-compute-apps=index``. Older drivers do not know the
# field at all and reject the whole query, hence the legacy fallback below.
_COMPUTE_APPS_QUERY = '--query-compute-apps=gpu_uuid,pid,process_name,used_memory'
_LEGACY_COMPUTE_APPS_QUERY = '--query-compute-apps=pid,process_name,used_memory'

# ``nvidia-smi`` talks to the driver through a local ioctl, not over a network
# or a serial BMC channel like ``ipmitool``, so a healthy call is expected to
# be fast -- nobody here timed it, so 3 seconds is a judgement about how long
# a stuck one may block, not a multiple of a measured baseline. It bounds a
# wedged-driver hang tightly: this collector runs
# inside the sampler's background loop (see sys_stats.sampler), and
# _query_compute_apps can attempt this call twice (UUID-aware query, then the
# legacy fallback), so the timeout here directly caps how long one sampling
# pass can be stuck on GPU process attribution alone.
_NVIDIA_SMI_TIMEOUT = 3.0


def _query_compute_apps(timeout: float = _NVIDIA_SMI_TIMEOUT) -> str | None:
    """Ask nvidia-smi for the running compute apps, newest query shape first.

    The UUID-aware query is tried once; if the driver rejects it the legacy
    three-field query is tried as well, so a card attribution failure never
    costs the caller the whole process list.

    Parameters
    ----------
    timeout : float, optional
        Seconds to wait for each ``nvidia-smi`` invocation before giving up
        on it and moving to the next query (or giving up entirely).

    Returns
    -------
    str or None
        Raw CSV output of whichever query succeeded, or ``None`` when both
        invocations failed.
    """
    last_error = None
    for query in (_COMPUTE_APPS_QUERY, _LEGACY_COMPUTE_APPS_QUERY):
        try:
            result = subprocess.run(
                ['nvidia-smi', query, '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True,
                # A wedged driver hangs nvidia-smi indefinitely instead of
                # failing fast. This collector runs inside the sampler's
                # background thread, so an unbounded call would freeze the
                # cached snapshot for every consumer while /stats and /panel
                # kept serving stale numbers that look perfectly fresh. See
                # _NVIDIA_SMI_TIMEOUT for why this value.
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Timed out waiting for nvidia-smi ({query}): {e}")
            last_error = e
            continue
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            last_error = e
            continue
        return result.stdout

    # ``stderr`` is only set on ``CalledProcessError`` (and possibly
    # ``TimeoutExpired`` when the pipes were captured before the timeout
    # fired); ``FileNotFoundError`` (binary missing entirely) has none, so
    # guard both the attribute and the None case before stripping.
    stderr = (getattr(last_error, "stderr", None) or "").strip() if last_error is not None else ""
    logger.error(f"Error fetching GPU processes: {stderr or last_error}")
    return None


def _parse_compute_app_line(
    line: str, uuid_to_index: dict[str, int] | None
) -> dict[str, Any]:
    """Turn one nvidia-smi compute-app CSV row into a process entry.

    Both query shapes are handled here so the psutil command-line lookup is
    written once: four fields means the UUID-aware query answered, three fields
    means the legacy fallback did and the GPU stays unknown.

    Parameters
    ----------
    line : str
        A single non-empty CSV row, without the trailing newline.
    uuid_to_index : dict of str to int, or None
        Mapping from GPU UUID to the GPU index reported in the ``gpu`` section
        of ``/stats``. ``None`` (or a UUID missing from it) leaves
        ``gpu_index`` unresolved.

    Returns
    -------
    dict
        Keys ``pid``, ``name``, ``memory_used`` (bytes), ``cmdline``,
        ``gpu_uuid`` and ``gpu_index``.

    Raises
    ------
    ValueError
        If the row does not carry three or four fields, or if the numeric
        fields are not numbers. Callers treat this as "skip that line".
    """
    fields = line.split(', ')
    if len(fields) == 4:
        gpu_uuid, pid_str, process_name, used_memory_str = fields
    elif len(fields) == 3:
        gpu_uuid = None
        pid_str, process_name, used_memory_str = fields
    else:
        raise ValueError(f"expected 3 or 4 fields, got {len(fields)}")

    process_name = process_name.split('/')[-1]
    pid = int(pid_str)

    # Try to get command line using psutil
    try:
        p = psutil.Process(pid)
        cmdline = " ".join(p.cmdline()) if p.cmdline() else "N/A"
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        cmdline = "N/A"

    gpu_index = uuid_to_index.get(gpu_uuid) if (uuid_to_index and gpu_uuid) else None

    return {
        "pid": pid,
        "name": process_name,
        "memory_used": int(used_memory_str)*1024*1024, # Convert MiB to bytes
        "cmdline": cmdline,
        "gpu_uuid": gpu_uuid,
        "gpu_index": gpu_index,
    }


def get_gpu_processes(
    limit: int = 5, uuid_to_index: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Retrieve the top GPU compute processes reported by nvidia-smi.

    Parameters
    ----------
    limit : int, optional
        Maximum number of processes to return, applied after sorting.
    uuid_to_index : dict of str to int, or None, optional
        Mapping from GPU UUID to GPU index, normally built by
        :func:`collect_stats` from the GPUtil device list. Without it every
        entry reports ``gpu_index`` as ``None``, which keeps the function
        testable on its own.

    Returns
    -------
    list of dict
        The ``limit`` processes with the highest VRAM usage across every
        card, ordered for display by GPU index then by descending VRAM usage
        so the processes of a single card stay contiguous. Entries whose card
        could not be resolved are pushed to the end. Empty when nvidia-smi is
        missing or fails.
    """
    stdout = _query_compute_apps()
    if stdout is None:
        return []

    lines = stdout.strip().split('\n')
    gpu_processes = []
    for line in lines:
        if not line.strip():
            continue
        try:
            gpu_processes.append(_parse_compute_app_line(line, uuid_to_index))
        except ValueError:
            logger.warning(f"Skipping malformed GPU process line: '{line}'")
            continue

    # Selection happens first, purely by VRAM usage, so ``limit`` picks the
    # heaviest processes regardless of which card they run on. Grouping by
    # GPU is then applied only to the surviving slice, for display: sorting
    # by card index before truncating would instead keep whichever cards
    # happen to sort first and silently drop heavier processes on
    # higher-numbered cards.
    gpu_processes.sort(key=lambda x: -x["memory_used"])
    top_processes = gpu_processes[:limit]

    # ``gpu_index`` may be None, which does not compare with int: sort on a
    # tuple whose first element pushes the unresolved rows last, deterministically.
    top_processes.sort(
        key=lambda x: (x["gpu_index"] is None, x["gpu_index"] or 0, -x["memory_used"])
    )
    logger.debug(f"Top GPU processes: {top_processes}")
    return top_processes

def get_gpu_fan_and_power(timeout: float = _NVIDIA_SMI_TIMEOUT) -> dict[int, dict[str, float]]:
    """
    Retrieve fan speed (%) and power draw (W) for each GPU via nvidia-smi.
    Returns a dict keyed by GPU index: {"fan_speed": float, "power_draw": float}.

    Parameters
    ----------
    timeout : float, optional
        Seconds to wait for ``nvidia-smi`` before giving up. See
        :data:`_NVIDIA_SMI_TIMEOUT` for why this value.
    """
    data = {}
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,fan.speed,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True,
            # See _query_compute_apps: a wedged driver hangs this call
            # indefinitely otherwise, freezing the sampler's cached snapshot
            # while /stats and /panel keep serving stale numbers that look
            # perfectly fresh.
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error fetching GPU fan/power: {e.stderr.strip()}")
        return {}
    except FileNotFoundError as e:
        # The ``nvidia-smi`` binary is not installed at all (no NVIDIA driver
        # on this host), as opposed to the binary existing but failing.
        logger.error(f"Error fetching GPU fan/power: {e}")
        return {}
    except subprocess.TimeoutExpired as e:
        logger.warning(f"Timed out waiting for nvidia-smi (fan/power query): {e}")
        return {}

    lines = result.stdout.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue
        try:
            idx_str, fan_str, power_str = line.split(', ')
            idx = int(idx_str)
            fan_speed = float(fan_str)       # e.g. 25 means 25%
            power_draw = float(power_str)    # e.g. 30 means 30 W
            data[idx] = {
                "fan_speed": fan_speed,
                "power_draw": power_draw
            }
        except ValueError:
            logger.warning(f"Skipping malformed GPU fan/power line: '{line}'")
            continue

    return data

def get_ollama_process():
    """Retrieve the Ollama process information."""

    ollama_data = {"models": []}

    if OLLAMA_API_URL is None:
        return ollama_data

    try:
        response = requests.get(urljoin(OLLAMA_API_URL, "/api/ps"), timeout=5)
        response.raise_for_status()
        ollama_data = response.json()

    except requests.RequestException as e:
        print(f"Error fetching Ollama data: {e}")

    return ollama_data

# --- DCGM (NVIDIA Data Center GPU Manager) exporter scraping ---------------
#
# Some deployments run this server on a host with no NVIDIA driver of its
# own -- e.g. a Proxmox hypervisor whose GPUs are PCI-passed-through to a
# Kubernetes VM, leaving the host with neither /dev/nvidia* nor nvidia-smi.
# On that host, GPU metrics instead come from dcgm-exporter running inside
# the cluster on the VM the GPUs were passed into, scraped as a Prometheus
# ``/metrics`` endpoint (curled directly against one such endpoint during
# development: http://10.50.0.106:30940/metrics, 200 OK in ~21ms for ~10KB
# of body).
#
# Only the metric families below are read, all confirmed present on that
# endpoint's own ``# HELP`` lines: GPU utilization (%), framebuffer used/free
# (MiB), GPU temperature (C), power draw (W) and fan speed (%).

#: Metric families read from the DCGM exporter body; anything else in the
#: response is ignored.
_DCGM_METRIC_FIELDS = frozenset({
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_FB_FREE",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_FAN_SPEED",
})

# Prometheus text exposition format, one sample per line:
#   METRIC_NAME{label="value",label2="value2"} 123.45
# Blank lines and comment lines (# HELP / # TYPE) are skipped by the caller
# before either of these ever runs against a line.
_DCGM_METRIC_LINE_RE = re.compile(
    r'^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)\{(?P<labels>[^}]*)\}\s+(?P<value>\S+)\s*$'
)
_DCGM_LABEL_RE = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:[^"\\]|\\.)*)"')

# requests' `timeout=` bounds each individual socket send/recv, not the call
# as a whole: a server that returns one byte every 2.9 seconds never trips a
# 3-second read timeout and can hold the connection open indefinitely. The
# tuple form applies the first value to the connect phase and the second to
# each read phase, individually and repeatedly.
_DCGM_CONNECT_TIMEOUT = 2.0
_DCGM_READ_TIMEOUT = 3.0

# Hard wall-clock budget for one whole scrape attempt, enforced independently
# of the per-socket timeouts above -- see _fetch_dcgm_text -- so a connection
# that keeps dribbling bytes without ever going idle long enough to trip
# _DCGM_READ_TIMEOUT still gets abandoned. The measured healthy case (curl
# above) answers in ~21ms for the whole body, so this leaves ample headroom
# while still bounding how long this collector can hold up the sampler pass
# it runs inside (see sys_stats.sampler._collect_panel_extras).
_DCGM_HARD_DEADLINE = 5.0

# requests' timeout does not cover DNS resolution: socket.getaddrinfo runs
# before either half of the timeout tuple above starts counting, so an
# unreachable or slow resolver can stall this call regardless of
# _DCGM_CONNECT_TIMEOUT / _DCGM_READ_TIMEOUT. This is documented upstream in
# requests' own timeout reference, not something measured on this network.
# The mitigation is deployment-side: SYS_STATS_DCGM_URL is expected to carry
# a literal IP (as in the deployment this collector targets), and nothing
# here resolves or pre-resolves a hostname on the caller's behalf.

# Circuit breaker tuning. Measured on the target network: a NodePort with no
# backing Service silently drops packets on some nodes rather than refusing
# the connection, and the host runs pve-firewall with a DROP policy on
# unauthorised ports -- both hang instead of failing fast, and a wedged
# scrape retried on every sampling pass (default SYS_STATS_SAMPLE_INTERVAL is
# 2s) would make the sampler miss its own cadence, letting /panel's ``age``
# climb while the underlying cause is a missing firewall rule, not the
# server. 3 consecutive failures is chosen to tolerate a transient blip or
# two before backing off; backoff then starts at 5s and doubles up to a 60s
# cap, so a dead endpoint is retried occasionally rather than hammered, and a
# fixed endpoint recovers within a bounded window rather than staying open
# forever.
_DCGM_BREAKER_THRESHOLD = 3
_DCGM_BREAKER_BASE_BACKOFF = 5.0
_DCGM_BREAKER_MAX_BACKOFF = 60.0


class _DcgmCircuitBreaker:
    """Gates DCGM scrape attempts after repeated consecutive failures.

    Holds no more than a failure count and a "next attempt allowed at"
    monotonic timestamp. One instance is shared module-wide (see
    ``_dcgm_breaker`` below): this process talks to at most one configured
    ``SYS_STATS_DCGM_URL``, so there is exactly one endpoint's health to
    track, not one per call.
    """

    def __init__(self, threshold: int, base_backoff: float, max_backoff: float) -> None:
        """
        Parameters
        ----------
        threshold : int
            Consecutive failures required before the breaker starts
            skipping attempts.
        base_backoff : float
            Seconds to wait before the first retry once the breaker opens.
        max_backoff : float
            Cap on the backoff delay, however many failures accumulate past
            ``threshold``.
        """
        self._threshold = threshold
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self.consecutive_failures = 0
        self._next_attempt_at = 0.0

    def allow_attempt(self) -> bool:
        """Whether a scrape attempt may be made right now.

        Returns
        -------
        bool
            ``True`` when the breaker is closed, or open but past its
            backoff window; ``False`` while a backoff window is active.
        """
        return time.monotonic() >= self._next_attempt_at

    def record_success(self) -> None:
        """Close the breaker: a successful scrape resets the failure count."""
        self.consecutive_failures = 0
        self._next_attempt_at = 0.0

    def record_failure(self) -> None:
        """Record one failed attempt, opening the breaker past the threshold.

        Backoff is computed from how far past ``threshold`` the failure
        streak has gone, doubling each additional failure and capped at
        ``max_backoff``, so the delay before the *next* allowed attempt
        keeps growing the longer the endpoint stays down.
        """
        self.consecutive_failures += 1
        if self.consecutive_failures >= self._threshold:
            backoff_steps = self.consecutive_failures - self._threshold
            backoff = min(self._base_backoff * (2 ** backoff_steps), self._max_backoff)
            self._next_attempt_at = time.monotonic() + backoff

    def reset(self) -> None:
        """Return to a fresh, closed state. Test-only."""
        self.consecutive_failures = 0
        self._next_attempt_at = 0.0


# Single module-level breaker instance; see _DcgmCircuitBreaker's docstring
# for why one instance is enough.
_dcgm_breaker = _DcgmCircuitBreaker(
    _DCGM_BREAKER_THRESHOLD, _DCGM_BREAKER_BASE_BACKOFF, _DCGM_BREAKER_MAX_BACKOFF
)


def _reset_dcgm_breaker_for_tests() -> None:
    """Reset the module-level DCGM circuit breaker to a closed state.

    Not part of the public API. Without this, one test's induced failures
    would leave the breaker open for whichever test runs next, exactly like
    :func:`sys_stats.sampler._reset_for_tests` exists to stop a leftover
    background thread from corrupting later tests.
    """
    _dcgm_breaker.reset()


def _fetch_dcgm_text(url: str) -> str:
    """Fetch the raw dcgm-exporter Prometheus text body, bounded by a hard deadline.

    Streams the response instead of using ``requests``' buffered ``.text``,
    checking elapsed wall-clock time after every chunk against
    ``_DCGM_HARD_DEADLINE``. This is what catches a connection that dribbles
    bytes slowly enough to never trip ``_DCGM_READ_TIMEOUT`` on any single
    read -- see the trap documented above ``_DCGM_CONNECT_TIMEOUT``.

    Parameters
    ----------
    url : str
        The dcgm-exporter ``/metrics`` URL.

    Returns
    -------
    str
        The decoded response body.

    Raises
    ------
    requests.RequestException
        On a connection failure, an HTTP error status, or a per-socket
        timeout (see ``_DCGM_CONNECT_TIMEOUT`` / ``_DCGM_READ_TIMEOUT``).
    TimeoutError
        When the transfer as a whole exceeds ``_DCGM_HARD_DEADLINE`` even
        though no single socket operation individually timed out.
    """
    deadline = time.monotonic() + _DCGM_HARD_DEADLINE
    response = requests.get(
        url,
        timeout=(_DCGM_CONNECT_TIMEOUT, _DCGM_READ_TIMEOUT),
        stream=True,
    )
    try:
        response.raise_for_status()
        chunks: list[bytes] = []
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"DCGM scrape of {url} exceeded the {_DCGM_HARD_DEADLINE}s hard deadline"
                )
        return b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()


def _parse_dcgm_metrics(text: str) -> list[dict[str, Any]]:
    """Parse a dcgm-exporter Prometheus text body into per-GPU ``/panel`` entries.

    Keys ONLY on the ``gpu`` label of each sample line; every other label
    (``namespace``, ``pod``, ``container``, ``hostname``, ``UUID``,
    ``pci_bus_id``, ...) is ignored. Measured on the target endpoint: the
    same GPU index was seen carrying a different pod's labels an hour apart,
    and later carrying the labels of a pod that also held the other GPU,
    with nothing redeployed in between -- those labels describe whichever
    workload currently holds the device and move on their own, so grouping
    on them would silently stop matching.

    Parameters
    ----------
    text : str
        The raw ``/metrics`` response body.

    Returns
    -------
    list of dict
        One entry per distinct ``gpu`` label, in the exact ``/panel`` GPU
        shape (``i``, ``n``, ``load``, ``mem_used``, ``mem_total``,
        ``mem_pct``, ``temp``, ``fan``, ``power``), sorted by ``i`` for a
        stable display order (see get_temperatures for why this codebase
        always sorts positionally-rendered lists). A line that does not
        match the expected exposition-format shape, carries a metric family
        outside ``_DCGM_METRIC_FIELDS``, has no ``gpu`` label, or has a
        non-numeric value, is skipped rather than aborting the whole parse.
    """
    per_gpu: dict[str, dict[str, Any]] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        match = _DCGM_METRIC_LINE_RE.match(line)
        if not match:
            continue

        name = match.group('name')
        if name not in _DCGM_METRIC_FIELDS:
            continue

        try:
            value = float(match.group('value'))
        except ValueError:
            continue

        labels = dict(_DCGM_LABEL_RE.findall(match.group('labels')))
        gpu_label = labels.get('gpu')
        if gpu_label is None:
            continue

        entry = per_gpu.setdefault(gpu_label, {})
        entry[name] = value
        model_name = labels.get('modelName')
        if model_name:
            entry['modelName'] = model_name

    results: list[dict[str, Any]] = []
    for gpu_label, metrics in per_gpu.items():
        # MiB -> bytes: /panel reports every memory figure in bytes (see
        # _build_panel_gpu_entry in server.py); FB_USED/FB_FREE are the
        # exporter's own "Framebuffer memory used/free (in MiB)" families.
        mem_used = int(metrics.get('DCGM_FI_DEV_FB_USED', 0.0) * 1024 * 1024)
        mem_free = int(metrics.get('DCGM_FI_DEV_FB_FREE', 0.0) * 1024 * 1024)

        # DCGM exposes no total-memory metric, so it is derived as used +
        # free. Measured on an idle card: used=0 MiB, free=24125 MiB, giving
        # a derived total of 24125 MiB rather than the card's nominal 24576
        # -- the gap is framebuffer the driver reserves for itself. That is
        # the correct total for this endpoint and is deliberately not
        # rounded up to the nominal figure.
        mem_total = mem_used + mem_free
        mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0.0

        results.append({
            "i": int(gpu_label),
            "n": metrics.get("modelName") or f"GPU {gpu_label}",
            "load": round(metrics.get("DCGM_FI_DEV_GPU_UTIL", 0.0), 1),
            "mem_used": mem_used,
            "mem_total": mem_total,
            "mem_pct": round(mem_pct, 1),
            "temp": round(metrics.get("DCGM_FI_DEV_GPU_TEMP", 0.0), 1),
            # Fan speed is the exporter's own "Fan speed (in %)" family, not
            # RPM. Measured on healthy idle 3090s: 0 -- genuine zero-RPM
            # mode, not a missing field, so it is not treated as absent.
            "fan": int(round(metrics.get("DCGM_FI_DEV_FAN_SPEED", 0.0))),
            "power": round(metrics.get("DCGM_FI_DEV_POWER_USAGE", 0.0), 1),
        })

    results.sort(key=lambda e: e["i"])
    return results


def get_dcgm_gpus(url: str) -> list[dict[str, Any]]:
    """Retrieve per-GPU metrics from a dcgm-exporter Prometheus endpoint.

    Used in place of the GPUtil/nvidia-smi path (see :func:`collect_stats`)
    on hosts with no NVIDIA driver of their own -- e.g. a Proxmox hypervisor
    whose GPUs are PCI-passed-through to a Kubernetes VM, where dcgm-exporter
    runs inside that VM instead. Returns entries already in the exact shape
    ``/panel``'s ``gpu[]`` uses, so the caller (see
    :func:`sys_stats.sampler._collect_panel_extras`) can use them as a
    drop-in replacement with no further reshaping.

    Same graceful-degradation contract as this module's nvidia-smi
    collectors (:func:`get_gpu_fan_and_power`, :func:`get_gpu_processes`):
    never raises, degrades to an empty list on any failure -- a dead
    connection, an HTTP error, a timeout, a body that does not parse -- so a
    scrape failure can never take down the sampler's pass. Gated by a
    circuit breaker (see ``_dcgm_breaker`` / ``_DcgmCircuitBreaker``): after
    ``_DCGM_BREAKER_THRESHOLD`` consecutive failures, further attempts are
    skipped and spaced out with capped backoff instead of being retried on
    every sampling pass; the breaker closes again on the first success.

    Parameters
    ----------
    url : str
        The dcgm-exporter ``/metrics`` URL, expected to carry a literal IP
        rather than a hostname (see the DNS-resolution note above
        ``_DCGM_BREAKER_THRESHOLD``).

    Returns
    -------
    list of dict
        One entry per GPU reported. Empty when the breaker is open, the
        scrape failed, or the body carried no recognised metric lines.
    """
    if not _dcgm_breaker.allow_attempt():
        logger.debug(f"DCGM breaker open; skipping scrape of {url}")
        return []

    try:
        text = _fetch_dcgm_text(url)
        # Parsing stays inside the guard: a malformed body -- a non-numeric
        # ``gpu`` label, a truncated response -- is a failure of this
        # endpoint like any network error, and this function documents that
        # it never raises. sampler catches it too, but a contract that only
        # holds because someone else also guards it is not a contract.
        gpus = _parse_dcgm_metrics(text)
    except Exception as e:
        logger.warning(f"Error fetching DCGM GPU metrics from {url}: {e}")
        _dcgm_breaker.record_failure()
        return []

    _dcgm_breaker.record_success()
    return gpus


def get_temperatures() -> list[dict[str, Any]]:
    """Retrieve hardware temperature sensors, sorted for stable display order.

    ``psutil.sensors_temperatures`` does not exist as an attribute at all on
    macOS (as opposed to existing and returning an empty dict), so the
    platform gate happens through ``getattr`` before any call is attempted,
    exactly like the ``nvidia-smi`` degradation elsewhere in this module.

    Returns
    -------
    list of dict
        Entries ``{"n": "<chip>/<label>", "c": <float, rounded to 1
        decimal>}``, sorted by ``"n"``. A probe with an empty ``label``
        falls back to the chip name plus its index within that chip (e.g.
        ``"k10temp/1"``) so several unlabeled probes on one chip do not
        collide on ``"n"``. Entries whose ``current`` reading is ``None``
        are skipped. Empty on a platform without sensor support.
    """
    sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
    if sensors_temperatures is None:
        return []

    entries: list[dict[str, Any]] = []
    for chip, chip_entries in sensors_temperatures().items():
        for index, entry in enumerate(chip_entries):
            if entry.current is None:
                continue
            label = entry.label or str(index)
            entries.append({"n": f"{chip}/{label}", "c": round(entry.current, 1)})

    # Observed: psutil.sensors_temperatures() returns an UNSORTED mapping.
    # Measured on the target host (AMD EPYC, Debian), the k10temp entries
    # came back as Tctl, Tccd8, Tccd1, Tccd2 ... Tccd7 -- neither
    # alphabetical nor numeric. That single observation is the whole
    # justification for sorting here.
    #
    # Why it matters: the consumer is a wall-mounted display that renders
    # this list POSITIONALLY. An order that differs between two samples
    # makes rows physically swap places on screen while someone is looking
    # at it. Sorting removes the question entirely.
    #
    # Deliberately NOT claimed: that the order tracks sysfs hwmon*
    # enumeration, or that it varies across reboots or module reloads.
    # Those are plausible mechanisms nobody here verified, and an earlier
    # version of this comment asserted them as fact. The sort is correct
    # on the observation alone; it does not need the story.
    #
    # Do not remove this sort believing psutil already orders the mapping.
    entries.sort(key=lambda e: e["n"])
    return entries


def get_fans() -> list[dict[str, Any]]:
    """Retrieve fan speed sensors, sorted for stable display order.

    Same macOS gate as :func:`get_temperatures`: ``psutil.sensors_fans`` is
    an absent attribute there, not a function returning an empty dict.

    Returns
    -------
    list of dict
        Entries ``{"n": "<chip>/<label>", "rpm": <int>}``, sorted by
        ``"n"``. A probe with an empty ``label`` falls back to the chip
        name plus its index within that chip, same as
        :func:`get_temperatures`. Entries whose ``current`` reading is
        ``None`` are skipped. Empty on a platform without sensor support.
    """
    sensors_fans = getattr(psutil, "sensors_fans", None)
    if sensors_fans is None:
        return []

    entries: list[dict[str, Any]] = []
    for chip, chip_entries in sensors_fans().items():
        for index, entry in enumerate(chip_entries):
            if entry.current is None:
                continue
            label = entry.label or str(index)
            entries.append({"n": f"{chip}/{label}", "rpm": int(entry.current)})

    # See get_temperatures: psutil was observed returning its sensor
    # mapping unsorted, and the wall panel renders this list positionally,
    # so it is re-sorted on every sample rather than trusted to be ordered.
    # No claim is made here about WHY the order comes out as it does.
    entries.sort(key=lambda e: e["n"])
    return entries


# Last failure reason reported by each IPMI collector, keyed by sensor kind.
#
# The sampler calls both IPMI collectors on its very first pass and every
# SYS_STATS_IPMI_INTERVAL seconds after that, whatever the hardware. A machine
# with no BMC is therefore not an edge case, it is the common one: every
# container without /dev/ipmi0 passed in, every laptop, every VM. Logging each
# failure at ERROR turned that into two lines every 30s forever, around 5,700 a
# day, which buries the errors that do deserve attention.
#
# So a failure is reported once at its natural level, then at DEBUG for as long
# as the reason is unchanged, and escalates again the moment the reason changes
# or the sensor starts answering. Only the sampler's background thread touches
# this dict, so it needs no lock.
_ipmi_last_failure: dict[str, str] = {}


def _log_ipmi_failure(sensor: str, message: str, level: int = logging.ERROR) -> None:
    """Report an IPMI collector failure without flooding the log.

    Parameters
    ----------
    sensor : str
        Which collector failed, ``"fan"`` or ``"temperature"``. The two are
        tracked apart on purpose: a BMC can answer one ``sdr type`` query and
        fail the other, and collapsing them would hide the second failure
        behind the first.
    message : str
        The failure reason. It is both the logged text and the identity of the
        failure: an unchanged reason is a repeat, a different one is a new
        condition that deserves a line of its own.
    level : int, optional
        Level for the first report of a given reason. Defaults to
        ``logging.ERROR``; a timeout passes ``logging.WARNING`` so that
        damping the volume does not also promote its severity.
    """
    if _ipmi_last_failure.get(sensor) == message:
        logger.debug(f"IPMI {sensor} sensors still failing, unchanged: {message}")
        return

    _ipmi_last_failure[sensor] = message
    logger.log(level, f"Error fetching IPMI {sensor} sensors: {message}")


def _clear_ipmi_failure(sensor: str) -> None:
    """Forget a collector's last failure, so the next one logs in full again.

    Called on every successful ``ipmitool`` call. A BMC that recovers and then
    breaks again the same way is a new incident, not a continuation of the old
    one, and deserves to be reported as such.
    """
    _ipmi_last_failure.pop(sensor, None)


def _reset_ipmi_failure_log_for_tests() -> None:
    """Forget every recorded IPMI failure. Test-only.

    Same purpose as :func:`_reset_dcgm_breaker_for_tests`: without it, one
    test's induced failure would damp the next test's first log line.
    """
    _ipmi_last_failure.clear()


def _run_ipmitool_sdr_fan(timeout: float = 5.0) -> str | None:
    """Ask ``ipmitool`` for the chassis fan sensors over IPMI.

    ``sdr type fan`` is the narrowest subcommand that still reports every
    fan sensor the BMC knows about, readable or not, one pipe-separated
    line per sensor -- exactly the shape :func:`get_ipmi_fans` expects to
    parse.

    Parameters
    ----------
    timeout : float, optional
        Seconds to wait for ``ipmitool`` before giving up.

    Returns
    -------
    str or None
        Raw stdout of the command, or ``None`` when ``ipmitool`` is
        missing, fails, times out, or is otherwise unreachable.
    """
    try:
        result = subprocess.run(
            ['ipmitool', 'sdr', 'type', 'fan'],
            capture_output=True,
            text=True,
            check=True,
            # ipmitool talks to a BMC over /dev/ipmi*; a wedged or
            # unreachable BMC hangs the command indefinitely instead of
            # failing fast. This collector runs inside the sampler's
            # background thread, so an unbounded call would freeze the
            # cache while the wall panel kept showing stale numbers that
            # look perfectly fresh. A short, explicit timeout turns that
            # hang into an ordinary degrade-to-empty case instead.
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        _log_ipmi_failure("fan", e.stderr.strip())
        return None
    except FileNotFoundError as e:
        # ``ipmitool`` is not installed at all, as opposed to installed but
        # failing (CalledProcessError) or hanging (TimeoutExpired).
        _log_ipmi_failure("fan", str(e))
        return None
    except subprocess.TimeoutExpired as e:
        _log_ipmi_failure("fan", f"timed out: {e}", level=logging.WARNING)
        return None

    _clear_ipmi_failure("fan")
    return result.stdout


def get_ipmi_fans() -> list[dict[str, Any]]:
    """Retrieve chassis fan speed sensors reported over IPMI, sorted for stable display.

    Complements :func:`get_fans`: some chassis (e.g. Proxmox hypervisor
    hosts) do not expose their fans through hwmon/sysfs at all, only
    through the BMC. This collector talks to ``ipmitool`` the same way the
    rest of this module talks to ``nvidia-smi`` -- a subprocess call,
    parsed defensively, degrading to an empty list rather than raising.

    Returns
    -------
    list of dict
        Entries ``{"n": "ipmi/<sensor name>", "rpm": <int>}``, sorted by
        ``"n"`` after the prefix is applied. See :func:`get_ipmi_temperatures`
        for why the ``ipmi/`` prefix exists; the same reasoning applies here
        (BMC fan labels like ``FAN1``/``FAN2`` collide in shape with the
        unprefixed hwmon convention used by :func:`get_fans`).
        A sensor the BMC reports as unreadable (``No Reading``,
        ``Disabled``, ``N/A``, ...) is a sensor that is not there, not a
        fan spinning at 0 RPM -- publishing it as 0 would render on the
        wall display as a stopped-fan alarm for a sensor that simply has
        nothing to say, so such entries are omitted rather than coerced.
        Empty when ``ipmitool`` is missing, fails, times out, or its
        output does not parse.
    """
    stdout = _run_ipmitool_sdr_fan()
    if stdout is None:
        return []

    entries: list[dict[str, Any]] = []
    for line in stdout.strip().split('\n'):
        if not line.strip():
            continue

        # ipmitool's `sdr type fan` output is pipe-separated, typically
        # "<name> | <id> | <status> | <entity> | <reading>", but the field
        # count is not load-bearing here: only the first field (sensor
        # name) and the last field (reading) are used, so this keeps
        # working across BMC firmware that formats the middle differently.
        fields = line.split('|')
        if len(fields) < 2:
            logger.warning(f"Skipping malformed IPMI fan sensor line: '{line}'")
            continue

        name = fields[0].strip()
        reading = fields[-1].strip()
        try:
            # The reading field is free text: "6300 RPM" for a working
            # sensor, "No Reading" / "Disabled" / "N/A" for one the BMC
            # cannot poll right now. Only the leading numeric token is
            # kept; anything that does not start with one drops the whole
            # entry rather than being coerced to 0 (see the "No Reading"
            # rationale above).
            rpm = int(reading.split()[0])
        except (ValueError, IndexError):
            continue

        # Prefixed here, before sorting: see get_ipmi_temperatures for why
        # IPMI entries carry an "ipmi/" prefix while hwmon entries carry a
        # "chip/" prefix (get_fans/get_temperatures), and why the two
        # conventions must stay visually distinct rather than unified.
        entries.append({"n": f"ipmi/{name}", "rpm": rpm})

    # Same treatment as get_fans/get_temperatures, for the same reason:
    # the wall panel renders this list positionally, so a stable order is
    # part of the contract and sorting guarantees it whatever ipmitool
    # hands back. Unlike the psutil case we have no observation of this
    # BMC's output order, and no claim is made about it -- sorting is
    # cheap enough that it does not need one. Sorted on the prefixed name,
    # never the raw BMC label, so the order matches what ships in /panel.
    entries.sort(key=lambda e: e["n"])
    return entries


def _run_ipmitool_sdr_temperature(timeout: float = 5.0) -> str | None:
    """Ask ``ipmitool`` for the chassis temperature sensors over IPMI.

    Sibling of :func:`_run_ipmitool_sdr_fan`: ``sdr type temperature`` is
    the narrowest subcommand that still reports every temperature sensor
    the BMC knows about, readable or not, one pipe-separated line per
    sensor -- exactly the shape :func:`get_ipmi_temperatures` expects to
    parse.

    Parameters
    ----------
    timeout : float, optional
        Seconds to wait for ``ipmitool`` before giving up.

    Returns
    -------
    str or None
        Raw stdout of the command, or ``None`` when ``ipmitool`` is
        missing, fails, times out, or is otherwise unreachable.
    """
    try:
        result = subprocess.run(
            ['ipmitool', 'sdr', 'type', 'temperature'],
            capture_output=True,
            text=True,
            check=True,
            # See _run_ipmitool_sdr_fan: ipmitool talks to a BMC over
            # /dev/ipmi*, and a wedged or unreachable BMC hangs indefinitely
            # instead of failing fast. This collector runs inside the
            # sampler's background thread, so an unbounded call would freeze
            # the cache while the wall panel kept showing stale numbers that
            # look perfectly fresh.
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        _log_ipmi_failure("temperature", e.stderr.strip())
        return None
    except FileNotFoundError as e:
        # ``ipmitool`` is not installed at all, as opposed to installed but
        # failing (CalledProcessError) or hanging (TimeoutExpired).
        _log_ipmi_failure("temperature", str(e))
        return None
    except subprocess.TimeoutExpired as e:
        _log_ipmi_failure("temperature", f"timed out: {e}", level=logging.WARNING)
        return None

    _clear_ipmi_failure("temperature")
    return result.stdout


def get_ipmi_temperatures() -> list[dict[str, Any]]:
    """Retrieve chassis temperature sensors reported over IPMI, sorted for stable display.

    Sibling of :func:`get_ipmi_fans`, complementing :func:`get_temperatures`:
    some chassis (e.g. Proxmox hypervisor hosts) report temperatures the BMC
    knows about but hwmon/sysfs does not. Returns the exact same entry shape
    as :func:`get_temperatures` so the two can be concatenated into one list
    by the caller (see :func:`sys_stats.sampler._collect_panel_extras`).

    Some BMCs report GPU temperatures among their sensors -- these are named
    probes like any other and belong here, not folded into the ``gpu[]``
    section of ``/panel``: a GPU entry with only a temperature and zeros
    everywhere else would read as an idle card, not as missing data.

    Every entry name is prefixed ``ipmi/``, e.g. ``ipmi/CPU Temp``,
    ``ipmi/GPU1 Temp``. hwmon entries are already prefixed ``chip/label``
    by :func:`get_temperatures` (``k10temp/Tctl``, ``nvme/Composite``); this
    matches the two lists to one convention instead of mixing a prefixed
    and an unprefixed name in the same ``temps`` list. Two concrete
    ambiguities this removes, both observed on the same host:

    - The BMC reports ``GPU1 Temp`` / ``GPU4 Temp`` (motherboard slot
      labels, on a board with four slots of which two are populated). The
      ``/panel`` payload separately carries a ``gpu[]`` array indexed
      ``i=0`` / ``i=1`` from GPUtil/nvidia-smi. These two numbering systems
      are unrelated and nothing joins them; an unprefixed ``GPU1 Temp``
      sitting next to ``gpu[].i = 1`` invites pairing them, and the pairing
      would be wrong.
    - CPU temperature arrives twice, from two different measurement
      points: ``k10temp/Tctl`` (on-die, ~58 degrees C observed) and
      ``ipmi/CPU Temp`` (motherboard sensor, ~52 degrees C observed). Both
      readings are correct; they measure different points. Without the
      prefixes these are two lines that disagree by six degrees with
      nothing on the line explaining why.

    ``ipmi/`` names a protocol, not a chip, so ``bmc/`` (the component)
    would be more consistent with the ``chip/label`` convention above. It
    was not used: ``ipmi`` is the term someone searches for or greps when
    they hit an unfamiliar line on the wall display, and that
    recognisability was judged more useful here than taxonomic consistency
    with ``k10temp/``/``nvme/``.

    Returns
    -------
    list of dict
        Entries ``{"n": "ipmi/<sensor name>", "c": <float, rounded to 1
        decimal>}``, sorted by ``"n"`` after the prefix is applied. A
        sensor the BMC reports as unreadable (``No Reading``, ``Disabled``,
        ``N/A``, ...) is a sensor that is not there, not a probe reading
        0degC -- such entries are omitted rather than coerced, exactly
        like :func:`get_ipmi_fans`. Empty when ``ipmitool`` is missing,
        fails, times out, or its output does not parse.
    """
    stdout = _run_ipmitool_sdr_temperature()
    if stdout is None:
        return []

    entries: list[dict[str, Any]] = []
    for line in stdout.strip().split('\n'):
        if not line.strip():
            continue

        # Same pipe-separated shape as get_ipmi_fans: only the first field
        # (sensor name) and the last field (reading) are used, so this keeps
        # working across BMC firmware that formats the middle differently.
        fields = line.split('|')
        if len(fields) < 2:
            logger.warning(f"Skipping malformed IPMI temperature sensor line: '{line}'")
            continue

        name = fields[0].strip()
        reading = fields[-1].strip()
        try:
            # The reading field is free text: "45 degrees C" for a working
            # sensor, "No Reading" / "Disabled" / "N/A" for one the BMC
            # cannot poll right now. Only the leading numeric token is kept;
            # anything that does not start with one drops the whole entry
            # rather than being coerced to 0 (see the "No Reading" rationale
            # above).
            celsius = float(reading.split()[0])
        except (ValueError, IndexError):
            continue

        # See the docstring above for why this prefix exists: it keeps
        # BMC slot labels like "GPU1 Temp" from being mistaken for the
        # /panel gpu[] array's own i=0/i=1 index, and keeps the two CPU
        # temperature readings (k10temp/Tctl vs ipmi/CPU Temp) visibly
        # distinct instead of looking like a contradiction.
        entries.append({"n": f"ipmi/{name}", "c": round(celsius, 1)})

    # Same treatment as get_ipmi_fans, for the same reason: the wall panel
    # renders this list positionally, so a stable order is part of the
    # contract and sorting guarantees it whatever ipmitool hands back. We
    # have no observation of this BMC's output order and no claim is made
    # about it -- sorting is cheap enough that it does not need one.
    # Sorted on the prefixed name, applied before this sort and
    # before concatenation with the hwmon list in
    # sys_stats.sampler._collect_panel_extras, never after either.
    entries.sort(key=lambda e: e["n"])
    return entries


def get_swap() -> dict[str, Any]:
    """Retrieve swap memory usage.

    Returns
    -------
    dict
        ``{"used": <int bytes>, "total": <int bytes>, "pct": <float,
        rounded to 1 decimal>}``, straight from ``psutil.swap_memory()``.
    """
    swap = psutil.swap_memory()
    return {
        "used": swap.used,
        "total": swap.total,
        "pct": round(swap.percent, 1),
    }


def get_per_core_cpu() -> list[float]:
    """Retrieve per-logical-core CPU usage percentages.

    Uses the same non-blocking ``interval=None`` convention as the
    aggregate figure in :func:`collect_stats`: it reports usage since the
    previous ``percpu=True`` call. Critically, psutil keeps a SEPARATE
    internal baseline for ``cpu_percent(percpu=True)`` from the one it
    keeps for the plain ``cpu_percent()`` call; they do not share state.
    :mod:`sys_stats.sampler` primes both baselines once at startup before
    the first real sample, so this function must never be called before
    that priming has happened, or its first reading will be a meaningless
    near-zero regardless of actual load.

    Returns
    -------
    list of float
        One percentage per logical core, each rounded to 1 decimal, in
        ``psutil.cpu_percent(percpu=True)`` order.
    """
    return [round(p, 1) for p in psutil.cpu_percent(interval=None, percpu=True)]


def get_load_average() -> list[float]:
    """Retrieve the 1/5/15 minute load average.

    ``os.getloadavg()`` is unavailable on some platforms (Windows) and
    raises ``OSError`` there instead of returning a fallback value itself.
    The ``/panel`` contract always needs three numbers regardless, so the
    degradation happens here rather than in every caller.

    Returns
    -------
    list of float
        ``[1min, 5min, 15min]`` load average, or ``[0.0, 0.0, 0.0]`` on a
        platform without ``os.getloadavg()`` support.
    """
    try:
        return list(os.getloadavg())
    except OSError:
        return [0.0, 0.0, 0.0]


def get_cpu_frequency_mhz() -> int:
    """Retrieve the current CPU frequency in MHz.

    ``psutil.cpu_freq()`` is documented upstream as returning ``None`` on
    platforms that cannot report it, and it may also raise. Both cases
    degrade to ``0`` here rather than propagating a misleading reading.

    An earlier version of this docstring also asserted that it "commonly
    reports 0 inside a container". That was never measured, so it is gone:
    the ``None`` guard and the ``except`` are justified by the documented
    behaviour alone and do not need the extra claim.

    Returns
    -------
    int
        Current CPU frequency in MHz, or ``0`` when unavailable.
    """
    freq = psutil.cpu_freq()
    if freq is None or freq.current is None:
        return 0
    return int(freq.current)


def collect_stats(limit: int = 5) -> dict:
    """Collect the full ``/stats`` payload from the host machine.

    Merges psutil, GPUtil and Ollama data into a single dict. Units are
    normalised here: memory in bytes, loads in percent, power in watts. Takes
    no Flask context, so it can be called from anywhere, not just the
    ``/stats`` route.

    Parameters
    ----------
    limit : int, optional
        Maximum number of entries returned for each per-process ranking
        (``top_cpu``, ``top_memory`` and ``top_gpu_processes``).

    Returns
    -------
    dict
        The payload whose top-level keys are the public contract of the
        project; adding keys is a minor bump, renaming one is a major bump.
    """
    # CPU usage. Deliberately non-blocking (interval=None): it reports usage
    # since the previous call rather than sampling for a second, which is
    # only meaningful because sys_stats.sampler is the sole caller of
    # psutil.cpu_percent and primes the baseline before the first real
    # sample. Calling this with a blocking interval here would fight that
    # baseline every time collect_stats() runs.
    cpu_usage = psutil.cpu_percent(interval=None)
    cpu_cores = psutil.cpu_count(logical=True)

    # RAM usage
    ram_info = psutil.virtual_memory()
    ram_stats = {
        "total": ram_info.total,
        "used": ram_info.used,
        "percent": ram_info.percent
    }

    # Check if there's at least one GPU
    gpus = GPUtil.getGPUs()
    has_gpu = (len(gpus) > 0)
    gpu_stats = []
    top_gpu_processes = []

    if has_gpu:
        # Retrieve extra info from nvidia-smi (fan + power)
        fan_power_data = get_gpu_fan_and_power()

        for gpu in gpus:
            # combine GPUtil info + fan/power
            gpu_index = gpu.id
            fan_speed = fan_power_data.get(gpu_index, {}).get("fan_speed", 0.0)
            power_draw = fan_power_data.get(gpu_index, {}).get("power_draw", 0.0)

            gpu_stats.append({
                "id": gpu_index,
                "name": gpu.name,
                "load": gpu.load * 100,
                "memoryTotal": gpu.memoryTotal,
                "memoryUsed": gpu.memoryUsed * 1024 * 1024, # in bytes
                "memoryPercent": (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                "temperature": gpu.temperature,
                "fanSpeed": fan_speed,     # in %
                "powerDraw": power_draw   # in W
            })

        # nvidia-smi identifies the card of a compute app by UUID only, so hand
        # it the UUID -> index mapping to translate into the same ``id`` the
        # ``gpu`` section above exposes. ``getattr`` keeps stub GPUs (tests,
        # exotic GPUtil versions) from breaking the endpoint.
        uuid_to_index = {}
        for gpu in gpus:
            gpu_uuid = getattr(gpu, "uuid", None)
            if gpu_uuid:
                uuid_to_index[gpu_uuid] = gpu.id

        # If there is a GPU, we call nvidia-smi for GPU processes
        top_gpu_processes = get_gpu_processes(limit=limit, uuid_to_index=uuid_to_index)

    # Summary (display the first GPU if any)
    if has_gpu:
        gpu_summary_load = gpu_stats[0]["load"]
        gpu_summary_vram = gpu_stats[0]["memoryPercent"]
        gpu_summary_name = gpu_stats[0]["name"]
    else:
        gpu_summary_load = 0.0
        gpu_summary_vram = 0.0
        gpu_summary_name = "N/A"

    ollama_processes = get_ollama_process()

    summary = {
        "cpu": {
            "usage": cpu_usage,
            "cores": cpu_cores
        },
        "ram": {
            "total": ram_stats["total"],
            "percent": ram_stats["percent"]
        },
        "gpu": [
            {
                "name": gpu_summary_name,
                "load": gpu_summary_load,
                "vram": gpu_summary_vram
            }
        ] if has_gpu else []
    }

    # Top processes
    top_cpu = get_top_processes_by_cpu(limit=limit)
    top_memory = get_top_processes_by_memory(limit=limit)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "current_time": current_time,
        "has_gpu": has_gpu,
        "summary": summary,
        "cpu": cpu_usage,
        "ram": ram_stats,
        "gpu": gpu_stats,
        "top_cpu": top_cpu,
        "top_memory": top_memory,
        "top_gpu_processes": top_gpu_processes,
        "ollama_processes": ollama_processes
    }
