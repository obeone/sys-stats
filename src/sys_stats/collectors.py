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
import subprocess
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
            # processes when the server runs unprivileged (macOS, container
            # without `pid: host` + `privileged`), and sorting None against a
            # float would blow up the whole endpoint.
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
# or a serial BMC channel like ``ipmitool`` -- on a healthy host it answers in
# well under 100ms. 3 seconds is therefore ample slack over the healthy case
# while still bounding a wedged-driver hang tightly: this collector runs
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
        logger.error(f"Error fetching IPMI fan sensors: {e.stderr.strip()}")
        return None
    except FileNotFoundError as e:
        # ``ipmitool`` is not installed at all, as opposed to installed but
        # failing (CalledProcessError) or hanging (TimeoutExpired).
        logger.error(f"Error fetching IPMI fan sensors: {e}")
        return None
    except subprocess.TimeoutExpired as e:
        logger.warning(f"Timed out waiting for IPMI fan sensors: {e}")
        return None

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
        Entries ``{"n": <sensor name>, "rpm": <int>}``, sorted by ``"n"``.
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

        entries.append({"n": name, "rpm": rpm})

    # Same treatment as get_fans/get_temperatures, for the same reason:
    # the wall panel renders this list positionally, so a stable order is
    # part of the contract and sorting guarantees it whatever ipmitool
    # hands back. Unlike the psutil case we have no observation of this
    # BMC's output order, and no claim is made about it -- sorting is
    # cheap enough that it does not need one.
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
        logger.error(f"Error fetching IPMI temperature sensors: {e.stderr.strip()}")
        return None
    except FileNotFoundError as e:
        # ``ipmitool`` is not installed at all, as opposed to installed but
        # failing (CalledProcessError) or hanging (TimeoutExpired).
        logger.error(f"Error fetching IPMI temperature sensors: {e}")
        return None
    except subprocess.TimeoutExpired as e:
        logger.warning(f"Timed out waiting for IPMI temperature sensors: {e}")
        return None

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

    Returns
    -------
    list of dict
        Entries ``{"n": <sensor name>, "c": <float, rounded to 1 decimal>}``,
        sorted by ``"n"``. A sensor the BMC reports as unreadable (``No
        Reading``, ``Disabled``, ``N/A``, ...) is a sensor that is not
        there, not a probe reading 0degC -- such entries are omitted rather
        than coerced, exactly like :func:`get_ipmi_fans`. Empty when
        ``ipmitool`` is missing, fails, times out, or its output does not
        parse.
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

        entries.append({"n": name, "c": round(celsius, 1)})

    # Same rationale as get_ipmi_fans/get_temperatures: sensor enumeration
    # order is not guaranteed stable across BMC firmware versions or
    # reboots, and the wall panel renders this list positionally, so it is
    # re-sorted on every single sample rather than trusted to already be
    # ordered.
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
