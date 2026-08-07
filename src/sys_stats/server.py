#!/usr/bin/env python3

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
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

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


def _query_compute_apps() -> str | None:
    """Ask nvidia-smi for the running compute apps, newest query shape first.

    The UUID-aware query is tried once; if the driver rejects it the legacy
    three-field query is tried as well, so a card attribution failure never
    costs the caller the whole process list.

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
                check=True
            )
        except subprocess.CalledProcessError as e:
            last_error = e
            continue
        return result.stdout

    # ``stderr`` is None when the caller did not capture it, so guard the strip.
    stderr = (last_error.stderr or "").strip() if last_error is not None else ""
    logger.error(f"Error fetching GPU processes: {stderr}")
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
        :func:`get_stats` from the GPUtil device list. Without it every entry
        reports ``gpu_index`` as ``None``, which keeps the function testable on
        its own.

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

def get_gpu_fan_and_power() -> dict[int, dict[str, float]]:
    """
    Retrieve fan speed (%) and power draw (W) for each GPU via nvidia-smi.
    Returns a dict keyed by GPU index: {"fan_speed": float, "power_draw": float}.
    """
    data = {}
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,fan.speed,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error fetching GPU fan/power: {e.stderr.strip()}")
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


@app.route('/stats', methods=['GET'])
def get_stats():
    """Serve the ``/stats`` payload consumed by the web UI and the Rich CLI.

    Merges psutil, GPUtil and Ollama data into a single JSON document. Units
    are normalised here: memory in bytes, loads in percent, power in watts.

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

    # CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)
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

    return jsonify({
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
    })

def main() -> None:
    """Console-script entry point: run the Flask metrics server.

    Honours the ``FLASK_DEBUG``, ``HOST`` and ``PORT`` environment variables.
    """
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=debug_mode)


if __name__ == '__main__':
    main()
