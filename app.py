#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import psutil
import subprocess
from typing import List, Dict, Any
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
import GPUtil
import coloredlogs
import requests
import datetime
from urllib.parse import urljoin


OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

def get_top_processes_by_cpu(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top processes by CPU usage.
    """
    processes = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "cmdline"]):
        try:
            processes.append({
                "pid": p.info["pid"],
                "name": p.info["name"],
                "cpu_percent": p.info["cpu_percent"],
                "cmdline": " ".join(p.info["cmdline"]) if p.info["cmdline"] else "N/A"
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    processes.sort(key=lambda x: x["cpu_percent"], reverse=True)
    logger.debug(f"Top CPU processes: {processes[:limit]}")
    return processes[:limit]

def get_top_processes_by_memory(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top processes by memory usage.
    """
    processes = []
    for p in psutil.process_iter(["pid", "name", "memory_percent", "memory_info", "cmdline"]):
        try:
            processes.append({
                "pid": p.info["pid"],
                "name": p.info["name"],
                "memory_usage": p.info["memory_info"].rss,
                "memory_percent": p.info["memory_percent"],
                "cmdline": " ".join(p.info["cmdline"]) if p.info["cmdline"] else "N/A"
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    processes.sort(key=lambda x: x["memory_percent"], reverse=True)
    logger.debug(f"Top Memory processes: {processes[:limit]}")
    return processes[:limit]

def get_gpu_processes(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top GPU processes using nvidia-smi for process usage.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error fetching GPU processes: {e.stderr.strip()}")
        return []

    lines = result.stdout.strip().split('\n')
    gpu_processes = []
    for line in lines:
        if not line.strip():
            continue
        try:
            pid_str, process_name, used_memory_str = line.split(', ')
            process_name = process_name.split('/')[-1]
            pid = int(pid_str)
            # Try to get command line using psutil
            try:
                p = psutil.Process(pid)
                cmdline = " ".join(p.cmdline()) if p.cmdline() else "N/A"
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                cmdline = "N/A"

            gpu_processes.append({
                "pid": pid,
                "name": process_name,
                "memory_used": int(used_memory_str)*1024*1024, # Convert MiB to bytes
                "cmdline": cmdline
            })
        except ValueError:
            logger.warning(f"Skipping malformed GPU process line: '{line}'")
            continue

    gpu_processes.sort(key=lambda x: x["memory_used"], reverse=True)
    logger.debug(f"Top GPU processes: {gpu_processes[:limit]}")
    return gpu_processes[:limit]

def get_gpu_fan_and_power() -> Dict[int, Dict[str, float]]:
    """
    Retrieve fan speed (%) and power draw (W) for each GPU via nvidia-smi.
    Returns a dict keyed by GPU index: {"fan_speed": float, "power_draw": float}.
    """
    data = {}
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,fan.speed,power.draw', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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

        # If there is a GPU, we call nvidia-smi for GPU processes
        top_gpu_processes = get_gpu_processes(limit=limit)

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

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
