#!/usr/bin/env python3

import requests
import time
import argparse
from prettytable import PrettyTable
import shutil
from termcolor import colored
import os
from datetime import datetime, timedelta, timezone

SYS_STATS_API_URL = os.getenv('SYS_STATS_API_URL', 'http://localhost:5000/stats')

def fetch_stats(api_url):
    """
    Fetch statistics from the specified API URL.

    Args:
        api_url (str): The URL of the API to fetch stats from.

    Returns:
        dict or None: The JSON response from the API if successful, None otherwise.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(colored(f"Error fetching stats: {e}", "red"))
        return None

def human_readable_size(size):
    """
    Convert bytes into a human-readable format.

    Args:
        size (int): The size in bytes.

    Returns:
        str: The size converted into a human-readable format (e.g., KB, MB).
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024

def time_until(expiration):
    """
    Calculate the time remaining until a given expiration datetime.

    Args:
        expiration (str): The expiration datetime in ISO format.

    Returns:
        str: A string representing the time until expiration or "Expired" if already expired.
    """
    exp_time = datetime.fromisoformat(expiration.replace('Z', '+00:00')).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    delta = exp_time - now
    if delta.total_seconds() <= 0:
        return "Expired"
    return str(timedelta(seconds=int(delta.total_seconds())))

def truncate_cmdline(cmdline, width):
    """
    Truncate command line string if it exceeds a specified width.

    Args:
        cmdline (str): The command line string to truncate.
        width (int): The maximum width allowed.

    Returns:
        str: The truncated command line string.
    """
    return cmdline if len(cmdline) <= width else cmdline[:width - 3] + "..."

def clear_terminal():
    """
    Clear the terminal screen.
    """
    print("\033[H\033[J", end="")  # ANSI escape sequence to clear screen and reset cursor

def display_oneline_stats(data):
    """
    Display statistics on a single line with icons.

    Args:
        data (dict): The fetched statistics data.
    """
    # Icons to represent metrics
    cpu_icon = "💻"
    ram_icon = "🧠"
    gpu_icon = "🎮"

    cpu_usage_str = colored(f"{data['cpu']:.1f}%", "green", attrs=["bold"])
    ram_usage_str = colored(f"{data['ram']['percent']:.1f}%", "yellow", attrs=["bold"])
    ram_total_str = colored(human_readable_size(data['ram']['total']), "cyan")

    usage_line = (
        f"{cpu_icon} CPU: {cpu_usage_str}  "
        f"{ram_icon} RAM: {ram_usage_str} / {ram_total_str}"
    )

    if data.get("has_gpu") and data.get("gpu"):
        gpu_data = data['gpu'][0]
        gpu_usage_str = colored(f"{gpu_data['load']:.1f}%", "blue", attrs=["bold"])
        vram_usage_str = colored(f"{gpu_data['memoryPercent']:.1f}%", "magenta", attrs=["bold"])
        usage_line += f"  {gpu_icon} GPU: {gpu_usage_str} VRAM: {vram_usage_str}"

    if data.get("ollama_processes", {}).get("models"):
        model = data['ollama_processes']['models'][0]  # First entry
        model_name = model['name']
        model_size = colored(human_readable_size(model['size_vram']), "blue")
        usage_line += f"  (Ollama: {model_name}, {model_size})"

    print(colored("\nOneline Stats:", "cyan", attrs=["bold"]))
    print(usage_line)

def display_summary(data, terminal_width):
    """
    Display a summary of the statistics in a table format.

    Args:
        data (dict): The fetched statistics data.
        terminal_width (int): The width of the terminal for formatting purposes.
    """
    print(colored("\nSummary:", "cyan", attrs=["bold"]))
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]

    # Calculate average space per column
    nb_columns = len(table.field_names)
    width_per_col = max((terminal_width - 4) // nb_columns, 5)  # 4 for margin, 5 min

    # Adjust column alignment
    for field in table.field_names:
        table.align[field] = "l"

    cpu_usage = colored(f"{data['cpu']:.1f}%", "green", attrs=["bold"])
    ram_usage = colored(f"{data['ram']['percent']:.1f}%", "yellow", attrs=["bold"])
    ram_total = colored(human_readable_size(data['ram']['total']), "cyan")
    current_time = colored(data["current_time"], "white", attrs=["bold"])

    table.add_row(["Date & Time", current_time])
    table.add_row(["CPU Usage", cpu_usage])
    table.add_row(["RAM Usage", ram_usage])
    table.add_row(["RAM Total", ram_total])

    if data.get("has_gpu") and data.get("gpu"):
        gpu_data = data['gpu'][0]
        gpu_usage = colored(f"{gpu_data['load']:.1f}%", "blue", attrs=["bold"])
        gpu_vram = colored(f"{gpu_data['memoryPercent']:.1f}%", "magenta", attrs=["bold"])
        table.add_row(["GPU Usage", gpu_usage])
        table.add_row(["GPU VRAM Usage", gpu_vram])

    print(table)

def display_top_processes(data, key, title, terminal_width):
    """
    Display the top processes based on either CPU or memory usage.

    Args:
        data (dict): The fetched statistics data.
        key (str): The key to determine which processes to display ('top_cpu' or 'top_memory').
        title (str): The title to display for the processes table.
        terminal_width (int): The width of the terminal for formatting purposes.
    """
    print(colored(f"\n{title}:", "cyan", attrs=["bold", "underline"]))
    table = PrettyTable()
    if key == 'top_cpu':
        table.field_names = ["PID", "Name", "CPU %", "Cmdline"]
    elif key == 'top_memory':
        table.field_names = ["PID", "Name", "Memory Usage", "Memory %", "Cmdline"]

    # Calculate average space per column
    nb_columns = len(table.field_names)
    width_per_col = max((terminal_width - 4) // nb_columns, 5)

    table._max_width = {field: width_per_col for field in table.field_names}
    for field in table.field_names:
        table.align[field] = "l"

    for proc in data.get(key, []):
        if key == 'top_cpu':
            table.add_row([
                colored(proc['pid'], "cyan"),
                colored(proc['name'], "green"),
                colored(f"{proc['cpu_percent']:.1f}%", "yellow", attrs=["bold"]),
                colored(truncate_cmdline(proc['cmdline'], width_per_col), "white")
            ])
        elif key == 'top_memory':
            table.add_row([
                colored(proc['pid'], "cyan"),
                colored(proc['name'], "green"),
                colored(human_readable_size(proc['memory_usage']), "blue"),
                colored(f"{proc['memory_percent']:.1f}%", "yellow", attrs=["bold"]),
                colored(truncate_cmdline(proc['cmdline'], width_per_col), "white")
            ])

    print(table)

def display_gpu_stats(data, terminal_width):
    """
    Display GPU statistics in a tabular format.

    Args:
        data (dict): The fetched statistics data.
        terminal_width (int): The width of the terminal for formatting purposes.
    """
    if not data.get("has_gpu") or not data.get("gpu"):
        return

    print(colored("\nGPU Stats:", "cyan", attrs=["bold", "underline"]))
    table = PrettyTable()
    table.field_names = ["Name", "Load (%)", "VRAM", "VRAM (%)", "Fan (%)", "Power (W)", "Temp (°C)"]

    nb_columns = len(table.field_names)
    width_per_col = max((terminal_width - 4) // nb_columns, 5)
    table._max_width = {field: width_per_col for field in table.field_names}
    for field in table.field_names:
        table.align[field] = "l"

    for gpu in data['gpu']:
        table.add_row([
            colored(gpu['name'], "green", attrs=["bold"]),
            colored(f"{gpu['load']:.1f}%", "blue"),
            colored(f"{human_readable_size(gpu['memoryUsed'])}", "magenta"),
            colored(f"{gpu['memoryPercent']:.1f}%", "magenta"),
            colored(f"{gpu['fanSpeed']:.1f}%", "cyan"),
            colored(f"{gpu['powerDraw']:.1f} W", "yellow"),
            colored(f"{gpu['temperature']:.1f}°C", "red", attrs=["bold"])
        ])

    print(table)

def display_top_gpu_processes(data, terminal_width):
    """
    Display the top GPU processes (provided by 'top_gpu_processes') in a tabular format.

    Args:
        data (dict): The fetched statistics data.
        terminal_width (int): The width of the terminal for formatting purposes.
    """
    if not data.get("top_gpu_processes"):
        return

    print(colored("\nTop GPU Processes:", "cyan", attrs=["bold", "underline"]))
    table = PrettyTable()
    table.field_names = ["PID", "Name", "Memory Used", "Cmdline"]

    nb_columns = len(table.field_names)
    width_per_col = max((terminal_width - 4) // nb_columns, 5)
    table._max_width = {field: width_per_col for field in table.field_names}
    for field in table.field_names:
        table.align[field] = "l"

    for proc in data["top_gpu_processes"]:
        table.add_row([
            colored(proc['pid'], "cyan"),
            colored(proc['name'], "green"),
            colored(human_readable_size(proc['memory_used']), "blue"),
            colored(truncate_cmdline(proc['cmdline'], width_per_col), "white")
        ])

    print(table)

def display_ollama_stats(data, terminal_width):
    """
    Display statistics related to Ollama processes in a tabular format.

    Args:
        data (dict): The fetched statistics data.
        terminal_width (int): The width of the terminal for formatting purposes.
    """
    if not data.get("ollama_processes", {}).get("models"):
        return

    print(colored("\nOllama Stats:", "cyan", attrs=["bold", "underline"]))
    table = PrettyTable()
    table.field_names = ["Model Name", "Size", "VRAM Usage", "GPU Loaded", "Expiration"]

    nb_columns = len(table.field_names)
    width_per_col = max((terminal_width - 4) // nb_columns, 5)
    table._max_width = {field: width_per_col for field in table.field_names}
    for field in table.field_names:
        table.align[field] = "l"

    for model in data['ollama_processes']['models']:
        vram_str = human_readable_size(model['size_vram'])
        total_str = human_readable_size(model['size'])
        gpu_loaded_str = ""
        if model['size'] > 0:
            ratio = (model["size_vram"]/model['size'])*100
            gpu_loaded_str = colored(f"{ratio:.0f}% GPU", "red", attrs=["bold"])

        table.add_row([
            colored(model['name'], "green", attrs=["bold"]),
            colored(total_str, "blue"),
            colored(vram_str, "blue"),
            gpu_loaded_str,
            colored(time_until(model['expires_at']), "yellow")
        ])

    print(table)

def main():
    """
    Main function to execute the server stats dashboard CLI.
    """
    parser = argparse.ArgumentParser(description="CLI for Server Stats Dashboard")
    parser.add_argument("--url", type=str, default=SYS_STATS_API_URL, help="API URL for stats endpoint")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--oneline", action='store_true', help="Display stats on a single line with icons.")
    args = parser.parse_args()

    while True:
        clear_terminal()
        terminal_width, _ = shutil.get_terminal_size(fallback=(80, 20))  # Get terminal size
        stats = fetch_stats(args.url)
        if stats:
            if args.oneline:
                display_oneline_stats(stats)
            else:
                display_summary(stats, terminal_width)
                display_top_processes(stats, 'top_cpu', 'Top CPU Processes', terminal_width)
                display_top_processes(stats, 'top_memory', 'Top Memory Processes', terminal_width)
                display_gpu_stats(stats, terminal_width)
                display_top_gpu_processes(stats, terminal_width)
                display_ollama_stats(stats, terminal_width)

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
