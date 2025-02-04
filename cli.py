#!/usr/bin/env python3

import requests
import time
import argparse
import shutil
import os
import threading
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import readchar  # To capture key presses

# Default API URL
SYS_STATS_API_URL = os.getenv('SYS_STATS_API_URL', 'http://localhost:5000/stats')

console = Console()

# Dashboard states
is_paused = False
show_help_flag = False
refresh_interval = 5  # Default refresh interval in seconds

# Locks for thread-safe operations
state_lock = threading.Lock()
stats_lock = threading.Lock()

# Events for synchronization
exit_event = threading.Event()
rebuild_layout_event = threading.Event()

# Shared variable to store the latest statistics
latest_stats = None


def fetch_stats(api_url):
    """Fetch statistics from the specified API URL."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error fetching stats:[/bold red] {e}")
        return None


def human_readable_size(size):
    """Converts bytes into a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def time_until(expiration):
    """Calculates the remaining time until a given expiration date."""
    try:
        exp_time = datetime.fromisoformat(expiration.replace('Z', '+00:00')).astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        delta = exp_time - now
        if delta.total_seconds() <= 0:
            return "[bold red]Expired[/bold red]"
        return str(timedelta(seconds=int(delta.total_seconds())))
    except Exception:
        return "N/A"


def truncate_cmdline(cmdline, width):
    """Truncates the command line string if it exceeds a specified width."""
    return cmdline if len(cmdline) <= width else cmdline[:width - 1] + "…"


def truncate_name(name, max_length=15):
    """Truncates the name if it exceeds a specified maximum length."""
    return name if len(name) <= max_length else name[:max_length - 1] + "…"


def create_layout():
    """Creates the initial layout for the dashboard."""
    layout = Layout()

    # Split the main layout into header and body
    layout.split(
        Layout(name="header", size=1),
        Layout(name="body")
    )

    # Split the body into upper and lower sections
    layout["body"].split(
        Layout(name="upper", ratio=1),
        Layout(name="lower", ratio=1)
    )

    # Split the upper section into summary and processes
    layout["upper"].split_row(
        Layout(name="summary", ratio=1),
        Layout(name="processes", ratio=2)
    )

    # Split the lower section into GPU processes and Ollama statistics
    layout["lower"].split_row(
        Layout(name="gpu_processes", ratio=1),
        Layout(name="ollama", ratio=1)
    )

    return layout


def build_header():
    """Builds the header panel."""
    header_text = Text("🖥️ Sys Stats", style="bold white on blue")
    return Panel(header_text, height=1, style="blue", padding=(0, 2))


def build_summary(data, interval):
    """Builds the summary panel showing CPU, RAM, and GPU usage."""
    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_column(justify="right")

    current_time = f"[bold]{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}[/bold]"

    table.add_row("Current Time", current_time)
    table.add_row("", "")

    # CPU and RAM information
    cpu_usage = f"[bold green]CPU:[/bold green] {data.get('cpu', 0):.1f}%"
    ram_percent = data.get('ram', {}).get('percent', 0)
    ram_total = human_readable_size(data.get('ram', {}).get('total', 0))
    ram_usage = f"[bold yellow]RAM:[/bold yellow] {ram_percent:.1f}% / {ram_total}"
    table.add_row(cpu_usage, ram_usage)

    # GPU information
    if data.get("has_gpu") and data.get("gpu"):
        table.add_row('','')

        gpu_data = data['gpu'][0]
        gpu_name = truncate_name(gpu_data.get('name', 'N/A'), 15)
        gpu_load = gpu_data.get('load', 0)
        gpu_fan_speed = gpu_data.get('fanSpeed', '')
        gpu_power_draw = gpu_data.get('powerDraw', '')

        # Total VRAM calculation
        memory_used = gpu_data.get('memoryUsed', 0)
        memory_percent = gpu_data.get('memoryPercent', 0)
        if memory_percent > 0:
            memory_total = memory_used / (memory_percent / 100)
        else:
            memory_total = 0
        memory_used_str = human_readable_size(memory_used)
        memory_total_str = human_readable_size(memory_total)
        vram_percent = memory_percent

        gpu_title = f"[bold blue]GPU:[/bold blue] {gpu_name} ({memory_total_str})"
        table.add_row(gpu_title)

        gpu_fan = f"[bold blue]Fan:[/bold blue] {gpu_fan_speed}%"
        gpu_power = f"[bold blue]Power draw:[/bold blue] {gpu_power_draw}W"
        table.add_row(gpu_fan, gpu_power)

        gpu_vram = f"[bold blue]VRAM:[/bold blue] {memory_used_str} ({vram_percent:.1f}%)"
        gpu_load_str = f"[bold blue]Load:[/bold blue] {gpu_load:.1f}%"
        table.add_row(gpu_load_str, gpu_vram)

    # Status (Paused or Running)
    with state_lock:
        status = "[bold red]PAUSED[/bold red]" if is_paused else ""

    table.add_row("", status)

    summary_panel = Panel(
        table,
        border_style="cyan",
        padding=(0, 1),
        subtitle=f"Refresh rate: {interval}s"
    )
    return summary_panel


def build_process_table(processes, key, title):
    """Builds a table for the most resource-intensive processes."""
    if not processes:
        return Panel(
            f"No data for {title}.",
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    if key == 'top_cpu':
        table.add_column("PID", style="cyan", no_wrap=True, width=6)
        table.add_column("Name", style="green", width=15)
        table.add_column("CPU%", style="yellow", justify="right", width=6)
        table.add_column("Cmdline", style="white", max_width=20)
    elif key == 'top_memory':
        table.add_column("PID", style="cyan", no_wrap=True, width=6)
        table.add_column("Name", style="green", width=15)
        table.add_column("Memory%", style="blue", justify="right", width=6)
        table.add_column("Cmdline", style="white", max_width=20)

    for proc in processes:
        pid = str(proc.get('pid', 'N/A'))
        name = truncate_name(proc.get('name', 'N/A'))
        cmdline = truncate_cmdline(proc.get('cmdline', ''), 20)
        if key == 'top_cpu':
            cpu_percent = f"{proc.get('cpu_percent', 0):.1f}%"
            table.add_row(pid, name, cpu_percent, cmdline)
        elif key == 'top_memory':
            mem_percent = f"{proc.get('memory_percent', 0):.1f}%"
            table.add_row(pid, name, mem_percent, cmdline)

    return table


def build_processes_panel(data):
    """Builds the panel for Top CPU and Top Memory processes."""
    top_cpu = data.get('top_cpu', [])
    top_memory = data.get('top_memory', [])

    table_cpu = build_process_table(top_cpu, 'top_cpu', 'CPU')
    table_mem = build_process_table(top_memory, 'top_memory', 'Memory')

    processes_table = Table.grid(expand=True)
    processes_table.add_column()
    processes_table.add_column()

    processes_table.add_row(
        Panel(table_cpu, title="[bold cyan]Top CPU[/bold cyan]", border_style="cyan", padding=(0, 1)),
        Panel(table_mem, title="[bold cyan]Top Memory[/bold cyan]", border_style="cyan", padding=(0, 1))
    )

    return processes_table


def build_gpu_processes_panel(data):
    """Builds the panel for GPU processes."""
    processes = data.get("top_gpu_processes", [])
    if not processes:
        return Panel(
            "No GPU processes.",
            title="[bold cyan]GPU Processes[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    table.add_column("PID", style="cyan", no_wrap=True, width=6)
    table.add_column("Name", style="green", width=15)
    table.add_column("Memory Used", style="blue", justify="right", width=10)
    table.add_column("Cmdline", style="white", max_width=20)

    for proc in processes:
        pid = str(proc.get('pid', 'N/A'))
        name = truncate_name(proc.get('name', 'N/A'))
        memory_used = human_readable_size(proc.get('memory_used', 0))
        cmdline = truncate_cmdline(proc.get('cmdline', ''), 20)
        table.add_row(pid, name, memory_used, cmdline)

    return Panel(
        table,
        title="[bold cyan]GPU Processes[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_ollama_panel(data):
    """Builds the panel for Ollama statistics."""
    models = data.get("ollama_processes", {}).get("models", [])
    if not models:
        return Panel(
            "No Ollama models.",
            title="[bold cyan]Ollama Statistics[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    table.add_column("Model", style="green", width=15)
    table.add_column("Size", style="blue", justify="right", width=10)
    table.add_column("VRAM", style="blue", justify="right", width=10)
    table.add_column("GPU%", style="red", justify="right", width=6)
    table.add_column("Expires", style="yellow", justify="right", width=12)

    for model in models:
        model_name = truncate_name(model.get('name', 'N/A'))
        size_total = human_readable_size(model.get('size', 0))
        size_vram = human_readable_size(model.get('size_vram', 0))
        gpu_loaded_ratio = (model.get("size_vram", 0) / model.get('size', 1)) * 100 if model.get('size', 1) > 0 else 0
        gpu_loaded_str = f"{gpu_loaded_ratio:.0f}%"
        expiration = time_until(model.get('expires_at', ''))
        table.add_row(
            model_name,
            size_total,
            size_vram,
            gpu_loaded_str,
            expiration
        )

    return Panel(
        table,
        title="[bold cyan]Ollama Statistics[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_layout_content(layout, data, interval, terminal_width):
    """Fills the layout with the fetched data."""
    # Update header
    layout["header"].update(build_header())

    # Update summary
    summary_panel = build_summary(data, interval)
    layout["upper"]["summary"].update(summary_panel)

    # Update processes (Top CPU and Top Memory)
    processes_table = build_processes_panel(data)
    layout["upper"]["processes"].update(processes_table)

    # Update GPU processes
    gpu_processes_panel = build_gpu_processes_panel(data)
    layout["lower"]["gpu_processes"].update(gpu_processes_panel)

    # Update Ollama statistics
    ollama_panel = build_ollama_panel(data)
    layout["lower"]["ollama"].update(ollama_panel)


def build_full_screen_help():
    """Builds the full-screen help panel."""
    help_text = """
[bold yellow]Keyboard Shortcuts:[/bold yellow]

[bold green]q[/bold green] - Quit
[bold green]r[/bold green] - Refresh
[bold green]h[/bold green] - Show/Hide help
[bold green]p[/bold green] - Pause/Resume
[bold green]-[/bold green] - Decrease interval
[bold green]+[/bold green] - Increase interval

Press [bold green]h[/bold green] again to return.
"""

    help_panel = Panel.fit(
        help_text,
        title="[bold cyan]Help[/bold cyan]",
        border_style="green",
        padding=(1, 2)
    )
    return help_panel


def keyboard_listener():
    """Listens for keyboard input and modifies state accordingly."""
    global is_paused, show_help_flag, refresh_interval, latest_stats
    while not exit_event.is_set():
        key = readchar.readkey()
        with state_lock:
            if key.lower() == 'q':
                exit_event.set()
            elif key.lower() == 'r':
                # Signal to refresh data
                rebuild_layout_event.set()  # We want to rebuild the layout with the same data
            elif key.lower() == 'h':
                show_help_flag = not show_help_flag
                rebuild_layout_event.set()  # Rebuild layout to show/hide help
            elif key.lower() == 'p':
                is_paused = not is_paused
                rebuild_layout_event.set()  # Rebuild layout to show pause state
            elif key == '-':
                if refresh_interval > 1:
                    refresh_interval -= 1
                    rebuild_layout_event.set()  # Rebuild layout to show new interval
            elif key == '+':
                if refresh_interval < 60:
                    refresh_interval += 1
                    rebuild_layout_event.set()  # Rebuild layout to show new interval


def main():
    """Main function to run the server statistics CLI dashboard."""
    global latest_stats
    parser = argparse.ArgumentParser(description="CLI for server statistics dashboard")
    parser.add_argument("--url", type=str, default=SYS_STATS_API_URL, help="API URL for the statistics")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds (can be adjusted with '+' and '-')")
    args = parser.parse_args()

    global refresh_interval
    with state_lock:
        refresh_interval = args.interval

    layout = create_layout()

    # Start keyboard listener thread
    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    with Live(layout, refresh_per_second=4, screen=True):
        while not exit_event.is_set():
            with state_lock:
                current_interval = refresh_interval
                paused = is_paused
                help_flag = show_help_flag

            if help_flag:
                # Show the help panel in full screen
                help_panel = build_full_screen_help()
                layout.update(help_panel)
            else:
                if not paused:
                    # Fetch statistics if not paused
                    stats = fetch_stats(args.url)
                    if stats:
                        with stats_lock:
                            latest_stats = stats
                        terminal_size = shutil.get_terminal_size(fallback=(80, 24))
                        terminal_width = terminal_size.columns
                        build_layout_content(layout, latest_stats, current_interval, terminal_width)
                else:
                    # If paused, only rebuild the layout with the latest data
                    if latest_stats:
                        build_layout_content(layout, latest_stats, current_interval, terminal_width=shutil.get_terminal_size(fallback=(80, 20)).columns)

            # Check if a layout rebuild is required
            if rebuild_layout_event.is_set():
                if help_flag:
                    help_panel = build_full_screen_help()
                    layout.update(help_panel)
                elif not paused and latest_stats:
                    build_layout_content(layout, latest_stats, current_interval, terminal_width=shutil.get_terminal_size(fallback=(80, 24)).columns)
                elif paused and latest_stats:
                    build_layout_content(layout, latest_stats, current_interval, terminal_width=shutil.get_terminal_size(fallback=(80, 20)).columns)
                rebuild_layout_event.clear()

            # Wait for the refresh interval or an event
            sleep_time = current_interval
            start_time = time.time()
            while (time.time() - start_time) < sleep_time:
                if exit_event.is_set() or rebuild_layout_event.is_set():
                    break
                time.sleep(0.1)  # Wait in 100ms slices to react quickly to events

    console.print("[bold red]Closing the dashboard...[/bold red]")


if __name__ == "__main__":
    main()
