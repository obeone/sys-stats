#!/usr/bin/env python3

import argparse
import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone

import readchar  # To capture key presses
import requests
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

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

# Rendering thresholds for the GPU detail panel.
# Past three side-by-side vertical tables the columns become too narrow to read
# on a standard terminal, so the panel falls back to one row per GPU.
MAX_SIDE_BY_SIDE_GPUS = 3
# Minimum number of columns a vertical GPU table needs to stay legible.
MIN_GPU_COLUMN_WIDTH = 26

# Style applied to the GPU process rows owned by Ollama, so its share of the
# VRAM stands out among the other compute apps.
OLLAMA_ROW_STYLE = "bold magenta"


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
    """Converts bytes into a human-readable format.

    ``TB`` is handled by the fallback rather than by the loop: a unit listed in
    the loop gets divided once more before the fallback is reached, which would
    under-report anything past a terabyte.
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
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
    """Create the static 2x2 grid backing the dashboard.

    The four regions are named by position rather than by content: which panel
    lands where depends on the number of GPUs, and is decided at render time by
    :func:`build_layout_content`. The structure itself never changes, so the
    ``Live`` loop keeps rendering the very same ``Layout`` object and never has
    to re-split it.

    Returns
    -------
    rich.layout.Layout
        Root layout holding ``top_left``, ``top_right``, ``bottom_left`` and
        ``bottom_right`` regions.
    """
    layout = Layout()

    # Two rows of equal height.
    layout.split(
        Layout(name="upper", ratio=1),
        Layout(name="lower", ratio=1)
    )

    # Upper row: a narrow left column and a wider right one.
    layout["upper"].split_row(
        Layout(name="top_left", ratio=1),
        Layout(name="top_right", ratio=2)
    )

    # Lower row: even split by default, adjusted per mode at render time.
    layout["lower"].split_row(
        Layout(name="bottom_left", ratio=1),
        Layout(name="bottom_right", ratio=1)
    )

    return layout


def gpu_memory_total_bytes(gpu_data):
    """Return the total VRAM of a GPU, in bytes.

    The ``/stats`` payload mixes units on purpose: ``memoryUsed`` is normalised
    to bytes server-side while ``memoryTotal`` is left in the MiB that GPUtil
    reports. This helper hides that asymmetry, and keeps working on payloads
    that predate ``memoryTotal`` by deriving the total from the used amount and
    its percentage.

    Parameters
    ----------
    gpu_data : dict
        A single entry of the ``gpu`` list returned by ``/stats``.

    Returns
    -------
    float
        Total VRAM in bytes, or ``0`` when neither source is usable.
    """
    memory_total_mib = gpu_data.get('memoryTotal') or 0
    if memory_total_mib:
        return float(memory_total_mib) * 1024 * 1024

    # Fallback for payloads without memoryTotal: used / percent gives the total.
    memory_used = gpu_data.get('memoryUsed') or 0
    memory_percent = gpu_data.get('memoryPercent') or 0
    if memory_percent > 0:
        return memory_used / (memory_percent / 100)

    return 0


def build_summary(data, interval, multi_gpu=False):
    """Build the summary panel with the clock, CPU, RAM and GPU figures.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.
    interval : int
        Current refresh interval in seconds, shown as the panel subtitle.
    multi_gpu : bool, optional
        When ``True`` the GPU section is condensed into cumulated totals
        (:func:`build_gpu_totals`) because the per-GPU detail is rendered in its
        own panel. When ``False`` the per-GPU tables are appended as before.

    Returns
    -------
    rich.panel.Panel
        The summary panel.
    """
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

    # Status (Paused or Running)
    with state_lock:
        status = "[bold red]PAUSED[/bold red]" if is_paused else ""

    table.add_row("", status)

    gpu_section = build_gpu_totals(data) if multi_gpu else build_gpu_summary(data)

    summary_panel = Panel(
        Group(table, gpu_section),
        border_style="cyan",
        padding=(0, 1),
        subtitle=f"Refresh rate: {interval}s"
    )
    return summary_panel


def build_single_gpu_table(gpu_data):
    """Build the vertical table describing one GPU.

    Parameters
    ----------
    gpu_data : dict
        A single entry of the ``gpu`` list returned by ``/stats``.

    Returns
    -------
    rich.table.Table
        A two-column table (label, value) titled with the GPU name.
    """
    gpu_name = truncate_name(gpu_data.get('name', 'N/A'), 25)
    gpu_load = gpu_data.get('load', 0)
    gpu_fan_speed = gpu_data.get('fanSpeed', '')
    gpu_power_draw = gpu_data.get('powerDraw', '')
    gpu_temperature = gpu_data.get('temperature', '')

    memory_used = gpu_data.get('memoryUsed', 0)
    memory_used_str = human_readable_size(memory_used)
    memory_total_str = human_readable_size(gpu_memory_total_bytes(gpu_data))
    # The reported percentage is kept as-is rather than recomputed: it is what
    # the driver claims, and it stays consistent with the web UI.
    vram_percent = gpu_data.get('memoryPercent', 0)

    table = Table(title=gpu_name, show_header=False, padding=(0, 1), expand=True)

    table.add_column(style='green')
    table.add_column()
    table.add_row('Total VRAM', memory_total_str)
    table.add_row('Power draw', f"{gpu_power_draw:.0f} W")
    table.add_row('Temperature', f"{gpu_temperature:.0f} °C")
    table.add_row('Fan speed', f"{gpu_fan_speed:.0f} %")
    table.add_row('VRAM used', f"{memory_used_str} ({vram_percent:.2f} %)")
    table.add_row('Utilization', f"{gpu_load:.1f} %")

    return table


def build_gpu_summary(data):
    """Build one vertical table per GPU, stacked.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.

    Returns
    -------
    rich.console.Group
        One :func:`build_single_gpu_table` per card, empty on a GPU-less host.
    """
    tables = []

    if data.get("has_gpu") and data.get("gpu"):
        for gpu_data in data['gpu']:
            tables.append(build_single_gpu_table(gpu_data))

    return Group(*tables)


def build_gpu_totals(data):
    """Build the cumulated GPU figures across every card.

    Used in multi-GPU mode, where listing each card in the summary panel would
    overflow the narrow region it lives in. Sums (VRAM, power) and means
    (utilization, fan) are computed here, while temperature is reported as the
    maximum: a single hot card is what matters, not the average.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.

    Returns
    -------
    rich.table.Table
        A two-column table (label, value). On a GPU-less host it only reports a
        count of zero, without ever dividing by it.
    """
    gpus = data.get('gpu') or []
    gpu_count = len(gpus)

    table = Table(title="GPU Totals", show_header=False, padding=(0, 1), expand=True)
    table.add_column(style='green')
    table.add_column()
    table.add_row('GPUs', str(gpu_count))

    if not gpu_count:
        return table

    memory_used = sum(gpu.get('memoryUsed') or 0 for gpu in gpus)
    memory_total = sum(gpu_memory_total_bytes(gpu) for gpu in gpus)
    # A driver that reports neither total nor percentage leaves memory_total at
    # zero, hence the guard before the ratio.
    memory_percent = (memory_used / memory_total * 100) if memory_total else 0
    mean_load = sum(gpu.get('load') or 0 for gpu in gpus) / gpu_count
    total_power = sum(gpu.get('powerDraw') or 0 for gpu in gpus)
    max_temperature = max((gpu.get('temperature') or 0) for gpu in gpus)
    mean_fan_speed = sum(gpu.get('fanSpeed') or 0 for gpu in gpus) / gpu_count

    table.add_row(
        'VRAM used',
        f"{human_readable_size(memory_used)} / {human_readable_size(memory_total)}"
        f" ({memory_percent:.1f} %)",
    )
    table.add_row('Utilization', f"{mean_load:.1f} % (mean)")
    table.add_row('Power draw', f"{total_power:.0f} W (total)")
    table.add_row('Temperature', f"{max_temperature:.0f} °C (max)")
    table.add_row('Fan speed', f"{mean_fan_speed:.0f} % (mean)")

    return table


def build_gpu_rows_table(gpus):
    """Build one wide table holding a single row per GPU.

    The horizontal fallback of :func:`build_gpu_detail_panel`, used when there
    are too many cards to show their vertical tables side by side.

    Parameters
    ----------
    gpus : list of dict
        The ``gpu`` list returned by ``/stats``.

    Returns
    -------
    rich.table.Table
        A table with the GPU / Name / Util / VRAM / % / Temp / Fan / Power
        columns.
    """
    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1), expand=True)
    table.add_column("GPU", style="cyan", justify="right", no_wrap=True, min_width=3)
    table.add_column("Name", style="green", min_width=8, overflow="ellipsis")
    table.add_column("Util", style="yellow", justify="right", no_wrap=True, min_width=5)
    table.add_column("VRAM", style="blue", justify="right", no_wrap=True, min_width=8)
    table.add_column("%", style="blue", justify="right", no_wrap=True, min_width=4)
    table.add_column("Temp", style="red", justify="right", no_wrap=True, min_width=5)
    table.add_column("Fan", style="white", justify="right", no_wrap=True, min_width=4)
    table.add_column("Power", style="white", justify="right", no_wrap=True, min_width=5)

    for position, gpu_data in enumerate(gpus):
        # ``id`` is the driver index; fall back to the position in the list when
        # a payload omits it, so the column is never blank.
        gpu_index = gpu_data.get('id', position)
        memory_used = gpu_data.get('memoryUsed') or 0
        table.add_row(
            str(gpu_index),
            truncate_name(gpu_data.get('name', 'N/A'), 20),
            f"{gpu_data.get('load') or 0:.0f} %",
            human_readable_size(memory_used),
            f"{gpu_data.get('memoryPercent') or 0:.0f} %",
            f"{gpu_data.get('temperature') or 0:.0f} °C",
            f"{gpu_data.get('fanSpeed') or 0:.0f} %",
            f"{gpu_data.get('powerDraw') or 0:.0f} W",
        )

    return table


def build_gpu_detail_panel(data, terminal_width=None):
    """Build the per-GPU detail panel used in multi-GPU mode.

    Two renderings, picked from the number of cards and the room available: up
    to :data:`MAX_SIDE_BY_SIDE_GPUS` vertical tables are laid out side by side,
    anything beyond that (or too narrow a terminal) falls back to one row per
    GPU.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.
    terminal_width : int, optional
        Width of the terminal in columns. ``None`` means unconstrained, in
        which case only the GPU count decides.

    Returns
    -------
    rich.panel.Panel
        The GPU detail panel.
    """
    gpus = data.get('gpu') or []
    if not data.get('has_gpu') or not gpus:
        return Panel(
            "No GPU detected.",
            title="[bold cyan]GPU Detail[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    # In multi-GPU mode this panel occupies roughly half of the screen width,
    # so that is the budget the vertical tables have to fit in.
    available_width = terminal_width // 2 if terminal_width else None
    side_by_side = len(gpus) <= MAX_SIDE_BY_SIDE_GPUS and (
        available_width is None or available_width >= len(gpus) * MIN_GPU_COLUMN_WIDTH
    )

    if side_by_side:
        grid = Table.grid(expand=True)
        for _ in gpus:
            grid.add_column(ratio=1)
        grid.add_row(*[build_single_gpu_table(gpu_data) for gpu_data in gpus])
        content = grid
    else:
        content = build_gpu_rows_table(gpus)

    return Panel(
        content,
        title="[bold cyan]GPU Detail[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_process_table(processes, key, title):
    """Build a table for the most resource-intensive processes.

    Parameters
    ----------
    processes : list of dict
        Entries of ``top_cpu`` or ``top_memory``.
    key : str
        Either ``"top_cpu"`` or ``"top_memory"``; selects the metric column.
    title : str
        Human readable name of the ranking, used by the empty-state panel.

    Returns
    -------
    rich.table.Table or rich.panel.Panel
        The ranking table, or an explanatory panel when there is no data.
    """
    if not processes:
        return Panel(
            f"No data for {title}.",
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    # Columns are elastic: fixed widths clipped everything as soon as a region
    # narrowed down, so only a floor is imposed and Rich shares out the rest.
    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    if key == 'top_cpu':
        table.add_column("PID", style="cyan", no_wrap=True, min_width=5)
        table.add_column("Name", style="green", min_width=8, overflow="ellipsis")
        table.add_column("CPU%", style="yellow", justify="right", no_wrap=True, min_width=5)
        table.add_column("Cmdline", style="white", max_width=20, overflow="ellipsis")
    elif key == 'top_memory':
        table.add_column("PID", style="cyan", no_wrap=True, min_width=5)
        table.add_column("Name", style="green", min_width=8, overflow="ellipsis")
        table.add_column("Memory%", style="blue", justify="right", no_wrap=True, min_width=5)
        table.add_column("Cmdline", style="white", max_width=20, overflow="ellipsis")

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
    """Build the panel holding the Top CPU and Top Memory rankings.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.

    Returns
    -------
    rich.table.Table
        A two-column grid with one panel per ranking.
    """
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


def build_gpu_processes_panel(data, multi_gpu=False):
    """Build the panel listing the processes holding VRAM.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.
    multi_gpu : bool, optional
        When ``True`` a ``GPU`` column is prepended with the index of the card
        each process runs on. On a single-GPU host that column would only
        repeat ``0`` on every row, so it is left out.

    Returns
    -------
    rich.panel.Panel
        The GPU processes panel, or an empty-state panel.
    """
    processes = data.get("top_gpu_processes", [])
    if not processes:
        return Panel(
            "No GPU processes.",
            title="[bold cyan]GPU Processes[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    # Padding is collapsed and the edges dropped: this table lives in a third of
    # the screen and the extra column has to come from somewhere.
    table = Table(
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
        collapse_padding=True,
        pad_edge=False,
    )
    if multi_gpu:
        table.add_column("GPU", style="cyan", justify="right", no_wrap=True, min_width=3)
    table.add_column("PID", style="cyan", no_wrap=True, min_width=4)
    table.add_column("Name", style="green", min_width=6, overflow="ellipsis")
    table.add_column("Memory Used", style="blue", justify="right", no_wrap=True, min_width=5)
    table.add_column("Cmdline", style="white", max_width=20, overflow="ellipsis")

    for proc in processes:
        pid = str(proc.get('pid', 'N/A'))
        name = proc.get('name', 'N/A')
        memory_used = human_readable_size(proc.get('memory_used', 0))
        cmdline = truncate_cmdline(proc.get('cmdline', ''), 20)
        cells = [pid, truncate_name(name), memory_used, cmdline]
        if multi_gpu:
            # A driver too old to report gpu_uuid leaves the index unresolved.
            gpu_index = proc.get('gpu_index')
            cells.insert(0, "?" if gpu_index is None else str(gpu_index))
        # Ollama's own workers are highlighted so its share stands out.
        row_style = OLLAMA_ROW_STYLE if name == "ollama" else None
        table.add_row(*cells, style=row_style)

    return Panel(
        table,
        title="[bold cyan]GPU Processes[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def format_context_length(value):
    """Format an Ollama context length for display.

    Exact multiples of 1024 (and of 1024²) are abbreviated, since that is how
    context windows are usually quoted; anything else is shown raw rather than
    rounded, because a truncated context length is misleading.

    Parameters
    ----------
    value : int or None
        The ``context_length`` reported by ``/api/ps``, or ``None`` when the
        Ollama server is too old to expose it.

    Returns
    -------
    str
        ``"1M"``, ``"32K"``, the raw integer, or ``"N/A"``.
    """
    if value is None:
        return "N/A"

    try:
        context_length = int(value)
    except (TypeError, ValueError):
        return "N/A"

    if context_length > 0:
        if context_length % (1024 * 1024) == 0:
            return f"{context_length // (1024 * 1024)}M"
        if context_length % 1024 == 0:
            return f"{context_length // 1024}K"

    return str(context_length)


def ollama_gpu_indices(gpu_processes):
    """Collect the GPU indices Ollama currently holds VRAM on.

    Parameters
    ----------
    gpu_processes : list of dict or None
        Entries of ``top_gpu_processes``.

    Returns
    -------
    tuple of (bool, str)
        Whether an Ollama process was found at all, and the label to display:
        the sorted, comma-joined indices, or ``"-"`` when none could be
        resolved.
    """
    processes = gpu_processes or []
    ollama_processes = [proc for proc in processes if proc.get('name') == "ollama"]
    if not ollama_processes:
        return False, "-"

    indices = sorted({
        proc['gpu_index'] for proc in ollama_processes if proc.get('gpu_index') is not None
    })
    return True, (",".join(str(index) for index in indices) if indices else "-")


def build_ollama_panel(data, gpu_processes=None):
    """Build the panel listing the models Ollama keeps loaded.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.
    gpu_processes : list of dict, optional
        Entries of ``top_gpu_processes``, used to fill the ``GPU`` column.

    Returns
    -------
    rich.panel.Panel
        The Ollama panel, or an empty-state panel.

    Notes
    -----
    The ``GPU`` column is a process-level heuristic, not a per-model truth.
    Ollama serves N models from a single OS process and ``/api/ps`` exposes no
    per-model PID, so the only thing that can be correlated is "the Ollama
    process holds VRAM on these cards". Every row therefore shows the same
    indices, even when the models are actually spread across different GPUs.
    """
    models = data.get("ollama_processes", {}).get("models", [])
    if not models:
        return Panel(
            "No Ollama models.",
            title="[bold cyan]Ollama Statistics[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    has_ollama_process, gpu_label = ollama_gpu_indices(gpu_processes)

    # Seven columns in a third of the screen leave no room for cell padding: with
    # it the table overflows its region and Rich clips the rightmost columns off.
    # The box lines still separate the values, and every floor is kept low so the
    # shrinking happens inside the cells rather than at the panel border.
    table = Table(
        show_header=True,
        header_style="bold magenta",
        padding=(0, 0),
        pad_edge=False,
    )
    table.add_column("Model", style="green", min_width=8, overflow="ellipsis")
    if has_ollama_process:
        table.add_column("GPU", style="cyan", justify="right", no_wrap=True, min_width=3)
    table.add_column("Ctx", style="magenta", justify="right", no_wrap=True, min_width=3)
    table.add_column("Size", style="blue", justify="right", no_wrap=True, min_width=5)
    table.add_column("VRAM", style="blue", justify="right", no_wrap=True, min_width=5)
    table.add_column("GPU%", style="red", justify="right", no_wrap=True, min_width=4)
    table.add_column("Expires", style="yellow", justify="right", no_wrap=True, min_width=5)

    for model in models:
        model_name = truncate_name(model.get('name', 'N/A'))
        context_length = format_context_length(model.get('context_length'))
        size_total = human_readable_size(model.get('size', 0))
        size_vram = human_readable_size(model.get('size_vram', 0))
        gpu_loaded_ratio = (model.get("size_vram", 0) / model.get('size', 1)) * 100 if model.get('size', 1) > 0 else 0
        gpu_loaded_str = f"{gpu_loaded_ratio:.0f}%"
        expiration = time_until(model.get('expires_at', ''))
        cells = [
            model_name,
            context_length,
            size_total,
            size_vram,
            gpu_loaded_str,
            expiration,
        ]
        if has_ollama_process:
            cells.insert(1, gpu_label)
        table.add_row(*cells)

    return Panel(
        table,
        title="[bold cyan]Ollama Statistics[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_layout_content(layout, data, interval, terminal_width=None):
    """Fill the four layout regions with the fetched data.

    The routing depends on the number of GPUs. With zero or one card the
    per-GPU detail fits inside the summary panel, so the layout stays as it
    always was. From two cards on, the detail moves to its own region, the GPU
    processes join Ollama on the left (they describe the same VRAM), and the
    summary keeps only cumulated figures.

    Parameters
    ----------
    layout : rich.layout.Layout
        The layout built by :func:`create_layout`; updated in place.
    data : dict
        The ``/stats`` payload.
    interval : int
        Current refresh interval in seconds.
    terminal_width : int, optional
        Width of the terminal in columns, forwarded to
        :func:`build_gpu_detail_panel` to pick its rendering.

    Returns
    -------
    None
    """
    gpus = data.get("gpu") or []
    multi_gpu = bool(data.get("has_gpu")) and len(gpus) > 1

    gpu_processes = data.get("top_gpu_processes") or []
    ollama_panel = build_ollama_panel(data, gpu_processes=gpu_processes)
    gpu_processes_panel = build_gpu_processes_panel(data, multi_gpu=multi_gpu)

    # Both modes keep the process rankings top right, they need the width.
    layout["top_right"].update(build_processes_panel(data))

    if multi_gpu:
        # The left column carries two stacked panels, hence a bit more room.
        layout["top_left"].ratio = 2
        layout["top_right"].ratio = 3
        layout["bottom_left"].ratio = 1
        layout["bottom_right"].ratio = 2
        layout["top_left"].update(Group(ollama_panel, gpu_processes_panel))
        layout["bottom_left"].update(build_summary(data, interval, multi_gpu=True))
        layout["bottom_right"].update(build_gpu_detail_panel(data, terminal_width))
    else:
        layout["top_left"].ratio = 1
        layout["top_right"].ratio = 2
        layout["bottom_left"].ratio = 1
        layout["bottom_right"].ratio = 1
        layout["top_left"].update(ollama_panel)
        layout["bottom_left"].update(gpu_processes_panel)
        layout["bottom_right"].update(build_summary(data, interval))


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
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (can be adjusted with '+' and '-')",
    )
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
                        build_layout_content(
                            layout,
                            latest_stats,
                            current_interval,
                            terminal_width=shutil.get_terminal_size(fallback=(80, 20)).columns,
                        )

            # Check if a layout rebuild is required
            if rebuild_layout_event.is_set():
                if help_flag:
                    help_panel = build_full_screen_help()
                    layout.update(help_panel)
                elif not paused and latest_stats:
                    build_layout_content(
                        layout,
                        latest_stats,
                        current_interval,
                        terminal_width=shutil.get_terminal_size(fallback=(80, 24)).columns,
                    )
                elif paused and latest_stats:
                    build_layout_content(
                        layout,
                        latest_stats,
                        current_interval,
                        terminal_width=shutil.get_terminal_size(fallback=(80, 20)).columns,
                    )
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
