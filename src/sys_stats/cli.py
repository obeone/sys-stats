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

# Rendering thresholds for the panels stacked in the left column of the
# multi-GPU layout (Ollama and GPU Processes): below these available widths
# the least valuable column is dropped rather than squeezed into
# illegibility. Rich shares out space among flexible columns by their
# measured content width, not their ``min_width``, so once a table is asked
# to fit into less than the sum of its columns' minimums it can starve some
# of them down to zero rather than shrinking every column proportionally.
# Dropping a column outright is more reliable than fighting that algorithm.
OLLAMA_VRAM_COLUMN_MIN_WIDTH = 30
OLLAMA_SIZE_COLUMN_MIN_WIDTH = 36
OLLAMA_GPU_PERCENT_COLUMN_MIN_WIDTH = 42
OLLAMA_EXPIRES_COLUMN_MIN_WIDTH = 50
GPU_PROCESSES_CMDLINE_COLUMN_MIN_WIDTH = 36
# Below GPU_PROCESSES_CMDLINE_COLUMN_MIN_WIDTH (Cmdline already dropped), the
# GPU Processes table still overflows its panel because ``Name`` and
# ``Memory Used`` naturally want more room than is left: their headers are
# wider than a single value. These caps force Rich to actually shrink them
# instead of reverting to their natural content width (see
# :func:`build_gpu_processes_panel`).
GPU_PROCESSES_NAME_COLUMN_MAX_WIDTH = 4
GPU_PROCESSES_MEMORY_COLUMN_MAX_WIDTH = 8

# Rendering thresholds for the GPU Detail panel's row-per-GPU fallback
# (:func:`build_gpu_rows_table`). Columns are dropped in ascending order of
# importance as the estimated available width shrinks: ``%`` first (not
# ranked among the named columns below), then ``Name``, ``Fan``, ``Power``,
# ``Temp`` and finally ``Util``. ``GPU`` and ``VRAM`` are never dropped.
GPU_DETAIL_PERCENT_COLUMN_MIN_WIDTH = 76
GPU_DETAIL_NAME_COLUMN_MIN_WIDTH = 70
GPU_DETAIL_FAN_COLUMN_MIN_WIDTH = 50
GPU_DETAIL_POWER_COLUMN_MIN_WIDTH = 43
GPU_DETAIL_TEMP_COLUMN_MIN_WIDTH = 36
GPU_DETAIL_UTIL_COLUMN_MIN_WIDTH = 28

# Rendering thresholds for the Top CPU / Top Memory process tables
# (:func:`build_process_table`). Columns are dropped in ascending order of
# importance: ``Cmdline`` first, then the metric (``CPU%``/``Memory%``), then
# ``PID``. ``Name`` is never dropped, it is the whole point of the panel.
PROCESS_TABLE_CMDLINE_COLUMN_MIN_WIDTH = 55
PROCESS_TABLE_METRIC_COLUMN_MIN_WIDTH = 30
PROCESS_TABLE_PID_COLUMN_MIN_WIDTH = 24

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


def left_column_available_width(terminal_width):
    """Estimate the width available to the Ollama and GPU Processes panels.

    Both panels live in the layout's left column: on its own in mono-GPU
    mode (roughly a third of the terminal, ``top_left`` ratio 1 against
    ``top_right`` ratio 2) or stacked with each other in multi-GPU mode
    (roughly two fifths, ratio 2 against 3). A third of the terminal width is
    used as a single, slightly conservative estimate for both modes: it
    matches the mono-GPU case closely and under-estimates the multi-GPU one,
    which only means columns are dropped a little earlier than strictly
    necessary there, never later.

    Parameters
    ----------
    terminal_width : int or None
        Width of the terminal in columns, or ``None`` when unconstrained.

    Returns
    -------
    int or None
        The estimated available width, or ``None`` when ``terminal_width``
        is ``None``, in which case callers should treat every column as
        fitting.
    """
    if terminal_width is None:
        return None
    return terminal_width // 3


def bottom_right_column_available_width(terminal_width):
    """Estimate the width available to the GPU Detail panel's row fallback.

    Unlike the Ollama and GPU Processes panels, the GPU Detail panel lives in
    ``bottom_right``, whose ratio against ``bottom_left`` is 2:1 in multi-GPU
    mode (the only mode that reaches the row-per-GPU fallback, see
    :data:`MAX_SIDE_BY_SIDE_GPUS`), so it gets roughly two thirds of the
    terminal rather than a third. Reusing
    :func:`left_column_available_width` here would under-estimate the room
    available by about half, dropping columns that would actually fit.

    Parameters
    ----------
    terminal_width : int or None
        Width of the terminal in columns, or ``None`` when unconstrained.

    Returns
    -------
    int or None
        The estimated available width, or ``None`` when ``terminal_width``
        is ``None``, in which case callers should treat every column as
        fitting.
    """
    if terminal_width is None:
        return None
    return terminal_width * 2 // 3


def process_table_available_width(terminal_width):
    """Estimate the width available to each Top CPU / Top Memory table.

    ``top_right`` gets a ratio of 2 against ``top_left``'s 1 in mono-GPU mode
    and 3 against 2 in multi-GPU mode (roughly two thirds and three fifths of
    the terminal, respectively), and :func:`build_processes_panel` then
    splits that region into two side-by-side tables, so each one only gets
    about half of it. The smaller of the two ratios, three fifths, is used as
    a single, slightly conservative estimate that covers both modes: it
    matches the multi-GPU case closely and under-estimates the mono-GPU one,
    which only means columns are dropped a little earlier than strictly
    necessary there, never later.

    Parameters
    ----------
    terminal_width : int or None
        Width of the terminal in columns, or ``None`` when unconstrained.

    Returns
    -------
    int or None
        The estimated available width, or ``None`` when ``terminal_width``
        is ``None``, in which case callers should treat every column as
        fitting.
    """
    if terminal_width is None:
        return None
    return (terminal_width * 3 // 5) // 2


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


def build_gpu_rows_table(gpus, terminal_width=None):
    """Build one wide table holding a single row per GPU.

    The horizontal fallback of :func:`build_gpu_detail_panel`, used when there
    are too many cards to show their vertical tables side by side.

    Parameters
    ----------
    gpus : list of dict
        The ``gpu`` list returned by ``/stats``.
    terminal_width : int, optional
        Width of the terminal in columns. ``None`` means unconstrained, in
        which case every column is shown. Otherwise the ``%``, ``Name``,
        ``Fan``, ``Power``, ``Temp`` and ``Util`` columns are dropped one by
        one, in that order (least important first), as the width estimated
        through :func:`bottom_right_column_available_width` drops below
        :data:`GPU_DETAIL_PERCENT_COLUMN_MIN_WIDTH`,
        :data:`GPU_DETAIL_NAME_COLUMN_MIN_WIDTH`,
        :data:`GPU_DETAIL_FAN_COLUMN_MIN_WIDTH`,
        :data:`GPU_DETAIL_POWER_COLUMN_MIN_WIDTH`,
        :data:`GPU_DETAIL_TEMP_COLUMN_MIN_WIDTH` and
        :data:`GPU_DETAIL_UTIL_COLUMN_MIN_WIDTH`. ``GPU`` and ``VRAM`` are
        never dropped: they carry the values this panel exists to show.

    Returns
    -------
    rich.table.Table
        A table with as many of the GPU / Name / Util / VRAM / % / Temp /
        Fan / Power columns as fit.
    """
    available_width = bottom_right_column_available_width(terminal_width)
    show_percent = available_width is None or available_width >= GPU_DETAIL_PERCENT_COLUMN_MIN_WIDTH
    show_name = available_width is None or available_width >= GPU_DETAIL_NAME_COLUMN_MIN_WIDTH
    show_fan = available_width is None or available_width >= GPU_DETAIL_FAN_COLUMN_MIN_WIDTH
    show_power = available_width is None or available_width >= GPU_DETAIL_POWER_COLUMN_MIN_WIDTH
    show_temp = available_width is None or available_width >= GPU_DETAIL_TEMP_COLUMN_MIN_WIDTH
    show_util = available_width is None or available_width >= GPU_DETAIL_UTIL_COLUMN_MIN_WIDTH

    # Padding is collapsed and the edges dropped, and every column is
    # ``no_wrap``, for the same reasons as the GPU Processes table: without
    # them a starved column wraps its text onto extra lines rather than
    # truncating it, stretching the row and leaving the other cells looking
    # like blank rows underneath it.
    table = Table(
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
        collapse_padding=True,
        pad_edge=False,
    )
    # Every numeric/name column also gets an explicit ``max_width``. Without
    # one, Rich measures a column's *natural* content width (e.g. the full
    # GPU name before ellipsis truncation) to decide its initial width, and
    # that natural width is what competes for space; if the columns'
    # combined natural widths overflow the panel, Rich's last-resort shrink
    # kicks in and can crush a whole trailing column to zero width or clip a
    # value's unit off, regardless of ``min_width``/``no_wrap``. Capping
    # every column keeps the combined natural width predictable so that
    # reaching this fallback path is avoided in the first place.
    table.add_column("GPU", style="cyan", justify="right", no_wrap=True, min_width=3, max_width=3)
    if show_name:
        table.add_column(
            "Name", style="green", min_width=8, max_width=18, overflow="ellipsis", no_wrap=True
        )
    if show_util:
        table.add_column(
            "Util", style="yellow", justify="right", no_wrap=True, min_width=5, max_width=5,
            overflow="ellipsis",
        )
    table.add_column(
        "VRAM", style="blue", justify="right", no_wrap=True, min_width=8, max_width=8,
        overflow="ellipsis",
    )
    if show_percent:
        table.add_column(
            "%", style="blue", justify="right", no_wrap=True, min_width=4, max_width=4,
            overflow="ellipsis",
        )
    if show_temp:
        table.add_column(
            "Temp", style="red", justify="right", no_wrap=True, min_width=6, max_width=6,
            overflow="ellipsis",
        )
    if show_fan:
        table.add_column(
            "Fan", style="white", justify="right", no_wrap=True, min_width=5, max_width=5,
            overflow="ellipsis",
        )
    if show_power:
        table.add_column(
            "Power", style="white", justify="right", no_wrap=True, min_width=5, max_width=5,
            overflow="ellipsis",
        )

    for position, gpu_data in enumerate(gpus):
        # ``id`` is the driver index; fall back to the position in the list when
        # a payload omits it, so the column is never blank.
        gpu_index = gpu_data.get('id', position)
        memory_used = gpu_data.get('memoryUsed') or 0
        cells = [str(gpu_index)]
        if show_name:
            cells.append(truncate_name(gpu_data.get('name', 'N/A'), 20))
        if show_util:
            cells.append(f"{gpu_data.get('load') or 0:.0f} %")
        cells.append(human_readable_size(memory_used))
        if show_percent:
            cells.append(f"{gpu_data.get('memoryPercent') or 0:.0f} %")
        if show_temp:
            cells.append(f"{gpu_data.get('temperature') or 0:.0f} °C")
        if show_fan:
            cells.append(f"{gpu_data.get('fanSpeed') or 0:.0f} %")
        if show_power:
            cells.append(f"{gpu_data.get('powerDraw') or 0:.0f} W")
        table.add_row(*cells)

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
        content = build_gpu_rows_table(gpus, terminal_width)

    return Panel(
        content,
        title="[bold cyan]GPU Detail[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_process_table(processes, key, title, terminal_width=None):
    """Build a table for the most resource-intensive processes.

    Parameters
    ----------
    processes : list of dict
        Entries of ``top_cpu`` or ``top_memory``.
    key : str
        Either ``"top_cpu"`` or ``"top_memory"``; selects the metric column.
    title : str
        Human readable name of the ranking, used by the empty-state panel.
    terminal_width : int, optional
        Width of the terminal in columns. ``None`` means unconstrained, in
        which case every column is shown. Otherwise the ``Cmdline``, metric
        (``CPU%``/``Memory%``) and ``PID`` columns are dropped one by one, in
        that order (least important first), as the width estimated through
        :func:`process_table_available_width` drops below
        :data:`PROCESS_TABLE_CMDLINE_COLUMN_MIN_WIDTH`,
        :data:`PROCESS_TABLE_METRIC_COLUMN_MIN_WIDTH` and
        :data:`PROCESS_TABLE_PID_COLUMN_MIN_WIDTH`. ``Name`` is never
        dropped: identifying the process is the whole point of the panel.

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

    available_width = process_table_available_width(terminal_width)
    show_cmdline = (
        available_width is None or available_width >= PROCESS_TABLE_CMDLINE_COLUMN_MIN_WIDTH
    )
    show_metric = (
        available_width is None or available_width >= PROCESS_TABLE_METRIC_COLUMN_MIN_WIDTH
    )
    show_pid = available_width is None or available_width >= PROCESS_TABLE_PID_COLUMN_MIN_WIDTH

    # Padding is collapsed and the edges dropped, and every column is
    # ``no_wrap``, for the same reasons as the GPU Processes table: without
    # them a starved column wraps its text onto extra lines rather than
    # truncating it, stretching the row and leaving the other cells looking
    # like blank rows underneath it.
    table = Table(
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
        collapse_padding=True,
        pad_edge=False,
    )
    metric_header = "CPU%" if key == 'top_cpu' else "Memory%"
    if show_pid:
        table.add_column("PID", style="cyan", no_wrap=True, min_width=4)
    table.add_column("Name", style="green", min_width=6, overflow="ellipsis", no_wrap=True)
    if show_metric:
        table.add_column(metric_header, style="yellow" if key == 'top_cpu' else "blue",
                          justify="right", no_wrap=True, min_width=5)
    if show_cmdline:
        table.add_column("Cmdline", style="white", max_width=20, overflow="ellipsis", no_wrap=True)

    for proc in processes:
        pid = str(proc.get('pid', 'N/A'))
        name = truncate_name(proc.get('name', 'N/A'))
        cmdline = truncate_cmdline(proc.get('cmdline', ''), 20)
        metric = f"{proc.get('cpu_percent', 0):.1f}%" if key == 'top_cpu' \
            else f"{proc.get('memory_percent', 0):.1f}%"

        cells = []
        if show_pid:
            cells.append(pid)
        cells.append(name)
        if show_metric:
            cells.append(metric)
        if show_cmdline:
            cells.append(cmdline)
        table.add_row(*cells)

    return table


def build_processes_panel(data, terminal_width=None):
    """Build the panel holding the Top CPU and Top Memory rankings.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.
    terminal_width : int, optional
        Width of the terminal in columns, forwarded to
        :func:`build_process_table` so it can size the two side-by-side
        rankings correctly (see :func:`process_table_available_width`).

    Returns
    -------
    rich.table.Table
        A two-column grid with one panel per ranking.
    """
    top_cpu = data.get('top_cpu', [])
    top_memory = data.get('top_memory', [])

    table_cpu = build_process_table(top_cpu, 'top_cpu', 'CPU', terminal_width)
    table_mem = build_process_table(top_memory, 'top_memory', 'Memory', terminal_width)

    processes_table = Table.grid(expand=True)
    processes_table.add_column()
    processes_table.add_column()

    processes_table.add_row(
        Panel(table_cpu, title="[bold cyan]Top CPU[/bold cyan]", border_style="cyan", padding=(0, 1)),
        Panel(table_mem, title="[bold cyan]Top Memory[/bold cyan]", border_style="cyan", padding=(0, 1))
    )

    return processes_table


def build_gpu_processes_panel(data, multi_gpu=False, terminal_width=None):
    """Build the panel listing the processes holding VRAM.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.
    multi_gpu : bool, optional
        When ``True`` a ``GPU`` column is prepended with the index of the card
        each process runs on. On a single-GPU host that column would only
        repeat ``0`` on every row, so it is left out.
    terminal_width : int, optional
        Width of the terminal in columns. ``None`` means unconstrained, in
        which case every column is shown at its natural width. Below
        :data:`GPU_PROCESSES_CMDLINE_COLUMN_MIN_WIDTH` (estimated through
        :func:`left_column_available_width`) the ``Cmdline`` column, the
        widest and least essential one, is dropped, and the ``Name`` and
        ``Memory Used`` columns are additionally capped at
        :data:`GPU_PROCESSES_NAME_COLUMN_MAX_WIDTH` and
        :data:`GPU_PROCESSES_MEMORY_COLUMN_MAX_WIDTH` respectively: their
        headers are wider than a single value, so left uncapped Rich keeps
        them at their natural width and the table overflows its panel even
        with ``Cmdline`` gone. Capping them keeps the GPU, PID and the memory
        figure with its unit legible instead of being clipped mid-word.

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

    available_width = left_column_available_width(terminal_width)
    show_cmdline = (
        available_width is None or available_width >= GPU_PROCESSES_CMDLINE_COLUMN_MIN_WIDTH
    )
    name_max_width = None if show_cmdline else GPU_PROCESSES_NAME_COLUMN_MAX_WIDTH
    memory_max_width = None if show_cmdline else GPU_PROCESSES_MEMORY_COLUMN_MAX_WIDTH

    # Padding is collapsed and the edges dropped: this table lives in a third of
    # the screen and the extra column has to come from somewhere. Every column
    # is ``no_wrap``: without it, a column starved for width wraps its text
    # onto extra lines instead of truncating it, which stretches the row and
    # leaves the other cells looking like blank rows underneath it.
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
    table.add_column(
        "Name", style="green", min_width=6, max_width=name_max_width,
        overflow="ellipsis", no_wrap=True,
    )
    table.add_column(
        "Memory Used", style="blue", justify="right", no_wrap=True, min_width=5,
        max_width=memory_max_width, overflow="ellipsis",
    )
    if show_cmdline:
        table.add_column("Cmdline", style="white", max_width=20, overflow="ellipsis", no_wrap=True)

    for proc in processes:
        pid = str(proc.get('pid', 'N/A'))
        name = proc.get('name', 'N/A')
        memory_used = human_readable_size(proc.get('memory_used', 0))
        cells = [pid, truncate_name(name), memory_used]
        if show_cmdline:
            cells.append(truncate_cmdline(proc.get('cmdline', ''), 20))
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


def build_ollama_panel(data, gpu_processes=None, terminal_width=None):
    """Build the panel listing the models Ollama keeps loaded.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.
    gpu_processes : list of dict, optional
        Entries of ``top_gpu_processes``, used to fill the ``GPU`` column.
    terminal_width : int, optional
        Width of the terminal in columns. ``None`` means unconstrained, in
        which case every column is shown. Otherwise the ``Size``, ``VRAM``,
        ``GPU%`` and ``Expires`` columns are dropped one by one, in that
        order (least important first), as the width estimated through
        :func:`left_column_available_width` drops below
        :data:`OLLAMA_VRAM_COLUMN_MIN_WIDTH`,
        :data:`OLLAMA_SIZE_COLUMN_MIN_WIDTH`,
        :data:`OLLAMA_GPU_PERCENT_COLUMN_MIN_WIDTH` and
        :data:`OLLAMA_EXPIRES_COLUMN_MIN_WIDTH`. ``Model``, ``GPU`` and
        ``Ctx`` are never dropped: they carry the values this panel exists to
        show.

    Returns
    -------
    rich.panel.Panel
        The Ollama panel, or an empty-state panel.

    Notes
    -----
    The ``GPU`` column is a process-level heuristic, not a per-model truth.
    Ollama serves N models from a single OS process and ``/api/ps`` exposes no
    per-model PID, so the only thing that can be correlated is "the Ollama
    process holds VRAM on these cards". Every row resident in VRAM therefore
    shows the same indices, even when the models are actually spread across
    different GPUs. A model reporting ``size_vram`` of ``0`` is not resident
    in VRAM at all, so attributing it to those cards would be actively
    misleading; its cell falls back to the same placeholder used when no
    index could be resolved.
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

    available_width = left_column_available_width(terminal_width)
    show_vram = available_width is None or available_width >= OLLAMA_VRAM_COLUMN_MIN_WIDTH
    show_size = available_width is None or available_width >= OLLAMA_SIZE_COLUMN_MIN_WIDTH
    show_gpu_percent = (
        available_width is None or available_width >= OLLAMA_GPU_PERCENT_COLUMN_MIN_WIDTH
    )
    show_expires = available_width is None or available_width >= OLLAMA_EXPIRES_COLUMN_MIN_WIDTH

    # Up to seven columns in a third of the screen leave no room for cell
    # padding: with it the table overflows its region and Rich clips the
    # rightmost columns off. The box lines still separate the values, every
    # floor is kept low, and every column is ``no_wrap`` so a starved one
    # truncates its text instead of wrapping it onto extra lines.
    table = Table(
        show_header=True,
        header_style="bold magenta",
        padding=(0, 0),
        pad_edge=False,
    )
    table.add_column("Model", style="green", min_width=8, overflow="ellipsis", no_wrap=True)
    if has_ollama_process:
        table.add_column("GPU", style="cyan", justify="right", no_wrap=True, min_width=3)
    table.add_column("Ctx", style="magenta", justify="right", no_wrap=True, min_width=3)
    if show_size:
        table.add_column("Size", style="blue", justify="right", no_wrap=True, min_width=5)
    if show_vram:
        table.add_column("VRAM", style="blue", justify="right", no_wrap=True, min_width=5)
    if show_gpu_percent:
        table.add_column("GPU%", style="red", justify="right", no_wrap=True, min_width=4)
    if show_expires:
        table.add_column("Expires", style="yellow", justify="right", no_wrap=True, min_width=5)

    for model in models:
        model_name = truncate_name(model.get('name', 'N/A'))
        context_length = format_context_length(model.get('context_length'))
        size_value = model.get('size', 0)
        # ``or 0`` also absorbs an explicit ``None``, which a payload predating
        # this field would otherwise turn into a ``TypeError`` below.
        size_vram_value = model.get('size_vram') or 0
        size_total = human_readable_size(size_value)
        size_vram = human_readable_size(size_vram_value)
        gpu_loaded_ratio = (size_vram_value / size_value) * 100 if size_value > 0 else 0
        gpu_loaded_str = f"{gpu_loaded_ratio:.0f}%"
        expiration = time_until(model.get('expires_at', ''))

        cells = [model_name]
        if has_ollama_process:
            # A model absent from VRAM was never actually placed on any of the
            # cards Ollama holds; showing the process-level indices there
            # would misattribute it, so it gets the "unresolved" placeholder.
            cells.append(gpu_label if size_vram_value > 0 else "-")
        cells.append(context_length)
        if show_size:
            cells.append(size_total)
        if show_vram:
            cells.append(size_vram)
        if show_gpu_percent:
            cells.append(gpu_loaded_str)
        if show_expires:
            cells.append(expiration)
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
    ollama_panel = build_ollama_panel(data, gpu_processes=gpu_processes, terminal_width=terminal_width)
    gpu_processes_panel = build_gpu_processes_panel(
        data, multi_gpu=multi_gpu, terminal_width=terminal_width
    )

    # Both modes keep the process rankings top right, they need the width.
    layout["top_right"].update(build_processes_panel(data, terminal_width))

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

    with Live(layout, refresh_per_second=4, screen=True) as live:
        # Tracks whether the previous iteration was showing the help screen,
        # so ``live.update`` is only called on an actual transition rather
        # than every loop iteration.
        previous_help_flag = False
        while not exit_event.is_set():
            with state_lock:
                current_interval = refresh_interval
                paused = is_paused
                help_flag = show_help_flag

            if help_flag:
                # Show the help panel in full screen. ``layout.update`` would
                # be a no-op here: once a ``Layout`` has children, Rich
                # renders those in preference to its own set renderable, so
                # the help screen has to be swapped in on the ``Live``
                # instance instead, which is what actually controls what
                # gets drawn.
                if not previous_help_flag:
                    live.update(build_full_screen_help())
            else:
                if previous_help_flag:
                    # Coming back from the help screen: restore the live
                    # layout so refreshes resume underneath it.
                    live.update(layout)
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

            previous_help_flag = help_flag

            # Check if a layout rebuild is required
            if rebuild_layout_event.is_set():
                if help_flag:
                    live.update(build_full_screen_help())
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
