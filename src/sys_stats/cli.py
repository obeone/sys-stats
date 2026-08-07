#!/usr/bin/env python3

import argparse
import io
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import readchar  # To capture key presses
import requests
from rich.cells import cell_len
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.measure import Measurement
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

# Floors for the columns that hold free text rather than a formatted figure.
# Every other column is sized to its own widest value and is dropped outright
# rather than shown truncated (see :class:`AdaptiveColumn`); these ones carry
# names and command lines, which stay useful once ellipsised, so they are
# allowed to shrink down to these widths and to soak up whatever room is left
# over afterwards.
MODEL_COLUMN_MIN_WIDTH = 8
PROCESS_NAME_COLUMN_MIN_WIDTH = 8
GPU_NAME_COLUMN_MIN_WIDTH = 8
CMDLINE_COLUMN_MIN_WIDTH = 7

# Which column each table sacrifices first when it does not fit the room it
# was given: the lowest priority goes first, and the last one standing is
# kept whatever happens. The numbers themselves are meaningless, only their
# order matters.
OLLAMA_COLUMN_PRIORITIES = {
    "Model": 100,
    "Ctx": 90,
    "GPU": 80,
    "VRAM": 40,
    "Size": 30,
    "GPU%": 20,
    "Expires": 10,
}
GPU_PROCESSES_COLUMN_PRIORITIES = {
    "Name": 100,
    "GPU": 90,
    "Memory Used": 80,
    "PID": 20,
    "Cmdline": 10,
}
GPU_DETAIL_COLUMN_PRIORITIES = {
    "GPU": 100,
    "VRAM": 90,
    "Util": 60,
    "Temp": 50,
    "Power": 40,
    "Fan": 30,
    "Name": 20,
    "%": 10,
}
# The metric outranks the PID here: a ranking stripped of the figure it
# ranks by is a list of names in an order nobody can see, while a name and a
# percentage still say everything the panel is for.
PROCESS_TABLE_COLUMN_PRIORITIES = {
    "Name": 100,
    "CPU%": 40,
    "Memory%": 40,
    "PID": 30,
    "Cmdline": 10,
}

# Style applied to the GPU process rows owned by Ollama, so its share of the
# VRAM stands out among the other compute apps.
OLLAMA_ROW_STYLE = "bold magenta"

# Width handed to the measurement console below. It only has to be larger
# than any table this dashboard can produce: Rich clamps a measurement to the
# width it is measured against, and a clamped measurement would report every
# table as fitting.
MEASUREMENT_WIDTH = 10_000

# Console used solely to measure candidate tables before rendering them. It
# writes into a buffer nobody reads: what is wanted is Rich's own column
# arithmetic, which needs a console to resolve styles and character widths.
_measurement_console = Console(file=io.StringIO(), width=MEASUREMENT_WIDTH, legacy_windows=False)


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


class AdaptiveRenderable:
    """A renderable that is only built once Rich reveals its real width.

    Every panel of this dashboard has to decide which columns it can afford,
    and the only reliable source for the room it actually gets is Rich
    itself: ``options.max_width`` in :meth:`__rich_console__` is the exact
    number of columns the renderable may occupy, already net of the enclosing
    panel's borders and padding. Guessing that number from the terminal width
    and the layout ratios cannot account for the panel chrome or for Rich's
    own rounding, which is how tables ended up overflowing their panel and
    losing their right border.

    Parameters
    ----------
    builder : callable
        Takes the available width in columns and returns the renderable to
        display. It is called once per render pass, and again whenever Rich
        measures the object, so it must be free of side effects.
    """

    def __init__(self, builder):
        self._builder = builder

    def build(self, available_width):
        """Build the wrapped renderable for a given width.

        Parameters
        ----------
        available_width : int
            Width in columns the renderable may occupy.

        Returns
        -------
        rich.console.RenderableType
            Whatever the builder produced.
        """
        return self._builder(available_width)

    def __rich_console__(self, console, options):
        """Render the wrapped renderable at its real width."""
        yield self.build(options.max_width)

    def __rich_measure__(self, console, options):
        """Report the wrapped renderable's own measurement.

        Without this, Rich falls back to "anything up to the full width",
        which would make the two side-by-side process panels claim the whole
        region instead of sharing it.
        """
        return Measurement.get(console, options, self.build(options.max_width))


class AdaptiveColumn:
    """One candidate column of a table sized by :func:`build_adaptive_table`.

    A column knows the widest string it will ever have to show, and refuses
    to be rendered narrower than that unless it explicitly opts into
    shrinking through ``minimum``. That is what keeps a value such as
    ``128K`` from being ellipsised into ``12…``: a column that cannot show
    its own content is dropped, never mangled.

    Parameters
    ----------
    header : str
        Column header.
    cells : list of str
        One value per row, in row order.
    style : str, optional
        Rich style applied to the column's values.
    justify : str, optional
        Rich justification, ``"left"`` by default.
    minimum : int, optional
        Width the column may be squeezed down to. Defaults to the natural
        width, meaning "never squeezed". Only ever set on columns holding
        free text, which stays readable once ellipsised. The header always
        fits: a column whose own header is truncated tells the reader
        nothing.
    grow : bool, optional
        Whether the column may take a share of the width left over once
        every kept column has its minimum. Only makes sense together with
        ``minimum``.
    """

    def __init__(self, header, cells, *, style=None, justify="left", minimum=None, grow=False):
        self.header = header
        self.cells = [str(cell) for cell in cells]
        self.style = style
        self.justify = justify
        self.grow = grow
        self.natural_width = max(cell_len(text) for text in [header, *self.cells])
        if minimum is None:
            self.minimum = self.natural_width
        else:
            # Never below the header, never above what the content needs.
            self.minimum = min(self.natural_width, max(minimum, cell_len(header)))


def measure_table_width(table):
    """Return the exact number of columns a table renders into.

    Only meaningful for tables whose every column has a fixed ``width``, as
    :func:`assemble_table` builds them: their measurement has no slack, so
    Rich's minimum and maximum coincide with the rendered width, borders and
    padding included.

    Parameters
    ----------
    table : rich.table.Table
        A table built by :func:`assemble_table`, without an overall width.

    Returns
    -------
    int
        The rendered width in columns.
    """
    options = _measurement_console.options.update_width(MEASUREMENT_WIDTH)
    return Measurement.get(_measurement_console, options, table).maximum


def assemble_table(columns, widths, row_styles=None, total_width=None):
    """Build a table whose columns all have a fixed width.

    Fixed widths are not a stylistic choice, they are the only shape Rich
    keeps its promises on. When a table overflows, ``Table._calculate_column_widths``
    shrinks the columns and then re-measures them, and that second
    measurement re-applies each column's ``min_width``, undoing the shrink
    and letting the table run past its panel. A column pinned with ``width``
    is clamped by the re-measure instead of growing back.

    Parameters
    ----------
    columns : list of AdaptiveColumn
        Columns to render, in display order.
    widths : list of int
        Content width of each column, excluding padding.
    row_styles : list, optional
        One Rich style (or ``None``) per row.
    total_width : int, optional
        Overall width of the table. Acts as a hard cap: with fixed columns
        Rich can honour it, so it is the last line of defence against an
        overflow.

    Returns
    -------
    rich.table.Table
        The assembled table.
    """
    # Padding is collapsed and the edges dropped: these tables live in a
    # fraction of the screen and every spare column buys another value.
    table = Table(
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
        collapse_padding=True,
        pad_edge=False,
        width=total_width,
    )
    for column, width in zip(columns, widths, strict=True):
        # ``no_wrap`` keeps a starved cell on a single line: without it the
        # text wraps onto extra lines, stretching the row and leaving the
        # other cells looking like blank rows underneath it.
        table.add_column(
            column.header,
            style=column.style,
            justify=column.justify,
            width=width,
            no_wrap=True,
            overflow="ellipsis",
        )

    row_count = len(columns[0].cells) if columns else 0
    styles = row_styles if row_styles is not None else [None] * row_count
    for index in range(row_count):
        table.add_row(*[column.cells[index] for column in columns], style=styles[index])

    return table


def build_adaptive_table(columns, available_width, priorities, row_styles=None):
    """Fit a set of candidate columns into an exactly known width.

    The least important column is dropped, and the result measured again,
    until the table fits. Whatever room is left over then goes to the columns
    that can use it, so the table fills its panel rather than leaving a
    ragged gap. Because the width is measured rather than estimated, the
    column set only ever grows as the terminal widens.

    Parameters
    ----------
    columns : list of AdaptiveColumn
        Candidate columns, in display order.
    available_width : int
        Room the table has, in columns, as reported by Rich.
    priorities : dict
        Maps a column header to its importance; the lowest is dropped first.
    row_styles : list, optional
        One Rich style (or ``None``) per row.

    Returns
    -------
    rich.table.Table
        A table guaranteed to fit ``available_width``, unless a single column
        is already too wide for it.
    """
    kept = list(columns)
    widths = [column.minimum for column in kept]
    width = measure_table_width(assemble_table(kept, widths, row_styles))

    while width > available_width and len(kept) > 1:
        index = min(range(len(kept)), key=lambda position: priorities[kept[position].header])
        del kept[index]
        del widths[index]
        width = measure_table_width(assemble_table(kept, widths, row_styles))

    # Hand the slack to the free-text columns, left to right.
    slack = available_width - width
    for index, column in enumerate(kept):
        if slack <= 0:
            break
        if not column.grow:
            continue
        extra = min(slack, column.natural_width - widths[index])
        widths[index] += extra
        slack -= extra

    return assemble_table(kept, widths, row_styles, total_width=available_width)


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
    # Every figure goes through ``or 0``: a driver that cannot report one of
    # them sends ``null``, and formatting ``None`` as a number raises. The
    # row-per-GPU rendering has always guarded this; this one used to crash
    # on the very same payload, purely because the terminal was wider.
    gpu_name = truncate_name(gpu_data.get('name') or 'N/A', 25)
    gpu_load = gpu_data.get('load') or 0
    gpu_fan_speed = gpu_data.get('fanSpeed') or 0
    gpu_power_draw = gpu_data.get('powerDraw') or 0
    gpu_temperature = gpu_data.get('temperature') or 0

    memory_used = gpu_data.get('memoryUsed') or 0
    memory_used_str = human_readable_size(memory_used)
    memory_total_str = human_readable_size(gpu_memory_total_bytes(gpu_data))
    # The reported percentage is kept as-is rather than recomputed: it is what
    # the driver claims, and it stays consistent with the web UI.
    vram_percent = gpu_data.get('memoryPercent') or 0

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


def build_gpu_rows_table(gpus, available_width):
    """Build one wide table holding a single row per GPU.

    The horizontal fallback of :func:`build_gpu_detail_panel`, used when there
    are too many cards to show their vertical tables side by side.

    Parameters
    ----------
    gpus : list of dict
        The ``gpu`` list returned by ``/stats``.
    available_width : int
        Room the table has, in columns, as measured by Rich. Columns are
        dropped in the order given by :data:`GPU_DETAIL_COLUMN_PRIORITIES`
        until the rest fits.

    Returns
    -------
    rich.table.Table
        A table with as many of the GPU / Name / Util / VRAM / % / Temp /
        Fan / Power columns as fit.
    """
    # ``id`` is the driver index; fall back to the position in the list when a
    # payload omits it, so the column is never blank.
    indices = [str(gpu.get('id', position)) for position, gpu in enumerate(gpus)]
    names = [truncate_name(gpu.get('name') or 'N/A', 20) for gpu in gpus]
    loads = [f"{gpu.get('load') or 0:.0f} %" for gpu in gpus]
    memories = [human_readable_size(gpu.get('memoryUsed') or 0) for gpu in gpus]
    percents = [f"{gpu.get('memoryPercent') or 0:.0f} %" for gpu in gpus]
    temperatures = [f"{gpu.get('temperature') or 0:.0f} °C" for gpu in gpus]
    fan_speeds = [f"{gpu.get('fanSpeed') or 0:.0f} %" for gpu in gpus]
    power_draws = [f"{gpu.get('powerDraw') or 0:.0f} W" for gpu in gpus]

    columns = [
        AdaptiveColumn("GPU", indices, style="cyan", justify="right"),
        AdaptiveColumn(
            "Name", names, style="green", minimum=GPU_NAME_COLUMN_MIN_WIDTH, grow=True
        ),
        AdaptiveColumn("Util", loads, style="yellow", justify="right"),
        AdaptiveColumn("VRAM", memories, style="blue", justify="right"),
        AdaptiveColumn("%", percents, style="blue", justify="right"),
        AdaptiveColumn("Temp", temperatures, style="red", justify="right"),
        AdaptiveColumn("Fan", fan_speeds, style="white", justify="right"),
        AdaptiveColumn("Power", power_draws, style="white", justify="right"),
    ]

    return build_adaptive_table(columns, available_width, GPU_DETAIL_COLUMN_PRIORITIES)


def build_gpu_detail_content(gpus, available_width):
    """Pick and build the rendering of the GPU detail panel.

    Up to :data:`MAX_SIDE_BY_SIDE_GPUS` vertical tables are laid out side by
    side as long as each of them gets at least
    :data:`MIN_GPU_COLUMN_WIDTH` columns; anything beyond that falls back to
    one row per GPU.

    Parameters
    ----------
    gpus : list of dict
        The ``gpu`` list returned by ``/stats``.
    available_width : int
        Room the panel's content has, in columns, as measured by Rich.

    Returns
    -------
    rich.table.Table
        Either a grid of vertical tables, or the row-per-GPU table.
    """
    side_by_side = (
        len(gpus) <= MAX_SIDE_BY_SIDE_GPUS
        and available_width >= len(gpus) * MIN_GPU_COLUMN_WIDTH
    )
    if not side_by_side:
        return build_gpu_rows_table(gpus, available_width)

    grid = Table.grid(expand=True)
    for _ in gpus:
        grid.add_column(ratio=1)
    grid.add_row(*[build_single_gpu_table(gpu_data) for gpu_data in gpus])
    return grid


def build_gpu_detail_panel(data):
    """Build the per-GPU detail panel used in multi-GPU mode.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.

    Returns
    -------
    rich.panel.Panel
        The GPU detail panel, whose content is chosen from the width Rich
        hands it (see :func:`build_gpu_detail_content`).
    """
    gpus = data.get('gpu') or []
    if not data.get('has_gpu') or not gpus:
        return Panel(
            "No GPU detected.",
            title="[bold cyan]GPU Detail[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        )

    return Panel(
        AdaptiveRenderable(lambda width: build_gpu_detail_content(gpus, width)),
        title="[bold cyan]GPU Detail[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_process_table(processes, key, available_width):
    """Build a table for the most resource-intensive processes.

    Parameters
    ----------
    processes : list of dict
        Entries of ``top_cpu`` or ``top_memory``.
    key : str
        Either ``"top_cpu"`` or ``"top_memory"``; selects the metric column.
    available_width : int
        Room the table has, in columns, as measured by Rich. Columns are
        dropped in the order given by :data:`PROCESS_TABLE_COLUMN_PRIORITIES`
        until the rest fits.

    Returns
    -------
    rich.table.Table
        The ranking table.
    """
    pids = [str(proc.get('pid', 'N/A')) for proc in processes]
    names = [truncate_name(proc.get('name') or 'N/A') for proc in processes]
    cmdlines = [truncate_cmdline(proc.get('cmdline') or '', 20) for proc in processes]
    if key == 'top_cpu':
        metric_header = "CPU%"
        metric_style = "yellow"
        metrics = [f"{proc.get('cpu_percent') or 0:.1f}%" for proc in processes]
    else:
        metric_header = "Memory%"
        metric_style = "blue"
        metrics = [f"{proc.get('memory_percent') or 0:.1f}%" for proc in processes]

    columns = [
        AdaptiveColumn("PID", pids, style="cyan"),
        AdaptiveColumn(
            "Name", names, style="green", minimum=PROCESS_NAME_COLUMN_MIN_WIDTH, grow=True
        ),
        AdaptiveColumn(metric_header, metrics, style=metric_style, justify="right"),
        AdaptiveColumn(
            "Cmdline", cmdlines, style="white", minimum=CMDLINE_COLUMN_MIN_WIDTH, grow=True
        ),
    ]

    return build_adaptive_table(columns, available_width, PROCESS_TABLE_COLUMN_PRIORITIES)


def build_process_panel(processes, key, title):
    """Wrap a process ranking in its titled panel.

    Parameters
    ----------
    processes : list of dict
        Entries of ``top_cpu`` or ``top_memory``.
    key : str
        Either ``"top_cpu"`` or ``"top_memory"``.
    title : str
        Human readable name of the ranking, used as the panel title and by
        the empty state.

    Returns
    -------
    rich.panel.Panel
        The ranking panel, showing an explanatory line when there is no data.
    """
    content = (
        AdaptiveRenderable(lambda width: build_process_table(processes, key, width))
        if processes
        else f"No data for {title}."
    )
    return Panel(
        content,
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    )


def build_processes_panel(data):
    """Build the panel holding the Top CPU and Top Memory rankings.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.

    Returns
    -------
    rich.table.Table
        A two-column grid with one panel per ranking. Each panel sizes its
        own table from the half of the region Rich gives it, so neither has
        to guess how the grid was split.
    """
    processes_table = Table.grid(expand=True)
    processes_table.add_column()
    processes_table.add_column()

    processes_table.add_row(
        build_process_panel(data.get('top_cpu', []), 'top_cpu', 'Top CPU'),
        build_process_panel(data.get('top_memory', []), 'top_memory', 'Top Memory'),
    )

    return processes_table


def build_gpu_processes_table(processes, multi_gpu, available_width):
    """Build the table listing the processes holding VRAM.

    Parameters
    ----------
    processes : list of dict
        Entries of ``top_gpu_processes``.
    multi_gpu : bool
        When ``True`` a ``GPU`` column is prepended with the index of the card
        each process runs on. On a single-GPU host that column would only
        repeat ``0`` on every row, so it is left out.
    available_width : int
        Room the table has, in columns, as measured by Rich. Columns are
        dropped in the order given by
        :data:`GPU_PROCESSES_COLUMN_PRIORITIES` until the rest fits.

    Returns
    -------
    rich.table.Table
        The GPU processes table.
    """
    names = [proc.get('name') or 'N/A' for proc in processes]
    columns = []
    if multi_gpu:
        # A driver too old to report gpu_uuid leaves the index unresolved.
        indices = [
            "?" if proc.get('gpu_index') is None else str(proc['gpu_index'])
            for proc in processes
        ]
        columns.append(AdaptiveColumn("GPU", indices, style="cyan", justify="right"))

    columns.append(
        AdaptiveColumn("PID", [str(proc.get('pid', 'N/A')) for proc in processes], style="cyan")
    )
    columns.append(
        AdaptiveColumn(
            "Name", [truncate_name(name) for name in names], style="green",
            minimum=PROCESS_NAME_COLUMN_MIN_WIDTH, grow=True,
        )
    )
    columns.append(
        AdaptiveColumn(
            "Memory Used",
            [human_readable_size(proc.get('memory_used') or 0) for proc in processes],
            style="blue", justify="right",
        )
    )
    columns.append(
        AdaptiveColumn(
            "Cmdline",
            [truncate_cmdline(proc.get('cmdline') or '', 20) for proc in processes],
            style="white", minimum=CMDLINE_COLUMN_MIN_WIDTH, grow=True,
        )
    )

    # Ollama's own workers are highlighted so its share stands out.
    row_styles = [OLLAMA_ROW_STYLE if name == "ollama" else None for name in names]

    return build_adaptive_table(
        columns, available_width, GPU_PROCESSES_COLUMN_PRIORITIES, row_styles=row_styles
    )


def build_gpu_processes_panel(data, multi_gpu=False):
    """Build the panel listing the processes holding VRAM.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.
    multi_gpu : bool, optional
        Forwarded to :func:`build_gpu_processes_table`.

    Returns
    -------
    rich.panel.Panel
        The GPU processes panel, or an empty-state panel.
    """
    processes = data.get("top_gpu_processes", [])
    content = (
        AdaptiveRenderable(
            lambda width: build_gpu_processes_table(processes, multi_gpu, width)
        )
        if processes
        else "No GPU processes."
    )
    return Panel(
        content,
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


def build_ollama_table(models, has_ollama_process, gpu_label, available_width):
    """Build the table listing the models Ollama keeps loaded.

    Parameters
    ----------
    models : list of dict
        The ``models`` list of the ``/api/ps`` response.
    has_ollama_process : bool
        Whether an Ollama process was found among the GPU compute apps; the
        ``GPU`` column only exists when there is something to attribute.
    gpu_label : str
        Comma-joined indices Ollama holds VRAM on.
    available_width : int
        Room the table has, in columns, as measured by Rich. Columns are
        dropped in the order given by :data:`OLLAMA_COLUMN_PRIORITIES` until
        the rest fits.

    Returns
    -------
    rich.table.Table
        The Ollama table.

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
    names, gpu_cells, contexts = [], [], []
    sizes, vrams, ratios, expirations = [], [], [], []

    for model in models:
        size_value = model.get('size') or 0
        # ``or 0`` also absorbs an explicit ``None``, which a payload predating
        # this field would otherwise turn into a ``TypeError`` below.
        size_vram_value = model.get('size_vram') or 0

        names.append(truncate_name(model.get('name') or 'N/A'))
        # A model absent from VRAM was never actually placed on any of the
        # cards Ollama holds; showing the process-level indices there would
        # misattribute it, so it gets the "unresolved" placeholder.
        gpu_cells.append(gpu_label if size_vram_value > 0 else "-")
        contexts.append(format_context_length(model.get('context_length')))
        sizes.append(human_readable_size(size_value))
        vrams.append(human_readable_size(size_vram_value))
        ratio = (size_vram_value / size_value) * 100 if size_value > 0 else 0
        ratios.append(f"{ratio:.0f}%")
        expirations.append(time_until(model.get('expires_at', '')))

    columns = [
        AdaptiveColumn(
            "Model", names, style="green", minimum=MODEL_COLUMN_MIN_WIDTH, grow=True
        ),
    ]
    if has_ollama_process:
        columns.append(AdaptiveColumn("GPU", gpu_cells, style="cyan", justify="right"))
    columns.extend([
        AdaptiveColumn("Ctx", contexts, style="magenta", justify="right"),
        AdaptiveColumn("Size", sizes, style="blue", justify="right"),
        AdaptiveColumn("VRAM", vrams, style="blue", justify="right"),
        AdaptiveColumn("GPU%", ratios, style="red", justify="right"),
        AdaptiveColumn("Expires", expirations, style="yellow", justify="right"),
    ])

    return build_adaptive_table(columns, available_width, OLLAMA_COLUMN_PRIORITIES)


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

    return Panel(
        AdaptiveRenderable(
            lambda width: build_ollama_table(models, has_ollama_process, gpu_label, width)
        ),
        title="[bold cyan]Ollama Statistics[/bold cyan]",
        border_style="cyan",
        padding=(0, 1)
    )


def build_layout_content(layout, data, interval):
    """Fill the four layout regions with the fetched data.

    The routing depends on the number of GPUs. With zero or one card the
    per-GPU detail fits inside the summary panel, so the layout stays as it
    always was. From two cards on, the detail moves to its own region, the GPU
    processes join Ollama on the left (they describe the same VRAM), and the
    summary keeps only cumulated figures.

    No width is threaded through: every panel sizes itself from the room Rich
    hands it at render time (see :class:`AdaptiveRenderable`), which is the
    only figure that accounts for the layout ratios *and* the panel chrome.

    Parameters
    ----------
    layout : rich.layout.Layout
        The layout built by :func:`create_layout`; updated in place.
    data : dict
        The ``/stats`` payload.
    interval : int
        Current refresh interval in seconds.

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
        layout["bottom_right"].update(build_gpu_detail_panel(data))
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


def read_dashboard_state():
    """Take a consistent snapshot of the state the keyboard thread owns.

    Returns
    -------
    tuple of (bool, bool, int)
        Whether the help screen is up, whether refreshing is paused, and the
        current refresh interval in seconds.
    """
    with state_lock:
        return show_help_flag, is_paused, refresh_interval


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
            # Consume any pending wake-up *before* reading the state it
            # announces. A key press landing after this point sets the event
            # again, which cuts the sleep at the end of the iteration short
            # instead of being swallowed until the next refresh.
            rebuild_layout_event.clear()
            help_flag, paused, current_interval = read_dashboard_state()

            if not help_flag and not paused:
                stats = fetch_stats(args.url)
                if stats:
                    with stats_lock:
                        latest_stats = stats
                # The request blocks for as long as the server takes to
                # answer, and every key press in that window changed the
                # state behind our back. Acting on the pre-fetch snapshot is
                # what used to swallow a key press for a whole interval.
                help_flag, paused, current_interval = read_dashboard_state()

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
                with stats_lock:
                    stats_snapshot = latest_stats
                if stats_snapshot:
                    build_layout_content(layout, stats_snapshot, current_interval)

            previous_help_flag = help_flag

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
