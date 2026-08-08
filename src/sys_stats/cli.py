#!/usr/bin/env python3

import argparse
import io
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import readchar  # To capture key presses
import requests
from readchar import key as readchar_key
from rich.cells import cell_len
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.measure import Measurement
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

# Default API URL
SYS_STATS_API_URL = os.getenv('SYS_STATS_API_URL', 'http://localhost:5000/stats')

console = Console()

# Dashboard states
is_paused = False
show_help_flag = False
refresh_interval = 5  # Default refresh interval in seconds

# Scrolling state, owned by the keyboard thread and read back by the panels
# while they render. ``scroll_offsets`` maps a panel key to the index of its
# first visible row; ``panel_row_counts`` and ``panel_page_sizes`` are written
# by the render pass so the keyboard thread knows how far ``End`` and
# ``Page Down`` should go without having to measure anything itself.
scroll_offsets = {}
panel_row_counts = {}
panel_page_sizes = {}
# Keys of the scrollable panels currently on screen, in Tab order, and the one
# the scroll keys drive. Both are refreshed by :func:`build_layout_content`,
# because which panels exist depends on the payload and on the layout mode.
scrollable_panel_keys = []
focused_panel = None

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

# Border styles telling the scroll keys' target apart from the rest.
PANEL_BORDER_STYLE = "cyan"
FOCUSED_PANEL_BORDER_STYLE = "bold yellow"

# Identity of every scrollable panel. These double as the keys of
# :data:`scroll_offsets` and as the Tab order once
# :func:`build_layout_content` has listed the ones actually on screen.
PANEL_OLLAMA = "ollama"
PANEL_GPU_PROCESSES = "gpu_processes"
PANEL_TOP_CPU = "top_cpu"
PANEL_TOP_MEMORY = "top_memory"
PANEL_GPU_DETAIL = "gpu_detail"

# What each scrolling key can arrive as. Terminals do not agree on the escape
# sequences, and :mod:`readchar` only knows one spelling of each: it answers
# ``\x1b[F`` for End, while tmux, screen and the Linux console send ``\x1b[4~``
# and xterm in application mode sends ``\x1bOF``. Worse, readchar's escape
# parser does not expect a ``4`` in that position and stops one byte short,
# handing back ``\x1b[4`` and then a stray ``~`` on the next read. Every
# spelling is accepted, and the leftover ``~`` matches nothing and is ignored.
SCROLL_UP_KEYS = frozenset({readchar_key.UP, "\x1bOA"})
SCROLL_DOWN_KEYS = frozenset({readchar_key.DOWN, "\x1bOB"})
SCROLL_PAGE_UP_KEYS = frozenset({readchar_key.PAGE_UP, "\x1b[5~", "\x1bOy"})
SCROLL_PAGE_DOWN_KEYS = frozenset({readchar_key.PAGE_DOWN, "\x1b[6~", "\x1bOs"})
SCROLL_HOME_KEYS = frozenset({readchar_key.HOME, "\x1bOH", "\x1b[1~", "\x1b[7~"})
SCROLL_END_KEYS = frozenset({readchar_key.END, "\x1bOF", "\x1b[4~", "\x1b[4", "\x1b[8~"})

# Chrome a panel adds around its content: one border column on each side plus
# the ``(0, 1)`` padding horizontally, one border line above and below
# vertically. Rich hands the real width to the renderable inside the panel, but
# the *height* has to be worked out before the panel exists, because it is what
# decides how many rows the panel is built with in the first place.
PANEL_CHROME_WIDTH = 4
PANEL_CHROME_HEIGHT = 2

# Smallest panel worth drawing: two borders, a header, its rule and one row.
# A scrollable panel squeezed to this still says what it is and shows one line
# of data with an indicator of how much it is hiding.
MIN_SCROLLABLE_PANEL_HEIGHT = 5
# Panels with no rows to window (the summary) cannot shrink gracefully, so they
# are only ever taken down to their borders plus a line of content, and only
# once every scrollable panel is already at its own minimum.
MIN_PLAIN_PANEL_HEIGHT = 3

# Terminal width from which the dashboard lays its panels out in a single row
# of columns rather than in a 2x2 grid. Below it, four columns would leave
# every table too narrow to keep more than a couple of its own.
WIDE_LAYOUT_MIN_WIDTH = 200
LAYOUT_MODE_WIDE = "wide"
LAYOUT_MODE_NARROW = "narrow"

# Width and height handed to the measurement console below. They only have to
# be larger than anything this dashboard can produce: Rich clamps a measurement
# to the size it is measured against, and a clamped measurement would report
# every table as fitting.
MEASUREMENT_WIDTH = 10_000
MEASUREMENT_HEIGHT = 10_000

# Console used solely to measure candidate tables before rendering them. It
# writes into a buffer nobody reads: what is wanted is Rich's own column
# arithmetic, which needs a console to resolve styles and character widths.
# The height matters as much as the width: ``ConsoleOptions.update`` leaves
# ``max_height`` alone when the height is cleared, so a console sized to a real
# terminal would silently cap every height measurement at 25 lines.
_measurement_console = Console(
    file=io.StringIO(),
    width=MEASUREMENT_WIDTH,
    height=MEASUREMENT_HEIGHT,
    legacy_windows=False,
)


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


def measure_render_height(renderable, width):
    """Return the number of lines a renderable draws at a given width.

    The vertical counterpart of :func:`measure_table_width`, and the reason
    the layout can size a row to its content instead of splitting the screen
    down the middle. :class:`~rich.measure.Measurement` only ever speaks about
    widths, so the honest way to learn a height is to render the thing and
    count the lines it produced. Clearing the height keeps Rich from padding
    or cropping the result to a screen it is not going to.

    Parameters
    ----------
    renderable : rich.console.RenderableType
        Anything Rich can draw.
    width : int
        Width in columns the renderable is measured against. Heights depend on
        widths, so this has to be the width the caller will really give it.

    Returns
    -------
    int
        Number of lines the renderable occupies.
    """
    options = _measurement_console.options.update(width=max(1, width), height=None)
    return len(_measurement_console.render_lines(renderable, options, pad=False))


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


class FixedHeight:
    """Renders a renderable into an exact number of lines.

    The vertical half of the contract :class:`StackedPanels` relies on: Rich
    pads or crops the wrapped renderable to :attr:`height`, so whatever it
    does with the room it is handed, the stack still occupies the number of
    lines it was allocated and the panel underneath starts where it should.
    Panels honour ``options.height`` themselves, drawing both borders and
    fitting their content in between, so nothing here ever cuts a box in half.

    Parameters
    ----------
    renderable : rich.console.RenderableType
        What to draw.
    height : int
        Exact number of lines to occupy.
    """

    def __init__(self, renderable, height):
        self.renderable = renderable
        self.height = height

    def __rich_console__(self, console, options):
        """Yield exactly :attr:`height` lines of the wrapped renderable."""
        lines = console.render_lines(
            self.renderable, options.update_height(self.height), pad=True
        )
        for line in lines:
            yield from line
            yield Segment.line()


class ScrollablePanel:
    """A panel showing whichever window of its rows fits the room it gets.

    The vertical counterpart of :class:`AdaptiveRenderable`. That class reads
    ``options.max_width`` to choose the columns it can afford; this one also
    reads ``options.height`` to choose how many rows it can afford, because on
    a busy host there are simply more processes and models than there are
    lines on the screen and no layout fixes that. Rows the panel cannot show
    are counted in an indicator above and below the table, and the window
    slides over them through :data:`scroll_offsets`.

    The offset is clamped on every single render rather than when a key is
    pressed: the row count changes with every fetch, so an offset that was
    valid a second ago can point past the end of a list that has since shrunk,
    which would render an empty panel with no way back.

    Parameters
    ----------
    key : str
        Identity of the panel, used as its :data:`scroll_offsets` key and as
        its place in the Tab order.
    title : str
        Panel title, without markup.
    rows : list
        The items to list, one per table row.
    table_builder : callable
        Takes ``(rows, available_width)`` and returns the renderable listing
        exactly those rows. Called once per measurement, so it must be free of
        side effects.
    empty_message : str
        Shown instead of a table when there is nothing to list.
    """

    def __init__(self, key, title, rows, table_builder, empty_message):
        self.key = key
        self.title = title
        self.rows = list(rows)
        self.empty_message = empty_message
        self._table_builder = table_builder

    def build_table(self, available_width, rows=None):
        """Build the table listing a window of rows, all of them by default.

        Parameters
        ----------
        available_width : int
            Room the table has, in columns.
        rows : list, optional
            The window to list. Defaults to every row.

        Returns
        -------
        rich.console.RenderableType
            Whatever the table builder produced.
        """
        return self._table_builder(self.rows if rows is None else rows, available_width)

    def _rows_height(self, available_width, offset, count):
        """Height in lines of the table listing ``count`` rows from ``offset``."""
        window = self.rows[offset:offset + count]
        return measure_render_height(self.build_table(available_width, window), available_width)

    def _fit_count(self, available_width, offset, budget, anchor_at_end=False):
        """Most rows that fit ``budget`` lines, from ``offset`` or from the end.

        A binary search, which assumes the table never gets shorter as rows
        are added to it. Every rendering this dashboard uses satisfies that:
        the row-per-item tables grow by exactly one line per row, and the
        side-by-side GPU tables keep a constant height.

        Parameters
        ----------
        available_width : int
            Room the table has, in columns.
        offset : int
            Index of the first row of the window, ignored when anchoring at
            the end.
        budget : int
            Lines the table may occupy.
        anchor_at_end : bool, optional
            When ``True`` the window ends on the last row instead of starting
            at ``offset``, which is what "show me the end of the list" means.

        Returns
        -------
        int
            Number of rows, possibly zero when not even one of them fits.
        """
        total = len(self.rows)
        low, high, best = 1, total if anchor_at_end else total - offset, 0
        while low <= high:
            middle = (low + high) // 2
            start = total - middle if anchor_at_end else offset
            if self._rows_height(available_width, start, middle) <= budget:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        return best

    def _max_offset(self, available_width, available_height):
        """Furthest the window may be scrolled before it stops being full.

        Past this offset the panel would show a handful of rows over an
        expanse of nothing, so ``End`` and an overshooting ``Page Down`` both
        land here rather than on the last row.
        """
        total = len(self.rows)
        # Anchored at the end nothing is hidden below, so the only indicator
        # left to make room for is the one above.
        return max(0, total - self._fit_count(
            available_width, 0, available_height - 1, anchor_at_end=True
        ))

    def _count_at(self, available_width, available_height, offset):
        """Rows visible from ``offset``, once the indicators are paid for.

        Whether a line has to be reserved for the "more below" indicator
        depends on how many rows fit, which is what the reservation decides:
        both readings are computed and the one showing the most rows without
        contradicting itself wins.
        """
        total = len(self.rows)
        reserve_above = 1 if offset > 0 else 0
        best, consistent_best = 0, 0
        for reserve_below in (1, 0):
            count = self._fit_count(
                available_width, offset, available_height - reserve_above - reserve_below
            )
            best = max(best, count)
            if count and (offset + count < total) == bool(reserve_below):
                consistent_best = max(consistent_best, count)
        return consistent_best or best

    def _window(self, available_width, available_height, offset):
        """Resolve the window of rows to draw.

        Parameters
        ----------
        available_width : int
            Room the table has, in columns.
        available_height : int or None
            Lines the panel's content may occupy, or ``None`` when Rich has
            not constrained the height at all.
        offset : int
            Requested index of the first visible row.

        Returns
        -------
        tuple of (int, int, int, int)
            The clamped offset, how many rows are shown, how many are hidden
            above and how many below.
        """
        total = len(self.rows)
        if available_height is None or not total:
            return 0, total, 0, 0
        if self._rows_height(available_width, 0, total) <= available_height:
            # Everything fits: there is nothing to scroll and no offset worth
            # keeping, so the panel always shows the top of the list.
            return 0, total, 0, 0

        offset = min(max(offset, 0), self._max_offset(available_width, available_height))
        count = self._count_at(available_width, available_height, offset)
        return offset, count, offset, max(0, total - offset - count)

    def render_panel(self, width, height, publish=False):
        """Build the panel for an exactly known width and height.

        Parameters
        ----------
        width : int
            Total width of the panel, borders included.
        height : int or None
            Total height of the panel, borders included, or ``None`` to let it
            be as tall as its content.
        publish : bool, optional
            Whether to write the resolved geometry back into the shared state
            the keyboard thread reads. Only the real render pass does. The
            measurement passes must not: :meth:`natural_height` builds the
            panel unconstrained, where nothing is ever hidden, so publishing
            from there would reset every offset to zero on each frame.

        Returns
        -------
        rich.panel.Panel
            The panel, whole, at exactly the requested height.
        """
        with state_lock:
            offset = scroll_offsets.get(self.key, 0)
            focused = focused_panel == self.key

        border_style = FOCUSED_PANEL_BORDER_STYLE if focused else PANEL_BORDER_STYLE
        panel_options = {
            "title": f"[bold cyan]{self.title}[/bold cyan]",
            "border_style": border_style,
            "padding": (0, 1),
            "height": height,
        }

        if not self.rows:
            return Panel(self.empty_message, **panel_options)

        content_width = max(1, width - PANEL_CHROME_WIDTH)
        content_height = None if height is None else max(1, height - PANEL_CHROME_HEIGHT)
        offset, count, hidden_above, hidden_below = self._window(
            content_width, content_height, offset
        )

        if publish:
            with state_lock:
                scroll_offsets[self.key] = offset
                panel_row_counts[self.key] = len(self.rows)
                # A page is the window as it is at the top of the list, where
                # no indicator above eats a line. Publishing the window as
                # rendered instead would make it one row shorter as soon as
                # anything is hidden above, and Page Up would then land one
                # row below wherever Page Down came from.
                panel_page_sizes[self.key] = count + (1 if hidden_above else 0)

        if not count:
            # Too short for even one row and its indicators. Saying how much
            # is there beats drawing a table the panel would have to cut.
            return Panel(_overflow_indicator("↕", len(self.rows), "hidden"), **panel_options)

        parts = []
        if hidden_above:
            parts.append(_overflow_indicator("↑", hidden_above, "above"))
        # The table itself still sizes its columns from the width Rich hands
        # it, exactly as it did before there was a vertical axis to worry
        # about; only which rows reach it is decided here.
        window = self.rows[offset:offset + count]
        parts.append(AdaptiveRenderable(lambda w: self.build_table(w, window)))
        if hidden_below:
            parts.append(_overflow_indicator("↓", hidden_below, "below"))

        return Panel(Group(*parts), **panel_options)

    def natural_height(self, width):
        """Height the panel would take if nothing constrained it."""
        return measure_render_height(self.render_panel(width, None), width)

    def minimum_height(self):
        """Shortest this panel is still worth drawing."""
        return MIN_SCROLLABLE_PANEL_HEIGHT

    def __rich_console__(self, console, options):
        """Render the panel into the room Rich has just revealed."""
        yield self.render_panel(options.max_width, options.height, publish=True)


def _overflow_indicator(arrow, count, where):
    """Build the one-line notice naming rows the panel could not show."""
    plural = "" if count == 1 else "s"
    return Text(
        f"{arrow} {count} more row{plural} {where}",
        style="dim italic",
        justify="center",
        no_wrap=True,
        overflow="ellipsis",
    )


def panel_natural_height(renderable, width):
    """Height a stacked panel takes when nothing constrains it.

    Panels that size themselves answer for themselves; anything else is
    measured by rendering it (see :func:`measure_render_height`).
    """
    if hasattr(renderable, "natural_height"):
        return renderable.natural_height(width)
    return measure_render_height(renderable, width)


def panel_minimum_height(renderable):
    """Shortest a stacked panel may be squeezed to before it is dropped."""
    if hasattr(renderable, "minimum_height"):
        return renderable.minimum_height()
    return MIN_PLAIN_PANEL_HEIGHT


def distribute_heights(naturals, minimums, available):
    """Share the lines of a column between the panels stacked in it.

    Free space is never handed out. A natural height is already the height at
    which a panel shows every row it has, so stretching one past it buys no
    information: it only draws a border around blank lines. The lines a column
    does not need are left unpainted instead, which is the whole reason a
    column is one region holding a stack rather than one region per panel.

    Room only has to be *taken* from a panel, and it is taken from whichever
    one has the most of it above its own minimum, so the crowded panels give
    way before the ones already down to a header and a row. That is also what
    hands a growable panel the lines its neighbours do not use: a stack of a
    four-row table and a fifty-row one settles with the short one whole and
    the long one filling the rest of the column.

    Parameters
    ----------
    naturals : list of int
        Height each panel would take unconstrained.
    minimums : list of int
        Height below which each panel is not worth drawing.
    available : int
        Lines the column has.

    Returns
    -------
    list of int
        One height per panel, never summing to more than ``available``, and
        summing to less whenever the panels' own content does not fill the
        column. Shorter than ``naturals`` when the column is too short to hold
        every panel, the trailing ones being dropped.
    """
    if not naturals:
        return []

    sizes = list(naturals)
    total = sum(sizes)
    if total <= available:
        # Everything fits at the height of its own content. Whatever is left
        # of the column stays blank rather than inflating the last panel.
        return sizes

    # Over-subscribed. Lines are taken back from whichever panel has the most
    # room above its own minimum, so the crowded panels give way before the
    # ones already down to a header and a row.
    while total > available:
        index = max(range(len(sizes)), key=lambda position: sizes[position] - minimums[position])
        if sizes[index] <= minimums[index]:
            break
        sizes[index] -= 1
        total -= 1

    # Not even the minimums fit: a terminal this short drops the trailing
    # panels rather than draw a column of half-boxes.
    while total > available and len(sizes) > 1:
        total -= sizes.pop()
    if total > available:
        sizes[0] = max(0, available)
    return sizes


class StackedPanels:
    """Stacks panels in one layout region, each at the height it needs.

    This is what replaced the fixed 1:1 row split. A region used to be handed
    a :class:`~rich.console.Group` of panels, and Rich passed the region's
    full height to every one of them: two panels each drew a region's worth of
    lines, the second half of the result fell off the bottom, and the panel it
    was cut through lost its closing border with nothing raised and nothing
    logged. Here every panel is measured first and then pinned to the height
    it was allocated, so the stack always occupies exactly its region.

    Parameters
    ----------
    *panels : rich.console.RenderableType
        The panels to stack, top to bottom.
    """

    def __init__(self, *panels):
        self.panels = tuple(panels)

    def natural_height(self, width):
        """Combined height of the stacked panels, unconstrained."""
        return sum(panel_natural_height(panel, width) for panel in self.panels)

    def minimum_height(self):
        """Combined height below which panels start being dropped."""
        return sum(panel_minimum_height(panel) for panel in self.panels)

    def __rich_console__(self, console, options):
        """Allocate the region's lines and render each panel into its share."""
        width = options.max_width
        height = options.height if options.height is not None else options.max_height
        naturals = [panel_natural_height(panel, width) for panel in self.panels]
        # A minimum above the natural height would inflate a panel that has
        # nothing to show, so it is capped by what the panel actually needs.
        minimums = [
            min(panel_minimum_height(panel), natural)
            for panel, natural in zip(self.panels, naturals, strict=True)
        ]
        sizes = distribute_heights(naturals, minimums, height)
        yield Group(*[
            FixedHeight(panel, size)
            for panel, size in zip(self.panels, sizes, strict=False)
            if size > 0
        ])


class SideBySidePanels:
    """Renders panels next to each other, each as tall as it needs to be.

    A :class:`~rich.table.Table` grid would do the horizontal split just as
    well, but it renders each cell at that cell's own natural height and then
    pads the row to the tallest, which puts the shorter panel's border in the
    wrong place and lets the taller one be cropped by whatever sits above.
    Here every panel is handed the row's height capped by its own content, so
    a ranking of two processes closes right under its second row instead of
    framing the blank lines its neighbour needed.

    Parameters
    ----------
    *panels : rich.console.RenderableType
        The panels to lay out, left to right.
    """

    def __init__(self, *panels):
        self.panels = tuple(panels)

    def _widths(self, width):
        """Split a width into equal shares, the remainder going leftwards."""
        count = len(self.panels)
        base, extra = divmod(max(width, count), count)
        return [base + (1 if index < extra else 0) for index in range(count)]

    def natural_height(self, width):
        """Height of the tallest panel, which is what the row needs."""
        return max(
            panel_natural_height(panel, share)
            for panel, share in zip(self.panels, self._widths(width), strict=True)
        )

    def minimum_height(self):
        """Height of the most demanding panel of the row."""
        return max(panel_minimum_height(panel) for panel in self.panels)

    def __rich_console__(self, console, options):
        """Render every panel at its own height and stitch the lines."""
        widths = self._widths(options.max_width)
        columns = []
        for panel, share in zip(self.panels, widths, strict=True):
            # Never taller than the row it was given, never taller than what
            # it has to show. The cap is what keeps the shorter of two
            # rankings from being padded out to its neighbour's row count.
            height = panel_natural_height(panel, share)
            if options.height is not None:
                height = min(height, options.height)
            columns.append(
                console.render_lines(panel, options.update(width=share, height=height), pad=True)
            )
        # The columns rarely have the same number of lines; the shorter ones
        # are padded with blanks below their closing border so the row stays
        # rectangular for whatever is stacked underneath it.
        line_count = max(len(column) for column in columns)
        for index in range(line_count):
            for column, share in zip(columns, widths, strict=True):
                yield from column[index] if index < len(column) else [Segment(" " * share)]
            yield Segment.line()


def layout_mode_for_width(width):
    """Pick the layout mode a terminal of a given width calls for.

    Parameters
    ----------
    width : int
        Terminal width in columns.

    Returns
    -------
    str
        :data:`LAYOUT_MODE_WIDE` from :data:`WIDE_LAYOUT_MIN_WIDTH` columns on,
        :data:`LAYOUT_MODE_NARROW` below it.
    """
    return LAYOUT_MODE_WIDE if width >= WIDE_LAYOUT_MIN_WIDTH else LAYOUT_MODE_NARROW


def layout_shape(data, width):
    """Return the structure the dashboard needs for a payload and a width.

    The two things the ``Layout`` object itself depends on, and the only two
    the main loop has to compare before deciding to rebuild it. Everything
    else, including which panel goes where and how wide each column is, is
    settled at render time.

    Parameters
    ----------
    data : dict or None
        The ``/stats`` payload, or ``None`` before the first successful fetch.
    width : int
        Terminal width in columns.

    Returns
    -------
    tuple of (str, bool)
        The layout mode and whether there is more than one GPU to detail.
    """
    payload = data or {}
    gpus = payload.get("gpu") or []
    return layout_mode_for_width(width), bool(payload.get("has_gpu")) and len(gpus) > 1


def terminal_width():
    """Return the width of the terminal the dashboard draws into.

    Split out so the main loop reads the real console instead of guessing, and
    so the layout mode can be driven from a test without a terminal. This is
    the one width the dashboard is allowed to look up rather than measure: it
    selects the *structure*, which has to exist before Rich can measure
    anything inside it. Every panel still sizes itself from the room it is
    actually given.
    """
    return console.size.width


def create_layout(mode=LAYOUT_MODE_NARROW, multi_gpu=False):
    """Create the region tree backing the dashboard, for one layout shape.

    Two shapes exist. The narrow one is the 2x2 grid the dashboard has always
    had, split into columns rather than rows so that each column can size its
    own panels to their content; the wide one lays every panel out in a single
    row of columns, which is what an ultrawide terminal has the room for.

    In both cases the regions only carry the horizontal split. The vertical
    one lives inside :class:`StackedPanels`, which is the only place that
    knows how tall each panel's content actually is.

    Parameters
    ----------
    mode : str, optional
        :data:`LAYOUT_MODE_NARROW` or :data:`LAYOUT_MODE_WIDE`.
    multi_gpu : bool, optional
        Whether there is more than one GPU. Only the wide layout cares: with a
        single card there is no separate GPU detail panel, since the per-GPU
        table lives inside the summary, so the fourth column would be empty
        and the layout drops to three.

    Returns
    -------
    rich.layout.Layout
        Root layout. Region names are unique across the whole tree, because
        Rich resolves ``layout["name"]`` by searching it.
    """
    layout = Layout()

    if mode == LAYOUT_MODE_WIDE:
        names = ["wide_vram", "wide_processes", "wide_summary"]
        if multi_gpu:
            names.append("wide_gpu_detail")
        layout.split_row(*[Layout(name=name, ratio=1) for name in names])
        return layout

    # Narrow: two columns, their ratio set per mode at render time.
    layout.split_row(
        Layout(name="narrow_left", ratio=1),
        Layout(name="narrow_right", ratio=2),
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
    ScrollablePanel
        The GPU detail panel, whose content is chosen from the width Rich
        hands it (see :func:`build_gpu_detail_content`) and whose cards are
        windowed to the height it is given.
    """
    # A host reporting no GPU at all gets the empty state, whatever leftover
    # the payload happens to carry in its ``gpu`` key.
    gpus = (data.get('gpu') or []) if data.get('has_gpu') else []

    return ScrollablePanel(
        PANEL_GPU_DETAIL,
        "GPU Detail",
        gpus,
        build_gpu_detail_content,
        "No GPU detected.",
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
    ScrollablePanel
        The ranking panel, showing an explanatory line when there is no data.
        ``key`` doubles as the panel's scroll identity, since the two rankings
        are exactly the two panels a user tells apart by the metric they rank.
    """
    return ScrollablePanel(
        key,
        title,
        processes,
        lambda rows, width: build_process_table(rows, key, width),
        f"No data for {title}.",
    )


def build_processes_panel(data):
    """Build the pair of Top CPU and Top Memory rankings, side by side.

    Parameters
    ----------
    data : dict
        The ``/stats`` payload.

    Returns
    -------
    SideBySidePanels
        The two ranking panels sharing a row. Each sizes its own table from
        the half of the region Rich gives it, so neither has to guess how the
        row was split, and both are rendered at the row's full height so
        neither can be cut by the other.
    """
    return SideBySidePanels(
        build_process_panel(data.get('top_cpu', []), PANEL_TOP_CPU, 'Top CPU'),
        build_process_panel(data.get('top_memory', []), PANEL_TOP_MEMORY, 'Top Memory'),
    )


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
    ScrollablePanel
        The GPU processes panel, showing an explanatory line when there is
        nothing holding VRAM.
    """
    return ScrollablePanel(
        PANEL_GPU_PROCESSES,
        "GPU Processes",
        data.get("top_gpu_processes", []),
        lambda rows, width: build_gpu_processes_table(rows, multi_gpu, width),
        "No GPU processes.",
    )


def format_context_length(value):
    """Format an Ollama context length for display.

    Exact multiples of 1024 (and of 1024²) are abbreviated, since that is how
    context windows are usually quoted; anything else is shown raw rather than
    rounded, because a truncated context length is misleading.

    Anything that is not a positive whole number renders ``"N/A"``: a zero or
    negative context window is meaningless, and so is a value the server did
    not send. This mirrors ``formatContextLength`` in the web UI, which reads
    the same field of the same payload.

    Parameters
    ----------
    value : int or str or None
        The ``context_length`` reported by ``/api/ps``, or ``None`` when the
        Ollama server is too old to expose it.

    Returns
    -------
    str
        ``"1M"``, ``"32K"``, the raw integer, or ``"N/A"``.
    """
    try:
        context_length = int(value)
    except (TypeError, ValueError):
        return "N/A"

    if context_length <= 0:
        return "N/A"

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
    ScrollablePanel
        The Ollama panel, showing an explanatory line when nothing is loaded.
    """
    models = data.get("ollama_processes", {}).get("models", [])
    has_ollama_process, gpu_label = ollama_gpu_indices(gpu_processes)

    return ScrollablePanel(
        PANEL_OLLAMA,
        "Ollama Statistics",
        models,
        lambda rows, width: build_ollama_table(rows, has_ollama_process, gpu_label, width),
        "No Ollama models.",
    )


def register_scrollable_panels(panels):
    """Publish the panels the scroll keys can drive, and keep focus valid.

    Which panels exist depends on the payload and on the layout mode, and a
    panel with nothing to list has no window to move, so the Tab order is
    rebuilt on every render pass rather than fixed once. A focus left pointing
    at a panel that has since gone away falls back to the first one, and the
    offsets of vanished panels are dropped so a panel coming back starts at
    the top rather than wherever it was left months of uptime ago.

    Parameters
    ----------
    panels : list of ScrollablePanel
        Every scrollable panel currently in the layout, in Tab order.

    Returns
    -------
    None
    """
    global focused_panel

    keys = [panel.key for panel in panels if panel.rows]
    with state_lock:
        scrollable_panel_keys[:] = keys
        if focused_panel not in keys:
            focused_panel = keys[0] if keys else None
        for stale in [key for key in scroll_offsets if key not in keys]:
            del scroll_offsets[stale]


def build_layout_content(layout, data, interval, mode=LAYOUT_MODE_NARROW):
    """Fill the layout regions with the fetched data.

    The routing depends on the number of GPUs. With zero or one card the
    per-GPU detail fits inside the summary panel, so there is no separate
    detail panel at all. From two cards on, the detail gets its own region,
    the GPU processes join Ollama in the same column (they describe the same
    VRAM), and the summary keeps only cumulated figures.

    No size is threaded through: every panel sizes itself from the room Rich
    hands it at render time (see :class:`AdaptiveRenderable` for the columns
    and :class:`ScrollablePanel` for the rows), which is the only figure that
    accounts for the layout ratios *and* the panel chrome.

    Parameters
    ----------
    layout : rich.layout.Layout
        The layout built by :func:`create_layout`; updated in place. Its mode
        must match ``mode``, which is what the main loop compares before
        deciding to rebuild it.
    data : dict
        The ``/stats`` payload.
    interval : int
        Current refresh interval in seconds.
    mode : str, optional
        :data:`LAYOUT_MODE_NARROW` or :data:`LAYOUT_MODE_WIDE`.

    Returns
    -------
    None
    """
    gpus = data.get("gpu") or []
    multi_gpu = bool(data.get("has_gpu")) and len(gpus) > 1

    gpu_processes = data.get("top_gpu_processes") or []
    ollama_panel = build_ollama_panel(data, gpu_processes=gpu_processes)
    gpu_processes_panel = build_gpu_processes_panel(data, multi_gpu=multi_gpu)
    # Built as a pair even in wide mode, where the two rankings are stacked
    # rather than laid side by side: the panels themselves are the same
    # objects either way, only the arrangement differs.
    rankings = build_processes_panel(data)
    summary_panel = build_summary(data, interval, multi_gpu=multi_gpu)

    # Tab order: down the leftmost column first, then across. It follows the
    # reading order of both layouts, which is the only order a user can guess.
    scrollable = [ollama_panel, gpu_processes_panel, *rankings.panels]

    if mode == LAYOUT_MODE_WIDE:
        layout["wide_vram"].update(StackedPanels(ollama_panel, gpu_processes_panel))
        layout["wide_processes"].update(StackedPanels(*rankings.panels))
        layout["wide_summary"].update(StackedPanels(summary_panel))
        if multi_gpu:
            gpu_detail_panel = build_gpu_detail_panel(data)
            scrollable.append(gpu_detail_panel)
            layout["wide_gpu_detail"].update(StackedPanels(gpu_detail_panel))
    elif multi_gpu:
        # The left column carries three stacked panels, hence a bit more room.
        layout["narrow_left"].ratio = 2
        layout["narrow_right"].ratio = 3
        gpu_detail_panel = build_gpu_detail_panel(data)
        scrollable.append(gpu_detail_panel)
        layout["narrow_left"].update(
            StackedPanels(ollama_panel, gpu_processes_panel, summary_panel)
        )
        layout["narrow_right"].update(StackedPanels(rankings, gpu_detail_panel))
    else:
        layout["narrow_left"].ratio = 1
        layout["narrow_right"].ratio = 2
        layout["narrow_left"].update(StackedPanels(ollama_panel, gpu_processes_panel))
        layout["narrow_right"].update(StackedPanels(rankings, summary_panel))

    register_scrollable_panels(scrollable)


def build_full_screen_help():
    """Builds the full-screen help panel.

    The only place a user ever learns the keys, so every one of them is listed
    here, including the ones that only do something once a panel is focused.
    """
    help_text = """
[bold yellow]Keyboard Shortcuts:[/bold yellow]

[bold green]q[/bold green] - Quit
[bold green]r[/bold green] - Refresh
[bold green]h[/bold green] - Show/Hide help
[bold green]p[/bold green] - Pause/Resume
[bold green]-[/bold green] - Decrease interval
[bold green]+[/bold green] - Increase interval

[bold yellow]Scrolling:[/bold yellow]

[bold green]Tab[/bold green]         - Focus the next panel (highlighted border)
[bold green]Up/Down[/bold green]     - Scroll the focused panel by one row
[bold green]PgUp/PgDn[/bold green]   - Scroll the focused panel by one screenful
[bold green]Home/End[/bold green]    - Jump to the first or last row

A panel hiding rows says so above and below its table.

Press [bold green]h[/bold green] again to return.
"""

    help_panel = Panel.fit(
        help_text,
        title="[bold cyan]Help[/bold cyan]",
        border_style="green",
        padding=(1, 2)
    )
    return help_panel


def cycle_panel_focus(step=1):
    """Move the focus to the next scrollable panel.

    The caller must hold :data:`state_lock`.

    Parameters
    ----------
    step : int, optional
        How many panels to move by, negative to go backwards. Wraps around.

    Returns
    -------
    None
    """
    global focused_panel

    if not scrollable_panel_keys:
        focused_panel = None
    elif focused_panel in scrollable_panel_keys:
        position = scrollable_panel_keys.index(focused_panel)
        focused_panel = scrollable_panel_keys[(position + step) % len(scrollable_panel_keys)]
    else:
        focused_panel = scrollable_panel_keys[0]


def scroll_focused_panel(rows=0, pages=0, to=None):
    """Move the focused panel's window over its rows.

    The caller must hold :data:`state_lock`. Only a coarse clamp is applied
    here, to the last row: the exact one depends on how many rows the panel
    can show, which is not known until it is rendered against a width and a
    height. :class:`ScrollablePanel` clamps properly and writes the result
    back, so a key that overshoots settles on the next frame.

    Parameters
    ----------
    rows : int, optional
        Rows to move by, negative to go up.
    pages : int, optional
        Screenfuls to move by. A screenful is however many rows the panel fits
        when it is scrolled to the top, so paging back always lands where
        paging forward came from.
    to : int, optional
        Absolute offset to jump to, overriding ``rows`` and ``pages``.

    Returns
    -------
    None
    """
    if focused_panel is None:
        return

    total = panel_row_counts.get(focused_panel, 0)
    if to is None:
        # A panel that has never been rendered has no page size yet; one row
        # at a time is the safe reading, and the next frame fixes it.
        page = panel_page_sizes.get(focused_panel, 1) or 1
        offset = scroll_offsets.get(focused_panel, 0) + rows + pages * page
    else:
        offset = to

    scroll_offsets[focused_panel] = max(0, min(offset, max(0, total - 1)))


def keyboard_listener():
    """Listens for keyboard input and modifies state accordingly.

    The scrolling keys deliberately do not set :data:`rebuild_layout_event`.
    They change nothing about *what* is on screen, only which slice of it is
    drawn, and the offsets are read back by the panels on every render, so the
    ``Live`` instance's own refresh picks them up within its next frame. Waking
    the main loop would fire an HTTP request per key press for nothing.
    """
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
            elif key == readchar_key.TAB:
                cycle_panel_focus()
            elif key in SCROLL_UP_KEYS:
                scroll_focused_panel(rows=-1)
            elif key in SCROLL_DOWN_KEYS:
                scroll_focused_panel(rows=1)
            elif key in SCROLL_PAGE_UP_KEYS:
                scroll_focused_panel(pages=-1)
            elif key in SCROLL_PAGE_DOWN_KEYS:
                scroll_focused_panel(pages=1)
            elif key in SCROLL_HOME_KEYS:
                scroll_focused_panel(to=0)
            elif key in SCROLL_END_KEYS:
                # Past the last row on purpose: the render clamps it down to
                # the offset that fills the window with the final rows.
                scroll_focused_panel(to=panel_row_counts.get(focused_panel, 0))


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

    # The structure the very first frame needs. It is re-derived on every
    # iteration and the layout is only rebuilt when it actually changes.
    shape = layout_shape(None, terminal_width())
    layout = create_layout(*shape)

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
                with stats_lock:
                    stats_snapshot = latest_stats

                # A terminal that was resized past the wide threshold, or a
                # card that appeared or went away, needs a different region
                # tree. Everything else is handled by the panels themselves,
                # so the ``Layout`` is only rebuilt on an actual change of
                # shape rather than on every iteration.
                new_shape = layout_shape(stats_snapshot, terminal_width())
                reshaped = new_shape != shape
                if reshaped:
                    shape = new_shape
                    layout = create_layout(*shape)

                if previous_help_flag or reshaped:
                    # Coming back from the help screen, or onto a layout that
                    # did not exist a moment ago: either way the ``Live``
                    # instance is the thing that decides what gets drawn, so
                    # the new object has to be pushed through it.
                    live.update(layout)

                if stats_snapshot:
                    build_layout_content(layout, stats_snapshot, current_interval, shape[0])

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
