"""Tests for the Rich panel builders of the terminal dashboard.

The builders are pure: they take a ``/stats`` payload and return renderables, so
they can be asserted on without a terminal. Structure is checked wherever it is
enough (how many tables, which columns); the assembled layout is rendered to
plain text when the question is whether it physically fits the screen.

Panels do not choose their columns when they are built, they choose them when
Rich tells them how wide they are, so a panel's table is obtained through
``panel.renderable.build(width)`` (see :class:`sys_stats.cli.AdaptiveRenderable`).
"""

import functools
import io
import re
import sys

import pytest
from rich.console import Console
from rich.table import Table

from sys_stats import cli
from sys_stats.cli import (
    OLLAMA_ROW_STYLE,
    build_full_screen_help,
    build_gpu_detail_panel,
    build_gpu_processes_panel,
    build_gpu_processes_table,
    build_gpu_rows_table,
    build_gpu_summary,
    build_gpu_totals,
    build_layout_content,
    build_ollama_panel,
    build_process_panel,
    build_process_table,
    build_processes_panel,
    build_single_gpu_table,
    build_summary,
    create_layout,
    format_context_length,
    gpu_memory_total_bytes,
)

# Every terminal width the assembled layout is swept over. 60 is narrower than
# anything usable and 260 is wider than an ultrawide terminal, so the range
# brackets reality on both sides.
SWEEP_WIDTHS = range(60, 261)

# The two dimensional sweep is sampled rather than exhaustive: the full cross
# product is 201 widths x 41 heights x 5 payloads, which is over forty thousand
# full dashboard renders and minutes of wall clock. Widths are sampled every 10
# columns and heights every 4 lines, for 21 x 11 = 231 renders per payload.
# Sampling is safe on the width axis because the one dimensional sweep above
# still walks every single column; on the height axis because a panel's height
# only ever changes by whole rows, so four lines cannot hide a transition that
# the neighbouring samples do not also show.
SWEEP_LAYOUT_WIDTHS = range(60, 261, 10)
SWEEP_LAYOUT_HEIGHTS = range(20, 61, 4)

# Openers and their closers, for the box-integrity check below. A table that
# overflows the panel it lives in keeps the character that starts its border
# and loses the one that ends it.
BOX_PAIRS = (
    ("┏", "┓"),
    ("┡", "┩"),
    ("┗", "┛"),
    ("└", "┘"),
    ("┌", "┐"),
    ("╭", "╮"),
    ("╰", "╯"),
)


def _gpu(gpu_id: int, name: str) -> dict:
    """Build a GPU entry shaped like the one ``/stats`` returns."""
    return {
        "id": gpu_id,
        "name": name,
        "load": 42.0,
        "memoryUsed": 12288 * 1024 * 1024,
        "memoryPercent": 50.0,
        "temperature": 61.0,
        "fanSpeed": 30.0,
        "powerDraw": 220.0,
    }


def _null_gpu() -> dict:
    """A card whose driver reports nothing but its name.

    ``nvidia-smi`` leaves every figure at ``null`` on a GPU it cannot query
    (a card in an exotic power state, a container without the right
    capabilities), and the payload carries those nulls through untouched.
    """
    return {
        "id": 0,
        "name": "NVIDIA GeForce RTX 3090",
        "load": None,
        "memoryUsed": None,
        "memoryTotal": None,
        "memoryPercent": None,
        "temperature": None,
        "fanSpeed": None,
        "powerDraw": None,
    }


def _ollama_models() -> list:
    """Four loaded models, covering every shape of the Ollama payload."""
    return [
        {
            "name": "qwen3-coder:30b",
            "size": 19_000_000_000,
            "size_vram": 19_000_000_000,
            "expires_at": "2099-01-01T00:00:00Z",
            "context_length": 32768,
        },
        {
            "name": "llama3.3:70b-instruct-q4_K_M",
            "size": 43_000_000_000,
            "size_vram": 21_000_000_000,
            "expires_at": "2099-01-01T00:00:00Z",
            "context_length": 131072,
        },
        # No context_length at all: an Ollama older than 0.32.
        {
            "name": "nomic-embed-text",
            "size": 274_000_000,
            "size_vram": 274_000_000,
            "expires_at": "2099-01-01T00:00:00Z",
        },
        # Evicted from VRAM, and a context window that is not a power of two.
        {
            "name": "mistral-small:24b",
            "size": 14_000_000_000,
            "size_vram": 0,
            "expires_at": "2099-01-01T00:00:00Z",
            "context_length": 5000,
        },
    ]


def _gpu_processes() -> list:
    """Compute apps holding VRAM, two of them owned by Ollama."""
    return [
        {
            "pid": 4242, "name": "ollama", "memory_used": 17_825_792_000,
            "cmdline": "/usr/local/bin/ollama runner --model qwen3-coder",
            "gpu_uuid": "GPU-0", "gpu_index": 0,
        },
        {
            "pid": 4242, "name": "ollama", "memory_used": 9_437_184_000,
            "cmdline": "/usr/local/bin/ollama runner --model llama3.3",
            "gpu_uuid": "GPU-2", "gpu_index": 2,
        },
        {
            "pid": 8888, "name": "python3", "memory_used": 3_145_728_000,
            "cmdline": "python3 train.py --epochs 40 --batch 8",
            "gpu_uuid": "GPU-1", "gpu_index": 1,
        },
        {
            "pid": 9999, "name": "blender", "memory_used": 1_258_291_200,
            "cmdline": "blender -b scene.blend -f 120",
            "gpu_uuid": None, "gpu_index": None,
        },
    ]


def _stats(gpu_count: int) -> dict:
    """Build a realistic ``/stats`` payload with the requested number of GPUs."""
    return {
        "cpu": 37.4,
        "ram": {"percent": 61.2, "total": 128 * 1024**3},
        "has_gpu": gpu_count > 0,
        "gpu": [_gpu(index, f"NVIDIA GeForce RTX {3090 + index}") for index in range(gpu_count)],
        "top_cpu": [
            {"pid": 4242, "name": "ollama", "cpu_percent": 412.0,
             "cmdline": "/usr/local/bin/ollama serve"},
            {"pid": 8888, "name": "python3", "cpu_percent": 98.6,
             "cmdline": "python3 train.py --epochs 40 --batch 8"},
            {"pid": 771, "name": "kworker/u64:3", "cpu_percent": 12.1, "cmdline": "N/A"},
        ],
        "top_memory": [
            {"pid": 4242, "name": "ollama", "memory_percent": 29.8,
             "cmdline": "/usr/local/bin/ollama serve"},
            {"pid": 3120, "name": "postgres", "memory_percent": 6.0,
             "cmdline": "postgres: writer process"},
        ],
        "top_gpu_processes": _gpu_processes() if gpu_count else [],
        "ollama_processes": {"models": _ollama_models()},
    }


def _busy_stats(gpu_count: int) -> dict:
    """A ``/stats`` payload from a host with more rows than any screen holds.

    Eight loaded models and eight compute apps is what a saturated inference
    box actually looks like, and it is the payload that used to push the
    stacked left column past its half of the screen.
    """
    data = _stats(gpu_count)
    models = _ollama_models()
    data["ollama_processes"]["models"] = models + [
        dict(model, name=f"{model['name']}-b") for model in models
    ]
    processes = _gpu_processes()
    data["top_gpu_processes"] = processes + [
        dict(proc, pid=proc["pid"] + 1) for proc in processes
    ]
    return data


def _expected_panel_count(data: dict) -> int:
    """Number of panels the assembled dashboard must draw for a payload.

    Ollama, GPU Processes, Top CPU, Top Memory and the summary are always
    there; the GPU Detail panel only exists once there is more than one card
    to detail. A render holding fewer boxes than this has lost a panel
    outright, which is what vertical clipping does before it ever gets around
    to cutting a border in half.
    """
    gpus = data.get("gpu") or []
    return 6 if data.get("has_gpu") and len(gpus) > 1 else 5


def _render(renderable, width: int = 200, height: int = 50) -> str:
    """Render a renderable to plain text.

    The console writes to a :class:`io.StringIO`, so Rich emits no ANSI
    escapes and the result can be asserted on character by character.
    """
    console = Console(file=io.StringIO(), width=width, height=height, legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


def _headers(table: Table) -> list:
    """Return the header of every column of a table."""
    return [column.header for column in table.columns]


def _table(panel, width: int = 200) -> Table:
    """Build the table a panel renders into, for a given panel width.

    Parameters
    ----------
    panel : sys_stats.cli.ScrollablePanel
        Any panel of the dashboard that lists rows.
    width : int, optional
        Width in columns handed to the builder. Note this is the width of the
        table itself, not of the panel: Rich deducts the panel's borders and
        padding before the renderable ever sees it.
    """
    return panel.build_table(width)


def _has_blank_row(text: str) -> bool:
    """Detect a table row made only of box-drawing bars and whitespace.

    That shape is the signature of a starved column that wrapped its
    (already truncated) text onto extra physical lines: the row grows taller
    than one line, but every other cell only has content on the first line,
    leaving the rest looking like a fully blank row.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and set(stripped) <= {"│", " "}:
            return True
    return False


def _clipped_lines(text: str) -> list:
    """Return the rendered lines holding a box that never closes.

    When a table is wider than the panel it sits in, Rich draws it anyway and
    the enclosing panel cuts off whatever ran past its right edge. The table
    keeps the character that opens each of its border lines and loses the one
    that closes it, so an unbalanced line is the fingerprint of an overflow.
    Both counts and order are checked: a line may legitimately hold several
    boxes side by side, but never an opener with no closer after it.
    """
    clipped = []
    for line in text.splitlines():
        for opener, closer in BOX_PAIRS:
            unbalanced = line.count(opener) != line.count(closer)
            unclosed = opener in line and line.rfind(opener) > line.rfind(closer)
            if unbalanced or unclosed:
                clipped.append(line)
                break
    return clipped


def _boxed_blank_lines(text: str) -> list:
    """Return the rendered lines that are empty inside a box.

    The measure of framed void, and the one the earlier probe got wrong: it
    counted trailing whitespace after ``rstrip``, which strips nothing at all
    when a panel border sits at the right edge of the screen, and so reported
    a perfect score on the very dimension the dashboard was failing.

    A line counts here when it carries at least one vertical border and no
    other ink: the reader sees the sides of one or more boxes and nothing
    between them. Lines of plain background are not counted, because an
    unpainted remainder is the point rather than the problem.

    Parameters
    ----------
    text : str
        Plain text rendering of the assembled layout.

    Returns
    -------
    list of str
        The offending lines, in render order.
    """
    blanks = []
    for line in text.splitlines():
        ink = line.replace(" ", "")
        if ink and set(ink) <= {"│", "┃", "║"}:
            blanks.append(line)
    return blanks


def _panel_boxes(text: str) -> list:
    """Locate every panel drawn in a rendered layout.

    Panels are the only boxes Rich draws with rounded corners here, tables
    using the square and heavy sets, so the corner characters are enough to
    tell a panel from its content. A panel is found by pairing the ``╭`` and
    ``╮`` of its top border and then walking down for the line closing the
    same two columns.

    Parameters
    ----------
    text : str
        Plain text rendering of the assembled layout.

    Returns
    -------
    list of tuple of (int, int, int, int)
        One ``(top, bottom, left, right)`` per panel, in line and column
        indices into the rendered text.
    """
    lines = text.splitlines()
    boxes = []
    for top, line in enumerate(lines):
        start = line.find("╭")
        while start != -1:
            end = line.find("╮", start)
            if end == -1:
                break
            for bottom in range(top + 1, len(lines)):
                row = lines[bottom]
                if len(row) > end and row[start] == "╰" and row[end] == "╯":
                    boxes.append((top, bottom, start, end))
                    break
            start = line.find("╭", end)
    return boxes


def _blank_panel_interiors(text: str, titled_only: bool = True) -> list:
    """Return the blank lines each panel draws between its own borders.

    Sharper than :func:`_boxed_blank_lines`, which only sees a screen line
    where *every* box happens to be empty at once: a panel stretched to four
    times its content is invisible to that count as soon as a neighbouring
    column still has rows to draw on the same lines. This one reads each
    panel's own span, so it catches the stretch wherever it happens.

    Parameters
    ----------
    text : str
        Plain text rendering of the assembled layout.
    titled_only : bool, optional
        Whether to ignore the untitled panels. Only the summary is untitled,
        and it deliberately draws a spacer row and a slot for the ``PAUSED``
        marker, which are content rather than stretch.

    Returns
    -------
    list of str
        One ``"line,left-right"`` locator per blank interior line.
    """
    lines = text.splitlines()
    blanks = []
    for top, bottom, left, right in _panel_boxes(text):
        titled = any(character.isalnum() for character in lines[top][left:right + 1])
        if titled_only and not titled:
            continue
        for index in range(top + 1, bottom):
            if not lines[index][left + 1:right].strip():
                blanks.append(f"line {index + 1}, columns {left}-{right}")
    return blanks


def _header_cells(text: str) -> set:
    """Collect every table header cell of a rendered layout.

    A table's header row is the line right below the one carrying its top
    border, and header cells are separated by the heavy bar Rich uses inside
    a table (the enclosing panels use a light one).
    """
    lines = text.splitlines()
    cells = set()
    for index, line in enumerate(lines[:-1]):
        if "┏" not in line:
            continue
        for cell in lines[index + 1].split("┃"):
            cell = cell.strip()
            if cell and "│" not in cell:
                cells.add(cell)
    return cells


def _rendered_column(text: str, header: str) -> list:
    """Return the rendered cells sitting under a given table header.

    Finds the one table whose header row carries ``header``, works out that
    table's horizontal span from its own top border (several tables share a
    line once they are laid out side by side), and reads every row inside
    that span. Everything is recovered from the characters that were drawn,
    so this proves what reached the screen rather than what a renderable
    intended to put there.

    Parameters
    ----------
    text : str
        Plain text rendering of the assembled layout.
    header : str
        Exact header of the column to read.

    Returns
    -------
    list of str
        One stripped value per row, in row order.

    Raises
    ------
    AssertionError
        When no rendered table carries that header.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        start = line.find("┏")
        while start != -1:
            end = line.find("┓", start)
            if end == -1:
                break
            # Header cells are separated by the heavy bar, values by the light
            # one, but both sit at the same positions, so one index fits both.
            cells = [cell.strip() for cell in lines[index + 1][start:end + 1].split("┃")]
            if header in cells:
                position = cells.index(header)
                column = []
                for row in lines[index + 2:]:
                    span = row[start:end + 1]
                    if span.startswith("┡"):
                        continue  # The rule under the header.
                    if not span.startswith("│"):
                        break     # The bottom border, or whatever follows it.
                    column.append(span.split("│")[position].strip())
                return column
            start = line.find("┏", end)
    raise AssertionError(f"no rendered table carries a {header!r} column")


def _render_layout(data: dict, width: int, height: int = 50) -> str:
    """Assemble the whole dashboard and render it at a given terminal size.

    Goes through :func:`~sys_stats.cli.layout_shape` rather than picking a
    mode by hand, so what is rendered here is what ``main`` would put on the
    screen for that payload at that width.
    """
    mode, multi_gpu = cli.layout_shape(data, width)
    layout = create_layout(mode, multi_gpu)
    build_layout_content(layout, data, 5, mode)
    return _render(layout, width=width, height=height)


# Every adaptive table of the dashboard, keyed by the panel it belongs to and
# built from a realistic payload. Each entry takes the width the table is
# allowed to occupy and returns the table itself.
TABLE_FACTORIES = {
    "ollama": lambda width: _table(
        build_ollama_panel(
            {"ollama_processes": {"models": _ollama_models()}},
            gpu_processes=_gpu_processes(),
        ),
        width,
    ),
    "gpu_processes": lambda width: _table(
        build_gpu_processes_panel({"top_gpu_processes": _gpu_processes()}, multi_gpu=True),
        width,
    ),
    "top_cpu": lambda width: build_process_table(_stats(1)["top_cpu"], "top_cpu", width),
    "top_memory": lambda width: build_process_table(
        _stats(1)["top_memory"], "top_memory", width
    ),
    "gpu_rows": lambda width: build_gpu_rows_table(_stats(5)["gpu"], width),
}


class TestAssembledLayoutFitsTheTerminal:
    """The panels are only ever correct together.

    Every earlier regression test here rendered a panel on its own, against a
    width someone had estimated by hand, which is precisely how a dashboard
    whose tables overflowed at most terminal widths kept a green suite. These
    sweep the real thing: the layout as ``main`` assembles it, rendered
    through a console of the width under test.
    """

    @pytest.mark.parametrize("gpu_count", [0, 1, 3, 5])
    def test_no_table_is_ever_clipped_by_its_panel(self, gpu_count):
        """Regression: tables overflowed their panel at most terminal widths.

        Before the panels started measuring the room they get instead of
        estimating it from the terminal width, the mono-GPU dashboard clipped
        a table at every width from 60 to 83 and from 90 to 206, and the
        multi-GPU one at 60-74, 90-94, 100, 108-142 and 150-172.
        """
        data = _stats(gpu_count)

        offenders = {
            width: _clipped_lines(_render_layout(data, width))
            for width in SWEEP_WIDTHS
        }
        clipped_widths = sorted(width for width, lines in offenders.items() if lines)

        assert not clipped_widths, (
            f"{gpu_count} GPU: clipped at {clipped_widths}\n"
            + "\n".join(offenders[clipped_widths[0]])
        )

    @pytest.mark.parametrize("gpu_count", [0, 1, 5])
    def test_columns_only_ever_appear_as_the_terminal_widens(self, gpu_count):
        """A wider terminal must never show fewer columns than a narrower one.

        Swept inside each layout mode rather than straight across both. At the
        wide threshold the dashboard re-splits the screen into a different
        number of columns, so a panel that had two thirds of the width can
        come out of it with a quarter and legitimately drop a column. Within a
        mode the panels only ever get wider, so their column sets only ever
        grow.

        Three GPUs are left out: that is the one payload where the GPU Detail
        panel swaps its row-per-GPU table for side-by-side vertical ones as
        the terminal widens, and those have no headers at all, so the column
        set legitimately changes shape rather than growing. Zero, one and five
        cards cover every table without that mode switch.
        """
        data = _stats(gpu_count)
        modes = (
            range(SWEEP_WIDTHS.start, cli.WIDE_LAYOUT_MIN_WIDTH),
            range(cli.WIDE_LAYOUT_MIN_WIDTH, SWEEP_WIDTHS.stop),
        )

        for widths in modes:
            previous = set()
            for width in widths:
                headers = _header_cells(_render_layout(data, width))
                assert previous <= headers, (
                    f"{gpu_count} GPU: {sorted(previous - headers)} disappeared at width {width}"
                )
                previous = headers

    @pytest.mark.parametrize("gpu_count", [0, 1, 3, 5])
    def test_nothing_is_drawn_past_the_terminals_last_column(self, gpu_count):
        """Rich pads to the console width; anything longer is a broken render."""
        data = _stats(gpu_count)

        for width in SWEEP_WIDTHS:
            rendered = _render_layout(data, width)
            assert all(len(line) <= width for line in rendered.splitlines())


SWEEP_PAYLOADS = {
    "0-gpu": lambda: _stats(0),
    "1-gpu": lambda: _stats(1),
    "3-gpu": lambda: _stats(3),
    "5-gpu": lambda: _stats(5),
    "busy-3-gpu": lambda: _busy_stats(3),
}


@functools.cache
def _sweep_payload(payload_id: str) -> dict:
    """Build a sweep payload once and share it across the tests."""
    return SWEEP_PAYLOADS[payload_id]()


@functools.cache
def _sweep_render(payload_id: str, width: int, height: int) -> str:
    """Render one sweep cell, remembering the result.

    The sweep is a couple of thousand full dashboard renders and every test
    below walks the same grid, so rendering each cell once instead of once per
    test is the difference between a suite that is run and one that is not.
    Nothing mutates the payloads, so the cached text stays true.
    """
    return _render_layout(_sweep_payload(payload_id), width, height)


class TestAssembledLayoutFitsEveryTerminalSize:
    """The dashboard is swept over both axes, not just the horizontal one.

    Width alone was never the whole story: the layout used to split into two
    rows of equal height whatever each row held, so a stacked column simply
    ran past its half and Rich cropped it. Nothing raised, nothing was
    logged, the panel just lost its bottom border and everything below it
    disappeared. These sweep the assembled dashboard over the sampled sizes
    of :data:`SWEEP_LAYOUT_WIDTHS` x :data:`SWEEP_LAYOUT_HEIGHTS`.
    """

    @pytest.mark.parametrize("payload_id", SWEEP_PAYLOADS)
    def test_every_panel_is_drawn_whole_at_every_terminal_size(self, payload_id):
        """Regression: a stacked region overflowed its row and got cropped.

        On the fixed 1:1 split, three GPUs at 240 columns lost a panel's
        bottom border at heights 24, 30 and 36, and the busy payload lost one
        at every height up to 50. Two things are asserted, because a crop can
        take either shape: an opened box with no closing one (the panel was
        cut through the middle) and a box count below the number of panels
        the payload calls for (the panel was cut away entirely).
        """
        expected = _expected_panel_count(_sweep_payload(payload_id))

        offenders = []
        for width in SWEEP_LAYOUT_WIDTHS:
            for height in SWEEP_LAYOUT_HEIGHTS:
                text = _sweep_render(payload_id, width, height)
                opened, closed = text.count("╭"), text.count("╰")
                if opened != closed or opened != expected:
                    offenders.append(
                        f"{width}x{height}: {opened} opened, {closed} closed, "
                        f"{expected} panels expected"
                    )
                elif _clipped_lines(text):
                    offenders.append(f"{width}x{height}: {_clipped_lines(text)[0]!r}")

        assert not offenders, "\n".join(offenders[:20])

    @pytest.mark.parametrize("payload_id", SWEEP_PAYLOADS)
    def test_nothing_is_drawn_past_the_last_row_or_column(self, payload_id):
        """Rich pads to the console size; anything beyond it is a broken render."""
        for width in SWEEP_LAYOUT_WIDTHS:
            for height in SWEEP_LAYOUT_HEIGHTS:
                lines = _sweep_render(payload_id, width, height).splitlines()

                assert len(lines) <= height, f"{width}x{height}: {len(lines)} lines"
                assert all(len(line) <= width for line in lines), f"{width}x{height}"


class TestNoPanelFramesEmptySpace:
    """A border is a promise that something is inside it.

    The layout used to hand every panel the full height of its region, so the
    dashboard drew boxes around large blocks of nothing: at 240x37 with three
    cards, the last eighteen lines of the screen were empty inside their
    borders, the Ollama panel held four rows in a box nine lines tall and the
    GPU Detail panel held three in a box thirty-seven lines tall. Panels now
    stop at the height of their own content, and the lines a column does not
    need are left as plain background.
    """

    # The sizes the complaint was made at, plus the two the earlier probe
    # claimed were clean. 240x37 with three cards is the exact one captured.
    REPORTED_SIZES = [(240, 37), (240, 50), (200, 30), (100, 30)]

    @pytest.mark.parametrize("payload_id", SWEEP_PAYLOADS)
    def test_no_panel_is_ever_drawn_taller_than_its_content(self, payload_id):
        """Swept over both axes: the stretch used to depend on both.

        Every panel that names itself is checked, at every sampled terminal
        size. The summary is left out because the two blank lines it draws are
        its own: a spacer under the clock, and the slot the ``PAUSED`` marker
        appears in, which cannot be given up without the panel jumping by a
        line every time refreshing is paused.
        """
        offenders = []
        for width in SWEEP_LAYOUT_WIDTHS:
            for height in SWEEP_LAYOUT_HEIGHTS:
                blanks = _blank_panel_interiors(_sweep_render(payload_id, width, height))
                if blanks:
                    offenders.append(f"{width}x{height}: {len(blanks)} blank ({blanks[0]})")

        assert not offenders, "\n".join(offenders[:20])

    @pytest.mark.parametrize("payload_id", SWEEP_PAYLOADS)
    @pytest.mark.parametrize(("width", "height"), REPORTED_SIZES)
    def test_the_reported_sizes_frame_no_empty_line_at_all(self, payload_id, width, height):
        """The metric from the complaint, at the sizes it was made about.

        Before this, 240x37 drew 17 such lines out of 37 and 240x50 drew 30
        out of 50, whatever the number of cards. What is left is the summary's
        own two blank rows, and only when nothing else is drawn beside them.
        """
        text = _sweep_render(payload_id, width, height)

        blanks = _boxed_blank_lines(text)
        assert len(blanks) <= 2, f"{width}x{height}: {len(blanks)} blank lines\n" + "\n".join(blanks)
        # Whatever those lines are, none of them may belong to a titled panel.
        assert not _blank_panel_interiors(text)

    def test_a_column_with_room_to_spare_leaves_it_unpainted(self):
        """Regression: the leftover used to be given to the last panel.

        That is what turned a nine-line Ollama panel into a thirty-seven-line
        one on a screen with a single column of content.
        """
        sizes = cli.distribute_heights([9, 12], [5, 5], 40)

        assert sizes == [9, 12]

    def test_a_crowded_column_hands_its_lines_to_whoever_can_use_them(self):
        """The counterpart: free space still reaches the panel that needs it.

        A four-row table next to a fifty-row one keeps the short one whole and
        gives the rest of the column to the long one, so growing it hides
        fewer rows rather than leaving the lines blank.
        """
        sizes = cli.distribute_heights([9, 54], [5, 5], 40)

        assert sizes == [9, 31]
        assert sum(sizes) == 40

    def test_the_shorter_of_two_side_by_side_panels_ends_at_its_last_row(self):
        """Regression: both rankings were drawn at the taller one's height.

        Top Memory lists two processes where Top CPU lists three, so it used
        to close one line below its last row with a blank line in between.
        """
        row = build_processes_panel(_stats(1))

        text = _render(cli.FixedHeight(row, row.natural_height(120)), width=120, height=20)

        assert not _blank_panel_interiors(text)
        assert text.count("╭") == text.count("╰") == 2


class TestAdaptiveColumnFitting:
    """Unit-level guarantees of the column fitting the panels share."""

    def _panel(self):
        """The Ollama panel, the widest and most crowded of them all."""
        return build_ollama_panel(
            {"ollama_processes": {"models": _ollama_models()}},
            gpu_processes=_gpu_processes(),
        )

    def test_a_wide_panel_shows_every_column(self):
        """Nothing is dropped when there is room for everything."""
        assert _headers(_table(self._panel(), 200)) == [
            "Model", "GPU", "Ctx", "Size", "VRAM", "GPU%", "Expires",
        ]

    def test_the_context_column_is_never_ellipsised(self):
        """Regression: ``Ctx`` was declared 3 wide while its values are 4, so
        ``128K`` rendered as ``12…`` at every width up to 200.

        A column that cannot show its own widest value is dropped, never
        shown mangled, so wherever ``Ctx`` survives it is legible.
        """
        for width in range(10, 201):
            table = _table(self._panel(), width)
            if "Ctx" not in _headers(table):
                continue
            assert "128K" in _render(table, width=width)

    @pytest.mark.parametrize("factory", TABLE_FACTORIES.values(), ids=TABLE_FACTORIES)
    def test_a_table_never_outgrows_the_width_it_was_given(self, factory):
        """The whole point of measuring: every table fits, at every width."""
        for width in range(12, 161):
            rendered = _render(factory(width), width=width)

            assert all(len(line.rstrip()) <= width for line in rendered.splitlines()), width
            assert not _clipped_lines(rendered), width
            # A column starved for room has to truncate its text: wrapping it
            # would stretch the row and leave the other cells looking like
            # blank rows underneath it.
            assert not _has_blank_row(rendered), width

    @pytest.mark.parametrize("factory", TABLE_FACTORIES.values(), ids=TABLE_FACTORIES)
    def test_columns_only_ever_appear_as_the_table_widens(self, factory):
        """The column set grows with the width, it never shuffles."""
        previous = set()
        for width in range(10, 201):
            headers = set(_headers(factory(width)))
            assert previous <= headers, f"lost {sorted(previous - headers)} at width {width}"
            previous = headers

    def test_the_last_column_standing_is_the_most_important_one(self):
        """Squeezed to nothing, the Ollama table keeps the model name."""
        assert _headers(_table(self._panel(), 10)) == ["Model"]


class TestBuildGpuSummary:
    def test_renders_one_table_per_gpu(self):
        """Regression: every card gets its own table, not GPU 0 repeated N times."""
        data = {
            "has_gpu": True,
            "gpu": [_gpu(0, "RTX 3090"), _gpu(1, "RTX 4090")],
        }

        titles = [table.title for table in build_gpu_summary(data).renderables]

        assert titles == ["RTX 3090", "RTX 4090"]

    def test_renders_nothing_without_a_gpu(self):
        """A GPU-less host produces an empty group, not a placeholder table."""
        assert build_gpu_summary({"has_gpu": False, "gpu": []}).renderables == []


class TestBuildSummary:
    def _data(self):
        """Two cards with distinct figures, so a sum differs from a mean."""
        data = {
            "cpu": 10.0,
            "ram": {"percent": 25.0, "total": 32 * 1024**3},
            "has_gpu": True,
            "gpu": [_gpu(0, "RTX 3090"), _gpu(1, "RTX 4090")],
        }
        data["gpu"][1]["load"] = 62.0
        data["gpu"][1]["temperature"] = 78.0
        return data

    def test_multi_gpu_mode_condenses_the_cards_into_totals(self):
        """Regression: the multi-GPU summary must cumulate, not list.

        Listing one vertical table per card is what overflows the narrow
        region the summary gets in multi-GPU mode, and it duplicates the
        panel that already shows the per-card detail.
        """
        text = _render(build_summary(self._data(), 5, multi_gpu=True))

        assert "GPU Totals" in text
        assert "52.0 % (mean)" in text
        assert "440 W (total)" in text
        assert "RTX 3090" not in text

    def test_mono_gpu_mode_keeps_the_per_card_tables(self):
        """One card has no total worth computing, so the detail stays here."""
        text = _render(build_summary(self._data(), 5))

        assert "RTX 3090" in text
        assert "RTX 4090" in text
        assert "GPU Totals" not in text


class TestBuildProcessTable:
    def _processes(self):
        """Two processes with distinct, identifiable names."""
        return [
            {"pid": 4242, "name": "python3.12", "cpu_percent": 412.0,
             "cmdline": "/usr/bin/python3.12 -m sys_stats.server"},
            {"pid": 8888, "name": "ollama", "cpu_percent": 88.5,
             "cmdline": "/usr/local/bin/ollama serve"},
        ]

    @pytest.mark.parametrize(("key", "title"), [("top_cpu", "Top CPU"), ("top_memory", "Top Memory")])
    def test_empty_rankings_render_a_placeholder(self, key, title):
        """No data yields an explanatory line rather than an empty table."""
        panel = build_process_panel([], key, title)

        assert panel.empty_message == f"No data for {title}."
        assert "No data for" in _render(panel)

    def test_cpu_table_has_one_row_per_process(self):
        """Each process becomes a row under the PID/Name/CPU%/Cmdline columns."""
        table = build_process_table(self._processes(), "top_cpu", 200)

        assert table.row_count == 2
        assert _headers(table) == ["PID", "Name", "CPU%", "Cmdline"]

    def test_memory_table_uses_the_memory_columns(self):
        """The memory ranking swaps the CPU% column for Memory%."""
        processes = [{"pid": 1, "name": "big", "memory_percent": 50.0, "cmdline": "big"}]

        table = build_process_table(processes, "top_memory", 200)

        assert _headers(table) == ["PID", "Name", "Memory%", "Cmdline"]

    def test_a_narrow_ranking_keeps_its_name_and_its_metric(self):
        """Regression: the Name and Cmdline columns used to vanish entirely,
        leaving a PID and a percentage with no way to tell which process they
        belonged to. The metric now outranks the PID as well: a ranking
        without the figure it ranks by says nothing."""
        table = build_process_table(self._processes(), "top_cpu", 20)

        assert _headers(table) == ["Name", "CPU%"]
        assert "python3" in _render(table, width=20)

    def test_each_ranking_sizes_itself_from_its_half_of_the_region(self):
        """Regression: both tables used to be sized against an estimate of the
        whole region, ignoring that the grid splits it in two."""
        grid = build_processes_panel({"top_cpu": self._processes(), "top_memory": []})

        rendered = _render(grid, width=80)

        assert "Top CPU" in rendered
        assert "Top Memory" in rendered
        assert not _clipped_lines(rendered)
        assert all(len(line) <= 80 for line in rendered.splitlines())


class TestOptionalPanels:
    def test_gpu_processes_panel_degrades_gracefully(self):
        """Hosts without GPU compute apps get a message, not a crash."""
        panel = build_gpu_processes_panel({"top_gpu_processes": []})

        assert panel.empty_message == "No GPU processes."
        assert "No GPU processes." in _render(panel)

    def test_ollama_panel_degrades_gracefully(self):
        """An unset or unreachable Ollama renders an empty-state panel."""
        panel = build_ollama_panel({"ollama_processes": {"models": []}})

        assert panel.empty_message == "No Ollama models."
        assert "No Ollama models." in _render(panel)

    def test_ollama_panel_lists_loaded_models(self):
        """Each loaded model becomes a row of the Ollama table."""
        data = {
            "ollama_processes": {
                "models": [
                    {"name": "llama3", "size": 8 * 1024**3, "size_vram": 4 * 1024**3},
                    {"name": "qwen3", "size": 4 * 1024**3, "size_vram": 4 * 1024**3},
                ]
            }
        }

        assert _table(build_ollama_panel(data)).row_count == 2

    def test_ollama_panel_survives_a_zero_sized_model(self):
        """A model reporting ``size`` 0 must not raise ZeroDivisionError."""
        data = {"ollama_processes": {"models": [{"name": "ghost", "size": 0, "size_vram": 0}]}}

        assert _table(build_ollama_panel(data)).row_count == 1


class TestGpuMemoryTotalBytes:
    def test_prefers_memory_total_and_converts_from_mib(self):
        """``memoryTotal`` is reported in MiB while ``memoryUsed`` is in bytes."""
        gpu_data = {"memoryTotal": 24576, "memoryUsed": 1024, "memoryPercent": 99.0}

        assert gpu_memory_total_bytes(gpu_data) == 24 * 1024**3

    def test_falls_back_to_the_used_over_percent_derivation(self):
        """Payloads predating ``memoryTotal`` still yield a usable total."""
        gpu_data = {"memoryUsed": 12 * 1024**3, "memoryPercent": 50.0}

        assert gpu_memory_total_bytes(gpu_data) == 24 * 1024**3

    @pytest.mark.parametrize(
        "gpu_data",
        [
            {},
            {"memoryTotal": 0, "memoryUsed": 0, "memoryPercent": 0},
            {"memoryUsed": 1024, "memoryPercent": 0},
        ],
    )
    def test_returns_zero_when_nothing_is_usable(self, gpu_data):
        """A driver reporting neither total nor percentage must not divide by zero."""
        assert gpu_memory_total_bytes(gpu_data) == 0


class TestBuildGpuTotals:
    def test_cumulates_across_every_card(self):
        """Sums for VRAM and power, means for load and fan, max for temperature."""
        data = {"has_gpu": True, "gpu": [_gpu(0, "RTX 3090"), _gpu(1, "RTX 4090")]}
        data["gpu"][1]["temperature"] = 78.0
        data["gpu"][1]["load"] = 62.0

        text = _render(build_gpu_totals(data))

        assert "2" in text                       # GPU count
        assert "24.0 GB / 48.0 GB (50.0 %)" in text
        assert "52.0 % (mean)" in text           # (42 + 62) / 2
        assert "440 W (total)" in text           # 220 + 220
        assert "78 °C (max)" in text
        assert "30 % (mean)" in text

    def test_reads_memory_total_as_mib(self):
        """Regression: treating ``memoryTotal`` as bytes reported 48 KB instead of 48 GB."""
        gpus = [_gpu(0, "RTX 3090"), _gpu(1, "RTX 4090")]
        for gpu_data in gpus:
            gpu_data["memoryTotal"] = 24576

        text = _render(build_gpu_totals({"has_gpu": True, "gpu": gpus}))

        assert "24.0 GB / 48.0 GB" in text

    def test_a_gpu_less_host_only_reports_a_count(self):
        """Zero card means no mean to compute, and no ZeroDivisionError."""
        table = build_gpu_totals({"has_gpu": False, "gpu": []})

        assert table.row_count == 1
        assert "0" in _render(table)

    def test_cards_reporting_no_vram_at_all_do_not_divide_by_zero(self):
        """A driver exposing neither total nor percentage leaves the summed
        total at zero, which the VRAM ratio has to survive."""
        gpus = [_null_gpu(), _null_gpu()]

        text = _render(build_gpu_totals({"has_gpu": True, "gpu": gpus}))

        assert "0.0 B / 0.0 B (0.0 %)" in text


class TestBuildGpuDetailPanel:
    def test_up_to_three_gpus_are_laid_out_side_by_side(self):
        """One grid column per card, each holding the vertical per-GPU table."""
        gpus = [_gpu(index, f"GPU{index}") for index in range(3)]

        grid = _table(build_gpu_detail_panel({"has_gpu": True, "gpu": gpus}), 200)

        assert len(grid.columns) == 3
        assert grid.row_count == 1
        assert [table.title for table in grid.columns[0]._cells + grid.columns[1]._cells] == [
            "GPU0",
            "GPU1",
        ]

    def test_more_than_three_gpus_switch_to_one_row_each(self):
        """Four vertical tables side by side are unreadable, so rows take over."""
        gpus = [_gpu(index, f"GPU{index}") for index in range(4)]

        table = _table(build_gpu_detail_panel({"has_gpu": True, "gpu": gpus}), 200)

        assert table.row_count == 4
        assert _headers(table) == ["GPU", "Name", "Util", "VRAM", "%", "Temp", "Fan", "Power"]

    def test_a_narrow_region_switches_to_rows_below_the_threshold(self):
        """Three cards in 60 columns leave 20 each: rows are the only option."""
        gpus = [_gpu(index, f"GPU{index}") for index in range(3)]

        table = _table(build_gpu_detail_panel({"has_gpu": True, "gpu": gpus}), 60)

        assert table.row_count == 3

    def test_degrades_gracefully_without_a_gpu(self):
        """A GPU-less host gets a message, not an empty grid."""
        panel = build_gpu_detail_panel({"has_gpu": False, "gpu": []})

        assert panel.empty_message == "No GPU detected."
        assert "No GPU detected." in _render(panel)

    def test_rows_use_the_driver_index_not_the_position_in_the_list(self):
        """``id`` is what nvidia-smi and every other tool call the card."""
        gpus = [_gpu(3, "GPU3"), _gpu(7, "GPU7")]

        table = build_gpu_rows_table(gpus, 200)

        assert list(table.columns[0]._cells) == ["3", "7"]

    def test_narrow_rows_keep_the_index_and_the_vram(self):
        """Regression: the GPU column used to be starved to zero width and the
        Power column ran past the panel's right edge."""
        gpus = [_gpu(index, f"GPU{index}") for index in range(3)]

        table = build_gpu_rows_table(gpus, 30)
        text = _render(table, width=30)

        assert _headers(table)[:1] == ["GPU"]
        assert "VRAM" in _headers(table)
        assert "12.0 GB" in text
        assert not _clipped_lines(text)
        assert not _has_blank_row(text)


class TestNullGpuFields:
    """A card the driver cannot query reports ``null`` for every figure.

    Which of the two GPU renderings runs is decided by the width, so an
    unguarded one turns a working dashboard into a ``TypeError`` the moment
    the terminal is resized.
    """

    def test_the_vertical_table_survives_null_figures(self):
        """Regression: this path formatted ``None`` as a number and crashed."""
        text = _render(build_single_gpu_table(_null_gpu()))

        assert "0 W" in text
        assert "0 °C" in text

    def test_the_row_per_gpu_table_survives_null_figures(self):
        """The wide fallback has always guarded them; it must keep doing so."""
        text = _render(build_gpu_rows_table([_null_gpu()], 200))

        assert "0 W" in text

    @pytest.mark.parametrize("width", [20, 30, 60, 100, 200])
    def test_the_detail_panel_survives_null_figures_at_every_width(self, width):
        """Both renderings are reachable from the same payload."""
        panel = build_gpu_detail_panel({"has_gpu": True, "gpu": [_null_gpu()]})

        assert _render(_table(panel, width), width=width)


class TestFormatContextLength:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (32768, "32K"),
            (4096, "4K"),
            (131072, "128K"),
            (1024 * 1024, "1M"),
            (2 * 1024 * 1024, "2M"),
            (40000, "40000"),
            ("32768", "32K"),
            (0, "N/A"),
            (-1, "N/A"),
            (None, "N/A"),
            ("nope", "N/A"),
        ],
    )
    def test_formats_known_shapes(self, value, expected):
        """Exact multiples of 1024 are abbreviated, the rest is shown raw.

        Anything that is not a positive whole number is ``N/A``, which is what
        ``formatContextLength`` in the web UI answers for the same payload:
        the two are documented as paired consumers of ``/stats`` and used to
        disagree on ``0``, on negatives and on numeric strings.
        """
        assert format_context_length(value) == expected


class TestOllamaPanelColumns:
    def test_context_length_gets_its_own_column(self):
        """The Ctx column shows the abbreviated context window."""
        data = {
            "ollama_processes": {
                "models": [
                    {
                        "name": "llama3",
                        "size": 8 * 1024**3,
                        "size_vram": 8 * 1024**3,
                        "context_length": 32768,
                        "expires_at": "2099-01-01T00:00:00Z",
                    }
                ]
            }
        }

        panel = build_ollama_panel(data)

        assert "Ctx" in _headers(_table(panel))
        assert "32K" in _render(_table(panel))

    def test_a_missing_context_length_renders_na(self):
        """Older Ollama servers omit the field entirely."""
        data = {
            "ollama_processes": {
                "models": [
                    {
                        "name": "llama3",
                        "size": 8 * 1024**3,
                        "size_vram": 8 * 1024**3,
                        "expires_at": "2099-01-01T00:00:00Z",
                    }
                ]
            }
        }

        # The expiry is far in the future, so N/A can only come from the Ctx column.
        assert "N/A" in _render(_table(build_ollama_panel(data)))

    def test_gpu_column_lists_the_indices_held_by_ollama(self):
        """The indices come from the compute apps named ``ollama``."""
        data = {"ollama_processes": {"models": [{"name": "llama3", "size": 1, "size_vram": 1}]}}
        gpu_processes = [
            {"pid": 1, "name": "ollama", "memory_used": 1, "gpu_index": 1},
            {"pid": 1, "name": "ollama", "memory_used": 1, "gpu_index": 0},
            {"pid": 2, "name": "python", "memory_used": 1, "gpu_index": 2},
        ]

        table = _table(build_ollama_panel(data, gpu_processes=gpu_processes))

        assert "GPU" in _headers(table)
        assert "0,1" in _render(table)

    def test_gpu_column_is_absent_without_an_ollama_process(self):
        """Nothing to attribute means no column at all."""
        data = {"ollama_processes": {"models": [{"name": "llama3", "size": 1, "size_vram": 1}]}}

        panel = build_ollama_panel(data, gpu_processes=[{"pid": 2, "name": "python"}])

        assert "GPU" not in _headers(_table(panel))

    def test_unresolved_indices_render_a_dash(self):
        """A driver too old to report ``gpu_uuid`` leaves the attribution unknown."""
        data = {"ollama_processes": {"models": [{"name": "llama3", "size": 1, "size_vram": 1}]}}
        gpu_processes = [{"pid": 1, "name": "ollama", "memory_used": 1, "gpu_index": None}]

        table = _table(build_ollama_panel(data, gpu_processes=gpu_processes))

        assert "GPU" in _headers(table)
        assert "-" in _render(table)

    def test_a_model_absent_from_vram_gets_the_placeholder_not_the_indices(self):
        """Regression: a model with ``size_vram`` 0 must not show the indices held
        by other, actually resident models."""
        data = {
            "ollama_processes": {
                "models": [
                    {"name": "llama3", "size": 1, "size_vram": 1},
                    {"name": "mistral-small:24b", "size": 14_000_000_000, "size_vram": 0},
                ]
            }
        }
        gpu_processes = [
            {"pid": 1, "name": "ollama", "memory_used": 1, "gpu_index": 0},
            {"pid": 1, "name": "ollama", "memory_used": 1, "gpu_index": 2},
        ]

        table = _table(build_ollama_panel(data, gpu_processes=gpu_processes))
        gpu_cells = table.columns[_headers(table).index("GPU")]._cells

        assert gpu_cells[0] == "0,2"
        assert gpu_cells[1] == "-"


class TestGpuProcessesPanelColumns:
    def test_multi_gpu_mode_adds_the_gpu_column(self):
        """The index is only worth a column when there is more than one card."""
        panel = build_gpu_processes_panel({"top_gpu_processes": _gpu_processes()}, multi_gpu=True)

        assert _headers(_table(panel)) == ["GPU", "PID", "Name", "Memory Used", "Cmdline"]

    def test_mono_gpu_mode_keeps_the_original_columns(self):
        """On a single card the column would only repeat ``0`` on every row."""
        panel = build_gpu_processes_panel({"top_gpu_processes": _gpu_processes()})

        assert _headers(_table(panel)) == ["PID", "Name", "Memory Used", "Cmdline"]

    def test_an_unresolved_index_renders_a_question_mark(self):
        """``gpu_index`` is ``None`` when the driver did not report ``gpu_uuid``."""
        panel = build_gpu_processes_panel({"top_gpu_processes": _gpu_processes()}, multi_gpu=True)

        assert "?" in _render(_table(panel))

    def test_ollama_rows_are_highlighted(self):
        """Ollama's share of the VRAM must be spottable among the other apps."""
        panel = build_gpu_processes_panel({"top_gpu_processes": _gpu_processes()}, multi_gpu=True)
        rows = _table(panel).rows

        assert rows[0].style == OLLAMA_ROW_STYLE
        assert rows[3].style is None

    def test_a_long_process_name_is_truncated(self):
        """A compute app can be named after its whole binary path; the column
        exists to identify it, not to reproduce it."""
        processes = [{"pid": 1, "name": "a-very-long-process-name", "memory_used": 1}]

        table = build_gpu_processes_table(processes, False, 200)
        name_cells = table.columns[_headers(table).index("Name")]._cells

        assert name_cells[0] == "a-very-long-pr…"

    def test_a_narrow_panel_keeps_the_memory_value_and_its_unit(self):
        """Regression: the memory figure used to lose its unit (``16.6``
        instead of ``16.6 GB``) and the table its right border."""
        panel = build_gpu_processes_panel(
            {"top_gpu_processes": _gpu_processes()}, multi_gpu=True
        )

        # 36 columns is what the panel's content really gets on a mono-GPU
        # dashboard in an 80 column terminal, measured rather than guessed.
        table = _table(panel, 36)
        text = _render(table, width=36)

        assert _headers(table) == ["GPU", "PID", "Name", "Memory Used"]
        assert "16.6 GB" in text
        assert "4242" in text
        assert not _clipped_lines(text)
        assert not _has_blank_row(text)
        assert all(len(line) <= 36 for line in text.splitlines())


def _titles(region) -> list:
    """Return the title of every panel stacked in a layout region."""
    titles = []
    for panel in region.renderable.panels:
        if hasattr(panel, "panels"):
            titles.extend(inner.title for inner in panel.panels)
        else:
            titles.append(str(panel.title))
    return titles


class TestBuildLayoutContent:
    def test_mono_gpu_keeps_the_summary_in_the_right_column(self):
        """One card: the VRAM panels on the left, rankings and summary right."""
        layout = create_layout()

        build_layout_content(layout, _stats(1), 5)

        assert _titles(layout["narrow_left"]) == ["Ollama Statistics", "GPU Processes"]
        assert _titles(layout["narrow_right"])[:2] == ["Top CPU", "Top Memory"]
        assert "Refresh rate" in str(layout["narrow_right"].renderable.panels[-1].subtitle)

    def test_multi_gpu_stacks_ollama_with_the_gpu_processes(self):
        """Two cards: the left column carries both VRAM panels and the totals."""
        layout = create_layout()

        build_layout_content(layout, _stats(2), 5)

        assert _titles(layout["narrow_left"])[:2] == ["Ollama Statistics", "GPU Processes"]
        assert "Refresh rate" in str(layout["narrow_left"].renderable.panels[-1].subtitle)
        assert _titles(layout["narrow_right"])[-1] == "GPU Detail"

    def test_multi_gpu_summary_is_the_cumulated_one(self):
        """Regression: the multi-GPU layout must ask for totals, not for the
        stack of per-card tables the mono layout uses."""
        layout = create_layout()

        build_layout_content(layout, _stats(2), 5)
        text = _render(layout["narrow_left"].renderable.panels[-1])

        assert "GPU Totals" in text
        assert "(mean)" in text
        assert "RTX 3090" not in text

    def test_the_ratios_flip_between_modes(self):
        """Multi-GPU widens the left column, which then carries three panels."""
        layout = create_layout()

        build_layout_content(layout, _stats(1), 5)
        mono = [layout[name].ratio for name in ("narrow_left", "narrow_right")]

        build_layout_content(layout, _stats(2), 5)
        multi = [layout[name].ratio for name in ("narrow_left", "narrow_right")]

        assert mono == [1, 2]
        assert multi == [2, 3]

    def test_a_gpu_less_host_uses_the_mono_routing(self):
        """No GPU must never trigger the multi-GPU layout."""
        layout = create_layout()

        build_layout_content(layout, _stats(0), 5)

        assert _titles(layout["narrow_left"]) == ["Ollama Statistics", "GPU Processes"]
        assert layout["narrow_left"].ratio == 1


class TestLayoutModes:
    """Which structure the dashboard picks, and that neither leaves a gap."""

    @pytest.mark.parametrize("width", [60, 120, 199])
    def test_a_narrow_terminal_gets_the_two_column_grid(self, width):
        """Below the threshold four columns would starve every table."""
        assert cli.layout_mode_for_width(width) == cli.LAYOUT_MODE_NARROW
        assert cli.layout_shape(_stats(3), width)[0] == cli.LAYOUT_MODE_NARROW

    @pytest.mark.parametrize("width", [200, 240, 260])
    def test_a_wide_terminal_gets_the_single_row_of_columns(self, width):
        """The threshold is inclusive: 200 columns is already wide."""
        assert cli.layout_mode_for_width(width) == cli.LAYOUT_MODE_WIDE
        assert cli.layout_shape(_stats(3), width)[0] == cli.LAYOUT_MODE_WIDE

    def test_the_wide_layout_drops_to_three_columns_without_a_second_card(self):
        """Regression risk: a fourth column with nothing to put in it.

        With one card the per-GPU table lives inside the summary, so there is
        no GPU Detail panel to fill a fourth column with.
        """
        mono = create_layout(cli.LAYOUT_MODE_WIDE, multi_gpu=False)
        multi = create_layout(cli.LAYOUT_MODE_WIDE, multi_gpu=True)

        assert len(mono.children) == 3
        assert len(multi.children) == 4

    @pytest.mark.parametrize("gpu_count", [0, 1, 3])
    @pytest.mark.parametrize("width", [120, 240])
    def test_no_region_is_ever_left_empty(self, gpu_count, width):
        """Every region of both layouts is filled, in every GPU configuration."""
        data = _stats(gpu_count)
        mode, multi_gpu = cli.layout_shape(data, width)
        layout = create_layout(mode, multi_gpu)

        build_layout_content(layout, data, 5, mode)

        for region in layout.children:
            assert isinstance(region.renderable, cli.StackedPanels)
            assert region.renderable.panels
            assert _render(region.renderable, width=width // len(layout.children)).strip()


class TestGpuColumnReachesTheScreen:
    """The headline feature of the multi-GPU layout, checked end to end.

    Every other test of the ``GPU`` column stops at the renderable the panel
    builder returns, so nothing proved that ``build_layout_content`` actually
    asks for it: passing ``multi_gpu=False`` at that call site left the whole
    suite green while the column silently vanished from a multi-GPU
    dashboard. These read the column back out of the drawn characters.
    """

    def _processes(self):
        """Compute apps spread over three cards, none of them Ollama's.

        Keeping Ollama out is deliberate. An Ollama process would give the
        Ollama table a ``GPU`` column of its own, and these tests have to pin
        down the one that belongs to the GPU Processes panel, not any ``GPU``
        header that happens to be on screen.
        """
        return [
            {"pid": 8888, "name": "python3", "memory_used": 3_145_728_000,
             "cmdline": "python3 train.py --epochs 40", "gpu_uuid": "GPU-0", "gpu_index": 0},
            {"pid": 9001, "name": "vllm", "memory_used": 2_147_483_648,
             "cmdline": "vllm serve mistral", "gpu_uuid": "GPU-2", "gpu_index": 2},
            {"pid": 9002, "name": "comfyui", "memory_used": 1_073_741_824,
             "cmdline": "python main.py --listen", "gpu_uuid": "GPU-1", "gpu_index": 1},
            # A driver too old to report gpu_uuid leaves the index unresolved.
            {"pid": 9999, "name": "blender", "memory_used": 1_258_291_200,
             "cmdline": "blender -b scene.blend", "gpu_uuid": None, "gpu_index": None},
        ]

    def _data(self, gpu_count):
        """A payload whose GPU processes are the ones above."""
        data = _stats(gpu_count)
        data["top_gpu_processes"] = self._processes()
        return data

    def test_a_multi_gpu_dashboard_draws_the_index_of_every_process(self):
        """Regression: the column has to survive the trip through the layout.

        Three cards at 190 columns keep the dashboard in its narrow mode,
        where the GPU Detail panel is wide enough for its headerless
        side-by-side rendering, and no Ollama process means no ``GPU`` column
        in the Ollama table, so the only one left on screen is the one under
        test. A wider terminal would give the detail panel a column of its own
        and a ``GPU`` header to go with it.
        """
        text = _render_layout(self._data(3), 190)

        assert "GPU" in _header_cells(text)
        assert _rendered_column(text, "GPU") == ["0", "2", "1", "?"]

    def test_a_mono_gpu_dashboard_draws_no_index_at_all(self):
        """One card means the column would only repeat ``0`` on every row."""
        text = _render_layout(self._data(1), 190)

        assert "GPU" not in _header_cells(text)
        # The placeholder for an unresolved index exists nowhere else.
        assert "?" not in text


def _scroll_processes(count: int) -> list:
    """Compute apps named so the rendered window can be read straight back."""
    return [
        {
            "pid": 1000 + index,
            "name": f"proc{index:02d}",
            "memory_used": 1024 ** 3,
            "cmdline": "compute",
        }
        for index in range(count)
    ]


def _render_panel(panel, width: int = 60, height: int = 12) -> str:
    """Render one panel at an exact size, the way a layout region would.

    ``publish=True`` is what the real render pass uses, so the row count and
    page size the keyboard thread reads are the ones this render resolved.
    """
    return _render(panel.render_panel(width, height, publish=True), width=width, height=height)


def _press(monkeypatch, keys):
    """Run the keyboard listener over a scripted sequence of key presses.

    The listener loops on ``readchar.readkey`` until ``q`` sets the exit
    event, so the sequence is simply terminated with one.
    """
    sequence = iter([*keys, "q"])

    class _ScriptedReadchar:
        @staticmethod
        def readkey():
            return next(sequence)

    monkeypatch.setattr(cli, "readchar", _ScriptedReadchar)
    cli.exit_event.clear()
    try:
        cli.keyboard_listener()
    finally:
        cli.exit_event.clear()


@pytest.fixture
def scroll_state():
    """Give a test the shared scrolling state to itself, and clean up after."""
    def reset():
        cli.scroll_offsets.clear()
        cli.panel_row_counts.clear()
        cli.panel_page_sizes.clear()
        cli.scrollable_panel_keys[:] = []
        cli.focused_panel = None

    reset()
    yield
    reset()


class TestPanelScrolling:
    """A panel with more rows than lines shows a window over them.

    No layout fixes a host running twelve models: the rows simply outnumber
    the screen. These drive the very keys a user would, then read the window
    back out of the characters that were drawn.
    """

    WIDTH = 60
    HEIGHT = 12

    def _panel(self, count: int = 12):
        """The GPU processes panel over a list too long for :attr:`HEIGHT`."""
        return build_gpu_processes_panel({"top_gpu_processes": _scroll_processes(count)})

    def _visible(self, panel) -> list:
        """Names of the rows the panel actually drew, in order."""
        return re.findall(r"proc\d\d", _render_panel(panel, self.WIDTH, self.HEIGHT))

    def _focus(self, key=None):
        """Point the scroll keys at one panel, as the layout would."""
        cli.scrollable_panel_keys[:] = [key or cli.PANEL_GPU_PROCESSES]
        cli.focused_panel = cli.scrollable_panel_keys[0]

    def test_a_panel_that_fits_shows_every_row_and_says_nothing(self, scroll_state):
        """The indicator is a statement about hidden rows, not decoration."""
        panel = self._panel(3)

        text = _render_panel(panel, self.WIDTH, self.HEIGHT)

        assert re.findall(r"proc\d\d", text) == ["proc00", "proc01", "proc02"]
        assert "more row" not in text

    def test_a_panel_that_overflows_counts_what_it_hides(self, scroll_state):
        """Below only, at the top of a list: there is nothing above yet."""
        text = _render_panel(self._panel(12), self.WIDTH, self.HEIGHT)

        hidden = 12 - len(re.findall(r"proc\d\d", text))
        assert f"↓ {hidden} more rows below" in text
        assert "above" not in text

    def test_scrolling_down_names_what_is_now_hidden_above(self, monkeypatch, scroll_state):
        """Both indicators once the window sits in the middle of the list."""
        panel = self._panel(12)
        self._visible(panel)
        self._focus()

        _press(monkeypatch, [cli.readchar_key.DOWN] * 3)
        text = _render_panel(panel, self.WIDTH, self.HEIGHT)

        assert "↑ 3 more rows above" in text
        assert "more rows below" in text

    def test_the_arrows_move_the_window_by_one_row(self, monkeypatch, scroll_state):
        """Down then up is a round trip, at the very top of the list."""
        panel = self._panel(12)
        assert self._visible(panel)[0] == "proc00"
        self._focus()

        _press(monkeypatch, [cli.readchar_key.DOWN])
        assert self._visible(panel)[0] == "proc01"

        _press(monkeypatch, [cli.readchar_key.UP])
        assert self._visible(panel)[0] == "proc00"

    def test_up_at_the_top_stays_at_the_top(self, monkeypatch, scroll_state):
        """A negative offset would render a window over nothing."""
        panel = self._panel(12)
        self._visible(panel)
        self._focus()

        _press(monkeypatch, [cli.readchar_key.UP] * 5)

        assert self._visible(panel)[0] == "proc00"
        assert cli.scroll_offsets[cli.PANEL_GPU_PROCESSES] == 0

    def test_the_page_keys_move_by_a_screenful_and_come_back(self, monkeypatch, scroll_state):
        """Regression risk: paging back has to land where paging forward left.

        The window is one row shorter as soon as something is hidden above,
        so a page measured from the window as rendered would drift by a row
        on every round trip.
        """
        panel = self._panel(12)
        page = len(self._visible(panel))
        self._focus()

        _press(monkeypatch, [cli.readchar_key.PAGE_DOWN])
        assert self._visible(panel)[0] == f"proc{page:02d}"

        _press(monkeypatch, [cli.readchar_key.PAGE_UP])
        assert self._visible(panel)[0] == "proc00"

    def test_end_shows_the_last_rows_and_home_comes_back(self, monkeypatch, scroll_state):
        """End fills the window with the tail of the list, not with one row."""
        panel = self._panel(12)
        page = len(self._visible(panel))
        self._focus()

        _press(monkeypatch, [cli.readchar_key.END])
        text = _render_panel(panel, self.WIDTH, self.HEIGHT)

        assert re.findall(r"proc\d\d", text)[-1] == "proc11"
        assert len(re.findall(r"proc\d\d", text)) == page
        assert "more rows below" not in text
        assert "↑ " in text

        _press(monkeypatch, [cli.readchar_key.HOME])
        assert self._visible(panel)[0] == "proc00"

    def test_an_offset_past_the_end_is_clamped_when_the_list_shrinks(
        self, monkeypatch, scroll_state
    ):
        """Regression risk: the row count changes on every single fetch.

        A panel left scrolled near the bottom of a long list, whose next
        payload is a short one, would render a window over rows that no
        longer exist: an empty panel, with no key that brings it back. Two
        shapes of shrink are checked, because they take different paths: one
        that still overflows, so the offset has to be pulled back to the last
        full window, and one that now fits, where there is no offset left to
        keep at all.
        """
        page = len(self._visible(self._panel(12)))
        self._focus()
        _press(monkeypatch, [cli.readchar_key.END])
        stale = cli.scroll_offsets[cli.PANEL_GPU_PROCESSES]
        assert stale > 0

        still_overflowing = re.findall(
            r"proc\d\d", _render_panel(self._panel(8), self.WIDTH, self.HEIGHT)
        )

        assert cli.scroll_offsets[cli.PANEL_GPU_PROCESSES] < stale
        assert still_overflowing[-1] == "proc07"
        assert len(still_overflowing) == page

        now_fits = re.findall(r"proc\d\d", _render_panel(self._panel(2), self.WIDTH, self.HEIGHT))

        assert cli.scroll_offsets[cli.PANEL_GPU_PROCESSES] == 0
        assert now_fits == ["proc00", "proc01"]

    @pytest.mark.parametrize(
        ("sequence", "expected_first"),
        [
            # readchar's own spellings, which is what a bare xterm sends.
            ("\x1b[F", "last"),
            ("\x1b[H", "proc00"),
            # tmux, screen and the Linux console. readchar stops one byte
            # short of the End sequence and returns the rest separately.
            ("\x1b[4", "last"),
            ("\x1b[1~", "proc00"),
            # xterm with application cursor keys on.
            ("\x1bOF", "last"),
            ("\x1bOH", "proc00"),
        ],
    )
    def test_home_and_end_answer_to_every_spelling_terminals_use(
        self, monkeypatch, scroll_state, sequence, expected_first
    ):
        """Regression: driving the real dashboard under tmux, End did nothing.

        Terminals disagree on what Home and End send and readchar only knows
        one spelling of each, so matching its constant alone left both keys
        dead under tmux, screen and the Linux console.
        """
        panel = self._panel(12)
        page = len(self._visible(panel))
        self._focus()
        # Start away from both ends, so either key has somewhere to go.
        _press(monkeypatch, [cli.readchar_key.DOWN] * 2)

        _press(monkeypatch, [sequence])
        visible = self._visible(panel)

        if expected_first == "last":
            assert visible[-1] == "proc11"
            assert len(visible) == page
        else:
            assert visible[0] == expected_first

    def test_a_panel_too_short_for_a_row_says_how_much_it_holds(self, scroll_state):
        """Four lines cannot hold a header, a rule and a row, let alone eight."""
        text = _render_panel(self._panel(8), self.WIDTH, 4)

        assert "8 more rows hidden" in text
        assert text.count("╭") == text.count("╰") == 1


class TestPanelFocus:
    """Which panel the scroll keys drive, and how a user can tell."""

    def test_the_focused_panel_gets_a_distinct_border(self, scroll_state):
        """Otherwise the keys act on a panel nobody can point at."""
        panel = build_gpu_processes_panel({"top_gpu_processes": _scroll_processes(4)})

        cli.focused_panel = None
        assert panel.render_panel(60, 12).border_style == cli.PANEL_BORDER_STYLE

        cli.focused_panel = cli.PANEL_GPU_PROCESSES
        assert panel.render_panel(60, 12).border_style == cli.FOCUSED_PANEL_BORDER_STYLE

    def test_building_the_layout_focuses_the_first_panel(self, scroll_state):
        """Something has to be focused, or the keys do nothing at all."""
        layout = create_layout()

        build_layout_content(layout, _stats(3), 5)

        assert cli.scrollable_panel_keys[0] == cli.PANEL_OLLAMA
        assert cli.focused_panel == cli.PANEL_OLLAMA

    def test_tab_cycles_through_every_scrollable_panel(self, monkeypatch, scroll_state):
        """And wraps around, so Tab alone reaches all of them."""
        layout = create_layout()
        build_layout_content(layout, _stats(3), 5)
        order = list(cli.scrollable_panel_keys)
        assert len(order) == 5

        for expected in order[1:]:
            _press(monkeypatch, [cli.readchar_key.TAB])
            assert cli.focused_panel == expected

        _press(monkeypatch, [cli.readchar_key.TAB])
        assert cli.focused_panel == order[0]

    def test_a_panel_with_nothing_to_list_is_not_in_the_tab_order(self, scroll_state):
        """There is no window to move over an empty-state panel."""
        data = _stats(1)
        data["ollama_processes"] = {"models": []}
        layout = create_layout()

        build_layout_content(layout, data, 5)

        assert cli.PANEL_OLLAMA not in cli.scrollable_panel_keys
        assert cli.focused_panel == cli.PANEL_GPU_PROCESSES

    def test_a_focus_left_on_a_vanished_panel_falls_back(self, scroll_state):
        """A card removed between two fetches takes its panel with it."""
        layout = create_layout()
        build_layout_content(layout, _stats(3), 5)
        cli.focused_panel = cli.PANEL_GPU_DETAIL

        build_layout_content(layout, _stats(1), 5)

        assert cli.PANEL_GPU_DETAIL not in cli.scrollable_panel_keys
        assert cli.focused_panel == cli.scrollable_panel_keys[0]

    def test_the_help_screen_documents_every_scrolling_key(self):
        """It is the only place a user ever learns they exist."""
        text = _render(build_full_screen_help())

        for documented in ("Tab", "Up/Down", "PgUp/PgDn", "Home/End"):
            assert documented in text


class _FakeLive:
    """Stand-in for :class:`rich.live.Live` that records ``update`` calls.

    Real ``Live`` needs a live terminal and would otherwise try to render to
    stdout in a background thread; this only has to prove which renderable
    ``main`` hands it and when.
    """

    def __init__(self, renderable, **kwargs):
        self.initial_renderable = renderable
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def update(self, renderable):
        self.updates.append(renderable)


class _MainHarness:
    """Run :func:`sys_stats.cli.main` against a fake ``Live`` and clock.

    The inner sleep loop checks ``exit_event``/``rebuild_layout_event`` before
    every ``time.sleep`` call, so driving those two events from a faked
    ``time.sleep`` steps the outer loop deterministically without any real
    waiting.
    """

    STATS = {
        "cpu": 1.0,
        "ram": {"percent": 1.0, "total": 1},
        "has_gpu": False,
        "gpu": [],
        "top_cpu": [],
        "top_memory": [],
        "top_gpu_processes": [],
        "ollama_processes": {"models": []},
    }

    def __init__(self, monkeypatch, on_sleep, on_fetch=None, width=100):
        self.live_instances = []
        self.monkeypatch = monkeypatch
        self.on_sleep = on_sleep
        self.on_fetch = on_fetch
        self.width = width

    def _fetch(self, _url):
        if self.on_fetch is not None:
            self.on_fetch()
        return self.STATS

    def _terminal_width(self):
        """Stand in for the real console, which pytest does not give us.

        Accepts a callable so a test can resize the terminal between two
        iterations of the loop.
        """
        return self.width() if callable(self.width) else self.width

    def run(self):
        """Reset the module globals, run ``main``, and return the fake ``Live``."""
        def live_factory(renderable, **kwargs):
            instance = _FakeLive(renderable, **kwargs)
            self.live_instances.append(instance)
            return instance

        self.monkeypatch.setattr(cli, "Live", live_factory)
        self.monkeypatch.setattr(cli, "keyboard_listener", lambda: None)
        self.monkeypatch.setattr(cli, "fetch_stats", self._fetch)
        self.monkeypatch.setattr(cli, "terminal_width", self._terminal_width)
        self.monkeypatch.setattr(cli.time, "sleep", self.on_sleep)
        self.monkeypatch.setattr(
            sys, "argv", ["sys-stats", "--url", "http://example.invalid/stats", "--interval", "1"]
        )

        cli.exit_event.clear()
        cli.rebuild_layout_event.clear()
        cli.show_help_flag = False
        cli.is_paused = False
        cli.latest_stats = None
        try:
            cli.main()
        finally:
            cli.exit_event.clear()
            cli.rebuild_layout_event.clear()
            cli.show_help_flag = False
            cli.is_paused = False
            cli.latest_stats = None
            cli.scroll_offsets.clear()
            cli.panel_row_counts.clear()
            cli.panel_page_sizes.clear()
            cli.scrollable_panel_keys[:] = []
            cli.focused_panel = None

        assert len(self.live_instances) == 1
        return self.live_instances[0]


class TestMainLoop:
    def test_toggling_help_swaps_the_lives_renderable(self, monkeypatch):
        """Regression: ``layout.update(help_panel)`` is a no-op once ``layout``
        has children (Rich renders a layout's children in preference to its
        own set renderable), which used to freeze the whole dashboard as soon
        as help was toggled on.

        Two ``h`` presses: show help, then return to the live layout.
        """
        calls = {"count": 0}

        def on_sleep(_seconds):
            calls["count"] += 1
            if calls["count"] == 1:
                with cli.state_lock:
                    cli.show_help_flag = True
                cli.rebuild_layout_event.set()
            elif calls["count"] == 2:
                with cli.state_lock:
                    cli.show_help_flag = False
                cli.rebuild_layout_event.set()
            else:
                cli.exit_event.set()

        live = _MainHarness(monkeypatch, on_sleep).run()

        assert live.updates, "expected at least one live.update() call"
        # First press: the help panel is swapped in, not the layout.
        assert live.updates[0] is not live.initial_renderable
        # Second press: the live layout is restored so refreshes resume.
        assert live.updates[-1] is live.initial_renderable

    def test_a_key_press_during_the_fetch_is_honoured_at_once(self, monkeypatch):
        """Regression: a key pressed while the HTTP request was in flight was
        swallowed for a whole refresh interval.

        The loop sampled the state, blocked in ``fetch_stats``, then acted on
        that now stale sample and cleared the wake-up event, destroying it.
        Pressing ``h`` mid-request has to show the help screen on this very
        iteration.
        """
        pressed = {"done": False}

        def on_fetch():
            if pressed["done"]:
                return
            pressed["done"] = True
            # The keyboard thread reacting while the request is in flight.
            with cli.state_lock:
                cli.show_help_flag = True
            cli.rebuild_layout_event.set()

        def on_sleep(_seconds):
            cli.exit_event.set()

        live = _MainHarness(monkeypatch, on_sleep, on_fetch=on_fetch).run()

        assert live.updates, "the help screen was not shown on the iteration that fetched"
        assert live.updates[0] is not live.initial_renderable

    def test_a_stable_terminal_never_rebuilds_the_layout(self, monkeypatch):
        """The ``Layout`` object is expensive to re-split and never has to be.

        Everything that varies with the data is decided inside the regions, so
        as long as the shape holds the very same object keeps being rendered.
        """
        def on_sleep(_seconds):
            cli.exit_event.set()

        live = _MainHarness(monkeypatch, on_sleep).run()

        assert live.updates == []
        assert len(live.initial_renderable.children) == 2

    def test_a_terminal_resized_past_the_threshold_gets_the_wide_layout(self, monkeypatch):
        """Regression risk: a rebuilt layout that never reaches the screen.

        ``Live`` renders the object it was handed, so a new ``Layout`` is
        invisible until it is pushed through ``live.update``, exactly as the
        help screen is.
        """
        width = [100]

        def on_sleep(_seconds):
            if width[0] == 100:
                width[0] = 240
                cli.rebuild_layout_event.set()
            else:
                cli.exit_event.set()

        live = _MainHarness(monkeypatch, on_sleep, width=lambda: width[0]).run()

        assert live.updates, "the wide layout was never pushed through Live"
        assert live.updates[-1] is not live.initial_renderable
        # Three columns: one card means no separate GPU detail panel.
        assert len(live.updates[-1].children) == 3

    def test_pausing_during_the_fetch_stops_the_next_request(self, monkeypatch):
        """The same staleness used to keep a paused dashboard fetching once more."""
        fetches = {"count": 0}

        def on_fetch():
            fetches["count"] += 1
            if fetches["count"] == 1:
                with cli.state_lock:
                    cli.is_paused = True
                cli.rebuild_layout_event.set()

        def on_sleep(_seconds):
            cli.exit_event.set()

        _MainHarness(monkeypatch, on_sleep, on_fetch=on_fetch).run()

        assert fetches["count"] == 1
