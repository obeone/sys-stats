"""Tests for the Rich panel builders of the terminal dashboard.

The builders are pure: they take a ``/stats`` payload and return renderables, so
they can be asserted on without a terminal. Only structure is checked here (how
many tables, which rows), never the ANSI output.
"""

import io

import pytest
from rich.console import Console, Group
from rich.table import Table

from sys_stats.cli import (
    OLLAMA_ROW_STYLE,
    build_gpu_detail_panel,
    build_gpu_processes_panel,
    build_gpu_summary,
    build_gpu_totals,
    build_layout_content,
    build_ollama_panel,
    build_process_table,
    build_processes_panel,
    create_layout,
    format_context_length,
    gpu_memory_total_bytes,
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


def _render(renderable, width: int = 200) -> str:
    """Render a renderable to plain text, for the assertions on values.

    Structure assertions are preferred everywhere else; this is only used when
    the computed figure itself is what matters. The console writes to a
    :class:`io.StringIO`, so Rich emits no ANSI escapes.
    """
    console = Console(file=io.StringIO(), width=width, legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


def _headers(table: Table) -> list:
    """Return the header of every column of a table."""
    return [column.header for column in table.columns]


def _has_blank_row(text: str) -> bool:
    """Detect a table row made only of box-drawing bars and whitespace.

    That shape is the signature of a starved column that wrapped its
    (already truncated) text onto extra physical lines: the row grows taller
    than one line, but every other cell only has content on the first line,
    leaving the rest looking like a fully blank row.

    Parameters
    ----------
    text : str
        Plain text rendering of a panel or table.

    Returns
    -------
    bool
        ``True`` if any non-empty line consists solely of ``│`` and spaces.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and set(stripped) <= {"│", " "}:
            return True
    return False


def _first_column_collapsed(text: str) -> bool:
    """Detect a table's first column having been starved down to zero width.

    Rich's last-resort column shrinking (triggered when even the sum of every
    column's minimum width does not fit) can crush a column to nothing
    regardless of its ``min_width``: the signature is the header row's
    top-left corner being immediately followed by a column separator instead
    of at least one border dash.

    Parameters
    ----------
    text : str
        Plain text rendering of a panel or table.

    Returns
    -------
    bool
        ``True`` if the first column has zero width.
    """
    for line in text.splitlines():
        if "┏" in line:
            index = line.index("┏")
            return line[index + 1] in ("┳", "┓")
    return False


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


class TestBuildProcessTable:
    @pytest.mark.parametrize(("key", "title"), [("top_cpu", "CPU"), ("top_memory", "Memory")])
    def test_empty_rankings_render_a_placeholder_panel(self, key, title):
        """No data yields an explanatory panel rather than an empty table."""
        panel = build_process_table([], key, title)

        assert title in str(panel.renderable)

    def test_cpu_table_has_one_row_per_process(self):
        """Each process becomes a row under the PID/Name/CPU%/Cmdline columns."""
        processes = [
            {"pid": 1, "name": "busy", "cpu_percent": 90.0, "cmdline": "busy --go"},
            {"pid": 2, "name": "idle", "cpu_percent": 0.1, "cmdline": "idle"},
        ]

        table = build_process_table(processes, "top_cpu", "CPU")

        assert table.row_count == 2
        assert [column.header for column in table.columns] == ["PID", "Name", "CPU%", "Cmdline"]

    def test_memory_table_uses_the_memory_columns(self):
        """The memory ranking swaps the CPU% column for Memory%."""
        processes = [{"pid": 1, "name": "big", "memory_percent": 50.0, "cmdline": "big"}]

        table = build_process_table(processes, "top_memory", "Memory")

        assert [column.header for column in table.columns] == ["PID", "Name", "Memory%", "Cmdline"]


class TestOptionalPanels:
    def test_gpu_processes_panel_degrades_gracefully(self):
        """Hosts without GPU compute apps get a message, not a crash."""
        panel = build_gpu_processes_panel({"top_gpu_processes": []})

        assert "No GPU processes." in str(panel.renderable)

    def test_ollama_panel_degrades_gracefully(self):
        """An unset or unreachable Ollama renders an empty-state panel."""
        panel = build_ollama_panel({"ollama_processes": {"models": []}})

        assert "No Ollama models." in str(panel.renderable)

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

        panel = build_ollama_panel(data)

        assert panel.renderable.row_count == 2

    def test_ollama_panel_survives_a_zero_sized_model(self):
        """A model reporting ``size`` 0 must not raise ZeroDivisionError."""
        data = {"ollama_processes": {"models": [{"name": "ghost", "size": 0, "size_vram": 0}]}}

        assert build_ollama_panel(data).renderable.row_count == 1


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


class TestBuildGpuDetailPanel:
    def test_up_to_three_gpus_are_laid_out_side_by_side(self):
        """One grid column per card, each holding the vertical per-GPU table."""
        gpus = [_gpu(index, f"GPU{index}") for index in range(3)]

        panel = build_gpu_detail_panel({"has_gpu": True, "gpu": gpus}, terminal_width=200)
        grid = panel.renderable

        assert len(grid.columns) == 3
        assert grid.row_count == 1
        assert [table.title for table in grid.columns[0]._cells + grid.columns[1]._cells] == [
            "GPU0",
            "GPU1",
        ]

    def test_more_than_three_gpus_switch_to_one_row_each(self):
        """Four vertical tables side by side are unreadable, so rows take over."""
        gpus = [_gpu(index, f"GPU{index}") for index in range(4)]

        panel = build_gpu_detail_panel({"has_gpu": True, "gpu": gpus}, terminal_width=200)
        table = panel.renderable

        assert table.row_count == 4
        assert _headers(table) == ["GPU", "Name", "Util", "VRAM", "%", "Temp", "Fan", "Power"]

    def test_a_narrow_terminal_switches_to_rows_below_the_threshold(self):
        """Three cards on 80 columns leave ~13 columns each: rows are the only option."""
        gpus = [_gpu(index, f"GPU{index}") for index in range(3)]

        panel = build_gpu_detail_panel({"has_gpu": True, "gpu": gpus}, terminal_width=80)

        assert panel.renderable.row_count == 3

    def test_degrades_gracefully_without_a_gpu(self):
        """A GPU-less host gets a message, not an empty grid."""
        panel = build_gpu_detail_panel({"has_gpu": False, "gpu": []})

        assert "No GPU detected." in str(panel.renderable)


class TestFormatContextLength:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (32768, "32K"),
            (4096, "4K"),
            (1024 * 1024, "1M"),
            (2 * 1024 * 1024, "2M"),
            (40000, "40000"),
            (0, "0"),
            (None, "N/A"),
            ("nope", "N/A"),
        ],
    )
    def test_formats_known_shapes(self, value, expected):
        """Exact multiples of 1024 are abbreviated, the rest is shown raw."""
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

        assert "Ctx" in _headers(panel.renderable)
        assert "32K" in _render(panel)

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
        assert "N/A" in _render(build_ollama_panel(data))

    def test_gpu_column_lists_the_indices_held_by_ollama(self):
        """The indices come from the compute apps named ``ollama``."""
        data = {"ollama_processes": {"models": [{"name": "llama3", "size": 1, "size_vram": 1}]}}
        gpu_processes = [
            {"pid": 1, "name": "ollama", "memory_used": 1, "gpu_index": 1},
            {"pid": 1, "name": "ollama", "memory_used": 1, "gpu_index": 0},
            {"pid": 2, "name": "python", "memory_used": 1, "gpu_index": 2},
        ]

        panel = build_ollama_panel(data, gpu_processes=gpu_processes)

        assert "GPU" in _headers(panel.renderable)
        assert "0,1" in _render(panel)

    def test_gpu_column_is_absent_without_an_ollama_process(self):
        """Nothing to attribute means no column at all."""
        data = {"ollama_processes": {"models": [{"name": "llama3", "size": 1, "size_vram": 1}]}}

        panel = build_ollama_panel(data, gpu_processes=[{"pid": 2, "name": "python"}])

        assert "GPU" not in _headers(panel.renderable)

    def test_unresolved_indices_render_a_dash(self):
        """A driver too old to report ``gpu_uuid`` leaves the attribution unknown."""
        data = {"ollama_processes": {"models": [{"name": "llama3", "size": 1, "size_vram": 1}]}}
        gpu_processes = [{"pid": 1, "name": "ollama", "memory_used": 1, "gpu_index": None}]

        panel = build_ollama_panel(data, gpu_processes=gpu_processes)

        assert "GPU" in _headers(panel.renderable)
        assert "-" in _render(panel)

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

        panel = build_ollama_panel(data, gpu_processes=gpu_processes)
        table = panel.renderable
        gpu_cells = table.columns[_headers(table).index("GPU")]._cells

        assert gpu_cells[0] == "0,2"
        assert gpu_cells[1] == "-"


class TestGpuProcessesPanelColumns:
    def _processes(self):
        """Two compute apps, one of them owned by Ollama, one unattributed."""
        return [
            {"pid": 1, "name": "ollama", "memory_used": 1024**3, "cmdline": "ollama serve",
             "gpu_index": 1, "gpu_uuid": "GPU-1"},
            {"pid": 2, "name": "python", "memory_used": 1024**3, "cmdline": "python train.py",
             "gpu_index": None, "gpu_uuid": None},
        ]

    def test_multi_gpu_mode_adds_the_gpu_column(self):
        """The index is only worth a column when there is more than one card."""
        panel = build_gpu_processes_panel({"top_gpu_processes": self._processes()}, multi_gpu=True)

        assert _headers(panel.renderable) == ["GPU", "PID", "Name", "Memory Used", "Cmdline"]

    def test_mono_gpu_mode_keeps_the_original_columns(self):
        """On a single card the column would only repeat ``0`` on every row."""
        panel = build_gpu_processes_panel({"top_gpu_processes": self._processes()})

        assert _headers(panel.renderable) == ["PID", "Name", "Memory Used", "Cmdline"]

    def test_an_unresolved_index_renders_a_question_mark(self):
        """``gpu_index`` is ``None`` when the driver did not report ``gpu_uuid``."""
        panel = build_gpu_processes_panel({"top_gpu_processes": self._processes()}, multi_gpu=True)

        assert "?" in _render(panel)

    def test_ollama_rows_are_highlighted(self):
        """Ollama's share of the VRAM must be spottable among the other apps."""
        panel = build_gpu_processes_panel({"top_gpu_processes": self._processes()}, multi_gpu=True)
        rows = panel.renderable.rows

        assert rows[0].style == OLLAMA_ROW_STYLE
        assert rows[1].style is None


class TestBuildLayoutContent:
    def _data(self, gpu_count: int) -> dict:
        """Build a minimal ``/stats`` payload with the requested number of GPUs."""
        return {
            "cpu": 10.0,
            "ram": {"percent": 25.0, "total": 32 * 1024**3},
            "has_gpu": gpu_count > 0,
            "gpu": [_gpu(index, f"GPU{index}") for index in range(gpu_count)],
            "top_cpu": [],
            "top_memory": [],
            "top_gpu_processes": [],
            "ollama_processes": {"models": []},
        }

    def test_mono_gpu_keeps_the_summary_bottom_right(self):
        """One card: Ollama top left, GPU processes bottom left, summary bottom right."""
        layout = create_layout()

        build_layout_content(layout, self._data(1), 5, 200)

        assert "Ollama" in str(layout["top_left"].renderable.title)
        assert "GPU Processes" in str(layout["bottom_left"].renderable.title)
        assert "Refresh rate" in str(layout["bottom_right"].renderable.subtitle)

    def test_multi_gpu_stacks_ollama_with_the_gpu_processes(self):
        """Two cards: the left column carries both VRAM panels, detail goes bottom right."""
        layout = create_layout()

        build_layout_content(layout, self._data(2), 5, 200)

        top_left = layout["top_left"].renderable
        assert isinstance(top_left, Group)
        assert "Ollama" in str(top_left.renderables[0].title)
        assert "GPU Processes" in str(top_left.renderables[1].title)
        assert "Refresh rate" in str(layout["bottom_left"].renderable.subtitle)
        assert "GPU Detail" in str(layout["bottom_right"].renderable.title)

    def test_the_ratios_flip_between_modes(self):
        """Multi-GPU widens the left column and the detail region."""
        layout = create_layout()

        build_layout_content(layout, self._data(1), 5, 200)
        mono = [layout[name].ratio for name in ("top_left", "top_right", "bottom_right")]

        build_layout_content(layout, self._data(2), 5, 200)
        multi = [layout[name].ratio for name in ("top_left", "top_right", "bottom_right")]

        assert mono == [1, 2, 1]
        assert multi == [2, 3, 2]

    def test_a_gpu_less_host_uses_the_mono_routing(self):
        """No GPU must never trigger the multi-GPU layout."""
        layout = create_layout()

        build_layout_content(layout, self._data(0), 5, 200)

        assert "Ollama" in str(layout["top_left"].renderable.title)
        assert layout["top_left"].ratio == 1


class TestOllamaPanelNarrowWidth:
    """Regression tests for the Model/GPU/Ctx columns collapsing at 80 columns."""

    def _data(self):
        """A realistic multi-GPU Ollama payload, shaped like ``/api/ps``."""
        return {
            "ollama_processes": {
                "models": [
                    {
                        "name": "qwen3-coder:30b",
                        "size": 19_000_000_000,
                        "size_vram": 19_000_000_000,
                        "context_length": 32768,
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                    {
                        "name": "llama3.3:70b-instruct-q4_K_M",
                        "size": 43_000_000_000,
                        "size_vram": 21_000_000_000,
                        "context_length": 131072,
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                    {
                        "name": "mistral-small:24b",
                        "size": 14_000_000_000,
                        "size_vram": 0,
                        "context_length": 5000,
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                ]
            }
        }

    def _gpu_processes(self):
        """Ollama holding VRAM on GPUs 0 and 2, as reported by ``top_gpu_processes``."""
        return [
            {"pid": 4242, "name": "ollama", "memory_used": 1, "gpu_index": 0},
            {"pid": 4242, "name": "ollama", "memory_used": 1, "gpu_index": 2},
        ]

    def test_narrow_region_keeps_model_and_ctx_readable(self):
        """Regression: at 80 columns Model/GPU/Ctx used to collapse to zero width.

        ``terminal_width=80`` mirrors what :func:`build_layout_content` passes on a
        standard terminal; 30 columns is a realistic estimate of the region that
        panel actually renders into once split off the rest of the layout.
        """
        panel = build_ollama_panel(
            self._data(), gpu_processes=self._gpu_processes(), terminal_width=80
        )

        text = _render(panel, width=30)

        assert "qwen3-coder" in text
        assert "32K" in text
        assert not _has_blank_row(text)

    def test_unconstrained_width_still_shows_every_column(self):
        """``terminal_width=None`` must keep the pre-fix, all-columns behaviour."""
        panel = build_ollama_panel(self._data(), gpu_processes=self._gpu_processes())

        assert _headers(panel.renderable) == [
            "Model", "GPU", "Ctx", "Size", "VRAM", "GPU%", "Expires",
        ]


class TestGpuProcessesPanelNarrowWidth:
    """Regression tests for the blank rows produced by a starved Cmdline column."""

    def _processes(self):
        """Two Ollama workers plus two unrelated compute apps, one unattributed."""
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

    def test_narrow_region_emits_no_blank_rows(self):
        """Regression: a starved Cmdline column used to wrap onto blank rows.

        ``terminal_width=80`` mirrors what :func:`build_layout_content` passes on a
        standard terminal; 30 columns is a realistic estimate of the region that
        panel actually renders into once split off the rest of the layout.
        """
        panel = build_gpu_processes_panel(
            {"top_gpu_processes": self._processes()}, multi_gpu=True, terminal_width=80
        )

        text = _render(panel, width=30)

        # The Name column itself may still be squeezed down to an ellipsis at
        # this width; what this fix guarantees is that every process keeps its
        # own single-line row instead of one bleeding into blank continuation
        # lines underneath it.
        assert "8888" in text
        assert "9999" in text
        assert not _has_blank_row(text)

    def test_narrow_region_keeps_the_memory_value_and_unit_intact(self):
        """Regression: even with Cmdline dropped, an uncapped Name and Memory
        Used still overflowed the panel, clipping the unit off the memory
        figure (e.g. ``16.6`` instead of ``16.6 GB``) and losing the right
        border entirely.

        ``terminal_width=80`` mirrors what :func:`build_layout_content` passes
        on a standard terminal; 32 columns is a realistic estimate of the
        panel's actual box width once split off the rest of the layout.
        """
        panel = build_gpu_processes_panel(
            {"top_gpu_processes": self._processes()}, multi_gpu=True, terminal_width=80
        )

        text = _render(panel, width=32)

        # 17_825_792_000 bytes converts to exactly "16.6 GB": the unit must
        # survive alongside the GPU index and PID.
        assert "16.6 GB" in text
        assert "4242" in text
        assert not _first_column_collapsed(text)
        assert not _has_blank_row(text)
        assert all(len(line) <= 32 for line in text.splitlines())

    def test_unconstrained_width_still_shows_the_cmdline_column(self):
        """``terminal_width=None`` must keep the pre-fix, all-columns behaviour."""
        panel = build_gpu_processes_panel(
            {"top_gpu_processes": self._processes()}, multi_gpu=True
        )

        assert _headers(panel.renderable) == ["GPU", "PID", "Name", "Memory Used", "Cmdline"]


class TestGpuDetailPanelNarrowWidth:
    """Regression tests for the row-per-GPU fallback collapsing at 80 columns."""

    def _gpus(self):
        """Three cards, shaped so the row fallback is the only viable layout."""
        return [_gpu(index, f"GPU{index}") for index in range(3)]

    def test_narrow_region_keeps_gpu_and_vram_intact(self):
        """Regression: at 80 columns the GPU column used to collapse to zero
        width (missing ``no_wrap`` on Name let Rich starve the whole table
        unpredictably), the Power column ran past the panel's right edge, and
        even with proactive column-dropping, uncapped columns (Name, Temp,
        Fan, Power, ``%``) still let Rich's last-resort shrink crush the Fan
        and Power columns to nothing and clip the unit off the Temp value
        (``78`` instead of ``78 °C``), because Rich measures a column's full
        *natural* content width regardless of ``min_width``/``no_wrap``.

        ``terminal_width=80`` mirrors what :func:`build_layout_content` passes
        on a standard terminal; 54 columns is the actual box width measured
        for the bottom-right region at that terminal width with 3 GPUs.
        """
        panel = build_gpu_detail_panel({"has_gpu": True, "gpu": self._gpus()}, terminal_width=80)

        text = _render(panel, width=54)

        assert not _first_column_collapsed(text)
        assert "12.0 GB" in text
        # Temp/Fan/Power must keep their units, not just their digits: this
        # is exactly what a partial Rich last-resort shrink used to clip.
        assert "61 °C" in text
        assert "30 %" in text
        assert "220 W" in text
        assert not _has_blank_row(text)
        assert all(len(line) <= 54 for line in text.splitlines())

    def test_unconstrained_width_still_shows_every_column(self):
        """``terminal_width=None`` must keep the pre-fix, all-columns behaviour."""
        gpus = [_gpu(index, f"GPU{index}") for index in range(4)]

        panel = build_gpu_detail_panel({"has_gpu": True, "gpu": gpus})

        assert _headers(panel.renderable) == [
            "GPU", "Name", "Util", "VRAM", "%", "Temp", "Fan", "Power",
        ]


class TestProcessTableNarrowWidth:
    """Regression tests for the Name/Cmdline columns vanishing at 80 columns."""

    def _processes(self):
        """Two processes with distinct, identifiable names."""
        return [
            {"pid": 4242, "name": "python3.12", "cpu_percent": 412.0,
             "cmdline": "/usr/bin/python3.12 -m sys_stats.server"},
            {"pid": 8888, "name": "ollama", "cpu_percent": 88.5,
             "cmdline": "/usr/local/bin/ollama serve"},
        ]

    def test_narrow_region_keeps_the_process_name_readable(self):
        """Regression: at 80 columns the Name and Cmdline columns used to
        vanish entirely, leaving a PID and a percentage with no way to tell
        which process they belonged to.

        ``terminal_width=80`` mirrors what :func:`build_layout_content` passes
        on a standard terminal; 26 columns is a realistic estimate of what
        each of the two side-by-side tables gets once top_right is split.
        """
        table = build_process_table(self._processes(), "top_cpu", "CPU", terminal_width=80)

        text = _render(table, width=26)

        assert "python3" in text
        assert "4242" in text
        assert not _has_blank_row(text)

    def test_unconstrained_width_still_shows_every_column(self):
        """``terminal_width=None`` must keep the pre-fix, all-columns behaviour."""
        table = build_process_table(self._processes(), "top_cpu", "CPU")

        assert _headers(table) == ["PID", "Name", "CPU%", "Cmdline"]

    def test_build_processes_panel_threads_the_terminal_width(self):
        """The Top CPU / Top Memory tables must see their halved share of
        top_right, not the full terminal width."""
        data = {"top_cpu": self._processes(), "top_memory": []}

        grid = build_processes_panel(data, terminal_width=80)
        cpu_panel = grid.columns[0]._cells[0]

        assert _headers(cpu_panel.renderable) == ["PID", "Name"]
