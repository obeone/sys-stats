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
