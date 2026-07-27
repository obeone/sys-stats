"""Tests for the Rich panel builders of the terminal dashboard.

The builders are pure: they take a ``/stats`` payload and return renderables, so
they can be asserted on without a terminal. Only structure is checked here (how
many tables, which rows), never the ANSI output.
"""

import pytest

from sys_stats.cli import (
    build_gpu_processes_panel,
    build_gpu_summary,
    build_ollama_panel,
    build_process_table,
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
