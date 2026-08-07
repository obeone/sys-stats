"""Tests for the metric collectors of :mod:`sys_stats.server`.

The collectors shell out to ``nvidia-smi`` and talk to the Ollama API, so every
test here replaces those boundaries with fakes. What matters is the parsing and
the degradation behaviour: a missing GPU or a dead Ollama must yield empty data,
never an exception, because ``/stats`` has no other error channel.
"""

import subprocess

import psutil
import pytest
import requests

from sys_stats import server


class _FakeCompletedProcess:
    """Minimal stand-in for :class:`subprocess.CompletedProcess`."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""


def _fake_run(stdout: str):
    """Build a ``subprocess.run`` replacement returning a fixed stdout."""

    def _run(*args, **kwargs):
        return _FakeCompletedProcess(stdout)

    return _run


def _failing_run(stderr: str = "no devices"):
    """Build a ``subprocess.run`` replacement raising ``CalledProcessError``."""

    def _run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi", stderr=stderr)

    return _run


def _missing_binary_run():
    """Build a ``subprocess.run`` replacement raising ``FileNotFoundError``.

    Simulates a host where the ``nvidia-smi`` binary is simply not installed
    (no NVIDIA driver at all), as opposed to ``_failing_run`` which simulates
    the binary existing but the command failing.
    """

    def _run(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "nvidia-smi")

    return _run


class TestGetGpuFanAndPower:
    def test_parses_one_entry_per_gpu_index(self, monkeypatch):
        """The CSV output of nvidia-smi is keyed by GPU index."""
        monkeypatch.setattr(
            server.subprocess, "run", _fake_run("0, 25, 30.5\n1, 40, 120.0\n")
        )

        assert server.get_gpu_fan_and_power() == {
            0: {"fan_speed": 25.0, "power_draw": 30.5},
            1: {"fan_speed": 40.0, "power_draw": 120.0},
        }

    def test_skips_malformed_lines(self, monkeypatch):
        """Cards reporting ``N/A`` (laptop dGPUs, passive cards) are dropped, not fatal."""
        monkeypatch.setattr(
            server.subprocess, "run", _fake_run("0, N/A, 30.5\n1, 40, 120.0\n")
        )

        assert list(server.get_gpu_fan_and_power()) == [1]

    def test_returns_empty_mapping_when_nvidia_smi_fails(self, monkeypatch):
        """A failing nvidia-smi degrades to no fan/power data at all."""
        monkeypatch.setattr(server.subprocess, "run", _failing_run())

        assert server.get_gpu_fan_and_power() == {}

    def test_returns_empty_mapping_when_nvidia_smi_is_absent(self, monkeypatch):
        """A missing ``nvidia-smi`` binary must degrade like a failing one.

        Regression guard: ``subprocess.run`` raises ``FileNotFoundError`` when
        the binary is not on ``PATH`` at all, which is a different exception
        than ``CalledProcessError``. The collector's contract (CLAUDE.md) is
        to degrade to ``{}``, not to propagate the exception.
        """
        monkeypatch.setattr(server.subprocess, "run", _missing_binary_run())

        assert server.get_gpu_fan_and_power() == {}


class TestGetGpuProcesses:
    @pytest.fixture(autouse=True)
    def _stub_psutil_process(self, monkeypatch):
        """Resolve every PID to a fixed command line."""

        class _FakeProcess:
            def __init__(self, pid):
                self.pid = pid

            def cmdline(self):
                return ["python", "train.py"]

        monkeypatch.setattr(server.psutil, "Process", _FakeProcess)

    def test_converts_mib_to_bytes_and_sorts_by_usage(self, monkeypatch):
        """nvidia-smi reports MiB; ``/stats`` must expose bytes, biggest first."""
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run(
                "GPU-aaa, 100, /usr/bin/python3, 512\n"
                "GPU-aaa, 200, /opt/ollama/ollama, 2048\n"
            ),
        )

        processes = server.get_gpu_processes()

        assert [p["pid"] for p in processes] == [200, 100]
        assert processes[0]["memory_used"] == 2048 * 1024 * 1024
        assert processes[0]["name"] == "ollama"
        assert processes[0]["cmdline"] == "python train.py"

    def test_honours_the_limit(self, monkeypatch):
        """Only the ``limit`` heaviest processes are returned."""
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run("GPU-aaa, 1, a, 10\nGPU-aaa, 2, b, 20\nGPU-aaa, 3, c, 30\n"),
        )

        assert len(server.get_gpu_processes(limit=2)) == 2

    def test_falls_back_to_na_for_vanished_processes(self, monkeypatch):
        """A PID that died between nvidia-smi and psutil yields ``N/A``, not a crash."""

        def _raise(pid):
            raise psutil.NoSuchProcess(pid)

        monkeypatch.setattr(server.psutil, "Process", _raise)
        monkeypatch.setattr(
            server.subprocess, "run", _fake_run("GPU-aaa, 100, python3, 512\n")
        )

        assert server.get_gpu_processes()[0]["cmdline"] == "N/A"

    def test_returns_empty_list_when_nvidia_smi_fails(self, monkeypatch):
        """No GPU compute apps and a broken nvidia-smi look the same to callers.

        Both the UUID-aware query and the legacy one fail here, which is the
        only case where the whole list is given up.
        """
        monkeypatch.setattr(server.subprocess, "run", _failing_run())

        assert server.get_gpu_processes() == []

    def test_returns_empty_list_when_nvidia_smi_is_absent(self, monkeypatch):
        """A missing ``nvidia-smi`` binary must degrade like a failing one.

        Regression guard: ``subprocess.run`` raises ``FileNotFoundError`` on a
        host without the NVIDIA driver installed at all (e.g. macOS), a
        different exception than ``CalledProcessError``. The collector's
        contract (CLAUDE.md) is to degrade to ``[]``, not to propagate it.
        """
        monkeypatch.setattr(server.subprocess, "run", _missing_binary_run())

        assert server.get_gpu_processes() == []

    def test_skips_malformed_lines_without_dropping_the_batch(self, monkeypatch):
        """A truncated row costs its own line, not the other processes."""
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run("GPU-aaa, 100\nGPU-aaa, 200, /usr/bin/python3, 512\n"),
        )

        processes = server.get_gpu_processes()

        assert [p["pid"] for p in processes] == [200]

    def test_resolves_the_gpu_index_through_the_uuid_mapping(self, monkeypatch):
        """The card of a compute app is only knowable through its UUID."""
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run(
                "GPU-bbb, 200, /opt/ollama/ollama, 2048\n"
                "GPU-aaa, 100, /usr/bin/python3, 512\n"
            ),
        )

        processes = server.get_gpu_processes(
            uuid_to_index={"GPU-aaa": 0, "GPU-bbb": 1}
        )

        assert [(p["gpu_index"], p["gpu_uuid"]) for p in processes] == [
            (0, "GPU-aaa"),
            (1, "GPU-bbb"),
        ]

    def test_leaves_the_gpu_index_unresolved_without_a_mapping(self, monkeypatch):
        """Called on its own, the collector still reports the raw UUID."""
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run("GPU-aaa, 100, /usr/bin/python3, 512\n"),
        )

        process = server.get_gpu_processes()[0]

        assert process["gpu_uuid"] == "GPU-aaa"
        assert process["gpu_index"] is None

    def test_leaves_the_gpu_index_unresolved_for_an_unknown_uuid(self, monkeypatch):
        """A UUID absent from the mapping is not an error, just an unknown card."""
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run("GPU-zzz, 100, /usr/bin/python3, 512\n"),
        )

        assert server.get_gpu_processes(uuid_to_index={"GPU-aaa": 0})[0]["gpu_index"] is None

    def test_sorts_by_gpu_index_then_by_descending_memory(self, monkeypatch):
        """Processes of one card stay contiguous, heaviest first within the card."""
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run(
                "GPU-bbb, 1, a, 4096\n"
                "GPU-aaa, 2, b, 512\n"
                "GPU-bbb, 3, c, 8192\n"
                "GPU-aaa, 4, d, 1024\n"
            ),
        )

        processes = server.get_gpu_processes(uuid_to_index={"GPU-aaa": 0, "GPU-bbb": 1})

        assert [p["pid"] for p in processes] == [4, 2, 3, 1]

    def test_pushes_unresolved_cards_last(self, monkeypatch):
        """``gpu_index`` is None for unknown cards, and None does not compare with int.

        Regression guard: a naive ``(gpu_index, -memory_used)`` key raises
        TypeError as soon as one row is unattributed.
        """
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run(
                "GPU-zzz, 1, a, 8192\n"
                "GPU-aaa, 2, b, 512\n"
                "GPU-zzz, 3, c, 16384\n"
            ),
        )

        processes = server.get_gpu_processes(uuid_to_index={"GPU-aaa": 0})

        assert [p["pid"] for p in processes] == [2, 3, 1]

    def test_limit_selects_by_vram_before_grouping_by_gpu(self, monkeypatch):
        """The limit must select the heaviest processes across all cards.

        Regression guard: the sort used to key on ``gpu_index`` first and
        truncate to ``limit`` afterwards, so on a multi-GPU host the cutoff
        kept whichever cards sorted first by index and dropped heavier
        processes sitting on higher-numbered cards. Here GPU 1 (the
        higher-index card) carries the single heaviest process (an "ollama"
        job) among five lighter ones on GPU 0; it must survive a ``limit=5``
        truncation, and the surviving rows must still be grouped by card for
        display.
        """
        monkeypatch.setattr(
            server.subprocess,
            "run",
            _fake_run(
                "GPU-aaa, 1, small1, 100\n"
                "GPU-aaa, 2, small2, 200\n"
                "GPU-aaa, 3, small3, 300\n"
                "GPU-aaa, 4, small4, 400\n"
                "GPU-aaa, 5, small5, 500\n"
                "GPU-bbb, 6, ollama, 999999\n"
            ),
        )

        processes = server.get_gpu_processes(
            limit=5, uuid_to_index={"GPU-aaa": 0, "GPU-bbb": 1}
        )

        pids = [p["pid"] for p in processes]
        assert len(pids) == 5
        assert 6 in pids, "heaviest process, on the higher-index card, was truncated away"
        assert 1 not in pids, "the lightest process should have been dropped instead"
        # Rows of the same card stay contiguous, heaviest first within the card.
        assert pids == [5, 4, 3, 2, 6]

    def test_falls_back_to_the_legacy_query_when_gpu_uuid_is_rejected(self, monkeypatch):
        """An old driver rejects ``gpu_uuid``; losing the card beats losing the list."""
        queries = []

        def _run(args, **kwargs):
            queries.append(args[1])
            if "gpu_uuid" in args[1]:
                raise subprocess.CalledProcessError(
                    1, "nvidia-smi", stderr="Field gpu_uuid is not supported"
                )
            return _FakeCompletedProcess("100, /usr/bin/python3, 512\n")

        monkeypatch.setattr(server.subprocess, "run", _run)

        processes = server.get_gpu_processes(uuid_to_index={"GPU-aaa": 0})

        assert queries == [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--query-compute-apps=pid,process_name,used_memory",
        ]
        assert len(processes) == 1
        assert processes[0]["pid"] == 100
        assert processes[0]["gpu_uuid"] is None
        assert processes[0]["gpu_index"] is None
        assert processes[0]["memory_used"] == 512 * 1024 * 1024


class TestGetOllamaProcess:
    def test_returns_no_models_when_url_is_unset(self, monkeypatch):
        """Without ``OLLAMA_API_URL`` the panel stays empty and no request is made."""
        monkeypatch.setattr(server, "OLLAMA_API_URL", None)

        def _explode(*args, **kwargs):
            raise AssertionError("no HTTP call expected when OLLAMA_API_URL is unset")

        monkeypatch.setattr(server.requests, "get", _explode)

        assert server.get_ollama_process() == {"models": []}

    def test_queries_the_ps_endpoint_and_returns_the_payload(self, monkeypatch):
        """The collector hits ``/api/ps`` and passes the JSON straight through."""
        monkeypatch.setattr(server, "OLLAMA_API_URL", "http://ollama.example:11434")
        payload = {"models": [{"name": "llama3", "size": 42}]}
        called = {}

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        def _get(url, timeout=None):
            called["url"] = url
            return _FakeResponse()

        monkeypatch.setattr(server.requests, "get", _get)

        assert server.get_ollama_process() == payload
        assert called["url"] == "http://ollama.example:11434/api/ps"

    def test_degrades_to_no_models_when_ollama_is_unreachable(self, monkeypatch):
        """A dead Ollama must not break the whole ``/stats`` response."""
        monkeypatch.setattr(server, "OLLAMA_API_URL", "http://ollama.example:11434")

        def _get(*args, **kwargs):
            raise requests.RequestException("connection refused")

        monkeypatch.setattr(server.requests, "get", _get)

        assert server.get_ollama_process() == {"models": []}


class TestTopProcesses:
    def test_cpu_ranking_is_sorted_and_limited(self, monkeypatch):
        """Processes come back sorted by CPU usage, truncated to ``limit``."""
        monkeypatch.setattr(
            server.psutil,
            "process_iter",
            lambda attrs: iter(
                [
                    _FakeProcInfo({"pid": 1, "name": "idle", "cpu_percent": 0.5, "cmdline": []}),
                    _FakeProcInfo(
                        {"pid": 2, "name": "busy", "cpu_percent": 90.0, "cmdline": ["busy", "--go"]}
                    ),
                ]
            ),
        )

        top = server.get_top_processes_by_cpu(limit=1)

        assert len(top) == 1
        assert top[0]["name"] == "busy"
        assert top[0]["cmdline"] == "busy --go"

    def test_cpu_ranking_reports_na_for_empty_cmdline(self, monkeypatch):
        """Kernel threads have no command line and must not render as an empty cell."""
        monkeypatch.setattr(
            server.psutil,
            "process_iter",
            lambda attrs: iter(
                [_FakeProcInfo({"pid": 1, "name": "kthreadd", "cpu_percent": 1.0, "cmdline": []})]
            ),
        )

        assert server.get_top_processes_by_cpu()[0]["cmdline"] == "N/A"

    def test_cpu_ranking_survives_unreadable_attributes(self, monkeypatch):
        """Regression: ``process_iter`` yields None for denied attributes.

        Sorting None against a float used to raise TypeError and turn the whole
        ``/stats`` response into a 500 on any unprivileged host (macOS, or a
        container started without ``pid: host`` and ``privileged``).
        """
        monkeypatch.setattr(
            server.psutil,
            "process_iter",
            lambda attrs: iter(
                [
                    _FakeProcInfo({"pid": 1, "name": "root-owned", "cpu_percent": None, "cmdline": None}),
                    _FakeProcInfo({"pid": 2, "name": "mine", "cpu_percent": 5.0, "cmdline": ["mine"]}),
                ]
            ),
        )

        top = server.get_top_processes_by_cpu()

        assert [p["name"] for p in top] == ["mine", "root-owned"]
        assert top[1]["cpu_percent"] == 0.0
        assert top[1]["cmdline"] == "N/A"

    def test_memory_ranking_survives_unreadable_attributes(self, monkeypatch):
        """Regression: a None ``memory_info`` must not raise AttributeError."""
        monkeypatch.setattr(
            server.psutil,
            "process_iter",
            lambda attrs: iter(
                [
                    _FakeProcInfo(
                        {
                            "pid": 1,
                            "name": "root-owned",
                            "memory_percent": None,
                            "memory_info": None,
                            "cmdline": None,
                        }
                    ),
                ]
            ),
        )

        top = server.get_top_processes_by_memory()

        assert top[0]["memory_usage"] == 0
        assert top[0]["memory_percent"] == 0.0

    def test_memory_ranking_exposes_rss_in_bytes(self, monkeypatch):
        """``memory_usage`` is the RSS in bytes, ranked by percentage."""
        monkeypatch.setattr(
            server.psutil,
            "process_iter",
            lambda attrs: iter(
                [
                    _FakeProcInfo(
                        {
                            "pid": 1,
                            "name": "small",
                            "memory_percent": 1.0,
                            "memory_info": _FakeMemoryInfo(1024),
                            "cmdline": ["small"],
                        }
                    ),
                    _FakeProcInfo(
                        {
                            "pid": 2,
                            "name": "large",
                            "memory_percent": 50.0,
                            "memory_info": _FakeMemoryInfo(4096),
                            "cmdline": ["large"],
                        }
                    ),
                ]
            ),
        )

        top = server.get_top_processes_by_memory()

        assert [p["name"] for p in top] == ["large", "small"]
        assert top[0]["memory_usage"] == 4096


class _FakeMemoryInfo:
    """Stand-in for the ``memory_info`` namedtuple exposed by psutil."""

    def __init__(self, rss: int) -> None:
        self.rss = rss


class _FakeProcInfo:
    """Stand-in for a :class:`psutil.Process` yielded by ``process_iter``."""

    def __init__(self, info: dict) -> None:
        self.info = info
