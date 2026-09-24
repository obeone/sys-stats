"""Contract tests for the ``/panel/procs`` payload.

``/panel/procs`` serves the process and Ollama lists of ``/stats`` to the
wall panel, under the same keys and nesting, but rebuilt from an explicit
allowlist so it stays safe to register in ``SYS_STATS_PANEL_ONLY`` mode:
``/stats`` carries full command lines, which routinely hold tokens and
passwords passed as arguments.
"""

import importlib
import os

import pytest

from sys_stats import collectors, sampler, server


class _FakeVirtualMemory:
    """Stand-in for the namedtuple returned by ``psutil.virtual_memory``."""

    total = 16 * 1024**3
    used = 8 * 1024**3
    percent = 50.0


class _FakeGPU:
    """Stand-in for a :class:`GPUtil.GPU`, which reports VRAM in MiB."""

    def __init__(self, gpu_id: int = 0) -> None:
        """Build a fake GPU with a fixed identity.

        Parameters
        ----------
        gpu_id : int, optional
            The index GPUtil would report.
        """
        self.id = gpu_id
        self.uuid = f"GPU-{gpu_id}"
        self.name = "NVIDIA GeForce RTX 3090"
        self.load = 0.42
        self.memoryTotal = 24576  # MiB
        self.memoryUsed = 12288  # MiB
        self.temperature = 61.0


#: A secret planted in every collector's ``cmdline`` so a leak anywhere in
#: the response is caught by a plain substring search on the raw body.
_SECRET = "--password=hunter2"


def _fake_top_cpu(limit: int = 5) -> list[dict]:
    """Return a CPU ranking whose entries carry a secret-bearing ``cmdline``.

    Parameters
    ----------
    limit : int, optional
        Maximum number of entries, like the real collector.

    Returns
    -------
    list of dict
        Entries sorted by descending ``cpu_percent``.
    """
    rows = [
        {"pid": 100 + i, "name": f"cpu{i}", "cpu_percent": 90.0 - i, "cmdline": f"cpu{i} {_SECRET}"}
        for i in range(8)
    ]
    return rows[:limit]


def _fake_top_memory(limit: int = 5) -> list[dict]:
    """Return a memory ranking whose entries carry extra, non-allowlisted keys.

    Parameters
    ----------
    limit : int, optional
        Maximum number of entries, like the real collector.

    Returns
    -------
    list of dict
        Entries sorted by descending ``memory_percent``.
    """
    rows = [
        {
            "pid": 200 + i,
            "name": f"mem{i}",
            "memory_usage": (8 - i) * 1024**2,
            "memory_percent": 40.0 - i,
            "cmdline": f"mem{i} {_SECRET}",
            "username": "root",
        }
        for i in range(8)
    ]
    return rows[:limit]


def _fake_gpu_processes(limit: int = 5, uuid_to_index=None) -> list[dict]:
    """Return GPU compute apps whose entries carry a secret-bearing ``cmdline``.

    Parameters
    ----------
    limit : int, optional
        Maximum number of entries, like the real collector.
    uuid_to_index : dict, optional
        Ignored; accepted for signature compatibility.

    Returns
    -------
    list of dict
        Entries in the collector's display order.
    """
    rows = [
        {
            "pid": 300 + i,
            "name": f"gpu{i}",
            "memory_used": (8 - i) * 1024**3,
            "gpu_index": 0,
            "cmdline": f"gpu{i} {_SECRET}",
        }
        for i in range(8)
    ]
    return rows[:limit]


def _fake_ollama() -> dict:
    """Return an Ollama ``/api/ps`` answer with fields beyond the allowlist.

    Returns
    -------
    dict
        One resident model carrying digest, details and expiry on top of the
        allowlisted name/model/size/size_vram, plus a stray top-level key.
    """
    return {
        "models": [
            {
                "name": "llama3:8b",
                "model": "llama3:8b",
                "size": 6 * 1024**3,
                "size_vram": 5 * 1024**3,
                "digest": "sha256:abc",
                "details": {"family": "llama"},
                "expires_at": "2026-01-01T00:00:00Z",
                "cmdline": _SECRET,
            }
        ],
        "extra": _SECRET,
    }


@pytest.fixture
def client(monkeypatch):
    """A Flask test client with every collector stubbed and one sample cached."""
    monkeypatch.setattr(collectors.psutil, "cpu_percent", lambda interval=None: 12.5)
    monkeypatch.setattr(collectors.psutil, "cpu_count", lambda logical=True: 8)
    monkeypatch.setattr(collectors.psutil, "virtual_memory", _FakeVirtualMemory)
    monkeypatch.setattr(collectors, "get_top_processes_by_cpu", _fake_top_cpu)
    monkeypatch.setattr(collectors, "get_top_processes_by_memory", _fake_top_memory)
    monkeypatch.setattr(collectors, "get_gpu_processes", _fake_gpu_processes)
    monkeypatch.setattr(collectors, "get_gpu_fan_and_power", lambda: {})
    monkeypatch.setattr(collectors, "get_ollama_process", _fake_ollama)
    monkeypatch.setattr(collectors.GPUtil, "getGPUs", lambda: [_FakeGPU(0)])

    monkeypatch.setattr(collectors, "get_temperatures", lambda: [])
    monkeypatch.setattr(collectors, "get_ipmi_temperatures", lambda: [])
    monkeypatch.setattr(collectors, "get_fans", lambda: [])
    monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [])
    monkeypatch.setattr(collectors, "get_swap", lambda: {"used": 0, "total": 0, "pct": 0.0})
    monkeypatch.setattr(collectors, "get_per_core_cpu", lambda: [])
    monkeypatch.setattr(collectors, "get_load_average", lambda: [0.0, 0.0, 0.0])
    monkeypatch.setattr(collectors, "get_cpu_frequency_mhz", lambda: 0)

    server.app.config.update(TESTING=True)

    sampler._reset_for_tests()
    sampler._ipmi_last_poll_monotonic = None
    sampler._sample_once(limit=sampler._get_top_processes_cap())

    yield server.app.test_client()

    sampler._reset_for_tests()


def _all_keys(value) -> set[str]:
    """Collect every dict key found anywhere in a decoded JSON value.

    Parameters
    ----------
    value : Any
        A decoded JSON value.

    Returns
    -------
    set of str
        Every key of every nested object.
    """
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys |= _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            keys |= _all_keys(child)
    return keys


def test_procs_returns_exactly_the_allowlisted_shape(client):
    """Top-level keys and per-entry fields are exactly the allowlist."""
    response = client.get("/panel/procs")

    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload) == {"top_cpu", "top_memory", "top_gpu_processes", "ollama_processes"}
    # _fake_top_cpu/_fake_top_memory carry a joined "cmdline" but no "argv",
    # exactly like a real entry whose cmdline collection failed: args is "".
    assert payload["top_cpu"][0] == {"pid": 100, "name": "cpu0", "cpu_percent": 90.0, "args": ""}
    assert payload["top_memory"][0] == {
        "pid": 200,
        "name": "mem0",
        "memory_usage": 8 * 1024**2,
        "args": "",
    }
    assert payload["top_gpu_processes"][0] == {
        "pid": 300,
        "name": "gpu0",
        "memory_used": 8 * 1024**3,
        "gpu_index": 0,
    }
    assert payload["ollama_processes"] == {
        "models": [
            {"name": "llama3:8b", "model": "llama3:8b", "size_vram": 5 * 1024**3, "size": 6 * 1024**3}
        ]
    }


def test_procs_never_carries_cmdline_even_when_collectors_return_one(client):
    """Regression guard: no ``cmdline`` key and no secret anywhere in the body."""
    response = client.get("/panel/procs?limit=50")

    assert "cmdline" not in _all_keys(response.get_json())
    assert _SECRET not in response.get_data(as_text=True)


def test_procs_honours_limit_like_stats(client):
    """``?limit=`` caps every ranking, and a bad value falls back to 5."""
    payload = client.get("/panel/procs?limit=2").get_json()
    for key in ("top_cpu", "top_memory", "top_gpu_processes"):
        assert len(payload[key]) == 2

    payload = client.get("/panel/procs?limit=banana").get_json()
    for key in ("top_cpu", "top_memory", "top_gpu_processes"):
        assert len(payload[key]) == 5


def test_procs_ranks_the_same_rows_as_stats(client):
    """For a given limit, the pids match ``/stats`` row for row."""
    stats = client.get("/stats?limit=3").get_json()
    procs = client.get("/panel/procs?limit=3").get_json()

    for key in ("top_cpu", "top_memory", "top_gpu_processes"):
        assert [p["pid"] for p in procs[key]] == [p["pid"] for p in stats[key]]


def test_procs_args_happy_path_drops_argv0_and_truncates(client, monkeypatch):
    """``args`` is ``argv[1:]`` joined by spaces, ``argv[0]`` dropped, capped at 60 chars."""
    long_arg = "x" * 80
    monkeypatch.setattr(
        collectors,
        "get_top_processes_by_cpu",
        lambda limit=5: [
            {
                "pid": 1,
                "name": "qemu-kvm",
                "cpu_percent": 42.0,
                "cmdline": f"/usr/bin/qemu-kvm -name guest1 --extra {long_arg}",
                "argv": ["/usr/bin/qemu-kvm", "-name", "guest1", "--extra", long_arg],
            }
        ],
    )
    sampler._sample_once(limit=sampler._get_top_processes_cap())

    entry = client.get("/panel/procs").get_json()["top_cpu"][0]

    assert entry["args"] == " ".join(["-name", "guest1", "--extra", long_arg])[:60]
    assert len(entry["args"]) == 60
    assert "qemu-kvm" not in entry["args"]


def test_procs_args_empty_when_no_arguments(client, monkeypatch):
    """A process with only ``argv[0]`` (no arguments) yields ``args == ""``."""
    monkeypatch.setattr(
        collectors,
        "get_top_processes_by_memory",
        lambda limit=5: [
            {"pid": 2, "name": "sshd", "memory_usage": 1024, "cmdline": "sshd", "argv": ["sshd"]}
        ],
    )
    sampler._sample_once(limit=sampler._get_top_processes_cap())

    payload = client.get("/panel/procs").get_json()

    assert payload["top_memory"][0]["args"] == ""


def test_procs_args_empty_when_cmdline_unavailable(client, monkeypatch):
    """An ``N/A`` cmdline (``AccessDenied``, a zombie, a kernel thread) yields ``args == ""``."""
    monkeypatch.setattr(
        collectors,
        "get_top_processes_by_cpu",
        lambda limit=5: [
            {"pid": 3, "name": "kthreadd", "cpu_percent": 0.1, "cmdline": "N/A", "argv": []}
        ],
    )
    sampler._sample_once(limit=sampler._get_top_processes_cap())

    payload = client.get("/panel/procs").get_json()

    assert payload["top_cpu"][0]["args"] == ""


def test_procs_tolerates_a_malformed_ollama_answer(client, monkeypatch):
    """A non-dict Ollama reply degrades to an empty model list, not a 500."""
    monkeypatch.setattr(collectors, "get_ollama_process", lambda: ["unexpected"])
    sampler._sample_once(limit=sampler._get_top_processes_cap())

    payload = client.get("/panel/procs").get_json()

    assert payload["ollama_processes"] == {"models": []}


def test_procs_returns_503_before_any_snapshot(monkeypatch):
    """No usable sample means a retryable 503, never a 200 of empty lists."""
    monkeypatch.setattr(sampler, "get_snapshot", lambda: (None, None, None))
    monkeypatch.setattr(sampler, "wait_for_first_snapshot", lambda timeout: (None, None, None))
    server.app.config.update(TESTING=True)

    response = server.app.test_client().get("/panel/procs")

    assert response.status_code == 503
    assert response.get_json() == {}


class TestPanelOnlyRegistersProcs:
    """``/panel/procs`` survives ``SYS_STATS_PANEL_ONLY`` while ``/stats`` does not."""

    @pytest.fixture(autouse=True)
    def _reload_server_afterwards(self):
        """Reload ``sys_stats.server`` with the env unset after each test."""
        yield
        os.environ.pop("SYS_STATS_PANEL_ONLY", None)
        importlib.reload(server)

    def test_panel_only_serves_procs_and_404s_stats(self, client, monkeypatch):
        """In panel-only mode ``/panel/procs`` answers 200 and ``/stats`` 404."""
        monkeypatch.setenv("SYS_STATS_PANEL_ONLY", "1")
        importlib.reload(server)
        server.app.config.update(TESTING=True)
        panel_client = server.app.test_client()

        rules = {rule.rule for rule in server.app.url_map.iter_rules()}
        assert "/panel/procs" in rules
        assert "/stats" not in rules

        assert panel_client.get("/stats").status_code == 404
        response = panel_client.get("/panel/procs?limit=2")
        assert response.status_code == 200
        assert len(response.get_json()["top_cpu"]) == 2
        assert "cmdline" not in _all_keys(response.get_json())
