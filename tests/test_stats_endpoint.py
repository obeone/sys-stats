"""Contract tests for the ``/stats`` payload.

Both consumers (the inline JS dashboard and the Rich CLI) read this JSON, so the
key names and the units are the actual public API of the project. These tests
pin them down.
"""

import pytest

from sys_stats import server


class _FakeVirtualMemory:
    """Stand-in for the namedtuple returned by ``psutil.virtual_memory``."""

    total = 16 * 1024**3
    used = 8 * 1024**3
    percent = 50.0


class _FakeGPU:
    """Stand-in for a :class:`GPUtil.GPU`, which reports VRAM in MiB."""

    def __init__(self, gpu_id: int = 0) -> None:
        self.id = gpu_id
        self.name = "NVIDIA GeForce RTX 3090"
        self.load = 0.42
        self.memoryTotal = 24576
        self.memoryUsed = 12288
        self.temperature = 61.0


@pytest.fixture
def client(monkeypatch):
    """A Flask test client with every host-level collector stubbed out."""
    monkeypatch.setattr(server.psutil, "cpu_percent", lambda interval=None: 12.5)
    monkeypatch.setattr(server.psutil, "cpu_count", lambda logical=True: 8)
    monkeypatch.setattr(server.psutil, "virtual_memory", _FakeVirtualMemory)
    monkeypatch.setattr(server, "get_top_processes_by_cpu", lambda limit=5: [])
    monkeypatch.setattr(server, "get_top_processes_by_memory", lambda limit=5: [])
    monkeypatch.setattr(server, "get_ollama_process", lambda: {"models": []})
    monkeypatch.setattr(server.GPUtil, "getGPUs", lambda: [])

    server.app.config.update(TESTING=True)
    return server.app.test_client()


def test_stats_exposes_the_full_payload_without_a_gpu(client):
    """On a GPU-less host every GPU section is empty but still present."""
    payload = client.get("/stats").get_json()

    assert set(payload) == {
        "current_time",
        "has_gpu",
        "summary",
        "cpu",
        "ram",
        "gpu",
        "top_cpu",
        "top_memory",
        "top_gpu_processes",
        "ollama_processes",
    }
    assert payload["has_gpu"] is False
    assert payload["gpu"] == []
    assert payload["top_gpu_processes"] == []
    assert payload["summary"]["gpu"] == []
    assert payload["cpu"] == 12.5
    assert payload["ram"] == {"total": 16 * 1024**3, "used": 8 * 1024**3, "percent": 50.0}


def test_stats_converts_gpu_memory_to_bytes(client, monkeypatch):
    """GPUtil reports MiB; the payload must carry bytes and a percentage."""
    monkeypatch.setattr(server.GPUtil, "getGPUs", lambda: [_FakeGPU()])
    monkeypatch.setattr(
        server, "get_gpu_fan_and_power", lambda: {0: {"fan_speed": 30.0, "power_draw": 220.0}}
    )
    monkeypatch.setattr(server, "get_gpu_processes", lambda limit=5: [])

    gpu = client.get("/stats").get_json()["gpu"][0]

    assert gpu["memoryUsed"] == 12288 * 1024 * 1024
    assert gpu["memoryTotal"] == 24576  # still MiB, as GPUtil reports it
    assert gpu["memoryPercent"] == pytest.approx(50.0)
    assert gpu["load"] == pytest.approx(42.0)  # fraction scaled to a percentage
    assert gpu["fanSpeed"] == 30.0
    assert gpu["powerDraw"] == 220.0


def test_stats_summary_mirrors_the_first_gpu(client, monkeypatch):
    """The summary block is a condensed view of GPU 0 for the compact panels."""
    monkeypatch.setattr(server.GPUtil, "getGPUs", lambda: [_FakeGPU()])
    monkeypatch.setattr(server, "get_gpu_fan_and_power", lambda: {})
    monkeypatch.setattr(server, "get_gpu_processes", lambda limit=5: [])

    summary = client.get("/stats").get_json()["summary"]

    assert summary["cpu"] == {"usage": 12.5, "cores": 8}
    assert summary["gpu"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert summary["gpu"][0]["vram"] == pytest.approx(50.0)


def test_stats_defaults_to_zero_fan_and_power_when_nvidia_smi_is_silent(client, monkeypatch):
    """Missing fan/power data must not drop the GPU from the payload."""
    monkeypatch.setattr(server.GPUtil, "getGPUs", lambda: [_FakeGPU()])
    monkeypatch.setattr(server, "get_gpu_fan_and_power", lambda: {})
    monkeypatch.setattr(server, "get_gpu_processes", lambda limit=5: [])

    gpu = client.get("/stats").get_json()["gpu"][0]

    assert gpu["fanSpeed"] == 0.0
    assert gpu["powerDraw"] == 0.0


def test_stats_forwards_the_limit_query_parameter(client, monkeypatch):
    """``?limit=`` controls how many processes each ranking returns."""
    seen = {}
    monkeypatch.setattr(
        server, "get_top_processes_by_cpu", lambda limit=5: seen.setdefault("cpu", limit) and []
    )

    client.get("/stats?limit=3")

    assert seen["cpu"] == 3


def test_stats_falls_back_to_five_on_a_non_numeric_limit(client, monkeypatch):
    """A bogus ``limit`` is ignored rather than returning a 500."""
    seen = {}
    monkeypatch.setattr(
        server, "get_top_processes_by_cpu", lambda limit=5: seen.setdefault("cpu", limit) and []
    )

    response = client.get("/stats?limit=banana")

    assert response.status_code == 200
    assert seen["cpu"] == 5


def test_index_serves_the_dashboard(client):
    """The web UI is served from the packaged template directory."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_favicon_is_packaged_alongside_the_template(client):
    """``/favicon.png`` is served from inside the installed package."""
    response = client.get("/favicon.png")

    assert response.status_code == 200
    assert response.mimetype == "image/png"
