"""Contract tests for the ``/stats`` payload.

Both consumers (the inline JS dashboard and the Rich CLI) read this JSON, so the
key names and the units are the actual public API of the project. These tests
pin them down.
"""

import importlib

import pytest

from sys_stats import collectors, sampler, server


class _FakeVirtualMemory:
    """Stand-in for the namedtuple returned by ``psutil.virtual_memory``."""

    total = 16 * 1024**3
    used = 8 * 1024**3
    percent = 50.0


class _FakeGPU:
    """Stand-in for a :class:`GPUtil.GPU`, which reports VRAM in MiB."""

    def __init__(self, gpu_id: int = 0, uuid: str | None = None) -> None:
        self.id = gpu_id
        self.name = "NVIDIA GeForce RTX 3090"
        self.load = 0.42
        self.memoryTotal = 24576
        self.memoryUsed = 12288
        self.temperature = 61.0
        # The attribute is deliberately absent unless a UUID is asked for: that
        # is what an older GPUtil looks like, and ``get_stats`` must survive it.
        if uuid is not None:
            self.uuid = uuid


class _FakeCompletedProcess:
    """Minimal stand-in for :class:`subprocess.CompletedProcess`."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""


class _FakeProcess:
    """Stand-in for :class:`psutil.Process`, resolving any PID to a command line."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def cmdline(self) -> list[str]:
        """Return a fixed command line.

        Returns
        -------
        list of str
            The argv of the fake process.
        """
        return ["python", "train.py"]


def _seed_cache() -> None:
    """Collect one sample synchronously under whatever stubs are active.

    ``/stats`` now serves ``sampler``'s cache instead of collecting inline
    (see :mod:`sys_stats.sampler`), so a test that changes a collector stub
    after the ``client`` fixture ran must re-seed the cache before hitting
    the route, exactly where the old inline call would have picked the
    change up on its own.
    """
    sampler._sample_once(limit=sampler._get_top_processes_cap())


@pytest.fixture
def client(monkeypatch):
    """A Flask test client with every host-level collector stubbed out."""
    monkeypatch.setattr(collectors.psutil, "cpu_percent", lambda interval=None: 12.5)
    monkeypatch.setattr(collectors.psutil, "cpu_count", lambda logical=True: 8)
    monkeypatch.setattr(collectors.psutil, "virtual_memory", _FakeVirtualMemory)
    monkeypatch.setattr(collectors, "get_top_processes_by_cpu", lambda limit=5: [])
    monkeypatch.setattr(collectors, "get_top_processes_by_memory", lambda limit=5: [])
    monkeypatch.setattr(collectors, "get_ollama_process", lambda: {"models": []})
    monkeypatch.setattr(collectors.GPUtil, "getGPUs", lambda: [])

    server.app.config.update(TESTING=True)

    # Start every test from a clean sampler: no leftover thread from a
    # previous test, no leftover cached sample. Seed one sample synchronously
    # under the stubs above rather than starting the real background thread,
    # so tests stay instant and deterministic instead of depending on
    # wall-clock timing.
    sampler._reset_for_tests()
    _seed_cache()

    yield server.app.test_client()

    sampler._reset_for_tests()


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
    monkeypatch.setattr(collectors.GPUtil, "getGPUs", lambda: [_FakeGPU()])
    monkeypatch.setattr(
        collectors, "get_gpu_fan_and_power", lambda: {0: {"fan_speed": 30.0, "power_draw": 220.0}}
    )
    monkeypatch.setattr(collectors, "get_gpu_processes", lambda limit=5, uuid_to_index=None: [])
    _seed_cache()

    gpu = client.get("/stats").get_json()["gpu"][0]

    assert gpu["memoryUsed"] == 12288 * 1024 * 1024
    assert gpu["memoryTotal"] == 24576  # still MiB, as GPUtil reports it
    assert gpu["memoryPercent"] == pytest.approx(50.0)
    assert gpu["load"] == pytest.approx(42.0)  # fraction scaled to a percentage
    assert gpu["fanSpeed"] == 30.0
    assert gpu["powerDraw"] == 220.0


def test_stats_summary_mirrors_the_first_gpu(client, monkeypatch):
    """The summary block is a condensed view of GPU 0 for the compact panels."""
    monkeypatch.setattr(collectors.GPUtil, "getGPUs", lambda: [_FakeGPU()])
    monkeypatch.setattr(collectors, "get_gpu_fan_and_power", lambda: {})
    monkeypatch.setattr(collectors, "get_gpu_processes", lambda limit=5, uuid_to_index=None: [])
    _seed_cache()

    summary = client.get("/stats").get_json()["summary"]

    assert summary["cpu"] == {"usage": 12.5, "cores": 8}
    assert summary["gpu"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert summary["gpu"][0]["vram"] == pytest.approx(50.0)


def test_stats_defaults_to_zero_fan_and_power_when_nvidia_smi_is_silent(client, monkeypatch):
    """Missing fan/power data must not drop the GPU from the payload."""
    monkeypatch.setattr(collectors.GPUtil, "getGPUs", lambda: [_FakeGPU()])
    monkeypatch.setattr(collectors, "get_gpu_fan_and_power", lambda: {})
    monkeypatch.setattr(collectors, "get_gpu_processes", lambda limit=5, uuid_to_index=None: [])
    _seed_cache()

    gpu = client.get("/stats").get_json()["gpu"][0]

    assert gpu["fanSpeed"] == 0.0
    assert gpu["powerDraw"] == 0.0


def test_stats_attributes_gpu_processes_to_their_card(client, monkeypatch):
    """``top_gpu_processes`` entries name their GPU by UUID and by index.

    Exercises the real collector rather than a stub: the UUID mapping is built
    from the GPUtil devices inside ``get_stats``, so stubbing the collector
    would test nothing about that wiring.
    """
    monkeypatch.setattr(
        collectors.GPUtil,
        "getGPUs",
        lambda: [_FakeGPU(gpu_id=0, uuid="GPU-aaa"), _FakeGPU(gpu_id=1, uuid="GPU-bbb")],
    )
    monkeypatch.setattr(collectors, "get_gpu_fan_and_power", lambda: {})
    monkeypatch.setattr(
        collectors.subprocess,
        "run",
        lambda *a, **kw: _FakeCompletedProcess(
            "GPU-bbb, 200, /opt/ollama/ollama, 2048\nGPU-aaa, 100, /usr/bin/python3, 512\n"
        ),
    )
    monkeypatch.setattr(collectors.psutil, "Process", lambda pid: _FakeProcess(pid))
    _seed_cache()

    processes = client.get("/stats").get_json()["top_gpu_processes"]

    assert [(p["gpu_index"], p["gpu_uuid"], p["pid"]) for p in processes] == [
        (0, "GPU-aaa", 100),
        (1, "GPU-bbb", 200),
    ]


def test_stats_leaves_the_gpu_index_unresolved_for_an_unknown_uuid(client, monkeypatch):
    """A compute app on a card GPUtil did not enumerate still shows up."""
    monkeypatch.setattr(collectors.GPUtil, "getGPUs", lambda: [_FakeGPU(uuid="GPU-aaa")])
    monkeypatch.setattr(collectors, "get_gpu_fan_and_power", lambda: {})
    monkeypatch.setattr(
        collectors.subprocess,
        "run",
        lambda *a, **kw: _FakeCompletedProcess("GPU-zzz, 100, /usr/bin/python3, 512\n"),
    )
    monkeypatch.setattr(collectors.psutil, "Process", lambda pid: _FakeProcess(pid))
    _seed_cache()

    process = client.get("/stats").get_json()["top_gpu_processes"][0]

    assert process["gpu_uuid"] == "GPU-zzz"
    assert process["gpu_index"] is None


def test_stats_forwards_the_limit_query_parameter(client, monkeypatch):
    """``?limit=`` slices the sampler's cached ranking down to that many entries.

    The sampler, not the route, decides how many entries to collect (up to
    its own cap); ``?limit=`` only controls how much of that cache a given
    request gets back.
    """
    processes = [
        {"pid": pid, "name": f"proc{pid}", "cpu_percent": 0.0, "cmdline": "N/A"} for pid in range(10)
    ]
    monkeypatch.setattr(collectors, "get_top_processes_by_cpu", lambda limit=5: processes[:limit])
    _seed_cache()

    response = client.get("/stats?limit=3").get_json()

    assert [p["pid"] for p in response["top_cpu"]] == [0, 1, 2]


def test_stats_falls_back_to_five_on_a_non_numeric_limit(client, monkeypatch):
    """A bogus ``limit`` is ignored, defaulting to 5, rather than returning a 500."""
    processes = [
        {"pid": pid, "name": f"proc{pid}", "cpu_percent": 0.0, "cmdline": "N/A"} for pid in range(10)
    ]
    monkeypatch.setattr(collectors, "get_top_processes_by_cpu", lambda limit=5: processes[:limit])
    _seed_cache()

    response = client.get("/stats?limit=banana")

    assert response.status_code == 200
    assert [p["pid"] for p in response.get_json()["top_cpu"]] == [0, 1, 2, 3, 4]


def test_stats_limit_above_the_sampled_cap_returns_what_was_sampled(client, monkeypatch, caplog):
    """A ``?limit=`` beyond what the sampler collected returns what exists.

    The route must never re-collect inline to satisfy an oversized limit:
    it just hands back everything the sampler already has, and logs once.
    """
    processes = [
        {"pid": pid, "name": f"proc{pid}", "cpu_percent": 0.0, "cmdline": "N/A"} for pid in range(3)
    ]
    monkeypatch.setattr(collectors, "get_top_processes_by_cpu", lambda limit=5: processes[:limit])
    _seed_cache()  # sampled at the cap (default 50), but the stub only ever has 3 to give

    with caplog.at_level("WARNING"):
        response = client.get("/stats?limit=1000").get_json()

    assert [p["pid"] for p in response["top_cpu"]] == [0, 1, 2]
    assert any("exceeds" in record.message for record in caplog.records)


def test_stats_served_from_cache_matches_the_sampled_payload(client):
    """The route answers from the sampler's cache, not a fresh collection.

    Requesting exactly the sampled cap makes slicing a no-op, so the
    response must equal the snapshot the sampler already stored.
    """
    cached_stats, _wall_ts, _monotonic_ts = sampler.get_snapshot()
    cap = sampler._get_top_processes_cap()

    response = client.get(f"/stats?limit={cap}").get_json()

    assert response == cached_stats


def test_slice_to_limit_returns_independent_list_copies():
    """Mutating a per-request response must never corrupt the shared cache."""
    cached = {
        "top_cpu": [{"pid": 1}],
        "top_memory": [{"pid": 2}],
        "top_gpu_processes": [{"pid": 3, "gpu_index": 0, "memory_used": 100}],
        "other_key": "unchanged",
    }

    sliced = server._slice_to_limit(cached, limit=5)
    sliced["top_cpu"].append({"pid": 999, "name": "intruder"})
    sliced["top_memory"] = "replaced"

    assert cached["top_cpu"] == [{"pid": 1}]
    assert cached["top_memory"] == [{"pid": 2}]
    assert cached["other_key"] == "unchanged"


def test_slice_to_limit_selects_top_gpu_processes_by_memory_before_truncating():
    """``limit`` must keep the heaviest GPU processes, not whichever GPU sorts first.

    The sampler stores ``top_gpu_processes`` in *display* order (grouped by
    ``gpu_index``, heaviest first within each card), not in descending
    ``memory_used`` order. Slicing that display-ordered list directly keeps
    whichever card happens to sort first rather than the biggest VRAM
    consumers. Reproduces the review's repro case: 10 processes at 100 MiB on
    GPU 0 plus 3 processes at 10 GiB on GPU 1, with ``limit=3``.
    """
    light = [
        {
            "pid": pid,
            "gpu_index": 0,
            "gpu_uuid": "GPU-light",
            "memory_used": 100 * 1024 * 1024,
            "name": f"light{pid}",
        }
        for pid in range(10)
    ]
    heavy = [
        {
            "pid": 100 + pid,
            "gpu_index": 1,
            "gpu_uuid": "GPU-heavy",
            "memory_used": 10 * 1024**3,
            "name": f"heavy{pid}",
        }
        for pid in range(3)
    ]
    # Already in display order: GPU 0's (lighter) processes first, then GPU 1's.
    cached = {
        "top_cpu": [],
        "top_memory": [],
        "top_gpu_processes": light + heavy,
    }

    sliced = server._slice_to_limit(cached, limit=3)

    assert {p["pid"] for p in sliced["top_gpu_processes"]} == {100, 101, 102}


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


def test_stats_returns_503_when_no_snapshot_lands(client, monkeypatch):
    """An empty payload is a lie; a genuine cold-start failure must 503.

    Both consumers (the inline JS and the Rich CLI) expect every key in the
    contract to be present. A ``200 {}`` response after the wait window
    elapses without a sample silently breaks both instead of surfacing the
    failure.
    """
    monkeypatch.setattr(sampler, "get_snapshot", lambda: (None, None, None))
    monkeypatch.setattr(sampler, "wait_for_first_snapshot", lambda timeout: (None, None, None))

    response = client.get("/stats")

    assert response.status_code == 503
    assert response.get_json() == {}


def test_importing_the_server_module_starts_the_sampler(monkeypatch):
    """Any WSGI entry point that only imports ``app`` must get a live sampler.

    Before this fix, ``sampler.start()`` was called solely from
    ``server.main()``. Running under ``gunicorn sys_stats.server:app`` or
    ``flask --app sys_stats.server run`` never invoked it, so ``/stats``
    burned the full ``wait_for_first_snapshot`` timeout on every request,
    forever.
    """
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    started = []
    # Patched before the reload, so the module-scope call the reload triggers
    # hits this stub instead of spawning a real background thread.
    monkeypatch.setattr(sampler, "start", lambda: started.append(True))

    importlib.reload(server)

    assert started == [True]
