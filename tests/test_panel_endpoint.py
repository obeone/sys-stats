"""Contract tests for the ``/panel`` payload.

``/panel`` is a frozen, negotiated contract consumed by an ESP32-S3 wall
display (LVGL, no heap, a fixed POD struct, ArduinoJson filter). Every test
here pins the exact schema down: fixed keys always present, units, rounding,
truncation and the single most important non-functional property -- that it
shares one sampling pass with ``/stats`` instead of doubling the
``nvidia-smi`` cost this whole refactor exists to remove.
"""

import time

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
        self.id = gpu_id
        self.name = "NVIDIA GeForce RTX 3090"
        self.load = 0.42
        self.memoryTotal = 24576  # MiB
        self.memoryUsed = 12288  # MiB
        self.temperature = 61.0


def _seed_cache() -> None:
    """Collect one sample synchronously under whatever stubs are active."""
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
    _seed_cache()

    yield server.app.test_client()

    sampler._reset_for_tests()


def test_panel_returns_503_with_exact_body_before_any_snapshot(monkeypatch):
    """No sample yet must 503 with exactly ``{"v": 1, "ready": false}``.

    A payload of zeroes would render as a dead machine on the wall display,
    which is the whole reason this is a 503 and not a 200.
    """
    monkeypatch.setattr(sampler, "get_panel_snapshot", lambda: (None, None, None, None))
    server.app.config.update(TESTING=True)

    response = server.app.test_client().get("/panel")

    assert response.status_code == 503
    assert response.get_json() == {"v": 1, "ready": False}


def test_panel_exposes_exactly_the_frozen_top_level_keys(client):
    """Every key in the contract is always present, never conditional."""
    payload = client.get("/panel").get_json()

    assert set(payload) == {
        "v",
        "ts",
        "age",
        "ready",
        "host",
        "cpu",
        "mem",
        "swap",
        "gpu",
        "gpu_n",
        "temps",
        "temps_n",
        "fans",
        "fans_n",
        "err",
    }
    assert payload["v"] == 1
    assert payload["ready"] is True


def test_panel_cpu_and_mem_sections_without_a_gpu(client):
    """CPU and memory come straight from the shared /stats-side collection."""
    payload = client.get("/panel").get_json()

    assert payload["cpu"] == {
        "pct": 12.5,
        "n": 8,
        "mhz": 0,
        "load": [0.0, 0.0, 0.0],
        "per": [],
    }
    assert payload["mem"] == {"used": 8 * 1024**3, "total": 16 * 1024**3, "pct": 50.0}
    assert payload["swap"] == {"used": 0, "total": 0, "pct": 0.0}
    assert payload["gpu"] == []
    assert payload["gpu_n"] == 0
    assert payload["err"] == []


class TestPanelHost:
    """``host`` identifies which machine produced the sample.

    Unlike every other ``/panel`` guard (``ready``, ``age``, ``err``), this
    one exists to catch data that is fresh and correct but describing the
    wrong machine -- a duplicated IP, a repointed DNS record, a migrated
    DHCP lease, none of which raise an error on their own.
    """

    def test_falls_back_to_socket_gethostname_when_unset(self, client, monkeypatch):
        """Unset (default) reports the real host, verbatim, not shortened."""
        monkeypatch.delenv("SYS_STATS_HOSTNAME", raising=False)
        monkeypatch.setattr(server.socket, "gethostname", lambda: "bart.virt.obeone.org")

        payload = client.get("/panel").get_json()

        assert payload["host"] == "bart.virt.obeone.org"

    def test_reflects_the_env_override_when_set(self, client, monkeypatch):
        """SYS_STATS_HOSTNAME wins over socket.gethostname() when set."""
        monkeypatch.setattr(server.socket, "gethostname", lambda: "bart.virt.obeone.org")
        monkeypatch.setenv("SYS_STATS_HOSTNAME", "pod-abc123")

        payload = client.get("/panel").get_json()

        assert payload["host"] == "pod-abc123"

    def test_is_re_read_per_request_rather_than_cached(self, client, monkeypatch):
        """A hostname change must be picked up on the very next request.

        Regression guard: ``hostnamectl set-hostname`` changes the value at
        runtime, and a process that cached it at import/sample time would
        keep announcing the old name indefinitely.
        """
        monkeypatch.delenv("SYS_STATS_HOSTNAME", raising=False)
        hostnames = iter(["first-name", "second-name"])
        monkeypatch.setattr(server.socket, "gethostname", lambda: next(hostnames))

        first = client.get("/panel").get_json()
        second = client.get("/panel").get_json()

        assert first["host"] == "first-name"
        assert second["host"] == "second-name"


def test_panel_converts_gpu_memory_total_to_bytes(client, monkeypatch):
    """/stats keeps memoryTotal in MiB (issue #16); /panel must not inherit that.

    mem_used is already bytes upstream; mem_total is the field that needs an
    explicit conversion, tested here on its own so a regression that only
    breaks one of the two units is caught.
    """
    monkeypatch.setattr(collectors.GPUtil, "getGPUs", lambda: [_FakeGPU()])
    monkeypatch.setattr(
        collectors, "get_gpu_fan_and_power", lambda: {0: {"fan_speed": 30.0, "power_draw": 220.0}}
    )
    monkeypatch.setattr(collectors, "get_gpu_processes", lambda limit=5, uuid_to_index=None: [])
    _seed_cache()

    gpu = client.get("/panel").get_json()["gpu"][0]

    assert gpu["mem_used"] == 12288 * 1024 * 1024
    assert gpu["mem_total"] == 24576 * 1024 * 1024
    assert gpu["mem_pct"] == pytest.approx(50.0)
    assert gpu["i"] == 0
    assert gpu["n"] == "NVIDIA GeForce RTX 3090"
    assert gpu["load"] == pytest.approx(42.0)
    assert gpu["temp"] == 61.0
    assert gpu["fan"] == 30
    assert gpu["power"] == 220.0
    assert client.get("/panel").get_json()["gpu_n"] == 1


def test_panel_gpu_entry_is_an_int_fan_speed(client, monkeypatch):
    """``fan`` is an int in the contract, not a float like /stats' fanSpeed."""
    monkeypatch.setattr(collectors.GPUtil, "getGPUs", lambda: [_FakeGPU()])
    monkeypatch.setattr(
        collectors, "get_gpu_fan_and_power", lambda: {0: {"fan_speed": 30.6, "power_draw": 1.0}}
    )
    monkeypatch.setattr(collectors, "get_gpu_processes", lambda limit=5, uuid_to_index=None: [])
    _seed_cache()

    gpu = client.get("/panel").get_json()["gpu"][0]

    assert gpu["fan"] == 31
    assert isinstance(gpu["fan"], int)


def test_panel_load_is_the_loadavg_triple_not_a_percentage(client, monkeypatch):
    """cpu.load is os.getloadavg(), distinct from /stats' percentage under the same key name."""
    monkeypatch.setattr(collectors, "get_load_average", lambda: [1.5, 2.25, 3.0])
    _seed_cache()

    payload = client.get("/panel").get_json()

    assert payload["cpu"]["load"] == [1.5, 2.25, 3.0]
    assert payload["cpu"]["pct"] == 12.5  # the percentage lives under "pct", not "load"


def test_panel_load_average_degrades_to_zeros_on_oserror(client, monkeypatch):
    """A platform without os.getloadavg() (Windows) must not break the sample."""

    def _raise():
        raise OSError("getloadavg() not supported")

    monkeypatch.setattr(collectors, "get_load_average", _raise)
    _seed_cache()

    payload = client.get("/panel").get_json()

    assert payload["cpu"]["load"] == [0.0, 0.0, 0.0]
    assert "loadavg" in payload["err"]


def test_panel_cpu_frequency_degrades_to_zero(client, monkeypatch):
    """cpu.mhz degrades to 0 when psutil.cpu_freq() is unavailable."""
    monkeypatch.setattr(collectors, "get_cpu_frequency_mhz", lambda: 0)
    _seed_cache()

    assert client.get("/panel").get_json()["cpu"]["mhz"] == 0


def test_panel_reports_err_tags_for_failing_collectors(client, monkeypatch):
    """A collector that raises during the sampling pass is named in err."""
    monkeypatch.setattr(
        collectors, "get_swap", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    _seed_cache()

    payload = client.get("/panel").get_json()

    assert payload["err"] == ["swap"]
    assert payload["swap"] == {"used": 0, "total": 0, "pct": 0.0}


def test_panel_never_exposes_process_lists_or_ollama_data(client, monkeypatch):
    """The frozen contract explicitly forbids process lists, Ollama data and cmdlines."""
    monkeypatch.setattr(
        collectors, "get_top_processes_by_cpu", lambda limit=5: [{"pid": 1, "cmdline": "x"}]
    )
    monkeypatch.setattr(
        collectors, "get_ollama_process", lambda: {"models": [{"name": "llama3"}]}
    )
    _seed_cache()

    payload = client.get("/panel")
    body = payload.get_data(as_text=True)

    assert "top_cpu" not in payload.get_json()
    assert "ollama_processes" not in payload.get_json()
    assert "cmdline" not in body
    assert "ollama" not in body.lower()


class TestTempsAndFansTruncation:
    """``temps``/``fans`` truncation: caps from env, always-present counts."""

    def _seed_five_temps_and_fans(self, monkeypatch):
        temps = [{"n": f"chip/core{i}", "c": float(i)} for i in range(5)]
        fans = [{"n": f"chip/fan{i}", "rpm": i * 100} for i in range(5)]
        monkeypatch.setattr(collectors, "get_temperatures", lambda: temps)
        monkeypatch.setattr(collectors, "get_fans", lambda: fans)
        return temps, fans

    def test_uncapped_by_default_returns_every_entry(self, client, monkeypatch):
        """No env var set means no cap: the true count and the list length match."""
        temps, fans = self._seed_five_temps_and_fans(monkeypatch)
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == temps
        assert payload["temps_n"] == 5
        assert payload["fans"] == fans
        assert payload["fans_n"] == 5

    def test_caps_truncate_but_report_the_true_count(self, client, monkeypatch):
        """temps_n/fans_n stay the TRUE count even when the list is capped."""
        temps, fans = self._seed_five_temps_and_fans(monkeypatch)
        monkeypatch.setenv("SYS_STATS_PANEL_MAX_TEMPS", "2")
        monkeypatch.setenv("SYS_STATS_PANEL_MAX_FANS", "3")
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == temps[:2]
        assert payload["temps_n"] == 5
        assert len(payload["temps"]) < payload["temps_n"]
        assert payload["fans"] == fans[:3]
        assert payload["fans_n"] == 5
        assert len(payload["fans"]) < payload["fans_n"]

    def test_caps_slice_after_the_incoming_sort_never_before(self, client, monkeypatch):
        """Truncation must keep the first N of the already-sorted list, deterministically."""
        temps = [{"n": "a", "c": 1.0}, {"n": "b", "c": 2.0}, {"n": "c", "c": 3.0}]
        monkeypatch.setattr(collectors, "get_temperatures", lambda: temps)
        monkeypatch.setenv("SYS_STATS_PANEL_MAX_TEMPS", "1")
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == [{"n": "a", "c": 1.0}]

    @pytest.mark.parametrize("raw", ["0", "-1", "not-a-number"])
    def test_bad_cap_values_mean_no_cap(self, client, monkeypatch, raw):
        """Zero, negative or non-numeric caps must not truncate anything."""
        temps, _fans = self._seed_five_temps_and_fans(monkeypatch)
        monkeypatch.setenv("SYS_STATS_PANEL_MAX_TEMPS", raw)
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == temps
        assert payload["temps_n"] == 5

    def test_gpu_list_is_capped_by_its_own_env_var(self, client, monkeypatch):
        """SYS_STATS_PANEL_MAX_GPUS caps the gpu list independently of temps/fans."""
        monkeypatch.setattr(
            collectors.GPUtil, "getGPUs", lambda: [_FakeGPU(gpu_id=0), _FakeGPU(gpu_id=1)]
        )
        monkeypatch.setattr(collectors, "get_gpu_fan_and_power", lambda: {})
        monkeypatch.setattr(collectors, "get_gpu_processes", lambda limit=5, uuid_to_index=None: [])
        monkeypatch.setenv("SYS_STATS_PANEL_MAX_GPUS", "1")
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["gpu_n"] == 2
        assert len(payload["gpu"]) == 1
        assert payload["gpu"][0]["i"] == 0


class TestTempsUnion:
    """``temps`` is the union of the hwmon and IPMI sources, concatenated then sorted."""

    def test_both_sources_populated_are_interleaved_by_the_sort(self, client, monkeypatch):
        """hwmon and IPMI entries land in one list, ordered together by ``n``."""
        monkeypatch.setattr(
            collectors, "get_temperatures", lambda: [{"n": "k10temp/Tctl", "c": 45.0}]
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_temperatures",
            lambda: [{"n": "ipmi/CPU1 Temp", "c": 38.0}, {"n": "ipmi/CPU2 Temp", "c": 41.5}],
        )
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == [
            {"n": "ipmi/CPU1 Temp", "c": 38.0},
            {"n": "ipmi/CPU2 Temp", "c": 41.5},
            {"n": "k10temp/Tctl", "c": 45.0},
        ]
        assert payload["temps_n"] == 3

    def test_only_hwmon_populated(self, client, monkeypatch):
        """A desktop with hwmon sensors and no BMC reports hwmon temps alone."""
        monkeypatch.setattr(
            collectors, "get_temperatures", lambda: [{"n": "k10temp/Tctl", "c": 45.0}]
        )
        monkeypatch.setattr(collectors, "get_ipmi_temperatures", lambda: [])
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == [{"n": "k10temp/Tctl", "c": 45.0}]
        assert payload["temps_n"] == 1

    def test_only_ipmi_populated(self, client, monkeypatch):
        """A Proxmox hypervisor with zero hwmon sensors reports IPMI temps alone."""
        monkeypatch.setattr(collectors, "get_temperatures", lambda: [])
        monkeypatch.setattr(
            collectors, "get_ipmi_temperatures", lambda: [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        )
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        assert payload["temps_n"] == 1

    def test_neither_source_populated(self, client, monkeypatch):
        """No hwmon and no IPMI temps is an empty list and a zero count, not an error."""
        monkeypatch.setattr(collectors, "get_temperatures", lambda: [])
        monkeypatch.setattr(collectors, "get_ipmi_temperatures", lambda: [])
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == []
        assert payload["temps_n"] == 0

    def test_one_source_raising_does_not_lose_the_other(self, client, monkeypatch):
        """A raising collector degrades only its own contribution to the union."""
        monkeypatch.setattr(
            collectors, "get_temperatures", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        monkeypatch.setattr(
            collectors, "get_ipmi_temperatures", lambda: [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        )
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        assert payload["temps_n"] == 1
        assert "temps_hwmon" in payload["err"]

    def test_cap_applies_to_the_union_with_temps_n_reporting_the_true_total(
        self, client, monkeypatch
    ):
        """SYS_STATS_PANEL_MAX_TEMPS caps the sorted union; temps_n stays the pre-cap total."""
        monkeypatch.setattr(
            collectors, "get_temperatures", lambda: [{"n": "k10temp/Tctl", "c": 45.0}]
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_temperatures",
            lambda: [{"n": "ipmi/CPU1 Temp", "c": 38.0}, {"n": "ipmi/CPU2 Temp", "c": 41.5}],
        )
        monkeypatch.setenv("SYS_STATS_PANEL_MAX_TEMPS", "2")
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["temps"] == [
            {"n": "ipmi/CPU1 Temp", "c": 38.0},
            {"n": "ipmi/CPU2 Temp", "c": 41.5},
        ]
        assert payload["temps_n"] == 3
        assert len(payload["temps"]) < payload["temps_n"]


class TestFansUnion:
    """``fans`` is the union of the hwmon and IPMI sources, concatenated then sorted."""

    def test_both_sources_populated_are_interleaved_by_the_sort(self, client, monkeypatch):
        """hwmon and IPMI entries land in one list, ordered together by ``n``."""
        monkeypatch.setattr(
            collectors, "get_fans", lambda: [{"n": "nct6775/fan2", "rpm": 900}]
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_fans",
            lambda: [{"n": "ipmi/FAN1", "rpm": 7100}, {"n": "ipmi/FAN2", "rpm": 6900}],
        )
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["fans"] == [
            {"n": "ipmi/FAN1", "rpm": 7100},
            {"n": "ipmi/FAN2", "rpm": 6900},
            {"n": "nct6775/fan2", "rpm": 900},
        ]
        assert payload["fans_n"] == 3

    def test_only_hwmon_populated(self, client, monkeypatch):
        """A desktop with a Super I/O chip and no BMC reports hwmon fans alone."""
        monkeypatch.setattr(
            collectors, "get_fans", lambda: [{"n": "nct6775/fan1", "rpm": 1200}]
        )
        monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [])
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["fans"] == [{"n": "nct6775/fan1", "rpm": 1200}]
        assert payload["fans_n"] == 1

    def test_only_ipmi_populated(self, client, monkeypatch):
        """A Proxmox hypervisor with zero hwmon fans reports IPMI fans alone."""
        monkeypatch.setattr(collectors, "get_fans", lambda: [])
        monkeypatch.setattr(
            collectors, "get_ipmi_fans", lambda: [{"n": "ipmi/FAN1", "rpm": 7100}]
        )
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["fans"] == [{"n": "ipmi/FAN1", "rpm": 7100}]
        assert payload["fans_n"] == 1

    def test_neither_source_populated(self, client, monkeypatch):
        """No hwmon and no IPMI fans is an empty list and a zero count, not an error."""
        monkeypatch.setattr(collectors, "get_fans", lambda: [])
        monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [])
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["fans"] == []
        assert payload["fans_n"] == 0

    def test_one_source_raising_does_not_lose_the_other(self, client, monkeypatch):
        """A raising collector degrades only its own contribution to the union."""
        monkeypatch.setattr(
            collectors, "get_fans", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        monkeypatch.setattr(
            collectors, "get_ipmi_fans", lambda: [{"n": "ipmi/FAN1", "rpm": 7100}]
        )
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["fans"] == [{"n": "ipmi/FAN1", "rpm": 7100}]
        assert payload["fans_n"] == 1
        assert "fans_hwmon" in payload["err"]

    def test_cap_applies_to_the_union_with_fans_n_reporting_the_true_total(
        self, client, monkeypatch
    ):
        """SYS_STATS_PANEL_MAX_FANS caps the sorted union; fans_n stays the pre-cap total."""
        monkeypatch.setattr(
            collectors, "get_fans", lambda: [{"n": "nct6775/fan1", "rpm": 1200}]
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_fans",
            lambda: [{"n": "ipmi/FAN1", "rpm": 7100}, {"n": "ipmi/FAN2", "rpm": 6900}],
        )
        monkeypatch.setenv("SYS_STATS_PANEL_MAX_FANS", "2")
        _seed_cache()

        payload = client.get("/panel").get_json()

        assert payload["fans"] == [
            {"n": "ipmi/FAN1", "rpm": 7100},
            {"n": "ipmi/FAN2", "rpm": 6900},
        ]
        assert payload["fans_n"] == 3
        assert len(payload["fans"]) < payload["fans_n"]


class TestAge:
    """``age`` is computed at response time from the monotonic stamp, never from ``ts``."""

    def test_age_grows_as_the_monotonic_clock_advances(self, client, monkeypatch):
        """A snapshot aged by advancing the monotonic clock must report a growing age."""
        base_monotonic = time.monotonic()
        fake_now = [base_monotonic]
        monkeypatch.setattr(server.time, "monotonic", lambda: fake_now[0])

        # The patch is already active, so this _seed_cache() stores the
        # sample's monotonic stamp as base_monotonic. Advancing fake_now
        # afterwards only moves the clock the /panel *route* reads at
        # response time, not the one the sample itself was stamped with.
        _seed_cache()
        fake_now[0] = base_monotonic + 7.9

        payload = client.get("/panel").get_json()

        assert payload["age"] == 7

    def test_age_is_clamped_at_zero_never_negative(self, client, monkeypatch):
        """A monotonic reading at or before the sample's own stamp clamps to 0."""
        stats, extras, wall_ts, monotonic_ts = sampler.get_panel_snapshot()
        monkeypatch.setattr(server.time, "monotonic", lambda: monotonic_ts - 5)

        payload = client.get("/panel").get_json()

        assert payload["age"] == 0

    def test_backwards_wall_clock_jump_does_not_corrupt_age(self, client, monkeypatch):
        """age is computed from the monotonic stamp; a wall-clock step must not affect it.

        Simulates an NTP step: the sample lands right as the wall clock
        jumps an hour into the past (``ts`` legitimately follows the jump,
        it is a plain ``time.time()`` reading for a human running curl) but
        ``age`` must still come out sane, computed purely from the
        monotonic stamp captured at that same instant, never from ``ts``.
        """
        base_monotonic = time.monotonic()
        monkeypatch.setattr(sampler.time, "time", lambda: 1_000_000.0)  # far in the past
        monkeypatch.setattr(sampler.time, "monotonic", lambda: base_monotonic)
        _seed_cache()

        monkeypatch.setattr(server.time, "monotonic", lambda: base_monotonic + 4)

        payload = client.get("/panel").get_json()

        assert payload["ts"] == 1_000_000
        assert payload["age"] == 4


def test_panel_ts_is_the_sample_wall_clock_not_the_request_time(client, monkeypatch):
    """ts is recorded when the SAMPLE was taken, not when the request arrived."""
    _stats, _extras, wall_ts, _monotonic_ts = sampler.get_panel_snapshot()

    payload = client.get("/panel").get_json()

    assert payload["ts"] == int(wall_ts)


def test_panel_response_is_a_deep_copy_of_the_cache(client):
    """Mutating a /panel response must never poison the sampler's cache."""
    first = client.get("/panel").get_json()
    first["swap"]["used"] = 999999999
    first["temps"].append({"n": "intruder", "c": 1.0})

    second = client.get("/panel").get_json()

    assert second["swap"]["used"] != 999999999
    assert second["temps"] == []


def test_stats_route_is_untouched_by_panel_extras(client):
    """/stats' key set must stay exactly what it always was.

    Proves the panel-only extras collected alongside it never leak into the
    /stats response: no "temps", "fans", "swap", "per_core", "load", "mhz"
    or "err" key appears there.
    """
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
