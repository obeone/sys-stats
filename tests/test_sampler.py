"""Tests for the background sampler of :mod:`sys_stats.sampler`.

The sampler owns the ``psutil.cpu_percent`` baseline and runs its loop on a
daemon thread, so most of what matters here is timing-sensitive behaviour:
priming the baseline before the first real sample, and surviving a collector
that raises. Every test drives ``sampler._run`` directly on a short-lived
thread it starts and stops itself, rather than the real
``SYS_STATS_SAMPLE_INTERVAL``-based ``start()``, so the suite stays fast and
deterministic instead of depending on wall-clock timing.
"""

import threading
import time

import pytest

from sys_stats import collectors, sampler


@pytest.fixture(autouse=True)
def _clean_sampler():
    """Stop any sampler thread and clear the cache before and after each test.

    A thread left running from one test would keep sampling on its own
    schedule after this test's monkeypatches revert, touching the real
    machine and corrupting the next test's cache.
    """
    sampler._reset_for_tests()
    yield
    sampler._reset_for_tests()


@pytest.fixture(autouse=True)
def _stub_panel_extra_collectors(monkeypatch):
    """Stub the ``/panel``-only collectors for every test in this file.

    ``_sample_once`` now calls ``_collect_panel_extras`` in the same pass as
    ``collect_stats`` (see the "one sampling pass" guarantee /panel relies
    on), so any test here that drives ``sampler._run`` or calls
    ``_sample_once`` directly would otherwise reach the real ``psutil``
    sensors/frequency/load-average calls through it. Tests that care about
    the extras' own behaviour override these stubs locally.
    """
    monkeypatch.setattr(collectors, "get_temperatures", lambda: [])
    monkeypatch.setattr(collectors, "get_ipmi_temperatures", lambda: [])
    monkeypatch.setattr(collectors, "get_fans", lambda: [])
    monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [])
    monkeypatch.setattr(collectors, "get_swap", lambda: {"used": 0, "total": 0, "pct": 0.0})
    monkeypatch.setattr(collectors, "get_per_core_cpu", lambda: [])
    monkeypatch.setattr(collectors, "get_load_average", lambda: [0.0, 0.0, 0.0])
    monkeypatch.setattr(collectors, "get_cpu_frequency_mhz", lambda: 0)


def test_get_snapshot_is_none_before_any_sample():
    """A fresh sampler with no thread running has nothing cached yet."""
    stats, wall_ts, monotonic_ts = sampler.get_snapshot()

    assert (stats, wall_ts, monotonic_ts) == (None, None, None)


def test_run_primes_the_cpu_percent_baseline_before_the_first_sample(monkeypatch):
    """The loop calls ``psutil.cpu_percent`` once, before ``collect_stats`` runs at all.

    ``psutil.cpu_percent(interval=None)`` only means something relative to
    the previous call. The very first call in a process has no previous call
    to compare against, so the loop makes (and discards) that first call
    before entering the sample/sleep cycle, and every real sample afterwards
    is meaningful.
    """
    calls: list[str] = []

    def _cpu_percent(interval=None, percpu=False):
        # The percpu priming call is recorded under its own label so this
        # test can still assert the plain-variant call happened exactly
        # once, undisturbed by the second priming call added alongside it
        # (see test_run_primes_the_percpu_cpu_percent_baseline_before_the_first_sample).
        calls.append("prime_percpu" if percpu else "prime")
        return [] if percpu else 0.0

    monkeypatch.setattr(sampler.psutil, "cpu_percent", _cpu_percent)
    monkeypatch.setattr(
        sampler.collectors,
        "collect_stats",
        lambda limit=5: calls.append("collect") or {"top_cpu": [], "top_memory": [], "top_gpu_processes": []},
    )

    thread = threading.Thread(target=sampler._run, args=(0.01, 5), daemon=True)
    thread.start()
    try:
        assert sampler._first_snapshot_event.wait(timeout=2), "no sample landed in time"
    finally:
        sampler._stop_event.set()
        thread.join(timeout=2)

    # Exactly one priming call, and it happened before the first collection.
    assert calls[0] == "prime"
    assert calls.count("prime") == 1
    assert "collect" in calls[1:]


def test_run_primes_the_percpu_cpu_percent_baseline_before_the_first_sample(monkeypatch):
    """The loop primes the percpu variant too, before ``collect_stats`` runs.

    psutil keeps a SEPARATE internal baseline for
    ``cpu_percent(percpu=True)`` from the one it keeps for the plain call:
    priming one does nothing for the other. If the loop only primed the
    plain variant, ``collectors.get_per_core_cpu()``'s first real reading
    would be a meaningless near-zero regardless of actual load. Both
    baselines must be primed, in the same crash-guarded step, before the
    first real sample.

    Also asserts the THIRD baseline alongside these two:
    ``collectors.get_top_processes_by_cpu()`` reads ``cpu_percent`` off
    ``psutil.process_iter(...)``, which keeps its baseline per-Process
    object, entirely separate from both module-level baselines above.
    Priming the plain and percpu variants does nothing for it; without its
    own priming sweep, every process reports ``cpu_percent: 0.0`` on the
    first real sample and ``top_cpu`` sorts a meaningless all-zero column.
    """
    calls: list[tuple[str, bool | None]] = []

    def _cpu_percent(interval=None, percpu=False):
        calls.append(("prime", percpu))
        return [] if percpu else 0.0

    class _FakeProcess:
        def __init__(self, pid):
            self.info = {"cpu_percent": 0.0}
            calls.append((f"prime_proc_{pid}", None))

    def _process_iter(attrs=None):
        calls.append(("process_iter", None))
        return iter([_FakeProcess(1), _FakeProcess(2)])

    monkeypatch.setattr(sampler.psutil, "cpu_percent", _cpu_percent)
    monkeypatch.setattr(sampler.psutil, "process_iter", _process_iter)
    monkeypatch.setattr(
        sampler.collectors,
        "collect_stats",
        lambda limit=5: calls.append(("collect", None))
        or {"top_cpu": [], "top_memory": [], "top_gpu_processes": []},
    )

    thread = threading.Thread(target=sampler._run, args=(0.01, 5), daemon=True)
    thread.start()
    try:
        assert sampler._first_snapshot_event.wait(timeout=2), "no sample landed in time"
    finally:
        sampler._stop_event.set()
        thread.join(timeout=2)

    priming_calls = [c for c in calls if c[0] == "prime"]
    # Both cpu_percent variants primed, plain first then percpu, each exactly once.
    assert priming_calls == [("prime", False), ("prime", True)]
    # The per-process baseline sweep runs once too, strictly after the two
    # module-level baselines and strictly before the first collect_stats()
    # call -- proving all three baselines are established together, in
    # order, before any real sample is taken.
    assert [c[0] for c in calls] == [
        "prime",
        "prime",
        "process_iter",
        "prime_proc_1",
        "prime_proc_2",
        "collect",
    ]


def test_run_primes_the_per_process_cpu_percent_baseline_tolerating_vanished_processes(
    monkeypatch,
):
    """Priming the per-process baseline must survive processes that vanish mid-sweep.

    ``collectors.get_top_processes_by_cpu()`` already guards its own
    ``process_iter`` sweep against ``NoSuchProcess``, ``AccessDenied`` and
    ``ZombieProcess`` -- the CLAUDE.md-documented reality that unreadable
    attributes come back as ``None`` rather than raising, and that a process
    can exit between being listed and being read. The priming sweep added
    alongside the two system-wide baselines must tolerate the same
    conditions instead of taking the whole iteration (and thus the sampler
    thread) down with it.
    """
    calls: list[str] = []

    class _VanishingProcess:
        def __init__(self, exc):
            self._exc = exc

        @property
        def info(self):
            raise self._exc

    class _HealthyProcess:
        def __init__(self, pid):
            self.info = {"cpu_percent": 0.0}
            calls.append(f"prime_proc_{pid}")

    def _process_iter(attrs=None):
        calls.append("process_iter")
        return iter(
            [
                _VanishingProcess(sampler.psutil.NoSuchProcess(1)),
                _HealthyProcess(2),
                _VanishingProcess(sampler.psutil.AccessDenied(3)),
                _VanishingProcess(sampler.psutil.ZombieProcess(4)),
                _HealthyProcess(5),
            ]
        )

    monkeypatch.setattr(
        sampler.psutil,
        "cpu_percent",
        lambda interval=None, percpu=False: [] if percpu else 0.0,
    )
    monkeypatch.setattr(sampler.psutil, "process_iter", _process_iter)
    monkeypatch.setattr(
        sampler.collectors,
        "collect_stats",
        lambda limit=5: calls.append("collect")
        or {"top_cpu": [], "top_memory": [], "top_gpu_processes": []},
    )

    thread = threading.Thread(target=sampler._run, args=(0.01, 5), daemon=True)
    thread.start()
    try:
        assert sampler._first_snapshot_event.wait(timeout=2), "no sample landed in time"
    finally:
        sampler._stop_event.set()
        thread.join(timeout=2)

    # The sweep ran, the healthy processes were still primed despite the
    # vanishing ones interleaved among them, and a real sample landed
    # afterwards -- none of the three exception types killed the thread.
    assert not thread.is_alive()
    assert calls == ["process_iter", "prime_proc_2", "prime_proc_5", "collect"]


def test_run_primes_immediately_and_first_sample_lands_one_interval_later(monkeypatch):
    """A cold start must cost ONE interval, not two.

    Regression test for the loop that ran its interruptible sleep *before*
    the priming call, giving a wait-prime-wait-sample sequence: the first
    snapshot landed at ``2 * interval``, so at
    ``SYS_STATS_SAMPLE_INTERVAL=12`` the ``/panel`` route answered 503 for
    24 seconds and ``/stats`` blocked for just as long. The interval between
    priming and the first sample is load-bearing -- it is what gives
    ``psutil.cpu_percent`` a baseline to measure against -- so what is
    asserted here is that priming happens immediately and exactly one
    interval separates it from the first collection.

    Deliberately runs at a NON-DEFAULT interval: the default 2s made the
    wrong 4s cold start look unremarkable, which is why the rest of the
    suite never caught this.
    """
    interval = 0.4  # non-default, and long enough to tell one from two
    started = time.monotonic()
    prime_at: list[float] = []
    collect_at: list[float] = []

    def _cpu_percent(interval=None, percpu=False):
        if not percpu:
            prime_at.append(time.monotonic() - started)
        return [] if percpu else 0.0

    def _collect_stats(limit=5):
        collect_at.append(time.monotonic() - started)
        return {"top_cpu": [], "top_memory": [], "top_gpu_processes": []}

    monkeypatch.setattr(sampler.psutil, "cpu_percent", _cpu_percent)
    monkeypatch.setattr(sampler.collectors, "collect_stats", _collect_stats)

    thread = threading.Thread(target=sampler._run, args=(interval, 5), daemon=True)
    thread.start()
    try:
        assert sampler._first_snapshot_event.wait(timeout=5), "no sample landed in time"
    finally:
        sampler._stop_event.set()
        thread.join(timeout=5)

    # Primed on entry, without burning an interval first.
    assert prime_at[0] < interval / 2
    # First real sample one interval later -- never two. The upper bound sits
    # well below 2 * interval so a slow CI box cannot make a two-interval
    # cold start pass as a one-interval one.
    assert interval * 0.8 <= collect_at[0] < interval * 1.6


def test_run_survives_a_raising_collector_and_keeps_sampling(monkeypatch, caplog):
    """A collector that raises must not kill the sampler thread.

    The next iteration has to run regardless, or the cache would freeze on
    whatever was last cached (``None`` on a cold start) forever.
    """
    monkeypatch.setattr(
        sampler.psutil,
        "cpu_percent",
        lambda interval=None, percpu=False: [] if percpu else 0.0,
    )

    outcomes = iter(
        [RuntimeError("boom"), {"top_cpu": [1], "top_memory": [], "top_gpu_processes": []}]
    )

    def _collect_stats(limit=5):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(sampler.collectors, "collect_stats", _collect_stats)

    thread = threading.Thread(target=sampler._run, args=(0.01, 5), daemon=True)
    with caplog.at_level("ERROR"):
        thread.start()
        try:
            assert sampler._first_snapshot_event.wait(timeout=2), "no sample landed after the failure"
        finally:
            sampler._stop_event.set()
            thread.join(timeout=2)

    assert not thread.is_alive()
    stats, _wall_ts, _monotonic_ts = sampler.get_snapshot()
    assert stats == {"top_cpu": [1], "top_memory": [], "top_gpu_processes": []}
    assert any("Sampler iteration failed" in record.message for record in caplog.records)


def test_run_survives_a_raising_priming_call_and_keeps_sampling(monkeypatch, caplog):
    """A ``psutil.cpu_percent`` priming failure must not kill the thread either.

    The priming call used to run before the loop's try/except, so a failure
    there killed the thread silently with no retry and the cache never
    filled. It must be logged and retried on the next interval, exactly like
    any other iteration failure.
    """
    # The first priming attempt fails on the plain-variant call before ever
    # reaching the percpu one. The retried attempt succeeds on both calls:
    # plain, then percpu, in the same guarded step.
    outcomes = iter([RuntimeError("boom"), 0.0, 0.0])

    def _cpu_percent(interval=None, percpu=False):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return [] if percpu else outcome

    monkeypatch.setattr(sampler.psutil, "cpu_percent", _cpu_percent)
    monkeypatch.setattr(
        sampler.collectors,
        "collect_stats",
        lambda limit=5: {"top_cpu": [], "top_memory": [], "top_gpu_processes": []},
    )

    thread = threading.Thread(target=sampler._run, args=(0.01, 5), daemon=True)
    with caplog.at_level("ERROR"):
        thread.start()
        try:
            assert sampler._first_snapshot_event.wait(
                timeout=2
            ), "no sample landed after the priming failure"
        finally:
            sampler._stop_event.set()
            thread.join(timeout=2)

    assert not thread.is_alive()
    stats, _wall_ts, _monotonic_ts = sampler.get_snapshot()
    assert stats == {"top_cpu": [], "top_memory": [], "top_gpu_processes": []}
    assert any("Sampler iteration failed" in record.message for record in caplog.records)


def test_wait_for_first_snapshot_returns_immediately_once_a_sample_exists():
    """A snapshot that already landed is returned without waiting out the timeout."""
    sampler._store_snapshot({"marker": "already-here"})

    start = time.monotonic()
    stats, _wall_ts, _monotonic_ts = sampler.wait_for_first_snapshot(timeout=5)
    elapsed = time.monotonic() - start

    assert stats == {"marker": "already-here"}
    assert elapsed < 1  # did not block for anywhere near the full timeout


def test_wait_for_first_snapshot_times_out_when_nothing_lands():
    """With no writer at all, the wait gives up after ``timeout`` seconds."""
    start = time.monotonic()
    stats, wall_ts, monotonic_ts = sampler.wait_for_first_snapshot(timeout=0.05)
    elapsed = time.monotonic() - start

    assert (stats, wall_ts, monotonic_ts) == (None, None, None)
    assert elapsed >= 0.05


@pytest.mark.parametrize("raw", [None, "0", "-1", "not-a-number"])
def test_get_sample_interval_falls_back_to_the_default_on_bad_input(monkeypatch, raw):
    """Unset, non-numeric or non-positive input all fall back to the default."""
    if raw is None:
        monkeypatch.delenv("SYS_STATS_SAMPLE_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("SYS_STATS_SAMPLE_INTERVAL", raw)

    assert sampler._get_sample_interval() == sampler.DEFAULT_SAMPLE_INTERVAL


def test_get_sample_interval_honours_a_valid_override(monkeypatch):
    """A well-formed positive float is used as-is."""
    monkeypatch.setenv("SYS_STATS_SAMPLE_INTERVAL", "7.5")

    assert sampler._get_sample_interval() == 7.5


@pytest.mark.parametrize("raw", [None, "0", "-1", "not-a-number"])
def test_get_ipmi_interval_falls_back_to_the_default_on_bad_input(monkeypatch, raw):
    """Unset, non-numeric or non-positive input all fall back to the default.

    Same validation shape as SYS_STATS_SAMPLE_INTERVAL (see
    test_get_sample_interval_falls_back_to_the_default_on_bad_input above).
    """
    if raw is None:
        monkeypatch.delenv("SYS_STATS_IPMI_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("SYS_STATS_IPMI_INTERVAL", raw)

    assert sampler._get_ipmi_interval() == sampler.DEFAULT_IPMI_INTERVAL


def test_get_ipmi_interval_honours_a_valid_override(monkeypatch):
    """A well-formed positive float is used as-is."""
    monkeypatch.setenv("SYS_STATS_IPMI_INTERVAL", "60")

    assert sampler._get_ipmi_interval() == 60.0


@pytest.mark.parametrize("raw", [None, "0", "-5", "banana"])
def test_get_top_processes_cap_falls_back_to_the_default_on_bad_input(monkeypatch, raw):
    """Unset, non-numeric or non-positive input all fall back to the default."""
    if raw is None:
        monkeypatch.delenv("SYS_STATS_TOP_PROCESSES_MAX", raising=False)
    else:
        monkeypatch.setenv("SYS_STATS_TOP_PROCESSES_MAX", raw)

    assert sampler._get_top_processes_cap() == sampler.DEFAULT_TOP_PROCESSES_MAX


def test_get_top_processes_cap_honours_a_valid_override(monkeypatch):
    """A well-formed positive integer is used as-is."""
    monkeypatch.setenv("SYS_STATS_TOP_PROCESSES_MAX", "10")

    assert sampler._get_top_processes_cap() == 10


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_get_dcgm_url_is_none_when_unset_or_blank(monkeypatch, raw):
    """Unset or blank both mean "current behaviour unchanged" (see _get_dcgm_url)."""
    if raw is None:
        monkeypatch.delenv("SYS_STATS_DCGM_URL", raising=False)
    else:
        monkeypatch.setenv("SYS_STATS_DCGM_URL", raw)

    assert sampler._get_dcgm_url() is None


def test_get_dcgm_url_strips_surrounding_whitespace(monkeypatch):
    """A configured URL is returned with surrounding whitespace stripped."""
    monkeypatch.setenv("SYS_STATS_DCGM_URL", "  http://10.50.0.106:30940/metrics  ")

    assert sampler._get_dcgm_url() == "http://10.50.0.106:30940/metrics"


def test_start_is_idempotent(monkeypatch):
    """Calling ``start()`` twice must not spawn a second thread.

    Two live loops would each call ``psutil.cpu_percent`` on their own
    schedule, corrupting each other's baseline exactly like the two-caller
    problem the sampler exists to prevent.
    """
    monkeypatch.setenv("SYS_STATS_SAMPLE_INTERVAL", "0.01")
    monkeypatch.setattr(
        sampler.psutil,
        "cpu_percent",
        lambda interval=None, percpu=False: [] if percpu else 0.0,
    )
    monkeypatch.setattr(
        collectors,
        "collect_stats",
        lambda limit=5: {"top_cpu": [], "top_memory": [], "top_gpu_processes": []},
    )

    sampler.start()
    first_thread = sampler._thread
    sampler.start()

    assert sampler._thread is first_thread


class TestCollectPanelExtras:
    """Tests for ``_collect_panel_extras``, the ``/panel``-only side of a sampling pass."""

    def test_collects_every_field_with_no_errors(self, monkeypatch):
        """The happy path returns every field and an empty ``err`` list."""
        monkeypatch.setattr(collectors, "get_temperatures", lambda: [{"n": "cpu", "c": 55.0}])
        monkeypatch.setattr(collectors, "get_fans", lambda: [{"n": "fan1", "rpm": 1200}])
        monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [])
        monkeypatch.setattr(collectors, "get_swap", lambda: {"used": 1, "total": 2, "pct": 50.0})
        monkeypatch.setattr(collectors, "get_per_core_cpu", lambda: [10.0, 20.0])
        monkeypatch.setattr(collectors, "get_load_average", lambda: [0.1, 0.2, 0.3])
        monkeypatch.setattr(collectors, "get_cpu_frequency_mhz", lambda: 3200)

        extras = sampler._collect_panel_extras()

        assert extras == {
            "temps": [{"n": "cpu", "c": 55.0}],
            "fans": [{"n": "fan1", "rpm": 1200}],
            "swap": {"used": 1, "total": 2, "pct": 50.0},
            "per_core": [10.0, 20.0],
            "load": [0.1, 0.2, 0.3],
            "mhz": 3200,
            "err": [],
        }

    @pytest.mark.parametrize(
        "collector_name, default",
        [
            ("get_temperatures", []),
            ("get_ipmi_temperatures", []),
            ("get_fans", []),
            ("get_ipmi_fans", []),
            ("get_swap", {"used": 0, "total": 0, "pct": 0.0}),
            ("get_per_core_cpu", []),
            ("get_load_average", [0.0, 0.0, 0.0]),
            ("get_cpu_frequency_mhz", 0),
        ],
    )
    def test_one_failing_collector_degrades_only_its_own_field(
        self, monkeypatch, collector_name, default
    ):
        """A single raising collector must not take the whole extras dict down.

        Every other field still collects normally; only the failing one
        falls back to its safe default, and its short tag lands in ``err``.
        """
        monkeypatch.setattr(
            collectors,
            collector_name,
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        extras = sampler._collect_panel_extras()

        field = {
            "get_temperatures": "temps",
            "get_ipmi_temperatures": "temps",
            "get_fans": "fans",
            "get_ipmi_fans": "fans",
            "get_swap": "swap",
            "get_per_core_cpu": "per_core",
            "get_load_average": "load",
            "get_cpu_frequency_mhz": "mhz",
        }[collector_name]

        assert extras[field] == default
        assert len(extras["err"]) == 1

    def test_temps_are_the_union_of_hwmon_and_ipmi_sorted_by_name(self, monkeypatch):
        """The two temperature sources are concatenated, then sorted together by ``n``.

        Their names cannot collide in practice (hwmon: ``chip/label``, IPMI:
        ``ipmi/``-prefixed sensor names like ``ipmi/CPU1 Temp``), so a plain
        concatenate-then-sort is the whole contract -- no deduplication, no
        priority rule.
        """
        monkeypatch.setattr(
            collectors, "get_temperatures", lambda: [{"n": "k10temp/Tctl", "c": 45.0}]
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_temperatures",
            lambda: [{"n": "ipmi/CPU1 Temp", "c": 38.0}, {"n": "ipmi/CPU2 Temp", "c": 41.5}],
        )

        extras = sampler._collect_panel_extras()

        assert extras["temps"] == [
            {"n": "ipmi/CPU1 Temp", "c": 38.0},
            {"n": "ipmi/CPU2 Temp", "c": 41.5},
            {"n": "k10temp/Tctl", "c": 45.0},
        ]

    def test_temps_are_empty_when_neither_source_reports_anything(self, monkeypatch):
        """No hwmon temps and no IPMI temps is a legitimate empty union, not an error."""
        monkeypatch.setattr(collectors, "get_temperatures", lambda: [])
        monkeypatch.setattr(collectors, "get_ipmi_temperatures", lambda: [])

        extras = sampler._collect_panel_extras()

        assert extras["temps"] == []
        assert extras["err"] == []

    def test_ipmi_temps_survive_a_raising_hwmon_collector(self, monkeypatch):
        """hwmon raising must not cost the IPMI data, only its own field's slice."""
        monkeypatch.setattr(
            collectors, "get_temperatures", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        monkeypatch.setattr(
            collectors, "get_ipmi_temperatures", lambda: [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        )

        extras = sampler._collect_panel_extras()

        assert extras["temps"] == [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        assert extras["err"] == ["temps_hwmon"]

    def test_hwmon_temps_survive_a_raising_ipmi_collector(self, monkeypatch):
        """IPMI raising must not cost the hwmon data, only its own field's slice."""
        monkeypatch.setattr(
            collectors, "get_temperatures", lambda: [{"n": "k10temp/Tctl", "c": 45.0}]
        )
        monkeypatch.setattr(
            collectors, "get_ipmi_temperatures", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        extras = sampler._collect_panel_extras()

        assert extras["temps"] == [{"n": "k10temp/Tctl", "c": 45.0}]
        assert extras["err"] == ["temps_ipmi"]

    def test_fans_are_the_union_of_hwmon_and_ipmi_sorted_by_name(self, monkeypatch):
        """The two fan sources are concatenated, then sorted together by ``n``.

        Their names cannot collide in practice (hwmon: ``chip/fanN``, IPMI:
        ``ipmi/FANn``), so a plain concatenate-then-sort is the whole
        contract -- no deduplication, no priority rule.
        """
        monkeypatch.setattr(
            collectors, "get_fans", lambda: [{"n": "nct6775/fan2", "rpm": 900}]
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_fans",
            lambda: [{"n": "ipmi/FAN1", "rpm": 7100}, {"n": "ipmi/FAN2", "rpm": 6900}],
        )

        extras = sampler._collect_panel_extras()

        assert extras["fans"] == [
            {"n": "ipmi/FAN1", "rpm": 7100},
            {"n": "ipmi/FAN2", "rpm": 6900},
            {"n": "nct6775/fan2", "rpm": 900},
        ]

    def test_fans_are_empty_when_neither_source_reports_anything(self, monkeypatch):
        """No hwmon fans and no IPMI fans is a legitimate empty union, not an error."""
        monkeypatch.setattr(collectors, "get_fans", lambda: [])
        monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [])

        extras = sampler._collect_panel_extras()

        assert extras["fans"] == []
        assert extras["err"] == []

    def test_ipmi_fans_survive_a_raising_hwmon_collector(self, monkeypatch):
        """hwmon raising must not cost the IPMI data, only its own field's slice."""
        monkeypatch.setattr(
            collectors, "get_fans", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [{"n": "ipmi/FAN1", "rpm": 7100}])

        extras = sampler._collect_panel_extras()

        assert extras["fans"] == [{"n": "ipmi/FAN1", "rpm": 7100}]
        assert extras["err"] == ["fans_hwmon"]

    def test_hwmon_fans_survive_a_raising_ipmi_collector(self, monkeypatch):
        """IPMI raising must not cost the hwmon data, only its own field's slice."""
        monkeypatch.setattr(
            collectors, "get_fans", lambda: [{"n": "nct6775/fan1", "rpm": 1200}]
        )
        monkeypatch.setattr(
            collectors, "get_ipmi_fans", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        extras = sampler._collect_panel_extras()

        assert extras["fans"] == [{"n": "nct6775/fan1", "rpm": 1200}]
        assert extras["err"] == ["fans_ipmi"]

    def test_dcgm_gpu_is_absent_and_uncalled_when_url_is_unset(self, monkeypatch):
        """Unset SYS_STATS_DCGM_URL: no scrape attempt, no ``dcgm_gpu`` key at all.

        The key's absence -- not an empty list under it -- is what tells
        server._build_panel_payload to keep building gpu[] from
        stats["gpu"] the way it always has.
        """
        monkeypatch.delenv("SYS_STATS_DCGM_URL", raising=False)

        def _explode(url):
            raise AssertionError("no DCGM scrape expected when the URL is unset")

        monkeypatch.setattr(collectors, "get_dcgm_gpus", _explode)

        extras = sampler._collect_panel_extras()

        assert "dcgm_gpu" not in extras
        assert extras["err"] == []

    def test_dcgm_gpu_is_included_and_url_forwarded_when_configured(self, monkeypatch):
        """A configured URL is forwarded verbatim and the collector's result kept as-is."""
        monkeypatch.setenv("SYS_STATS_DCGM_URL", "http://10.50.0.106:30940/metrics")
        seen_urls = []
        entry = {
            "i": 0, "n": "RTX 3090", "load": 12.0, "mem_used": 1, "mem_total": 2,
            "mem_pct": 50.0, "temp": 40.0, "fan": 0, "power": 100.0,
        }

        def _get_dcgm_gpus(url):
            seen_urls.append(url)
            return [entry]

        monkeypatch.setattr(collectors, "get_dcgm_gpus", _get_dcgm_gpus)

        extras = sampler._collect_panel_extras()

        assert seen_urls == ["http://10.50.0.106:30940/metrics"]
        assert extras["dcgm_gpu"] == [entry]
        assert extras["err"] == []

    def test_dcgm_empty_result_tags_gpu_in_err(self, monkeypatch):
        """get_dcgm_gpus never raises on failure -- an empty result while
        configured is the failure signal itself, so it must land in err.
        """
        monkeypatch.setenv("SYS_STATS_DCGM_URL", "http://10.50.0.106:30940/metrics")
        monkeypatch.setattr(collectors, "get_dcgm_gpus", lambda url: [])

        extras = sampler._collect_panel_extras()

        assert extras["dcgm_gpu"] == []
        assert extras["err"] == ["gpu"]

    def test_dcgm_gpu_survives_an_unexpected_raise(self, monkeypatch):
        """Defensive backstop, matching every other field above: even though
        get_dcgm_gpus is documented to never raise, an unexpected exception
        must degrade to an empty list and the "gpu" tag, not lose the rest
        of the extras dict.
        """
        monkeypatch.setenv("SYS_STATS_DCGM_URL", "http://10.50.0.106:30940/metrics")
        monkeypatch.setattr(
            collectors, "get_dcgm_gpus", lambda url: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        extras = sampler._collect_panel_extras()

        assert extras["dcgm_gpu"] == []
        assert extras["err"] == ["gpu"]


class TestIpmiPollingCadence:
    """Tests for the throttled IPMI cadence inside ``_collect_panel_extras``.

    ``get_ipmi_temperatures``/``get_ipmi_fans`` poll the host's BMC over
    ``ipmitool`` and run on their own ``SYS_STATS_IPMI_INTERVAL`` cadence,
    separate from ``get_temperatures``/``get_fans``, which stay on the
    normal per-pass cadence. Every test here drives the clock with a fake
    ``time.monotonic`` so the cadence is deterministic instead of
    depending on wall-clock timing.
    """

    def test_ipmi_collected_on_the_first_pass(self, monkeypatch):
        """The very first call polls IPMI immediately, with no prior state."""
        monkeypatch.setattr(
            collectors, "get_ipmi_temperatures", lambda: [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        )
        monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [{"n": "ipmi/FAN1", "rpm": 7100}])

        extras = sampler._collect_panel_extras()

        assert extras["temps"] == [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        assert extras["fans"] == [{"n": "ipmi/FAN1", "rpm": 7100}]

    def test_ipmi_not_re_collected_on_a_pass_inside_the_interval(self, monkeypatch):
        """A second pass before SYS_STATS_IPMI_INTERVAL elapses must not call IPMI again."""
        fake_now = [1000.0]
        monkeypatch.setattr(sampler.time, "monotonic", lambda: fake_now[0])

        temps_calls = []
        fans_calls = []
        monkeypatch.setattr(
            collectors,
            "get_ipmi_temperatures",
            lambda: temps_calls.append(1) or [{"n": "ipmi/CPU1 Temp", "c": 38.0}],
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_fans",
            lambda: fans_calls.append(1) or [{"n": "ipmi/FAN1", "rpm": 7100}],
        )

        sampler._collect_panel_extras()  # first pass: polls, primes the cache

        # Advance the clock, but stay well inside the default 30s interval.
        fake_now[0] += 5.0
        extras = sampler._collect_panel_extras()

        assert len(temps_calls) == 1
        assert len(fans_calls) == 1
        # The cached value from the first pass must still be present.
        assert extras["temps"] == [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        assert extras["fans"] == [{"n": "ipmi/FAN1", "rpm": 7100}]

    def test_ipmi_re_collected_once_the_interval_has_elapsed(self, monkeypatch):
        """Once SYS_STATS_IPMI_INTERVAL has elapsed, the next pass polls again."""
        fake_now = [1000.0]
        monkeypatch.setattr(sampler.time, "monotonic", lambda: fake_now[0])

        temps_calls = []
        results = iter(
            [
                [{"n": "ipmi/CPU1 Temp", "c": 38.0}],
                [{"n": "ipmi/CPU1 Temp", "c": 41.0}],
            ]
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_temperatures",
            lambda: temps_calls.append(1) or next(results),
        )
        monkeypatch.setattr(collectors, "get_ipmi_fans", lambda: [])

        sampler._collect_panel_extras()  # first pass: polls, primes the cache

        fake_now[0] += sampler.DEFAULT_IPMI_INTERVAL  # exactly the interval
        extras = sampler._collect_panel_extras()

        assert len(temps_calls) == 2
        assert extras["temps"] == [{"n": "ipmi/CPU1 Temp", "c": 41.0}]

    def test_hwmon_is_still_collected_every_pass(self, monkeypatch):
        """hwmon collectors are unaffected: they run every pass, regardless of the IPMI cadence."""
        fake_now = [1000.0]
        monkeypatch.setattr(sampler.time, "monotonic", lambda: fake_now[0])

        hwmon_temp_calls = []
        monkeypatch.setattr(
            collectors,
            "get_temperatures",
            lambda: hwmon_temp_calls.append(1) or [{"n": "k10temp/Tctl", "c": 45.0}],
        )

        sampler._collect_panel_extras()
        fake_now[0] += 1.0  # well inside the IPMI interval
        sampler._collect_panel_extras()
        fake_now[0] += 1.0
        sampler._collect_panel_extras()

        assert len(hwmon_temp_calls) == 3

    def test_env_var_controls_the_cadence(self, monkeypatch):
        """SYS_STATS_IPMI_INTERVAL, not the hardcoded default, gates the re-poll."""
        monkeypatch.setenv("SYS_STATS_IPMI_INTERVAL", "10")
        fake_now = [1000.0]
        monkeypatch.setattr(sampler.time, "monotonic", lambda: fake_now[0])

        temps_calls = []
        monkeypatch.setattr(
            collectors,
            "get_ipmi_temperatures",
            lambda: temps_calls.append(1) or [],
        )

        sampler._collect_panel_extras()  # first pass

        fake_now[0] += 9.0  # inside the configured 10s interval
        sampler._collect_panel_extras()
        assert len(temps_calls) == 1

        fake_now[0] += 1.0  # now exactly 10s since the first poll
        sampler._collect_panel_extras()
        assert len(temps_calls) == 2

    def test_a_failed_ipmi_poll_keeps_serving_the_last_good_value(self, monkeypatch):
        """A raising IPMI collector must not poison the cache.

        Instead of clearing the cached readings, a failed poll keeps
        serving the last good value and tags the failure into ``err`` --
        the design choice documented in sampler._collect_panel_extras: a
        wall panel losing its fan/temperature readings entirely is worse
        than showing readings up to SYS_STATS_IPMI_INTERVAL old.
        """
        fake_now = [1000.0]
        monkeypatch.setattr(sampler.time, "monotonic", lambda: fake_now[0])

        monkeypatch.setattr(
            collectors, "get_ipmi_temperatures", lambda: [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        )
        monkeypatch.setattr(
            collectors, "get_ipmi_fans", lambda: [{"n": "ipmi/FAN1", "rpm": 7100}]
        )
        sampler._collect_panel_extras()  # first pass: primes the cache with good data

        # Now the BMC starts failing, once the interval has elapsed again.
        fake_now[0] += sampler.DEFAULT_IPMI_INTERVAL
        monkeypatch.setattr(
            collectors,
            "get_ipmi_temperatures",
            lambda: (_ for _ in ()).throw(RuntimeError("ipmitool timed out")),
        )
        monkeypatch.setattr(
            collectors,
            "get_ipmi_fans",
            lambda: (_ for _ in ()).throw(RuntimeError("ipmitool timed out")),
        )

        extras = sampler._collect_panel_extras()

        # The stale-but-last-known-good reading is still served...
        assert extras["temps"] == [{"n": "ipmi/CPU1 Temp", "c": 38.0}]
        assert extras["fans"] == [{"n": "ipmi/FAN1", "rpm": 7100}]
        # ...and the failure is signalled, not silently swallowed.
        assert extras["err"] == ["temps_ipmi", "fans_ipmi"]


class TestGetPanelSnapshot:
    """Tests for ``get_panel_snapshot``, the accessor exclusive to ``/panel``."""

    def test_returns_all_nones_before_any_sample(self):
        """A fresh sampler with no thread running has nothing cached yet."""
        assert sampler.get_panel_snapshot() == (None, None, None, None)

    def test_returns_the_stats_and_extras_stored_together(self, monkeypatch):
        """``_sample_once`` stores both halves under the same timestamps."""
        monkeypatch.setattr(
            collectors, "collect_stats", lambda limit=5: {"top_cpu": [], "top_memory": [], "top_gpu_processes": []}
        )
        monkeypatch.setattr(collectors, "get_temperatures", lambda: [{"n": "cpu", "c": 40.0}])

        sampler._sample_once(limit=5)
        stats, extras, wall_ts, monotonic_ts = sampler.get_panel_snapshot()

        assert stats == {"top_cpu": [], "top_memory": [], "top_gpu_processes": []}
        assert extras["temps"] == [{"n": "cpu", "c": 40.0}]
        assert wall_ts is not None
        assert monotonic_ts is not None

        # get_snapshot() -- the accessor /stats actually uses -- must return
        # the very same stats dict and timestamps, proving /panel's extras
        # ride alongside the existing cache rather than replacing it.
        stats_only, snap_wall_ts, snap_monotonic_ts = sampler.get_snapshot()
        assert stats_only is stats
        assert snap_wall_ts == wall_ts
        assert snap_monotonic_ts == monotonic_ts

    def test_reset_for_tests_clears_the_panel_extras_too(self, monkeypatch):
        """``_reset_for_tests`` must not leave a stale extras dict behind."""
        monkeypatch.setattr(
            collectors, "collect_stats", lambda limit=5: {"top_cpu": [], "top_memory": [], "top_gpu_processes": []}
        )

        sampler._sample_once(limit=5)
        sampler._reset_for_tests()

        assert sampler.get_panel_snapshot() == (None, None, None, None)


def test_sample_once_collects_stats_exactly_once_per_pass(monkeypatch):
    """``/panel`` must never double the ``collect_stats`` cost of a pass.

    ``collect_stats`` is the single place GPU data (and its ``nvidia-smi``
    subprocess calls) gets collected; the whole point of sharing one sampler
    pass between ``/stats`` and ``/panel`` is that it runs exactly once per
    iteration, never once per consumer.
    """
    calls: list[int] = []
    monkeypatch.setattr(
        collectors,
        "collect_stats",
        lambda limit=5: calls.append(1)
        or {"top_cpu": [], "top_memory": [], "top_gpu_processes": [], "gpu": []},
    )

    sampler._sample_once(limit=5)

    assert len(calls) == 1
