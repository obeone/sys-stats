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
    monkeypatch.setattr(collectors, "get_fans", lambda: [])
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
    """
    calls: list[tuple[str, bool]] = []

    def _cpu_percent(interval=None, percpu=False):
        calls.append(("prime", percpu))
        return [] if percpu else 0.0

    monkeypatch.setattr(sampler.psutil, "cpu_percent", _cpu_percent)
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
    # Both variants primed, plain first then percpu, each exactly once,
    # and both strictly before the first collect_stats() call.
    assert priming_calls == [("prime", False), ("prime", True)]
    assert calls[2] == ("collect", None)


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
            ("get_fans", []),
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
            "get_fans": "fans",
            "get_swap": "swap",
            "get_per_core_cpu": "per_core",
            "get_load_average": "load",
            "get_cpu_frequency_mhz": "mhz",
        }[collector_name]

        assert extras[field] == default
        assert len(extras["err"]) == 1


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
