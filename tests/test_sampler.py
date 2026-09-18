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
    monkeypatch.setattr(
        sampler.psutil, "cpu_percent", lambda interval=None: calls.append("prime") or 0.0
    )
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


def test_run_survives_a_raising_collector_and_keeps_sampling(monkeypatch, caplog):
    """A collector that raises must not kill the sampler thread.

    The next iteration has to run regardless, or the cache would freeze on
    whatever was last cached (``None`` on a cold start) forever.
    """
    monkeypatch.setattr(sampler.psutil, "cpu_percent", lambda interval=None: 0.0)

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
    monkeypatch.setattr(sampler.psutil, "cpu_percent", lambda interval=None: 0.0)
    monkeypatch.setattr(
        collectors,
        "collect_stats",
        lambda limit=5: {"top_cpu": [], "top_memory": [], "top_gpu_processes": []},
    )

    sampler.start()
    first_thread = sampler._thread
    sampler.start()

    assert sampler._thread is first_thread
