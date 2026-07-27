"""Tests for the pure formatting helpers of the terminal dashboard.

These functions are the presentation contract of :mod:`sys_stats.cli`: they turn
the raw byte counts and ISO timestamps returned by ``/stats`` into the strings
rendered in the Rich tables.
"""

import re
from datetime import datetime, timedelta, timezone

import pytest

from sys_stats.cli import (
    human_readable_size,
    time_until,
    truncate_cmdline,
    truncate_name,
)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0.0 B"),
        (512, "512.0 B"),
        (1023, "1023.0 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1024**2, "1.0 MB"),
        (1024**3, "1.0 GB"),
        (1024**4, "1.0 TB"),
    ],
)
def test_human_readable_size_scales_by_binary_unit(size, expected):
    """Byte counts are rendered with the largest unit below 1024."""
    assert human_readable_size(size) == expected


def test_human_readable_size_caps_at_terabytes():
    """Values beyond the unit table stay in TB rather than overflowing it."""
    assert human_readable_size(5 * 1024**5) == "5120.0 TB"


def test_truncate_cmdline_leaves_short_values_untouched():
    """A command line shorter than the budget is returned verbatim."""
    assert truncate_cmdline("python -m sys_stats", 40) == "python -m sys_stats"


def test_truncate_cmdline_ellipsises_and_respects_width():
    """A long command line is cut to exactly ``width`` characters, ellipsis included."""
    result = truncate_cmdline("a" * 50, 10)

    assert len(result) == 10
    assert result.endswith("…")


def test_truncate_name_uses_its_own_default_budget():
    """Process names default to 15 characters, ellipsis included."""
    assert truncate_name("short") == "short"

    result = truncate_name("an-extremely-long-process-name")
    assert len(result) == 15
    assert result.endswith("…")


def test_time_until_formats_a_future_deadline():
    """A future expiry is rendered as a ``H:MM:SS`` countdown."""
    expiration = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    assert re.fullmatch(r"\d+:\d{2}:\d{2}", time_until(expiration))


def test_time_until_accepts_the_zulu_suffix_ollama_returns():
    """Ollama reports ``expires_at`` with a ``Z`` suffix, which must parse."""
    expiration = (
        (datetime.now(timezone.utc) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    assert re.fullmatch(r"\d+:\d{2}:\d{2}", time_until(expiration))


def test_time_until_reports_expired_deadlines():
    """A deadline in the past is flagged instead of showing a negative delta."""
    expiration = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

    assert "Expired" in time_until(expiration)


@pytest.mark.parametrize("value", ["", "not-a-date", "2024-13-45T99:99:99"])
def test_time_until_falls_back_to_na_on_garbage(value):
    """Unparsable timestamps degrade to ``N/A`` instead of raising."""
    assert time_until(value) == "N/A"
