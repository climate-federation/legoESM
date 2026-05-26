"""Unit tests for the plane-CRM test-helper parsers.

iter-40 added ``_parse_rad_call_count(stdout)`` to extract the
``rad_calls=N`` token from the driver's final ``Done.`` line; it
gates the iter-38/39/41 radiation-schedule assertions. Until iter-53
the regex was exercised only through the slow + fast integration
tests; iter-53 adds direct unit tests so a regex regression fails
in < 1 s instead of waiting for the next nightly.

Mirrors the iter-52 unit-test layer for the hydrostatic
``_assert_dt_used`` + ``_assert_max_wind_peak_below`` helpers.
"""
from __future__ import annotations

import pytest

from tests.atmosphere.nonhydrostatic.integration.test_plane_crm_end_to_end_smoke import (
    _parse_rad_call_count,
)


# ---------------------------------------------------------------------------
# happy-path matches
# ---------------------------------------------------------------------------


def test_parse_rad_call_count_basic():
    """Driver's actual ``Done.`` line format from
    ``scripts/run_rce_mpi_long.py:896-899``."""
    stdout = (
        "step  N\n"
        "Done. 60 steps, 0.003 days sim. Wall: 2.1 min. rad_calls=5.\n"
    )
    assert _parse_rad_call_count(stdout) == 5


def test_parse_rad_call_count_zero():
    """--no-radiation case (iter-39): rad_calls=0."""
    stdout = "Done. 86 steps, 0.005 days sim. Wall: 0.3 min. rad_calls=0.\n"
    assert _parse_rad_call_count(stdout) == 0


def test_parse_rad_call_count_large():
    """A long run (e.g. 30-day at production scale) — many calls."""
    stdout = (
        "Done. 518400 steps, 30.000 days sim. Wall: 9720.0 min. "
        "rad_calls=43200.\n"
    )
    assert _parse_rad_call_count(stdout) == 43200


# ---------------------------------------------------------------------------
# anchor robustness (iter-40 Codex LOW#4 fix)
# ---------------------------------------------------------------------------


def test_parse_rad_call_count_ignores_substring_in_other_lines():
    """An unrelated line containing ``rad_calls=`` must NOT confuse
    the parser. iter-40 regex anchors to ``^Done\\..*\\brad_calls=``
    via re.MULTILINE so only the Done. line counts."""
    stdout = (
        "# debug: total_rad_calls=999 (sum across ranks)\n"
        "Done. 60 steps, 0.003 days sim. Wall: 2.1 min. rad_calls=5.\n"
    )
    assert _parse_rad_call_count(stdout) == 5


def test_parse_rad_call_count_rejects_substring_only_match():
    """A future log line like ``total_rad_calls=42`` without a
    ``Done.`` prefix must NOT match — the line-anchor + word-bound
    combo enforces this."""
    stdout = "diagnostic: total_rad_calls=42 emitted\n"
    assert _parse_rad_call_count(stdout) is None


def test_parse_rad_call_count_rejects_float_suffix():
    """``rad_calls=5.0`` is invalid — iter-54 regex requires
    integer-then-period as the LAST non-whitespace token
    (``\\d+\\.[^\\S\\n]*$`` multiline). The float form fails
    because ``5.`` is followed by ``0``, not by line-end.

    iter-53 pinned the looser ``\\b\\d+\\b`` regex as "current
    behaviour" (silently extracted ``5`` from ``5.0``); iter-54
    closes that gap.
    """
    stdout = "Done. 60 steps. Wall: 2.1 min. rad_calls=5.0.\n"
    assert _parse_rad_call_count(stdout) is None


def test_parse_rad_call_count_rejects_trailing_token():
    """iter-54 Codex LOW fix: ``rad_calls=5. (cached)`` would have
    silently matched 5 under the iter-54 first-cut ``(?:\\s|$)``
    suffix (any one whitespace, then arbitrary trailing). The
    multiline ``$`` anchor + non-newline-whitespace tolerance
    closes that gap so any future debug-suffix on the Done. line
    must be addressed explicitly rather than silently absorbed.
    """
    stdout = "Done. 60 steps. Wall: 2.1 min. rad_calls=5. (cached=true)\n"
    assert _parse_rad_call_count(stdout) is None


def test_parse_rad_call_count_tolerates_trailing_whitespace():
    """Trailing spaces / tabs before the newline must STILL match —
    legitimate terminal-padding from f-string formatting or CRLF
    line endings should not break the parser."""
    stdout = "Done. 60 steps. Wall: 2.1 min. rad_calls=5.   \n"
    assert _parse_rad_call_count(stdout) == 5
    stdout_crlf = "Done. 60 steps. Wall: 2.1 min. rad_calls=5.\r\n"
    assert _parse_rad_call_count(stdout_crlf) == 5


# ---------------------------------------------------------------------------
# missing / malformed inputs
# ---------------------------------------------------------------------------


def test_parse_rad_call_count_missing_returns_none():
    """No ``rad_calls=`` marker → None (older driver / parser-side
    regression). Tests must then assert == 0 explicitly."""
    stdout = "Done. 60 steps, 0.003 days sim. Wall: 2.1 min.\n"
    assert _parse_rad_call_count(stdout) is None


def test_parse_rad_call_count_empty_stdout():
    stdout = ""
    assert _parse_rad_call_count(stdout) is None


def test_parse_rad_call_count_done_without_marker():
    """A ``Done.`` line without ``rad_calls=`` — must return None,
    not silently match a value from elsewhere."""
    stdout = (
        "rad_calls=999 (orphan)\n"
        "Done. 60 steps, 0.003 days sim.\n"
    )
    assert _parse_rad_call_count(stdout) is None


def test_parse_rad_call_count_two_done_lines_returns_none():
    """Pathological: two ``Done.`` lines in the same stdout (e.g.
    from a botched retry). iter-40 contract returns None when
    multiple matches found, rather than silently picking one."""
    stdout = (
        "Done. 60 steps, 0.003 days sim. Wall: 2.1 min. rad_calls=5.\n"
        "Done. 60 steps, 0.003 days sim. Wall: 2.1 min. rad_calls=7.\n"
    )
    assert _parse_rad_call_count(stdout) is None
