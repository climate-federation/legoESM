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

from tests.atmosphere.nonhydrostatic.integration._plane_crm_helpers import (
    _parse_rad_call_count,
    _read_log,
)


# ---------------------------------------------------------------------------
# happy-path matches
# ---------------------------------------------------------------------------


def test_parse_rad_call_count_basic():
    """Driver's actual ``Done.`` line format from
    ``scripts/run/run_rce_mpi_long.py:896-899``."""
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


# ---------------------------------------------------------------------------
# iter-84: unit coverage for _read_log (moved from
# test_plane_crm_end_to_end_smoke.py to _plane_crm_helpers.py).
# ---------------------------------------------------------------------------


def _write_log(tmp_path, content):
    out = tmp_path / "rce_out"
    out.mkdir()
    (out / "log.txt").write_text(content)
    return out


def test_read_log_basic_format(tmp_path):
    """Driver's actual log.txt format (header lines + CSV rows)."""
    out = _write_log(tmp_path, (
        "# RCE MPI LONG  n_ranks=1 grid=12x12 nlev=20 dx=2000.0 dt=5.0\n"
        "# physics: NO radiation + Kessler microphysics + Smag\n"
        "# step,day,CWV_mean,MSE_mean,max|w|\n"
        "1,0.000058,5.5001e+01,4.2049e+09,0.0000e+00\n"
        "20,0.001157,5.5001e+01,4.2049e+09,4.7400e-04\n"
    ))
    rows = _read_log(out)
    assert len(rows) == 2
    assert rows[0]["step"] == "1"
    assert rows[0]["CWV_mean"] == "5.5001e+01"
    assert rows[1]["max|w|"] == "4.7400e-04"


def test_read_log_returns_empty_when_missing(tmp_path):
    """Missing log.txt → empty list (driver crashed before writing)."""
    out = tmp_path / "no_log"
    out.mkdir()
    assert _read_log(out) == []


def test_read_log_skips_blank_lines(tmp_path):
    out = _write_log(tmp_path, (
        "# step,day,max|w|\n"
        "\n"
        "1,0.0,1e-3\n"
        "\n"
        "2,0.0001,2e-3\n"
    ))
    rows = _read_log(out)
    assert len(rows) == 2


def test_read_log_skips_extra_comments(tmp_path):
    """Lines starting with ``#`` after the header are skipped (e.g.
    a ``# BAIL: NaN at step N`` marker the driver writes on failure)."""
    out = _write_log(tmp_path, (
        "# RCE MPI LONG\n"
        "# step,day,max|w|\n"
        "1,0.0,1e-3\n"
        "# BAIL: NaN at step 2\n"
    ))
    rows = _read_log(out)
    assert len(rows) == 1


def test_read_log_rejects_row_with_wrong_column_count(tmp_path):
    """A row with too few/many fields is silently dropped (rather
    than producing a malformed dict). Defensive against driver
    schema drift mid-run."""
    out = _write_log(tmp_path, (
        "# step,day,max|w|\n"
        "1,0.0,1e-3\n"
        "2,0.0001\n"       # missing one field
        "3,0.0002,3e-3,extra\n"  # extra field
    ))
    rows = _read_log(out)
    # Only the well-formed row 1 makes it through.
    assert len(rows) == 1
    assert rows[0]["step"] == "1"


def test_read_log_without_header_returns_empty(tmp_path):
    """A log.txt that never contains the ``# step,...`` schema line
    has no parseable rows."""
    out = _write_log(tmp_path, (
        "# RCE MPI LONG header only\n"
        "# no schema line\n"
        "1,2,3\n"
    ))
    rows = _read_log(out)
    assert rows == []
