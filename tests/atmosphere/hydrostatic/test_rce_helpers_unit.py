"""Unit tests for the shared helpers in ``tests.atmosphere.hydrostatic._rce_helpers``.

iter-46 factored ``_assert_dt_used`` + ``_assert_max_wind_peak_below``
out of the iter-44 C72 nightly so the same regression pattern could
be applied to C48 (iter-46), C96 (iter-47), V4 (iter-50), and LL32/T21
(iter-51) without duplicating the parse-and-assert logic. iter-52
adds direct unit tests for those helpers so a regression in the
assertion logic itself fails in < 1 s, well before any of the slow
nightly tests (which take 100-2400 s each) would run.

Covers:
* ``_assert_dt_used`` happy path (strict ``==``, with ``abs_tol``).
* ``_assert_dt_used`` failure modes (wrong dt, missing results.txt,
  missing dt field, abs_tol violation).
* ``_assert_max_wind_peak_below`` happy path (peak < cap).
* ``_assert_max_wind_peak_below`` failure modes (peak > cap, missing
  csv, header-only csv → iter-46 Codex MEDIUM vacuous-pass fix,
  missing max_wind column).
"""
from __future__ import annotations

import pytest

from tests.atmosphere.hydrostatic._rce_helpers import (
    _assert_dt_used,
    _assert_max_wind_peak_below,
    _assert_rce_pass,
    _parse_notes,
    _parse_results,
)


# ---------------------------------------------------------------------------
# _assert_dt_used
# ---------------------------------------------------------------------------


def _write_results(tmp_path, lines):
    """Write a fake ``results.txt`` and return its parent dir."""
    out = tmp_path / "fake_run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.txt").write_text("\n".join(lines) + "\n")
    return out


def test_assert_dt_used_strict_pass(tmp_path):
    out = _write_results(tmp_path, ["dt: 150.0", "status: PASS"])
    _assert_dt_used(out, label="strict", expected_dt=150.0)


def test_assert_dt_used_strict_fail_on_wrong_dt(tmp_path):
    out = _write_results(tmp_path, ["dt: 75.0", "status: PASS"])
    with pytest.raises(AssertionError, match="dt=75.0 != 150.0"):
        _assert_dt_used(out, label="strict-wrong", expected_dt=150.0)


def test_assert_dt_used_strict_fail_on_close_but_not_equal(tmp_path):
    """Strict mode (abs_tol=None) must reject any deviation, however
    small. Catches a regression that silently rounds dt."""
    out = _write_results(tmp_path, ["dt: 150.001", "status: PASS"])
    with pytest.raises(AssertionError):
        _assert_dt_used(out, label="strict-tiny", expected_dt=150.0)


def test_assert_dt_used_missing_results_txt(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    with pytest.raises(AssertionError, match="did not produce results.txt"):
        _assert_dt_used(out, label="missing", expected_dt=150.0)


def test_assert_dt_used_missing_dt_field(tmp_path):
    """results.txt without a dt: line → float("nan") → dt_used==nan
    → NaN compares False against anything → assertion fires."""
    out = _write_results(tmp_path, ["status: PASS", "grid: cubed_sphere"])
    with pytest.raises(AssertionError):
        _assert_dt_used(out, label="no-dt", expected_dt=150.0)


def test_assert_dt_used_abs_tol_pass(tmp_path):
    """LL32 use case: dt is post-clamped to a non-integer (~81.844)."""
    out = _write_results(tmp_path, ["dt: 81.844088", "status: PASS"])
    _assert_dt_used(
        out, label="abs-tol-pass", expected_dt=81.844, abs_tol=1e-2,
    )


def test_assert_dt_used_abs_tol_at_boundary(tmp_path):
    """abs_tol uses ``<=`` so values within tolerance are accepted.
    Use a value slightly inside the boundary to avoid float-precision
    edge cases (``0.01`` arithmetic on doubles can round either way).
    """
    out = _write_results(tmp_path, ["dt: 81.852"])  # diff = 0.008 < 0.01
    _assert_dt_used(
        out, label="abs-tol-boundary", expected_dt=81.844, abs_tol=1e-2,
    )


def test_assert_dt_used_abs_tol_fail(tmp_path):
    """abs_tol violation = clear, informative fail message."""
    out = _write_results(tmp_path, ["dt: 82.0"])
    with pytest.raises(AssertionError, match="81.844 ± 0.01"):
        _assert_dt_used(
            out, label="abs-tol-fail", expected_dt=81.844, abs_tol=1e-2,
        )


def test_assert_dt_used_abs_tol_zero_is_strict_eq(tmp_path):
    """``abs_tol=0`` must accept exact match (uses <=). Codex iter-51
    confirmed no confounding between strict and tolerant paths."""
    out = _write_results(tmp_path, ["dt: 600.0"])
    _assert_dt_used(
        out, label="zero-tol", expected_dt=600.0, abs_tol=0.0,
    )
    out2 = _write_results(tmp_path / "x", ["dt: 600.0001"])
    with pytest.raises(AssertionError):
        _assert_dt_used(
            out2, label="zero-tol-fail", expected_dt=600.0, abs_tol=0.0,
        )


# ---------------------------------------------------------------------------
# _assert_max_wind_peak_below
# ---------------------------------------------------------------------------


def _write_timeseries(tmp_path, rows, *, header=("step", "max_wind")):
    """Write a fake ``mean_timeseries.csv`` and return its parent dir."""
    out = tmp_path / "fake_ts"
    out.mkdir()
    import csv
    with open(out / "mean_timeseries.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return out


def test_assert_max_wind_peak_below_happy_path(tmp_path):
    out = _write_timeseries(tmp_path, [(1, 2.5), (2, 5.0), (3, 4.8)])
    _assert_max_wind_peak_below(out, label="happy", cap=10.0)


def test_assert_max_wind_peak_below_takes_abs_value(tmp_path):
    """The helper does ``abs(float(row["max_wind"]))`` — a negative
    spike must count as a peak. Protects a future regression that
    silently signs the diagnostic."""
    out = _write_timeseries(tmp_path, [(1, 2.0), (2, -15.0), (3, 3.0)])
    with pytest.raises(AssertionError, match="peak max"):
        _assert_max_wind_peak_below(out, label="abs", cap=10.0)


def test_assert_max_wind_peak_below_fail_on_peak_above_cap(tmp_path):
    out = _write_timeseries(tmp_path, [(1, 5.0), (2, 30.0), (3, 7.0)])
    with pytest.raises(AssertionError, match="peak max"):
        _assert_max_wind_peak_below(out, label="over", cap=25.0)


def test_assert_max_wind_peak_below_missing_csv(tmp_path):
    out = tmp_path / "no_csv"
    out.mkdir()
    with pytest.raises(AssertionError, match="missing"):
        _assert_max_wind_peak_below(out, label="missing", cap=25.0)


def test_assert_max_wind_peak_below_empty_csv_rejected(tmp_path):
    """iter-46 Codex MEDIUM fix: header-only csv must NOT pass
    vacuously (peak_v=0.0 would otherwise always satisfy <cap).
    """
    out = _write_timeseries(tmp_path, [])
    with pytest.raises(AssertionError, match="no parseable"):
        _assert_max_wind_peak_below(out, label="empty", cap=25.0)


def test_assert_max_wind_peak_below_missing_max_wind_column(tmp_path):
    """iter-45 column-name pin — different column names must FAIL
    rather than silently skip the data."""
    out = _write_timeseries(
        tmp_path,
        [(1, 5.0)],
        header=("step", "max_v"),  # was iter-44 substring-match
    )
    with pytest.raises(AssertionError, match="max_wind"):
        _assert_max_wind_peak_below(out, label="wrong-col", cap=25.0)


def test_assert_max_wind_peak_below_unparseable_rows_rejected(tmp_path):
    """If every row's max_wind is non-numeric, helper must fail —
    not pass with peak_v=0.0."""
    out = _write_timeseries(
        tmp_path,
        [(1, "n/a"), (2, "n/a"), (3, "n/a")],
    )
    with pytest.raises(AssertionError, match="no parseable"):
        _assert_max_wind_peak_below(out, label="bad-data", cap=25.0)


def test_assert_max_wind_peak_below_all_nan_rows_rejected(tmp_path):
    """iter-66 Codex HIGH fix: ``float("nan")`` parses successfully
    but ``max(0.0, nan)`` returns 0.0 in CPython (NaN-naive
    comparison) — pre-iter-66 helper would set seen_max_wind=True
    yet leave peak_v=0.0 → silent vacuous pass against cap.
    The math.isfinite guard now rejects NaN rows, falling back to
    the same "no parseable finite rows" assertion.
    """
    out = _write_timeseries(
        tmp_path,
        [(1, "nan"), (2, "nan"), (3, "nan")],
    )
    with pytest.raises(AssertionError, match="no parseable"):
        _assert_max_wind_peak_below(out, label="nan-rows", cap=25.0)


def test_assert_max_wind_peak_below_mixed_nan_and_finite(tmp_path):
    """Mixed NaN + finite rows: helper must use the finite values
    only. A regression that re-introduced max(0.0, nan)=0.0 would
    pass here vacuously (NaN row encountered first → seen_max_wind
    + peak_v=0.0 stays at 0)."""
    out = _write_timeseries(
        tmp_path,
        [(1, "nan"), (2, 5.0), (3, "nan"), (4, 12.0)],
    )
    # peak finite value is 12.0; cap 25 → pass.
    _assert_max_wind_peak_below(out, label="mixed-nan", cap=25.0)
    # peak finite value is 12.0; cap 10 → fail (would silent-pass
    # under the pre-iter-66 bug).
    with pytest.raises(AssertionError, match="peak max"):
        _assert_max_wind_peak_below(out, label="mixed-nan-fail", cap=10.0)


# ---------------------------------------------------------------------------
# iter-87: distinguish NaN (skip, junk data) from inf (propagate,
# CFL crash signal). The iter-66 ``not isfinite`` skip incorrectly
# treated inf the same as NaN — but inf is a real signal of CFL
# blowup that the cap check should fire on.
# ---------------------------------------------------------------------------


def test_assert_max_wind_peak_below_inf_propagates_to_cap_check(tmp_path):
    """A single ``inf`` row should make peak_v=inf and trip the cap
    check (NOT silently skip like NaN). Models a future driver that
    emits inf max_wind on CFL crash."""
    out = _write_timeseries(
        tmp_path,
        [(1, 5.0), (2, "inf"), (3, 3.0)],
    )
    with pytest.raises(AssertionError, match="peak max"):
        _assert_max_wind_peak_below(out, label="inf-crash", cap=25.0)


def test_assert_max_wind_peak_below_all_inf_rows_trip_cap(tmp_path):
    """All inf rows: seen_max_wind=True (vs iter-66 NaN case where
    it'd be False), peak_v=inf, cap check fires. The error message
    is the cap-exceeded assertion, NOT the "no parseable" one."""
    out = _write_timeseries(
        tmp_path,
        [(1, "inf"), (2, "inf"), (3, "inf")],
    )
    with pytest.raises(AssertionError, match="peak max"):
        _assert_max_wind_peak_below(out, label="all-inf", cap=25.0)


def test_assert_max_wind_peak_below_mixed_nan_inf_and_finite(tmp_path):
    """Mixed bag: NaN skipped, inf propagates, finite values counted.
    With one inf row, peak_v=inf → cap check fires regardless of
    finite-row values."""
    out = _write_timeseries(
        tmp_path,
        [(1, "nan"), (2, 5.0), (3, "inf"), (4, 12.0)],
    )
    with pytest.raises(AssertionError, match="peak max"):
        _assert_max_wind_peak_below(out, label="mixed-nan-inf", cap=25.0)


# ---------------------------------------------------------------------------
# iter-88: optional min_floor parameter closes all-zero silent-pass class.
# ---------------------------------------------------------------------------


def test_assert_max_wind_peak_below_min_floor_default_unchanged(tmp_path):
    """``min_floor=None`` (default) preserves iter-46/66/87 behavior.
    All-zero rows pass cap check without floor enforcement."""
    out = _write_timeseries(tmp_path, [(1, 0.0), (2, 0.0), (3, 0.0)])
    _assert_max_wind_peak_below(out, label="zeros-default", cap=10.0)


def test_assert_max_wind_peak_below_min_floor_catches_all_zeros(tmp_path):
    """``min_floor=0.5`` flags all-zero (broken dycore) runs."""
    out = _write_timeseries(tmp_path, [(1, 0.0), (2, 0.0), (3, 0.0)])
    with pytest.raises(AssertionError, match="below activity floor"):
        _assert_max_wind_peak_below(
            out, label="zeros-floor", cap=10.0, min_floor=0.5,
        )


def test_assert_max_wind_peak_below_min_floor_passes_real_run(tmp_path):
    """``min_floor=0.5`` accepts a real run with max|v|=2.28 (iter-12
    V4 30-day measurement) — must NOT false-fail."""
    out = _write_timeseries(tmp_path, [(1, 1.5), (2, 2.0), (3, 2.28)])
    _assert_max_wind_peak_below(
        out, label="real-run", cap=10.0, min_floor=0.5,
    )


def test_assert_max_wind_peak_below_min_floor_at_boundary(tmp_path):
    """Peak exactly at floor: ``peak_v >= min_floor`` (<= boundary
    accepted, like ``_assert_dt_used`` abs_tol)."""
    out = _write_timeseries(tmp_path, [(1, 0.5)])
    _assert_max_wind_peak_below(
        out, label="boundary", cap=10.0, min_floor=0.5,
    )


def test_assert_max_wind_peak_below_min_floor_cap_check_first(tmp_path):
    """Order matters: cap check fires BEFORE floor check. If peak
    exceeds cap AND floor, the user sees the cap error (more
    actionable). Test by passing peak > cap > floor."""
    out = _write_timeseries(tmp_path, [(1, 30.0)])
    with pytest.raises(AssertionError, match="exceeds 25.0 m/s cap"):
        _assert_max_wind_peak_below(
            out, label="cap-first", cap=25.0, min_floor=0.5,
        )


# ---------------------------------------------------------------------------
# iter-83: unit coverage for _parse_results / _parse_notes / _assert_rce_pass.
# These were extracted to _rce_helpers.py at iter-78 alongside _assert_dt_used
# + _assert_max_wind_peak_below (which iter-52 already covered). Now all 5
# pure helpers in the module have direct unit tests.
# ---------------------------------------------------------------------------


def test_parse_results_basic(tmp_path):
    """Parse a typical results.txt with key: value pairs."""
    out = _write_results(tmp_path, [
        "test: rce",
        "grid: cubed_sphere",
        "resolution: 24",
        "dt: 600.0",
        "status: PASS",
        "notes: mean_T_sfc=300.65, max|v|=7.23",
    ])
    fields = _parse_results(out)
    assert fields is not None
    assert fields["test"] == "rce"
    assert fields["dt"] == "600.0"
    assert fields["status"] == "PASS"
    assert "mean_T_sfc=300.65" in fields["notes"]


def test_parse_results_returns_none_when_missing(tmp_path):
    """No results.txt → None (driver crashed before writing)."""
    out = tmp_path / "no_results"
    out.mkdir()
    assert _parse_results(out) is None


def test_parse_results_handles_empty_value(tmp_path):
    """Some keys may have empty values (e.g. ``notes:`` with no
    diagnostics). Should parse as empty string, not crash."""
    out = _write_results(tmp_path, ["status: PASS", "notes:"])
    fields = _parse_results(out)
    assert fields["notes"] == ""


def test_parse_results_only_lines_with_colon(tmp_path):
    """Lines without a colon are skipped (e.g. blank lines, comments
    that don't follow the key:value convention). Codex iter-85 LOW:
    tightened from a tolerant ``or`` to a strict equality after
    re-reading ``_parse_results`` — the ``":" in line`` guard means
    no-colon lines are SKIPPED, never returned with empty value.
    """
    out = _write_results(tmp_path, [
        "",
        "# leading comment",
        "status: PASS",
        "trailing-no-colon",
    ])
    fields = _parse_results(out)
    assert fields == {"status": "PASS"}


def test_parse_notes_basic():
    """``key=val, key=val`` parse to float dict."""
    notes = "mean_T_sfc=300.65, mean_T=266.97, max|v|=7.23"
    out = _parse_notes(notes)
    assert out["mean_T_sfc"] == 300.65
    assert out["mean_T"] == 266.97
    assert out["max|v|"] == 7.23


def test_parse_notes_skips_non_floats():
    """Non-float values get dropped without raising."""
    notes = "status=PASS, max|v|=2.28, label=clean"
    out = _parse_notes(notes)
    assert out == {"max|v|": 2.28}


def test_parse_notes_handles_whitespace():
    """Whitespace around ``=`` and ``,`` is tolerated."""
    notes = "  mean_T_sfc  =  300.13  ,  max|v|  =  8.43  "
    out = _parse_notes(notes)
    assert out["mean_T_sfc"] == 300.13
    assert out["max|v|"] == 8.43


def test_parse_notes_empty():
    """Empty notes returns empty dict."""
    assert _parse_notes("") == {}


def test_assert_rce_pass_happy_path(tmp_path):
    """status=PASS + mean_T_sfc within tol + max|v| < cap."""
    out = _write_results(tmp_path, [
        "status: PASS",
        "notes: mean_T_sfc=299.95, max|v|=8.2",
    ])
    _assert_rce_pass(out, label="happy", temp_tol=1.0, max_v_cap=20.0)


def test_assert_rce_pass_fail_on_wrong_status(tmp_path):
    out = _write_results(tmp_path, [
        "status: FAIL",
        "notes: mean_T_sfc=299.95, max|v|=8.2",
    ])
    with pytest.raises(AssertionError, match="want PASS"):
        _assert_rce_pass(out, label="status-fail")


def test_assert_rce_pass_fail_on_temp_drift(tmp_path):
    """mean_T_sfc drift > temp_tol triggers."""
    out = _write_results(tmp_path, [
        "status: PASS",
        "notes: mean_T_sfc=305.0, max|v|=8.2",
    ])
    with pytest.raises(AssertionError, match="drifted"):
        _assert_rce_pass(out, label="temp-drift", temp_tol=1.0)


def test_assert_rce_pass_fail_on_max_v_cap(tmp_path):
    """max|v| > cap triggers."""
    out = _write_results(tmp_path, [
        "status: PASS",
        "notes: mean_T_sfc=300.05, max|v|=60.0",
    ])
    with pytest.raises(AssertionError, match="exceeds"):
        _assert_rce_pass(out, label="cap-fail", max_v_cap=50.0)


def test_assert_rce_pass_fail_on_missing_notes_fields(tmp_path):
    """notes line missing ``mean_T_sfc`` triggers."""
    out = _write_results(tmp_path, [
        "status: PASS",
        "notes: no_recognised_fields_here",
    ])
    with pytest.raises(AssertionError, match="missing mean_T_sfc"):
        _assert_rce_pass(out, label="notes-broken")


def test_assert_rce_pass_fail_on_missing_max_v(tmp_path):
    """notes line has mean_T_sfc but no max|v| → distinct assertion
    (Codex iter-85 LOW: iter-83 covered mean_T_sfc missing but not
    max|v| missing)."""
    out = _write_results(tmp_path, [
        "status: PASS",
        "notes: mean_T_sfc=300.0",
    ])
    with pytest.raises(AssertionError, match="missing max"):
        _assert_rce_pass(out, label="missing-maxv")


def test_assert_rce_pass_fail_on_missing_results(tmp_path):
    """No results.txt → AssertionError 'silently'."""
    out = tmp_path / "no_results"
    out.mkdir()
    with pytest.raises(AssertionError, match="did not produce"):
        _assert_rce_pass(out, label="no-results")
