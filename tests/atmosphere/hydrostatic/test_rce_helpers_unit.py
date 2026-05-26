"""Unit tests for the shared helpers in ``test_rce_cross_grid_smoke.py``.

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
