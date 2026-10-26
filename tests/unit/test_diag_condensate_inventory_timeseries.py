"""Direct tests for the condensate-inventory-versus-time probe.

The probe's job is to say whether a reservoir gap is a leak or a fixed-point
offset, so the tests pin the two things that could make it lie: a wrong layer
mass (which cancels out of ratios and so hides), and a silently skipped date.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from legoesm import constants

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "validate" / \
    "diag_condensate_inventory_timeseries.py"
_spec = importlib.util.spec_from_file_location("diag_condensate_inventory", _SRC)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def _write(tmp_path: Path, day: int, q_c, p_s, A, B, name="run"):
    run = tmp_path / name
    run.mkdir(exist_ok=True)
    np.savez(
        run / f"checkpoint_day_{day:04d}.npz",
        meta_vgrid=np.stack([np.asarray(A, float), np.asarray(B, float)]),
        p_s=np.asarray(p_s, float),
        trc_q_c=np.asarray(q_c, float),
    )
    return run


def _sigma_table(nlev):
    """Pure terrain-following: A is zero, B spans 0 to 1."""
    return np.zeros(nlev + 1), np.linspace(0.0, 1.0, nlev + 1)


def test_uniform_mixing_ratio_gives_analytic_burden(tmp_path):
    """q constant through a full sigma column integrates to q * p_s / g."""
    nlev, q, p_s = 4, 1.0e-3, 1.0e5
    A, B = _sigma_table(nlev)
    run = _write(tmp_path, 5, np.full((3, nlev), q), np.full(3, p_s), A, B)
    got = diag.inventory(run / "checkpoint_day_0005.npz")["trc_q_c"]
    # rel 1e-6, not tighter: the shared column integral accumulates in the
    # conservation dtype, which is single precision unless x64 is enabled.
    assert got == pytest.approx(q * p_s / constants.g, rel=1e-6)


def test_hybrid_table_uses_reference_pressure_not_surface_pressure(tmp_path):
    """On a hybrid column the A part must NOT scale with surface pressure.

    Two cells at very different surface pressures share the same pure-pressure
    top layer, so that layer's mass must be identical in both.  Treating the
    table as terrain-following would make it differ by tens of hectopascals.
    """
    nlev = 3
    A = np.array([0.002, 0.02, 0.01, 0.0])
    B = np.array([0.0, 0.0, 0.4, 1.0])
    p_s = np.array([1.01e5, 0.60e5])
    dp = diag.layer_thickness(
        np.load(_write(tmp_path, 1, np.zeros((2, nlev)), p_s, A, B) /
                "checkpoint_day_0001.npz"))
    # the top layer has no terrain-following part, so it is the same mass
    # in a sea-level column and a high-altitude one
    assert dp[0, 0] == pytest.approx((0.02 - 0.002) * constants.p_ref)
    assert dp[0, 0] == pytest.approx(dp[1, 0])


def test_non_monotonic_table_raises_rather_than_returning_a_number(tmp_path):
    """Non-vacuity: a bad layer mass cancels out of ratios, so it must raise."""
    A = np.array([0.0, 0.3, 0.2, 0.2])          # third interface goes back up
    B = np.array([0.0, 0.0, 0.0, 1.0])
    ckpt = _write(tmp_path, 1, np.zeros((2, 3)), np.full(2, 1.0e5), A, B) / \
        "checkpoint_day_0001.npz"
    with pytest.raises(ValueError, match="monotonically increasing"):
        diag.layer_thickness(np.load(ckpt))


def test_series_reports_only_dates_that_exist(tmp_path):
    nlev = 2
    A, B = _sigma_table(nlev)
    run = _write(tmp_path, 5, np.full((2, nlev), 1e-3), np.full(2, 1e5), A, B)
    _write(tmp_path, 15, np.full((2, nlev), 2e-3), np.full(2, 1e5), A, B)
    got = diag.series(run, [5, 10, 15], species=("trc_q_c",))
    assert sorted(got) == [5, 15]


def test_drift_slope_is_least_squares_not_endpoint_difference():
    """A single noisy endpoint must not set the trend."""
    days = [5, 10, 15, 20, 25]
    values = [1.0, 2.0, 3.0, 4.0, 9.0]
    assert diag._drift(values, days) == pytest.approx(0.36, rel=1e-9)


def test_main_runs_end_to_end_and_reports_a_ratio(tmp_path, capsys):
    nlev = 2
    A, B = _sigma_table(nlev)
    a = _write(tmp_path, 5, np.full((2, nlev), 1e-3), np.full(2, 1e5), A, B, name="a")
    b = _write(tmp_path, 5, np.full((2, nlev), 5e-4), np.full(2, 1e5), A, B, name="b")
    assert diag.main([str(a), str(b), "--days", "5", "--species", "trc_q_c"]) == 0
    assert "0.500" in capsys.readouterr().out
