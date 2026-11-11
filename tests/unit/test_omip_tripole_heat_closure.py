"""Known-answer control for the tripole heat-closure gate.

A synthetic two-snapshot pair whose heat-content change is exactly the
applied surface + geothermal heat must close to rounding; a doubled q_net
must leave the planted residual; a step-count mismatch must be fatal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "validate" / "ocean_fidelity"))

from omip_tripole_heat_closure import (  # noqa: E402
    build_z_coord, close_windows, step_month_weights)

RHO, CP = 1025.0, 3994.0
NY, NX, H = 4, 5, 1000.0
DT_WIN = 5 * 86400.0
Q, GEO = 30.0, 0.1          # W/m2 surface, W/m2 seafloor


def _zc():
    from legoesm.ocean.vertical import create_ocean_z_star
    return create_ocean_z_star(n_levels=3, H_max=H, dz_surface=200.0, dz_deep=400.0)


def _snap(day, step, T, q=None):
    s = {"T": np.full((NY, NX, 3), T), "eta": np.zeros((NY, NX)),
         "H_bathy": np.full((NY, NX), H), "land_mask": np.ones((NY, NX)),
         "cell_area": np.full((NY, NX), 1.0e10), "time_days": np.float64(day),
         "step": np.float64(step)}
    if q is not None:
        s.update(q_net_mean=np.full((NY, NX), q),
                 flux_mean_n_steps=np.float64(2880),
                 flux_mean_window_s=np.float64(DT_WIN))
    return s


def _pair(q_saved):
    dT = (Q + GEO) * DT_WIN / (RHO * CP * H)     # exact column warming
    a = _snap(5.0, 2880, 10.0)
    b = _snap(10.0, 5760, 10.0 + dT, q=q_saved)
    return [a, b]


def _close(snaps):
    return close_windows(snaps, lambda s: _zc(), RHO, CP, GEO, None, 3.34e5)


def test_known_answer_closes_to_rounding():
    [row] = _close(_pair(Q))
    assert abs(row["residual_Wm2"]) < 1e-9 * (Q + GEO)
    assert row["Q_geothermal_W"] == pytest.approx(GEO * NY * NX * 1.0e10, rel=1e-15)


def test_doubled_qnet_leaves_the_planted_residual():
    [row] = _close(_pair(2.0 * Q))
    assert row["residual_Wm2"] == pytest.approx(-Q, rel=1e-9)


def test_step_count_mismatch_is_fatal():
    snaps = _pair(Q)
    snaps[1]["flux_mean_n_steps"] = np.float64(2879)
    with pytest.raises(SystemExit, match="accumulator holds"):
        _close(snaps)


def test_missing_flux_means_is_fatal():
    snaps = _pair(Q)
    del snaps[1]["q_net_mean"]
    with pytest.raises(SystemExit, match="flux-accumulate"):
        _close(snaps)


def test_partial_bottom_and_nonzero_eta_close_through_the_run_geometry():
    """Thickness via the run's constructors: e3t ladder -> partial bottom
    cell (H=800 in a 1000 m ladder) -> eta stretch (eta=+0.5 m)."""
    e3t = np.array([100.0, 300.0, 600.0])
    Hb, eta = 800.0, 0.5
    h_top = 100.0 * (Hb + eta) / Hb          # top cell, partial-cell z* stretch
    a = _snap(5.0, 2880, 10.0)
    b = _snap(10.0, 5760, 10.0, q=Q)
    b["T"][..., 0] += Q * DT_WIN / (RHO * CP * h_top)   # heat lands in the top cell
    for s in (a, b):
        s["H_bathy"][:] = Hb
        s["eta"][:] = eta
    zc = lambda s: build_z_coord(e3t, None, s["H_bathy"])[1]  # noqa: E731
    [row] = close_windows([a, b], zc, RHO, CP, 0.0, None, 3.34e5)
    assert abs(row["residual_Wm2"]) < 1e-9 * Q


def test_duration_must_match_the_snapshot_span():
    snaps = _pair(Q)
    snaps[1]["flux_mean_window_s"] = np.float64(DT_WIN - 1800.0)
    with pytest.raises(SystemExit, match="snapshot span"):
        _close(snaps)


def test_isf_month_follows_the_driver_rule():
    """Driver: month of step s = searchsorted(month_end, s*dt/86400, right);
    a step ending exactly on day 31 already belongs to February."""
    w = step_month_weights(31 * 48 - 2, 31 * 48 + 2, 1800.0)
    assert w[0] == pytest.approx(0.25) and w[1] == pytest.approx(0.75)
