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

from omip_tripole_heat_closure import close_windows  # noqa: E402

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
