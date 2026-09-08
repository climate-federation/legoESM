"""Tests for the latlon_regional grid plumbing added for Petersen-scale runs.

The ocean test matrix gained ``latlon_regional`` support for the
lock-exchange case so legoESM can run a 64 km x 4 km equatorial channel
matching the Veros peer setup (``src/legoesm/ocean/fidelity/veros_configs/
lock_exchange.py``). The helpers exercised here used to handle only
``latlon`` / ``mpas`` / ``cubed_sphere``; this test pins the regional
dispatch so it does not silently regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "matrix"))

import run_ocean_test_matrix as m  # noqa: E402


def test_get_cell_latlon_rad_handles_latlon_regional():
    class FakeGrid:
        lat = np.linspace(-0.018 * np.pi / 180, 0.018 * np.pi / 180, 6)
        lon = np.linspace(0.0, 0.576 * np.pi / 180, 66)

    lat_2d, lon_2d = m._get_cell_latlon_rad("latlon_regional", FakeGrid())
    assert lat_2d.shape == (6, 66)
    assert lon_2d.shape == (6, 66)
    assert float(lon_2d.min()) >= 0.0


def test_get_cell_latlon_rad_handles_mpas_regional():
    class FakeMesh:
        latCell = np.array([0.0, 0.1, 0.2])
        lonCell = np.array([0.0, 0.5, 1.0])

    lat, lon = m._get_cell_latlon_rad("mpas_regional", FakeMesh())
    np.testing.assert_allclose(lat, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(lon, [0.0, 0.5, 1.0])


def test_lock_exchange_test_matrix_includes_latlon_regional_entry():
    matrix = m._build_test_matrix()
    entries = [t for t in matrix if t.case == "lock_exchange"
               and t.grid_type == "latlon_regional"]
    assert len(entries) == 1, (
        "expected exactly one Petersen-scale lock_exchange entry on "
        "latlon_regional"
    )
    tc = entries[0]
    assert tc.resolution == "4x64"
    kw = tc.run_kwargs
    assert kw["dt"] == 30.0, "Petersen dx ~ 1 km demands CFL-safe dt"
    assert kw["A_h"] == 0.0
    assert kw["A_v"] == 0.0
    assert kw["bottom_drag_r"] == 0.0
    # Free-surface diffusion / semi-implicit barotropic must be disabled
    # so the gravity-current signal is not damped before the comparison.
    assert kw["barotropic_diffusion_alpha"] == 0.0
    assert kw["bebt"] == 0.0
    assert kw["barotropic_time_filter"] == "box"
    assert kw["n_barotropic_substeps"] == 1
    # Linear EOS matching the Veros peer (eq_of_state_type = 1).
    assert kw["eos"] == "linear"
    assert kw["alpha_T"] == 2.0e-4
    assert kw["beta_S"] == 0.0
    assert kw["T_ref"] == 17.5
    # TRACER advection must be the SAME constant the global lat-lon and
    # tripole lock-exchange arms use, not a second literal -- that is what
    # stops the two lanes drifting apart, which is how this arm ended up on
    # WENO5. WENO5 was chosen for sharpness and is essentially-
    # non-oscillatory, NOT monotonicity-preserving; it finished this case
    # with water at -3.00 degC from an initial [5, 30] (2026-08-13).
    # Asserting the CONSTANT, not its value, so a deliberate family-wide
    # change stays a one-line edit.
    assert kw["tracer_advection"] == m.LOCKEX_CGRID_TRACER_ADV
    # MOMENTUM advection is untouched: the undershoot was measured against
    # the tracer scheme alone, and changing two things would have made the
    # sweep unreadable.
    assert kw["momentum_advection"] == "weno5"
    # ~64 km zonal, ~4 km meridional, centred on the equator
    assert kw["lon_east"] - kw["lon_west"] > 0.5
    assert kw["lat_north"] - kw["lat_south"] < 0.05
    assert abs(kw["lat_north"] + kw["lat_south"]) < 1e-3
