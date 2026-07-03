"""Pin the Oceananigans barotropic_gyre wind-stress convention.

The gyre comparison driver must convert the Oceananigans KINEMATIC wind stress
(tau0=1e-2 m^2/s^2, applied as a FluxBoundaryCondition on the velocity field, so
already per unit reference density) to the DYNAMIC stress [N/m^2] that legoESM's
``OceanSurfaceForcing.tau_x`` expects (the model divides by rho_0 internally).
That conversion is a factor of rho_0 (=1000). Omitting it under-forces the gyre
by exactly rho_0 -> a laminar ~3e-4 m/s flow instead of the oracle's O(1 m/s)
western boundary current (the bug fixed in commit a17843799). This test fails
loudly if the rho_0 factor (or the validated +ocean_stress sign) regresses.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from legoesm.grids.latlon import create_regional_latlon_grid

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def gyre_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.compare_oceananigans_barotropic_gyre as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_wind_is_dynamic_stress_rho0_times_kinematic(gyre_module, monkeypatch):
    """build_wind must return rho_0 * tau_kinematic (the dynamic stress)."""
    mod = gyre_module
    monkeypatch.delenv("WIND_SIGN", raising=False)
    grid, _wall = create_regional_latlon_grid(
        mod.NY, mod.NX, mod.LAT_S, mod.LAT_N,
        lon_west=mod.LON_W, lon_east=mod.LON_E, periodic_x=False)
    wind = mod.build_wind(grid)
    tau_x = np.asarray(wind.tau_x)

    # Peak |dynamic stress| must be rho_0 * (peak kinematic amplitude on the
    # grid) -- the 1000x = rho_0 factor is the fix. The grid latitudes do not
    # land exactly on the cos peak, so use the discrete cos maximum.
    lat_deg = np.degrees(np.asarray(grid.lat))
    cos_peak = float(np.max(np.abs(
        np.cos(2 * np.pi * (lat_deg - mod.PHI0) / mod.LPHI))))
    expected = mod.RHO0 * mod.TAU0 * cos_peak
    peak = float(np.max(np.abs(tau_x)))
    assert peak == pytest.approx(expected, rel=1e-3), (
        f"wind peak {peak} != rho_0*tau0*cos_peak {expected}; the "
        "kinematic->dynamic rho_0 conversion regressed (gyre under-forced 1000x)."
    )
    # Sanity: it is ~rho_0x the raw kinematic amplitude, not 1x.
    assert peak > 100.0 * mod.TAU0

    # tau_y is zero (zonal wind only).
    assert np.all(np.asarray(wind.tau_y) == 0.0)


def test_wind_sign_is_plus_ocean_stress(gyre_module, monkeypatch):
    """Validated convention: legoESM receives +ocean_stress = +rho_0*tau0*cos.

    With the default WIND_SIGN=-1 the driver passes -sign*ocean_tau = +ocean_tau,
    so tau_x is POSITIVE where cos(2*pi*(phi-phi0)/Lphi) > 0 (near the southern
    wall phi~phi0=15). The -0.68 vs +0.68 day-10 corr pinned this sign.
    """
    mod = gyre_module
    monkeypatch.delenv("WIND_SIGN", raising=False)
    grid, _wall = create_regional_latlon_grid(
        mod.NY, mod.NX, mod.LAT_S, mod.LAT_N,
        lon_west=mod.LON_W, lon_east=mod.LON_E, periodic_x=False)
    lat_deg = np.degrees(np.asarray(grid.lat))
    cosfac = np.cos(2 * np.pi * (lat_deg - mod.PHI0) / mod.LPHI)
    tau_x = np.asarray(mod.build_wind(grid).tau_x)[:, 0]
    # Same sign as cos everywhere it is non-trivial.
    nontrivial = np.abs(cosfac) > 1e-3
    assert np.all(np.sign(tau_x[nontrivial]) == np.sign(cosfac[nontrivial]))


def test_wind_sign_env_override_flips(gyre_module, monkeypatch):
    """WIND_SIGN=+1 flips to -ocean_stress (the MITgcm-recipe negation)."""
    mod = gyre_module
    grid, _wall = create_regional_latlon_grid(
        mod.NY, mod.NX, mod.LAT_S, mod.LAT_N,
        lon_west=mod.LON_W, lon_east=mod.LON_E, periodic_x=False)
    monkeypatch.delenv("WIND_SIGN", raising=False)
    plus = np.asarray(mod.build_wind(grid).tau_x)
    monkeypatch.setenv("WIND_SIGN", "1")
    flipped = np.asarray(mod.build_wind(grid).tau_x)
    assert np.allclose(plus, -flipped)
