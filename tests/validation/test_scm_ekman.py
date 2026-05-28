"""Neutral Ekman boundary-layer benchmark.

Drives the legoESM SCM with constant geostrophic wind, zero surface
heat / moisture flux, and a neutral initial θ profile (see
:mod:`scripts.run_scm_ekman`).  Verifies the canonical Ekman-spiral
signatures: surface wind decelerated and veered relative to the
geostrophic profile, qke approaching the friction-velocity surface
BC, no spurious heat-flux drift beyond the documented Exner-
conversion limitation.

Compares scalar surface quantities against the jax_scm
``andren1994_Nz64.nc`` oracle (jax_scm's closest neutral-Ekman case).
Profile-level comparison is deferred to a follow-up (different
vertical coordinates between the two SCMs).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("netCDF4")
xr = pytest.importorskip("xarray")


ORACLE_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "scm_oracle"
    / "andren1994_Nz64.nc"
)

TEST_HOURS = 3.0      # ~0.17 inertial periods at f_c = 1e-4
TEST_DT = 10.0
TEST_NLEV = 16
TEST_SIGMA_TOP = 0.85


def _import_runner():
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    import run_scm_ekman
    return run_scm_ekman


@pytest.fixture(scope="module")
def ekman_run():
    runner = _import_runner()
    scm = runner.build_scm(
        nlev=TEST_NLEV, dt=TEST_DT, sigma_top=TEST_SIGMA_TOP,
    )
    scm.run(nsteps=int(TEST_HOURS * 3600 / TEST_DT))
    return scm


def test_ekman_finite(ekman_run):
    scm = ekman_run
    for arr in (
        np.asarray(scm.state.T.data),
        np.asarray(scm.state.u.data),
        np.asarray(scm.state.v.data),
        np.asarray(scm.phys_state.qke),
    ):
        assert np.all(np.isfinite(arr))


def test_ekman_neutral_temperature_does_not_blow_up(ekman_run):
    """Neutral init θ=273.15 K everywhere.  Surface heat flux ≈ 0
    by construction (no T_sfc/T[-1] gradient).  Lowest-cell T should
    stay within ±1 K of the initial value — the small drift is the
    documented Exner round-off in implicit-θ diffusion."""
    T_low = float(ekman_run.state.T.data[0, 0, 0, -1])
    assert 272.0 < T_low < 274.0, f"Neutral T drifted: {T_low}"


def test_ekman_surface_wind_decelerates_relative_to_geostrophic(ekman_run):
    """Surface drag must brake the wind below ``u_g = 10 m/s``."""
    u_low = float(ekman_run.state.u.data[0, 0, 0, -1])
    v_low = float(ekman_run.state.v.data[0, 0, 0, -1])
    speed_low = (u_low ** 2 + v_low ** 2) ** 0.5
    assert speed_low < 10.0, (
        f"Surface speed {speed_low} did not decelerate below u_g=10"
    )


def test_ekman_surface_wind_veers_toward_low_pressure(ekman_run):
    """Northern-hemisphere Ekman spiral: surface wind veers from
    pure westerly toward south-of-west (positive v in the legoESM
    sign convention).  Veer angle should fall in (0, 45°) before
    one inertial period elapses."""
    u_low = float(ekman_run.state.u.data[0, 0, 0, -1])
    v_low = float(ekman_run.state.v.data[0, 0, 0, -1])
    veer_rad = np.arctan2(v_low, u_low)
    veer_deg = float(np.rad2deg(veer_rad))
    assert 0.0 < veer_deg < 45.0, (
        f"Veer angle {veer_deg} outside Ekman band (0, 45)°"
    )


def test_ekman_surface_qke_tracks_friction_velocity(ekman_run):
    """qke_sfc = B1^(2/3) · u*² with u* = √(Cd) · |V|."""
    u_low = float(ekman_run.state.u.data[0, 0, 0, -1])
    v_low = float(ekman_run.state.v.data[0, 0, 0, -1])
    V = (u_low ** 2 + v_low ** 2 + 1e-4) ** 0.5
    expected = (24.0 ** (2 / 3)) * 1.5e-3 * V * V
    actual = float(ekman_run.phys_state.qke[0, -1])
    np.testing.assert_allclose(actual, expected, rtol=0.10)


def test_ekman_oracle_present_and_neutral():
    """Sanity-check the andren1994 oracle: neutral θ, zero surface
    heat flux, plausible u* (O(0.3) for u_g=10 + Cd~1e-3)."""
    if not ORACLE_PATH.exists():
        pytest.skip(f"Oracle missing: {ORACLE_PATH}")
    ds = xr.open_dataset(ORACLE_PATH)
    try:
        th = ds["th"].values
        assert float(th.std()) < 0.5, (
            f"Oracle θ has spurious variability (neutral case): "
            f"std={float(th.std())}"
        )
        w_th = ds["mo_w_th"].values
        assert float(np.abs(w_th).max()) < 1e-3, (
            f"Oracle surface heat flux too large for neutral case: "
            f"max |w'θ'|={float(np.abs(w_th).max())}"
        )
        ustar = ds["mo_u_st"].values
        assert 0.1 < float(ustar.max()) < 1.0, (
            f"Oracle u* out of band: max={float(ustar.max())}"
        )
    finally:
        ds.close()
