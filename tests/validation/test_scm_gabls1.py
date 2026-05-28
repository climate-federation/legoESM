"""GABLS1 benchmark for the legoESM single-column model.

Integrates the legoESM SCM on the Cuxart et al. (2006) GABLS1 setup
(see :mod:`scripts.run_scm_gabls1`) and compares the resulting
trajectory against the jax_scm oracle stored at
``tests/validation/scm_oracle/gabls1_Nz64.nc``.

Phase F deferred-item #4 (profile-level RMSE via interpolation) is
addressed: ``test_gabls1_profile_rmse_within_band`` interpolates the
legoESM θ column onto the oracle's geometric-height grid and asserts
RMSE below a band that absorbs the constants drift + vertical-coord
mismatch + simplified initial profile.

What this test checks:

* Run completes without NaN over the test window (2 h vs the full
  9-h spec — the SCM driver is not jit-compiled so a longer window
  costs minutes per test; the surface-flux balance is diagnostic
  within 2 h).
* Lowest air temperature in the stable BL band; warmer than T_s.
* Surface qke ≈ ``B1^(2/3) · u*²`` (MY82 surface BC).
* Wind veers from the initial pure-westerly profile (Ekman veer).
* Oracle NetCDF surface-state magnitudes plausible.
* Profile-level θ RMSE vs oracle below 5 K (deferred-item #4).
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
    / "gabls1_Nz64.nc"
)

TEST_HOURS = 2.0
TEST_DT = 5.0
TEST_NLEV = 32
TEST_SIGMA_TOP = 0.7


def _import_runner():
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    import run_scm_gabls1
    return run_scm_gabls1


@pytest.fixture(scope="module")
def gabls1_run():
    """One shared GABLS1 SCM run — driven once per test session."""
    runner = _import_runner()
    scm = runner.build_scm(
        nlev=TEST_NLEV, dt=TEST_DT, sigma_top=TEST_SIGMA_TOP,
    )
    scm.run(nsteps=int(TEST_HOURS * 3600 / TEST_DT))
    return scm


def test_gabls1_finite(gabls1_run):
    """Run completes without NaN/inf at any level."""
    scm = gabls1_run
    for arr in (
        np.asarray(scm.state.T.data),
        np.asarray(scm.state.u.data),
        np.asarray(scm.state.v.data),
        np.asarray(scm.phys_state.qke),
    ):
        assert np.all(np.isfinite(arr))


def test_gabls1_lowest_cell_in_stable_band(gabls1_run):
    """Lowest cell in stable BL band; warmer than prescribed T_s(t_end)."""
    T_low = float(gabls1_run.state.T.data[0, 0, 0, -1])
    assert 255.0 < T_low < 270.0, (
        f"GABLS1 lowest cell out of band: T_low={T_low}"
    )
    T_s_end = 265.0 + (-0.25 / 3600.0) * (TEST_HOURS * 3600.0)
    assert T_low > T_s_end - 1.0, (
        f"Lowest air {T_low} colder than T_s {T_s_end} by >1 K — "
        "unphysical SBL"
    )


def test_gabls1_surface_qke_matches_friction_velocity_bc(gabls1_run):
    """qke_sfc ≈ B1^(2/3) · u*² with u* = √(Cd) · |V| (constant-Cd)."""
    u_low = float(gabls1_run.state.u.data[0, 0, 0, -1])
    v_low = float(gabls1_run.state.v.data[0, 0, 0, -1])
    V = np.sqrt(u_low ** 2 + v_low ** 2 + 1e-4)
    expected = (24.0 ** (2 / 3)) * 1.5e-3 * V * V
    actual = float(gabls1_run.phys_state.qke[0, -1])
    np.testing.assert_allclose(actual, expected, rtol=0.10)


def test_gabls1_wind_veers_under_coriolis(gabls1_run):
    """Pure-westerly init wind must veer toward +v under Coriolis +
    surface friction (Ekman-veer signature)."""
    v_low = float(gabls1_run.state.v.data[0, 0, 0, -1])
    assert v_low > 0.0, f"No Ekman veer: v_low={v_low}"


def test_gabls1_profile_rmse_within_band(gabls1_run):
    """Phase F deferred-item #4: profile-level θ RMSE vs jax_scm
    oracle.  Interpolates the legoESM θ column onto the oracle's
    geometric-height z-grid via ``np.interp``; asserts RMSE < 5 K.

    Tolerance absorbs the documented oracle-parity gap:
      * constants drift ~0.3 % (g, R_d, c_pd, L_v)
      * Simplified initial-θ profile vs Cuxart spec
      * Different vertical staggering (sigma-p vs geometric height)
      * Forward Euler vs Crank-Nicolson outer integration
    """
    if not ORACLE_PATH.exists():
        pytest.skip(f"Oracle missing: {ORACLE_PATH}")
    scm = gabls1_run
    p_s = float(scm.state.p_s.data[0, 0, 0])
    p_full = np.asarray(
        scm.sigma_coord.pressure_at_full(scm.state.p_s.data)[0, 0, 0]
    )
    z_lego = -8.0e3 * np.log(np.maximum(p_full / p_s, 1e-6))  # top→bottom
    from legoesm import constants as _c
    exner_lego = (p_full / _c.p_ref) ** _c.kappa
    T_lego = np.asarray(scm.state.T.data[0, 0, 0])
    theta_lego = T_lego / np.maximum(exner_lego, 1e-6)
    z_lego_asc = z_lego[::-1]
    theta_lego_asc = theta_lego[::-1]

    ds = xr.open_dataset(ORACLE_PATH)
    try:
        t_idx = int(np.argmin(np.abs(ds["time"].values - TEST_HOURS)))
        z_oracle = ds["z"].values
        theta_oracle = ds["th"].values[t_idx]
    finally:
        ds.close()
    z_min = float(z_lego_asc.min())
    z_max = float(z_lego_asc.max())
    mask = (z_oracle >= z_min) & (z_oracle <= z_max)
    if mask.sum() < 5:
        pytest.skip("Insufficient z-overlap between legoESM and oracle grids")
    theta_lego_on_oracle = np.interp(
        z_oracle[mask], z_lego_asc, theta_lego_asc,
    )
    residual = theta_lego_on_oracle - theta_oracle[mask]
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    max_abs = float(np.max(np.abs(residual)))

    # Tight band: observed RMSE is ~0.14 K at TEST_HOURS=2 h.  Codex
    # Phase F fix #4 review noted that a 5 K threshold passes for a
    # flat 263 K column (RMSE 3.35 K), defeating the parity gate.
    # 1.0 K rejects flat-column failure modes while leaving margin
    # for legitimate vertical-coord / constants drift.
    assert rmse < 1.0, (
        f"GABLS1 profile θ RMSE vs oracle at t≈{TEST_HOURS}h: "
        f"{rmse:.3f} K (band: < 1 K).  Drivers of any gap: "
        "vertical-coord mismatch, simplified init, constants drift."
    )
    # Shape-sensitive guard: max-abs error catches a localised
    # mismatch (e.g. lost inversion structure) that an averaged RMSE
    # would smooth out.  3 K leaves headroom for the inversion edge
    # where the legoESM linear interp + sigma stretching can amplify
    # small θ-grid differences.
    assert max_abs < 3.0, (
        f"GABLS1 profile θ max-abs vs oracle at t≈{TEST_HOURS}h: "
        f"{max_abs:.3f} K (band: < 3 K).  A high max-abs at low "
        "RMSE signals lost vertical structure (e.g. inversion edge)."
    )
    # Note: the masked overlap typically starts around z=37 m
    # because the lowest legoESM full level at (TEST_NLEV=32,
    # TEST_SIGMA_TOP=0.7) sits at ~37.6 m via -8000·log(σ_low).
    # The oracle's z=3.125-30 m surface layer is therefore excluded
    # from this RMSE — surface-layer parity is exercised by the
    # scalar `test_gabls1_surface_qke_matches_friction_velocity_bc`
    # and `test_gabls1_lowest_cell_in_stable_band` checks.  A
    # high-resolution near-surface profile comparison is a follow-up.


def test_gabls1_oracle_present_and_plausible_magnitudes():
    """Sanity gate: oracle NetCDF surface-state magnitudes are
    physically plausible.  Catches a corrupted or stale oracle file
    before benchmark comparison runs."""
    if not ORACLE_PATH.exists():
        pytest.skip(
            f"Oracle missing: {ORACLE_PATH}.  Run scripts/run_jax_scm_oracle.py"
        )
    ds = xr.open_dataset(ORACLE_PATH)
    try:
        ustar = ds["mo_u_st"].values
        assert np.all(np.isfinite(ustar))
        assert 0.05 < float(ustar.max()) < 1.0, (
            f"Oracle u* range out of band: max={float(ustar.max())}"
        )
        th_s = ds["mo_th_s"].values
        delta = float(th_s[0] - th_s[-1])
        assert 1.5 < delta < 3.0, (
            f"Oracle surface theta cooling not in GABLS1 band: "
            f"{delta} K (expected ~2.25 K over full 9-h spec)"
        )
    finally:
        ds.close()
