"""Tests for the legoESM-MITgcm baroclinic-gyre oracle.

Construction + analytic-forcing checks always run; the monitor-tier oracle match
against MITgcm's shipped ``results/output.txt`` dynstat — the production implicit
free-surface solver on a FORCED, STRATIFIED, spherical-grid spin-up — is the
``@slow`` test.
"""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.ocean.fidelity import mitgcm_baroclinic_gyre_recipe as bg

# MITgcm tutorial_baroclinic_gyre %MON dynstat at timestep 10 (on-box rebuild,
# globalFiles, dumpInitAndLast).  The %MON ``dynstat_*_max`` monitors are the
# SIGNED maxima (max of the signed field, not max-abs) — verified against the
# field dumps: dynstat_uvel_max == U.max() == 0.018792 exactly, while the field
# max|U| = 0.02295 is a larger NEGATIVE excursion.  Compare signed-max to
# signed-max (the monitor convention).
MITGCM_ETA_MAX = 0.00844351802125882
MITGCM_UVEL_MAX = 0.018792360955067
MITGCM_VVEL_MAX = 0.016034431643962


def test_grid_is_spherical_walled_box():
    g, mask = bg.build_baroclinic_gyre_grid()
    assert g.n_lat == bg.NY_INT + 2 and g.n_lon == bg.NX_INT + 2   # +1-cell wall ring
    # 60x60 ocean interior; spherical mid-latitude band 14.5-75.5N.
    assert int(np.asarray(mask).sum()) == bg.NX_INT * bg.NY_INT
    lat = np.degrees(np.asarray(g.lat))
    np.testing.assert_allclose([lat.min(), lat.max()], [14.5, 75.5], atol=1e-6)
    # Real spherical Coriolis (f = 2 Omega sin lat), positive in the NH.
    assert float(np.asarray(g.f).min()) > 0.0


def test_vertical_matches_mitgcm_delr():
    z = bg.create_z_star_from_thicknesses(np.asarray(bg.DELR_M))
    np.testing.assert_allclose(np.asarray(z.dz_ref), np.asarray(bg.DELR_M), rtol=1e-9)
    assert abs(float(-np.asarray(z.z_half_ref)[-1]) - bg.HO_M) < 1e-6   # Ho=1800


def test_wind_is_a_double_gyre():
    """tau_x(Y) = -tauMax cos(2pi (Y-15)/60): easterly at the N/S edges, westerly
    mid-domain (a subtropical+subpolar double gyre)."""
    g, _ = bg.build_baroclinic_gyre_grid()
    tau = np.asarray(bg.baroclinic_gyre_wind(g).tau_x)[:, 0]   # passed value (negated ocean)
    lat = np.degrees(np.asarray(g.lat))
    # Ocean stress = -tau_passed; reconstruct and check the double-gyre sign pattern.
    tau_ocean = -tau
    south = np.argmin(np.abs(lat - 15.5))
    mid = np.argmin(np.abs(lat - 45.0))
    assert tau_ocean[south] < 0.0 < tau_ocean[mid]               # easterly S, westerly mid
    np.testing.assert_allclose(np.abs(tau_ocean).max(), bg.TAU_MAX, rtol=0.02)


def test_restoring_target_is_linear_in_latitude():
    g, _ = bg.build_baroclinic_gyre_grid()
    t_star = np.asarray(bg.baroclinic_gyre_t_star(g))[:, 0]
    lat = np.degrees(np.asarray(g.lat))
    # Trest = 0.5 (75 - Y): ~29.75 at the south, ~-0.25 at the north.
    np.testing.assert_allclose(t_star, 0.5 * (75.0 - lat), atol=1e-6)


def test_initial_stratification_is_tref():
    r = bg.build_baroclinic_gyre_recipe()
    temp = np.asarray(r.state.T.data)
    mask = np.asarray(r.land_mask) > 0
    # Horizontally uniform tRef profile (30 surface -> 2 bottom), stable.
    surf = temp[..., 0][mask]
    bot = temp[..., -1][mask]
    np.testing.assert_allclose(surf, 30.0, atol=1e-6)
    np.testing.assert_allclose(bot, 2.0, atol=1e-6)


def test_config_is_faithful_baroclinic():
    r = bg.build_baroclinic_gyre_recipe()
    c = r.config
    assert c.eos == "linear" and c.eos_linear.alpha_T == bg.T_ALPHA
    assert c.A_h == bg.VISC_AH and c.lateral_viscosity_operator == "flux_divergence"
    assert c.lateral_side_bc == "no_slip" and c.K_h == bg.DIFF_KH_T
    assert c.barotropic.barotropic_solver == "implicit_unsplit"   # MITgcm-faithful unsplit FS
    assert c.physics.convection.scheme == "enhanced_diffusion"
    assert c.physics.surface_forcing.scheme == "restoring"


@pytest.mark.slow
def test_baroclinic_gyre_matches_mitgcm_monitor():
    """Monitor-tier oracle: the production implicit free surface reproduces MITgcm's
    10-step wind+buoyancy-driven baroclinic double-gyre spin-up on a spherical grid.

    Compared against MITgcm's ``%MON dynstat_*_max`` SIGNED maxima (the monitor
    convention): ``eta_max`` matches to ~1%, ``uvel_max`` to ~3%, ``vvel_max`` to
    ~10%.  (A field-tier dump comparison gives row-by-row max|u| agreement of
    1-5% at every latitude and an eta pattern correlation of 0.9989 — see
    ``scripts/tmp/_bgyre_field_compare.py``.)  The ``vvel`` ~10% at step 10 is a
    pure STARTUP TRANSIENT: a 200-step convergence test against MITgcm field dumps
    (``scripts/tmp/_bgyre_vratio_convergence.py``) shows the v.max ratio decays
    1.097 (step10) -> 1.013 (step50) -> 0.996 (step100) -> 0.987 (step200) while
    the v pattern correlation stays 0.999 throughout, so it self-corrects to ~1%
    as the flow spins up — not a structural amplitude error.  Theta stays in
    [2, 30].

    Per-term TENDENCY tier (``scripts/tmp/_bgyre_tendency_tier.py``, against MITgcm
    DIAGNOSTICS_PKG momentum dumps): the decomposition-independent total momentum
    tendency matches MITgcm's ``TOTUTEND`` to corr 0.987 (du/dt) / 0.9996 (dv/dt),
    and the path-independent depth-integrated wind input to corr 0.987 (peak
    identical, tau/rho_0)."""
    import os
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    r = bg.build_baroclinic_gyre_recipe()
    model = LatLonCGridOceanModel(r.geometry, r.z_coord, r.config)
    s = r.state
    for _ in range(10):
        s = model.step(s, r.dt_s, surface_forcing=r.wind_forcing)
    u = np.asarray(s.u.data)
    v = np.asarray(s.v.data)
    eta = np.asarray(s.eta.data)
    temp = np.asarray(s.T.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(eta))
    # MITgcm %MON dynstat_*_max are SIGNED maxima — compare like-for-like.
    eta_max = float(eta.max())
    uvel_max = float(u.max())
    vvel_max = float(v.max())
    # Free-surface amplitude — the solver under test — within 5% of MITgcm.
    # (The MITgcm-faithful UNSPLIT free surface gives ~3.9% at step 10 vs ~0.8% for
    # the old split implicit_cn — a tiny short-run trade for ELIMINATING the long-run
    # 2dx baroclinic checkerboard the split grew, see test_unsplit_freesurface.py.)
    assert abs(eta_max - MITGCM_ETA_MAX) / MITGCM_ETA_MAX < 0.05, \
        f"eta_max {eta_max:.5f} vs MITgcm {MITGCM_ETA_MAX:.5f}"
    # Zonal velocity within 6% (3% actual + margin) of the signed-max monitor.
    assert abs(uvel_max - MITGCM_UVEL_MAX) / MITGCM_UVEL_MAX < 0.06, \
        f"uvel_max {uvel_max:.5f} vs MITgcm {MITGCM_UVEL_MAX:.5f}"
    # Meridional velocity within 15% (10% actual + margin).
    assert abs(vvel_max - MITGCM_VVEL_MAX) / MITGCM_VVEL_MAX < 0.15, \
        f"vvel_max {vvel_max:.5f} vs MITgcm {MITGCM_VVEL_MAX:.5f}"
    # Stratification barely changes over 10 steps (30-day restoring): theta in [2, 30].
    mask = np.asarray(r.land_mask) > 0
    t_interior = temp[mask]
    assert 1.9 < float(t_interior.min()) and float(t_interior.max()) < 30.1
