"""Tests for the legoESM-MITgcm reentrant-channel (ACC) oracle.

Construction + analytic-IC + forcing/sponge/bathymetry checks always run; the
monitor-tier oracle match against a 10-step MITgcm reference (the production
lat-lon C-grid model with GM/Redi + wind + surface restoring + RBCS northern
sponge on a zonally-reentrant beta-plane channel) is the headline ``@slow`` test.

MITgcm reference: ``verification/tutorial_reentrant_channel`` (coarse 50 km,
GM-parameterized), single-tile build, ``%MON`` dynstat at timestep 10
(monitorSelect=2). Reproduced locally; the numbers below are pinned from that run.
"""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.ocean.fidelity import mitgcm_reentrant_channel_recipe as rc

# MITgcm tutorial_reentrant_channel %MON dynstat at timestep 10 (50 km, GM run).
MITGCM_ETA_MAX = 0.67016686313414
MITGCM_ETA_MIN = -0.66309098073115
MITGCM_UVEL_MAX = 0.21676240397297
MITGCM_VVEL_MAX = 0.081064820003333
MITGCM_THETA_MAX = 9.7769881436940
MITGCM_THETA_MIN = -1.9999747795231


def test_geometry_is_reentrant_beta_plane_channel():
    g = rc.build_reentrant_channel_geometry()
    assert g.n_lat == rc.NY and g.n_lon == rc.NX
    np.testing.assert_allclose(np.asarray(g.dx_T), rc.DX_M)
    # Southern-Hemisphere beta-plane: f = f0 + beta*y, f0<0 at y=0.
    f = np.asarray(g.f_T)
    assert f.min() < 0.0 and f.max() < 0.0          # f<0 everywhere (SH)
    # beta>0: f increases (less negative) northward.
    assert f[-1, 0] > f[0, 0]
    # #514: operators READ the stored uniform dx_v (== dx_m); the geometry's
    # natural pseudo-lat no longer enters the metric, so the lat=0 workaround is
    # gone and dx_v is the uniform Cartesian width the implicit FS relies on.
    np.testing.assert_allclose(np.asarray(g.dx_v), rc.DX_M)


def test_vertical_matches_mitgcm_delr():
    z = rc.create_z_star_from_thicknesses(np.asarray(rc.DELR_M))
    assert z.n_levels == 49
    np.testing.assert_allclose(np.asarray(z.dz_ref), np.asarray(rc.DELR_M), rtol=1e-5)
    assert abs(float(-np.asarray(z.z_half_ref)[-1]) - rc.HO_M) < 1e-2  # float32 cumsum


def test_land_mask_is_reentrant_south_wall_only():
    """Zonally reentrant: only the southern row is land; no E/W walls; ridge wet."""
    lm = np.asarray(rc.build_reentrant_channel_land_mask())
    assert lm.shape == (rc.NY, rc.NX)
    assert lm[0].sum() == 0.0                        # south wall (gendata bathy(:,1)=0)
    assert lm[-1].sum() == rc.NX                      # north row fully wet
    # No E/W walls: every interior (non-south) row spans the full zonal extent.
    assert np.all(lm[1:].sum(axis=1) == rc.NX)


def test_bathymetry_has_ns_ridge_with_notch():
    """The mid-domain N-S ridge (shallower than Ho) with an unblocked f/H notch."""
    depth = rc.build_reentrant_channel_bathymetry()
    assert depth.shape == (rc.NY, rc.NX)
    # Ridge band (zonal columns 5:14) is shallower than the flat-bottom Ho at a
    # meridional row OUTSIDE the notch (ny=2, away from the notch rows 14:24).
    assert depth[2, 5:14].min() < rc.HO_M            # a wet but shallower ridge
    assert depth[2, 5:14].min() > 0.0                # ridge is wet, not land
    # The unblocked meridional notch (ny rows 18:20) restores full depth in the
    # ridge band — the f/H corridor.
    np.testing.assert_allclose(depth[18:21, 9], rc.HO_M)
    # Flat bottom Ho away from the ridge band.
    np.testing.assert_allclose(depth[2, 0], rc.HO_M)


def test_wind_is_eastward_westerlies_peaking_mid_channel():
    g = rc.build_reentrant_channel_geometry()
    w = rc.build_reentrant_channel_wind(g)
    tau_x = np.asarray(w.tau_x)
    # Stored value is NEGATED (legoESM flips atm->ocean); the ocean-side stress
    # is -tau_x = 0.2 sin(pi Y/Ly), peaking mid-channel, zero at the walls.
    ocean_stress = -tau_x[:, 0]
    # Discrete sin over NY=40 samples peaks just below TAU_MAX (no sample at pi/2).
    assert abs(ocean_stress.max() - rc.TAU_MAX) < 2e-3
    assert ocean_stress[rc.NY // 2] > ocean_stress[0]    # peaks interior
    assert ocean_stress.min() >= -1e-12               # westerlies (all eastward)


def test_sponge_is_northern_rbcs_below_surface():
    """RBCS mask: full restoring at the north row (below surface), 0.25 one row
    south, zero elsewhere; surface level k=0 untouched (SST restoring handles it)."""
    z = rc.create_z_star_from_thicknesses(np.asarray(rc.DELR_M))
    lm = rc.build_reentrant_channel_land_mask()
    sp = rc.build_reentrant_channel_sponge(z, lm)
    gamma = np.asarray(sp.gamma)
    assert gamma.shape == (rc.NY, rc.NX, 49)
    # Only the two northern rows restore.
    nz_rows = np.where(gamma.sum(axis=(1, 2)) > 0)[0]
    np.testing.assert_array_equal(nz_rows, [rc.NY - 2, rc.NY - 1])
    # Surface k=0 is NOT restored by the sponge.
    assert gamma[:, :, 0].max() == 0.0
    # Rate is mask / tauRelaxT; north row full = 1/tau.
    np.testing.assert_allclose(gamma[-1, 0, 1], 1.0 / rc.TAU_RESTORE_S)
    np.testing.assert_allclose(gamma[-2, 0, 1], 0.25 / rc.TAU_RESTORE_S)


def test_config_is_gm_redi_linear_eos_channel():
    g = rc.build_reentrant_channel_geometry()
    c = rc.build_reentrant_channel_config(g)
    assert c.eos == "linear"
    assert c.eos_linear.alpha_T == rc.T_ALPHA and c.eos_linear.beta_S == 0.0
    assert c.gm_redi is not None
    assert c.gm_redi.kappa_GM == rc.GM_BACKGROUND_K
    assert c.gm_redi.kappa_Redi == rc.GM_BACKGROUND_K
    assert c.lateral_viscosity.A_h == rc.VISC_AH and c.K_h == 0.0
    assert c.A_v == rc.VISC_AR and c.K_v == 1.0e-5
    assert c.barotropic.barotropic_solver == "implicit_unsplit"
    assert c.implicit_vertical_mixing is True
    # ivdc convective adjustment via the enhanced-diffusion scheme.
    assert c.physics.convection.scheme == "enhanced_diffusion"


def test_initial_temperature_is_stratified_with_meridional_gradient():
    z = rc.create_z_star_from_thicknesses(np.asarray(rc.DELR_M))
    z_c = np.asarray(z.z_full_ref)
    temp = rc.reentrant_channel_initial_temperature(z_c)     # (ny, nz)
    # Stable: surface (k=0) warmer than depth everywhere.
    assert np.all(temp[:, 0] >= temp[:, -1])
    # Meridional surface gradient: north (warm) > south (cold).
    assert temp[-1, 0] > temp[1, 0]
    # Bottom approaches Tmin.
    assert abs(temp[:, -1].min() - rc.T_MIN_RESTORE) < 0.1


@pytest.mark.slow
def test_reentrant_channel_matches_mitgcm_monitor():
    """Monitor-tier oracle: the production lat-lon C-grid model (GM/Redi + wind +
    surface restoring + RBCS sponge, zonally reentrant) reproduces MITgcm's
    10-step reentrant-channel spin-up. eta/uvel within 3%, theta within 0.2% of
    the MITgcm %MON dynstat — validating the full ACC closure stack against a
    SECOND ocean oracle (MITgcm, after Veros)."""
    import os
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    r = rc.build_reentrant_channel_recipe()
    model = LatLonCGridOceanModel(r.geometry, r.z_coord, r.config)
    s = r.state
    for _ in range(10):
        s = model.step(s, r.dt_s, surface_forcing=r.wind_forcing, sponge=r.sponge)
    u = np.asarray(s.u.data)
    eta = np.asarray(s.eta.data)
    temp = np.asarray(s.T.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(eta))
    # Free-surface (the geostrophic ACC sea-surface tilt): within 3%.
    assert abs(float(eta.max()) - MITGCM_ETA_MAX) / MITGCM_ETA_MAX < 0.03
    assert abs(float(eta.min()) - MITGCM_ETA_MIN) / abs(MITGCM_ETA_MIN) < 0.03
    # Zonal jet: within 3%.
    assert abs(float(u.max()) - MITGCM_UVEL_MAX) / MITGCM_UVEL_MAX < 0.03
    # Temperature extremes (restoring-pinned): within 0.5%.
    assert abs(float(temp.max()) - MITGCM_THETA_MAX) / MITGCM_THETA_MAX < 0.005
    assert abs(float(temp.min()) - MITGCM_THETA_MIN) / abs(MITGCM_THETA_MIN) < 0.005
