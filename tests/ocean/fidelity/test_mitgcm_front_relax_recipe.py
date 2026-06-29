"""Tests for the legoESM-MITgcm front_relax baroclinic oracle.

Construction + analytic-IC checks always run; the monitor-tier oracle match
against MITgcm's shipped ``results/output.txt`` dynstat (the production implicit
free-surface solver on STRATIFIED geostrophic adjustment) is the headline
``@slow`` test.
"""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.ocean.fidelity import mitgcm_front_relax_recipe as fr

# MITgcm front_relax results/output.txt %MON dynstat at timestep 20.
MITGCM_ETA_MAX = 0.0722208208401975
MITGCM_UVEL_MAX = 0.12803771977288
MITGCM_VVEL_MAX = 0.0090404137970896
MITGCM_THETA_MAX = 20.576472287889


def test_geometry_is_fplane_channel():
    g = fr.build_front_relax_geometry()
    assert g.n_lat == fr.NY and g.n_lon == fr.NX
    np.testing.assert_allclose(np.asarray(g.dx_T), fr.DX_M)
    # f-plane: f constant (beta=0), == f0 at every row.
    np.testing.assert_allclose(np.asarray(g.f_T), fr.F0, rtol=1e-9)
    # #514: operators READ the stored uniform dx_v (== dx_m); the geometry's
    # natural pseudo-lat no longer enters the metric, so the lat=0 workaround is
    # gone and dx_v is the uniform Cartesian width the implicit FS relies on.
    np.testing.assert_allclose(np.asarray(g.dx_v), fr.DX_M)


def test_vertical_matches_mitgcm_delr():
    z = fr.create_z_star_from_thicknesses(np.asarray(fr.DELR_M))
    np.testing.assert_allclose(np.asarray(z.dz_ref), np.asarray(fr.DELR_M), rtol=1e-9)
    assert abs(float(-np.asarray(z.z_half_ref)[-1]) - fr.HO_M) < 1e-6  # Ho=2460 m


def test_initial_front_structure():
    """The analytic IC is a stratified front: warm over cold (dT/dz>0), with a
    south-to-north horizontal temperature contrast at the surface."""
    ly = fr.DY_M * (fr.NY - 1)
    y_c = -0.5 * ly + (np.arange(fr.NY) + 0.5) * fr.DY_M
    z_c = np.asarray(fr.create_z_star_from_thicknesses(np.asarray(fr.DELR_M)).z_full_ref)
    temp = fr.front_relax_initial_temperature(y_c, z_c)     # (ny, nz)
    # Stable stratification: surface (k=0) warmer than depth (k=-1) everywhere.
    assert np.all(temp[:, 0] > temp[:, -1])
    # A horizontal front at the surface (sin(pi Y/Ly) gives a S-N contrast).
    assert abs(temp[:, 0].max() - temp[:, 0].min()) > 0.5


def test_config_rejects_unknown_viscosity():
    with pytest.raises(ValueError, match="lateral_viscosity"):
        fr.build_front_relax_config(lateral_viscosity="bogus")


def test_config_is_faithful_baroclinic():
    c = fr.build_front_relax_config()              # faithful biharmonic default
    assert c.eos == "linear"
    assert c.eos_linear.alpha_T == fr.T_ALPHA and c.eos_linear.beta_S == 0.0
    assert c.lateral_viscosity.B_h == fr.VISC_A4 and c.lateral_viscosity.A_h == 0.0    # MITgcm viscA4 biharmonic
    assert c.A_v == fr.VISC_AR and c.K_v == fr.DIFF_KR_T
    assert c.lateral_side_bc == "free_slip"
    assert c.barotropic.barotropic_solver == "implicit_cn"
    assert c.implicit_vertical_mixing is True


@pytest.mark.slow
def test_baroclinic_adjustment_matches_mitgcm_monitor():
    """Monitor-tier oracle: the production implicit free-surface solver reproduces
    MITgcm's 20-step baroclinic geostrophic adjustment. eta_max within 8% and
    uvel_max (the thermal-wind jet) within 10% of MITgcm's dynstat — validating
    the implicit_cn solver on STRATIFIED flow (the single-layer gyre cannot).

    Uses the FAITHFUL MITgcm biharmonic (component del4), now STABLE via the
    ``flux_divergence`` component biharmonic — the vector grad(div)-curl(curl)
    biharmonic was ill-scaled here and was the prior instability."""
    import os
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    r = fr.build_front_relax_recipe()              # faithful biharmonic, stable
    model = LatLonCGridOceanModel(r.geometry, r.z_coord, r.config)
    s = r.state
    for _ in range(20):
        s = model.step(s, r.dt_s)
    u = np.asarray(s.u.data)
    eta = np.asarray(s.eta.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(eta))
    eta_max = float(np.abs(eta).max())
    uvel_max = float(np.abs(u).max())
    # Free-surface amplitude: within 8% of MITgcm (the FS solver under test).
    assert abs(eta_max - MITGCM_ETA_MAX) / MITGCM_ETA_MAX < 0.08, \
        f"eta_max {eta_max:.4f} vs MITgcm {MITGCM_ETA_MAX:.4f}"
    # Thermal-wind jet: within 10%.
    assert abs(uvel_max - MITGCM_UVEL_MAX) / MITGCM_UVEL_MAX < 0.10, \
        f"uvel_max {uvel_max:.4f} vs MITgcm {MITGCM_UVEL_MAX:.4f}"
