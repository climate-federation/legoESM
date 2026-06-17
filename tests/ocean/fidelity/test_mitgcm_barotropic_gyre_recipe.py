"""Tests for the legoESM-MITgcm barotropic-gyre recipe.

Construction/forcing/run checks always run; the oracle pattern-match against a
real MITgcm reference is opt-in (``$LEGOESM_OCEAN_FIDELITY_MITGCM_REF``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from legoesm.ocean.fidelity import mitgcm_barotropic_gyre_recipe as gyre

_REF_ROOT = os.environ.get("LEGOESM_OCEAN_FIDELITY_MITGCM_REF")
_CASE_DIR = Path(_REF_ROOT) / "barotropic_gyre" if _REF_ROOT else None


def test_wind_profile_matches_mitgcm_gendata():
    """tau_x(y) = -tauMax cos(pi Y), Y=(j-0.5)/(ny-2): -0.1 south, ~0 mid, +0.1 N."""
    tx = gyre.gyre_taux_ocean_profile()
    assert tx.shape == (gyre.NY,)
    np.testing.assert_allclose(tx[1], -0.1, atol=2e-3)       # near south wall
    np.testing.assert_allclose(tx[-2], 0.1, atol=2e-3)       # near north wall
    assert abs(tx[gyre.NY // 2]) < 0.01                      # ~0 mid-domain


def test_closed_box_land_mask():
    m = np.asarray(gyre.build_gyre_land_mask())
    assert m.shape == (gyre.NY, gyre.NX)
    assert int((m == 0).sum()) == 244  # 62*4 - 4 corners, matches bathy.bin
    assert m[1:-1, 1:-1].all()         # interior all ocean
    assert not m[0, :].any() and not m[:, 0].any()  # walls


def test_geometry_uses_beta_plane_with_mitgcm_params():
    g = gyre.build_gyre_geometry()
    assert g.n_lat == gyre.NY and g.n_lon == gyre.NX
    np.testing.assert_allclose(np.asarray(g.dx_T), gyre.DX_M)
    # f at the southern cell-centre row = f0 + beta*(y0 + dy/2).
    y_c0 = gyre.Y_ORIGIN_M + 0.5 * gyre.DY_M
    np.testing.assert_allclose(
        float(g.f_T[0, 0]), gyre.F0 + gyre.BETA * y_c0, rtol=1e-9
    )


def test_config_is_barotropic_constant_density():
    c = gyre.build_gyre_config()
    assert c.A_h == gyre.VISC_AH
    assert c.bottom_drag_r == 0.0
    assert c.gm_redi is None
    assert c.eos == "linear"
    assert c.eos_linear.alpha_T == 0.0 and c.eos_linear.beta_S == 0.0
    assert c.rho_0 == gyre.RHO_CONST


def test_recipe_runs_and_spins_up_a_gyre():
    """10 steps from rest with the MITgcm wind: a free-surface dipole develops,
    velocities at the O(1e-4 m/s) scale of the reference, all finite."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    r = gyre.build_gyre_recipe()
    model = LatLonCGridOceanModel(r.geometry, r.z_coord, r.config)
    s = r.state
    for _ in range(10):
        s = model.step(s, r.dt_s, surface_forcing=r.wind_forcing)
    eta, u, v = np.asarray(s.eta.data), np.asarray(s.u.data), np.asarray(s.v.data)
    assert np.all(np.isfinite(eta)) and np.all(np.isfinite(u))
    # Wind-curl forcing tilts the free surface into a dipole (min<0<max).
    assert eta.min() < 0.0 < eta.max()
    # Same order as MITgcm (u,v ~ 1e-4 m/s after 10 steps), not a blow-up.
    assert 1e-5 < np.abs(u).max() < 1e-2
    assert 1e-5 < np.abs(v).max() < 1e-2


@pytest.mark.skipif(
    _CASE_DIR is None or not _CASE_DIR.is_dir(),
    reason="no generated MITgcm barotropic_gyre reference",
)
def test_oracle_pattern_match_against_real_mitgcm():
    """legoESM reproduces the MITgcm free-surface pattern at iteration 10."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import mitgcm_runner

    r = gyre.build_gyre_recipe()
    model = LatLonCGridOceanModel(r.geometry, r.z_coord, r.config)
    s = r.state
    for _ in range(10):
        s = model.step(s, r.dt_s, surface_forcing=r.wind_forcing)
    lego_eta = np.asarray(s.eta.data)

    ref = mitgcm_runner.load_mitgcm_reference("barotropic_gyre")
    m_eta = np.asarray(ref.variables["Eta"])
    li = lego_eta[1:-1, 1:-1].ravel()
    mi = m_eta[1:-1, 1:-1].ravel()
    corr = float(np.corrcoef(li, mi)[0, 1])
    assert corr > 0.9, f"eta pattern correlation {corr:.3f} too low"
    # Magnitude within a factor of ~1.5 (solver/placement residuals aside).
    ratio = np.abs(lego_eta).max() / np.abs(m_eta).max()
    assert 0.5 < ratio < 2.0, f"eta magnitude ratio {ratio:.2f} off"
