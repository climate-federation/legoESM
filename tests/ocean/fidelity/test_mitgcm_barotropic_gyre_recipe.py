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


def test_config_pins_mitgcm_faithful_unsplit_numerics():
    """The recipe selects MITgcm's exact unsplit numerics: explicit-AB2 face-f
    Coriolis (abEps=0.01) + 2nd-order centered flux-form advection + fully-
    backward implicit free surface.  Pin them so a silent config drift (e.g. back
    to the misdiagnosed Sadourny vertex-f / upwind) is caught."""
    c = gyre.build_gyre_config()
    assert c.coriolis_scheme == "explicit_ab2"
    assert c.outer_integrator == "ab2"
    assert c.ab2_epsilon == 0.01
    assert c.coriolis_energy_conserving is False        # MITgcm face-f, not Sadourny
    assert c.momentum_advection == "flux_form"
    assert c.momentum_flux_scheme == "centered"          # MITgcm 2nd-order centered
    assert c.barotropic_solver == "implicit_cn"
    assert c.barotropic_implicit_theta_eta == 1.0
    assert c.barotropic_implicit_theta_pgf == 1.0


def test_geometry_uses_natural_pseudo_lat_with_stored_metric():
    """#514: the gyre geometry no longer needs the Cartesian (lat=0) pseudo-lat
    workaround.  The C-grid operators READ the stored uniform v-face metric
    (``dx_v == dx_m``) instead of recomputing ``cos(grid.lat_v)``, so the
    energy-conserving implicit free surface holds with the geometry's NATURAL
    nonzero pseudo-lat — the ~1% metric leak that ran this gyre turbulent is
    gone at the source.  The laminar dipole (test_mitgcm_gyre_canonical) confirms
    the dynamics stay laminar with this geometry."""
    g = gyre.build_gyre_geometry()
    # Natural pseudo-lat is now nonzero (the lat=0 workaround is removed).
    assert float(np.abs(np.asarray(g.lat)).max()) > 0.0
    # The metric the operators actually read is the uniform Cartesian dx_m.
    np.testing.assert_allclose(np.asarray(g.dx_v), gyre.DX_M)


def test_explicit_ab2_rejects_energy_conserving_coriolis():
    """Guard (adversarial-review MAJOR): explicit_ab2 already routes the FACE-f
    Coriolis through du_dt -> F_slow, and the implicit_cn solver only gates its
    OWN face-f FB term (_cori_fac=0).  Enabling the vertex-f energy-conserving
    barotropic Coriolis on top would double-count (and mix vertex-f barotropic
    with face-f du_dt).  Model construction must reject the combo, not silently
    double-count."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    r = gyre.build_gyre_recipe()
    bad = r.config._replace(coriolis_energy_conserving=True)  # + explicit_ab2 from recipe
    with pytest.raises(ValueError, match="coriolis_energy_conserving"):
        LatLonCGridOceanModel(r.geometry, r.z_coord, bad)


@pytest.mark.slow
def test_recipe_spin_up_stays_laminar():
    """End-to-end energy consequence of the metric fix: the FULL production
    LatLonCGridOceanModel on the recipe spins up LAMINAR toward MITgcm's
    |u|max≈0.031 / |v|max≈0.084 instead of overshooting to the turbulent
    0.15-0.37 attractor.  Marked slow (a few thousand steps); run with
    ``-m slow``.  Before the iteration-7 metric fix this reached ~0.05+ and
    climbed; after, it plateaus in the laminar band."""
    import os
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax
    from functools import partial
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    r = gyre.build_gyre_recipe()
    model = LatLonCGridOceanModel(r.geometry, r.z_coord, r.config)
    s = r.state
    model._ensure_vertex_mask(s)

    @jax.jit
    def chunk(state):
        def body(st, _):
            return model._step_impl(st, r.dt_s, surface_forcing=r.wind_forcing), None
        st, _ = jax.lax.scan(body, state, None, length=1000)
        return st

    for _ in range(6):                                   # 6000 steps (~0.23 yr)
        s = chunk(s)
    u = np.asarray(s.u.data); v = np.asarray(s.v.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
    umax = float(np.abs(u).max()); vmax = float(np.abs(v).max())
    # Laminar: well inside the turbulent band (which is 0.15-0.37). MITgcm = 0.031.
    assert umax < 0.045, f"|u|max={umax:.4f}: not laminar (turbulent overshoot)"
    assert 0.02 < umax, f"|u|max={umax:.4f}: gyre failed to spin up"
    assert vmax < 0.11, f"|v|max={vmax:.4f}: not laminar"


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
    # With the audit-settled config (fully-backward-Euler free surface theta=1.0,
    # flux-form momentum/viscosity) the 10-step pattern is near-perfect.
    assert corr > 0.999, f"eta pattern correlation {corr:.4f} too low"
    # Magnitude within ~15% (residual = time-scheme transient + free-slip vs
    # MITgcm no-slip walls, which is inactive over a 10-step spin-up).
    ratio = np.abs(lego_eta).max() / np.abs(m_eta).max()
    assert 0.85 < ratio < 1.15, f"eta magnitude ratio {ratio:.3f} off"
