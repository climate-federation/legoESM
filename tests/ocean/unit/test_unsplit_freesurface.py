"""Tests for the MITgcm-faithful UNSPLIT implicit free surface
(``barotropic_solver="implicit_unsplit"``) — the fix for the spurious 2dx
baroclinic instability the split free surface produces (the long-run
baroclinic-gyre checkerboard). See docs/ocean_fidelity/
mitgcm_unsplit_freesurface_fix.md.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")


def _zig(a):
    a = np.asarray(a)
    a = a[..., 0] if a.ndim == 3 else a
    s = 0.5 * (np.roll(a, 1, 1) + np.roll(a, -1, 1))
    d = a - s
    return float(np.sqrt((d[1:-1, 1:-1] ** 2).mean())
                 / (np.sqrt((a[1:-1, 1:-1] ** 2).mean()) + 1e-30))


def test_solve_unsplit_freesurface_reduces_divergence():
    """The unsplit solve + uniform correction makes the corrected depth-integrated
    transport ~non-divergent (the implicit free-surface kinematic constraint).
    Non-vacuous: the predictor (pre-correction) is strongly divergent."""
    import jax.numpy as jnp
    from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
    from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
        solve_unsplit_freesurface,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface,
        min_cell_to_vface,
    )
    from legoesm.ocean.fidelity import mitgcm_baroclinic_gyre_recipe as bg
    from legoesm.ocean.vertical import compute_layer_thickness

    r = bg.build_baroclinic_gyre_recipe()
    g = r.geometry
    st = r.state
    nz = len(np.asarray(r.z_coord.dz_ref))
    um = st.u_mask.data
    vm = st.v_mask.data
    cm = st.land_mask.data
    # a divergent predictor: warm-south-cold-north-driven zonal jet + a kick
    rng = np.random.default_rng(0)
    u_star = jnp.asarray(0.1 * rng.standard_normal((g.n_lat, g.n_lon + 1, nz))) * um[..., None]
    v_star = jnp.asarray(0.1 * rng.standard_normal((g.n_lat + 1, g.n_lon, nz))) * vm[..., None]
    h_k = compute_layer_thickness(st.eta.data, st.H_bathy.data, r.z_coord,
                                  min_water_column_m=r.config.min_water_column_m)
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, g)
    eta_new, u_new, v_new = solve_unsplit_freesurface(
        st.eta.data, u_star, v_star, h_u, h_v, r.dt_s, r.config.g, g, cm, um, vm)
    assert np.all(np.isfinite(np.asarray(eta_new)))
    assert np.all(np.isfinite(np.asarray(u_new)))
    # depth-integrated divergence: predictor vs corrected (must drop a lot).
    def coldiv(u, v):
        fu = jnp.sum(h_u * u, axis=-1) * um
        fv = jnp.sum(h_v * v, axis=-1) * vm
        return np.asarray(divergence_cgrid(fu, fv, g, u_mask=um, v_mask=vm))
    d_pred = coldiv(u_star, v_star)
    d_corr = coldiv(u_new, v_new)
    m = np.asarray(cm) > 0
    # the implicit FS removes the eta-tendency-balanced divergence: |div_corr| << |div_pred|
    assert np.abs(d_corr[m]).std() < 0.25 * np.abs(d_pred[m]).std()


def test_implicit_unsplit_requires_ab2():
    """Dispatch hardening: implicit_unsplit needs outer_integrator='ab2'."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import mitgcm_baroclinic_gyre_recipe as bg
    r = bg.build_baroclinic_gyre_recipe()
    # matsuno_split Coriolis so the explicit_ab2-Coriolis guard doesn't fire first;
    # the implicit_unsplit-requires-ab2 guard is checked at step time.
    cfg = r.config._replace(barotropic_solver="implicit_unsplit",
                            coriolis_scheme="matsuno_split",
                            outer_integrator="forward_euler")
    m = LatLonCGridOceanModel(r.geometry, r.z_coord, cfg)
    with pytest.raises(ValueError, match="requires.*outer_integrator='ab2'"):
        m.step(r.state, r.dt_s, surface_forcing=r.wind_forcing)


def test_implicit_unsplit_runs_finite():
    """The unsplit path dispatches + runs a few steps finite, stratification bounded."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import mitgcm_baroclinic_gyre_recipe as bg
    r = bg.build_baroclinic_gyre_recipe()
    cfg = r.config._replace(barotropic_solver="implicit_unsplit")
    m = LatLonCGridOceanModel(r.geometry, r.z_coord, cfg)
    s = r.state
    for _ in range(10):
        s = m.step(s, r.dt_s, surface_forcing=r.wind_forcing)
    assert np.all(np.isfinite(np.asarray(s.u.data)))
    assert np.all(np.isfinite(np.asarray(s.eta.data)))
    mask = np.asarray(r.land_mask) > 0
    t = np.asarray(s.T.data)[mask]
    assert 1.5 < float(t.min()) and float(t.max()) < 30.5


@pytest.mark.slow
def test_implicit_unsplit_suppresses_checkerboard():
    """The PROPERTY: the unsplit free surface keeps the velocity SMOOTH over a 30-day
    spin-up at the FAITHFUL A_h=5000/K_h=1000, where the split free surface grows a
    2dx checkerboard to zig~1.4.  Non-vacuous: the split baseline is asserted noisy."""
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity import mitgcm_baroclinic_gyre_recipe as bg
    r = bg.build_baroclinic_gyre_recipe()

    def run30(solver):
        cfg = r.config._replace(barotropic_solver=solver)
        m = LatLonCGridOceanModel(r.geometry, r.z_coord, cfg)
        s = r.state
        for _ in range(2160):
            s = m.step(s, r.dt_s, surface_forcing=r.wind_forcing)
        return _zig(np.asarray(s.u.data))

    zig_split = run30("implicit_cn")
    zig_unsplit = run30("implicit_unsplit")
    assert zig_split > 1.0, f"split baseline should checkerboard, got {zig_split:.3f}"
    assert zig_unsplit < 0.4, f"unsplit should stay smooth, got {zig_unsplit:.3f}"
    assert zig_unsplit < 0.3 * zig_split
