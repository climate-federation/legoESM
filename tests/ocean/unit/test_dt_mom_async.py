"""Asynchronous dt_mom≠dt_tracer ("distorted-physics") stepping — #44 Stage B.

Veros ACC integrates momentum with dt_mom=4800 and tracers + the clock with
dt_tracer=43200 (ratio 9): momentum() runs ONCE (no subcycle), the clock advances
by dt_tracer, and momentum is under-relaxed (dt_mom < dt_tracer) — accelerating the
transient to the SAME steady state. legoESM exposes this as
``LatLonCGridOceanConfig.dt_mom_ratio`` (dt_mom = dt / dt_mom_ratio; the public ``dt``
IS dt_tracer). Default 1.0 ⇒ dt_mom == dt ⇒ bit-identical.

dt_mom drives momentum + the barotropic solve + implicit vertical FRICTION; dt_tracer
(=``dt``) drives tracers + continuity/eta + implicit vertical DIFFUSION + the clock.
The split is exact ONLY under ``barotropic_solver="rigid_lid"`` (fixed column depth ⇒
the tracer flux-form update is dt-independent ⇒ tracer mass conserved, like Veros's
streamfunction rigid lid); free-surface solvers are rejected at config validation.

Run in fp64 + on CPU for deterministic conservation diagnostics.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_DT_TRACER = 3600.0
_RATIO = 9.0
_N_LAT, _N_LON = 8, 16


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _rigid_lid_basin(dt_mom_ratio=1.0, outer_integrator="ab2", **cfg_kw):
    """Closed flat-bottom channel (land walls N/S, periodic x), rigid lid, with a
    meridional T gradient that drives a geostrophic flow."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    lm = np.ones((_N_LAT, _N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((_N_LAT, _N_LON), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=jnp.asarray(lm),
        H_bathy_override=jnp.asarray(Hb))
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        enable_runtime_checks=False, barotropic_solver="rigid_lid",
        outer_integrator=outer_integrator, dt_mom_ratio=dt_mom_ratio, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _tracer_content(model, state, field):
    """Discrete area·thickness-weighted total tracer Σ_ijk area·h_k·X·land_mask
    (the quantity flux-form advection + zero-flux implicit mixing conserve)."""
    from legoesm.ocean.vertical import compute_layer_thickness
    h_k = np.asarray(compute_layer_thickness(
        state.eta.data, state.H_bathy.data, model.z_coord,
        min_water_column_m=model.config.min_water_column_m))
    area = np.asarray(model.grid.area)[:, :, None]
    lm = np.asarray(state.land_mask.data)[:, :, None]
    return float(np.sum(np.asarray(field) * h_k * area * lm, dtype=np.float64))


def test_dt_mom_ratio_validation():
    """ratio < 1 rejected; ratio != 1 requires rigid_lid; ratio != 1 + rigid_lid OK."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)

    with pytest.raises(ValueError, match="dt_mom_ratio must be >= 1.0"):
        LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(dt_mom_ratio=0.5))
    # ratio != 1 with a free-surface solver is rejected.
    for solver in ("explicit_substep", "implicit_cn"):
        with pytest.raises(ValueError, match="requires barotropic_solver='rigid_lid'"):
            LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(
                dt_mom_ratio=_RATIO, barotropic_solver=solver))
    # ratio != 1 with rigid_lid constructs fine.
    LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(
        dt_mom_ratio=_RATIO, barotropic_solver="rigid_lid"))


def test_dt_mom_ratio_one_matches_default():
    """Explicitly setting dt_mom_ratio=1.0 is byte-for-byte identical to the default
    (1.0) — i.e. ratio=1.0 is a true no-op (dt_mom == dt_tracer == dt)."""
    state_d, model_d = _rigid_lid_basin()                       # default ratio 1.0
    state_e, model_e = _rigid_lid_basin(dt_mom_ratio=1.0)       # explicit 1.0
    fd, _ = model_d.integrate_scan(state_d, n_steps=6, dt=_DT_TRACER)
    fe, _ = model_e.integrate_scan(state_e, n_steps=6, dt=_DT_TRACER)
    for a, b in ((fd.T.data, fe.T.data), (fd.u.data, fe.u.data),
                 (fd.v.data, fe.v.data), (fd.S.data, fe.S.data)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_dt_mom_ratio_active():
    """The knob is wired: ratio=9 under-relaxes momentum (dt_mom=dt/9), so after N
    steps the flow is markedly weaker than the synchronous ratio=1 run."""
    s1, m1 = _rigid_lid_basin(dt_mom_ratio=1.0)
    s9, m9 = _rigid_lid_basin(dt_mom_ratio=_RATIO)
    f1, _ = m1.integrate_scan(s1, n_steps=10, dt=_DT_TRACER)
    f9, _ = m9.integrate_scan(s9, n_steps=10, dt=_DT_TRACER)
    umax1 = float(np.max(np.abs(np.asarray(f1.u.data))))
    umax9 = float(np.max(np.abs(np.asarray(f9.u.data))))
    assert np.all(np.isfinite(np.asarray(f9.u.data)))
    assert umax9 < 0.9 * umax1, (
        f"ratio=9 should under-relax momentum vs ratio=1: "
        f"max|u| {umax9:.4g} not < 0.9·{umax1:.4g}")


def test_dt_mom_rigid_lid_conserves_tracer():
    """Rung 2 (headline): under rigid lid the column depth is fixed, so the async
    dt_mom≠dt_tracer split conserves the area·thickness-weighted total heat/salt to
    f64 round-off (the reason the split is restricted to rigid_lid). Closed basin,
    no surface forcing/freshwater (integrate_scan calls step with neither)."""
    state, model = _rigid_lid_basin(dt_mom_ratio=_RATIO, outer_integrator="ab2")
    heat0 = _tracer_content(model, state, state.T.data)
    salt0 = _tracer_content(model, state, state.S.data)
    final, _ = model.integrate_scan(state, n_steps=30, dt=_DT_TRACER)
    assert np.all(np.isfinite(np.asarray(final.T.data)))
    assert float(np.max(np.abs(np.asarray(final.eta.data)))) == 0.0   # rigid lid
    heat1 = _tracer_content(model, final, final.T.data)
    salt1 = _tracer_content(model, final, final.S.data)
    rel_heat = abs(heat1 - heat0) / abs(heat0)
    rel_salt = abs(salt1 - salt0) / max(abs(salt0), 1e-30)
    assert rel_heat < 1e-11, f"async rigid-lid heat not conserved: rel drift {rel_heat:.3e}"
    assert rel_salt < 1e-11, f"async rigid-lid salt not conserved: rel drift {rel_salt:.3e}"


def test_dt_mom_async_differentiable():
    """end-to-end jax.grad through integrate_scan with the async + faithful-AB2 path
    (rigid-lid SPD-CG adjoint) is finite and nonzero."""
    state, model = _rigid_lid_basin(dt_mom_ratio=_RATIO, outer_integrator="ab2")
    T0 = state.T.data

    def loss(scale):
        st = state._replace(T=state.T.replace(data=T0 * scale))
        f, _ = model.integrate_scan(st, n_steps=3, dt=_DT_TRACER)
        return jnp.sum(f.u.data ** 2) + jnp.sum(f.T.data ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0
