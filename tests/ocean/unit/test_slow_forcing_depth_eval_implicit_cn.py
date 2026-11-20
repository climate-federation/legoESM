"""``barotropic_slow_forcing_depth_evaluation`` is LIVE under the implicit-CN
free surface (the tripole/MPAS OMIP production solver), not only inside a
split-explicit window: the depth mean feeds ``F_slow`` before the solver
dispatch in ``_step_impl``.  It must change the step over partial cells and
leave a full-step mesh unchanged (there the reference and live min-rule
weights differ by one per-face scalar that cancels).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

_DT = 600.0


@pytest.fixture(autouse=True)
def _fp64():
    # fp64 policy: in float32 the two rules' different summation order alone
    # leaves ~1 ulp differences that would pass an "is it live" check.
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    old = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(old)


def _channel(evaluation, partial):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
    ny, nx, nz, H = 8, 16, 5, 3000.0
    grid = create_latlon_grid(n_lat=ny, n_lon=nx)
    z0c = create_ocean_z_star(n_levels=nz, H_max=H)
    H_bathy = jnp.asarray(np.full((ny, nx), H * 0.62)
                          - 300.0 * np.sin(2 * np.pi * np.arange(nx) / nx)[None, :] ** 2)
    z = create_partial_cell_coordinate(z0c, H_bathy) if partial else z0c
    state = rest_state_latlon_cgrid_ocean(
        grid, z0c, T_water_init_C=10.0, T_deep=10.0, S_uniform=35.0,
        H_bathy_override=H_bathy if partial else None)
    ny, nx = state.eta.data.shape
    u = np.array(np.asarray(state.u.data))
    u[:] = 0.3 * np.cos(np.linspace(0, np.pi, u.shape[2]))[None, None, :]
    # A sloping free surface: at eta = 0 the live thicknesses ARE the
    # reference ones and the two rules agree identically; they part only
    # where ssh stretches two neighbouring columns of different depth.
    eta = 0.5 * np.sin(2 * np.pi * np.arange(nx) / nx)[None, :] * np.ones((ny, 1))
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)),
                           eta=state.eta.replace(data=jnp.asarray(eta)))
    cfg = LatLonCGridOceanConfig.from_flat(
        barotropic_solver="implicit_cn", implicit_vertical_mixing=True,
        A_h=2.0e4, A_v=1.0e-3, K_v=1.0e-4, enable_runtime_checks=False,
        barotropic_slow_forcing_depth_evaluation=evaluation)
    return state, LatLonCGridOceanModel(grid, z, cfg)


def _step(evaluation, partial):
    state, model = _channel(evaluation, partial)
    out = model.step(state, _DT)
    return np.asarray(out.eta.data), np.asarray(out.u.data)


def test_nemo_literal_changes_the_cn_step_over_partial_cells():
    # non-vacuity: identical inputs give a bitwise-identical step, so a
    # nonzero difference below can only come from the setting
    a, b = _step("min_rule_live", partial=True), _step("min_rule_live", partial=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    eta_l, u_l = _step("nemo_literal", partial=True)
    eta_m, u_m = _step("min_rule_live", partial=True)
    assert np.all(np.isfinite(eta_l)) and np.all(np.isfinite(u_l))
    assert eta_l.dtype == np.float64 and u_l.dtype == np.float64
    # far above fp64 roundoff of these O(1) fields: a real change of the forcing
    assert np.max(np.abs(eta_l - eta_m)) > 1e-10, "eta: setting inert under implicit_cn"
    assert np.max(np.abs(u_l - u_m)) > 1e-12, "u: setting inert under implicit_cn"


def test_full_step_mesh_is_unchanged():
    eta_l, u_l = _step("nemo_literal", partial=False)
    eta_m, u_m = _step("min_rule_live", partial=False)
    assert np.array_equal(u_l, u_m) and np.array_equal(eta_l, eta_m)
