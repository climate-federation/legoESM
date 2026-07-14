"""FV3_3D iter 483: aggressive NH edge-min factory.

Stacks iter-466 hurting-flag drops + iter-481 d2_bg boost.
iter-482 verified 59.5% reduction over factory (1.42× vs 3.50).

Tests
-----

1. ``test_factory_importable``.
2. ``test_factory_disables_hurting_flags``.
3. ``test_factory_boosts_d2_bg``.
4. ``test_factory_overrides_take_precedence``.
5. ``test_factory_step_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_legoesm_nh_min_edge_aggressive_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def test_factory_importable():
    assert callable(make_legoesm_nh_min_edge_aggressive_config)


def test_factory_disables_hurting_flags():
    cfg = make_legoesm_nh_min_edge_aggressive_config()
    assert cfg.use_fv3_metric_aware_d_con is False
    assert cfg.heat_source_del2_iters == 0
    assert cfg.d_con_top_zero_levels == 0


def test_factory_boosts_d2_bg():
    cfg = make_legoesm_nh_min_edge_aggressive_config()
    assert cfg.corner_div_damp_d2_bg == 5e-2


def test_factory_overrides_take_precedence():
    cfg = make_legoesm_nh_min_edge_aggressive_config(
        corner_div_damp_d2_bg=1e-3,
        use_fv3_metric_aware_d_con=True,
    )
    assert cfg.corner_div_damp_d2_bg == 1e-3
    assert cfg.use_fv3_metric_aware_d_con is True


def test_factory_step_finite():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=483)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d,
                          units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d,
                        units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    cfg = make_legoesm_nh_min_edge_aggressive_config(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    model = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    new_state = model.step(state, dt=10.0)
    for f in (new_state.u, new_state.v, new_state.w,
              new_state.theta_prime, new_state.rho_prime):
        assert jnp.all(jnp.isfinite(f.data))
