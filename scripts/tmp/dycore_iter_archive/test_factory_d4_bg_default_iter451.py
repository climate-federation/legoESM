"""FV3_3D iter 451: factory defaults expose FV3 production
``d4_bg = 0.16`` background del-4 divergence damping coefficient.

Per FV3 ``fv_arrays.F90``::

    real :: d4_bg = 0.16   ! coefficient for background del-4(6)
                            ! divergence damping

Both factories now set ``corner_div_damp_d4_bg=0.16``.
Combined with iter-437 ``corner_div_damp_nord=1`` (del-4), the
factory configs activate the FV3 higher-order del-4 damping
path by default.

Tests
-----

1. ``test_nh_factory_default_d4_bg`` — 0.16.
2. ``test_pe_factory_default_d4_bg`` — 0.16.
3. ``test_nh_factory_override_zero`` — explicit override
   recovers baseline (d4_bg=0 disables higher-order path).
4. ``test_factory_steps_still_finite_post_d4_bg`` — iter-427
   factory step test still produces finite state with the new
   default.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
    make_fv3_faithful_pe_config,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
    standard_hybrid_levels,
)


def test_nh_factory_default_d4_bg():
    cfg = make_fv3_faithful_nh_config()
    assert cfg.corner_div_damp_d4_bg == 0.16


def test_pe_factory_default_d4_bg():
    cfg = make_fv3_faithful_pe_config()
    assert cfg.corner_div_damp_d4_bg == 0.16


def test_nh_factory_override_zero():
    cfg = make_fv3_faithful_nh_config(corner_div_damp_d4_bg=0.0)
    assert cfg.corner_div_damp_d4_bg == 0.0


def test_factory_steps_still_finite_post_d4_bg():
    """Quick sanity: factory + 1-step at C8 still finite with
    the new ``d4_bg=0.16`` default."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=451)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state_nh = NonHydrostaticState(
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
    cfg_nh = make_fv3_faithful_nh_config(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    m_nh = CDGridCompressibleEulerModel(grid, hc, tm, cfg_nh)
    s_nh = m_nh.step(state_nh, dt=10.0)
    assert jnp.all(jnp.isfinite(s_nh.u.data))

    coord = standard_hybrid_levels(nlev)
    cdg = create_cubed_sphere_cdgrid(grid)
    state_pe = hydrostatic_to_fv3(held_suarez_init(grid, coord), cdg)
    cfg_pe = make_fv3_faithful_pe_config(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    m_pe = CDGridPrimitiveEquationModel(grid, coord, cfg_pe)
    s_pe = m_pe.step(state_pe, dt=10.0)
    assert jnp.all(jnp.isfinite(s_pe.T.data))
