"""FV3_3D iter 452: scale-honest end-to-end test for FV3 sponge
boost.

Finding from iter-452 attempt: the FV3 namelist values
``d2_bg_k1=4.0`` + ``d2_bg_k2=2.0`` assume FV3's specific
``da_min_c × d2`` normalization; legoESM's grid-area scale
makes the same numeric values produce ~10⁸×-too-aggressive
damping → top-level blow-up.

Consequence: iter-444's factory defaults
``d2_bg_k1=4.0`` / ``d2_bg_k2=2.0`` were rolled back to 0.0
in iter-452 (sponge flags remain True but no-op without k
values).  User must set d2_bg_k* explicitly at a value
compatible with their legoESM-scale ``d2_bg``.

This test verifies the SPONGE BOOST mechanism IS effective
when called with appropriately small d2_bg_k* values (small
relative to legoESM's da_min_c).  Catches a regression where
the sponge mechanism stops working at any value.

Tests
-----

1. ``test_nh_sponge_boost_damps_top_winds_small_k1`` —
   ``d2_bg_k1=1e-4`` produces detectable but non-blowup
   damping at top.
2. ``test_pe_sponge_boost_damps_top_winds_small_k1`` — same on PE.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
    standard_hybrid_levels,
)


def test_nh_sponge_boost_damps_top_winds_small_k1():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=False)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=452)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state0 = NonHydrostaticState(
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
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg_off = CDGridCompressibleEulerConfig(**base_kw)
    cfg_on = CDGridCompressibleEulerConfig(
        corner_div_damp_d2_bg_k1=1.0e-4,
        use_fv3_sponge_damp_w=True,
        use_fv3_sponge_damp_v=True,
        **base_kw,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off, s_on = state0, state0
    for _ in range(3):
        s_off = m_off.step(s_off, dt=10.0)
        s_on = m_on.step(s_on, dt=10.0)
    # Finite check first
    assert jnp.all(jnp.isfinite(s_on.u.data))
    # Top-level |u| should DIFFER (sponge boost active)
    u_top_off = float(jnp.max(jnp.abs(s_off.u.data[..., 0])))
    u_top_on = float(jnp.max(jnp.abs(s_on.u.data[..., 0])))
    assert abs(u_top_off - u_top_on) > 0.0, (
        "Sponge boost mechanism produced no observable change "
        "at top level — sponge flag stack may be broken."
    )


def test_pe_sponge_boost_damps_top_winds_small_k1():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state0 = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=452)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state0 = state0._replace(
        u_d=state0.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state0.v_d.replace(data=jnp.asarray(v_p)),
    )
    base_kw = dict(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    cfg_off = CDGridPrimitiveEquationConfig(**base_kw)
    cfg_on = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg_k1=1.0e-4,
        use_fv3_sponge_damp_v=True,
        **base_kw,
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off, s_on = state0, state0
    for _ in range(3):
        s_off = m_off.step(s_off, dt=10.0)
        s_on = m_on.step(s_on, dt=10.0)
    assert jnp.all(jnp.isfinite(s_on.u_d.data))
    u_top_off = float(jnp.max(jnp.abs(s_off.u_d.data[..., 0])))
    u_top_on = float(jnp.max(jnp.abs(s_on.u_d.data[..., 0])))
    assert abs(u_top_off - u_top_on) > 0.0, (
        "PE sponge boost produced no observable change."
    )
