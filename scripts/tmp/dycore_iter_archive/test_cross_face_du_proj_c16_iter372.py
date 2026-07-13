"""FV3_3D iter 372: C16 does-not-amplify regression for iter-370
cross-face du projection flag.

Mirror of iter-357/358 pattern.  At C16 (where cube imprint
signal is bigger than C8), enabling iter-370 cross_face flag must
not amplify cube imprint by > 10 % vs default mode='edge'.

Tests
-----

1. ``test_pe_c16_cross_face_does_not_amplify``
2. ``test_nh_c16_cross_face_does_not_amplify``
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
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
    standard_hybrid_levels,
)


def _imprint_T(state, n_cells, edge_width=3):
    T = state.T.data
    i_idx = jnp.arange(n_cells)
    edge_i = (i_idx < edge_width) | (i_idx >= n_cells - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    e = float(jnp.std(T[:, edge_mask, :].reshape(-1)))
    i = float(jnp.std(T[:, interior_mask, :].reshape(-1)))
    return e / max(i, 1e-30)


def _imprint_v(state, n_cells, edge_width=3):
    v = state.v.data
    i_idx = jnp.arange(n_cells)
    edge_i = (i_idx < edge_width) | (i_idx >= n_cells - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    e = float(jnp.std(v[:, edge_mask, :].reshape(-1)))
    i = float(jnp.std(v[:, interior_mask, :].reshape(-1)))
    return e / max(i, 1e-30)


def test_pe_c16_cross_face_does_not_amplify():
    n = 16
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=372)
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n + 1, n + 1, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    def cfg(cross_face):
        return CDGridPrimitiveEquationConfig(
            damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
            corner_div_damp_d2_bg=0.0,
            A_h=0.0, hyperdiff_coeff=0.0,
            hyperdiff_ps_coeff=0.0, div_damp_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False,
            use_fv3_cross_face_du_proj=cross_face,
        )

    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg(False))
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg(True))
    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)
    r_off = _imprint_T(s_off, n_cells=16)
    r_on = _imprint_T(s_on, n_cells=16)
    assert r_on <= r_off * 1.10, (
        f"PE cross_face at C16 AMPLIFIES imprint: r_off={r_off:.4f}, "
        f"r_on={r_on:.4f}."
    )


def test_nh_c16_cross_face_does_not_amplify():
    n = 16
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    tm = compute_terrain_metric(terrain, hc)
    rng = np.random.default_rng(seed=372)
    u_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
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
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )

    def cfg(cross_face):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            damp_v=0.030, nord_v=1,
            use_fv3_cross_face_du_proj=cross_face,
        )

    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg(False))
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg(True))
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    r_off = _imprint_v(s_off, n_cells=16)
    r_on = _imprint_v(s_on, n_cells=16)
    assert r_on <= r_off * 1.10, (
        f"NH cross_face at C16 AMPLIFIES imprint: r_off={r_off:.4f}, "
        f"r_on={r_on:.4f}."
    )
