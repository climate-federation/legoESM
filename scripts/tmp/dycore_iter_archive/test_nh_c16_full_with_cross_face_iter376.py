"""FV3_3D iter 376: C16 NH full FV3-fidelity stack with iter-370
cross_face flag included.

iter-358 + iter-360 covered C16 NH stack without cross_face;
iter-376 adds cross_face to confirm full 5-flag stack composes
at production resolution.

Tests
-----

1. ``test_nh_c16_full_5_flags_does_not_amplify`` — all 5 NH
   flags + duogrid + full toolkit at C16 does not amplify
   v-imprint vs default flags + same toolkit.
2. ``test_nh_c16_full_5_flags_changes_state`` — measurably
   different θ_p from default at C16.
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
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


def _build_c16(use_duogrid):
    n = 16
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    hc = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    tm = compute_terrain_metric(terrain, hc)
    rng = np.random.default_rng(seed=376)
    u_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def _imprint_v(state, n_cells, edge_width=3):
    v = state.v.data
    i_idx = jnp.arange(n_cells)
    edge_i = (i_idx < edge_width) | (i_idx >= n_cells - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    e = float(jnp.std(v[:, edge_mask, :].reshape(-1)))
    i = float(jnp.std(v[:, interior_mask, :].reshape(-1)))
    return e / max(i, 1e-30)


def _nh_cfg(all_flags):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        use_fv3_a2b_zeta_corner=all_flags,
        div_damp_coeff=1e6, div_damp_dddmp=0.20, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
        delt_max=1.0,
        use_fv3_d_con_cv=all_flags,
        use_fv3_vector_halo_uv=all_flags,
        use_fv3_dynamic_exner=all_flags,
        use_fv3_metric_aware_d_con=all_flags,
        use_fv3_cross_face_du_proj=all_flags,
    )


def test_nh_c16_full_5_flags_does_not_amplify():
    grid_d, hc, tm, state = _build_c16(use_duogrid=False)
    grid_f, hc2, tm2, state_f = _build_c16(use_duogrid=True)
    m_d = CDGridCompressibleEulerModel(grid_d, hc, tm, _nh_cfg(False))
    m_f = CDGridCompressibleEulerModel(grid_f, hc2, tm2, _nh_cfg(True))
    s_d = m_d.step(state, 5.0)
    s_f = m_f.step(state_f, 5.0)
    r_d = _imprint_v(s_d, n_cells=16)
    r_f = _imprint_v(s_f, n_cells=16)
    assert r_f <= r_d * 1.10, (
        f"NH full 5-flag stack at C16 AMPLIFIES imprint: "
        f"r_default={r_d:.4f}, r_full={r_f:.4f}."
    )


def test_nh_c16_full_5_flags_changes_state():
    grid_d, hc, tm, state = _build_c16(use_duogrid=False)
    grid_f, hc2, tm2, state_f = _build_c16(use_duogrid=True)
    m_d = CDGridCompressibleEulerModel(grid_d, hc, tm, _nh_cfg(False))
    m_f = CDGridCompressibleEulerModel(grid_f, hc2, tm2, _nh_cfg(True))
    s_d = m_d.step(state, 5.0)
    s_f = m_f.step(state_f, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_d.theta_prime.data)
        - np.asarray(s_f.theta_prime.data),
    )))
    assert diff > 1e-6
