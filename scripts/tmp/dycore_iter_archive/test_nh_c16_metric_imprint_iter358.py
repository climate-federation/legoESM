"""FV3_3D iter 358: C16 NH counterpart of iter-357.

NH does-not-amplify check at C16 with full FV3-fidelity stack
(cv + vector_halo + dyn_exner + metric) vs default flags.

Tests
-----

1. ``test_nh_c16_full_fv3_stack_does_not_amplify`` — full NH
   FV3-fidelity stack must not amplify v-field cube-imprint
   ratio by > 10 % vs default flags + same toolkit.
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


def _build_c16(use_duogrid=False):
    n = 16
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=358)
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
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def _imprint_ratio_v(state, n_cells, edge_width=3):
    v = state.v.data
    i_idx = jnp.arange(n_cells)
    edge_i = (i_idx < edge_width) | (i_idx >= n_cells - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    edge = v[:, edge_mask, :].reshape(-1)
    interior = v[:, interior_mask, :].reshape(-1)
    e = float(jnp.std(edge))
    i = float(jnp.std(interior))
    return e / max(i, 1e-30)


def _nh_toolkit_cfg(all_flags):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        damp_v=0.030, nord_v=1,
        damp_v_d_con=1.0,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
        delt_max=1.0,
        use_fv3_d_con_cv=all_flags,
        use_fv3_vector_halo_uv=all_flags,
        use_fv3_dynamic_exner=all_flags,
        use_fv3_metric_aware_d_con=all_flags,
    )


def test_nh_c16_full_fv3_stack_does_not_amplify():
    """Full NH FV3-fidelity stack at C16 must not amplify v-field
    cube-imprint ratio by > 10 % vs default flags + same toolkit.
    """
    grid_plain, hc, tm, state = _build_c16(use_duogrid=False)
    grid_duo, hc2, tm2, state2 = _build_c16(use_duogrid=True)

    m_default = CDGridCompressibleEulerModel(
        grid_plain, hc, tm, _nh_toolkit_cfg(all_flags=False),
    )
    m_full = CDGridCompressibleEulerModel(
        grid_duo, hc2, tm2, _nh_toolkit_cfg(all_flags=True),
    )
    s_d = m_default.step(state, 5.0)
    s_f = m_full.step(state2, 5.0)
    r_d = _imprint_ratio_v(s_d, n_cells=16)
    r_f = _imprint_ratio_v(s_f, n_cells=16)
    assert r_f <= r_d * 1.10, (
        f"NH full FV3-fidelity stack at C16 AMPLIFIES v-imprint "
        f"by > 10 %: r_default={r_d:.4f}, r_full={r_f:.4f}.  "
        f"Combined flags interact destructively."
    )
