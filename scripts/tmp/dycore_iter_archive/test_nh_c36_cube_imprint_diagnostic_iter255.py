"""FV3_3D iter 255: cube-imprint diagnostic on a NH C36
production-style run (NH counterpart of iter-254).

iter-251 verified NH C36 stability with d_con on.  iter-179/
234 cover NH cube-imprint at C8.  iter-255 ties them together
at C36 production resolution.

Tests
-----

1. ``test_nh_c36_production_d_con_imprint_bounded`` — C36 NH
   + iter-184-style toolkit + full d_con stack, integrate
   for 5 steps × dt=10 from a perturbation IC, measure v
   edge_std/interior_std ratio.  Must stay in (0.1, 10.0)
   sanity range.
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
    compute_terrain_metric, create_height_coordinate,
)


def _edge_interior_std_ratio_nh(state, edge_width=4):
    """NH cube-imprint metric at C36 (edge_width=4)."""
    v = state.v.data    # (6, n, n, nlev)
    n = v.shape[1]

    i_idx = jnp.arange(n)
    j_idx = jnp.arange(n)
    edge_i = (i_idx < edge_width) | (i_idx >= n - edge_width)
    edge_j = (j_idx < edge_width) | (j_idx >= n - edge_width)
    edge_mask = edge_i[:, None] | edge_j[None, :]
    interior_mask = ~edge_mask

    v_edge = v[:, edge_mask, :].reshape(-1)
    v_interior = v[:, interior_mask, :].reshape(-1)
    return float(jnp.std(v_edge)) / max(float(jnp.std(v_interior)), 1e-30)


def test_nh_c36_production_d_con_imprint_bounded():
    """C36 NH + production toolkit + full d_con stack must keep
    cube-imprint ratio in sanity bounds after 5 steps."""
    n = 36
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=255)
    u_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev))
    v_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.05, 0.05, size=(6, n, n, nlev + 1))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )

    cfg = CDGridCompressibleEulerConfig(
        # NH production toolkit (matches iter-184 umbrella)
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        # Full NH d_con stack
        damp_w_d_con=1.0,
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=1.0,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    s = state
    dt = 10.0
    for _ in range(5):
        s = model.step(s, dt)

    final_ratio = _edge_interior_std_ratio_nh(s)
    assert jnp.isfinite(final_ratio), (
        f"NH C36 production: imprint ratio NaN/Inf — possible "
        f"d_con or damping instability."
    )
    assert 0.1 < final_ratio < 10.0, (
        f"NH C36 production imprint ratio={final_ratio:.4f} "
        f"outside sanity bound (0.1, 10.0).  Cube imprint may "
        f"be runaway."
    )

    max_u = float(jnp.max(jnp.abs(s.u.data)))
    assert max_u < 50.0, (
        f"NH C36 max|u|={max_u:.2f} > 50 m/s after 5 steps."
    )
