"""FV3_3D iter 269: NH cube-imprint metric OVER TIME (NH
counterpart of PE iter-268).

Tests
-----

1. ``test_nh_imprint_ratio_bounded_over_time`` — 50 NH steps
   with iter-184 toolkit + full 5-knob d_con stack; record
   imprint ratio every 10 steps; all sampled values within
   (0.1, 5.0) + growth factor < 5×.
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


def _edge_interior_std_ratio_nh(state, edge_width=2):
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


def test_nh_imprint_ratio_bounded_over_time():
    """50 NH steps with iter-184 toolkit + full d_con stack;
    imprint ratio bounded across all sample times."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=269)
    u_p = rng.uniform(-5.0, 5.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-5.0, 5.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev + 1))

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
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
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
    ratios = [_edge_interior_std_ratio_nh(s)]
    for i in range(50):
        s = model.step(s, 10.0)
        if (i + 1) % 10 == 0:
            ratios.append(_edge_interior_std_ratio_nh(s))

    for k, r in enumerate(ratios):
        assert jnp.isfinite(r), (
            f"NH imprint NaN at sample {k} (step {k * 10})."
        )
        assert 0.1 < r < 5.0, (
            f"NH imprint ratio at step {k * 10} = {r:.4f} "
            f"outside (0.1, 5.0).  All: "
            f"{[f'{x:.3f}' for x in ratios]}."
        )

    max_r = max(ratios)
    initial_r = ratios[0]
    growth_factor = max_r / max(initial_r, 1e-30)
    assert growth_factor < 5.0, (
        f"NH imprint growth {growth_factor:.2f}x — possible "
        f"slow-growth edge artifact accumulation."
    )
