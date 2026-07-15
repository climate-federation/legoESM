"""FV3_3D iter 234: cube-imprint regression for the full NH
d_con stack (mirror of PE iter-233).

Verifies that turning ALL 5 NH d_con knobs ON
(corner_div_damp_d_con, div_damp_d_con, ah_d_con,
damp_v_d_con, damp_w_d_con) does not introduce edge artifacts
in v.  d_con only modifies dθ_p_dt at cell centres (no edge
stencil) so the v field structure at panel boundaries should
be essentially unchanged.

Tests
-----

1. ``test_nh_d_con_does_not_amplify_imprint_ratio`` — full NH
   damping toolkit ON: imprint ratio with all 5 d_con knobs ON
   must remain within 50 % of d_con-OFF baseline.
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


def _edge_interior_std_ratio(state):
    """NH cube-imprint metric (matches iter-179 definition)."""
    v = state.v.data    # (6, n, n, nlev)
    n = v.shape[1]
    edge_width = 2

    i_idx = jnp.arange(n)
    j_idx = jnp.arange(n)
    edge_i = (i_idx < edge_width) | (i_idx >= n - edge_width)
    edge_j = (j_idx < edge_width) | (j_idx >= n - edge_width)
    edge_mask = edge_i[:, None] | edge_j[None, :]
    interior_mask = ~edge_mask

    v_edge = v[:, edge_mask, :].reshape(-1)
    v_interior = v[:, interior_mask, :].reshape(-1)
    return float(jnp.std(v_edge)) / max(float(jnp.std(v_interior)), 1e-30)


def test_nh_d_con_does_not_amplify_imprint_ratio():
    """NH d_con stack ON must not change cube-imprint ratio by
    more than 50% relative to OFF baseline."""
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

    rng = np.random.default_rng(seed=234)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
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

    common = dict(
        # Full NH FV3 toolkit (matches iter-184 umbrella).
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        damp_w=0.030, nord_w=1,
        A_h=1e6, smagorinsky_cs=0.20,
    )
    cfg_off = CDGridCompressibleEulerConfig(
        **common,
        damp_v_d_con=0.0,
        damp_w_d_con=0.0,
        corner_div_damp_d_con=0.0,
        div_damp_d_con=0.0,
        ah_d_con=0.0,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        **common,
        damp_v_d_con=1.0,
        damp_w_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
    )

    def step5(cfg):
        m = CDGridCompressibleEulerModel(
            grid, height_coord, terrain_metric, cfg,
        )
        s = state
        for _ in range(5):
            s = m.step(s, 10.0)
        return s

    s_off = step5(cfg_off)
    s_on = step5(cfg_on)

    ratio_off = _edge_interior_std_ratio(s_off)
    ratio_on = _edge_interior_std_ratio(s_on)

    assert jnp.isfinite(ratio_off) and jnp.isfinite(ratio_on)
    assert 0.5 < ratio_off < 5.0
    assert 0.5 < ratio_on < 5.0

    rel = abs(ratio_on - ratio_off) / max(ratio_off, 1e-30)
    assert rel < 0.5, (
        f"NH d_con stack must not change cube-imprint ratio by "
        f">50%: ratio_off={ratio_off:.4f}, "
        f"ratio_on={ratio_on:.4f}, rel={rel:.3f}.  Possible "
        f"d_con-induced edge artifact in NH path."
    )
