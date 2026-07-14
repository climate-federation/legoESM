"""FV3_3D iter 252: cube-imprint regression test with the
iter-187 smag_vort cap + iter-190 dedup'd a2b ζ corner path
engaged on the NH 3D path.

iter-179 NH cube-imprint metric tests engage the basic iter-168
corner-div (nord=0).  iter-197/206/212 extend to damp_w +
damp_w_d_con + damp_v_d_con.  But the iter-187 smag_vort cap
(``nord=1`` + ``d4_bg > 0`` activates the
``|dt|*sqrt(delpc²+ζ²)`` cap) and the iter-190 dedup'd
``_zeta_a2b_ord4`` path are NOT covered by any cube-imprint
test on either PE or NH.

iter 252 closes that gap.  Verifies that engaging iter-187 +
iter-190 simultaneously does not amplify cube-imprint relative
to the baseline iter-168 + iter-179 toolkit.

Tests
-----

1. ``test_nh_iter187_smag_vort_does_not_amplify_imprint`` —
   iter-184-style toolkit + iter-187 (nord=1, d4_bg > 0) +
   iter-190 (use_fv3_a2b_zeta_corner=True) ON; imprint ratio
   stays within 50 % of the baseline (iter-168 nord=0 only)
   ratio at C8.
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
    """NH cube-imprint metric (iter-179 definition)."""
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


def test_nh_iter187_smag_vort_does_not_amplify_imprint():
    """iter-187 + iter-190 must not amplify cube-imprint ratio
    beyond 50 % of baseline (iter-168 nord=0)."""
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

    rng = np.random.default_rng(seed=252)
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
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.001, corner_div_damp_dddmp=0.20,
    )
    cfg_baseline = CDGridCompressibleEulerConfig(
        **common,
        # nord=0: iter-168 only (no iter-187 smag_vort cap)
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=0,
        use_fv3_a2b_zeta_corner=False,
    )
    cfg_iter187_iter190 = CDGridCompressibleEulerConfig(
        **common,
        # nord=1 + d4_bg > 0: engages iter-187 smag_vort cap
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        # iter-190 dedup path (use_fv3_a2b_zeta_corner=True
        # AND iter-187 active uses dedup'd _zeta_a2b_ord4)
        use_fv3_a2b_zeta_corner=True,
    )

    def step5(cfg):
        m = CDGridCompressibleEulerModel(
            grid, height_coord, terrain_metric, cfg,
        )
        s = state
        for _ in range(5):
            s = m.step(s, 10.0)
        return s

    s_base = step5(cfg_baseline)
    s_iter187 = step5(cfg_iter187_iter190)

    ratio_base = _edge_interior_std_ratio(s_base)
    ratio_iter187 = _edge_interior_std_ratio(s_iter187)

    assert jnp.isfinite(ratio_base) and jnp.isfinite(ratio_iter187)
    assert 0.5 < ratio_base < 5.0
    assert 0.5 < ratio_iter187 < 5.0

    # iter-187 + iter-190 should produce STRONGER damping (because
    # smag_vort cap > |delpc|*dt cap when ζ ≠ 0).  So
    # ratio_iter187 should be ≤ ratio_baseline + tolerance.
    # If iter-187 wiring has an edge bug, ratio could AMPLIFY.
    rel = abs(ratio_iter187 - ratio_base) / max(ratio_base, 1e-30)
    assert rel < 0.5, (
        f"iter-187 smag_vort + iter-190 dedup must not change "
        f"cube-imprint ratio by >50%: baseline={ratio_base:.4f}, "
        f"iter187={ratio_iter187:.4f}, rel={rel:.3f}.  Possible "
        f"edge artifact from the iter-187 cap or iter-190 dedup."
    )
