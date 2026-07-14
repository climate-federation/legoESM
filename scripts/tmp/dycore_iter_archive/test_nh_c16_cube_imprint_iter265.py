"""FV3_3D iter 265: NH C16 cube-imprint reduction validation
(NH counterpart of PE iter-264).

Tests
-----

1. ``test_nh_c16_iter184_toolkit_does_not_amplify_imprint`` —
   NH C16 + iter-184-style toolkit + iter-187 path vs
   iter-168 nord=0 baseline.  Imprint ratio within 50 % of
   baseline.
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


def test_nh_c16_iter184_toolkit_does_not_amplify_imprint():
    """NH C16 iter-184 toolkit + iter-187 + iter-190 path
    must not amplify v cube-imprint ratio by >50% vs the
    iter-168 nord=0 baseline."""
    n = 16
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=265)
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
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
    )
    cfg_baseline = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=0,
        use_fv3_a2b_zeta_corner=False,
    )
    cfg_iter187 = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
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
    s_iter187 = step5(cfg_iter187)

    ratio_base = _edge_interior_std_ratio_nh(s_base)
    ratio_iter187 = _edge_interior_std_ratio_nh(s_iter187)

    assert jnp.isfinite(ratio_base) and jnp.isfinite(ratio_iter187)
    assert 0.5 < ratio_base < 5.0
    assert 0.5 < ratio_iter187 < 5.0

    rel = abs(ratio_iter187 - ratio_base) / max(ratio_base, 1e-30)
    assert rel < 0.5, (
        f"NH C16 iter-187 + iter-190 must not change cube-"
        f"imprint ratio by >50%: baseline={ratio_base:.4f}, "
        f"iter187={ratio_iter187:.4f}, rel={rel:.3f}.  "
        f"Possible NH-side edge artifact at C16 from the "
        f"iter-187 cap or iter-190 dedup path."
    )
