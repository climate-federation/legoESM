"""FV3_3D iter 198: quantitative test for the iter-187 nord >= 1
corner-divergence damping path on the NH 3D path.

iter-174 added a quantitative correctness test that the iter-168
corner-div damping (nord=0) reduces ``mean(|div_v|)`` on a
divergent IC.  iter-187 added the FV3-faithful smag_vort cap for
nord >= 1 with finiteness tests but NO quantitative test that the
higher-order branch actually reduces divergence MORE than the
nord=0 path alone — a silent regression where the higher-order
branch becomes a no-op (e.g., dd8 coefficient bug, divg_d_iter
all-zeros) would not be caught.

This iter mirrors iter-174's pattern: same divergent IC fixture,
3 configs (no damping, nord=0, nord=1), assert nord >= 1 reduces
mean(|div_v|) MORE than nord=0 alone (i.e., the higher-order
branch ADDS damping on top of the d2-only baseline).

Tests
-----

1. ``test_corner_div_damp_nord1_reduces_more_than_d2_only`` —
   nord=1 + d4_bg=1e-3 + same d2_bg as nord=0 reduces div MORE
   than nord=0 alone.  Catches the silent no-op of the d4 branch.
2. ``test_corner_div_damp_nord2_finite_on_divergent_ic`` —
   nord=2 (FV3 AM4 production order) + d4_bg=1e-4 (n=8-scaled
   from FV3 production 0.16) runs to completion on the divergent
   IC without going NaN.  This complements iter-187's nord=2
   test (which uses a perturbed-from-rest IC, not a strongly
   divergent one).
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
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid, cgrid_divergence, interp_center_to_corner,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def divergent_nh_state():
    """Same divergent NH state as iter-174 fixture."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    i_idx = jnp.arange(n)
    j_idx = jnp.arange(n)
    k_idx = jnp.arange(nlev)
    pattern = (
        jnp.sin(2 * jnp.pi * i_idx[None, :, None, None] / n)
        * jnp.cos(2 * jnp.pi * j_idx[None, None, :, None] / n)
        * jnp.ones_like(k_idx[None, None, None, :], dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n, n, nlev))
    u_perturb = jnp.asarray(5.0 * pattern, dtype=jnp.float64)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=u_perturb, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
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
    return grid, cdgrid, height_coord, terrain_metric, state


def _mean_abs_div(state, cdgrid):
    """Same metric as iter-174."""
    n_face_uv, n_i_uv, n_j_uv, nlev_uv = state.u.data.shape
    _uv_stack = jnp.stack([state.u.data, state.v.data], axis=-1)
    _uv_flat = _uv_stack.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv * 2)
    _uv_d_flat = interp_center_to_corner(_uv_flat, cdgrid)
    _uv_d = _uv_d_flat.reshape(
        _uv_d_flat.shape[0], _uv_d_flat.shape[1],
        _uv_d_flat.shape[2], nlev_uv, 2,
    )
    u_d = _uv_d[..., 0]
    v_d = _uv_d[..., 1]
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    div_v = cgrid_divergence(u_c, v_c, cdgrid)
    return float(jnp.mean(jnp.abs(div_v)))


def _step5(model, state, dt=10.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_corner_div_damp_nord1_reduces_more_than_d2_only(divergent_nh_state):
    """nord=1 + d4_bg > 0 reduces ``mean(|div_v|)`` MORE than the
    nord=0 path with same d2_bg.  Catches a silent regression where
    the iter-18-equivalent higher-order branch becomes no-op
    (e.g., dd8 coefficient or divg_d_iter wired wrong)."""
    grid, cdgrid, height_coord, terrain_metric, state = divergent_nh_state

    common = dict(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
    )
    cfg_d2_only = CDGridCompressibleEulerConfig(
        **common,
        # nord=0, no d4 — iter-16-equivalent path only
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=0,
    )
    cfg_d4_nord1 = CDGridCompressibleEulerConfig(
        **common,
        # nord=1 + d4_bg > 0 — engages iter-187 smag_vort cap +
        # iter-18-equivalent del-4 ke_correction term
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
    )

    m_d2 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d2_only,
    )
    m_d4 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d4_nord1,
    )

    s_d2 = _step5(m_d2, state)
    s_d4 = _step5(m_d4, state)

    div_d2 = _mean_abs_div(s_d2, cdgrid)
    div_d4 = _mean_abs_div(s_d4, cdgrid)

    assert div_d4 < div_d2, (
        f"nord=1 + d4_bg > 0 must reduce mean|div_v| MORE than "
        f"nord=0 alone: nord=0 div={div_d2:.4e}, "
        f"nord=1 div={div_d4:.4e} (higher-order branch did not "
        f"add damping — silent regression in the d4_bg/dd8 "
        f"wiring, or the iter-187 smag_vort cap is too aggressive "
        f"and reducing damping below d2-only)."
    )


def test_corner_div_damp_nord2_finite_on_divergent_ic(divergent_nh_state):
    """nord=2 (FV3 d_sw5 production order, del-6) + d4_bg n=8-scaled
    runs to completion on a strongly divergent IC without going
    NaN.  Complements iter-187's nord=2 test which uses a
    perturbed-from-rest IC; this exercises the higher-order branch
    on a state where it should be doing real work."""
    grid, cdgrid, height_coord, terrain_metric, state = divergent_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
        # n=8-scaled (FV3 production at C96 uses 0.16; at C8
        # da_min_c is much larger so dd8 = (da_min_c*d4_bg)^3
        # grows quickly and 1e-4 keeps it in scale)
        corner_div_damp_d4_bg=1e-4,
        corner_div_damp_nord=2,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    s = _step5(model, state)

    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.v.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))
    assert jnp.all(jnp.isfinite(s.rho_prime.data))

    div_after = _mean_abs_div(s, cdgrid)
    assert div_after < 1e-2, (
        f"nord=2 path should keep div_v bounded after 5 steps "
        f"({div_after:.4e})"
    )
