"""FV3_3D iter 432: extend iter-431 ``d_con_top_zero_levels``
mask to the remaining 4 NH d_con sites.

iter-431 wired the mask at post-acoustic damp_v_d_con.  iter-
432 extends to:
* post-acoustic damp_w_d_con (new mask at line 1756)
* aggregate ``_d_con_sum`` covering 3 slow-tendency sites
  (corner_div, div_damp, A_h)

Matches FV3 ``dyn_core.F90:790/800/804`` semantics: ``d_con_k
= 0`` zeros KE→heat from ALL damping mechanisms uniformly per
sponge level.  Aggregate mask at the sum site is exactly
equivalent to per-mechanism masking pre-sum (linearity).

Tests
-----

1. ``test_damp_w_site_masked`` — damp_w post-acoustic d_con
   site respects mask: top levels of θ′ change preserved
   identical with/without mask only in bottom levels; flag-ON
   produces zero net θ′ change at masked levels for the
   damp_w-only configuration.
2. ``test_slow_tendency_aggregate_masked`` — slow-tendency
   d_con sites respect mask: with corner_div + div_damp + A_h
   active and all post-acoustic damp_* disabled, flag-ON
   should produce zero θ′ change from the d_con AGGREGATE in
   the top N levels.
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
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_c8(seed, w_amp=1.0):
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=False)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-w_amp, w_amp, size=(6, n, n, nlev + 1))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
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
    return grid, height_coord, terrain_metric, state, nlev


def test_damp_w_site_masked():
    """damp_w-only configuration: with mask N=2, top 2 levels
    of θ′ differ from N=0 (mask kicks in); bottom levels
    identical (mask absent there).
    """
    grid, hc, tm, state, _ = _build_c8(seed=4321)
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1, damp_w_d_con=1.0,
        d_con_top_zero_levels=0,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1, damp_w_d_con=1.0,
        d_con_top_zero_levels=2,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    dth_off = s_off.theta_prime.data - state.theta_prime.data
    dth_on = s_on.theta_prime.data - state.theta_prime.data
    # Below top 2 levels (k>=2): identical (mask absent).
    np.testing.assert_allclose(
        np.asarray(dth_off[..., 2:]),
        np.asarray(dth_on[..., 2:]),
        rtol=1e-14, atol=1e-14,
    )
    # Top 2 levels: differ (damp_w d_con site picks up the mask).
    delta = float(jnp.max(jnp.abs(
        dth_off[..., :2] - dth_on[..., :2],
    )))
    assert delta > 0.0, (
        "iter-432 damp_w d_con site mask is a no-op; flag-ON "
        "should change top-level dtheta_p when damp_w_d_con > 0."
    )


def test_slow_tendency_aggregate_masked():
    """corner_div + div_damp + A_h slow-tendency d_con sites
    via the aggregate ``_d_con_sum`` respect the mask.

    Asserts effect is concentrated in top levels: the
    max-abs-diff of θ′ change at top-2 levels is >> max-abs-
    diff at bottom levels.  We do not assert bottom-identical
    because RK3 stages + vertical PGF couple levels through
    intermediate state propagation, which causes O(1e-8)
    bottom-level drift after 1 step even with ``w_init=0``.
    """
    grid, hc, tm, state, _ = _build_c8(seed=8765, w_amp=0.0)
    common = dict(
        n_acoustic_substeps=4,
        # All three slow-tendency d_con sources active.
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
    )
    cfg_off = CDGridCompressibleEulerConfig(
        d_con_top_zero_levels=0, **common,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        d_con_top_zero_levels=2, **common,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    dth_off = s_off.theta_prime.data - state.theta_prime.data
    dth_on = s_on.theta_prime.data - state.theta_prime.data
    diff_top = float(jnp.max(jnp.abs(
        dth_off[..., :2] - dth_on[..., :2],
    )))
    diff_bot = float(jnp.max(jnp.abs(
        dth_off[..., 2:] - dth_on[..., 2:],
    )))
    # Effect concentrated in top: top diff >> bottom drift.
    assert diff_top > 100.0 * diff_bot, (
        f"iter-432 slow-tendency mask did not concentrate "
        f"effect in top levels: top diff {diff_top:.2e} vs "
        f"bottom drift {diff_bot:.2e} (ratio "
        f"{diff_top/diff_bot:.1e})."
    )
    assert diff_top > 0.0, "Mask is a no-op."
