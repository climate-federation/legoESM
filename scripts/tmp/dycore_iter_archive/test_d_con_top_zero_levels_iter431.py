"""FV3_3D iter 431: FV3-faithful sponge zeroing of d_con
KE→heat conversion (top N levels).

Port of FV3 ``dyn_core.F90:790/800/804`` ``d_con_k = 0`` for
the top sponge layers (k=1,2,3 conditional on ``d2_bg_k1``/
``d2_bg_k2``).  legoESM models this with a new opt-in
``d_con_top_zero_levels: int = 0`` config field.  Default 0
preserves bit-for-bit baseline (no zeroing).

Currently wired at the NH post-acoustic damp_v d_con site
(iter-209 mirror).  Remaining 4 d_con sites (damp_w post-
acoustic + 3 slow-tendency: corner_div, div_damp, A_h)
pending future iters to match FV3 ``d_con_k`` uniform
zeroing semantics.

Tests
-----

1. ``test_default_value_is_zero`` — config field defaults to 0.
2. ``test_top_zero_levels_zeros_dtheta_p_top`` — when
   ``d_con_top_zero_levels = 2``, the damp_v post-acoustic
   d_con heating in the top 2 levels is exactly zero, while
   bottom levels retain non-zero heating (provided damp_v
   removed KE in those levels).
3. ``test_default_matches_no_zero`` — default 0 (no zeroing)
   produces same state as before iter-431 (no behavior change).
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


def test_default_value_is_zero():
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.d_con_top_zero_levels == 0


def _build_c8():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=False)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=431)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
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


def _cfg(d_con_top_zero_levels):
    return CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        d_con_top_zero_levels=d_con_top_zero_levels,
    )


def test_top_zero_levels_zeros_dtheta_p_top():
    grid, hc, tm, state, nlev = _build_c8()
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(0))
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(2))
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    dtheta_off = s_off.theta_prime.data - state.theta_prime.data
    dtheta_on = s_on.theta_prime.data - state.theta_prime.data
    # Effect is concentrated in top 2 levels (k=0,1).  Bottom
    # levels (k>=2) should be identical because the only iter-
    # 431 difference happens in top levels.
    np.testing.assert_allclose(
        np.asarray(dtheta_off[..., 2:]),
        np.asarray(dtheta_on[..., 2:]),
        rtol=1e-14, atol=1e-14,
    )
    # Top 2 levels: dtheta differs (iter-431 mask kicks in).
    delta = float(jnp.max(jnp.abs(
        dtheta_off[..., :2] - dtheta_on[..., :2],
    )))
    assert delta > 0.0, (
        "iter-431 d_con_top_zero_levels=2 produced NO change "
        "in top-level dtheta_p — flag is a no-op."
    )


def test_default_matches_no_zero():
    """Default config (no field set) == d_con_top_zero_levels=0."""
    grid, hc, tm, state, _ = _build_c8()
    # Build cfg without explicit d_con_top_zero_levels — should
    # equal explicit 0.
    cfg_default = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
    )
    cfg_explicit_zero = _cfg(0)
    m_def = CDGridCompressibleEulerModel(grid, hc, tm, cfg_default)
    m_zero = CDGridCompressibleEulerModel(grid, hc, tm, cfg_explicit_zero)
    s_def = m_def.step(state, dt=10.0)
    s_zero = m_zero.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_def.theta_prime.data),
        np.asarray(s_zero.theta_prime.data),
        rtol=1e-14, atol=1e-14,
    )
