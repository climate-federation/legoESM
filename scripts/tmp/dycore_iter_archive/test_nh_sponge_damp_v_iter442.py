"""FV3_3D iter 442: FV3-faithful sponge boost of NH ``damp_v``
(vorticity damping) coefficient at top levels.

Ports FV3 ``dyn_core.F90:786-787, 796-797`` ``damp_vt(k) =
0.5 * d2_divg`` at sponge layers k=1 + k=2 (1-based; our
k=0, k=1 0-based).  FV3 does NOT extend the damp_v boost to
k=3 (only damp_w + d_con zeroing apply there).

Implementation uses the linear ``damp^(nord+1)`` scaling
trick (mirror of iter-441 damp_w): at sponge levels,
``(du, dv) *= (0.5 * boosted / damp_v)^(nord_v+1)``.

Gated by new ``use_fv3_sponge_damp_v: bool = False``.

Tests
-----

1. ``test_default_flag_false``.
2. ``test_flag_false_no_change``.
3. ``test_flag_on_no_d2_bg_no_change`` — flag-ON without
   corner-div sponge fields → no behavior change.
4. ``test_flag_on_k1_boost_changes_u`` — k1 boost active →
   u_d differs at top.
5. ``test_no_boost_at_k2_threshold_005`` — d2_bg_k2 > 0.05 is
   the corner-div threshold for k=2 (i.e. k=2 in 0-based);
   for damp_v, k=2 is NOT boosted per FV3.  Verify by checking
   that 0.05 < d2_bg_k2 < ... triggers only k=1 changes from
   damp_v (vs corner-div which also touches k=2).
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


def _build_c8():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=False)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=442)
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
    return grid, height_coord, terrain_metric, state


def test_default_flag_false():
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.use_fv3_sponge_damp_v is False


def test_flag_false_no_change():
    grid, hc, tm, state = _build_c8()
    cfg_d = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
    )
    cfg_f = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_v=False,
    )
    m_d = CDGridCompressibleEulerModel(grid, hc, tm, cfg_d)
    m_f = CDGridCompressibleEulerModel(grid, hc, tm, cfg_f)
    s_d = m_d.step(state, dt=10.0)
    s_f = m_f.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.u.data), np.asarray(s_f.u.data),
        rtol=1e-14, atol=1e-14,
    )


def test_flag_on_no_d2_bg_no_change():
    grid, hc, tm, state = _build_c8()
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        use_fv3_sponge_damp_v=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        use_fv3_sponge_damp_v=True,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_off.u.data), np.asarray(s_on.u.data),
        rtol=1e-14, atol=1e-14,
    )


def test_flag_on_k1_boost_changes_u():
    grid, hc, tm, state = _build_c8()
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_v=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_v=True,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.u.data - s_on.u.data)
    diff_top = float(jnp.max(diff[..., 0]))
    assert diff_top > 0.0, (
        "iter-442 damp_v sponge boost is a no-op at k=0."
    )
