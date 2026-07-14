"""FV3_3D iter 441: FV3-faithful sponge boost of NH ``damp_w``
coefficient at top levels.

Ports FV3 ``dyn_core.F90:782, 793, 803`` ``damp_w = d2_divg``
at sponge layers — meaning damp_w receives the SAME boosted
coefficient as the corner-divergence damping at those levels.

Implementation uses the linear ``damp^(nord+1)`` scaling trick:
``dw_top_sponge *= (boosted_damp_w / damp_w)^(nord+1)``.
Exact equivalent of recomputing the del-n flux with the
boosted coefficient (since del-n flux is linear in damp).

Gated by new ``use_fv3_sponge_damp_w: bool = False``.

Tests
-----

1. ``test_default_flag_false`` — default value False.
2. ``test_flag_false_no_change`` — flag-off matches no-flag.
3. ``test_flag_on_no_d2_bg_no_change`` — flag-ON but
   ``corner_div_damp_d2_bg_k1/k2`` not set → no behaviour
   change (gated on corner-div sponge fields).
4. ``test_flag_on_k1_boost_changes_w`` — flag-ON + non-zero
   k1 boost → w differs at k=0 vs baseline.
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
    rng = np.random.default_rng(seed=441)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev + 1))
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
    return grid, height_coord, terrain_metric, state


def test_default_flag_false():
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.use_fv3_sponge_damp_w is False


def test_flag_false_no_change():
    """Default vs explicit-False matches bit-for-bit."""
    grid, hc, tm, state = _build_c8()
    cfg_d = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
    )
    cfg_f = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_w=False,
    )
    m_d = CDGridCompressibleEulerModel(grid, hc, tm, cfg_d)
    m_f = CDGridCompressibleEulerModel(grid, hc, tm, cfg_f)
    s_d = m_d.step(state, dt=10.0)
    s_f = m_f.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.w.data), np.asarray(s_f.w.data),
        rtol=1e-14, atol=1e-14,
    )


def test_flag_on_no_d2_bg_no_change():
    """Flag-ON but corner-div sponge fields not set → no change."""
    grid, hc, tm, state = _build_c8()
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        corner_div_damp_d2_bg=0.0005,
        use_fv3_sponge_damp_w=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        corner_div_damp_d2_bg=0.0005,
        use_fv3_sponge_damp_w=True,   # but no d2_bg_k* → no effect
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_off.w.data), np.asarray(s_on.w.data),
        rtol=1e-14, atol=1e-14,
    )


def test_flag_on_k1_boost_changes_w():
    grid, hc, tm, state = _build_c8()
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_w=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
        use_fv3_sponge_damp_w=True,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.w.data - s_on.w.data)
    diff_top = float(jnp.max(diff[..., 0]))
    assert diff_top > 0.0, (
        "iter-441 damp_w sponge boost is a no-op at k=0."
    )
