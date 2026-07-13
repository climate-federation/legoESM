"""FV3_3D iter 440: NH mirror of iter-438/439 corner-div sponge
boost.

Adds ``corner_div_damp_d2_bg_k1`` + ``corner_div_damp_d2_bg_k2``
fields to ``CDGridCompressibleEulerConfig``.  Wired via the
shared helper ``_apply_top_sponge_damp_boost`` at BOTH NH corner-
div sites (the adaptive |delpc|·dt branch and the Smagorinsky
+ nord branch — both consume the same ``_damp_corner`` field
that we override per-level).

FV3 reference: ``dyn_core.F90:780, 792, 802``.

Tests
-----

1. ``test_default_value_is_zero`` — both fields default to 0.
2. ``test_baseline_equals_no_field`` — explicit 0 matches
   default (no field set).
3. ``test_k1_boost_changes_state_at_top`` — non-zero k1 changes
   state.
4. ``test_k2_boost_above_005_overrides_k1_and_k2`` — k2 boost
   above 0.05 affects both k=1 + k=2 sponge levels.
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
    rng = np.random.default_rng(seed=440)
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


def test_default_value_is_zero():
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.corner_div_damp_d2_bg_k1 == 0.0
    assert cfg.corner_div_damp_d2_bg_k2 == 0.0


def test_baseline_equals_no_field():
    grid, hc, tm, state = _build_c8()
    cfg_d = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
    )
    cfg_z = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=0.0,
        corner_div_damp_d2_bg_k2=0.0,
    )
    m_d = CDGridCompressibleEulerModel(grid, hc, tm, cfg_d)
    m_z = CDGridCompressibleEulerModel(grid, hc, tm, cfg_z)
    s_d = m_d.step(state, dt=10.0)
    s_z = m_z.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.u.data), np.asarray(s_z.u.data),
        rtol=1e-14, atol=1e-14,
    )


def test_k1_boost_changes_state_at_top():
    grid, hc, tm, state = _build_c8()
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.u.data - s_on.u.data)
    diff_top = float(jnp.max(diff[..., 0]))
    assert diff_top > 0.0, "NH k1 boost is a no-op."


def test_k2_boost_above_005_overrides_k1_and_k2():
    grid, hc, tm, state = _build_c8()
    cfg_off = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k2=2.0,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.u.data - s_on.u.data)
    diff_k1 = float(jnp.max(diff[..., 1]))
    diff_k2 = float(jnp.max(diff[..., 2]))
    assert diff_k1 > 0.0, "NH k1 (d2_bg_k2 path) override no-op."
    assert diff_k2 > 0.0, "NH k2 (d2_bg_k2 path) override no-op."
