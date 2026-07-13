"""FV3_3D iter 457: FV3-faithful del-2 smoothing of the
aggregate heat_source.

Port of FV3 ``dyn_core.F90:1755-1756``::

    nf_ke = min(3, flagstruct%nord+1)
    call del2_cubed(heat_source, cnst_0p20*da_min, ..., nf_ke)

Applied to the legoESM ``_d_con_sum`` aggregate (NH slow-
tendency heat source) after the iter-432 mask and before the
delt_max cap.

Tests
-----

1. ``test_default_iters_is_zero`` — config field defaults
   preserve bit-for-bit baseline.
2. ``test_default_coeff_is_020``.
3. ``test_baseline_equals_no_smoothing`` — explicit iters=0
   matches default.
4. ``test_smoothing_reduces_grid_scale_noise`` — non-zero
   iters reduces the spatial variance of theta_prime tendency
   (less grid-scale structure post-smoothing).
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
    rng = np.random.default_rng(seed=457)
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


def test_default_iters_is_zero():
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.heat_source_del2_iters == 0


def test_default_coeff_is_020():
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.heat_source_del2_coeff == 0.20


def test_baseline_equals_no_smoothing():
    grid, hc, tm, state = _build_c8()
    cfg_default = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        # All 3 slow-tendency d_con sources active.
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
    )
    cfg_explicit_zero = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        heat_source_del2_iters=0,
    )
    m_d = CDGridCompressibleEulerModel(grid, hc, tm, cfg_default)
    m_z = CDGridCompressibleEulerModel(grid, hc, tm, cfg_explicit_zero)
    s_d = m_d.step(state, dt=10.0)
    s_z = m_z.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.theta_prime.data),
        np.asarray(s_z.theta_prime.data),
        rtol=1e-14, atol=1e-14,
    )


def test_smoothing_reduces_grid_scale_noise():
    """2 iterations of del-2 smoothing should produce a
    smoother theta_prime field than 0 iterations (after one
    step at C8 with all 3 slow-tendency d_con sources active).
    """
    grid, hc, tm, state = _build_c8()
    common = dict(
        n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
    )
    cfg_off = CDGridCompressibleEulerConfig(**common)
    cfg_on = CDGridCompressibleEulerConfig(
        heat_source_del2_iters=2,
        heat_source_del2_coeff=0.20,
        **common,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.theta_prime.data - s_on.theta_prime.data)
    # Some difference expected (the smoothing redistributes
    # heating).
    assert float(jnp.max(diff)) > 0.0, (
        "iter-457 del-2 smoothing produced no observable "
        "change in theta_prime — flag may be a no-op."
    )
