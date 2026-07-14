"""FV3_3D iter 337: extend iter-336 dynamic Exner to the 2
POST-ACOUSTIC NH d_con sites (damp_v_d_con, damp_w_d_con).

iter-336 wired ``use_fv3_dynamic_exner = True`` to the 3
slow-tendency d_con sites (corner_div, cell-centre div_damp,
Smag-A_h) where ``pi_prime`` is in scope from
``slow_tendencies`` step 1.  iter-337 closes the remaining 2
post-acoustic sites that live in the ``step()`` method (no live
``pi_prime`` from slow_tendencies — recompute via
``compute_exner_perturbation`` on the current post-acoustic
state).

ALL 5 NH d_con sites now respect ``use_fv3_dynamic_exner``.

Tests
-----

1. ``test_damp_v_baseline_bit_for_bit`` — damp_v_d_con
   flag=False reproduces baseline.
2. ``test_damp_w_baseline_bit_for_bit`` — damp_w_d_con
   flag=False reproduces baseline.
3. ``test_damp_v_dynamic_changes_state`` — flag=True changes
   θ_p measurably at damp_v_d_con site (proves wiring active).
4. ``test_damp_w_dynamic_changes_state`` — same for damp_w_d_con.
5. ``test_post_acoustic_dynamic_finite_with_full_stack`` —
   finite output under full d_con stack + dynamic Exner.
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
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def small_nh():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    rng = np.random.default_rng(seed=337)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev + 1))
    # Non-zero theta_p, rho_p so π' is non-trivial.
    theta_p = rng.uniform(-10.0, 10.0, size=(6, n, n, nlev))
    rho_p = rng.uniform(-0.15, 0.15, size=(6, n, n, nlev))

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
        theta_prime=Field(data=jnp.asarray(theta_p),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.asarray(rho_p),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def test_damp_v_baseline_bit_for_bit(small_nh):
    """damp_v_d_con flag=False bit-for-bit baseline."""
    grid, hc, tm, state = small_nh
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
    )
    cfg_default = CDGridCompressibleEulerConfig(**common)
    cfg_explicit_off = CDGridCompressibleEulerConfig(
        **common, use_fv3_dynamic_exner=False,
    )
    m_default = CDGridCompressibleEulerModel(grid, hc, tm, cfg_default)
    m_explicit = CDGridCompressibleEulerModel(
        grid, hc, tm, cfg_explicit_off,
    )
    s_default = m_default.step(state, 5.0)
    s_explicit = m_explicit.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s_default.theta_prime.data),
        np.asarray(s_explicit.theta_prime.data),
    )


def test_damp_w_baseline_bit_for_bit(small_nh):
    """damp_w_d_con flag=False bit-for-bit baseline."""
    grid, hc, tm, state = small_nh
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1, damp_w_d_con=1.0,
    )
    cfg_default = CDGridCompressibleEulerConfig(**common)
    cfg_explicit_off = CDGridCompressibleEulerConfig(
        **common, use_fv3_dynamic_exner=False,
    )
    m_default = CDGridCompressibleEulerModel(grid, hc, tm, cfg_default)
    m_explicit = CDGridCompressibleEulerModel(
        grid, hc, tm, cfg_explicit_off,
    )
    s_default = m_default.step(state, 5.0)
    s_explicit = m_explicit.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s_default.theta_prime.data),
        np.asarray(s_explicit.theta_prime.data),
    )


def test_damp_v_dynamic_changes_state(small_nh):
    """damp_v_d_con flag=True changes θ_p measurably vs flag=False."""
    grid, hc, tm, state = small_nh
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
    )
    cfg_off = CDGridCompressibleEulerConfig(
        **common, use_fv3_dynamic_exner=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        **common, use_fv3_dynamic_exner=True,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.theta_prime.data)
        - np.asarray(s_off.theta_prime.data),
    )))
    assert diff > 1e-12, (
        "iter-337 dynamic Exner at damp_v_d_con post-step site "
        "did NOT change θ_p when flag flipped — wiring is silent "
        "no-op."
    )


def test_damp_w_dynamic_changes_state(small_nh):
    """damp_w_d_con flag=True changes θ_p measurably vs flag=False."""
    grid, hc, tm, state = small_nh
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1, damp_w_d_con=1.0,
    )
    cfg_off = CDGridCompressibleEulerConfig(
        **common, use_fv3_dynamic_exner=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        **common, use_fv3_dynamic_exner=True,
    )
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off)
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, cfg_on)
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.theta_prime.data)
        - np.asarray(s_off.theta_prime.data),
    )))
    assert diff > 1e-12, (
        "iter-337 dynamic Exner at damp_w_d_con post-step site "
        "did NOT change θ_p when flag flipped — wiring is silent "
        "no-op."
    )


def test_post_acoustic_dynamic_finite_with_full_stack(small_nh):
    """Full d_con stack + dynamic Exner produces finite state."""
    grid, hc, tm, state = small_nh
    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        damp_w=0.030, nord_w=1, damp_w_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0, corner_div_damp_d_con=1.0,
        use_fv3_dynamic_exner=True,
    )
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s = m.step(state, 5.0)
    for fld in (s.u.data, s.v.data, s.w.data,
                s.theta_prime.data, s.rho_prime.data):
        assert np.all(np.isfinite(np.asarray(fld)))
