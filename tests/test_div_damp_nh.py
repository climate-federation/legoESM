"""FV3_3D iter 171: cell-centre divergence damping on the
non-hydrostatic 3D path (``compressible_euler_cdgrid.py``).

Faithful port of FV3 ``sw_core.F90:1720``::

    damp = da_min_c * max(d2_bg, min(0.20, dddmp * |div|))

with ``d2_bg = div_damp_coeff / da_min_c``.  Mirrors the iter-5
PE-side wiring.

Tests
-----
1. Default ``div_damp_coeff=0.0`` (and field unset) is bit-for-bit
   baseline — Python-static gate.
2. ``div_damp_coeff > 0, dddmp = 0`` (constant path) measurably
   changes wind state.
3. ``div_damp_coeff > 0, dddmp > 0`` (adaptive Smag path)
   measurably changes wind state and differs from the constant
   path.
4. ``jax.grad`` flows through 5 steps with damping active.
5. Rest state stays finite for 20 steps with damping active.
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
def small_nh_state():
    """Same fixture as the iter-168/169/170 NH tests."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, grid.n, grid.n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
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
    return grid, height_coord, terrain_metric, state


def test_nh_div_damp_zero_is_baseline(small_nh_state):
    """div_damp_coeff=0.0 is bit-for-bit identical to the field-unset
    baseline — Python-static gate."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_unset = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    cfg_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=0.0, div_damp_dddmp=0.20,  # dddmp set but coeff=0
    )

    model_unset = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_unset,
    )
    model_zero = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_zero,
    )

    s_unset = state
    s_zero = state
    for _ in range(5):
        s_unset = model_unset.step(s_unset, 10.0)
        s_zero = model_zero.step(s_zero, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_unset.u.data), np.asarray(s_zero.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_unset.v.data), np.asarray(s_zero.v.data),
    )


def test_nh_div_damp_constant_changes_winds(small_nh_state):
    """Constant ``div_damp_coeff > 0`` (dddmp=0) measurably changes
    winds on a perturbed state."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=171)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=0.0,
    )
    cfg_active = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e7, div_damp_dddmp=0.0,    # constant path
    )

    model_zero = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_zero,
    )
    model_active = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_active,
    )

    s_zero = model_zero.step(s, 10.0)
    s_active = model_active.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(s_zero.u.data - s_active.u.data)))
    base = float(jnp.max(jnp.abs(s_zero.u.data)))
    assert diff > 1e-6 * base, (
        f"div_damp_coeff must change winds (diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_active.u.data))
    assert jnp.all(jnp.isfinite(s_active.v.data))


def test_nh_div_damp_adaptive_differs_from_constant(small_nh_state):
    """Adaptive Smag path with HUGE dddmp engages the
    ``min(0.20, dddmp*|div|)`` cap and differs measurably from the
    constant path.  Mirrors the PE-side
    ``test_huge_dddmp_changes_tendencies`` pattern: at realistic
    divergence levels the cap does NOT engage, so to verify the
    adaptive code path is wired we force it via a 1e6 dddmp value."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=172)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_const = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e7, div_damp_dddmp=0.0,
    )
    cfg_adapt = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e7, div_damp_dddmp=1.0e6,    # huge — engages cap
    )

    model_const = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_const,
    )
    model_adapt = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_adapt,
    )

    s_const = model_const.step(s, 10.0)
    s_adapt = model_adapt.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(s_const.u.data - s_adapt.u.data)))
    base = float(jnp.max(jnp.abs(s_const.u.data)))
    assert diff > 1e-6 * base, (
        f"adaptive Smag path must differ from constant "
        f"(diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_adapt.u.data))


def test_nh_div_damp_differentiable(small_nh_state):
    """``jax.grad`` flows through 5 steps with adaptive div damp."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e7, div_damp_dddmp=0.20,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state._replace(
            theta_prime=state.theta_prime.replace(data=theta_p_data),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.mean(s.u.data ** 2 + s.v.data ** 2)

    grad = jax.grad(loss_fn)(state.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad))


def test_nh_div_damp_rest_state_smoke(small_nh_state):
    """Rest-state stability over 20 steps with adaptive div damp."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        div_damp_coeff=1e7, div_damp_dddmp=0.20,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    s = state
    for _ in range(20):
        s = model.step(s, 10.0)

    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.v.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))
    assert jnp.all(jnp.isfinite(s.rho_prime.data))
    rho_drift = float(jnp.max(jnp.abs(s.rho_prime.data)))
    assert rho_drift < 1.0, (
        f"rho_prime drift {rho_drift:.2e} too large for rest state"
    )
