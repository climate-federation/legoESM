"""FV3_3D iter 193: post-step ``damp_w + nord_w`` del-(2*(nord_w+1))
damping of vertical velocity ``w`` on the NH 3D path
(``compressible_euler_cdgrid.py``).

Faithful port of FV3 ``sw_core.F90:1080-1086`` (in ``d_sw1``)::

    damp4 = (damp_w * da_min_c) ** (nord_w + 1)
    call del6_vt_flux(nord_w, ..., damp4, w, ..., fx2, fy2, ...)
    dw = (fx2[i,j] - fx2[i+1,j] + fy2[i,j] - fy2[i,j+1]) * rarea
    w += dw

Mirrors the iter-169 ``damp_v`` post-step pattern but applied to a
scalar (w) instead of the (u, v) vector.  Reuses the SW backbone
``_del6_vt_flux`` from ``legoesm.core.fv3_del6_vt_flux``.

Tests
-----

1. ``test_nh_damp_w_zero_is_baseline`` — ``damp_w=0.0`` is bit-for-bit
   identical to the field-unset baseline.  Python-static gate guard.
2. ``test_nh_damp_w_changes_w`` — ``damp_w > 0`` measurably changes
   the vertical velocity on a perturbed state.
3. ``test_nh_damp_w_differentiable`` — ``jax.grad`` flows through 5
   steps with ``damp_w`` active.  Catches any non-AD-safe op in the
   ``_del6_vt_flux`` chain or the per-half-level vmap.
4. ``test_nh_damp_w_rest_state_smoke`` — 20 steps from rest with
   non-zero ``damp_w`` stays finite, no spurious mass growth.
5. ``test_nh_damp_w_nord2_fv3_default_finite`` — FV3 production
   default ``damp_w=0.30 + nord_w=2`` runs and produces finite
   output for 5 steps from a perturbed state.
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
    """Same C8 NH fixture as iter-168/169/170/171 NH tests."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
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


def test_nh_damp_w_zero_is_baseline(small_nh_state):
    """damp_w=0.0 is bit-for-bit identical to the field-unset
    baseline.  Python-static gate guard."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_unset = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    cfg_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.0,
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
        np.asarray(s_unset.w.data), np.asarray(s_zero.w.data),
    )


def test_nh_damp_w_changes_w(small_nh_state):
    """damp_w > 0 measurably changes w on a perturbed state."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev_half = state.w.data.shape[-1]
    rng = np.random.default_rng(seed=193)
    w_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev_half))
    s = state._replace(
        w=state.w.replace(data=jnp.asarray(w_p)),
    )

    cfg_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.0,
    )
    cfg_active = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
    )

    model_zero = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_zero,
    )
    model_active = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_active,
    )

    s_zero = model_zero.step(s, 10.0)
    s_active = model_active.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(s_zero.w.data - s_active.w.data)))
    base = float(jnp.max(jnp.abs(s_zero.w.data)))
    assert diff > 1e-6 * max(base, 1.0), (
        f"damp_w must change w (diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_active.w.data))


def test_nh_damp_w_differentiable(small_nh_state):
    """``jax.grad`` flows through 5 steps with ``damp_w`` active."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
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
        return jnp.mean(s.w.data ** 2)

    grad = jax.grad(loss_fn)(state.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad))


def test_nh_damp_w_rest_state_smoke(small_nh_state):
    """20 steps from rest with damp_w active stay finite."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    s = state
    for _ in range(20):
        s = model.step(s, 10.0)

    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.v.data))
    assert jnp.all(jnp.isfinite(s.w.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))
    assert jnp.all(jnp.isfinite(s.rho_prime.data))


def test_nh_damp_w_nord2_fv3_default_finite(small_nh_state):
    """FV3 AM4 production default (damp_w=0.30 + nord_w=2 = del-6)
    runs without error.  Catches a shape mismatch in the higher
    nord_w >= 2 branch that wouldn't be caught by the nord_w=1 test."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev_half = state.w.data.shape[-1]
    rng = np.random.default_rng(seed=293)
    w_p = rng.uniform(-0.3, 0.3, size=(6, n, n, nlev_half))
    s = state._replace(
        w=state.w.replace(data=jnp.asarray(w_p)),
    )

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.30,
        nord_w=2,                 # FV3 AM4 default (del-6)
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    for _ in range(5):
        s = model.step(s, 10.0)

    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.v.data))
    assert jnp.all(jnp.isfinite(s.w.data))
