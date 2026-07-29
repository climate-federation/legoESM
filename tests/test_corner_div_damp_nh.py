"""FV3_3D iter 168: corner-divergence damping on the non-hydrostatic
3D path (``compressible_euler_cdgrid.py``).

Mirrors the iter-16 / iter-18 tests in ``tests/test_div_damp_adaptive.py``
which cover the same FV3 mechanism on the hydrostatic PE 3D path
(``primitive_eq_cdgrid.py``).

Faithful to FV3 ``sw_core.F90:1641-1822`` (subroutine ``d_sw5``)::

    delpc       = corner divergence (B-grid, sin_sg + corner removal)
    damp        = da_min_c * max(d2_bg, min(0.20, dddmp*|delpc|*dt))
    divg_d_iter = (Laplacian)^nord(delpc)            [if nord > 0]
    dd8         = (da_min_c * d4_bg) ** (nord + 1)   [if d4_bg > 0]
    ke_corr     = damp * delpc + dd8 * divg_d_iter
    du -= grad_x(ke_corr) ; dv -= grad_y(ke_corr)

Tests
-----
1. Default ``corner_div_damp_d2_bg=0.0`` (and the field unset) produces
   bit-for-bit identical output to the existing baseline — Python-static
   gate guard.
2. ``corner_div_damp_d2_bg > 0`` measurably changes wind tendencies on
   a perturbed state.
3. Higher-order branch (``d4_bg > 0`` AND ``nord == 0``) is bit-for-bit
   identical to the d2-only path — Python-static gate guard for the
   iter-18-equivalent block.
4. Differentiability: ``jax.grad`` flows through 5 steps with the
   damping active.
5. Smoke: rest state remains finite after 20 steps with non-zero
   ``corner_div_damp_d2_bg`` — no catastrophic NaN.
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
    """A small (n=8, nlev=5) NH state at rest, with terrain-following
    metric.  Mirrors the fixture in
    ``tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py``.
    """
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


def test_nh_corner_div_damp_zero_is_baseline(small_nh_state):
    """corner_div_damp_d2_bg=0.0 is bit-for-bit identical to the
    field-unset baseline — Python-static gate guard."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_unset = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    cfg_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0,
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


def test_nh_corner_div_damp_changes_winds(small_nh_state):
    """corner_div_damp_d2_bg > 0 measurably changes the momentum
    tendencies on a perturbed state."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=33)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0,
    )
    cfg_active = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
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
        f"corner_div_damp must change winds (diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_active.u.data))
    assert jnp.all(jnp.isfinite(s_active.v.data))


def test_nh_corner_div_damp_d4_disabled_bit_for_bit_with_d2(small_nh_state):
    """When d4_bg > 0 but nord == 0 the higher-order block is gated
    off; output matches the d2-only path bit-for-bit.  Mirror of the
    iter-18 PE-side guard."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_d2_only = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
    )
    cfg_nord0 = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.16,    # set, but nord=0 disables
        corner_div_damp_nord=0,
    )

    model_d2 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d2_only,
    )
    model_nord0 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_nord0,
    )

    s_d2 = model_d2.step(state, 10.0)
    s_nord0 = model_nord0.step(state, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_d2.u.data), np.asarray(s_nord0.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_d2.v.data), np.asarray(s_nord0.v.data),
    )


def test_nh_corner_div_damp_del4_active_without_d2(small_nh_state):
    """The del-4 pair (d4_bg>0, nord>0) activates the corner damping
    WITHOUT d2_bg — the standard FV3 configuration and the exact matrix
    NH TC1/TC2/TC3 config family (d2_bg=0, nord=1, d4_bg=0.16).

    Regression for the 2026-07-29 DCMIP TC2/TC3 cube vertex blow-up:
    the old ``d2_bg > 0`` master gate (absent in FV3 sw_core.F90:1641,
    which branches only on ``nord``) left these configs with ZERO
    corner damping — this test is bit-identical-therefore-RED on that
    gate."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=34)
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(
            rng.uniform(-3.0, 3.0, size=(6, n, n, nlev)))),
        v=state.v.replace(data=jnp.asarray(
            rng.uniform(-3.0, 3.0, size=(6, n, n, nlev)))),
    )

    cfg_inert = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    cfg_del4 = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.16,
        corner_div_damp_nord=1,
    )

    model_inert = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_inert,
    )
    model_del4 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_del4,
    )

    s_inert = model_inert.step(s, 10.0)
    s_del4 = model_del4.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(s_inert.u.data - s_del4.u.data)))
    base = float(jnp.max(jnp.abs(s_inert.u.data)))
    assert diff > 1e-6 * base, (
        "del-4 corner damping (d2_bg=0, d4_bg=0.16, nord=1) must be "
        f"ACTIVE — bit-identical output means the d2_bg master gate is "
        f"back (diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_del4.u.data))
    assert jnp.all(jnp.isfinite(s_del4.v.data))


def test_corner_div_damp_activation_helpers():
    """Pure-helper truth table: activation matches FV3 semantics
    (d2_bg>0 OR (d4_bg>0 AND nord>0)); dddmp alone stays inert
    (deliberate deviation — its default 0.20 would flip every
    all-zero config)."""
    from legoesm.core._fv3_divergence_corner import (
        corner_div_damp_active,
        corner_div_damp_del4_active,
    )

    def cfg(d2=0.0, d4=0.0, nord=0, dddmp=0.20):
        return CDGridCompressibleEulerConfig(
            corner_div_damp_d2_bg=d2, corner_div_damp_d4_bg=d4,
            corner_div_damp_nord=nord, corner_div_damp_dddmp=dddmp,
        )

    assert not corner_div_damp_active(cfg())                      # all-inert
    assert corner_div_damp_active(cfg(d2=0.001))                  # d2 path
    assert corner_div_damp_active(cfg(d4=0.16, nord=1))           # del-4 pair
    assert not corner_div_damp_active(cfg(d4=0.16, nord=0))       # pair broken
    assert not corner_div_damp_active(cfg(nord=1))                # pair broken
    assert not corner_div_damp_active(cfg(dddmp=0.5))             # dddmp alone
    assert corner_div_damp_del4_active(cfg(d4=0.16, nord=1))
    assert not corner_div_damp_del4_active(cfg(d2=0.001))


def test_nh_corner_div_damp_differentiable(small_nh_state):
    """``jax.grad`` flows through 5 steps with the damping active.

    Same differentiability check as the existing
    ``test_differentiable_10_steps`` (NH baseline) but exercising the
    new code path.  Catches any non-AD-safe call inside
    ``fv3_divergence_corner_3d`` / ``fv3_corner_laplacian_iteration``
    or the new ``_pad_halo_4d_module`` invocation under reverse-mode
    autodiff.
    """
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
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


def test_nh_corner_div_damp_rest_state_smoke(small_nh_state):
    """Rest-state stability with damping active: 20 steps, finite +
    no spurious mass growth."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
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
