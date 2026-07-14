"""FV3_3D iter 170: FV3-faithful 4th-order A→B interpolation for
ζ_corner on the non-hydrostatic 3D path.

Mirrors the iter-14 PE-side wiring; reuses the SW backbone helper
``interp_center_to_corner_a2b_ord4`` (port of FV3
``a2b_edge.F90:a2b_ord4``).

Tests
-----
1. Default ``use_fv3_a2b_zeta_corner=False`` (and field unset) is
   bit-for-bit baseline — Python-static gate.
2. ``use_fv3_a2b_zeta_corner=True`` measurably changes wind state
   on a perturbed input.
3. ``jax.grad`` flows through 5 steps with the flag active.
4. Rest state stays finite for 20 steps with flag active.
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
    """Same fixture as the iter-168 / iter-169 NH tests."""
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


def test_nh_a2b_zeta_corner_off_is_baseline(small_nh_state):
    """use_fv3_a2b_zeta_corner=False matches the field-unset baseline
    bit-for-bit — Python-static gate guard."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_unset = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    cfg_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_a2b_zeta_corner=False,
    )

    model_unset = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_unset,
    )
    model_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_off,
    )

    s_unset = state
    s_off = state
    for _ in range(5):
        s_unset = model_unset.step(s_unset, 10.0)
        s_off = model_off.step(s_off, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_unset.u.data), np.asarray(s_off.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_unset.v.data), np.asarray(s_off.v.data),
    )


def test_nh_a2b_zeta_corner_on_changes_winds(small_nh_state):
    """use_fv3_a2b_zeta_corner=True measurably changes winds on a
    perturbed state."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=170)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_a2b_zeta_corner=False,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_a2b_zeta_corner=True,
    )

    model_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_off,
    )
    model_on = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_on,
    )

    s_off = model_off.step(s, 10.0)
    s_on = model_on.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(s_off.u.data - s_on.u.data)))
    base = float(jnp.max(jnp.abs(s_off.u.data)))
    assert diff > 1e-6 * base, (
        f"a2b_zeta_corner must change winds (diff={diff:.3e}, "
        f"base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_on.u.data))
    assert jnp.all(jnp.isfinite(s_on.v.data))


def test_nh_a2b_zeta_corner_differentiable(small_nh_state):
    """``jax.grad`` flows through 5 steps with a2b ζ corner active."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_a2b_zeta_corner=True,
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


def test_nh_a2b_zeta_corner_rest_state_smoke(small_nh_state):
    """Rest-state stability over 20 steps with a2b ζ corner active."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        use_fv3_a2b_zeta_corner=True,
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
