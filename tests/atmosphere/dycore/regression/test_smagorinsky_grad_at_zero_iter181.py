"""FV3_3D iter 181: differentiability of compute_smagorinsky_ah at
zero strain.

iter 180 noted that ``compute_smagorinsky_ah_2d`` had a singular
gradient at zero strain (``d sqrt(strain_mag_sq) / d strain_mag_sq``
is infinite at strain=0), which made the iter-180 NH
differentiability test fail at rest-state IC.  iter 180 worked
around this by perturbing the IC.

iter 181 fixes the helper using the JAX "double-where" trick so
the gradient is finite at zero strain.  The forward pass remains
bit-for-bit identical at zero strain (returns exactly 0; pinned
by the existing ``test_smagorinsky_zero_winds`` test).

Tests
-----
1. ``test_smag_grad_finite_at_zero_strain`` — ``jax.grad`` of a
   loss function that depends on the Smagorinsky output at the
   zero-strain input gives finite gradients (was NaN before
   iter 181).
2. ``test_smag_grad_finite_on_partial_zero_strain`` — same with
   half the cells zero strain and half non-zero; the per-cell
   gradient must be finite everywhere.
3. ``test_smag_forward_at_zero_winds_still_zero`` — sanity:
   the forward-pass bit-for-bit preservation at zero winds
   (existing iter-58 contract) is not broken by the iter-181
   gradient fix.
4. ``test_nh_smag_differentiable_at_rest`` — model-level
   differentiability through the NH path with smag ON, starting
   from EXACTLY the rest state (no perturbation).  Was the
   workaround motivation in iter 180.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._smagorinsky_visc import compute_smagorinsky_ah_2d
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


@pytest.fixture(scope="module")
def small_cube():
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return grid, cdgrid, n


def test_smag_grad_finite_at_zero_strain(small_cube):
    """``jax.grad`` of a loss that depends on the Smagorinsky output
    at zero-strain input gives finite gradients.  Before iter 181
    this returned NaN (sqrt(0) singularity)."""
    _, cdgrid, n = small_cube

    def loss(uv):
        u, v = uv[0], uv[1]
        ah = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.2)
        return jnp.sum(ah ** 2)

    uv0 = jnp.zeros((2, 6, n + 1, n + 1))    # rest state
    grad = jax.grad(loss)(uv0)
    assert jnp.all(jnp.isfinite(grad)), (
        "Smagorinsky gradient at zero strain must be finite "
        "(iter 181 fix).  NaN here indicates the double-where trick "
        "is not active or the safe_strain_sq fallback is not >= 1."
    )
    # At zero strain, the loss = 0 and the gradient should also be
    # 0 (small change in u/v at zero strain produces negligible
    # ah, so dL/du = 0).  The double-where masks the gradient to 0.
    np.testing.assert_array_equal(
        np.asarray(grad), np.zeros_like(grad),
    )


def test_smag_grad_finite_on_partial_zero_strain(small_cube):
    """Half the cells have zero strain (zero u/v), half have
    nonzero.  The per-cell gradient must be finite everywhere
    (no NaN propagation from zero-strain cells to nonzero cells)."""
    _, cdgrid, n = small_cube

    def loss(uv):
        u, v = uv[0], uv[1]
        ah = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.2)
        return jnp.sum(ah ** 2)

    rng = np.random.default_rng(seed=181)
    u = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)))
    v = jnp.asarray(rng.uniform(-1.0, 1.0, size=(6, n + 1, n + 1)))
    # Zero out the first three faces
    u = u.at[:3].set(0.0)
    v = v.at[:3].set(0.0)
    uv = jnp.stack([u, v])

    grad = jax.grad(loss)(uv)
    assert jnp.all(jnp.isfinite(grad)), (
        "Mixed zero / nonzero strain cells must produce finite "
        "gradient everywhere — NaN in zero cells would propagate "
        "via reductions to the rest of the gradient."
    )


def test_smag_forward_at_zero_winds_still_zero(small_cube):
    """Sanity check: the iter-58 contract that ah=0 at zero winds
    is preserved by the iter-181 gradient fix.  Mirrors the
    existing ``test_smagorinsky_zero_winds`` but explicitly
    verifies the iter-181 change didn't break it."""
    _, cdgrid, n = small_cube
    u = jnp.zeros((6, n + 1, n + 1))
    v = jnp.zeros((6, n + 1, n + 1))
    ah = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.2)
    np.testing.assert_array_equal(
        np.asarray(ah), np.zeros((6, n + 1, n + 1)),
    )


def test_nh_smag_differentiable_at_rest():
    """Model-level differentiability through the NH path with smag
    ON, starting from the EXACT rest state (no IC perturbation).
    This was the iter-180 NH differentiability failure mode that
    iter 181 fixes."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.vertical import (
        create_height_coordinate, compute_terrain_metric,
    )

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

    state_rest = NonHydrostaticState(
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

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6, smagorinsky_cs=0.20,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state_rest._replace(
            theta_prime=state_rest.theta_prime.replace(data=theta_p_data),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.mean(s.u.data ** 2 + s.v.data ** 2)

    grad = jax.grad(loss_fn)(state_rest.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad)), (
        "AD through NH with smag ON at rest state must be finite "
        "after iter 181 fix to compute_smagorinsky_ah_2d "
        "(was NaN in iter 180 due to sqrt(0) singularity)."
    )
