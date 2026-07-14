"""FV3_3D iter 180: FV3-style Smagorinsky-adaptive A_h on the
non-hydrostatic 3D path.

Mirrors the iter-57/58/59 PE-side wiring; reuses the
``compute_smagorinsky_ah_3d`` helper from ``legoesm.core._smagorinsky_visc``.

Tests
-----
1. Default ``smagorinsky_cs=0.0`` (and field unset) is bit-for-bit
   baseline — Python-static gate.
2. ``smagorinsky_cs > 0`` with ``A_h > 0`` measurably changes
   wind state on a perturbed input.
3. ``smagorinsky_cs > 0`` with ``A_h == 0`` is bit-for-bit
   identical to the default — Smagorinsky is added on TOP of
   ``A_h``, so when A_h is off Smagorinsky has no effect.
4. ``jax.grad`` flows through 5 steps with smag ON.
5. Rest state stays finite for 20 steps with smag ON.
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
    """Same fixture as iter-168...-179 NH tests."""
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


def test_nh_smag_zero_is_baseline(small_nh_state):
    """smagorinsky_cs=0.0 with A_h>0 is bit-for-bit identical to
    field-unset baseline — Python-static gate."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_unset = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6,
    )
    cfg_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6, smagorinsky_cs=0.0,
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


def test_nh_smag_changes_winds_when_ah_positive(small_nh_state):
    """smagorinsky_cs > 0 with A_h > 0 measurably changes wind
    state on a perturbed input."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=180)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6, smagorinsky_cs=0.0,
    )
    cfg_smag = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6, smagorinsky_cs=0.20,    # PE-tested useful range
    )

    model_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_off,
    )
    model_smag = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_smag,
    )

    s_off = model_off.step(s, 10.0)
    s_smag = model_smag.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(s_off.u.data - s_smag.u.data)))
    base = float(jnp.max(jnp.abs(s_off.u.data)))
    assert diff > 1e-6 * base, (
        f"smagorinsky_cs=0.20 must change winds when A_h>0 "
        f"(diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_smag.u.data))


def test_nh_smag_no_effect_when_ah_zero(small_nh_state):
    """smagorinsky_cs > 0 with A_h == 0 is bit-for-bit identical
    to the default (smag=0).  The Smagorinsky branch is gated
    inside the ``if config.A_h > 0:`` block; when A_h is off the
    branch isn't reached."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_off = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=0.0, smagorinsky_cs=0.0,
    )
    cfg_smag_no_ah = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=0.0, smagorinsky_cs=0.20,    # set, but A_h=0 disables
    )

    model_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_off,
    )
    model_smag = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_smag_no_ah,
    )

    s_off = state
    s_smag = state
    for _ in range(5):
        s_off = model_off.step(s_off, 10.0)
        s_smag = model_smag.step(s_smag, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_off.u.data), np.asarray(s_smag.u.data),
    )


def test_nh_smag_differentiable(small_nh_state):
    """``jax.grad`` flows through 5 steps with smag ON + A_h ON.

    NOTE: The Smagorinsky helper computes ``sqrt(strain_mag)`` whose
    gradient is singular at zero strain (rest state).  We start from
    a non-rest IC (small u perturbation) so the strain is non-zero
    at every step, avoiding the sqrt-at-zero singularity.  The
    underlying production helper has this property; future work
    could regularise it for differentiable training that touches
    the rest state."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=181)
    u_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev))
    state_perturbed = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
    )

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6, smagorinsky_cs=0.20,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state_perturbed._replace(
            theta_prime=state_perturbed.theta_prime.replace(data=theta_p_data),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.mean(s.u.data ** 2 + s.v.data ** 2)

    grad = jax.grad(loss_fn)(state_perturbed.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad))


def test_nh_smag_rest_state_smoke(small_nh_state):
    """Rest-state stability over 20 steps with smag ON + A_h ON."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6, smagorinsky_cs=0.20,
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
    assert rho_drift < 1.0
