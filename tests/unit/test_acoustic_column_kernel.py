"""Direct tests for ``_acoustic_column_kernel`` extracted from
``compressible_euler.acoustic_substeps``.

Two responsibilities:

1. *Refactor-equivalence:* on a small cubed-sphere-shaped state the
   public ``acoustic_substeps`` returns identical arrays before and
   after the extraction (verified indirectly by ``test_cdgrid.py``
   regression; here we add a direct synthetic call so future tweaks to
   the kernel are caught immediately).
2. *Algorithm soundness:* a single substep on a rest column with zero
   perturbations leaves the column at rest to machine precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.compressible_euler import (
    _acoustic_column_kernel,
    acoustic_substeps,
    CompressibleEulerConfig,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.grids.vertical import create_height_coordinate, TerrainMetric
from legoesm.timestepping.split_explicit import SplitExplicitConfig


jax.config.update("jax_enable_x64", True)


def _column_setup(nlev=8):
    height_coord = create_height_coordinate(nlev, H=20.0e3)
    return height_coord


def test_rest_column_kernel_one_substep_stays_at_rest():
    nlev = 8
    height_coord = _column_setup(nlev)
    w = jnp.zeros((nlev + 1,), dtype=jnp.float64)
    theta_p = jnp.zeros((nlev,), dtype=jnp.float64)
    rho_p = jnp.zeros((nlev,), dtype=jnp.float64)
    J = jnp.array(1.0, dtype=jnp.float64)

    w_new, theta_p_new, rho_p_new = _acoustic_column_kernel(
        w, theta_p, rho_p, height_coord, J, dt_s=1.0, beta=0.0, g=9.81,
    )

    assert jnp.max(jnp.abs(w_new)) == 0.0
    assert jnp.max(jnp.abs(theta_p_new)) == 0.0
    assert jnp.max(jnp.abs(rho_p_new)) == 0.0


def test_kernel_preserves_rigid_w_boundaries():
    """``w_new[0]`` and ``w_new[-1]`` should equal the input boundary
    values regardless of interior perturbations."""
    nlev = 6
    height_coord = _column_setup(nlev)
    w = jnp.zeros((nlev + 1,), dtype=jnp.float64)
    theta_p = jnp.linspace(-0.5, 0.5, nlev, dtype=jnp.float64)
    rho_p = jnp.linspace(-0.01, 0.01, nlev, dtype=jnp.float64)
    J = jnp.array(1.0, dtype=jnp.float64)

    w_new, _, _ = _acoustic_column_kernel(
        w, theta_p, rho_p, height_coord, J, dt_s=0.1, beta=0.0, g=9.81,
    )
    assert float(w_new[0]) == 0.0
    assert float(w_new[-1]) == 0.0


def test_acoustic_substeps_wrapper_calls_kernel_equivalently():
    """Calling the kernel directly inside a fori_loop produces the same
    output as the public ``acoustic_substeps`` wrapper on a single
    horizontal cell. This catches regressions where the wrapper does
    extra work beyond the kernel body."""
    nlev = 6
    height_coord = _column_setup(nlev)
    nface, n = 1, 1
    shape_full = (nface, n, n, nlev)
    shape_half = (nface, n, n, nlev + 1)
    w_data = jnp.zeros(shape_half, dtype=jnp.float64)
    theta_p_data = jnp.full(shape_full, 0.3, dtype=jnp.float64)
    rho_p_data = jnp.full(shape_full, 1.0e-3, dtype=jnp.float64)
    u_data = jnp.zeros(shape_full, dtype=jnp.float64)
    phis_data = jnp.zeros((nface, n, n), dtype=jnp.float64)
    tracers_data = jnp.zeros((nface, n, n, nlev, 0), dtype=jnp.float64)

    state = NonHydrostaticState(
        u=Field(u_data, name="u", dims=("face", "x", "y", "z")),
        v=Field(u_data, name="v", dims=("face", "x", "y", "z")),
        w=Field(w_data, name="w", dims=("face", "x", "y", "z_half")),
        theta_prime=Field(theta_p_data, name="theta_prime"),
        rho_prime=Field(rho_p_data, name="rho_prime"),
        phis=Field(phis_data, name="phis"),
        tracers=Field(tracers_data, name="tracers"),
    )
    slow_tend = NonHydrostaticTendencies(
        du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
        dv_dt=state.v.replace(data=jnp.zeros_like(state.v.data)),
        dw_dt=state.w.replace(data=jnp.zeros_like(state.w.data)),
        dtheta_prime_dt=state.theta_prime.replace(
            data=jnp.zeros_like(state.theta_prime.data)),
        drho_prime_dt=state.rho_prime.replace(
            data=jnp.zeros_like(state.rho_prime.data)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        dtracers_dt=state.tracers.replace(
            data=jnp.zeros_like(state.tracers.data)),
    )

    jacobian = jnp.ones((nface, n, n), dtype=jnp.float64)
    z_full_3d = jnp.broadcast_to(height_coord.z_full, (nface, n, n, nlev))
    z_half_3d = jnp.broadcast_to(height_coord.z_half, (nface, n, n, nlev + 1))
    terrain = TerrainMetric(
        z_s=jnp.zeros((nface, n, n), dtype=jnp.float64),
        jacobian=jacobian,
        z_full_3d=z_full_3d,
        z_half_3d=z_half_3d,
    )
    euler_config = CompressibleEulerConfig(n_acoustic_substeps=1)
    se_config = SplitExplicitConfig()

    out_state = acoustic_substeps(
        state, slow_tend, dt_s=0.05, n_substeps=1, config=se_config,
        height_coord=height_coord, terrain_metric=terrain,
        euler_config=euler_config,
    )

    # Direct kernel call on the single column for comparison.
    w_direct, theta_direct, rho_direct = _acoustic_column_kernel(
        w_data[0, 0, 0], theta_p_data[0, 0, 0], rho_p_data[0, 0, 0],
        height_coord, jacobian[0, 0, 0],
        dt_s=0.05, beta=euler_config.acoustic_off_centering, g=euler_config.g,
    )

    assert jnp.allclose(out_state.w.data[0, 0, 0], w_direct)
    assert jnp.allclose(out_state.theta_prime.data[0, 0, 0], theta_direct)
    assert jnp.allclose(out_state.rho_prime.data[0, 0, 0], rho_direct)
