"""FV3_3D iter 209: KE→heat conversion (``d_con``) for the iter-169
NH ``damp_v`` post-step damping.  NH mirror of PE iter-208.

When ``damp_v`` removes KE from (u, v) cell-centre winds via the
post-step (du_cc, dv_cc) increments, the lost KE is converted to
heat in θ_p (energy conservation):

    ΔKE_cc = u * du + 0.5*du² + v * dv + 0.5*dv²
    Δθ_p = -damp_v_d_con * ΔKE_cc / (c_pd * Π_ref)

Π_ref comes from ``HeightCoordinate.exner_ref`` (matches iter-207
refinement of the iter-203 damp_w_d_con).

Tests
-----

1. ``test_nh_damp_v_d_con_off_baseline`` — bit-for-bit baseline.
2. ``test_nh_damp_v_d_con_changes_theta_p`` — d_con > 0 changes
   θ_p AND mean θ_p does not decrease.
3. ``test_nh_damp_v_d_con_differentiable_at_rest`` — AD-safe at
   rest state.
4. ``test_nh_damp_v_d_con_no_op_when_damp_v_off`` — gated INSIDE
   damp_v block.
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


def test_nh_damp_v_d_con_off_baseline(small_nh_state):
    """damp_v_d_con=0.0 is bit-for-bit baseline (gated off)."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=209)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_baseline = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
    )
    cfg_d_con_zero = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
    )

    m_baseline = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_baseline,
    )
    m_d_con_zero = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d_con_zero,
    )

    s_baseline = m_baseline.step(s, 10.0)
    s_d_con_zero = m_d_con_zero.step(s, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_baseline.theta_prime.data),
        np.asarray(s_d_con_zero.theta_prime.data),
    )


def test_nh_damp_v_d_con_changes_theta_p(small_nh_state):
    """damp_v_d_con > 0 changes θ_p and mean θ_p does not decrease."""
    grid, height_coord, terrain_metric, state = small_nh_state

    n = grid.n
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=209)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    s = state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
    )

    cfg_no_d_con = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
    )
    cfg_d_con = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
    )

    m_no = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_d_con,
    )
    m_d = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d_con,
    )

    s_no = m_no.step(s, 10.0)
    s_d = m_d.step(s, 10.0)

    diff = float(jnp.max(jnp.abs(
        s_no.theta_prime.data - s_d.theta_prime.data
    )))
    assert diff > 1e-10, (
        f"damp_v_d_con > 0 must change θ_p (diff={diff:.3e})"
    )

    mean_no = float(jnp.mean(s_no.theta_prime.data))
    mean_d = float(jnp.mean(s_d.theta_prime.data))
    assert mean_d + 1e-6 >= mean_no, (
        f"d_con > 0 must not REDUCE mean θ_p: "
        f"mean(no_d_con)={mean_no:.4e}, mean(d_con)={mean_d:.4e}"
    )


def test_nh_damp_v_d_con_differentiable_at_rest(small_nh_state):
    """``jax.grad`` through 3 steps with damp_v + d_con at rest stays finite."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(theta_p_data):
        s = state._replace(
            theta_prime=state.theta_prime.replace(data=theta_p_data),
        )
        for _ in range(3):
            s = model.step(s, 10.0)
        return jnp.mean(s.theta_prime.data ** 2)

    grad = jax.grad(loss_fn)(state.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad))


def test_nh_damp_v_d_con_no_op_when_damp_v_off(small_nh_state):
    """damp_v == 0 + damp_v_d_con > 0 = bit-for-bit baseline."""
    grid, height_coord, terrain_metric, state = small_nh_state

    cfg_unset = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    cfg_d_con_only = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.0, damp_v_d_con=1.0,
    )

    m_unset = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_unset,
    )
    m_d_con_only = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_d_con_only,
    )

    s_unset = m_unset.step(state, 10.0)
    s_d_con_only = m_d_con_only.step(state, 10.0)

    np.testing.assert_array_equal(
        np.asarray(s_unset.theta_prime.data),
        np.asarray(s_d_con_only.theta_prime.data),
    )
