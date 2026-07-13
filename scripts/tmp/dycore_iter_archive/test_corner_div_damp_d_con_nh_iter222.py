"""FV3_3D iter 222: KE→heat ``d_con`` for the iter-168 corner-
divergence damping (NH mirror of PE iter-221).

iter-221 ported corner-div d_con on the PE 3D path.  iter-222
mirrors the implementation for the NH compressible-Euler path.

Heat tendency formula (per second, leading order in dt):

    dKE/dt_corner = u_d * du_d_dt_cdd + v_d * dv_d_dt_cdd
    dθ_p/dt += -corner_div_damp_d_con * (dKE/dt) / (c_pd * Π_ref)

where ``Π_ref`` is from ``HeightCoordinate.exner_ref`` (matching
the iter-207 refinement of damp_w_d_con).

Tests
-----

1. ``test_nh_corner_div_damp_d_con_off_baseline`` — default 0.0
   is bit-for-bit baseline.
2. ``test_nh_corner_div_damp_d_con_changes_theta_p`` — d_con > 0
   changes θ_p when winds non-zero and corner-div is active.
3. ``test_nh_corner_div_damp_d_con_no_op_when_corner_div_off`` —
   gated INSIDE ``corner_div_damp_d2_bg > 0``.
4. ``test_nh_corner_div_damp_d_con_differentiable_at_rest`` —
   AD-safe at the rest state.
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
    compute_terrain_metric, create_height_coordinate,
)


def _nh_state_with_winds(seed=222):
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

    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
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


def _step5(model, state, dt=10.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_nh_corner_div_damp_d_con_off_baseline():
    """corner_div_damp_d_con = 0.0 (default) is bit-for-bit
    baseline."""
    grid, hc, tm, state = _nh_state_with_winds()
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
    )
    cfg_explicit_off = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d_con=0.0,
    )
    cfg_default = CDGridCompressibleEulerConfig(**common)

    s_explicit = CDGridCompressibleEulerModel(
        grid, hc, tm, cfg_explicit_off,
    ).step(state, 10.0)
    s_default = CDGridCompressibleEulerModel(
        grid, hc, tm, cfg_default,
    ).step(state, 10.0)

    np.testing.assert_array_equal(
        s_explicit.theta_prime.data, s_default.theta_prime.data,
    )


def test_nh_corner_div_damp_d_con_changes_theta_p():
    """d_con > 0 changes θ_p when winds non-zero AND corner-div
    active."""
    grid, hc, tm, state = _nh_state_with_winds()
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
    )
    cfg_off = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d_con=0.0,
    )
    cfg_on = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d_con=1.0,
    )

    s_off = _step5(CDGridCompressibleEulerModel(
        grid, hc, tm, cfg_off,
    ), state)
    s_on = _step5(CDGridCompressibleEulerModel(
        grid, hc, tm, cfg_on,
    ), state)

    diff = float(jnp.max(jnp.abs(
        s_on.theta_prime.data - s_off.theta_prime.data
    )))
    assert diff > 1e-8, (
        f"NH corner_div_damp_d_con=1.0 must change θ_p relative "
        f"to d_con=0 baseline; got max|Δθ_p|={diff:.4e}."
    )


def test_nh_corner_div_damp_d_con_no_op_when_corner_div_off():
    """d_con is gated INSIDE the corner-div block; if corner-div
    is off, d_con > 0 must be a no-op."""
    grid, hc, tm, state = _nh_state_with_winds()
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0,    # corner-div OFF
    )
    cfg_a = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d_con=0.0,
    )
    cfg_b = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d_con=1.0,
    )
    s_a = CDGridCompressibleEulerModel(grid, hc, tm, cfg_a).step(
        state, 10.0,
    )
    s_b = CDGridCompressibleEulerModel(grid, hc, tm, cfg_b).step(
        state, 10.0,
    )
    np.testing.assert_array_equal(
        s_a.theta_prime.data, s_b.theta_prime.data,
    )


def test_nh_corner_div_damp_d_con_differentiable_at_rest():
    """AD-safe at rest state."""
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
        u=Field(data=jnp.zeros((6, n, n, nlev)),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss(eps):
        u_pert = state_rest.u.data + eps * jnp.ones_like(
            state_rest.u.data,
        )
        s = state_rest._replace(
            u=state_rest.u.replace(data=u_pert),
        )
        s_new = model.step(s, 10.0)
        return jnp.sum(s_new.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        f"NH corner_div_damp_d_con must be AD-safe at rest; "
        f"got grad={g}"
    )
