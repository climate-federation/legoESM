"""FV3_3D iter 522: AD-at-rest through ``monotone_halo_clip_context``.

The iter-505 helper is implemented via ``unittest.mock.patch``
on module-level halo aliases.  jax.grad must still propagate
through the patched code, since the inner implementation is a
``functools.partial`` of the original (JAX-AD-safe) function.

Tests
-----

1. ``test_grad_through_clip_context`` — define
   ``loss = mean(model.step(state).theta_prime² + ...)``;
   verify ``jax.grad(loss, state)`` returns finite nonzero
   gradients under the context.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_legoesm_nh_min_edge_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import monotone_halo_clip_context
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _build_state(n, seed):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
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
    return grid, hc, tm, state


def test_grad_through_clip_context():
    """AD must flow through the iter-505 helper."""
    grid, hc, tm, state = _build_state(8, 522)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    u0 = state.u.data
    v0 = state.v.data

    def loss_fn(u_init, v_init):
        s = state._replace(
            u=state.u.replace(data=u_init),
            v=state.v.replace(data=v_init),
        )
        with monotone_halo_clip_context(slack=0.5):
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            new_state = m.step(s, dt=10.0)
        return jnp.mean(new_state.theta_prime.data ** 2)

    grad_u_fn = jax.grad(loss_fn, argnums=0)
    grad_v_fn = jax.grad(loss_fn, argnums=1)
    grad_u = grad_u_fn(u0, v0)
    grad_v = grad_v_fn(u0, v0)
    assert jnp.all(jnp.isfinite(grad_u)), "grad u must be finite"
    assert jnp.all(jnp.isfinite(grad_v)), "grad v must be finite"
    nonzero_u = float(jnp.linalg.norm(grad_u))
    nonzero_v = float(jnp.linalg.norm(grad_v))
    assert nonzero_u > 1e-12, (
        f"grad u must be nonzero (got norm {nonzero_u})"
    )
    assert nonzero_v > 1e-12, (
        f"grad v must be nonzero (got norm {nonzero_v})"
    )


def test_grad_through_clip_context_multi_step():
    """AD through 3 steps under clip context."""
    grid, hc, tm, state = _build_state(8, 523)
    kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    cfg = make_legoesm_nh_min_edge_config(**kw)
    u0 = state.u.data

    def loss_fn(u_init):
        s = state._replace(u=state.u.replace(data=u_init))
        with monotone_halo_clip_context(slack=0.5):
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            for _ in range(3):
                s = m.step(s, dt=10.0)
        return jnp.mean(s.theta_prime.data ** 2)

    grad_u_fn = jax.grad(loss_fn)
    g = grad_u_fn(u0)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.linalg.norm(g)) > 1e-12
