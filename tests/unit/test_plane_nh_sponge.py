"""Rayleigh sponge layer behaviour tests for the plane NH dycore.

Sponge is the top-of-model damping that absorbs vertically-propagating
gravity / acoustic waves before they reflect off the rigid upper
boundary. PR2d wires it through
``plane_compressible_euler_slow_tendencies`` using the shared
``sponge_profile`` taper. These tests confirm the sponge:

1. Stays at rest when ``sponge_coeff > 0`` is the only setting.
2. Damps a uniform ``u`` placed in the top layers towards zero;
   leaves a uniform ``u`` near the surface unchanged.
3. Sponge profile evaluates to zero below ``H - sponge_width`` and to
   ``sponge_coeff`` exactly at the model top.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
    sponge_profile,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------- #
# sponge_profile sanity                                                #
# --------------------------------------------------------------------- #


def test_sponge_profile_is_zero_below_sponge_window():
    H = 20.0e3
    sponge_width = 5.0e3
    sponge_coeff = 0.05
    z_full = jnp.linspace(H, 0.0, 41)[1:][::-1]
    # Below H - sponge_width = 15 km the profile must be exactly zero.
    profile = sponge_profile(z_full, H, sponge_width, sponge_coeff)
    below_mask = z_full < (H - sponge_width)
    assert jnp.max(jnp.abs(profile * below_mask)) == 0.0


def test_sponge_profile_reaches_coeff_at_model_top():
    H = 20.0e3
    sponge_width = 5.0e3
    sponge_coeff = 0.05
    profile = sponge_profile(
        jnp.array([H]), H, sponge_width, sponge_coeff,
    )
    assert float(profile[0]) == pytest.approx(sponge_coeff, rel=1.0e-12)


# --------------------------------------------------------------------- #
# Sponge inside the plane dycore                                        #
# --------------------------------------------------------------------- #


def _setup_with_sponge(sponge_coeff=0.5, sponge_width=10.0e3, nlev=10, H=20.0e3):
    grid = create_plane_grid(
        nx=4, ny=4, nlev=nlev, dx=1.0e3, dy=1.0e3, dtype=jnp.float64,
    )
    height_coord = create_height_coordinate(grid.nlev, H=H)
    terrain = make_flat_plane_terrain_metric(grid, height_coord)
    config = CompressibleEulerConfig(
        sponge_coeff=sponge_coeff,
        sponge_width=sponge_width,
        hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False,
        use_coriolis=False,
        fix_mass=False,
    )
    model = PlaneCompressibleEulerModel(grid, height_coord, terrain, config)
    return model, grid, height_coord


def test_rest_state_under_sponge_stays_at_rest():
    """``sponge * 0 = 0``; the rest state must remain at rest even
    when the sponge is on."""
    model, grid, height_coord = _setup_with_sponge(sponge_coeff=0.5)
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    state = model.step(state, dt=1.0)
    assert float(jnp.max(jnp.abs(state.u.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.v.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.w.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.theta_prime.data))) == 0.0


def test_sponge_damps_u_in_top_layers_but_leaves_surface_layers_alone():
    """Place a uniform ``u = 1 m/s`` everywhere. After one step:
    layers inside the sponge window (top of model) lose amplitude;
    layers below the window are untouched by the sponge term (but may
    still see horizontal advection of zero gradients = zero
    tendency)."""
    nlev = 10
    H = 20.0e3
    sponge_width = 10.0e3
    sponge_coeff = 0.5
    model, grid, height_coord = _setup_with_sponge(
        sponge_coeff=sponge_coeff, sponge_width=sponge_width, nlev=nlev, H=H,
    )
    state = make_rest_state(grid, height_coord, dtype=jnp.float64)
    u_uniform = jnp.ones_like(state.u.data)
    state = state._replace(u=state.u.replace(data=u_uniform))

    dt = 0.1  # well below 1/sponge_coeff so Euler step stable
    next_state = model.step(state, dt=dt)
    u_new = next_state.u.data

    # Top level (k=0) is at z_full[0] = H - dz/2 — deep inside the
    # sponge window, so the damping rate is close to ``sponge_coeff``.
    # Expected: u_top < u_initial.
    u_top_initial = float(jnp.mean(u_uniform[..., 0]))
    u_top_new = float(jnp.mean(u_new[..., 0]))
    assert u_top_new < u_top_initial, (
        f"Sponge did not damp top-level u: {u_top_initial} -> {u_top_new}"
    )

    # Bottom level (k=nlev-1) is at z_full[-1] = dz/2 — well below
    # the sponge window. Sponge taper is exactly zero there, so the
    # only u tendency would come from horizontal advection (zero for
    # uniform u). Expected: u_bottom_new == u_bottom_initial to
    # round-off.
    u_bottom_initial = float(jnp.mean(u_uniform[..., -1]))
    u_bottom_new = float(jnp.mean(u_new[..., -1]))
    assert u_bottom_new == pytest.approx(u_bottom_initial, abs=1.0e-12)
