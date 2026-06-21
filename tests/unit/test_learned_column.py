"""Smoke tests for atmosphere/physics/learned_column.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.learned_column import (
    build_column_physics,
    make_column_physics_fn,
)
from legoesm.atmosphere.physics.neural_physics import NeuralPhysics


def test_build_column_physics_returns_neural_physics():
    nlev = 8
    key = jax.random.PRNGKey(0)
    model = build_column_physics(nlev=nlev, hidden_dim=16, n_layers=2, key=key)
    assert isinstance(model, NeuralPhysics)
    assert model.nlev == nlev


def test_build_column_physics_residual_scale_passes_through():
    nlev = 4
    key = jax.random.PRNGKey(1)
    model = build_column_physics(
        nlev=nlev, hidden_dim=8, n_layers=2, residual_scale=0.05, key=key
    )
    assert isinstance(model, NeuralPhysics)


def test_make_column_physics_fn_returns_callable():
    """Check that make_column_physics_fn produces a callable physics fn.

    We don't exercise the full spectral round-trip here (that needs a
    GaussianGrid + SpectralHydrostaticState) — just that the factory
    returns a function with the documented signature.
    """
    from legoesm.grids.gaussian import create_gaussian_grid

    nlev = 4
    key = jax.random.PRNGKey(0)
    model = build_column_physics(nlev=nlev, hidden_dim=8, n_layers=2, key=key)
    grid = create_gaussian_grid(n_max=10)

    fn = make_column_physics_fn(model, grid)
    assert callable(fn)


def test_neural_flux_head_writes_predicted_fluxes_into_held():
    """The bridge must write the NN's predicted TOA/surface fluxes into
    held_* (so the flux loss supervises them), NOT pass the (zeroed) input
    held through. held_dT_rad and sw_down_toa stay passthrough.
    """
    from legoesm.atmosphere.physics.neural_physics import (
        NeuralPhysics, make_neural_step_unified,
    )
    from legoesm.core.grid_adapters import make_adapter
    from legoesm.grids.latlon import create_latlon_grid

    nlev = 4
    grid = create_latlon_grid(n_lat=8)   # latlon (float32-OK; the AIMIP grid)
    adapter = make_adapter(grid)
    # Large flux scale + nonzero random weights -> nonzero predicted fluxes.
    nn = NeuralPhysics(nlev=nlev, hidden_dim=8, n_layers=2,
                       key=jax.random.PRNGKey(3), flux_output_scale=100.0)
    step = make_neural_step_unified(nn, adapter)

    nlat, nlon = int(grid.n_lat), int(grid.n_lon)
    s3 = (nlat, nlon, nlev)
    s2 = (nlat, nlon)
    T = jnp.full(s3, 280.0)
    z3 = jnp.zeros(s3)
    z2 = jnp.zeros(s2)
    p_s = jnp.full(s2, 1.0e5)
    lat = jnp.zeros(s2)
    # Positional tail after (need_rad, T, p_s, q_v, q_c, q_r, conv_prog):
    tail = (z3, z3, z2, z2, lat, lat, 0.0, 0.0, 600.0,  # u,v,sst,sic,lat,lon,doy,sod,dt
            z2, jnp.array(1361.0), z3, z2,              # solar_w, s_0, o3, aero
            z3, z2, z2, z2, z2, z2)                      # held_* (all zero)
    phys_out, held_new = step(jnp.bool_(True), T, p_s, z3, z3, z3, None, *tail)
    held_dT_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa, sw_down_toa = held_new

    # Predicted TOA/sfc fluxes were written (nonzero) ...
    assert float(jnp.abs(sw_up_toa).max()) > 0.0
    assert float(jnp.abs(lw_up_toa).max()) > 0.0
    assert float(jnp.abs(sw_net_sfc).max()) > 0.0
    # ... and match the PhysicsOutput the network produced.
    assert jnp.allclose(sw_up_toa, phys_out.sw_up_toa)
    assert jnp.allclose(lw_up_toa, phys_out.lw_up_toa)
    # held_dT_rad and sw_down_toa stay passthrough (zero in == zero out).
    assert float(jnp.abs(held_dT_rad).max()) == 0.0
    assert float(jnp.abs(sw_down_toa).max()) == 0.0
