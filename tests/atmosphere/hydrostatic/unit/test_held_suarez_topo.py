"""Sanity tests for ``legoesm.atmosphere.idealized.held_suarez_topo``."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.idealized.held_suarez_topo import (
    held_suarez_topo_forcing,
    held_suarez_topo_init,
    held_suarez_topo_init_latlon,
    held_suarez_topo_init_mpas,
    held_suarez_topo_init_spectral,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_forcing,
    held_suarez_forcing_mpas,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm.grids.voronoi import create_voronoi_mesh


@pytest.fixture(scope="module")
def sigma():
    return standard_hybrid_levels(8)


def test_topo_forcing_is_alias():
    """The topography variant reuses the canonical HS forcing exactly."""
    assert held_suarez_topo_forcing is held_suarez_forcing


def test_topo_init_cubed_sphere(sigma):
    grid = create_cubed_sphere(8)
    state = held_suarez_topo_init(grid, sigma, h_0=2000.0)

    assert state.T.data.shape == (6, 8, 8, 8)
    assert state.phis.data.shape == (6, 8, 8)
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.p_s.data)))

    # Mountain has non-zero peak, so phis must vary.
    phis_max = float(jnp.max(state.phis.data))
    phis_min = float(jnp.min(state.phis.data))
    assert phis_max > 0.0
    assert phis_min == 0.0  # cosine-bell is exactly zero outside R_m

    # Surface pressure must be hydrostatically lowered over the mountain
    p_max = float(jnp.max(state.p_s.data))
    p_min = float(jnp.min(state.p_s.data))
    assert p_max == pytest.approx(constants.p_ref, rel=1e-6)
    assert p_min < constants.p_ref


def test_topo_init_latlon(sigma):
    grid = create_latlon_grid(24, 48)
    state = held_suarez_topo_init_latlon(grid, sigma, h_0=2000.0)
    assert state.T.data.shape == (24, 48, 8)
    assert bool(jnp.all(jnp.isfinite(state.p_s.data)))
    assert float(jnp.max(state.phis.data)) > 0.0


def test_topo_init_mpas(sigma):
    mesh = create_voronoi_mesh(4)
    state = held_suarez_topo_init_mpas(mesh, sigma, h_0=2000.0)
    assert state.T.data.shape == (mesh.nCells, 8)
    assert bool(jnp.all(jnp.isfinite(state.T.data)))


def test_forcing_mpas_accepts_physics_fn_kwargs(sigma):
    """held_suarez_forcing_mpas must satisfy the MPAS hydrostatic dycore's
    operator-split physics_fn calling convention.

    Regression: ``primitive_eq_mpas`` calls its physics_fn as
    ``physics_fn(state, mesh, sigma_coord, *, phys_state=..., forcing=...)``
    (operator-split prognostic-physics carry).  ``held_suarez_forcing_mpas``
    lacked those kwargs, so the icosahedral held_suarez / held_suarez_topo /
    amip atmosphere-matrix cases errored at step 0 with
    ``held_suarez_forcing_mpas() got an unexpected keyword argument
    'phys_state'``.  HS is a stateless Newtonian relaxation, so the kwargs are
    accepted and ignored and the bare-tendencies return leaves the dycore's
    phys_state carry untouched.
    """
    mesh = create_voronoi_mesh(4)
    state = held_suarez_topo_init_mpas(mesh, sigma, h_0=2000.0)

    # Operator-split physics_fn convention: kwargs accepted.
    tend = held_suarez_forcing_mpas(
        state, mesh, sigma, phys_state=None, forcing=None)
    for field in (tend.du_dt.data, tend.dT_dt.data, tend.dp_s_dt.data):
        assert bool(jnp.all(jnp.isfinite(field)))
    # Newtonian relaxation off the equilibrium profile is non-trivial.
    assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0

    # The bare positional call (driver path) must still work and be identical:
    # phys_state / forcing are inert for stateless HS.
    tend_bare = held_suarez_forcing_mpas(state, mesh, sigma)
    assert bool(jnp.allclose(tend.dT_dt.data, tend_bare.dT_dt.data))
    assert bool(jnp.allclose(tend.du_dt.data, tend_bare.du_dt.data))
    assert bool(jnp.allclose(tend.dp_s_dt.data, tend_bare.dp_s_dt.data))


def test_topo_init_spectral(sigma):
    grid = create_gaussian_grid(21)
    state = held_suarez_topo_init_spectral(grid, sigma, h_0=2000.0)
    assert state.T_hat.data.shape == (grid.n_sh, 8)
    # phis should be non-zero in spectral coefficients (some power away
    # from zero mode)
    phis_amp = float(jnp.max(jnp.abs(state.phis_hat.data)))
    assert phis_amp > 0.0


def test_hs_parameter_overrides_reach_every_grid(sigma):
    """Held-Suarez tunables must reach the lat-lon, MPAS and spectral forcings,
    not only the cubed-sphere one.  Lat-lon and MPAS: the tendency with a
    non-default parameter set equals the shared relaxation helper's.  Spectral
    (output is in spectral space): every override, applied alone, changes it."""
    import jax
    from legoesm.atmosphere.forcing.idealized import held_suarez as HS

    kw = dict(k_a=2.0 * HS.K_A, k_s=0.5 * HS.K_S, sigma_b=0.6, delta_T_y=50.0,
              delta_theta_z=12.0, T_min=190.0)

    ll = create_latlon_grid(24, 48)
    s = held_suarez_topo_init_latlon(ll, sigma, h_0=2000.0)
    p_full = sigma.pressure_at_full(s.p_s.data)
    sig_eff = p_full / jnp.maximum(s.p_s.data[..., None], 1.0)
    want = HS.held_suarez_temperature_tendency(
        s.T.data, ll.lat[:, None, None], p_full, sig_eff, **kw)
    got = HS.held_suarez_forcing_latlon(s, ll, sigma, **kw).dT_dt.data
    assert bool(jnp.array_equal(got, want))
    assert not bool(jnp.array_equal(got, HS.held_suarez_forcing_latlon(s, ll, sigma).dT_dt.data))

    mesh = create_voronoi_mesh(4)
    s = held_suarez_topo_init_mpas(mesh, sigma, h_0=2000.0)
    p_full = sigma.pressure_at_full(s.p_s.data)
    sig_eff = p_full / jnp.maximum(s.p_s.data[:, None], 1.0)
    want = HS.held_suarez_temperature_tendency(
        s.T.data, mesh.latCell[:, None], p_full, sig_eff, **kw)
    got = HS.held_suarez_forcing_mpas(s, mesh, sigma, **kw).dT_dt.data
    assert bool(jnp.array_equal(got, want))

    gg = create_gaussian_grid(n_max=21)
    s = held_suarez_topo_init_spectral(gg, sigma, h_0=2000.0)
    base = HS.held_suarez_forcing_spectral(s, gg, sigma)
    for name, value in kw.items():
        tuned = HS.held_suarez_forcing_spectral(s, gg, sigma, **{name: value})
        same = jax.tree_util.tree_map(lambda a, b: bool(jnp.array_equal(a, b)), base, tuned)
        assert not all(jax.tree_util.tree_leaves(same)), name
