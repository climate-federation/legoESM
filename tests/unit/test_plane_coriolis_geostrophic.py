"""D2: Coriolis acts on the departure from the geostrophic reference wind.

gSAM ``coriolis.f90`` (f-plane branch)::

    dudt += fcory(j)·(v_av − vg0(k))
    dvdt −= fcor·(u_av − ug0(k))

so the large-scale balanced mean wind (u=ug0, v=vg0) feels ZERO Coriolis
acceleration — only the ageostrophic departure is rotated. legoESM stores
ug0/vg0 as optional ``HeightCoordinate.u_geo0/v_geo0`` profiles (None ⇒ 0,
the RCE / no-mean-wind case, which reduces to applying f to the full wind).

The slow-tendency is isolated to Coriolis here: with horizontally-uniform
winds varying only in z, w=0 and θ'=ρ'=0, advection / vertical / pressure-
gradient tendencies all vanish, so du_dt = du_cor and dv_dt = dv_cor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


jax.config.update("jax_enable_x64", True)

_F0 = 1.0e-4


def _setup(nlev=6):
    grid = create_plane_grid(
        nx=4, ny=4, nlev=nlev, dx=1.0e3, dy=1.0e3,
        coriolis_mode="f_plane", f0=_F0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(grid.nlev, H=20.0e3)
    terrain = make_flat_plane_terrain_metric(grid, hc)
    config = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False,
        use_coriolis=True,
    )
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    return grid, hc, terrain, config, state


def _with_uniform_wind(state, u_prof, v_prof):
    """Horizontally-uniform winds u(z), v(z); everything else at rest."""
    ny, nx, nlev = state.u.data.shape
    u = jnp.broadcast_to(u_prof[None, None, :], (ny, nx, nlev))
    v = jnp.broadcast_to(v_prof[None, None, :], (ny, nx, nlev))
    return state._replace(
        u=state.u.replace(data=u), v=state.v.replace(data=v),
    )


def _coriolis_tend(hc, state, grid, terrain, config):
    tend = plane_compressible_euler_slow_tendencies(
        state, grid, hc, terrain, config,
    )
    return tend.du_dt.data, tend.dv_dt.data


def test_geostrophic_wind_feels_no_coriolis():
    """u=ug0, v=vg0 (geostrophic balance) ⇒ Coriolis tendency is ~0: the
    mean wind is NOT spuriously spun up."""
    grid, hc, terrain, config, state = _setup()
    nlev = grid.nlev
    ug0 = jnp.linspace(5.0, 15.0, nlev)
    vg0 = jnp.linspace(-3.0, 3.0, nlev)
    hc = hc._replace(u_geo0=ug0, v_geo0=vg0)
    state = _with_uniform_wind(state, ug0, vg0)
    du, dv = _coriolis_tend(hc, state, grid, terrain, config)
    assert float(jnp.max(jnp.abs(du))) < 1.0e-14
    assert float(jnp.max(jnp.abs(dv))) < 1.0e-14


def test_no_reference_recovers_full_wind_coriolis():
    """With u_geo0/v_geo0 = None (RCE default) Coriolis acts on the FULL
    wind: du_cor = f·v (≠0 for the same winds) — i.e. the reference matters
    and the None path is the legacy full-wind form."""
    grid, hc, terrain, config, state = _setup()
    nlev = grid.nlev
    ug0 = jnp.linspace(5.0, 15.0, nlev)
    vg0 = jnp.linspace(-3.0, 3.0, nlev)
    # Same winds, but NO geostrophic reference on the height coordinate.
    assert hc.u_geo0 is None and hc.v_geo0 is None
    state = _with_uniform_wind(state, ug0, vg0)
    du, dv = _coriolis_tend(hc, state, grid, terrain, config)
    # du_cor = f·v = f0·vg0 (vg0 ranges to ±3 ⇒ |du| up to 3e-4).
    assert float(jnp.max(jnp.abs(du))) == pytest.approx(_F0 * 3.0, rel=1e-6)
    # And it is NOT zero — the geostrophic-balance cancellation requires the
    # reference subtraction.
    assert float(jnp.max(jnp.abs(du))) > 1.0e-6


def test_ageostrophic_departure_is_rotated():
    """A uniform departure Δu = u − ug0 produces dv_cor = −f·Δu (and the
    matching v-departure produces du_cor = +f·Δv): only the departure
    rotates, at the full Coriolis rate."""
    grid, hc, terrain, config, state = _setup()
    nlev = grid.nlev
    ug0 = jnp.linspace(5.0, 15.0, nlev)
    vg0 = jnp.linspace(-3.0, 3.0, nlev)
    hc = hc._replace(u_geo0=ug0, v_geo0=vg0)
    du_pert = 2.0
    state = _with_uniform_wind(state, ug0 + du_pert, vg0)
    du, dv = _coriolis_tend(hc, state, grid, terrain, config)
    # v = vg0 ⇒ du_cor = f·(v − vg0) = 0.
    assert float(jnp.max(jnp.abs(du))) < 1.0e-14
    # u − ug0 = +2 ⇒ dv_cor = −f·2 everywhere.
    assert jnp.allclose(dv, -_F0 * du_pert, atol=1.0e-14)


def test_none_reference_unchanged_from_zero_reference():
    """The None default is numerically identical to an explicit zero
    reference profile (backward-compatibility guarantee for RCE runs)."""
    grid, hc, terrain, config, state = _setup()
    nlev = grid.nlev
    u_prof = jnp.linspace(1.0, 4.0, nlev)
    v_prof = jnp.linspace(-2.0, 2.0, nlev)
    state = _with_uniform_wind(state, u_prof, v_prof)
    du_none, dv_none = _coriolis_tend(hc, state, grid, terrain, config)
    hc_zero = hc._replace(
        u_geo0=jnp.zeros(nlev), v_geo0=jnp.zeros(nlev),
    )
    du_zero, dv_zero = _coriolis_tend(hc_zero, state, grid, terrain, config)
    assert jnp.allclose(du_none, du_zero, atol=1.0e-16)
    assert jnp.allclose(dv_none, dv_zero, atol=1.0e-16)
