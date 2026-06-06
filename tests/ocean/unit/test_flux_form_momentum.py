"""Flux-form horizontal momentum advection — truth-tier gates (build spec F3-F6).

Directly exercises ``_bc_horizontal_momentum_advection_flux_form`` on a small flat
doubly-periodic domain:
- F3 zero-velocity -> zero advective tendency,
- F4 uniform flow -> zero advective tendency (analytic),
- F5 momentum conservation (volume-weighted domain integral ~ machine-eps),
- F6 differentiability.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _bc_horizontal_momentum_advection_flux_form,
)
from legoesm.ocean.state import LatLonCGridOceanConfig

_NLAT, _NLON, _NLEV = 8, 16, 3
_H = 100.0   # flat-bottom uniform layer thickness [m]


def _setup(u_vals, v_vals, scheme="centered"):
    """Build (du0, dv0, u, v, h_u, h_v, masks, grid, config) for a flat,
    all-ocean, doubly-periodic-in-lon domain. v is zeroed at the pole v-faces
    (wall). u_vals/v_vals are arrays of the right shape."""
    # float64 grid so the conservation gate measures the scheme's telescoping,
    # not float32 area round-off (production FV may run float32).
    grid = create_latlon_grid(_NLAT, _NLON, dtype=jnp.float64)
    nlat, nlon = grid.n_lat, grid.n_lon
    u = jnp.asarray(u_vals)
    v = jnp.asarray(v_vals)
    # Wall: no flow through the poles.
    v = v.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    h_u = jnp.full((nlat, nlon + 1, _NLEV), _H)
    h_v = jnp.full((nlat + 1, nlon, _NLEV), _H)
    u_mask_3d = jnp.ones((nlat, nlon + 1, _NLEV))
    v_mask_3d = jnp.ones((nlat + 1, nlon, _NLEV))
    mask = jnp.ones((nlat, nlon))
    du0 = jnp.zeros((nlat, nlon + 1, _NLEV))
    dv0 = jnp.zeros((nlat + 1, nlon, _NLEV))
    cfg = LatLonCGridOceanConfig(
        momentum_advection="flux_form", momentum_flux_scheme=scheme,
    )
    return du0, dv0, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg


def _call(du0, dv0, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg):
    return _bc_horizontal_momentum_advection_flux_form(
        du0, dv0, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg,
    )


def test_F3_zero_velocity_zero_tendency():
    """u=v=0 -> the flux-form advective tendency is exactly zero."""
    args = _setup(
        np.zeros((_NLAT, _NLON + 1, _NLEV)), np.zeros((_NLAT + 1, _NLON, _NLEV)),
    )
    du, dv, hu, hv = _call(*args)
    assert float(jnp.max(jnp.abs(hu))) == 0.0
    assert float(jnp.max(jnp.abs(hv))) == 0.0


def test_F4_uniform_flow_zero_tendency():
    """Uniform u=const, v=0 on a flat periodic domain -> advection of a constant
    is zero to round-off (div of a constant momentum flux = 0)."""
    u = np.full((_NLAT, _NLON + 1, _NLEV), 0.7)
    v = np.zeros((_NLAT + 1, _NLON, _NLEV))
    args = _setup(u, v, scheme="centered")
    du, dv, hu, hv = _call(*args)
    # Tendency scale for context: |u| * (typical 1/dt) — just assert ~0.
    assert float(jnp.max(jnp.abs(hu))) < 1e-12, float(jnp.max(jnp.abs(hu)))
    assert float(jnp.max(jnp.abs(hv))) < 1e-12, float(jnp.max(jnp.abs(hv)))


def _conservation_residual(scheme):
    rng = np.random.default_rng(3)
    u = 0.3 * rng.standard_normal((_NLAT, _NLON + 1, _NLEV))
    v = 0.3 * rng.standard_normal((_NLAT + 1, _NLON, _NLEV))
    # Interior flow: v -> 0 in the two lat rows nearest each pole, so the
    # meridional momentum flux through the boundary cell-centres vanishes. This
    # isolates the SCHEME's conservation (flux-divergence telescoping) from the
    # physical transfer of v-momentum to the N/S walls (which a walled lat-lon
    # domain legitimately does — the y-direction is not periodic). u-momentum is
    # periodic in lon and conserves regardless; this keeps v honest too.
    v[:2, :, :] = 0.0
    v[-2:, :, :] = 0.0
    args = _setup(u, v, scheme=scheme)
    du0, dv0, uu, vv, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg = args
    _, _, hadv_u, hadv_v = _call(*args)
    area = np.asarray(grid.area)
    # u-cell volume weight A_u*h_u (A_u = area avg to u-faces, periodic).
    a_uc = 0.5 * (area + np.roll(area, 1, axis=1))
    A_u = np.concatenate([a_uc, a_uc[:, 0:1]], axis=1)[..., None]
    vol_u = A_u * np.asarray(h_u)
    # Sum over UNIQUE u-points (exclude the periodic wrap column n_lon).
    mom_u = float(np.sum((vol_u * np.asarray(hadv_u))[:, :-1, :]))
    scale_u = float(np.sum(np.abs(vol_u * np.asarray(hadv_u))[:, :-1, :])) + 1e-300
    # v-cell volume weight.
    a_vc = 0.5 * (area[:-1] + area[1:])
    A_v = np.concatenate([a_vc[:1], a_vc, a_vc[-1:]], axis=0)[..., None]
    vol_v = A_v * np.asarray(h_v)
    mom_v = float(np.sum(vol_v * np.asarray(hadv_v)))
    scale_v = float(np.sum(np.abs(vol_v * np.asarray(hadv_v)))) + 1e-300
    return mom_u / scale_u, mom_v / scale_v


def test_F5_momentum_conservation_centered():
    """Volume-weighted domain-integrated flux-form advective tendency ~ 0 (the
    advection redistributes momentum; fluxes telescope on a periodic domain with
    v=0 walls). Centered scheme."""
    ru, rv = _conservation_residual("centered")
    assert abs(ru) < 1e-12, f"u-momentum not conserved: rel residual {ru:.2e}"
    assert abs(rv) < 1e-12, f"v-momentum not conserved: rel residual {rv:.2e}"


def test_F5_momentum_conservation_upwind():
    """Same conservation holds for the upwind reconstruction (telescoping is
    independent of the advected-value interpolation)."""
    ru, rv = _conservation_residual("upwind")
    assert abs(ru) < 1e-12, f"u-momentum not conserved (upwind): {ru:.2e}"
    assert abs(rv) < 1e-12, f"v-momentum not conserved (upwind): {rv:.2e}"


def test_F6_differentiable():
    """jax.grad of a scalar loss through the flux-form path is finite + nonzero."""
    rng = np.random.default_rng(5)
    u0 = jnp.asarray(0.3 * rng.standard_normal((_NLAT, _NLON + 1, _NLEV)))
    v0 = jnp.asarray(0.3 * rng.standard_normal((_NLAT + 1, _NLON, _NLEV)))
    args = _setup(np.asarray(u0), np.asarray(v0))
    du0, dv0, _, _, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg = args

    def loss(u):
        v = jnp.asarray(np.asarray(v0)).at[0].set(0.0).at[-1].set(0.0)
        _, _, hadv_u, _ = _bc_horizontal_momentum_advection_flux_form(
            du0, dv0, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg,
        )
        return jnp.sum(hadv_u ** 2)

    g = jax.grad(loss)(u0)
    assert jnp.all(jnp.isfinite(g)), "non-finite grad through flux-form advection"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "zero grad — path not differentiated"


def test_F7_gyre_stability_flux_form():
    """F7: a short forced-flow integration with momentum_advection='flux_form'
    stays finite and KE stays bounded (does not blow up). The upwind flux-form
    is dissipative, so KE plateaus/decays. vector_invariant runs too (baseline)."""
    import jax
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(12, 24)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=85.0,
    )
    # Active flow so horizontal momentum advection is exercised.
    up = 0.1 * jax.random.normal(jax.random.PRNGKey(11), state0.u.data.shape,
                                 dtype=jnp.float64)
    vp = 0.1 * jax.random.normal(jax.random.PRNGKey(12), state0.v.data.shape,
                                 dtype=jnp.float64)
    state0 = state0._replace(
        u=state0.u.replace(data=state0.u.data + up),
        v=state0.v.replace(data=state0.v.data + vp),
    )

    def _run(scheme):
        cfg = LatLonCGridOceanConfig(
            momentum_advection=scheme, momentum_flux_scheme="upwind",
            A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
            n_barotropic_substeps=8, enable_runtime_checks=False,
        )
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        state = state0
        ke0 = float(jnp.sum(state.u.data ** 2) + jnp.sum(state.v.data ** 2))
        for _ in range(80):
            state = model.step(state, dt=600.0)
        ke = float(jnp.sum(state.u.data ** 2) + jnp.sum(state.v.data ** 2))
        finite = bool(
            jnp.all(jnp.isfinite(state.u.data))
            and jnp.all(jnp.isfinite(state.v.data))
            and jnp.all(jnp.isfinite(state.T.data))
        )
        return finite, ke0, ke

    fin_ff, ke0, ke_ff = _run("flux_form")
    assert fin_ff, "flux_form integration produced non-finite state"
    assert ke_ff < 10.0 * ke0, (
        f"flux_form KE grew unboundedly: {ke_ff:.3e} vs initial {ke0:.3e}"
    )
    fin_vi, _, _ = _run("vector_invariant")
    assert fin_vi, "vector_invariant baseline produced non-finite state"
