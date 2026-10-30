"""Spectral-lane tracer MASS conservation (review 2026-10-10 F21).

The spectral PE transports tracers in advective form (pseudo-spectral
-v.grad q, upwind sigma-dot dq/dsigma) and filters q itself, neither of which
conserves a tracer's mass ``sum A sum_k dp_k q_k``; the lane has no water
fixer.  Fix: a per-stage transport mass fixer in ``spectral_pe_tendencies``
plus a global mass restore after the tracer filter.

Measured on this exact setup (T21 L10, 10-20 steps of 600 s, rotational +
divergent flow over a 1.5 km mountain, hyperdiffusion on, mass anchor on,
x64), relative tracer-mass drift after 20 steps:
  before the fix: sigma 1.0e-5, hybrid 2.3e-3;
  after:          sigma 2.1e-10, hybrid 7.9e-10 (ssp_rk54; the residual is
                  the integrator's error on the bilinear invariant).
The 1e-8 bound below therefore fails on the old code by >= 3 orders.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig, SpectralPrimitiveEquationModel,
    isothermal_rest_state_spectral,
)
from legoesm.core.field import Field
from legoesm.grids.gaussian import (
    create_gaussian_grid, sh_analysis_3d, sh_synthesis,
)
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels

jax.config.update("jax_enable_x64", True)

NLEV, N_STEPS, DT = 10, 10, 600.0


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(n_max=21)


def _setup(grid, coord):
    lat = jnp.broadcast_to(jnp.asarray(grid.lat)[:, None],
                           (grid.n_lat, grid.n_lon))
    lon = jnp.broadcast_to(jnp.asarray(grid.lon)[None, :],
                           (grid.n_lat, grid.n_lon))
    phis = constants.g * 1500.0 * jnp.exp(
        -((lat - 0.5) ** 2 + (lon - 2.0) ** 2) / 0.3)
    st = isothermal_rest_state_spectral(
        grid, coord, T_init=280.0, phis=phis, perturbation_amplitude=0.0)
    lev = jnp.linspace(-1.0, 1.0, NLEV)
    vor = 3e-5 * (jnp.sin(2 * lat) * jnp.cos(lon))[..., None] * jnp.ones(NLEV)
    div = 4e-6 * (jnp.cos(lat) * jnp.sin(2 * lon))[..., None] * lev
    st = st._replace(
        vor_hat=st.vor_hat.replace(data=sh_analysis_3d(grid, vor).astype(
            st.vor_hat.data.dtype)),
        div_hat=st.div_hat.replace(data=sh_analysis_3d(grid, div).astype(
            st.div_hat.data.dtype)))
    q = (0.01 * (1.0 + 0.5 * jnp.sin(lat) * jnp.cos(lon))[..., None]
         * jnp.linspace(0.1, 1.0, NLEV))
    st = st._replace(tracers={"q_v": Field(
        data=q, name="q_v", dims=("lat", "lon", "level"), units="kg/kg")})
    eig = grid.n_max * (grid.n_max + 1) / grid.radius ** 2
    cfg = SpectralPEConfig(
        hyperdiff_coeff=1.0 / (4 * 3600 * eig ** 2),
        time_integrator="ssp_rk54", fix_mass=True,
        anchor_mass_to_initial=True)
    return SpectralPrimitiveEquationModel(grid, coord, cfg), st


def _tracer_mass(grid, coord, s):
    ps = jnp.exp(sh_synthesis(grid, s.lnps_hat.data))
    dp = coord.layer_thickness_dp(ps)
    return float(jnp.sum(grid.grid_area[..., None] * dp
                         * s.tracers["q_v"].data))


@pytest.mark.parametrize("coord_kind", ["sigma", "hybrid"])
def test_spectral_transport_conserves_tracer_mass(grid, coord_kind):
    coord = (create_sigma_coordinate(NLEV) if coord_kind == "sigma"
             else make_hybrid_levels(NLEV, p_top_Pa=200.0, stretching=2.0))
    model, s = _setup(grid, coord)
    m0 = _tracer_mass(grid, coord, s)
    q0 = s.tracers["q_v"].data
    for _ in range(N_STEPS):
        s = model.step(s, DT)
    q1 = s.tracers["q_v"].data
    assert bool(jnp.all(jnp.isfinite(q1)))
    # the flow really transports (the bound is not met by a frozen field)
    assert float(jnp.max(jnp.abs(q1 - q0))) > 1e-5
    rel = abs(_tracer_mass(grid, coord, s) - m0) / m0
    assert rel < 1e-8, f"{coord_kind}: tracer mass drift {rel:.3e}"
