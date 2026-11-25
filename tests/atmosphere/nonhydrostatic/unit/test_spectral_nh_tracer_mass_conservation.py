"""Spectral NH core: dry AND tracer mass close to round-off (#2641 resolved).

#2641 recorded a tracer-mass leak (+1.883e-7 relative after 20 steps) when
``tracers_hat`` carried the mixing ratio q with an advective-form tendency.
The core now carries the Jacobian-weighted tracer DENSITY J*rho*q in flux
form (same spectral flux divergence as rho', vertical flux with the
substep-integrated mass flux), so the tracer mass is sum(w dz J*rho*q) and is
conserved like the dry mass: measured -8.0e-15 relative after 20 steps on the
case below (T21, 10 levels, 0.2 K theta perturbation, dt 5 s, fp64, flat
terrain so J = 1).
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
    SpectralCompressibleEulerModel, SpectralNHConfig, nh_rest_state_spectral)
from legoesm.grids.gaussian import (
    create_gaussian_grid, sh_analysis_3d, sh_synthesis_3d)
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate)



def test_spectral_nh_conserves_dry_and_tracer_mass():
    grid = create_gaussian_grid(n_max=21)
    hc = create_height_coordinate(10, 30000.0)
    tm = compute_terrain_metric(jnp.zeros((grid.n_lat, grid.n_lon)), hc)
    eig_max = grid.n_max * (grid.n_max + 1) / grid.radius ** 2
    cfg = SpectralNHConfig(hyperdiff_coeff=1.0 / (4.0 * 3600.0 * eig_max ** 2),
                           hyperdiff_order=2, sponge_coeff=0.0,
                           n_acoustic_substeps=4)
    model = SpectralCompressibleEulerModel(grid, hc, tm, cfg,
                                           allow_unsupported_backend=True)
    s = nh_rest_state_spectral(grid, hc, n_tracers=1)
    lat, lon = grid.lat[:, None, None], grid.lon[None, :, None]
    z = ((jnp.arange(10) + 0.5) / 10)[None, None, :]
    ones = jnp.ones((grid.n_lat, grid.n_lon, 10))
    th = 0.2 * jnp.cos(lat) * jnp.cos(lon) * jnp.sin(jnp.pi * z) * ones
    rho0 = hc.rho_ref * ones
    q = rho0 * 1e-2 * (1 + 0.3 * jnp.sin(lat) * jnp.cos(2 * lon) + 0.2 * z) * ones
    s = s._replace(
        theta_prime_hat=s.theta_prime_hat.replace(
            data=sh_analysis_3d(grid, th).astype(s.theta_prime_hat.data.dtype)),
        tracers_hat=s.tracers_hat.replace(
            data=sh_analysis_3d(grid, q)[..., None].astype(s.tracers_hat.data.dtype)))
    wgt = grid.weights[:, None, None] * hc.dz[None, None, :]

    def masses(st):
        rho = hc.rho_ref + sh_synthesis_3d(grid, st.rho_prime_hat.data)
        rq = sh_synthesis_3d(grid, st.tracers_hat.data[..., 0])  # J*rho*q, J = 1
        return float(jnp.sum(wgt * rho)), float(jnp.sum(wgt * rq))

    d0, t0 = masses(s)
    for _ in range(20):
        s = model.step(s, 5.0)
    d1, t1 = masses(s)
    assert abs(d1 / d0 - 1.0) < 1e-13
    leak = t1 / t0 - 1.0
    assert abs(leak) < 1e-13, leak
