"""Conservation on main's original T21, 20-step spectral NH defect fixture.

The old path leaked +1.883e-7 relative tracer inventory (main's historical
measurement). The paired mass path must conserve at the unchanged 1e-12 bar;
current T30 runtime is UNVERIFIED until this test runs on authorized compute.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
    SpectralCompressibleEulerModel, SpectralNHConfig, nh_rest_state_spectral)
from legoesm.grids.gaussian import (
    create_gaussian_grid, sh_analysis_3d, sh_synthesis_3d)
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate)

@pytest.fixture(autouse=True)
def fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(previous)


def test_spectral_tracer_mass_conserves_20_steps():
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
    q = 1e-2 * (1 + 0.3 * jnp.sin(lat) * jnp.cos(2 * lon) + 0.2 * z) * ones
    s = s._replace(
        theta_prime_hat=s.theta_prime_hat.replace(
            data=sh_analysis_3d(grid, th).astype(s.theta_prime_hat.data.dtype)),
        tracers_hat=s.tracers_hat.replace(
            data=sh_analysis_3d(grid, q)[..., None].astype(s.tracers_hat.data.dtype)))
    wgt = grid.weights[:, None, None] * hc.dz[None, None, :]

    def masses(st):
        rho = hc.rho_ref + sh_synthesis_3d(grid, st.rho_prime_hat.data)
        qq = sh_synthesis_3d(grid, st.tracers_hat.data[..., 0])
        return float(jnp.sum(wgt * rho)), float(jnp.sum(wgt * rho * qq))

    d0, t0 = masses(s)
    for _ in range(20):
        s = model.step(s, 5.0)
    d1, t1 = masses(s)
    assert abs(d1 / d0 - 1.0) < 1e-13
    leak = t1 / t0 - 1.0
    assert abs(leak) <= 1e-12, leak
