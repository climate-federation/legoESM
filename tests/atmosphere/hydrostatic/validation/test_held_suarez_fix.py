"""Test Held-Suarez simulation after PGF fix.

Runs a short spectral PE integration with Held-Suarez forcing
to verify stability after the pressure gradient force fix.
"""
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPrimitiveEquationModel,
    SpectralPEConfig,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_spectral


@pytest.mark.slow
def test_held_suarez_spectral_stability():
    """5-day Held-Suarez at T21/L20 should remain stable."""
    grid = create_gaussian_grid(n_max=21)
    sigma_coord = create_sigma_coordinate(20, sigma_top=0.01)

    eig_max = 21.0 * 22.0 / grid.radius**2
    HYPERDIFF = 1.0 / (0.5 * 3600.0 * eig_max**2)

    config = SpectralPEConfig(
        hyperdiff_coeff=HYPERDIFF,
        hyperdiff_order=2,
        time_integrator="ssp_rk3",
    )

    model = SpectralPrimitiveEquationModel(
        grid, sigma_coord, config, allow_unsupported_backend=True,
    )

    state = isothermal_rest_state_spectral(
        grid, sigma_coord, T_init=300.0, p_s_init=1e5,
    )

    DT = 600.0
    n_steps = 720  # 5 days

    for i in range(n_steps):
        state = model.step_with_physics(state, DT, held_suarez_forcing_spectral)

    diag = spectral_pe_to_grid(state, grid, sigma_coord)
    T = diag["T"]
    T_mean = float(jnp.mean(T))
    T_min = float(jnp.min(T))
    T_max = float(jnp.max(T))

    assert jnp.isfinite(jnp.array(T_mean)), "Temperature became non-finite"
    assert T_min > 100.0, f"Temperature too cold: {T_min:.1f} K"
    assert T_max < 500.0, f"Temperature too hot: {T_max:.1f} K"
