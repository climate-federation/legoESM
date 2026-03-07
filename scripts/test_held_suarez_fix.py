"""Test Held-Suarez simulation after PGF fix."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPrimitiveEquationModel,
    SpectralPEConfig,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)

# T21 for speed
grid = create_gaussian_grid(n_max=21)
sigma_coord = create_sigma_coordinate(20, sigma_top=0.01)

eig_max = 21.0 * 22.0 / grid.radius**2
HYPERDIFF = 1.0 / (0.5 * 3600.0 * eig_max**2)  # 30-min e-folding for max n

config = SpectralPEConfig(
    hyperdiff_coeff=HYPERDIFF,
    hyperdiff_order=2,
    time_integrator="ssp_rk3",
)

model = SpectralPrimitiveEquationModel(
    grid, sigma_coord, config, allow_unsupported_backend=True,
)

state = isothermal_rest_state_spectral(grid, sigma_coord, T_init=300.0, p_s_init=1e5)

# Simple Held-Suarez forcing
from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing_spectral
physics_fn = held_suarez_forcing_spectral

DT = 600.0
n_steps = 4320  # 30 days
print(f"Grid: T{grid.n_max}, {sigma_coord.n_levels} levels")
print(f"dt={DT}s, n_steps={n_steps}, total={n_steps*DT/86400:.0f} days")
print()

for i in range(n_steps):
    state = model.step_with_physics(state, DT, physics_fn)

    if (i + 1) % 144 == 0:
        diag = spectral_pe_to_grid(state, grid, sigma_coord)
        T = diag['T']
        u = diag['u']
        ps = diag['p_s']
        T_mean = float(jnp.mean(T))
        T_min = float(jnp.min(T))
        T_max = float(jnp.max(T))
        u_max = float(jnp.max(jnp.abs(u)))
        ps_range = float(jnp.max(ps) - jnp.min(ps))
        day = (i + 1) * DT / 86400.0
        print(f"  day {day:6.1f}: T=[{T_min:.1f}, {T_mean:.1f}, {T_max:.1f}]K "
              f"u_max={u_max:.1f}m/s ps_range={ps_range:.0f}Pa")

        if T_max > 500 or T_min < 100 or not jnp.isfinite(T_mean):
            print("  *** BLOWUP ***")
            break

print(f"\nDone. {'Stable' if T_max < 500 else 'BLEW UP'}.")
