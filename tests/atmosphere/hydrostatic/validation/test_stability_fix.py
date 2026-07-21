"""Test that the corrected PGF form is stable for isothermal rest."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
from legoesm.grids.gaussian import GaussianGrid, create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPrimitiveEquationModel,
    SpectralPEConfig,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)

# T42 resolution (~2.8 degree), 20 levels
grid = create_gaussian_grid(n_max=21)  # T21 for speed
sigma_coord = create_sigma_coordinate(20, sigma_top=0.01)

# Small hyperdiffusion for high wavenumbers
eig_max = 21.0 * 22.0 / grid.radius**2
HYPERDIFF = 1.0 / (1.0 * 3600.0 * eig_max**2)

config = SpectralPEConfig(
    hyperdiff_coeff=HYPERDIFF,
    hyperdiff_order=2,
    time_integrator="ssp_rk3",
    semi_implicit=False,
)

model = SpectralPrimitiveEquationModel(
    grid, sigma_coord, config, allow_unsupported_backend=True,
)

state = isothermal_rest_state_spectral(grid, sigma_coord, T_init=300.0, p_s_init=1e5)
DT = 600.0  # 10 minutes
n_steps = 240  # 40 hours

print(f"Grid: T{grid.n_max}, {sigma_coord.n_levels} levels")
print(f"dt={DT}s, n_steps={n_steps}, total={n_steps*DT/3600:.1f} hours")
print()

for i in range(n_steps):
    state = model.step(state, DT)

    if (i + 1) % 24 == 0:
        diag = spectral_pe_to_grid(state, grid, sigma_coord)
        T = diag['T']
        u = diag['u']
        div = diag['div']
        ps = diag['p_s']
        T_mean = float(jnp.mean(T))
        T_max = float(jnp.max(jnp.abs(T - 300.0)))
        u_max = float(jnp.max(jnp.abs(u)))
        div_max = float(jnp.max(jnp.abs(div)))
        ps_mean = float(jnp.mean(ps))
        ps_range = float(jnp.max(ps) - jnp.min(ps))
        day = (i + 1) * DT / 86400.0
        print(f"  day {day:5.2f}: T_mean={T_mean:.4f} T_dev={T_max:.2e} "
              f"u_max={u_max:.2e} div_max={div_max:.2e} "
              f"ps_mean={ps_mean:.1f} ps_range={ps_range:.2e}")

        if T_max > 100 or u_max > 100 or not jnp.isfinite(T_mean):
            print("  *** BLOWUP ***")
            break

print("\nDone. Model remained stable." if T_max < 100 else "\nModel blew up.")
