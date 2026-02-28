"""Test if earlier JAX ops slow down later JIT."""
import time, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.atmosphere.dynamics.spectral_sw import (
    SpectralSWConfig, spectral_sw_tendencies,
    williamson_test5_spectral, spectral_to_grid,
    compute_spectral_diagnostics,
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step

grid = create_gaussian_grid(42)
a = grid.radius
eig_max = 42 * 43 / (a**2)
nu = 1.0 / (4.0 * 3600.0 * eig_max**2)
config = SpectralSWConfig(hyperdiff_coeff=nu, hyperdiff_order=2)
state = williamson_test5_spectral(grid)

# These are called before step_jit in the real script
print('Running pre-JIT ops...', flush=True)
fields_init = spectral_to_grid(state, grid)
print(f'  h range: [{float(jnp.min(fields_init["h"])):.0f}, {float(jnp.max(fields_init["h"])):.0f}]')
diag = compute_spectral_diagnostics(state, grid)
print(f'  mass: {diag["mass"]:.6e}')

def tendency_fn(s):
    return spectral_sw_tendencies(s, grid, config)

step_jit = jax.jit(lambda s, dt: ssp_rk3_step(s, tendency_fn, dt))

print('JIT compiling step...', flush=True)
t0 = time.time()
s2 = step_jit(state, 60.0)
jax.block_until_ready(s2.vor_hat.data)
print(f'JIT done: {time.time()-t0:.1f}s', flush=True)
print('SUCCESS', flush=True)
