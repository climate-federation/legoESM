"""Debug JIT hang: exactly replicate the run script setup."""
import time, os, jax, jax.numpy as jnp, numpy as np
jax.config.update('jax_enable_x64', True)
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.atmosphere.dynamics.spectral_sw import (
    SpectralSWConfig, spectral_sw_tendencies,
    williamson_test5_spectral, spectral_to_grid,
    compute_spectral_diagnostics,
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm import constants

N_MAX = 42
DT = 60.0

os.makedirs("results/100day_spectral", exist_ok=True)

grid = create_gaussian_grid(N_MAX)
a = grid.radius
eig_max = N_MAX * (N_MAX + 1) / (a * a)
hyperdiff_coeff = 1.0 / (4.0 * 3600.0 * eig_max**2)

config = SpectralSWConfig(
    mean_depth=5960.0,
    hyperdiff_coeff=hyperdiff_coeff,
    hyperdiff_order=2,
)

state = williamson_test5_spectral(grid)
state_init = state

print('Step 1: spectral_to_grid...', flush=True)
t0 = time.time()
fields_init = spectral_to_grid(state, grid)
print(f'  done in {time.time()-t0:.2f}s', flush=True)

print('Step 2: compute_spectral_diagnostics...', flush=True)
t0 = time.time()
diag = compute_spectral_diagnostics(state, grid)
print(f'  done in {time.time()-t0:.2f}s', flush=True)

print('Step 3: jnp ops on fields_init...', flush=True)
t0 = time.time()
h_min = float(jnp.min(fields_init['h']))
h_max = float(jnp.max(fields_init['h']))
umax = float(jnp.max(jnp.sqrt(fields_init['u']**2 + fields_init['v']**2)))
hs_max = float(jnp.max(fields_init['h_s']))
print(f'  done in {time.time()-t0:.2f}s', flush=True)

# Storage (same as run script)
diagnostics = [diag]
snapshots = {0: state}
n_steps = int(100 * 86400 / DT)
STEPS_PER_DAY = int(86400 / DT)

print('Step 4: define tendency_fn and step_jit...', flush=True)
def tendency_fn(s):
    return spectral_sw_tendencies(s, grid, config)
step_jit = jax.jit(lambda s, dt: ssp_rk3_step(s, tendency_fn, dt))

print('Step 5: JIT compile...', flush=True)
t0 = time.time()
s2 = step_jit(state, DT)
jax.block_until_ready(s2.vor_hat.data)
print(f'  JIT done in {time.time()-t0:.1f}s', flush=True)
print('SUCCESS', flush=True)
