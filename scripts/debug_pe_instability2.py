#!/usr/bin/env python
"""Part 2: Find exact blowup point and growth mechanism.

The rest state is stable for 10 steps at T15/dt=480-540. But previous session
showed blowup at 200 steps. Find when it starts and what grows.
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np
import time

jax.config.update("jax_enable_x64", True)

from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis_3d
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPrimitiveEquationModel,
    SpectralPEConfig,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)
from legoesm import constants

print("=" * 70)
print("PE INSTABILITY: LONG-RUN REST STATE ANALYSIS")
print("=" * 70)

NLEV = 10
a = 6.371e6

# =====================================================================
# Test 1: T15/dt=480, run until blowup or 1000 steps
# =====================================================================
print("\n--- Test 1: T15/dt=480, explicit, up to 1000 steps ---")
T = 15
dt = 480.0

grid = create_gaussian_grid(T)
sigma = create_sigma_coordinate(NLEV)

eig_max = T * (T + 1) / (a * a)
HYPERDIFF = 1.0 / (0.5 * 3600.0 * eig_max ** 2)

config = SpectralPEConfig(hyperdiff_coeff=HYPERDIFF, hyperdiff_order=2)
model = SpectralPrimitiveEquationModel(grid, sigma, config)
state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

T_init = sh_synthesis_3d(grid, state.T_hat.data)
ns = np.array(grid.ls)

t0 = time.time()
# JIT warmup
state = model.step(state, dt)
jax.block_until_ready(state.T_hat.data)
print(f"  JIT warmup: {time.time()-t0:.1f}s")

# Reset
state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

for i in range(1000):
    state = model.step(state, dt)

    if (i + 1) % 10 == 0 or i < 10:
        T_now = sh_synthesis_3d(grid, state.T_hat.data)
        T_err = float(jnp.max(jnp.abs(T_now - T_init)))
        div_max = float(jnp.max(jnp.abs(state.div_hat.data)))
        lnps_err = float(jnp.max(jnp.abs(state.lnps_hat.data - isothermal_rest_state_spectral(grid, sigma).lnps_hat.data)))

        is_finite = bool(jnp.all(jnp.isfinite(state.T_hat.data)))
        if not is_finite:
            print(f"  step {i+1:5d}: T_err=NaN -> BLOWUP")
            break

        if (i + 1) % 50 == 0 or i < 10 or T_err > 1e-6:
            print(f"  step {i+1:5d}: T_err={T_err:.3e}, |div|={div_max:.3e}, lnps_err={lnps_err:.3e}")

        if T_err > 100:
            print(f"  step {i+1:5d}: T_err={T_err:.3e} -> BLOWUP")
            break

print(f"  Wall time: {time.time()-t0:.1f}s")

# =====================================================================
# Test 2: T15/dt=420, run 1000 steps (should stay stable)
# =====================================================================
print("\n--- Test 2: T15/dt=420, explicit, 1000 steps ---")
dt2 = 420.0
state2 = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

for i in range(1000):
    state2 = model.step(state2, dt2)

    if (i + 1) % 100 == 0:
        T_now = sh_synthesis_3d(grid, state2.T_hat.data)
        T_err = float(jnp.max(jnp.abs(T_now - T_init)))
        div_max = float(jnp.max(jnp.abs(state2.div_hat.data)))

        if not jnp.all(jnp.isfinite(state2.T_hat.data)):
            print(f"  step {i+1:5d}: BLOWUP")
            break

        print(f"  step {i+1:5d}: T_err={T_err:.3e}, |div|={div_max:.3e}")

# =====================================================================
# Test 3: T21/dt=120, run 500 steps (previously reported unstable)
# =====================================================================
print("\n--- Test 3: T21/dt=120, explicit, 500 steps ---")
T21 = 21
dt3 = 120.0

grid21 = create_gaussian_grid(T21)
sigma21 = create_sigma_coordinate(NLEV)

eig_max21 = T21 * (T21 + 1) / (a * a)
HYPERDIFF21 = 1.0 / (0.5 * 3600.0 * eig_max21 ** 2)

config21 = SpectralPEConfig(hyperdiff_coeff=HYPERDIFF21, hyperdiff_order=2)
model21 = SpectralPrimitiveEquationModel(grid21, sigma21, config21)
state3 = isothermal_rest_state_spectral(grid21, sigma21, T_init=300.0)

T_init21 = sh_synthesis_3d(grid21, state3.T_hat.data)

t0 = time.time()
state3 = model21.step(state3, dt3)
jax.block_until_ready(state3.T_hat.data)
print(f"  JIT warmup: {time.time()-t0:.1f}s")

state3 = isothermal_rest_state_spectral(grid21, sigma21, T_init=300.0)

for i in range(500):
    state3 = model21.step(state3, dt3)

    if (i + 1) % 50 == 0 or i < 5:
        T_now = sh_synthesis_3d(grid21, state3.T_hat.data)
        T_err = float(jnp.max(jnp.abs(T_now - T_init21)))
        div_max = float(jnp.max(jnp.abs(state3.div_hat.data)))

        if not jnp.all(jnp.isfinite(state3.T_hat.data)):
            print(f"  step {i+1:5d}: BLOWUP")
            break

        print(f"  step {i+1:5d}: T_err={T_err:.3e}, |div|={div_max:.3e}")

        if T_err > 100:
            print(f"  step {i+1:5d}: BLOWUP (T_err)")
            break

# =====================================================================
# Test 4: T21/dt=120 with Held-Suarez forcing (the real test)
# =====================================================================
print("\n--- Test 4: T21/dt=120, Held-Suarez, 500 steps ---")
from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing_spectral

config21_hs = SpectralPEConfig(hyperdiff_coeff=HYPERDIFF21, hyperdiff_order=2)
model21_hs = SpectralPrimitiveEquationModel(grid21, sigma21, config21_hs)
state4 = isothermal_rest_state_spectral(grid21, sigma21, T_init=300.0)

t0 = time.time()
state4 = model21_hs.step_with_physics(state4, dt3, held_suarez_forcing_spectral)
jax.block_until_ready(state4.T_hat.data)
print(f"  JIT warmup: {time.time()-t0:.1f}s")

state4 = isothermal_rest_state_spectral(grid21, sigma21, T_init=300.0)

for i in range(500):
    state4 = model21_hs.step_with_physics(state4, dt3, held_suarez_forcing_spectral)

    if (i + 1) % 50 == 0 or i < 5:
        fields = spectral_pe_to_grid(state4, grid21, sigma21)
        u_max = float(jnp.max(jnp.abs(fields['u'])))
        T_mean = float(jnp.mean(fields['T']))
        T_max = float(jnp.max(fields['T']))
        T_min = float(jnp.min(fields['T']))

        if not jnp.all(jnp.isfinite(state4.T_hat.data)):
            print(f"  step {i+1:5d}: BLOWUP (NaN)")
            break

        print(f"  step {i+1:5d}: |u|_max={u_max:.2f}, T=[{T_min:.1f}, {T_mean:.1f}, {T_max:.1f}]")

        if u_max > 500:
            print(f"  step {i+1:5d}: BLOWUP (u_max)")
            break

# =====================================================================
# Test 5: T15/dt=120 with Held-Suarez (should be stable reference)
# =====================================================================
print("\n--- Test 5: T15/dt=120, Held-Suarez, 500 steps ---")
config15_hs = SpectralPEConfig(hyperdiff_coeff=HYPERDIFF, hyperdiff_order=2)
model15_hs = SpectralPrimitiveEquationModel(grid, sigma, config15_hs)
state5 = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

state5 = model15_hs.step_with_physics(state5, 120.0, held_suarez_forcing_spectral)
jax.block_until_ready(state5.T_hat.data)

state5 = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

for i in range(500):
    state5 = model15_hs.step_with_physics(state5, 120.0, held_suarez_forcing_spectral)

    if (i + 1) % 50 == 0 or i < 5:
        fields = spectral_pe_to_grid(state5, grid, sigma)
        u_max = float(jnp.max(jnp.abs(fields['u'])))
        T_mean = float(jnp.mean(fields['T']))

        if not jnp.all(jnp.isfinite(state5.T_hat.data)):
            print(f"  step {i+1:5d}: BLOWUP")
            break

        print(f"  step {i+1:5d}: |u|_max={u_max:.2f}, T_mean={T_mean:.1f}")

# =====================================================================
# Test 6: SI at T21/dt=120 with Held-Suarez
# =====================================================================
print("\n--- Test 6: T21/dt=120, SI + Held-Suarez, 500 steps ---")
config21_si = SpectralPEConfig(
    hyperdiff_coeff=HYPERDIFF21, hyperdiff_order=2,
    semi_implicit=True, si_T_ref=300.0, si_alpha=0.5,
)
model21_si = SpectralPrimitiveEquationModel(grid21, sigma21, config21_si)
state6 = isothermal_rest_state_spectral(grid21, sigma21, T_init=300.0)

t0 = time.time()
state6 = model21_si.step_with_physics(state6, dt3, held_suarez_forcing_spectral)
jax.block_until_ready(state6.T_hat.data)
print(f"  JIT warmup: {time.time()-t0:.1f}s")

state6 = isothermal_rest_state_spectral(grid21, sigma21, T_init=300.0)

for i in range(500):
    state6 = model21_si.step_with_physics(state6, dt3, held_suarez_forcing_spectral)

    if (i + 1) % 50 == 0 or i < 5:
        fields = spectral_pe_to_grid(state6, grid21, sigma21)
        u_max = float(jnp.max(jnp.abs(fields['u'])))
        T_mean = float(jnp.mean(fields['T']))

        if not jnp.all(jnp.isfinite(state6.T_hat.data)):
            print(f"  step {i+1:5d}: BLOWUP (NaN)")
            break

        print(f"  step {i+1:5d}: |u|_max={u_max:.2f}, T_mean={T_mean:.1f}")

        if u_max > 500:
            print(f"  step {i+1:5d}: BLOWUP (u_max)")
            break

print("\nDone!")
