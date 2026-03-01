#!/usr/bin/env python
"""Diagnose the spectral PE instability at large dt.

Key questions:
1. What are the Gamma eigenvalues and the implied gravity wave CFL?
2. Is the rest-state tendency exactly zero?
3. Where does the error appear first (which field, which mode)?
4. Does the SI scheme actually remove the gravity wave constraint?
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis, sh_synthesis_3d
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPrimitiveEquationModel,
    SpectralPEConfig,
    isothermal_rest_state_spectral,
    spectral_pe_tendencies,
    spectral_pe_to_grid,
)
from legoesm.timestepping.semi_implicit import compute_Gamma_matrix, precompute_si_matrices
from legoesm import constants

print("=" * 70)
print("SPECTRAL PE INSTABILITY DIAGNOSTIC")
print("=" * 70)

# =====================================================================
# 1. Gamma matrix eigenvalues and gravity wave CFL
# =====================================================================
print("\n--- 1. Gamma Matrix Eigenvalues & Gravity Wave CFL ---")

T = 15
NLEV = 10
a = 6.371e6

grid = create_gaussian_grid(T)
sigma = create_sigma_coordinate(NLEV)

Gamma = compute_Gamma_matrix(sigma, 300.0)
Gamma_np = np.array(Gamma)

# Eigenvalues of Gamma
eigs_Gamma = np.linalg.eigvals(Gamma_np)
eigs_real = np.sort(np.real(eigs_Gamma))[::-1]
eigs_imag = np.imag(eigs_Gamma)

print(f"  Gamma shape: {Gamma.shape}")
print(f"  Gamma eigenvalues (real parts, sorted):")
for i, e in enumerate(eigs_real):
    print(f"    mode {i}: {e:.6e}")
print(f"  Max imaginary part: {np.max(np.abs(eigs_imag)):.2e}")
print(f"  Max eigenvalue: {eigs_real[0]:.6e}")
print(f"  Min eigenvalue: {eigs_real[-1]:.6e}")

# Gravity wave CFL
lambda_max = T * (T + 1) / a**2
print(f"\n  Spectral Laplacian eigenvalue at n={T}: {lambda_max:.6e}")

omega_max = np.sqrt(lambda_max * eigs_real[0])
print(f"  Max gravity wave angular frequency: {omega_max:.6e} rad/s")

c_gw_equiv = np.sqrt(eigs_real[0])
print(f"  Equivalent external GW speed: {c_gw_equiv:.1f} m/s")

# CFL limits for SSP-RK3
cfl_rk3 = 1.73  # stability boundary on imaginary axis
dt_max_gw = cfl_rk3 / omega_max
print(f"\n  SSP-RK3 CFL limit for GW: dt_max = {dt_max_gw:.1f} s")
print(f"  For comparison: explicit test uses dt=120s")
print(f"  Observed stability boundary: ~480-540s")

# Check if observed boundary matches GW CFL
for dt_test in [120, 300, 420, 480, 540, 600]:
    cfl_val = omega_max * dt_test
    print(f"  dt={dt_test:4d}s: omega*dt = {cfl_val:.3f} (limit={cfl_rk3:.2f})")

# =====================================================================
# 2. Per-vertical-mode gravity wave speeds
# =====================================================================
print("\n--- 2. Per-Vertical-Mode Gravity Wave Speeds ---")
for i, e in enumerate(eigs_real):
    if e > 0:
        c_mode = np.sqrt(e)
        omega_mode = np.sqrt(lambda_max * e)
        dt_max_mode = cfl_rk3 / omega_mode if omega_mode > 0 else float('inf')
        print(f"  Mode {i}: c={c_mode:.1f} m/s, omega_max={omega_mode:.3e}, dt_max={dt_max_mode:.0f}s")

# =====================================================================
# 3. Hyperdiffusion stability
# =====================================================================
print("\n--- 3. Hyperdiffusion Stability ---")
eig_max = T * (T + 1) / (a * a)
HYPERDIFF = 1.0 / (0.5 * 3600.0 * eig_max ** 2)
print(f"  Hyperdiffusion coefficient: {HYPERDIFF:.3e}")
print(f"  Damping rate at n={T}: {HYPERDIFF * eig_max**2:.6e} /s")
print(f"  e-folding time at n={T}: {1/(HYPERDIFF * eig_max**2):.0f}s")

# RK3 real-axis stability limit
diffusion_eigenvalue = -HYPERDIFF * eig_max**2
for dt_test in [120, 300, 420, 480, 540, 600]:
    z_diff = diffusion_eigenvalue * dt_test
    print(f"  dt={dt_test:4d}s: diffusion z = {z_diff:.4f} (limit=-2.51)")

# =====================================================================
# 4. Combined stability (diffusion + gravity waves)
# =====================================================================
print("\n--- 4. Combined RK3 Stability Region Check ---")
# For each dt, compute z = (diffusion + i*gravity_wave)*dt
# and check if it's inside the RK3 stability region
for dt_test in [120, 300, 420, 480, 540, 600]:
    z_real = diffusion_eigenvalue * dt_test
    z_imag = omega_max * dt_test
    z = complex(z_real, z_imag)
    # SSP-RK3 stability function: R(z) = 1 + z + z²/2 + z³/6
    R = 1 + z + z**2/2 + z**3/6
    amp = abs(R)
    print(f"  dt={dt_test:4d}s: z = {z_real:.4f} + {z_imag:.4f}i, |R(z)|={amp:.6f} {'STABLE' if amp<=1 else 'UNSTABLE'}")

# =====================================================================
# 5. Rest state tendency (should be exactly zero)
# =====================================================================
print("\n--- 5. Rest State Tendency Magnitudes ---")

config = SpectralPEConfig(hyperdiff_coeff=HYPERDIFF, hyperdiff_order=2)
state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

# Compute one tendency
tend = spectral_pe_tendencies(state, grid, sigma, config)

print(f"  |dvor/dt|_max   = {float(jnp.max(jnp.abs(tend.vor_hat.data))):.3e}")
print(f"  |ddiv/dt|_max   = {float(jnp.max(jnp.abs(tend.div_hat.data))):.3e}")
print(f"  |dT/dt|_max     = {float(jnp.max(jnp.abs(tend.T_hat.data))):.3e}")
print(f"  |dlnps/dt|_max  = {float(jnp.max(jnp.abs(tend.lnps_hat.data))):.3e}")
print(f"  |dphis/dt|_max  = {float(jnp.max(jnp.abs(tend.phis_hat.data))):.3e}")

# =====================================================================
# 6. Step-by-step evolution at boundary
# =====================================================================
print("\n--- 6. Step-by-Step Rest State Evolution ---")

def run_steps(T_res, dt, n_steps, use_si=False, verbose=True):
    """Run n_steps and track field magnitudes."""
    grid_l = create_gaussian_grid(T_res)
    sigma_l = create_sigma_coordinate(NLEV)

    a_l = grid_l.radius
    eig_max_l = T_res * (T_res + 1) / (a_l * a_l)
    hyperdiff_l = 1.0 / (0.5 * 3600.0 * eig_max_l ** 2)

    config_l = SpectralPEConfig(
        hyperdiff_coeff=hyperdiff_l,
        hyperdiff_order=2,
        semi_implicit=use_si,
        si_T_ref=300.0,
        si_alpha=0.5,
    )
    model = SpectralPrimitiveEquationModel(grid_l, sigma_l, config_l)
    state_l = isothermal_rest_state_spectral(grid_l, sigma_l, T_init=300.0)

    # Store initial T for comparison
    T_init = sh_synthesis_3d(grid_l, state_l.T_hat.data)

    for i in range(n_steps):
        state_l = model.step(state_l, dt)

        if verbose and (i < 5 or (i + 1) % 50 == 0 or i == n_steps - 1):
            T_now = sh_synthesis_3d(grid_l, state_l.T_hat.data)
            T_err = float(jnp.max(jnp.abs(T_now - T_init)))

            div_max = float(jnp.max(jnp.abs(state_l.div_hat.data)))
            vor_max = float(jnp.max(jnp.abs(state_l.vor_hat.data)))

            print(f"    step {i+1:4d}: T_err={T_err:.3e}, |div|={div_max:.3e}, |vor|={vor_max:.3e}")

            if not jnp.all(jnp.isfinite(state_l.T_hat.data)):
                print(f"    NaN at step {i+1}")
                return state_l, float('inf')

    T_final = sh_synthesis_3d(grid_l, state_l.T_hat.data)
    T_err = float(jnp.max(jnp.abs(T_final - T_init)))
    return state_l, T_err

# Test explicit at boundary dt
for dt in [420, 480, 540]:
    print(f"\n  === Explicit T15, dt={dt}s, 10 steps ===")
    _, err = run_steps(T, dt, 10, use_si=False)

# Test SI at boundary dt
for dt in [420, 480, 540]:
    print(f"\n  === SI T15, dt={dt}s, 10 steps ===")
    _, err = run_steps(T, dt, 10, use_si=True)

# =====================================================================
# 7. SI correction magnitude at boundary
# =====================================================================
print("\n--- 7. SI Correction Analysis ---")

si_data = precompute_si_matrices(grid, sigma, T_ref=300.0, alpha=0.5, dt=480.0)

# The SI system: (I + alpha^2*dt^2*lambda_n*Gamma) * D_new = D_explicit
# The correction factor at n=T (maximum n):
alpha_val = 0.5
dt_test = 480.0
lambda_T = T * (T + 1) / a**2
factor = alpha_val**2 * dt_test**2 * lambda_T * eigs_real[0]
print(f"  alpha^2 * dt^2 * lambda_max * max_eig(Gamma) = {factor:.4f}")
print(f"  SI matrix at n={T}: I + {factor:.4f} * ... ")
print(f"  This means SI correction is O({factor/(1+factor)*100:.1f}%) of divergence")
print(f"  If factor >> 1, SI strongly damps gravity waves")
print(f"  If factor << 1, gravity waves are not the issue")

# =====================================================================
# 8. Spectral content analysis of growing modes
# =====================================================================
print("\n--- 8. Growing Mode Analysis ---")

# Run 5 steps at dt=480 and examine which spectral modes grow
grid_diag = create_gaussian_grid(T)
sigma_diag = create_sigma_coordinate(NLEV)

config_diag = SpectralPEConfig(
    hyperdiff_coeff=1.0 / (0.5 * 3600.0 * (T*(T+1)/(a*a))**2),
    hyperdiff_order=2,
)
model_diag = SpectralPrimitiveEquationModel(grid_diag, sigma_diag, config_diag)
state_diag = isothermal_rest_state_spectral(grid_diag, sigma_diag, T_init=300.0)

# Compute initial state magnitudes by wavenumber
ns = np.array(grid_diag.ls)  # total wavenumber for each SH coefficient

print(f"  Running 5 steps at T{T}/dt=480...")
for step in range(5):
    state_diag = model_diag.step(state_diag, 480.0)

    # Analyze divergence by wavenumber
    div_data = np.abs(np.array(state_diag.div_hat.data))  # (n_sh, nlev)
    div_max_per_n = np.zeros(T + 1)
    for n in range(T + 1):
        mask = ns == n
        if np.any(mask):
            div_max_per_n[n] = np.max(div_data[mask])

    # Find the mode with largest divergence
    n_max_div = np.argmax(div_max_per_n)
    print(f"  Step {step+1}: max|div|={np.max(div_data):.3e}, "
          f"dominant n={n_max_div}, div[n={n_max_div}]={div_max_per_n[n_max_div]:.3e}")

    if step == 4:  # After 5 steps, show full spectrum
        print(f"  Divergence spectrum after 5 steps at dt=480:")
        for n in range(T + 1):
            if div_max_per_n[n] > 0:
                print(f"    n={n:2d}: max|div|={div_max_per_n[n]:.3e}")

# =====================================================================
# 9. Compare gravity wave CFL with observed stability
# =====================================================================
print("\n--- 9. Summary & Diagnosis ---")
print(f"  Theoretical GW CFL limit: dt_max = {dt_max_gw:.1f}s")
print(f"  Observed stability boundary: ~480-540s (at T15)")
print(f"  Ratio: observed/theoretical = {500/dt_max_gw:.2f}")

if abs(500 - dt_max_gw) / dt_max_gw < 0.3:
    print(f"  >>> DIAGNOSIS: Gravity wave CFL IS the limiting constraint!")
    print(f"  >>> The SI scheme should remove this limit.")
    print(f"  >>> If SI doesn't help, the SI implementation needs fixing.")
elif dt_max_gw > 1000:
    print(f"  >>> DIAGNOSIS: Gravity waves are NOT the bottleneck.")
    print(f"  >>> Something else limits stability (aliasing? adiabatic heating?)")
else:
    print(f"  >>> DIAGNOSIS: Unclear. Need further investigation.")

print("\nDone!")
