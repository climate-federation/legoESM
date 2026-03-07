"""Test eigenvalues of linearized PE with energy-conserving vs current discretization.

Builds the M (D→T coupling) and N (T→D coupling) matrices from the discretization,
then checks whether the coupled system eigenvalues are purely imaginary (stable)
or have positive real parts (unstable).
"""
import numpy as np
from numpy.linalg import eig

# ---- Parameters ----
nlev = 20
sigma_top = 0.01
T_ref = 300.0
R_d = 287.05
kappa = R_d / 1004.0
lnps_0 = np.log(1e5)  # ~11.51

# ---- Build sigma coordinate ----
sigma_half = np.linspace(sigma_top, 1.0, nlev + 1)
sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
dsigma = sigma_half[1:] - sigma_half[:-1]
sigma_range = 1.0 - sigma_top
fractional_sigma = (sigma_half[1:] - sigma_top) / sigma_range

# Simmons-Burridge coefficients
ln_ratio = np.log(sigma_half[1:] / sigma_half[:-1])
alpha_sb = np.log(sigma_half[1:] / sigma_full)

# ---- Build N matrix (geopotential coupling T→D, E-variable form) ----
S = np.zeros((nlev, nlev))
for k in range(nlev):
    for kp in range(nlev):
        if kp > k:
            S[k, kp] = ln_ratio[kp]
        elif kp == k:
            S[k, kp] = alpha_sb[k]

N = R_d * (S + lnps_0 * np.eye(nlev))

# ---- Build M matrix: CURRENT code (sigma_dot_full / sigma_full) ----
def build_M_current():
    M = np.zeros((nlev, nlev))
    for j in range(nlev):
        # D_j = 1, all other D = 0
        D = np.zeros(nlev)
        D[j] = 1.0
        D_total = np.sum(D * dsigma)
        cumsum_div = np.cumsum(D * dsigma)

        # sigma_dot at half-levels (nlev+1)
        sd = np.zeros(nlev + 1)
        for m in range(nlev):
            sd[m + 1] = fractional_sigma[m] * D_total - cumsum_div[m]
        sd[-1] = 0.0

        # sigma_dot_full (interpolated to full levels)
        sd_full = 0.5 * (sd[:-1] + sd[1:])

        # omega/p from sigma_dot part: sigma_dot_full / sigma_full
        sdos = sd_full / sigma_full

        # dlnps/dt = -D_total / sigma_range
        dlnps = -D_total / sigma_range

        # dT_k/dt = kappa * T_ref * (sdos_k + dlnps)
        for k in range(nlev):
            M[k, j] = kappa * T_ref * (sdos[k] + dlnps)
    return M

# ---- Build M matrix: ENERGY-CONSERVING (flux form of sigma*sigma_dot) ----
def build_M_EC():
    M = np.zeros((nlev, nlev))
    for j in range(nlev):
        D = np.zeros(nlev)
        D[j] = 1.0
        D_total = np.sum(D * dsigma)
        cumsum_div = np.cumsum(D * dsigma)

        # sigma_dot at half-levels
        sd = np.zeros(nlev + 1)
        for m in range(nlev):
            sd[m + 1] = fractional_sigma[m] * D_total - cumsum_div[m]
        sd[-1] = 0.0

        # EC adiabatic: (sigma_half[k+1]*sd[k+1] - sigma_half[k]*sd[k]) / dsigma[k]
        flux_upper = sigma_half[1:] * sd[1:]
        flux_lower = sigma_half[:-1] * sd[:-1]
        ec_sdos = (flux_upper - flux_lower) / dsigma

        dlnps = -D_total / sigma_range

        for k in range(nlev):
            M[k, j] = kappa * T_ref * (ec_sdos[k] + dlnps)
    return M

# ---- Build M matrix: THEORETICAL energy-conserving (from antisymmetry with N) ----
def build_M_theory():
    """M[k,j] = -(dsigma[j]/dsigma[k]) * N[j,k]"""
    M = np.zeros((nlev, nlev))
    for k in range(nlev):
        for j in range(nlev):
            M[k, j] = -(dsigma[j] / dsigma[k]) * N[j, k]
    return M

# ---- Eigenvalue analysis ----
def analyze(label, M):
    # Include lnps coupling
    # State vector: [D_0,...,D_{L-1}, T'_0,...,T'_{L-1}, lnps']
    # Size: 2*nlev + 1
    n_total = 2 * nlev + 1

    for n_wave in [1, 5, 10, 21]:
        a = 6.371e6
        lambda_n = n_wave * (n_wave + 1) / a**2

        A = np.zeros((n_total, n_total))

        # D equations: dD_k/dt = -lambda_n * [sum_j N[k,j]*T'_j + R_d*T_ref*lnps']
        for k in range(nlev):
            for j in range(nlev):
                A[k, nlev + j] = -lambda_n * N[k, j]  # T'_j -> D_k
            A[k, 2*nlev] = -lambda_n * R_d * T_ref  # lnps' -> D_k

        # T equations: dT'_k/dt = sum_j M[k,j]*D_j
        for k in range(nlev):
            for j in range(nlev):
                A[nlev + k, j] = M[k, j]

        # lnps equation: dlnps'/dt = -sum_k D_k * dsigma_k / sigma_range
        for k in range(nlev):
            A[2*nlev, k] = -dsigma[k] / sigma_range

        evals = eig(A)[0]
        max_real = np.max(np.real(evals))
        n_unstable = np.sum(np.real(evals) > 1e-12)

        print(f"  n={n_wave:3d}: max_real={max_real:+.6e}, unstable_modes={n_unstable}")

# ---- Check antisymmetry ----
def check_antisymmetry(label, M):
    """Check if dsigma_k*N[k,j] + dsigma_j*M[j,k] = 0"""
    err = np.zeros((nlev, nlev))
    for k in range(nlev):
        for j in range(nlev):
            err[k, j] = dsigma[k] * N[k, j] + dsigma[j] * M[j, k]
    print(f"  Antisymmetry error max: {np.max(np.abs(err)):.6e}")
    print(f"  Antisymmetry error RMS: {np.sqrt(np.mean(err**2)):.6e}")


print("=" * 70)
print("1. CURRENT discretization (sigma_dot_full / sigma_full)")
print("=" * 70)
M_current = build_M_current()
check_antisymmetry("current", M_current)
analyze("current", M_current)

print()
print("=" * 70)
print("2. ENERGY-CONSERVING flux form (sigma*sigma_dot flux divergence)")
print("=" * 70)
M_ec = build_M_EC()
check_antisymmetry("EC flux", M_ec)
analyze("EC flux", M_ec)

print()
print("=" * 70)
print("3. THEORETICAL (M = -dsigma_j/dsigma_k * N^T)")
print("=" * 70)
M_theory = build_M_theory()
check_antisymmetry("theory", M_theory)
analyze("theory", M_theory)

print()
print("=" * 70)
print("4. Comparison of M matrices (max abs difference)")
print("=" * 70)
print(f"  Current vs EC flux:  {np.max(np.abs(M_current - M_ec)):.6e}")
print(f"  Current vs Theory:   {np.max(np.abs(M_current - M_theory)):.6e}")
print(f"  EC flux vs Theory:   {np.max(np.abs(M_ec - M_theory)):.6e}")

# Print diagonal comparison
print()
print("  Diagonal comparison (first 5 levels):")
print(f"  {'k':>3s}  {'M_current':>12s}  {'M_EC':>12s}  {'M_theory':>12s}")
for k in range(min(5, nlev)):
    print(f"  {k:3d}  {M_current[k,k]:+12.4f}  {M_ec[k,k]:+12.4f}  {M_theory[k,k]:+12.4f}")
