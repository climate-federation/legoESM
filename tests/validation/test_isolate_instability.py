"""Isolate the exact source of instability by turning terms on/off.

Test combinations:
1. No adiabatic heating at all (M=0): just geopotential coupling
2. Only sigma_dot/sigma adiabatic (no dlnps/dt in M)
3. Only dlnps/dt adiabatic (no sigma_dot in M)
4. Full adiabatic with correct PGF
5. No T equation at all (D-lnps only)
6. No lnps equation at all (D-T only)
"""
import numpy as np
from numpy.linalg import eig

from legoesm import constants

nlev = 20
sigma_top = 0.01
T_ref = 300.0
R_d = constants.R_d
kappa = R_d / constants.c_pd
lnps_0 = np.log(1e5)
a = constants.R_earth

sigma_half = np.linspace(sigma_top, 1.0, nlev + 1)
sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
dsigma = sigma_half[1:] - sigma_half[:-1]
sigma_range = 1.0 - sigma_top
fractional_sigma = (sigma_half[1:] - sigma_top) / sigma_range
ln_ratio = np.log(sigma_half[1:] / sigma_half[:-1])
alpha_sb = np.log(sigma_half[1:] / sigma_full)

# S matrix (geopotential coupling)
S = np.zeros((nlev, nlev))
for k in range(nlev):
    for kp in range(nlev):
        if kp > k:
            S[k, kp] = ln_ratio[kp]
        elif kp == k:
            S[k, kp] = alpha_sb[k]

# Build different M matrices
def build_M(include_sdos=True, include_dlnps=True):
    M = np.zeros((nlev, nlev))
    for j in range(nlev):
        D = np.zeros(nlev)
        D[j] = 1.0
        D_total = np.sum(D * dsigma)
        cumsum_div = np.cumsum(D * dsigma)
        sd = np.zeros(nlev + 1)
        for m in range(nlev):
            sd[m + 1] = fractional_sigma[m] * D_total - cumsum_div[m]
        sd[-1] = 0.0
        sd_full = 0.5 * (sd[:-1] + sd[1:])
        sdos = sd_full / sigma_full if include_sdos else np.zeros(nlev)
        dlnps = -D_total / sigma_range if include_dlnps else 0.0
        for k in range(nlev):
            M[k, j] = kappa * T_ref * (sdos[k] + dlnps)
    return M


def eigenvalues_DT_lnps(N, M, n_wave, include_lnps=True, include_T=True):
    """Compute eigenvalues of coupled D-T-lnps system."""
    lambda_n = n_wave * (n_wave + 1) / a**2

    if include_T and include_lnps:
        n_total = 2 * nlev + 1
        A = np.zeros((n_total, n_total))
        for k in range(nlev):
            for j in range(nlev):
                A[k, nlev + j] = -lambda_n * N[k, j]
            A[k, 2*nlev] = -lambda_n * R_d * T_ref
        for k in range(nlev):
            for j in range(nlev):
                A[nlev + k, j] = M[k, j]
        for k in range(nlev):
            A[2*nlev, k] = -dsigma[k] / sigma_range
    elif include_T and not include_lnps:
        # D-T only (no lnps equation, no lnps->D coupling)
        n_total = 2 * nlev
        A = np.zeros((n_total, n_total))
        for k in range(nlev):
            for j in range(nlev):
                A[k, nlev + j] = -lambda_n * N[k, j]
        for k in range(nlev):
            for j in range(nlev):
                A[nlev + k, j] = M[k, j]
    elif not include_T and include_lnps:
        # D-lnps only (no T equation, no T->D coupling)
        n_total = nlev + 1
        A = np.zeros((n_total, n_total))
        for k in range(nlev):
            A[k, nlev] = -lambda_n * R_d * T_ref
        for k in range(nlev):
            A[nlev, k] = -dsigma[k] / sigma_range

    evals = eig(A)[0]
    return evals


n_test = 10

print("=" * 70)
print(f"All tests at n={n_test}, nlev={nlev}, sigma_top={sigma_top}")
print("=" * 70)

# Test 1: No adiabatic (M=0), correct PGF, full D-T-lnps
M_zero = np.zeros((nlev, nlev))
N_correct = R_d * S
evals = eigenvalues_DT_lnps(N_correct, M_zero, n_test)
print(f"\n1. M=0 (no adiabatic), correct PGF, D-T-lnps:")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")
print(f"   max_imag = {np.max(np.abs(np.imag(evals))):.4e}")

# Test 2: Only sigma_dot/sigma adiabatic (no dlnps contribution to T)
M_sdos_only = build_M(include_sdos=True, include_dlnps=False)
evals = eigenvalues_DT_lnps(N_correct, M_sdos_only, n_test)
print(f"\n2. Only sigma_dot/sigma adiabatic, correct PGF:")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")

# Test 3: Only dlnps/dt adiabatic (no sigma_dot contribution to T)
M_dlnps_only = build_M(include_sdos=False, include_dlnps=True)
evals = eigenvalues_DT_lnps(N_correct, M_dlnps_only, n_test)
print(f"\n3. Only dlnps/dt adiabatic, correct PGF:")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")

# Test 4: Full adiabatic, correct PGF
M_full = build_M(include_sdos=True, include_dlnps=True)
evals = eigenvalues_DT_lnps(N_correct, M_full, n_test)
print(f"\n4. Full adiabatic, correct PGF:")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")

# Test 5: D-T only (no lnps)
evals = eigenvalues_DT_lnps(N_correct, M_sdos_only, n_test, include_lnps=False)
print(f"\n5. D-T only (no lnps), sigma_dot/sigma adiabatic:")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")

evals = eigenvalues_DT_lnps(N_correct, M_full, n_test, include_lnps=False)
print(f"\n6. D-T only (no lnps), full adiabatic:")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")

# Test 7: D-lnps only (no T coupling to D)
evals = eigenvalues_DT_lnps(N_correct, M_zero, n_test, include_T=False)
print(f"\n7. D-lnps only (shallow water-like):")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")
print(f"   max_imag = {np.max(np.abs(np.imag(evals))):.4e}")

# Test 8: E-variable with M=0
N_evar = R_d * (S + lnps_0 * np.eye(nlev))
evals = eigenvalues_DT_lnps(N_evar, M_zero, n_test)
print(f"\n8. M=0, E-variable form, D-T-lnps:")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")

# Test 9: EC flux form, correct PGF
M_ec = np.zeros((nlev, nlev))
for j in range(nlev):
    D = np.zeros(nlev)
    D[j] = 1.0
    D_total = np.sum(D * dsigma)
    cumsum_div = np.cumsum(D * dsigma)
    sd = np.zeros(nlev + 1)
    for m in range(nlev):
        sd[m + 1] = fractional_sigma[m] * D_total - cumsum_div[m]
    sd[-1] = 0.0
    flux_upper = sigma_half[1:] * sd[1:]
    flux_lower = sigma_half[:-1] * sd[:-1]
    ec_sdos = (flux_upper - flux_lower) / dsigma
    dlnps = -D_total / sigma_range
    for k in range(nlev):
        M_ec[k, j] = kappa * T_ref * (ec_sdos[k] + dlnps)

evals = eigenvalues_DT_lnps(N_correct, M_ec, n_test)
print(f"\n9. EC flux adiabatic, correct PGF:")
print(f"   max_real = {np.max(np.real(evals)):+.4e}")

# Test 10: D-T only with EC flux sigma_dot and NO dlnps/dt in adiabatic
M_ec_sdos_only = np.zeros((nlev, nlev))
for j in range(nlev):
    D = np.zeros(nlev)
    D[j] = 1.0
    D_total = np.sum(D * dsigma)
    cumsum_div = np.cumsum(D * dsigma)
    sd = np.zeros(nlev + 1)
    for m in range(nlev):
        sd[m + 1] = fractional_sigma[m] * D_total - cumsum_div[m]
    sd[-1] = 0.0
    flux_upper = sigma_half[1:] * sd[1:]
    flux_lower = sigma_half[:-1] * sd[:-1]
    ec_sdos = (flux_upper - flux_lower) / dsigma
    for k in range(nlev):
        M_ec_sdos_only[k, j] = kappa * T_ref * ec_sdos[k]

evals = eigenvalues_DT_lnps(N_correct, M_ec_sdos_only, n_test, include_lnps=False)
print(f"\n10. D-T only, EC flux sigma_dot only (no dlnps):")
print(f"    max_real = {np.max(np.real(evals)):+.4e}")

# Same but with standard sigma_dot/sigma
M_sdos_only2 = build_M(include_sdos=True, include_dlnps=False)
evals = eigenvalues_DT_lnps(N_correct, M_sdos_only2, n_test, include_lnps=False)
print(f"\n11. D-T only, standard sigma_dot/sigma (no dlnps):")
print(f"    max_real = {np.max(np.real(evals)):+.4e}")

# Test 12: Antisymmetric M (exact), correct PGF, no lnps
M_anti = np.zeros((nlev, nlev))
for k in range(nlev):
    for j in range(nlev):
        M_anti[k, j] = -(dsigma[j] / dsigma[k]) * R_d * S[j, k]
evals = eigenvalues_DT_lnps(N_correct, M_anti, n_test, include_lnps=False)
print(f"\n12. D-T only, antisymmetric M, correct PGF:")
print(f"    max_real = {np.max(np.real(evals)):+.4e}")
print(f"    max_imag = {np.max(np.abs(np.imag(evals))):.4e}")

# Test 13: D-T-lnps with antisymmetric M, correct PGF
evals = eigenvalues_DT_lnps(N_correct, M_anti, n_test)
print(f"\n13. D-T-lnps, antisymmetric M, correct PGF:")
print(f"    max_real = {np.max(np.real(evals)):+.4e}")
