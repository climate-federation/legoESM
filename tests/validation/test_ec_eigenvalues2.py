"""Test eigenvalues with different PGF formulations and adiabatic discretizations.

Key question: is the instability from the E-variable form (extra R_d*lnps_0*T' in D equation)
or from the adiabatic heating discretization?
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

# Build sigma coordinate
sigma_half = np.linspace(sigma_top, 1.0, nlev + 1)
sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
dsigma = sigma_half[1:] - sigma_half[:-1]
sigma_range = 1.0 - sigma_top
fractional_sigma = (sigma_half[1:] - sigma_top) / sigma_range
ln_ratio = np.log(sigma_half[1:] / sigma_half[:-1])
alpha_sb = np.log(sigma_half[1:] / sigma_full)


def build_S():
    S = np.zeros((nlev, nlev))
    for k in range(nlev):
        for kp in range(nlev):
            if kp > k:
                S[k, kp] = ln_ratio[kp]
            elif kp == k:
                S[k, kp] = alpha_sb[k]
    return S


def build_M_current():
    """Current code: sigma_dot_full / sigma_full"""
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
        sdos = sd_full / sigma_full
        dlnps = -D_total / sigma_range
        for k in range(nlev):
            M[k, j] = kappa * T_ref * (sdos[k] + dlnps)
    return M


def build_M_EC_flux():
    """Energy-conserving flux form: (sigma*sd_upper - sigma*sd_lower)/dsigma"""
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
        flux_upper = sigma_half[1:] * sd[1:]
        flux_lower = sigma_half[:-1] * sd[:-1]
        ec_sdos = (flux_upper - flux_lower) / dsigma
        dlnps = -D_total / sigma_range
        for k in range(nlev):
            M[k, j] = kappa * T_ref * (ec_sdos[k] + dlnps)
    return M


def analyze_system(label, N, M, n_waves=[1, 5, 10, 21, 42]):
    print(f"\n  {label}")
    print(f"  {'n':>4s}  {'max_real':>12s}  {'unstable':>8s}  {'max_imag':>12s}")
    for n_wave in n_waves:
        a = constants.R_earth
        lambda_n = n_wave * (n_wave + 1) / a**2
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

        evals = eig(A)[0]
        max_real = np.max(np.real(evals))
        n_unstable = np.sum(np.real(evals) > 1e-12)
        max_imag = np.max(np.abs(np.imag(evals)))
        print(f"  {n_wave:4d}  {max_real:+12.4e}  {n_unstable:8d}  {max_imag:12.4e}")


S = build_S()
M_current = build_M_current()
M_ec = build_M_EC_flux()

# Case 1: E-variable form (current code)
print("=" * 70)
print("Case 1: E-variable form + current adiabatic")
N_evar = R_d * (S + lnps_0 * np.eye(nlev))
analyze_system("E-variable + current", N_evar, M_current)

# Case 2: E-variable + EC flux adiabatic
print("\n" + "=" * 70)
print("Case 2: E-variable form + EC flux adiabatic")
analyze_system("E-variable + EC flux", N_evar, M_ec)

# Case 3: Correct PGF (no E-variable extra) + current adiabatic
print("\n" + "=" * 70)
print("Case 3: Correct PGF (no lnps_0 diagonal) + current adiabatic")
N_correct = R_d * S
analyze_system("Correct PGF + current", N_correct, M_current)

# Case 4: Correct PGF + EC flux adiabatic
print("\n" + "=" * 70)
print("Case 4: Correct PGF + EC flux adiabatic")
analyze_system("Correct PGF + EC flux", N_correct, M_ec)

# Case 5: Correct PGF + theoretical M (antisymmetric with R_d*S)
print("\n" + "=" * 70)
print("Case 5: Correct PGF + theoretical M (exact antisymmetry)")
M_theory_correct = np.zeros((nlev, nlev))
for k in range(nlev):
    for j in range(nlev):
        M_theory_correct[k, j] = -(dsigma[j] / dsigma[k]) * R_d * S[j, k]
analyze_system("Correct PGF + theory", N_correct, M_theory_correct)

# Case 6: Include BOTH D-T antisymmetry AND D-lnps pairing
# For the D-lnps coupling to conserve energy:
# D -> lnps: dlnps/dt = -sum_k D_k*dsigma_k/sigma_range
# lnps -> D: dD_k/dt += -lambda*R_d*T_ref*lnps
# These should be "paired" in energy conservation.
# The T-lnps coupling through adiabatic:
# dT_k/dt includes kappa*T_ref*dlnps/dt = kappa*T_ref*(-D_total/sigma_range)
# This is part of M already.
#
# Actually, let me compute the FULL conserved energy quadratic form and see
# if any choice of M makes it work.
#
# Energy: Q = sum_k [D_k^2/(2*lambda) + c_T*T'_k^2] * dsigma_k + c_lnps*lnps'^2 + c_cross*sum_k T'_k*dsigma_k*lnps'
# where c_T, c_lnps, c_cross need to be determined.
#
# dQ/dt = 0 requires matching all terms.

# Actually, let me try the approach from Staniforth & Wood (2008) or similar:
# The proper energy for linearized PE around isothermal rest involves
# specific cross-terms. Let me try:
# Q = sum_k [D_k^2/(2*lambda) * dsigma_k + c_p*T'_k^2/(2*T_ref) * dsigma_k]
#     + c_p*T_ref*sigma_range/2 * lnps'^2 + c_p*lnps' * sum_k T'_k * dsigma_k

print("\n" + "=" * 70)
print("Case 6: Testing quadratic form conservation")
print("=" * 70)

# Define the energy matrix E such that Q = X^T E X
# X = [D_0,...,D_{L-1}, T'_0,...,T'_{L-1}, lnps']
# Q = sum_k D_k^2/(2*lambda) * dsigma_k + sum_k c_p*T'_k^2/(2*T_ref)*dsigma_k
#     + c_p*T_ref*sigma_range/2 * lnps'^2 + c_p*sum_k T'_k*dsigma_k*lnps'
c_p = constants.c_pd

for n_wave in [5, 10]:
    a = constants.R_earth
    lambda_n = n_wave * (n_wave + 1) / a**2
    n_total = 2 * nlev + 1

    # Energy matrix
    E_mat = np.zeros((n_total, n_total))
    for k in range(nlev):
        E_mat[k, k] = dsigma[k] / (2 * lambda_n)                    # D_k^2
        E_mat[nlev+k, nlev+k] = c_p * dsigma[k] / (2 * T_ref)      # T'^2
        # Cross term T'_k * lnps'
        E_mat[nlev+k, 2*nlev] = c_p * dsigma[k] / 2
        E_mat[2*nlev, nlev+k] = c_p * dsigma[k] / 2
    E_mat[2*nlev, 2*nlev] = c_p * T_ref * sigma_range / 2           # lnps'^2

    # Check if E_mat is positive definite
    evals_E = np.linalg.eigvalsh(E_mat)
    print(f"\n  n={n_wave}: Energy matrix min eigenvalue: {np.min(evals_E):.6e}")

    # For each choice of N, M, check if d(Q)/dt = 0, i.e., E*A + A^T*E = 0
    for label, N, M in [
        ("E-var + current", N_evar, M_current),
        ("E-var + EC flux", N_evar, M_ec),
        ("Correct PGF + current", N_correct, M_current),
        ("Correct PGF + EC flux", N_correct, M_ec),
    ]:
        A = np.zeros((n_total, n_total))
        for k in range(nlev):
            for j in range(nlev):
                A[k, nlev+j] = -lambda_n * N[k, j]
            A[k, 2*nlev] = -lambda_n * R_d * T_ref
        for k in range(nlev):
            for j in range(nlev):
                A[nlev+k, j] = M[k, j]
        for k in range(nlev):
            A[2*nlev, k] = -dsigma[k] / sigma_range

        sym_err = E_mat @ A + A.T @ E_mat
        print(f"    {label:30s}: ||E*A + A^T*E|| = {np.linalg.norm(sym_err):.4e}")
