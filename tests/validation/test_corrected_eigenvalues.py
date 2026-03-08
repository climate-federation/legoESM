"""Corrected eigenvalue analysis with the RIGHT sign for PGF.

The divergence equation in spectral space is:
  dD_hat/dt = flux_vor_curl - lap * E_hat - R_d*T_ref*lap*lnps_hat

Since lap = -n(n+1)/a^2 (NEGATIVE), we have:
  -lap * E_hat = +lambda * E_hat  (POSITIVE coupling)
  -R_d*T_ref*lap*lnps_hat = +lambda*R_d*T_ref*lnps_hat  (POSITIVE coupling)

Previous analysis used NEGATIVE sign -> showed spurious instability!
"""
import numpy as np
from numpy.linalg import eig

nlev = 20
sigma_top = 0.01
T_ref = 300.0
R_d = 287.05
kappa = R_d / 1004.0
lnps_0 = np.log(1e5)
a = 6.371e6

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

# M matrix (adiabatic: D -> T)
def build_M():
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


def analyze(label, N, M, n_waves=[1, 5, 10, 21, 42]):
    print(f"\n  {label}")
    print(f"  {'n':>4s}  {'max_real':>12s}  {'unstable':>8s}  {'max_imag':>12s}")
    for n_wave in n_waves:
        lambda_n = n_wave * (n_wave + 1) / a**2
        n_total = 2 * nlev + 1

        A = np.zeros((n_total, n_total))

        # D equation: dD_k/dt = +lambda_n * [sum_j N[k,j]*T'_j + R_d*T_ref*lnps']
        # Note: POSITIVE sign (from -lap where lap is negative)
        for k in range(nlev):
            for j in range(nlev):
                A[k, nlev + j] = +lambda_n * N[k, j]  # <<< CORRECTED: + not -
            A[k, 2*nlev] = +lambda_n * R_d * T_ref      # <<< CORRECTED: + not -

        # T equation: dT'_k/dt = sum_j M[k,j]*D_j
        for k in range(nlev):
            for j in range(nlev):
                A[nlev + k, j] = M[k, j]

        # lnps equation: dlnps'/dt = -sum_k D_k * dsigma_k / sigma_range
        for k in range(nlev):
            A[2*nlev, k] = -dsigma[k] / sigma_range

        evals = eig(A)[0]
        max_real = np.max(np.real(evals))
        n_unstable = np.sum(np.real(evals) > 1e-10)
        max_imag = np.max(np.abs(np.imag(evals)))
        print(f"  {n_wave:4d}  {max_real:+12.4e}  {n_unstable:8d}  {max_imag:12.4e}")


M_full = build_M()

# E-variable form
print("=" * 70)
print("CORRECTED SIGN: E-variable form (current code)")
N_evar = R_d * (S + lnps_0 * np.eye(nlev))
analyze("E-variable + current adiabatic", N_evar, M_full)

# Correct PGF (no E-variable extra)
print("\n" + "=" * 70)
print("CORRECTED SIGN: Correct PGF (no lnps_0 diagonal)")
N_correct = R_d * S
analyze("Correct PGF + current adiabatic", N_correct, M_full)

# D-lnps only (no T coupling)
print("\n" + "=" * 70)
print("CORRECTED SIGN: D-lnps only (shallow-water)")
for n_wave in [1, 5, 10, 21]:
    lambda_n = n_wave * (n_wave + 1) / a**2
    A_sw = np.zeros((nlev + 1, nlev + 1))
    for k in range(nlev):
        A_sw[k, nlev] = +lambda_n * R_d * T_ref
    for k in range(nlev):
        A_sw[nlev, k] = -dsigma[k] / sigma_range
    evals = eig(A_sw)[0]
    max_real = np.max(np.real(evals))
    max_imag = np.max(np.abs(np.imag(evals)))
    print(f"  n={n_wave:3d}: real={max_real:+.4e}, imag={max_imag:.4e}")

# M = 0 (no adiabatic heating)
print("\n" + "=" * 70)
print("CORRECTED SIGN: No adiabatic (M=0), correct PGF")
M_zero = np.zeros((nlev, nlev))
analyze("No adiabatic, correct PGF", N_correct, M_zero)

# M = 0, E-variable
print("\n" + "=" * 70)
print("CORRECTED SIGN: No adiabatic (M=0), E-variable")
analyze("No adiabatic, E-variable", N_evar, M_zero)
