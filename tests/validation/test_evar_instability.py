"""Prove the E-variable form is the root cause of the PE instability.

For a 1-level model (barotropic PE), the system reduces to:
  dD/dt = -lambda * [N*T' + R_d*T_ref*lnps']
  dT'/dt = M*D  (adiabatic heating)
  dlnps'/dt = -D

Characteristic equation: s^2 = -lambda*(N*M + R_d*T_ref)

If N*M + R_d*T_ref > 0: stable (imaginary eigenvalues = gravity waves)
If N*M + R_d*T_ref < 0: UNSTABLE (real eigenvalues = exponential growth)

The E-variable form adds R_d*lnps_0 to N, which makes N*M large and negative.
"""
import numpy as np

from legoesm import constants

R_d = constants.R_d
c_p = constants.c_pd
kappa = R_d / c_p
T_ref = 300.0
lnps_0 = np.log(1e5)  # ~11.51
a = constants.R_earth

# 1-level sigma coordinate
sigma_top = 0.01
sigma_half = np.array([sigma_top, 1.0])
sigma_full = 0.5 * (sigma_half[0] + sigma_half[1])
dsigma = 1.0 - sigma_top
sigma_range = 1.0 - sigma_top
ln_ratio = np.log(sigma_half[1] / sigma_half[0])
alpha_sb = np.log(sigma_half[1] / sigma_full)

print(f"sigma_full = {sigma_full:.4f}")
print(f"alpha_sb = {alpha_sb:.4f}")
print(f"ln_ratio = {ln_ratio:.4f}")
print(f"lnps_0 = {lnps_0:.4f}")
print()

# M: D -> T coupling through adiabatic heating
# For 1 level: sigma_dot = 0 everywhere (both boundaries), so sigma_dot_full = 0
# Only dlnps/dt = -D contributes
M = kappa * T_ref * (-1.0)  # = -kappa*T_ref
print(f"M (adiabatic D->T) = {M:.4f}")
print(f"  = -kappa*T_ref = {-kappa*T_ref:.4f}")
print()

# N with E-variable: N = R_d*(alpha_sb + lnps_0)
N_evar = R_d * (alpha_sb + lnps_0)
print(f"N (E-variable) = R_d*(alpha_sb + lnps_0) = {N_evar:.2f}")
criterion_evar = N_evar * M + R_d * T_ref
print(f"  N*M + R_d*T_ref = {criterion_evar:.2f}")
print(f"  {'STABLE' if criterion_evar > 0 else 'UNSTABLE'}")
print()

# N without E-variable: N = R_d*alpha_sb
N_correct = R_d * alpha_sb
print(f"N (correct PGF) = R_d*alpha_sb = {N_correct:.2f}")
criterion_correct = N_correct * M + R_d * T_ref
print(f"  N*M + R_d*T_ref = {criterion_correct:.2f}")
print(f"  {'STABLE' if criterion_correct > 0 else 'UNSTABLE'}")
print()

# Why is N*M + R_d*T_ref the criterion?
# For 1-level: s^2 = -lambda*(N*M + R_d*T_ref)
# Need N*M + R_d*T_ref > 0 for imaginary s (stable)
# N*M = R_d*(alpha_sb + lnps_0)*(-kappa*T_ref)
# = -R_d*kappa*T_ref*(alpha_sb + lnps_0)
# = -R_d^2*T_ref/c_p * (alpha_sb + lnps_0)
#
# R_d*T_ref = R_d*T_ref
#
# Sum = R_d*T_ref*(1 - kappa*(alpha_sb + lnps_0))  [E-variable]
# or  = R_d*T_ref*(1 - kappa*alpha_sb)              [correct PGF]
#
# Since kappa=0.286 and alpha_sb~0.68: kappa*alpha_sb ~ 0.195 < 1 -> STABLE
# But kappa*(alpha_sb+lnps_0) ~ 0.286*12.2 = 3.49 > 1 -> UNSTABLE!

print("=" * 60)
print("Physical interpretation:")
print("=" * 60)
print(f"  kappa = {kappa:.4f}")
print(f"  kappa * alpha_sb = {kappa*alpha_sb:.4f} {'< 1 STABLE' if kappa*alpha_sb < 1 else '> 1 UNSTABLE'}")
print(f"  kappa * (alpha_sb + lnps_0) = {kappa*(alpha_sb+lnps_0):.4f} {'< 1 STABLE' if kappa*(alpha_sb+lnps_0) < 1 else '> 1 UNSTABLE'}")
print()
print("The E-variable form adds R_d*lnps_0*lap*T' to the D equation.")
print("This is a same-level T'->D coupling that creates positive feedback:")
print("  Convergent D -> cooling dT' < 0 -> negative lap(T') -> MORE convergence")
print("The feedback is proportional to lnps_0 ~ 11.5, overwhelmingly dominant.")
print()

# Verify with full eigenvalue analysis
from numpy.linalg import eig

print("=" * 60)
print("Full eigenvalue verification (nlev=20)")
print("=" * 60)

nlev = 20
sigma_half = np.linspace(sigma_top, 1.0, nlev + 1)
sigma_full_v = 0.5 * (sigma_half[:-1] + sigma_half[1:])
dsigma_v = sigma_half[1:] - sigma_half[:-1]
sigma_range_v = 1.0 - sigma_top
fractional_sigma_v = (sigma_half[1:] - sigma_top) / sigma_range_v
ln_ratio_v = np.log(sigma_half[1:] / sigma_half[:-1])
alpha_sb_v = np.log(sigma_half[1:] / sigma_full_v)

S = np.zeros((nlev, nlev))
for k in range(nlev):
    for kp in range(nlev):
        if kp > k:
            S[k, kp] = ln_ratio_v[kp]
        elif kp == k:
            S[k, kp] = alpha_sb_v[k]

def build_M():
    M = np.zeros((nlev, nlev))
    for j in range(nlev):
        D = np.zeros(nlev)
        D[j] = 1.0
        D_total = np.sum(D * dsigma_v)
        cumsum_div = np.cumsum(D * dsigma_v)
        sd = np.zeros(nlev + 1)
        for m in range(nlev):
            sd[m + 1] = fractional_sigma_v[m] * D_total - cumsum_div[m]
        sd[-1] = 0.0
        sd_full = 0.5 * (sd[:-1] + sd[1:])
        sdos = sd_full / sigma_full_v
        dlnps = -D_total / sigma_range_v
        for k in range(nlev):
            M[k, j] = kappa * T_ref * (sdos[k] + dlnps)
    return M

M_mat = build_M()

for label, N_mat in [
    ("E-variable (current code)", R_d * (S + lnps_0 * np.eye(nlev))),
    ("Correct PGF (no lnps_0*I)", R_d * S),
]:
    print(f"\n  {label}:")
    for n_wave in [1, 5, 10, 21, 42]:
        lambda_n = n_wave * (n_wave + 1) / a**2
        n_total = 2 * nlev + 1
        A = np.zeros((n_total, n_total))
        for k in range(nlev):
            for j in range(nlev):
                A[k, nlev + j] = -lambda_n * N_mat[k, j]
            A[k, 2*nlev] = -lambda_n * R_d * T_ref
        for k in range(nlev):
            for j in range(nlev):
                A[nlev + k, j] = M_mat[k, j]
        for k in range(nlev):
            A[2*nlev, k] = -dsigma_v[k] / sigma_range_v
        evals = eig(A)[0]
        max_real = np.max(np.real(evals))
        n_unstable = np.sum(np.real(evals) > 1e-12)
        max_imag = np.max(np.abs(np.imag(evals)))
        print(f"    n={n_wave:3d}: real={max_real:+.4e}, imag={max_imag:.4e}, unstable={n_unstable}")
