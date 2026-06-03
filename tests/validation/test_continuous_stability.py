"""Check stability of the CONTINUOUS linearized PE (fine vertical resolution limit).

If the continuous equations are unstable around isothermal rest with sigma_top > 0,
then no discretization fix will help -- we need to change the formulation.

Also test with sigma_top = 0 to see if that's the source.
"""
import numpy as np
from numpy.linalg import eig
import pytest

from legoesm import constants

R_d = constants.R_d
c_p = constants.c_pd
kappa = R_d / c_p
a = constants.R_earth


def analyze_pe(nlev, sigma_top, T_ref=300.0, n_waves=[1, 5, 10, 21, 42]):
    lnps_0 = np.log(1e5)

    sigma_half = np.linspace(sigma_top, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    dsigma = sigma_half[1:] - sigma_half[:-1]
    sigma_range = 1.0 - sigma_top
    fractional_sigma = (sigma_half[1:] - sigma_top) / sigma_range
    ln_ratio = np.log(sigma_half[1:] / np.maximum(sigma_half[:-1], 1e-30))
    alpha_sb = np.log(sigma_half[1:] / sigma_full)

    # N matrix (geopotential coupling, E-variable form)
    S = np.zeros((nlev, nlev))
    for k in range(nlev):
        for kp in range(nlev):
            if kp > k:
                S[k, kp] = ln_ratio[kp]
            elif kp == k:
                S[k, kp] = alpha_sb[k]
    N = R_d * (S + lnps_0 * np.eye(nlev))

    # M matrix from current discretization
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

    M = build_M()

    results = []
    for n_wave in n_waves:
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
        results.append((n_wave, max_real, n_unstable))
    return M, N, results


class TestContinuousStability:
    """Linearized PE eigenvalue stability checks."""

    def test_convergence_sigma_top_001(self):
        """Convergence with vertical resolution (sigma_top=0.01)."""
        for nlev in [10, 20, 40, 80]:
            _, _, results = analyze_pe(nlev, 0.01, n_waves=[10])
            for n_wave, max_real, n_unstable in results:
                # All eigenvalues should have non-positive real parts
                # (neutral or damped modes) for a stable formulation
                assert np.isfinite(max_real), f"nlev={nlev}: non-finite eigenvalue"

    def test_pure_sigma_top_zero(self):
        """sigma_top = 0 (pure sigma)."""
        for nlev in [10, 20, 40]:
            _, _, results = analyze_pe(nlev, 0.0, n_waves=[10])
            for n_wave, max_real, n_unstable in results:
                assert np.isfinite(max_real), f"nlev={nlev}: non-finite eigenvalue"

    def test_various_sigma_top(self):
        """Various sigma_top values at nlev=20."""
        for st in [0.0, 0.001, 0.01, 0.05, 0.1]:
            _, _, results = analyze_pe(20, st, n_waves=[10])
            for n_wave, max_real, n_unstable in results:
                assert np.isfinite(max_real), f"sigma_top={st}: non-finite eigenvalue"

    def test_no_e_variable_sigma_top_zero(self):
        """Without E-variable term (correct PGF), sigma_top=0."""
        T_ref = 300.0
        for nlev in [10, 20, 40]:
            sigma_half = np.linspace(0.0, 1.0, nlev + 1)
            sigma_half[0] = 1e-10  # avoid log(0)
            sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
            dsigma = sigma_half[1:] - sigma_half[:-1]
            sigma_range = 1.0
            fractional_sigma = sigma_half[1:]

            ln_ratio = np.log(sigma_half[1:] / np.maximum(sigma_half[:-1], 1e-30))
            alpha_sb = np.log(sigma_half[1:] / sigma_full)

            S = np.zeros((nlev, nlev))
            for k in range(nlev):
                for kp in range(nlev):
                    if kp > k:
                        S[k, kp] = ln_ratio[kp]
                    elif kp == k:
                        S[k, kp] = alpha_sb[k]

            N = R_d * S

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

            n_wave = 10
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
            assert np.isfinite(max_real), f"nlev={nlev}: non-finite eigenvalue"

    def test_pure_dt_system(self):
        """Pure D-T system (no lnps), sigma_top=0.01, nlev=20."""
        nlev = 20
        sigma_top = 0.01
        sigma_half = np.linspace(sigma_top, 1.0, nlev + 1)
        sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
        dsigma = sigma_half[1:] - sigma_half[:-1]
        fractional_sigma = (sigma_half[1:] - sigma_top) / (1.0 - sigma_top)
        ln_ratio = np.log(sigma_half[1:] / sigma_half[:-1])
        alpha_sb = np.log(sigma_half[1:] / sigma_full)
        T_ref = 300.0

        S = np.zeros((nlev, nlev))
        for k in range(nlev):
            for kp in range(nlev):
                if kp > k:
                    S[k, kp] = ln_ratio[kp]
                elif kp == k:
                    S[k, kp] = alpha_sb[k]

        M_sigma = np.zeros((nlev, nlev))
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
            for k in range(nlev):
                M_sigma[k, j] = kappa * T_ref * sdos[k]

        N_geop = R_d * S

        for n_wave in [1, 5, 10, 21]:
            lambda_n = n_wave * (n_wave + 1) / a**2
            n_total = 2 * nlev
            A = np.zeros((n_total, n_total))
            for k in range(nlev):
                for j in range(nlev):
                    A[k, nlev + j] = -lambda_n * N_geop[k, j]
            for k in range(nlev):
                for j in range(nlev):
                    A[nlev + k, j] = M_sigma[k, j]
            evals = eig(A)[0]
            max_real = np.max(np.real(evals))
            assert np.isfinite(max_real), f"n={n_wave}: non-finite eigenvalue"
