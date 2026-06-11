"""Fast-SBM Bott coalescence: Courant tables, conservation, Golovin box.

Oracle: WRF ``module_mp_fast_sbm.F`` ``courant_bott_KS``/``coll_xxx_lwf``.
The Golovin box test mirrors the SDM one
(``test_sdm_coalescence.test_golovin_number_decay_matches_analytic``):
for the additive kernel K = b(X_i+X_j) the moment laws
N(t) = N0·exp(−b_m L t) and M2(t) = M2(0)·exp(+2 b_m L t) are exact for any
initial spectrum (b_m = b/ρ_w on mass), and L is conserved.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm import (
    bott_coalescence,
    collision_ck_matrix,
    discretize_exponential,
    g_from_f,
    mass_density_from_g,
    mass_doubling_grid,
    number_density,
    number_density_from_g,
    precompute_collision_tables,
    radius_from_mass,
)
from legoesm.atmosphere.physics.microphysics.sdm.kernels import golovin_kernel

jax.config.update("jax_enable_x64", True)

B_GOLOVIN = 1.5e3          # [s^-1 per m^3 of droplet volume] (SDM test value)
N0 = 1.0e9                 # [m^-3]
R0 = 1.4e-5                # mean-volume-ish radius [m] (SDM test value)


def _setup():
    m = mass_doubling_grid()
    tables = precompute_collision_tables(m)
    r = radius_from_mass(m)
    kernel = golovin_kernel(r[:, None], r[None, :], B_GOLOVIN)
    return m, tables, kernel


def test_courant_tables_doubling_grid_structure():
    m, tables, _ = _setup()
    n = m.shape[0]
    i = np.asarray(tables.i_idx)
    j = np.asarray(tables.j_idx)
    k = np.asarray(tables.k_idx)
    c = np.asarray(tables.c_pair)
    # All (i <= j) pairs over source bins 0..n-2, oracle order.
    assert len(i) == (n - 1) * n // 2
    assert np.all(i <= j) and np.all(j <= n - 2)
    # Self-collection doubles the mass exactly: target k = j+1, Courant 0.
    self_pairs = i == j
    np.testing.assert_array_equal(k[self_pairs],
                                  np.minimum(j[self_pairs] + 1, n - 2))
    np.testing.assert_array_equal(c[self_pairs], 0.0)
    # Mixed pairs: m_i + m_j lands inside bin j..j+1 → target j, c in (0,1).
    mixed = i < j
    np.testing.assert_array_equal(k[mixed], j[mixed])
    assert np.all(c[mixed] > 0.0) and np.all(c[mixed] < 1.0)
    # Courant value matches the oracle formula ln(x0/m_{k})/ln2 for mixed
    # pairs (m_{k-1} of the landing bin is bin j itself = k).
    mm = np.asarray(m)
    c_expect = np.log((mm[i[mixed]] + mm[j[mixed]]) / mm[j[mixed]]) / np.log(2.0)
    np.testing.assert_allclose(c[mixed], c_expect, rtol=1e-12)


def test_empty_spectrum_fixed_point():
    m, tables, kernel = _setup()
    ck = collision_ck_matrix(kernel, 1.0)
    g0 = jnp.zeros_like(m)
    g1 = bott_coalescence(g0, ck, m, tables)
    np.testing.assert_array_equal(np.asarray(g1), 0.0)


def test_single_bin_self_collection_target():
    # Mass in bin b alone: self-collection deposits into b+1 (Courant c = 0
    # on a doubling grid). Gauss-Seidel sweep semantics (oracle): the later
    # pair (b, b+1) in the SAME sweep sees the just-created b+1 mass and
    # fluxes a little of it on to b+2 — so assert exact conservation and
    # that b+1 dominates, not a single-target landing.
    m, tables, kernel = _setup()
    ck = collision_ck_matrix(kernel, 5.0)
    b = 10
    g0 = jnp.zeros_like(m).at[b].set(1.0e-3)
    g1 = bott_coalescence(g0, ck, m, tables)
    moved = float(g0[b] - g1[b])
    assert moved > 0.0
    deposited = np.asarray(g1)[b + 1:]
    # rtol covers the oracle's gmin floor lift of touched-empty bins
    # (≤ a few × 1e-16 kg m⁻³ absolute).
    np.testing.assert_allclose(deposited.sum(), moved, rtol=1e-9)
    assert float(g1[b + 1]) > 0.99 * moved
    # Nothing leaks downward.
    np.testing.assert_array_equal(np.asarray(g1)[:b], 0.0)


def test_mass_conserved_number_decreases():
    m, tables, kernel = _setup()
    ck = collision_ck_matrix(kernel, 1.0)
    f0 = discretize_exponential(
        m, N0, 4.0 / 3.0 * np.pi * constants.rho_water * R0**3)
    g = g_from_f(f0, m)
    M0 = float(mass_density_from_g(g))
    N_prev = float(number_density_from_g(g, m))
    for _ in range(20):
        g = bott_coalescence(g, ck, m, tables)
        N_now = float(number_density_from_g(g, m))
        assert N_now <= N_prev * (1.0 + 1.0e-12)
        N_prev = N_now
    M1 = float(mass_density_from_g(g))
    assert abs(M1 - M0) / M0 < 1.0e-10
    assert np.all(np.asarray(g) >= 0.0)


def test_golovin_box_moment_laws():
    # Discretization context (probed by dt sweep + the refinement test
    # below): at the oracle's 33-bin doubling resolution the Bott scheme
    # under-predicts N(t) by ~7% (dt-converged; spectral broadening sends
    # mass to large bins early) and over-predicts M2 by ~22%. Both errors
    # shrink monotonically under log-grid refinement — the tolerances here
    # bracket the converged NKR=33 numbers, not noise.
    m, tables, kernel = _setup()
    dt = 0.5
    n_steps = 116                       # t = 58 s (SDM test timeline)
    ck = collision_ck_matrix(kernel, dt)
    mbar = 4.0 / 3.0 * np.pi * constants.rho_water * R0**3
    f0 = discretize_exponential(m, N0, mbar)
    g0 = g_from_f(f0, m)

    def step(g, _):
        return bott_coalescence(g, ck, m, tables), None

    g_T, _ = jax.lax.scan(jax.jit(step), g0, None, length=n_steps)

    L = float(mass_density_from_g(g0))
    b_mass = B_GOLOVIN / constants.rho_water
    t = n_steps * dt
    # Number decay (exact moment law for the additive kernel).
    ratio = float(number_density_from_g(g_T, m) / number_density_from_g(g0, m))
    assert ratio == pytest.approx(np.exp(-b_mass * L * t), rel=0.10)
    # Second-moment growth M2 = Σ f m² dm = COL Σ g·m: broadening makes the
    # bin scheme overshoot — require both the analytic bracket and the
    # deterministic overshoot signature.
    M2_0 = float(COL_sum_m2(g0, m))
    M2_T = float(COL_sum_m2(g_T, m))
    analytic_M2 = np.exp(2.0 * b_mass * L * t)
    assert M2_T / M2_0 == pytest.approx(analytic_M2, rel=0.30)
    assert M2_T / M2_0 > analytic_M2
    # Mass conserved through the whole run.
    assert float(mass_density_from_g(g_T)) == pytest.approx(L, rel=1.0e-9)


def COL_sum_m2(g, m):
    from legoesm.atmosphere.physics.microphysics.fast_sbm import COL
    return COL * jnp.sum(g * m, axis=-1)


def test_golovin_refinement_convergence():
    # The solver is grid-general (Courant tables use the actual log
    # spacing): halving d(ln m) must cut the number-decay error — proves
    # the bias above is resolution, not a port bug.
    import math
    b_mass = B_GOLOVIN / constants.rho_water
    mbar = 4.0 / 3.0 * np.pi * constants.rho_water * R0**3
    m1 = 4.0 / 3.0 * math.pi * constants.rho_water * (2.0e-6) ** 3
    t, dt = 58.0, 0.5
    errors = []
    for s in (1, 2):
        n_bins = 33 * s
        masses = jnp.asarray(m1 * (2.0 ** (np.arange(n_bins) / s)))
        tables = precompute_collision_tables(masses)
        r = radius_from_mass(masses)
        ck = golovin_kernel(r[:, None], r[None, :], B_GOLOVIN) \
            * dt * (math.log(2.0) / (3.0 * s))
        # Exponential init on this grid (edges 2^(±1/(2s))).
        m_lo = masses * 2.0 ** (-0.5 / s)
        m_hi = masses * 2.0 ** (+0.5 / s)
        dn = N0 * (jnp.exp(-m_lo / mbar) - jnp.exp(-m_hi / mbar))
        g0 = 3.0 * masses**2 * dn / ((math.log(2.0) / s) * masses)
        dlnr = math.log(2.0) / (3.0 * s)
        L = float(dlnr * jnp.sum(g0))
        step = jax.jit(lambda g, _, ck=ck, masses=masses, tables=tables:
                       (bott_coalescence(g, ck, masses, tables), None))
        g_T, _ = jax.lax.scan(step, g0, None, length=int(t / dt))
        ratio = float(jnp.sum(g_T / masses) / jnp.sum(g0 / masses))
        errors.append(abs(ratio - np.exp(-b_mass * L * t)))
    assert errors[1] < 0.75 * errors[0]


def test_coalescence_differentiable():
    m, tables, kernel = _setup()
    mbar = 4.0 / 3.0 * np.pi * constants.rho_water * R0**3

    def n_after(b):
        ck = collision_ck_matrix(
            golovin_kernel(radius_from_mass(m)[:, None],
                           radius_from_mass(m)[None, :], 1.0) * b, 1.0)
        g = g_from_f(discretize_exponential(m, N0, mbar), m)
        for _ in range(3):
            g = bott_coalescence(g, ck, m, tables)
        return number_density_from_g(g, m)

    gb = jax.grad(n_after)(jnp.asarray(B_GOLOVIN))
    assert np.isfinite(float(gb))
    # Stronger kernel → faster number decay.
    assert float(gb) < 0.0


def test_number_density_consistent_f_and_g():
    m, _, _ = _setup()
    f = discretize_exponential(m, N0, 1.0e-11)
    g = g_from_f(f, m)
    np.testing.assert_allclose(float(number_density_from_g(g, m)),
                               float(number_density(f, m)), rtol=1e-13)
