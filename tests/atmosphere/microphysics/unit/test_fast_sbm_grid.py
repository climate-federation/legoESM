"""Fast-SBM bin grid: oracle landmarks, moment quadrature, differentiability.

Oracle: WRF ``phys/module_mp_fast_sbm.F`` (FSBM-2). Checks reproduce the
oracle's documented grid landmarks (33 mass-doubling bins, COL = ln2/3,
bin 15 ~ 50 um cloud/rain boundary, ~3.25 mm spectrum top) and the spectral
moment convention (QC/QNC diagnostics with dm = 3*COL*m).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm import (
    COL,
    KRDROP,
    NKR_LIQUID,
    bin_mass_widths,
    discretize_lognormal,
    mass_density,
    mass_doubling_grid,
    number_density,
    radius_from_mass,
)

jax.config.update("jax_enable_x64", True)


def test_col_matches_oracle_constant():
    # Oracle: DOUBLE PRECISION, PARAMETER :: COL = 0.23105 (= ln2/3 rounded).
    assert abs(COL - 0.23105) < 5.0e-6
    assert np.isclose(COL, np.log(2.0) / 3.0, rtol=0, atol=0)


def test_mass_doubling_exact():
    m = mass_doubling_grid()
    assert m.shape == (NKR_LIQUID,)
    # Bitwise doubling (2**k is exact in fp).
    np.testing.assert_array_equal(np.asarray(m[1:]), 2.0 * np.asarray(m[:-1]))
    assert np.all(np.asarray(m) > 0.0)


def test_oracle_grid_landmarks():
    m = mass_doubling_grid()
    r = radius_from_mass(m)
    # Smallest drop: 2 um by construction.
    assert np.isclose(float(r[0]), 2.0e-6, rtol=1e-12)
    # Oracle comment: "KRDROP=Bin 15 --> 50um" (2um * 2^(14/3) = 50.8 um).
    assert np.isclose(float(r[KRDROP - 1]), 50.0e-6, rtol=0.02)
    # FSBM documented drop-spectrum top ~ 3.25 mm.
    assert np.isclose(float(r[-1]), 3.25e-3, rtol=0.01)
    # Radius doubles every 3 bins (mass doubles every bin).
    np.testing.assert_allclose(np.asarray(r[3:]), 2.0 * np.asarray(r[:-3]),
                               rtol=1e-12)


def test_radius_mass_roundtrip():
    m = mass_doubling_grid()
    r = radius_from_mass(m)
    m_back = 4.0 / 3.0 * np.pi * constants.rho_water * np.asarray(r) ** 3
    np.testing.assert_allclose(m_back, np.asarray(m), rtol=1e-12)


def test_moment_quadrature_matches_oracle_formula():
    # Oracle FAST_SBM: QC = (1/ρ) Σ COL·f·x²·3 ; QNC = (1/ρ) Σ COL·f·x·3.
    m = mass_doubling_grid()
    f = jnp.ones_like(m) * 1.0e12  # arbitrary uniform spectrum [m^-3 kg^-1]
    n = float(number_density(f, m))
    q = float(mass_density(f, m))
    n_oracle = float(jnp.sum(3.0 * COL * f * m))
    q_oracle = float(jnp.sum(3.0 * COL * f * m * m))
    assert np.isclose(n, n_oracle, rtol=1e-14)
    assert np.isclose(q, q_oracle, rtol=1e-14)
    # dm_k = ln2 * m_k exactly.
    np.testing.assert_allclose(np.asarray(bin_mass_widths(m)),
                               np.log(2.0) * np.asarray(m), rtol=0, atol=0)


def test_moments_batched_axes():
    m = mass_doubling_grid()
    f = jnp.broadcast_to(jnp.ones_like(m), (4, 7, NKR_LIQUID)) * 1.0e10
    assert number_density(f, m).shape == (4, 7)
    assert mass_density(f, m).shape == (4, 7)


def test_discretize_lognormal_recovers_moments():
    # Maritime cloud-droplet-like mode well inside the grid.
    m = mass_doubling_grid()
    n_total = 1.0e8          # [m^-3] = 100 cm^-3
    r_med = 10.0e-6          # 10 um median
    geom_std = 1.4
    f = discretize_lognormal(m, n_total, r_med, geom_std)
    assert np.all(np.asarray(f) >= 0.0)
    # Number: exact up to grid truncation (mode is far from both ends).
    assert np.isclose(float(number_density(f, m)), n_total, rtol=1e-6)
    # Mass: midpoint-rule approximation of the analytic 3rd-radius moment
    # m̄ = (4/3)πρ r_med³ exp(4.5 ln²σ_g). The doubling grid carries a
    # systematic midpoint bias of ~ln²2/24 = +2.0% (same quadrature the
    # oracle uses), so the tolerance brackets it.
    q_analytic = n_total * (4.0 / 3.0 * np.pi * constants.rho_water
                            * r_med ** 3 * np.exp(4.5 * np.log(geom_std) ** 2))
    q = float(mass_density(f, m))
    assert np.isclose(q, q_analytic, rtol=0.03)
    # The bias is the predicted midpoint term, not noise: +2.0% ± 0.5%.
    assert 1.005 < q / q_analytic < 1.035


def test_discretize_lognormal_truncation_consistent():
    # A mode centred below the grid loses number — but never goes negative
    # and never exceeds n_total.
    m = mass_doubling_grid()
    f = discretize_lognormal(m, 1.0e8, 0.5e-6, 1.5)
    n = float(number_density(f, m))
    assert 0.0 <= n < 1.0e8


def test_grid_differentiable():
    m = mass_doubling_grid()

    def q_of_params(params):
        r_med, n_tot = params
        f = discretize_lognormal(m, n_tot, r_med, 1.4)
        return mass_density(f, m)

    g = jax.grad(q_of_params)(jnp.array([10.0e-6, 1.0e8]))
    assert np.all(np.isfinite(np.asarray(g)))
    # Mass grows with both median radius and number.
    assert float(g[0]) > 0.0
    assert float(g[1]) > 0.0
    # d(mass)/df_k = m_k · dm_k exactly (linear moment).
    f0 = discretize_lognormal(m, 1.0e8, 10.0e-6, 1.4)
    gf = jax.grad(lambda f: mass_density(f, m))(f0)
    np.testing.assert_allclose(np.asarray(gf),
                               np.asarray(m * bin_mass_widths(m)), rtol=1e-12)


def test_bin_mixing_ratio_roundtrip_and_qc_sum():
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        bin_mixing_ratios_from_f, f_from_bin_mixing_ratios)
    m = mass_doubling_grid()
    f = discretize_lognormal(m, 1.0e8, 10.0e-6, 1.4)
    f = jnp.broadcast_to(f, (3, NKR_LIQUID))
    rho_air = jnp.array([1.2, 1.0, 0.8])
    q_bins = bin_mixing_ratios_from_f(f, m, rho_air)
    # Oracle QC diagnostic = Σ_k q_k = mass_density/ρ_air.
    np.testing.assert_allclose(np.asarray(jnp.sum(q_bins, axis=-1)),
                               np.asarray(mass_density(f, m)) / np.asarray(rho_air),
                               rtol=1e-14)
    f_back = f_from_bin_mixing_ratios(q_bins, m, rho_air)
    np.testing.assert_allclose(np.asarray(f_back), np.asarray(f), rtol=1e-13)


def test_grid_jit_and_float32():
    m32 = mass_doubling_grid(dtype=jnp.float32)
    assert m32.dtype == jnp.float32
    fn = jax.jit(lambda f, m: (number_density(f, m), mass_density(f, m)))
    n, q = fn(jnp.ones_like(m32) * 1.0e10, m32)
    assert np.isfinite(float(n)) and np.isfinite(float(q))


def test_invalid_n_bins_raises():
    with pytest.raises(ValueError, match="n_bins"):
        mass_doubling_grid(0)
