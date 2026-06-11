"""Fast-SBM immersion freezing (oracle FREEZ, Bigg 1953)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm import (
    mass_density,
    mass_doubling_grid,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.freezing import (
    bigg_freezing_rate,
    freeze_step,
)

jax.config.update("jax_enable_x64", True)


def _spectrum():
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        discretize_lognormal)
    m = mass_doubling_grid()
    # Broad spectrum spanning cloud + drizzle so big bins are populated.
    f = discretize_lognormal(m, 1.0e8, 3.0e-5, 1.6)
    return m, f


def test_no_freezing_at_or_above_freezing():
    m, f = _spectrum()
    for T in (constants.T_freeze, constants.T_freeze + 5.0):
        out = freeze_step(f, m, jnp.asarray(T), jnp.asarray(1.0), 2.0)
        np.testing.assert_array_equal(np.asarray(out.f_ice), 0.0)
        np.testing.assert_array_equal(np.asarray(out.f_liquid), np.asarray(f))
        assert float(out.dT) == 0.0


def test_rate_matches_oracle_formula():
    cfg = FastSBMConfig()
    m = mass_doubling_grid()
    T = jnp.asarray(constants.T_freeze - 15.0)
    P = np.asarray(bigg_freezing_rate(m, T, cfg))
    # Oracle PF = m_g·AFREEZMY·exp(−BF·ΔT), BF constant (B_max=B0).
    dT = -15.0
    m_g = np.asarray(m) * 1.0e3
    P_oracle = m_g * cfg.bigg_a * np.exp(-cfg.bigg_b0 * dT)
    np.testing.assert_allclose(P, P_oracle, rtol=1e-12)


def test_freezing_conserves_mass_and_heats():
    m, f = _spectrum()
    rho = jnp.asarray(1.0)
    out = freeze_step(f, m, jnp.asarray(constants.T_freeze - 25.0), rho, 2.0)
    # Mass moved liquid→ice, total preserved per bin (same masses).
    np.testing.assert_allclose(
        np.asarray(mass_density(out.f_liquid, m) + mass_density(out.f_ice, m)),
        float(mass_density(f, m)), rtol=1e-12)
    assert float(mass_density(out.f_ice, m)) > 0.0
    # Fusion warming = (L_f/c_pd)·dq_ice.
    dq_ice = float(mass_density(out.f_ice, m)) / float(rho)
    assert float(out.dT) == pytest.approx(
        (constants.L_f / constants.c_pd) * dq_ice, rel=1e-12)
    assert np.all(np.asarray(out.f_liquid) >= 0.0)


def test_deeper_supercooling_freezes_more():
    m, f = _spectrum()
    rho = jnp.asarray(1.0)
    ice = [float(mass_density(
        freeze_step(f, m, jnp.asarray(constants.T_freeze - d), rho, 2.0).f_ice,
        m)) for d in (5.0, 15.0, 30.0)]
    assert ice[0] < ice[1] < ice[2]


def test_larger_drops_freeze_preferentially():
    # Bigg rate ∝ volume → big bins freeze a larger fraction than small.
    m = mass_doubling_grid()
    P = np.asarray(bigg_freezing_rate(
        m, jnp.asarray(constants.T_freeze - 15.0)))
    assert np.all(np.diff(P) > 0.0)        # monotone increasing in mass


def test_freezing_differentiable_in_T():
    m, f = _spectrum()
    rho = jnp.asarray(1.0)

    def ice_mass(T):
        return mass_density(freeze_step(f, m, T, rho, 2.0).f_ice, m)

    g = jax.grad(ice_mass)(jnp.asarray(constants.T_freeze - 15.0))
    assert np.isfinite(float(g))
    # Colder (lower T) → more ice, so d(ice)/dT < 0.
    assert float(g) < 0.0
