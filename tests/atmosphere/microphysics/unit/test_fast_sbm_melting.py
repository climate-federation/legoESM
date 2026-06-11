"""Fast-SBM ice melting (oracle J_W_MELT, Jiwen Fan)."""

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
from legoesm.atmosphere.physics.microphysics.fast_sbm.melting import (
    melt_fraction,
    melt_step,
)

jax.config.update("jax_enable_x64", True)


def _ice():
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        discretize_lognormal)
    m = mass_doubling_grid()
    f = discretize_lognormal(m, 1.0e6, 1.0e-4, 1.6)   # broad ice spectrum
    return m, f


def test_no_melting_below_freezing():
    m, f = _ice()
    out = melt_step(f, m, jnp.asarray(constants.T_freeze - 5.0),
                    jnp.asarray(1.0), 2.0)
    np.testing.assert_array_equal(np.asarray(out.f_liquid), 0.0)
    np.testing.assert_array_equal(np.asarray(out.f_ice), np.asarray(f))
    assert float(out.dT) == 0.0


def test_melt_fraction_ladder_matches_oracle():
    cfg = FastSBMConfig()
    dt = 30.0
    frac = np.asarray(melt_fraction(33, dt, cfg))
    # Small bins melt fully.
    assert np.all(frac[:cfg.melt_full_bin + 1] == 1.0)
    # Mid bins at min(rate_mid·dt, 1).
    mid = slice(cfg.melt_full_bin + 1, cfg.melt_mid_bin + 1)
    np.testing.assert_allclose(frac[mid], min(cfg.melt_rate_mid * dt, 1.0))
    # Large bins at min(rate_high·dt, 1).
    np.testing.assert_allclose(frac[cfg.melt_mid_bin + 1:],
                               min(cfg.melt_rate_high * dt, 1.0))


def test_melting_conserves_mass_and_cools():
    m, f = _ice()
    rho = jnp.asarray(1.0)
    out = melt_step(f, m, jnp.asarray(constants.T_freeze + 3.0), rho, 5.0)
    np.testing.assert_allclose(
        np.asarray(mass_density(out.f_ice, m)
                   + mass_density(out.f_liquid, m)),
        float(mass_density(f, m)), rtol=1e-12)
    assert float(mass_density(out.f_liquid, m)) > 0.0
    dq_melt = float(mass_density(out.f_liquid, m)) / float(rho)
    assert float(out.dT) == pytest.approx(
        -(constants.L_f / constants.c_pd) * dq_melt, rel=1e-12)
    assert float(out.dT) < 0.0                       # cooling
    assert np.all(np.asarray(out.f_ice) >= 0.0)


def test_small_bins_fully_melt_one_step():
    m, f = _ice()
    cfg = FastSBMConfig()
    out = melt_step(f, m, jnp.asarray(constants.T_freeze + 1.0),
                    jnp.asarray(1.0), 1.0, cfg)
    # Below the full-melt threshold the ice bins are emptied.
    np.testing.assert_allclose(
        np.asarray(out.f_ice[:cfg.melt_full_bin + 1]), 0.0, atol=1e-30)
    # All their mass appears as liquid.
    np.testing.assert_allclose(
        np.asarray(out.f_liquid[:cfg.melt_full_bin + 1]),
        np.asarray(f[:cfg.melt_full_bin + 1]), rtol=1e-12)


def test_freeze_melt_round_trip_conserves():
    # Freeze drops then melt the ice back: total condensed mass preserved,
    # heat releases then re-absorbs (net ~0 if fully reversed).
    from legoesm.atmosphere.physics.microphysics.fast_sbm.freezing import (
        freeze_step)
    m, _ = _ice()
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        discretize_lognormal)
    f_liq = discretize_lognormal(m, 1.0e8, 3.0e-5, 1.6)
    rho = jnp.asarray(1.0)
    frz = freeze_step(f_liq, m, jnp.asarray(constants.T_freeze - 30.0),
                      rho, 2.0)
    melt = melt_step(frz.f_ice, m, jnp.asarray(constants.T_freeze + 5.0),
                     rho, 200.0)          # long dt → melt all
    total_before = float(mass_density(f_liq, m))
    total_after = float(mass_density(frz.f_liquid, m)
                        + mass_density(melt.f_liquid, m)
                        + mass_density(melt.f_ice, m))
    np.testing.assert_allclose(total_after, total_before, rtol=1e-12)


def test_melting_differentiable_in_T():
    m, f = _ice()
    rho = jnp.asarray(1.0)

    def liquid_mass(T):
        return mass_density(melt_step(f, m, T, rho, 5.0).f_liquid, m)

    g = jax.grad(liquid_mass)(jnp.asarray(constants.T_freeze + 3.0))
    assert np.isfinite(float(g))
