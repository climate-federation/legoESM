"""Habit-routed Bigg freezing (oracle FREEZ KRFREEZ split)."""

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
    freeze_step,
    freeze_step_routed,
)

jax.config.update("jax_enable_x64", True)


def _spectrum():
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        discretize_lognormal)
    m = mass_doubling_grid()
    f = discretize_lognormal(m, 1.0e8, 5.0e-5, 1.8)   # broad, spans KRFREEZE
    return m, f


def test_routing_sum_equals_single_category():
    # The two categories must sum EXACTLY to the single-category frozen ice;
    # depleted liquid and fusion heat identical.
    m, f = _spectrum()
    rho = jnp.asarray(1.0)
    T = jnp.asarray(constants.T_freeze - 25.0)
    single = freeze_step(f, m, T, rho, 2.0)
    routed = freeze_step_routed(f, m, T, rho, 2.0)
    np.testing.assert_allclose(
        np.asarray(routed.f_crystals + routed.f_hail),
        np.asarray(single.f_ice), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(routed.f_liquid),
                               np.asarray(single.f_liquid), rtol=1e-14)
    assert float(routed.dT) == pytest.approx(float(single.dT), rel=1e-14)


def test_routing_split_at_krfreeze():
    # Small bins (< KRFREEZE) → crystals only; large bins → hail only.
    m, f = _spectrum()
    cfg = FastSBMConfig()
    out = freeze_step_routed(f, m, jnp.asarray(constants.T_freeze - 25.0),
                             jnp.asarray(1.0), 2.0, cfg)
    crystals = np.asarray(out.f_crystals)
    hail = np.asarray(out.f_hail)
    assert np.all(crystals[cfg.krfreeze:] == 0.0)     # no crystals in big bins
    assert np.all(hail[:cfg.krfreeze] == 0.0)         # no hail in small bins
    # Both categories non-empty for a spectrum spanning the threshold.
    assert float(mass_density(out.f_crystals, m)) > 0.0
    assert float(mass_density(out.f_hail, m)) > 0.0


def test_routing_conserves_mass():
    m, f = _spectrum()
    rho = jnp.asarray(1.0)
    out = freeze_step_routed(f, m, jnp.asarray(constants.T_freeze - 25.0),
                             rho, 2.0)
    total = (mass_density(out.f_liquid, m) + mass_density(out.f_crystals, m)
             + mass_density(out.f_hail, m))
    np.testing.assert_allclose(float(total), float(mass_density(f, m)),
                               rtol=1e-12)


def test_no_routing_above_freezing():
    m, f = _spectrum()
    out = freeze_step_routed(f, m, jnp.asarray(constants.T_freeze + 1.0),
                             jnp.asarray(1.0), 2.0)
    np.testing.assert_array_equal(np.asarray(out.f_crystals), 0.0)
    np.testing.assert_array_equal(np.asarray(out.f_hail), 0.0)
    assert float(out.dT) == 0.0


def test_routed_differentiable():
    m, f = _spectrum()
    rho = jnp.asarray(1.0)

    def hail_mass(T):
        return mass_density(
            freeze_step_routed(f, m, T, rho, 2.0).f_hail, m)

    g = jax.grad(hail_mass)(jnp.asarray(constants.T_freeze - 20.0))
    assert np.isfinite(float(g))
