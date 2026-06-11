"""Fast-SBM CCN activation (oracle JERNUCL01_KS / WATER_NUCLEATION).

Köhler critical radius scaling + the activated-number response.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm import mass_doubling_grid
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.nucleation import (
    activate_ccn,
    critical_dry_radius,
    hygroscopicity,
    kelvin_coefficient,
)

jax.config.update("jax_enable_x64", True)

T0 = 283.0


def test_kelvin_coefficient_value():
    # A = 2σ/(ρ_w R_v T): ~1.2 nm·... at 283 K; oracle AKOE=3.3e-5 cm·K /T
    # → A_oracle = 3.3e-7 m·K / T. Sub-percent agreement expected.
    A = float(kelvin_coefficient(jnp.asarray(T0)))
    A_oracle = 3.3e-7 / T0          # AKOE in SI (cm→m): 3.3e-5 cm = 3.3e-7 m
    assert A == pytest.approx(A_oracle, rel=0.05)
    assert A > 0.0


def test_critical_radius_scales_as_s_minus_two_thirds():
    cfg = FastSBMConfig()
    s = np.array([0.001, 0.004, 0.016])      # quadrupling steps
    r = np.array([float(critical_dry_radius(jnp.asarray(1.0 + si),
                                            jnp.asarray(T0), cfg))
                  for si in s])
    # r_crit ∝ s^(-2/3): doubling-of-4× s → r ratio (4)^(-2/3)=0.397.
    np.testing.assert_allclose(r[1] / r[0], 4.0 ** (-2.0 / 3.0), rtol=1e-6)
    np.testing.assert_allclose(r[2] / r[1], 4.0 ** (-2.0 / 3.0), rtol=1e-6)


def test_matches_oracle_rcriti_formula():
    cfg = FastSBMConfig()
    T, S = jnp.asarray(T0), jnp.asarray(1.005)
    A = kelvin_coefficient(T)
    B = hygroscopicity(cfg)
    s = 0.005
    r_oracle = float((A / 3.0) * (4.0 / (B * s * s)) ** (1.0 / 3.0))
    assert float(critical_dry_radius(S, T, cfg)) == pytest.approx(
        r_oracle, rel=1e-12)


def test_no_activation_at_or_below_saturation():
    cfg = FastSBMConfig()
    m = mass_doubling_grid()
    for S in (0.9, 1.0):
        out = activate_ccn(jnp.asarray(S), jnp.asarray(T0),
                           jnp.asarray(1.0e8), m, cfg)
        assert float(out.n_activated) == 0.0
        np.testing.assert_array_equal(np.asarray(out.df), 0.0)


def test_activation_monotone_in_supersaturation():
    cfg = FastSBMConfig()
    m = mass_doubling_grid()
    n_avail = jnp.asarray(1.0e8)
    ns = [float(activate_ccn(jnp.asarray(1.0 + s), jnp.asarray(T0),
                             n_avail, m, cfg).n_activated)
          for s in (0.001, 0.003, 0.01, 0.03)]
    assert all(np.diff(ns) > 0.0)              # more s → more activation
    assert ns[-1] <= float(n_avail)            # bounded by the reservoir
    # Activated number seeded into the smallest bin only.
    out = activate_ccn(jnp.asarray(1.02), jnp.asarray(T0), n_avail, m, cfg)
    assert float(out.df[0]) > 0.0
    np.testing.assert_array_equal(np.asarray(out.df[1:]), 0.0)
    # Bin-0 seed carries exactly the activated number.
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        bin_mass_widths)
    np.testing.assert_allclose(
        float(out.df[0] * bin_mass_widths(m)[0]),
        float(out.n_activated), rtol=1e-12)


def test_activation_bounded_by_reservoir():
    cfg = FastSBMConfig()
    m = mass_doubling_grid()
    # Huge supersaturation → essentially the whole reservoir activates.
    out = activate_ccn(jnp.asarray(1.5), jnp.asarray(T0),
                       jnp.asarray(2.0e8), m, cfg)
    assert float(out.n_activated) <= 2.0e8 + 1.0
    assert float(out.n_activated) > 0.5 * 2.0e8


def test_differentiable_in_S_and_T():
    cfg = FastSBMConfig()
    m = mass_doubling_grid()

    def n_act(x):
        S, T = x
        return activate_ccn(S, T, jnp.asarray(1.0e8), m, cfg).n_activated

    g = jax.grad(n_act)(jnp.array([1.01, T0]))
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(g[0]) > 0.0     # higher S activates more
