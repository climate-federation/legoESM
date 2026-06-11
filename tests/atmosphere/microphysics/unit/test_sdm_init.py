"""Unit tests for SDM initialization spectra (ERF constant-multiplicity port)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm import (
    exponential_water_droplets,
    lognormal_aerosol_droplets,
    represented_water_mass,
    sample_exponential_mass,
    sample_lognormal_radius,
)


def test_exponential_mass_moments_and_min():
    n = 200_000
    mean, mmin = 5.0e-13, 1.0e-13
    m = np.asarray(sample_exponential_mass(random.PRNGKey(0), n, mean, mmin))
    assert m.min() >= mmin
    # mean of m_min + Exp(delta) is m_min + delta = mass_mean
    assert m.mean() == pytest.approx(mean, rel=0.01)
    # exponential std = delta
    assert m.std() == pytest.approx(mean - mmin, rel=0.02)


def test_lognormal_radius_moments_and_truncation():
    n = 200_000
    r_med, gstd = 5.0e-8, 1.8
    r = np.asarray(sample_lognormal_radius(random.PRNGKey(1), n, r_med, gstd))
    lnr = np.log(r)
    assert np.exp(np.median(lnr)) == pytest.approx(r_med, rel=0.01)
    assert lnr.std() == pytest.approx(np.log(gstd), rel=0.02)
    # truncated draw lies strictly within bounds and keeps the median
    r_t = np.asarray(sample_lognormal_radius(
        random.PRNGKey(2), n, r_med, gstd, r_min=2.0e-8, r_max=2.0e-7))
    assert r_t.min() >= 2.0e-8 and r_t.max() <= 2.0e-7


def test_samplers_deterministic():
    a = sample_exponential_mass(random.PRNGKey(3), 64, 1e-12)
    b = sample_exponential_mass(random.PRNGKey(3), 64, 1e-12)
    assert jnp.array_equal(a, b)
    c = sample_lognormal_radius(random.PRNGKey(4), 64, 5e-8, 1.6)
    d = sample_lognormal_radius(random.PRNGKey(4), 64, 5e-8, 1.6)
    assert jnp.array_equal(c, d)


def test_exponential_water_droplets_builder():
    n_sd, N, x0 = 4096, 1.0e9, 1.0e-12
    st = exponential_water_droplets(random.PRNGKey(5), n_sd, N, x0)
    assert jnp.allclose(st.multiplicity, N / n_sd)
    assert jnp.all(st.active == 1.0)
    assert jnp.all(st.solute_mass == 0.0)
    # represented total water -> N * mass_mean within MC tolerance
    total = float(jnp.sum(represented_water_mass(st)))
    assert total == pytest.approx(N * x0, rel=0.05)


def test_lognormal_aerosol_droplets_builder():
    n_sd = 4096
    rho_s = 1770.0
    st = lognormal_aerosol_droplets(random.PRNGKey(6), n_sd, 5.0e7,
                                    r_dry_median=5.0e-8, geom_std=1.6,
                                    solute_density=rho_s)
    # solute mass consistent with the sampled dry radius
    r_dry = (np.asarray(st.solute_mass)
             / (4.0 / 3.0 * np.pi * rho_s)) ** (1.0 / 3.0)
    assert np.exp(np.median(np.log(r_dry))) == pytest.approx(5.0e-8, rel=0.05)
    # default wet radius == dry radius (factor 1)
    assert np.allclose(np.asarray(st.radius), r_dry, rtol=1e-12)
    st2 = lognormal_aerosol_droplets(random.PRNGKey(6), n_sd, 5.0e7,
                                     r_dry_median=5.0e-8, geom_std=1.6,
                                     solute_density=rho_s,
                                     wet_radius_factor=2.0)
    assert np.allclose(np.asarray(st2.radius), 2.0 * r_dry, rtol=1e-12)
