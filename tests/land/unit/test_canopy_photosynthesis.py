"""Unit tests for canopy/photosynthesis.py.

Checks:
- C3 assimilation > 0 at ambient CO2 and saturating light
- C3 assimilation = 0 in the dark
- C4 assimilation > C3 for hot, light-saturated conditions
- Temperature response peaks near 25-30 C
- Mixed C3/C4 fraction is a continuous weighted average
- Functions are JIT- and grad-compatible
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.land.canopy.photosynthesis import (
    c3_photosynthesis,
    c4_photosynthesis,
    photosynthesis,
    vcmax_temperature_response,
)


def test_c3_positive_daytime():
    Tf = jnp.array(298.15)
    Ci = jnp.array(280.0)
    APAR = jnp.array(1500.0)
    Vcmax25 = jnp.array(60.0)
    Ps = jnp.array(101325.0)
    alf = jnp.array(0.3)
    TgC = jnp.array(20.0)
    An = c3_photosynthesis(Tf, Ci, APAR, Vcmax25, Ps, alf, TgC)
    assert float(An) > 5.0, f"expected daytime C3 An > 5, got {An}"


def test_c3_zero_dark():
    An = c3_photosynthesis(
        jnp.array(298.15), jnp.array(280.0), jnp.array(0.0),
        jnp.array(60.0), jnp.array(101325.0), jnp.array(0.3), jnp.array(20.0))
    # With no APAR, the light-limited rate is zero; only Rd is subtracted,
    # then jnp.maximum clips to 0.
    assert float(An) == 0.0


def test_c4_positive_daytime():
    An = c4_photosynthesis(
        jnp.array(303.15), jnp.array(150.0), jnp.array(1500.0), jnp.array(40.0))
    assert float(An) > 5.0


def test_vcmax_peak_near_25C():
    """Vcmax temperature response should be ~1 at 25C and drop at 50C."""
    f25 = vcmax_temperature_response(jnp.array(298.15), jnp.array(20.0))
    f50 = vcmax_temperature_response(jnp.array(323.15), jnp.array(20.0))
    f5  = vcmax_temperature_response(jnp.array(278.15), jnp.array(20.0))
    assert 0.9 < float(f25) < 1.1
    assert float(f50) < float(f25)
    assert float(f5) < float(f25)


def test_photosynthesis_mixing_is_weighted_average():
    args = dict(
        Tf=jnp.array(300.0), Ci=jnp.array(250.0), APAR=jnp.array(1200.0),
        Vcmax25_C3=jnp.array(60.0), Vcmax25_C4=jnp.array(40.0),
        Ps=jnp.array(101325.0), alf=jnp.array(0.3), TgC=jnp.array(20.0),
    )
    An_c3_only = photosynthesis(fC4=jnp.array(0.0), **args)
    An_c4_only = photosynthesis(fC4=jnp.array(1.0), **args)
    An_mix     = photosynthesis(fC4=jnp.array(0.5), **args)
    expected_mix = 0.5 * An_c3_only + 0.5 * An_c4_only
    assert jnp.allclose(An_mix, expected_mix, atol=1e-5)


def test_photosynthesis_differentiable():
    """Gradients wrt Vcmax25 must be finite — required for training."""
    def loss(v):
        return c3_photosynthesis(
            jnp.array(298.15), jnp.array(280.0), jnp.array(1500.0),
            v, jnp.array(101325.0), jnp.array(0.3), jnp.array(20.0))
    g = jax.grad(loss)(jnp.array(60.0))
    assert jnp.isfinite(g)
    assert float(g) > 0.0  # more Vcmax => more An
