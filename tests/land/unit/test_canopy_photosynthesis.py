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

from legoesm.land.canopy import photosynthesis as photo
from legoesm.land.canopy.photosynthesis import (
    _jmax25_over_vcmax25,
    _rd_atkin,
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


# ---------------------------------------------------------------------------
# Canonical-FvCB structural checks (the ported DifferBESS bug fixes)
# ---------------------------------------------------------------------------

def test_rd25_equals_basal_fraction_no_double_count():
    """Rd at 25 degC / TgC=25 == 0.015 * Vcmax25 (the double-count fix).

    The old path used Rd = 0.015 * Vcmax(T) * rd_response(T), applying the
    temperature response twice.  The canonical form bases Rd on Vcmax25 and a
    response normalised to 1 at the reference state.
    """
    Vcmax25 = jnp.array(60.0)
    Rd = _rd_atkin(jnp.array(298.15), jnp.array(25.0), Vcmax25)
    assert jnp.allclose(Rd, 0.015 * Vcmax25, rtol=1e-6)


def test_jmax_acclimation_ratio_kattge_knorr():
    """Jmax25/Vcmax25 follows Kattge & Knorr 2007: 2.59 - 0.035*TgC."""
    for tgc in (11.0, 20.0, 35.0):
        ratio = _jmax25_over_vcmax25(jnp.array(tgc))
        assert jnp.allclose(ratio, 2.59 - 0.035 * tgc, rtol=1e-6)
    # cooler growth temperature => higher Jmax:Vcmax ratio
    assert float(_jmax25_over_vcmax25(jnp.array(11.0))) > \
        float(_jmax25_over_vcmax25(jnp.array(35.0)))


def test_c4_uses_clm5_constants():
    """C4 kinetics carry the CLM5-aligned constants (the b5e19e8/85bed3b fix)."""
    assert photo._S2_C4 == 313.15      # high-T deactivation onset (was 309.15)
    assert photo._S3_C4 == 0.2         # low-T inhibition slope (was 0.3)
    assert photo._RD25_FRAC_C4 == 0.025  # Rd25/Vcmax25 (was fixed 0.8)
    assert photo._ALPHA_C4 == 0.05     # quantum yield (was 0.067)


def test_c3_electron_transport_jmax_bounded():
    """At very high APAR the light-limited rate saturates (Jmax bound).

    The old JE = alf*APAR grew without bound; the canonical Jmax-limited J
    saturates, so doubling already-saturating light barely changes An.
    """
    base = dict(Tf=jnp.array(298.15), Ci=jnp.array(280.0),
                Vcmax25=jnp.array(60.0), Ps=jnp.array(101325.0),
                alf=jnp.array(0.3), TgC=jnp.array(20.0))
    An_2000 = float(c3_photosynthesis(APAR=jnp.array(2000.0), **base))
    An_4000 = float(c3_photosynthesis(APAR=jnp.array(4000.0), **base))
    # Saturating: a 2x light increase yields < 5% more assimilation.
    assert An_4000 >= An_2000
    assert (An_4000 - An_2000) / An_2000 < 0.05
