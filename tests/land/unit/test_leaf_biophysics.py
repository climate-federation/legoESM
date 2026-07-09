"""Direct tests for the shared leaf-biochemistry helpers.

Covers the single-source Arrhenius / peaked-Arrhenius temperature responses and
the Bernacchi (2001) kinetic constants that both land photosynthesis paths
(``land/stomata.py`` big-leaf and ``canopy/photosynthesis.py`` two-leaf) now
import instead of each carrying their own copy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land import leaf_biophysics as lb


def test_arrhenius_unity_at_reference():
    # Normalised: exactly 1 at 25 degC regardless of Ha / gas constant.
    for Ha in (37830.0, 65330.0, 79430.0):
        assert float(lb.arrhenius_factor(lb.T_REF_K, Ha)) == pytest.approx(1.0, abs=1e-12)


def test_arrhenius_monotonic_and_matches_formula():
    T = jnp.array([280.0, 290.0, lb.T_REF_K, 305.0, 315.0])
    Ha = 65330.0
    f = lb.arrhenius_factor(T, Ha)
    # Strictly increasing in T for Ha > 0.
    assert np.all(np.diff(np.asarray(f)) > 0.0)
    # Uses the canonical CODATA gas constant, not a truncated 8.314.  rtol is
    # float32-safe (this module runs under both x64 and default precision) yet
    # far tighter than the ~1.5e-4 relative signal a truncated-R regression
    # (8.314) would produce, so it still pins the CODATA value.
    expected = np.exp(Ha * (np.asarray(T) - lb.T_REF_K)
                      / (lb.T_REF_K * constants.R_universal * np.asarray(T)))
    np.testing.assert_allclose(np.asarray(f), expected, rtol=5e-6)


def test_peaked_arrhenius_unity_at_reference_and_peaks():
    Ha, Hd, S = 72000.0, 200000.0, 649.12
    assert float(lb.peaked_arrhenius_factor(lb.T_REF_K, Ha, Hd, S)) == pytest.approx(
        1.0, abs=1e-12)
    # Peaked response declines at high T: value at 45 degC below the 25-30 degC band.
    T_lo = lb.T_REF_K + 3.0
    T_hi = lb.T_REF_K + 20.0
    assert float(lb.peaked_arrhenius_factor(T_hi, Ha, Hd, S)) < float(
        lb.peaked_arrhenius_factor(T_lo, Ha, Hd, S))


def test_bernacchi_constants_units_and_ratios():
    # umol/mol values equal the canonical Bernacchi mmol/mol figures * 1000.
    assert lb.KO25_UMOL_MOL == pytest.approx(278.4 * 1000.0)
    assert lb.O2_UMOL_MOL == pytest.approx(209.0 * 1000.0)
    assert lb.KC25_UMOL_MOL == pytest.approx(404.9)
    assert lb.GAMMA_STAR25_UMOL_MOL == pytest.approx(42.75)
    # The O2/Ko ratio the Michaelis term uses is unit-independent (~0.75).
    assert lb.O2_UMOL_MOL / lb.KO25_UMOL_MOL == pytest.approx(0.7507, abs=1e-3)


def test_reference_temperature_is_25C():
    assert float(lb.T_REF_K) == pytest.approx(constants.T_freeze + 25.0)


def test_differentiable():
    g = jax.grad(lambda T: lb.arrhenius_factor(T, 65330.0))(300.0)
    assert np.isfinite(float(g)) and float(g) > 0.0
    gp = jax.grad(lambda T: lb.peaked_arrhenius_factor(T, 72000.0, 200000.0, 649.0))(300.0)
    assert np.isfinite(float(gp))
