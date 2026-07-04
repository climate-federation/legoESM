"""Direct tests for the thermo saturation mixing-ratio curves — the shared
``e_sat -> q_sat`` conversion and the Goff variant added for the #762 air-sea
thermodynamic convention (0.98 x Goff surface q_sat)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.thermo import (
    saturation_mixing_ratio,
    saturation_mixing_ratio_goff,
    saturation_vapor_pressure,
    saturation_vapor_pressure_goff,
    _mixing_ratio_from_esat,
)

_T = jnp.array([275.0, 285.0, 295.0, 305.0])   # SST range [K]
_P = jnp.full_like(_T, 1.0e5)                    # ~surface [Pa]


def test_tetens_is_shared_conversion_over_tetens_curve():
    """The refactor is exact: saturation_mixing_ratio == the shared
    conversion applied to the Tetens e_sat (no numeric drift)."""
    a = np.asarray(saturation_mixing_ratio(_T, _P))
    b = np.asarray(_mixing_ratio_from_esat(saturation_vapor_pressure(_T), _P))
    np.testing.assert_array_equal(a, b)


def test_goff_is_shared_conversion_over_goff_curve():
    a = np.asarray(saturation_mixing_ratio_goff(_T, _P))
    b = np.asarray(
        _mixing_ratio_from_esat(saturation_vapor_pressure_goff(_T), _P))
    np.testing.assert_array_equal(a, b)


def test_goff_differs_from_tetens_but_close():
    """Goff and Tetens are distinct curves (non-vacuous switch) yet agree
    to a few percent across the SST range — the ~0.5-1 % Δq class the
    #762 convention targets, not a blunder."""
    tet = np.asarray(saturation_mixing_ratio(_T, _P))
    gof = np.asarray(saturation_mixing_ratio_goff(_T, _P))
    assert not np.allclose(tet, gof)                     # actually differ
    rel = np.abs(gof - tet) / tet
    assert np.all(rel < 0.05)                            # same ballpark
    assert np.all(np.isfinite(gof)) and np.all(gof > 0)  # physical


def test_goff_monotone_in_T():
    gof = np.asarray(saturation_mixing_ratio_goff(_T, _P))
    assert np.all(np.diff(gof) > 0)                      # warmer -> wetter


def test_goff_differentiable():
    """Reverse-mode AD flows through the Goff q_sat (surface flux path)."""
    g = jax.grad(lambda t: saturation_mixing_ratio_goff(
        jnp.array([t]), jnp.array([1.0e5]))[0])(295.0)
    assert np.isfinite(float(g)) and float(g) > 0.0
